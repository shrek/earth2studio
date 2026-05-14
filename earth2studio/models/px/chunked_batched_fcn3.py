# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Earth2Studio FCN3 variant with true Makani batching and decoder chunking.

The stock Earth2Studio ``FCN3`` accepts a batch coordinate but loops over batch
members before calling Makani. This subclass keeps Makani untouched, but adapts
the loaded model instance so true batched inference can fit on 80GB GPUs.
"""

from __future__ import annotations

import json
import types
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import torch
from loguru import logger

from earth2studio.models.auto import Package
from earth2studio.models.px import fcn3 as earth2_fcn3
from earth2studio.models.px.fcn3 import FCN3, VARIABLES
from earth2studio.utils.imports import check_optional_dependencies
from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import CoordSystem

try:
    from makani.third_party.climt.zenith_angle import cos_zenith_angle
except ImportError:
    cos_zenith_angle = None


class ChunkedBatchedFCN3(FCN3):
    """FCN3 subclass that uses true Makani batches with a chunked decoder.

    Parameters
    ----------
    core_model : torch.nn.Module
        Loaded Makani model package.
    variables : np.ndarray
        FCN3 variable names.
    seed : int
        Makani stochastic preprocessor seed.
    atmo_decoder_chunk_size : int
        Number of flattened ``batch * pressure_group`` atmospheric slices to
        pass to the full-resolution atmospheric decoder at once. A value of
        ``0`` disables chunking.
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        variables: np.ndarray = np.array(VARIABLES),
        seed: int = 333,
        atmo_decoder_chunk_size: int = 4,
    ) -> None:
        self.atmo_decoder_chunk_size = atmo_decoder_chunk_size
        self._batch_seeds: list[int] | None = None
        super().__init__(core_model=core_model, variables=variables, seed=seed)
        self.install_chunked_decode()

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        variables: Sequence[str] = VARIABLES,
        atmo_decoder_chunk_size: int = 4,
    ) -> "ChunkedBatchedFCN3":
        """Load FCN3 and return the chunked true-batch subclass."""

        if not earth2_fcn3._cuda_extension_available:
            logger.warning(
                "torch-harmonics disco CUDA extension is not available.\n"
                "FCN3 run on GPU/CUDA will be slower.\n"
                "Please install torch-harmonics in the following way:\n"
                "export FORCE_CUDA_EXTENSION=1\n"
                "pip install --no-build-isolation torch-harmonics"
            )

        model = earth2_fcn3.load_model_package(package)
        model.eval()

        config_path = package.get("config.json")
        with open(config_path) as f:
            config = json.load(f)
            variables = config["channel_names"]

        return cls(
            model,
            variables=np.array(variables),
            atmo_decoder_chunk_size=atmo_decoder_chunk_size,
        )

    @property
    def makani_core(self) -> torch.nn.Module:
        """Return the loaded Makani ``AtmoSphericNeuralOperatorNet``."""

        return self.model.model.model

    def set_atmo_decoder_chunk_size(self, chunk_size: int) -> None:
        """Update the atmospheric decoder chunk size."""

        if chunk_size < 0:
            raise ValueError("atmo decoder chunk size must be >= 0")
        self.atmo_decoder_chunk_size = chunk_size
        self.makani_core.atmo_decoder_chunk_size = chunk_size

    def set_rng(self, seed: int = 333, reset: bool = True) -> None:
        """Set the underlying FCN3 RNG and clear any batch seed override."""

        self._batch_seeds = None
        super().set_rng(seed=seed, reset=reset)

    def set_batch_seeds(self, seeds: Sequence[int]) -> None:
        """Set explicit per-ensemble seeds for the next batched iterator reset."""

        if not seeds:
            raise ValueError("Expected at least one batch seed")
        self._batch_seeds = [int(seed) for seed in seeds]

    def install_chunked_decode(self) -> None:
        """Patch the loaded Makani core instance with a chunked decode method."""

        core = self.makani_core
        core.atmo_decoder_chunk_size = self.atmo_decoder_chunk_size

        def chunked_decode(self: Any, x: torch.Tensor) -> torch.Tensor:
            batchdims = x.shape[:-3]

            x_atmo = x[
                ..., : (self.n_atmo_groups * self.atmo_embed_dim), :, :
            ].reshape(-1, self.atmo_embed_dim, *x.shape[-2:])

            chunk_size = int(getattr(self, "atmo_decoder_chunk_size", 0) or 0)
            if chunk_size > 0 and x_atmo.shape[0] > chunk_size:
                x_atmo = torch.cat(
                    [
                        self.atmo_decoder(part.contiguous())
                        for part in x_atmo.split(chunk_size, dim=0)
                    ],
                    dim=0,
                )
            else:
                x_atmo = self.atmo_decoder(x_atmo)

            x_out = torch.zeros(
                *batchdims,
                self.n_out_chans,
                *x_atmo.shape[-2:],
                dtype=x.dtype,
                device=x.device,
            )
            x_out[..., self.atmo_channels, :, :] = x_atmo.reshape(
                *batchdims, -1, *x_atmo.shape[-2:]
            )

            if hasattr(self, "surf_decoder"):
                x_surf = x[..., -self.surf_embed_dim :, :, :]
                x_surf = self.surf_decoder(x_surf)
                x_out[..., self.surf_channels, :, :] = x_surf.reshape(
                    *batchdims, -1, *x_surf.shape[-2:]
                )

            return x_out

        core.decode = types.MethodType(chunked_decode, core)

    def _reset_internal_state(self, num_ensemble: int, num_time: int) -> None:
        """Reset per-ensemble noise states with one-member state tensors."""

        preprocessor = self.model.model.preprocessor
        if (
            self._batch_seeds is not None
            and len(self._batch_seeds) != num_ensemble
        ):
            raise ValueError(
                f"Expected {num_ensemble} batch seeds, got {len(self._batch_seeds)}"
            )
        internal_noise_states: list[list[torch.Tensor | None]] = [
            [None for _ in range(num_time)] for _ in range(num_ensemble)
        ]
        internal_rng_states: list[
            list[tuple[torch.Tensor | None, torch.Tensor | None] | None]
        ] = [[None for _ in range(num_time)] for _ in range(num_ensemble)]
        for i in range(num_ensemble):
            if self._batch_seeds is not None:
                noise = getattr(preprocessor, "input_noise", None)
                if noise is not None and getattr(noise, "state", None) is not None:
                    noise.state = noise.state.detach().clone()
                self.model.set_rng(reset=True, seed=self._batch_seeds[i])
            for j in range(num_time):
                preprocessor.update_internal_state(
                    replace_state=True, batch_size=1
                )
                internal_noise_states[i][j] = preprocessor.get_internal_state(
                    tensor=True
                )
                internal_rng_states[i][j] = preprocessor.get_internal_state(
                    tensor=False
                )
        self._internal_noise_states = internal_noise_states
        self._internal_rng_states = internal_rng_states

    def _get_batched_internal_state(
        self, batch_size: int, time_index: int
    ) -> torch.Tensor | None:
        states = [
            self._get_internal_state(ensemble_index, time_index)
            for ensemble_index in range(batch_size)
        ]
        first = states[0]
        if first is None:
            return None
        if not all(isinstance(state, torch.Tensor) for state in states):
            raise TypeError("Expected tensor internal states for batched FCN3")
        if not all(state.shape[0] == 1 for state in states):
            raise ValueError(
                "Expected stored FCN3 internal states with leading size 1"
            )
        return torch.cat(states, dim=0)

    def _advance_batched_internal_state(
        self, batch_size: int, time_index: int
    ) -> torch.Tensor | None:
        """Advance each ensemble member with its own Makani noise RNG stream."""

        preprocessor = self.model.model.preprocessor
        states = [
            self._get_internal_state(ensemble_index, time_index)
            for ensemble_index in range(batch_size)
        ]
        first = states[0]
        if first is None:
            return None

        advanced_states: list[torch.Tensor] = []
        for ensemble_index, state in enumerate(states):
            if not isinstance(state, torch.Tensor):
                raise TypeError("Expected tensor internal states for batched FCN3")

            rng_state = self._internal_rng_states[ensemble_index][time_index]
            self._set_preprocessor_tensor_state(state)
            if rng_state is not None:
                preprocessor.set_internal_state(rng_state)
            preprocessor.update_internal_state(replace_state=False, batch_size=1)
            advanced_states.append(preprocessor.get_internal_state(tensor=True))
            self._internal_rng_states[ensemble_index][time_index] = (
                preprocessor.get_internal_state(tensor=False)
            )

        batched_state = torch.cat(advanced_states, dim=0)
        self._set_preprocessor_tensor_state(batched_state)
        return batched_state

    def _set_preprocessor_tensor_state(self, state: torch.Tensor | None) -> None:
        if state is None:
            return
        preprocessor = self.model.model.preprocessor
        if not hasattr(preprocessor, "input_noise"):
            return

        noise = preprocessor.input_noise
        with torch.no_grad():
            if (
                getattr(noise, "state", None) is not None
                and noise.state.shape == state.shape
            ):
                noise.state.copy_(state)
            else:
                noise.state = state.detach().clone()

    def _set_batched_internal_state(
        self, batch_size: int, time_index: int
    ) -> None:
        preprocessor = self.model.model.preprocessor
        state = preprocessor.get_internal_state(tensor=True)
        if state is None:
            return
        if state.shape[0] != batch_size:
            raise ValueError(
                f"Expected batched state leading size {batch_size}, got {state.shape[0]}"
            )
        for ensemble_index in range(batch_size):
            self._internal_noise_states[ensemble_index][time_index] = (
                state[ensemble_index : ensemble_index + 1].detach().clone()
            )

    def _times_for_batch(
        self, t: np.datetime64, coords: CoordSystem
    ) -> list[datetime]:
        times = [
            datetime.fromisoformat(dt.isoformat() + "+00:00")
            for dt in timearray_to_datetime(t + coords["lead_time"])
        ]
        if len(times) != 1:
            raise ValueError(
                "ChunkedBatchedFCN3 expects exactly one FCN3 lead_time per call"
            )
        return times * len(coords["batch"])

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run FCN3 with a true Makani batch dimension."""

        output_coords = self.output_coords(coords)
        x = x.squeeze(2)
        batch_size = len(coords["batch"])
        autocast_dtype = (
            torch.bfloat16
            if earth2_fcn3._cuda_extension_available
            else torch.float32
        )

        for time_index, t in enumerate(coords["time"]):
            self._advance_batched_internal_state(batch_size, time_index)

            times = self._times_for_batch(t, coords)
            with torch.autocast(device_type=x.device.type, dtype=autocast_dtype):
                if self.model.add_zenith:
                    if cos_zenith_angle is None:
                        raise ImportError(
                            "Makani is required to compute FCN3 zenith-angle features"
                        )
                    cosz = cos_zenith_angle(
                        times,
                        self.model.lon_grid,
                        self.model.lat_grid,
                    ).astype("float32")
                    z = torch.as_tensor(cosz[:, None], device=x.device)
                    self.model.model.preprocessor.cache_unpredicted_features(
                        None, None, xz=z, yz=None
                    )

                xn = (x[:, time_index] - self.model.in_bias) / self.model.in_scale
                y = self.model.model(
                    xn, update_state=False, replace_state=False
                )
                x[:, time_index] = y * self.model.out_scale + self.model.out_bias

            self._set_batched_internal_state(batch_size, time_index)

        return x.unsqueeze(2), output_coords
