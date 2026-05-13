#!/usr/bin/env python3
"""Estimate FCN3 load-only model memory from the cached checkpoint.

This estimates resident model tensor memory, not full inference peak memory.
The default output includes:

1. Checkpoint tensors from best_ckpt_mp0.tar, read with FakeTensorMode when
   available so the 2.7 GB checkpoint does not need to be materialized.
2. Runtime buffers from a CPU load of earth2studio.models.px.FCN3, because FCN3
   constructs sizeable SHT/ISHT/preprocessor buffers that are not checkpoint
   parameters but do move to GPU with the model.

On the cached nvidia/fourcastnet3 package this should report about:
  checkpoint tensors: 2.648 GiB
  runtime buffers:    1.228 GiB
  resident tensors:   3.876 GiB
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


BYTES_IN_MIB = 1024**2
BYTES_IN_GIB = 1024**3


@dataclass(frozen=True)
class TensorSummary:
    tensor_count: int
    element_count: int
    byte_count: int

    @property
    def mib(self) -> float:
        return self.byte_count / BYTES_IN_MIB

    @property
    def gib(self) -> float:
        return self.byte_count / BYTES_IN_GIB


@dataclass(frozen=True)
class DTypeSummary:
    dtype: str
    tensor_count: int
    element_count: int
    byte_count: int

    @property
    def gib(self) -> float:
        return self.byte_count / BYTES_IN_GIB


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str
    element_count: int
    byte_count: int

    @property
    def mib(self) -> float:
        return self.byte_count / BYTES_IN_MIB


def default_fcn3_cache_dir() -> Path:
    root = Path.home() / ".cache" / "earth2studio"
    root = Path(os.environ.get("EARTH2STUDIO_CACHE", root))
    root = Path(os.environ.get("EARTH2STUDIO_MODEL_CACHE", root))
    return root / "fcn3"


def default_checkpoint_path() -> Path:
    return default_fcn3_cache_dir() / "best_ckpt_mp0.tar"


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def walk_tensors(obj: Any, prefix: str = "") -> Iterable[tuple[str, torch.Tensor]]:
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from walk_tensors(value, name)
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            name = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from walk_tensors(value, name)


def summarize_records(records: list[TensorRecord]) -> TensorSummary:
    return TensorSummary(
        tensor_count=len(records),
        element_count=sum(record.element_count for record in records),
        byte_count=sum(record.byte_count for record in records),
    )


def summarize_by_dtype(records: list[TensorRecord]) -> list[DTypeSummary]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for record in records:
        summary = totals[record.dtype]
        summary[0] += 1
        summary[1] += record.element_count
        summary[2] += record.byte_count
    return [
        DTypeSummary(
            dtype=dtype,
            tensor_count=values[0],
            element_count=values[1],
            byte_count=values[2],
        )
        for dtype, values in sorted(totals.items())
    ]


def checkpoint_tensor_records(checkpoint_path: Path) -> list[TensorRecord]:
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:
        FakeTensorMode = None  # type: ignore[assignment]

    if FakeTensorMode is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    else:
        with FakeTensorMode():
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

    records = []
    for name, tensor in walk_tensors(checkpoint):
        records.append(
            TensorRecord(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                element_count=tensor.numel(),
                byte_count=tensor_nbytes(tensor),
            )
        )
    return records


def runtime_model_records() -> tuple[list[TensorRecord], list[TensorRecord]]:
    # Imported lazily so checkpoint-only mode only requires torch.
    from earth2studio.models.px import FCN3

    package = FCN3.load_default_package()
    model = FCN3.load_model(package)

    parameter_records = [
        TensorRecord(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            element_count=tensor.numel(),
            byte_count=tensor_nbytes(tensor),
        )
        for name, tensor in model.named_parameters()
    ]
    buffer_records = [
        TensorRecord(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            element_count=tensor.numel(),
            byte_count=tensor_nbytes(tensor),
        )
        for name, tensor in model.named_buffers()
    ]
    return parameter_records, buffer_records


def round_cuda_allocator_estimate(byte_count: int, granularity_mib: int = 16) -> int:
    """Round to a coarse CUDA allocator bucket for a load-only reserved estimate."""
    granularity = granularity_mib * BYTES_IN_MIB
    return ((byte_count + granularity - 1) // granularity) * granularity


def record_to_dict(record: TensorRecord) -> dict[str, Any]:
    data = asdict(record)
    data["mib"] = record.mib
    return data


def dtype_summary_to_dict(summary: DTypeSummary) -> dict[str, Any]:
    data = asdict(summary)
    data["gib"] = summary.gib
    return data


def tensor_summary_to_dict(summary: TensorSummary) -> dict[str, Any]:
    data = asdict(summary)
    data["mib"] = summary.mib
    data["gib"] = summary.gib
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate cached FCN3 load-only model memory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint_path(),
        help="Path to best_ckpt_mp0.tar.",
    )
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="Only count checkpoint tensors; skip runtime FCN3 buffer loading.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of largest tensors to print.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    return parser


def print_text(result: dict[str, Any], top: int) -> None:
    print("FCN3 load-only memory estimate")
    print("--------------------------------")
    print(f"checkpoint: {result['checkpoint']}")
    print()

    checkpoint = result["checkpoint_tensors"]
    print("Checkpoint tensors")
    print(f"  tensors:  {checkpoint['tensor_count']:,}")
    print(f"  elements: {checkpoint['element_count']:,}")
    print(f"  memory:   {checkpoint['gib']:.3f} GiB ({checkpoint['mib']:.1f} MiB)")

    if "runtime_parameters" in result:
        parameters = result["runtime_parameters"]
        buffers = result["runtime_buffers"]
        total = result["resident_model_tensors"]
        reserved = result["cuda_reserved_estimate"]
        print()
        print("Runtime model tensors")
        print(
            "  parameters: "
            f"{parameters['gib']:.3f} GiB ({parameters['element_count']:,} elements)"
        )
        print(
            "  buffers:    "
            f"{buffers['gib']:.3f} GiB ({buffers['element_count']:,} elements)"
        )
        print(
            "  total:      "
            f"{total['gib']:.3f} GiB ({total['mib']:.1f} MiB)"
        )
        print(
            "  reserved:   "
            f"~{reserved['gib']:.3f} GiB with coarse CUDA allocator rounding"
        )
    elif "runtime_error" in result:
        print()
        print("Runtime model tensors")
        print(f"  skipped: {result['runtime_error']}")

    if result["checkpoint_by_dtype"]:
        print()
        print("Checkpoint by dtype")
        for item in result["checkpoint_by_dtype"]:
            print(
                f"  {item['dtype']}: {item['gib']:.3f} GiB, "
                f"{item['element_count']:,} elements"
            )

    if top > 0:
        print()
        print(f"Largest checkpoint tensors, top {top}")
        for record in result["largest_checkpoint_tensors"][:top]:
            print(
                f"  {record['mib']:8.1f} MiB  "
                f"{record['dtype']:15}  {tuple(record['shape'])!s:24}  "
                f"{record['name']}"
            )


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_path = args.checkpoint.expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint_records = checkpoint_tensor_records(checkpoint_path)
    checkpoint_summary = summarize_records(checkpoint_records)

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_tensors": tensor_summary_to_dict(checkpoint_summary),
        "checkpoint_by_dtype": [
            dtype_summary_to_dict(summary)
            for summary in summarize_by_dtype(checkpoint_records)
        ],
        "largest_checkpoint_tensors": [
            record_to_dict(record)
            for record in sorted(
                checkpoint_records,
                key=lambda record: record.byte_count,
                reverse=True,
            )
        ],
    }

    if not args.checkpoint_only:
        try:
            parameter_records, buffer_records = runtime_model_records()
        except Exception as exc:
            result["runtime_error"] = f"{type(exc).__name__}: {exc}"
        else:
            parameter_summary = summarize_records(parameter_records)
            buffer_summary = summarize_records(buffer_records)
            total_summary = TensorSummary(
                tensor_count=(
                    parameter_summary.tensor_count + buffer_summary.tensor_count
                ),
                element_count=(
                    parameter_summary.element_count + buffer_summary.element_count
                ),
                byte_count=parameter_summary.byte_count + buffer_summary.byte_count,
            )
            reserved_summary = TensorSummary(
                tensor_count=total_summary.tensor_count,
                element_count=total_summary.element_count,
                byte_count=round_cuda_allocator_estimate(total_summary.byte_count),
            )

            result["runtime_parameters"] = tensor_summary_to_dict(parameter_summary)
            result["runtime_buffers"] = tensor_summary_to_dict(buffer_summary)
            result["resident_model_tensors"] = tensor_summary_to_dict(total_summary)
            result["cuda_reserved_estimate"] = tensor_summary_to_dict(reserved_summary)
            result["runtime_parameter_by_dtype"] = [
                dtype_summary_to_dict(summary)
                for summary in summarize_by_dtype(parameter_records)
            ]
            result["runtime_buffer_by_dtype"] = [
                dtype_summary_to_dict(summary)
                for summary in summarize_by_dtype(buffer_records)
            ]

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_text(result, args.top)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
