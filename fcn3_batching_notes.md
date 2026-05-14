# FCN3 Batching Notes

Date: 2026-05-14

## Summary

True Makani-level FCN3 batching with `B=2` originally OOMed on an H100 80GB, but it can be made to fit by chunking the atmospheric decoder over the flattened `batch * pressure_group` dimension. With decoder chunking and a patched Earth2Studio wrapper that calls Makani with a real batch, `B=2` ran successfully and gave roughly a 1.8x throughput gain in the single-step profile.

This is not yet production behavior. The current Earth2Studio FCN3 wrapper accepts a batch coordinate but loops over batch members internally, so it is effectively serial `B=1` Makani execution.

## Baseline Behavior

Current wrapper path:

```text
earth2studio FCN3 wrapper
  -> Makani ModelWrapper.forward
  -> SingleStepWrapper._forward_eval
  -> AtmoSphericNeuralOperatorNet.forward
```

Relevant files:

```text
/usr/local/lib/python3.12/dist-packages/earth2studio/models/px/fcn3.py
/usr/local/lib/python3.12/dist-packages/makani/models/model_package.py
/usr/local/lib/python3.12/dist-packages/makani/models/stepper.py
/usr/local/lib/python3.12/dist-packages/makani/models/networks/fourcastnet3.py
```

The Earth2Studio wrapper loops here:

```python
for j, _ in enumerate(coords["batch"]):
    for i, t in enumerate(coords["time"]):
        x[j, i : i + 1] = self.model(...)
```

So `profile_fcn3_batch.py --batch-size 2` does not test true Makani `B=2`; it tests wrapper-level microbatching.

## Why True B=2 OOMs Without Chunking

Direct Makani `B=1` already peaks around:

```text
peak allocated: 49.313 GiB
peak reserved : 58.070 GiB
```

Direct Makani `B=2` without chunking OOMs:

```text
CUDA OOM: Tried to allocate 40.73 GiB
allocated at OOM: 54.281 GiB
reserved at OOM : 54.457 GiB
free at OOM     : 24.04 GiB
```

The failed allocation correlates with the atmospheric decoder DISCO expansion.

For `B=1`, the largest forward buffer is:

```text
atmo_decoder_disco_kernel_output
shape: (13, 45, 9, 721, 1440)
dtype: fp32
size : 20.364 GiB
```

For true `B=2`, `AtmoSphericNeuralOperatorNet.decode` reshapes the atmospheric latent tensor from:

```text
(2, 13 * 45, 360, 720)
```

to:

```text
(26, 45, 360, 720)
```

The `26` dimension is:

```text
batch_size * n_atmo_groups = 2 * 13
```

The DISCO expansion then becomes:

```text
(26, 45, 9, 721, 1440) fp32 = 40.727 GiB
```

This matches the OOM request exactly. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` did not help, confirming this is a single live allocation that is too large, not allocator fragmentation.

## Chunking Strategy

The localized chunking point is in `AtmoSphericNeuralOperatorNet.decode`:

```python
x_atmo = x[..., : (self.n_atmo_groups * self.atmo_embed_dim), :, :].reshape(
    -1, self.atmo_embed_dim, *x.shape[-2:]
)
x_atmo = self.atmo_decoder(x_atmo)
```

Chunk the flattened first dimension before calling `atmo_decoder`:

```python
chunks = [
    self.atmo_decoder(part.contiguous())
    for part in x_atmo.split(chunk_size, dim=0)
]
x_atmo = torch.cat(chunks, dim=0)
```

Estimated atmospheric decoder DISCO buffer sizes for true `B=2`:

```text
unchunked : 40.727 GiB
chunk=13  : 20.364 GiB
chunk=8   : 12.531 GiB
chunk=4   :  6.265 GiB
chunk=1   :  1.566 GiB
```

## Profiling Harness

Added profiling-only monkeypatch:

```text
profile_fcn3_chunked_batch.py
```

It does not modify installed packages. It monkeypatches:

1. `AtmoSphericNeuralOperatorNet.decode` to chunk `atmo_decoder`.
2. Earth2Studio FCN3 `_forward` to call Makani with a true batch instead of looping over batch members.

Command used:

```bash
TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 \
python profile_fcn3_chunked_batch.py --chunk-sizes 13 8 4 --batch-size 2 --wrapper-chunk-size 4
```

Measured results:

| Case | Elapsed | Seconds/sample | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: | ---: |
| direct `B=1`, no chunk | `2.064s` | `2.064s` | `49.313 GiB` | `58.070 GiB` |
| direct `B=2`, chunk `13` | `2.286s` | `1.143s` | `52.555 GiB` | `59.398 GiB` |
| direct `B=2`, chunk `8` | `2.251s` | `1.125s` | `35.989 GiB` | `43.109 GiB` |
| direct `B=2`, chunk `4` | `2.257s` | `1.129s` | `33.439 GiB` | `37.406 GiB` |
| patched wrapper `B=2`, chunk `4` | `2.258s` | `1.129s` | `34.027 GiB` | `37.992 GiB` |

`chunk=4` is the best current memory choice. Runtime was essentially the same as `chunk=8` and `chunk=13`, but with much lower peak memory.

## Estimated 20-Step Benefit

Using the measured patched-wrapper `B=2`, `chunk=4` timing:

```text
default serial B=1: 2.064s/sample/step
true B=2 chunk=4 : 2.258s for 2 samples/step = 1.129s/sample/step
```

For 2 ensemble members over 20 FCN3 steps:

```text
default B=1 serial: 20 * 2 * 2.064s = 82.56s
B=2 chunk=4       : 20 * 2.258s     = 45.16s
```

Estimated improvement:

```text
wall-time reduction: ~45.3%
throughput gain    : ~1.83x
```

This assumes the single-step timing scales across 20 autoregressive steps.

## Production Caveats

Before using this in the workflow, the profiling monkeypatch should be converted into a proper implementation.

## Local Implementation

A local Earth2Studio-side subclass has been added:

```text
serve/server/example_workflows/chunked_batched_fcn3.py
```

Class:

```python
ChunkedBatchedFCN3
```

It keeps the installed Makani package untouched and applies the batching changes after the Makani model is loaded:

1. Installs a chunked ``decode`` method on the loaded Makani core instance.
2. Overrides Earth2Studio ``FCN3._forward`` to call Makani with a true batch dimension.
3. Adds ``set_batch_seeds(seeds)`` so a workflow can initialize per-ensemble noise states from explicit seed lists before creating a batched iterator.

Basic load pattern:

```python
from serve.server.example_workflows.chunked_batched_fcn3 import ChunkedBatchedFCN3

package = ChunkedBatchedFCN3.load_default_package()
fcn3 = ChunkedBatchedFCN3.load_model(package, atmo_decoder_chunk_size=4)
fcn3.to(device)
fcn3.eval()
```

For a batched ensemble group:

```python
fcn3.set_batch_seeds(batch_seeds)
iterator = fcn3.create_iterator(x_batched, coords_batched)
```

Items to verify:

1. Preserve FCN3 stochastic preprocessor/noise-state semantics for multiple batch members.
2. Validate that batched outputs match the serial wrapper outputs for fixed seeds within acceptable numerical tolerance.
3. Confirm multi-step autoregressive rollout stability for 20+ steps.
4. Decide where chunk size should live: model attribute, workflow parameter, environment variable, or fixed constant.
5. Keep FCN3 conditioning for StormScope compatible with the workflow's ensemble seed mapping.

Recommended starting point:

```text
true Makani batch size: 2
atmo decoder chunk size: 4
```
