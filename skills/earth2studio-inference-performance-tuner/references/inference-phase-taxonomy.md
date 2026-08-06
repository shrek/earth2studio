# Inference phase taxonomy

Use these names in trace ranges, `phase-source-map.json`, and reports. Mark an absent phase `not_applicable` rather than omitting it.

| Phase | Boundary | Typical evidence |
|---|---|---|
| `output_initialize` | create output schema/store | host span, metadata calls |
| `data_fetch` | source request through returned payload | source range, bytes |
| `data_decode` | decode, normalize, or materialize source data | CPU range |
| `coordinate_mapping` | selection, reshape, concatenate, `map_coords` | CPU/CUDA range |
| `regrid` | spatial interpolation/regridding | CPU/CUDA range |
| `host_to_device` | input tensor transfer | memcpy kernels, range |
| `perturbation` | ensemble/member perturbation | CPU/CUDA range |
| `forecast_step` | one prognostic iterator transition | outer logical range |
| `diagnostic` | diagnostic transformation | nested model range |
| `sampler_or_denoising_step` | one sampler/denoiser iteration | nested logical range |
| `assimilation` | observation processing and analysis update | CPU/CUDA range |
| `output_filter` | variable/domain filtering and splitting | CPU/CUDA range |
| `device_to_host` | accelerator output transfer | memcpy kernels, range |
| `output_write` | one output submission/write | host range, bytes |
| `output_flush` | buffered or asynchronous completion | host range |
| `checkpoint` | checkpoint state/metadata write | host range, bytes |
| `distributed_sync` | collectives, barriers, rank synchronization | NCCL/host ranges |

Use `forecast_step` for deterministic inference, nest it inside `ensemble_batch` for ensemble workflows, nest `sampler_or_denoising_step` inside `sample` or `forecast_step` for diffusion, and use `work_item` outside model-specific units in recipes.
