# Inference benchmark and metrics contract

## Timing views

Record warm end-to-end and compute steady state by default. Start timing only after the model is loaded, moved to its device, and all one-time setup needed by the workload is complete. Do not measure package download, model loading, process startup, device setup, or time to first output unless the user explicitly requests startup analysis and confirms `include_startup: true`. Declare data, compiler, and filesystem cache state when they affect the measured path. Include final `flush()` or `close()` for async output. Use `time.perf_counter()` around complete host-visible work with `torch.cuda.synchronize()` at boundaries; use CUDA events for GPU-only spans.

## Aggregation

Use at least three process-level repetitions and retain per-unit samples after warmup. Report count, median, mean, standard deviation, p5, p95, minimum, and maximum. Preserve per-rank measurements. Do not discard outliers without a predeclared rule and raw results.

## Required metrics

- warm end-to-end latency, including final output completion;
- optional startup latency only when explicitly requested;
- logical-unit latency and throughput;
- simulated forecast hours/s, members/s, or samples/s;
- peak allocated and reserved GPU memory;
- fetched/written bytes and throughput when observable;
- CPU wait, GPU idle, launch overhead, and IO wait when traceable;
- per-rank throughput, imbalance, and scaling when distributed.

Profiled, compile-diagnostic, and NCU replay timings are diagnostic only. Derive speedup only from paired unprofiled samples with one intended difference.
