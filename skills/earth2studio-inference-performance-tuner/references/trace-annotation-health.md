# Trace annotation health

Require a trustworthy outer unit: native `ProfilerStep#N` or explicit `forecast_step`, `inference_step`, `sample`, `ensemble_batch`, or `work_item`. Require finite positive durations and at least one compute phase plus every phase claimed by the report. Validate each variant independently.

Recapture when the boundary is absent, overlapping units cannot be separated, required phases are missing, or durations are invalid. If compiled nested ranges are ignored, verify outer ranges in the raw trace. Reconstruct only when recapture is impractical; record algorithm, confidence, affected steps, and `provenance: reconstructed`.
