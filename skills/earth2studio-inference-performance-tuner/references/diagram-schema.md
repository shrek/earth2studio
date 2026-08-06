# HTA pipeline and dominant-kernel artifact schema

## Normalized input

`render_trace_diagrams.py` consumes one JSON object with `schema_version: "0.1"`, `provenance`, `pipeline`, and `forward`.

`pipeline` contains `step_id`, positive `duration_ms`, and at least CPU and GPU lanes. Each lane contains spans with `label`, `category`, `start_ms`, and `end_ms` relative to the validated logical boundary.

`forward` contains one or more steps. Each step contains:

- `step_id`, `rank`, positive `duration_ms`, and `forward_gpu_time_ms`;
- `kernels`: launch instances with exact `name`, normalized `family`, `start_ms`, `end_ms`, optional `source_range`, and optional `shapes`;
- `dominant_kernels`: aggregated rows with `name`, `family`, `call_count`, `total_gpu_ms`, `self_gpu_ms`, `mean_gpu_ms`, `median_gpu_ms`, `p95_gpu_ms`, `max_gpu_ms`, `pct_forward_gpu_time`, `pct_step_wall_time`, `source_range`, `shapes`, and `provenance`.

Use JSON `null`, an empty string, or an empty list for unavailable optional evidence. Do not fabricate statistics. Every duration must be finite, non-negative, and bounded by its enclosing step. Preserve exact kernel names even when the diagram aggregates by family.

## Required outputs

The renderer must write:

- `pipeline.svg`: CPU/GPU lanes, phase spans, and idle bubbles;
- `dominant-kernels.svg`: one panel per forward step, one lane per dominant family, launch ordering/gaps, cumulative family time, and share;
- `dominant-kernels.json`: validated aggregate rows plus provenance;
- `dominant-kernels.csv`: machine-readable detailed table;
- `dominant-kernels.md`: report-ready detailed table.

Sort the detailed table by descending `total_gpu_ms`, then exact name. Include enough rows to cover at least 80% of forward GPU time and at least ten rows when available; allow a stricter user-specified top-N cap only when stated in the report. Aggregate repeated launches without losing exact-name rows.

## Evidence rules

Derive boundaries and timings from HTA-normalized events, not screenshot inspection. Record HTA version, trace path/hash, rank, step-boundary provenance, extraction command, and assumptions in `provenance`. When shapes or launch source are not collected, leave them unavailable and say so.
