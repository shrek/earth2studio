---
name: earth2studio-inference-performance-tuner
description: Profile and tune Earth2Studio weather and climate inference workloads with reproducible warm end-to-end and steady-state baselines; validated Kineto and HolisticTraceAnalysis evidence; detailed forward-pass dominant-kernel tables and diagrams; conditional torch.compile and Nsight Compute diagnostics; phase-to-source review; and correctness-guarded optimization experiments. Use for slow deterministic, diagnostic, ensemble, diffusion/downscaling, data-assimilation, seasonal, recipe, IO-heavy, or distributed inference; low GPU utilization; excessive memory use; data or output stalls; compile regressions; and poor scaling. Do not use for training, model-accuracy comparison, data-only fetching, installation, or serving-load testing.
---

# Earth2Studio Inference Performance Tuner

## Overview

Measure the complete inference pipeline before changing it. Produce a validated analysis bundle, then optionally apply one reversible optimization at a time and measure it against the confirmed baseline. Never change physical fields, units, forecast semantics, stochastic sampling semantics, or output schemas for speed.

## Required inputs and execution gate

Collect or infer the complete launch command and working directory, representative model/config/data, forecast steps, times, variables, ensemble/sample/batch sizes, precision, device topology, IO/checkpoint settings, data cache state, budgets, primary performance goal, and executable correctness check.

Ask one targeted question if the command, representative data, or correctness check is missing. Read-only discovery is allowed, but do not run smoke tests, inference, profilers, or NCU until all three are known and the user explicitly confirms the resolved `test-config.md`. Regenerate and reconfirm it after any material change.

## Analysis workflow

### 1. Prepare and confirm the run bundle

Read `references/phase1-protocol.md`, `references/inference-phase-taxonomy.md`, and the relevant entry in `references/golden-workloads.md`. Create a fresh bundle with `scripts/create_run_bundle.py`, show `test-config.md`, obtain explicit confirmation, and run `scripts/confirm_run_config.py`. Never record secrets or an unfiltered environment.

### 2. Establish correctness and unprofiled baselines

Read `references/benchmark-metrics.md` and `references/correctness-protocol.md`. Run a correctness smoke test, then measure warm end-to-end including final IO completion and steady inference units after warmup. Do not measure package download, model loading, device setup, or other startup latency by default. Measure startup only when the user explicitly requests it and confirms `include_startup: true`. Use at least three repetitions unless the user limits the budget. Do not use profiler, compiler-diagnostic, or NCU replay timings as benchmark results.

### 3. Add minimal inference instrumentation

Prefer existing ranges. Otherwise add opt-in `record_function` or NVTX ranges around canonical phases without changing default behavior. Capture identical bounded logical units for baseline and candidate variants. Use native `ProfilerStep#N` ranges or an explicit outer `inference_step`, `forecast_step`, `sample`, `ensemble_batch`, or `work_item` boundary. Read `references/trace-annotation-health.md` and validate each trace with `scripts/validate_trace_annotations.py` before HTA.

### 4. Diagnose optional compilation

Read `references/compile-comparison.md` only when compilation is supported. Honor model-specific compile boundaries. Warm every intended graph before measurement and exclude lazy compilation and first-step setup from the default baseline. Measure compile or first-step latency only when explicitly requested. Record expected recompilations and cache state separately from warm steady state. Normalize diagnostic logs with `scripts/analyze_compile_logs.py`. Do not force a universal eager/compiled comparison.

### 5. Analyze traces and forward-pass dominant kernels

Inspect the installed HolisticTraceAnalysis version and use HTA as the only Python trace-analysis dependency. For each representative trace:

1. Validate trace/rank coverage and logical-step provenance.
2. Compute temporal, idle-time, kernel, launch, memory, communication, and rank-imbalance breakdowns as applicable.
3. Derive the critical path for a representative logical unit.
4. Normalize CPU/GPU and forward spans using `references/diagram-schema.md`.
5. Generate the required CPU/GPU pipeline SVG with `scripts/render_trace_diagrams.py`.
6. Generate a detailed forward-pass dominant-kernel table in JSON, CSV, and Markdown. Include rank, exact kernel name, normalized family, call count, total/self GPU duration, mean, median, p95, maximum, percentage of forward GPU time, percentage of logical-step wall time, launch source/range, shapes when captured, and evidence provenance. Never infer unavailable values.
7. Generate the required forward dominant-kernel diagram. Show each dominant kernel family on its own lane inside the validated forward boundary, retain launch order and gaps, and label cumulative duration and share. Generate per-step views when materially different forecast, sampler, or diagnostic steps exist.
8. Generate in-step and multi-step bubble diagrams when supported.

The dominant-kernel table and diagrams are completion requirements. Timings must come from normalized HTA evidence rather than visual estimation.

### 6. Review phase source

Read `references/source-review-protocol.md`. Map every canonical phase to ranges, symbols, repository-relative paths, configuration, and evidence. Mark absent phases `not_applicable`. Require measured evidence before recommending code changes.

### 7. Use NCU conditionally

Read `references/ncu-profiling.md`. Use NCU only after HTA identifies a critical-path kernel with a recommendation-relevant question. Show the exact bounded capture plan and obtain separate approval. Start with at most ten launches and the default set. Never install NCU, use sudo, or change performance-counter permissions.

### 8. Report and validate

Use `assets/report-template.md`. Rank findings with evidence, source, one isolated experiment, and status `recommendation_only`. Validate with `scripts/validate_run_bundle.py --ready`.

## Controlled tuning phase

Enter tuning only after the user requests implementation based on phase 1. Select one recommendation, state the change and semantic risk, implement it reversibly and opt-in when practical, and rerun the identical correctness and unprofiled benchmark protocol. Report speedup only from paired unprofiled results. Reject or label candidates that fail correctness or do not improve the target distribution.

## Completion criteria

Finish analysis only when confirmation is current; correctness passes; warm end-to-end and steady-state metrics are reported; trace annotations are healthy; HTA evidence exists; the CPU/GPU diagram and detailed forward dominant-kernel JSON/CSV/Markdown table and SVG exist; every phase is mapped or not applicable; NCU evidence exists or is explicitly unnecessary; and every conclusion links to an artifact.

## Guardrails

- Do not execute before explicit configuration confirmation.
- Do not vary data, outputs, forecast length, seeds, precision, batch/member/sample counts, topology, or cache state without labeling the comparison.
- Include async IO completion in end-to-end timing.
- Exclude model loading, lazy initialization, compilation, and other startup work from default benchmark samples.
- Do not report projected savings as measured speedups.
- Do not generalize smoke inputs to representative workloads.
- Do not edit workload code during recommendation-only analysis.
