# {{WORKLOAD}} inference performance analysis

Status: recommendation-only analysis unless a separately reported tuning phase follows.

## Executive summary

{{MEASURED_RESULT_AND_PRIMARY_BOTTLENECK}}

## Workload and confirmed configuration

| Item | Value |
|---|---|
| Command/config fingerprint | {{CONFIG}} |
| Model and workflow | {{MODEL}} |
| Data and cache state | {{DATA}} |
| Forecast/member/sample shape | {{SHAPE}} |
| Precision and devices | {{RUNTIME}} |
| IO/checkpoint configuration | {{IO}} |
| Correctness invariant | {{CORRECTNESS}} |

## Benchmark protocol and unprofiled results

Report warm end-to-end and steady-state logical-unit distributions. Include startup latency only when explicitly requested and label it optional. Include memory and per-rank results.

## Trace coverage and annotation health

{{TRACE_PROVENANCE_AND_LIMITATIONS}}

## Whole-pipeline temporal breakdown

{{PHASE_TABLE}}

## CPU/GPU inference pipeline

![CPU/GPU inference pipeline]({{PIPELINE_SVG}})

## Forward-pass dominant kernels

State the validated forward boundary, rank/step, forward wall time, forward GPU time, HTA version, and coverage percentage.

{{DOMINANT_KERNEL_MARKDOWN_TABLE}}

Required columns: rank; exact kernel; family; calls; total/self GPU ms; mean/median/p95/max ms; share of forward GPU time; share of logical-step wall time; launch source/range; shapes; provenance.

![Forward dominant-kernel timeline]({{DOMINANT_KERNEL_SVG}})

Explain launch order, gaps, repeated families, and whether dominance changes across forecast, sampler, or diagnostic steps.

## Critical path and idle bubbles

{{CRITICAL_PATH}}

## Optional torch.compile diagnostics

{{COMPILE_ANALYSIS_OR_NOT_APPLICABLE}}

## Optional NCU kernel analysis

{{NCU_EVIDENCE_OR_SKIP_REASON}}

## Phase-to-source map and code analysis

{{SOURCE_MAP_AND_EVIDENCE}}

## Ranked findings

| Rank | Finding | Severity | Confidence | Critical-path evidence | Source analysis | Isolated experiment | Status |
|---:|---|---|---|---|---|---|---|
| 1 | {{FINDING}} | {{SEVERITY}} | {{CONFIDENCE}} | {{EVIDENCE}} | {{SOURCE}} | {{EXPERIMENT}} | recommendation_only |

## Correctness, limitations, and residual bottlenecks

{{CORRECTNESS_LIMITATIONS_RESIDUALS}}

## Artifact index

{{ARTIFACT_LINKS}}

## Controlled tuning results

Include only after an explicitly requested optimization: exact change, semantic risk, paired correctness, unprofiled warm end-to-end and steady-state baseline/candidate distributions, measured delta with uncertainty, and disposition.
