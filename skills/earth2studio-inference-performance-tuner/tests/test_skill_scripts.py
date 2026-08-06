# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

SKILL = Path(__file__).parents[1]
SCRIPTS = SKILL / "scripts"


def run(
    script: str, *args: object, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPTS / script), *(str(arg) for arg in args)],
        check=check,
        capture_output=True,
        text=True,
    )


def test_bundle_confirmation_is_bound_to_config(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    run(
        "create_run_bundle.py",
        "--output",
        bundle,
        "--workdir",
        SKILL.parents[1],
        "--baseline-command",
        "uv run python inference.py",
        "--data",
        "local fixture",
        "--correctness-command",
        "uv run python check.py",
        "--performance-goal",
        "forecast-step latency",
        "--model-config",
        "DLWP smoke",
        "--forecast-steps",
        "4",
    )
    assert "explicit confirmation required" in (bundle / "test-config.md").read_text()
    config = json.loads((bundle / "test-config.json").read_text())
    assert config["protocol"]["include_startup"] is False
    assert config["protocol"]["default_measurements"] == [
        "warm_end_to_end",
        "steady_state",
    ]
    run("confirm_run_config.py", bundle)
    run("validate_run_bundle.py", bundle, "--confirmed")

    config = json.loads((bundle / "test-config.json").read_text())
    config["workload"]["forecast_steps"] = "5"
    (bundle / "test-config.json").write_text(json.dumps(config))
    result = run("validate_run_bundle.py", bundle, "--confirmed", check=False)
    assert result.returncode == 1
    assert "fingerprint is stale" in result.stderr


def diagram_document() -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "provenance": {
            "hta_version": "0.test",
            "trace": "trace.json",
            "step_boundary": "explicit",
            "extraction_command": "fixture",
        },
        "pipeline": {
            "title": "Inference pipeline",
            "step_id": "forecast_step#3",
            "duration_ms": 10.0,
            "lanes": [
                {
                    "name": "CPU",
                    "spans": [
                        {
                            "label": "forecast_step",
                            "category": "model",
                            "start_ms": 0.0,
                            "end_ms": 9.8,
                        }
                    ],
                },
                {
                    "name": "GPU",
                    "spans": [
                        {
                            "label": "forward kernels",
                            "category": "model",
                            "start_ms": 1.0,
                            "end_ms": 8.5,
                        },
                        {
                            "label": "idle",
                            "category": "idle",
                            "start_ms": 8.5,
                            "end_ms": 10.0,
                        },
                    ],
                },
            ],
        },
        "forward": {
            "title": "Forward dominant kernels",
            "steps": [
                {
                    "step_id": "forecast_step#3",
                    "rank": 0,
                    "duration_ms": 10.0,
                    "forward_gpu_time_ms": 6.0,
                    "kernels": [
                        {
                            "name": "gemm_128x64",
                            "label": "gemm_128x64",
                            "family": "gemm",
                            "start_ms": 1.0,
                            "end_ms": 3.0,
                            "source_range": "model.forward",
                        },
                        {
                            "name": "fused_attention",
                            "label": "fused_attention",
                            "family": "attention",
                            "start_ms": 3.5,
                            "end_ms": 7.5,
                            "source_range": "model.forward",
                        },
                    ],
                    "dominant_kernels": [
                        {
                            "name": "fused_attention",
                            "family": "attention",
                            "call_count": 1,
                            "total_gpu_ms": 4.0,
                            "self_gpu_ms": 4.0,
                            "mean_gpu_ms": 4.0,
                            "median_gpu_ms": 4.0,
                            "p95_gpu_ms": 4.0,
                            "max_gpu_ms": 4.0,
                            "pct_forward_gpu_time": 66.67,
                            "pct_step_wall_time": 40.0,
                            "source_range": "model.forward",
                            "shapes": [[1, 64, 128]],
                            "provenance": "HTA fixture",
                        },
                        {
                            "name": "gemm_128x64",
                            "family": "gemm",
                            "call_count": 1,
                            "total_gpu_ms": 2.0,
                            "self_gpu_ms": 2.0,
                            "mean_gpu_ms": 2.0,
                            "median_gpu_ms": 2.0,
                            "p95_gpu_ms": 2.0,
                            "max_gpu_ms": 2.0,
                            "pct_forward_gpu_time": 33.33,
                            "pct_step_wall_time": 20.0,
                            "source_range": "model.forward",
                            "shapes": [],
                            "provenance": "HTA fixture",
                        },
                    ],
                }
            ],
        },
    }


def test_renderer_writes_detailed_kernel_tables_and_diagrams(tmp_path: Path) -> None:
    source = tmp_path / "diagram.json"
    source.write_text(json.dumps(diagram_document()))
    output = tmp_path / "hta"
    run("render_trace_diagrams.py", source, "--output-dir", output)
    for name in (
        "pipeline.svg",
        "dominant-kernels.svg",
        "dominant-kernels.json",
        "dominant-kernels.csv",
        "dominant-kernels.md",
    ):
        assert (output / name).stat().st_size > 0
    rows = list(csv.DictReader((output / "dominant-kernels.csv").open()))
    assert rows[0]["name"] == "fused_attention"
    assert rows[0]["p95_gpu_ms"] == "4.0"
    assert rows[0]["source_range"] == "model.forward"
    assert "launch order and gaps" in (output / "dominant-kernels.svg").read_text()


def test_trace_annotation_health_accepts_explicit_forecast_step(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "forecast_step#3", "ph": "X", "ts": 0, "dur": 100},
                    {"name": "output_write", "ph": "X", "ts": 80, "dur": 10},
                ]
            }
        )
    )
    output = tmp_path / "health.json"
    run(
        "validate_trace_annotations.py",
        trace,
        "--output",
        output,
        "--required-phase",
        "output_write",
    )
    health = json.loads(output.read_text())
    assert health["healthy"] is True
    assert health["logical_boundary"]["provenance"] == "explicit"


def test_benchmark_summary_reports_distribution(tmp_path: Path) -> None:
    source = tmp_path / "samples.json"
    source.write_text(json.dumps({"steady_step_ms": [10, 11, 9, 10]}))
    output = tmp_path / "summary.json"
    run("summarize_benchmarks.py", source, "--output", output)
    metric = json.loads(output.read_text())["metrics"]["steady_step_ms"]
    assert metric["count"] == 4
    assert metric["median"] == 10.0
    assert metric["p95"] > 10.0
