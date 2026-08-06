#!/usr/bin/env python3
"""Create a fresh Earth2Studio inference performance-analysis bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1"


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--baseline-command", required=True)
    parser.add_argument("--candidate-command", default="")
    parser.add_argument("--comparison-label", default="baseline-only")
    parser.add_argument("--workload", default="Earth2Studio inference")
    parser.add_argument("--entry-point", default="")
    parser.add_argument("--model-config", default="")
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache-state", default="")
    parser.add_argument("--initialization-times", default="")
    parser.add_argument("--forecast-steps", default="")
    parser.add_argument("--variables", default="")
    parser.add_argument("--batch-size", default="")
    parser.add_argument("--ensemble-size", default="")
    parser.add_argument("--sample-count", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--precision", default="")
    parser.add_argument("--hardware", default="")
    parser.add_argument("--framework-stack", default="")
    parser.add_argument("--distributed", default="single-process")
    parser.add_argument("--io-config", default="")
    parser.add_argument("--checkpoint-config", default="")
    parser.add_argument("--compile-config", default="not-applicable")
    parser.add_argument(
        "--include-startup",
        action="store_true",
        help="Opt in to startup timing such as model loading and first output.",
    )
    parser.add_argument("--correctness-command", required=True)
    parser.add_argument("--performance-goal", required=True)
    parser.add_argument("--warmup-units", type=positive, default=5)
    parser.add_argument("--measure-units", type=positive, default=20)
    parser.add_argument("--profile-units", type=positive, default=5)
    parser.add_argument("--repetitions", type=positive, default=3)
    parser.add_argument("--run-id")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(workdir: Path, *args: str) -> str | None:
    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603
            [git_executable, *args],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def display(value: Any) -> str:
    if value in (None, "", []):
        return "UNKNOWN — resolve before confirmation"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def render_config(config: dict[str, Any]) -> str:
    w = config["workload"]
    r = config["runtime"]
    p = config["protocol"]
    rows = [
        ("Working directory", w["workdir"]),
        ("Baseline command", f"`{w['baseline_command'].replace('`', chr(39))}`"),
        (
            "Candidate command",
            (
                f"`{w['candidate_command'].replace('`', chr(39))}`"
                if w["candidate_command"]
                else "not applicable"
            ),
        ),
        ("Comparison", config["comparison_label"]),
        ("Revision", config["source"]["git_commit"]),
        ("Local changes", "dirty" if config["source"]["dirty"] else "clean"),
        ("Entry point", w["entry_point"]),
        ("Model/config", w["model_config"]),
        ("Resolved config", w["resolved_config_artifact"]),
        ("Data", config["data"]["identity"]),
        ("Cache state", config["data"]["cache_state"]),
        ("Initialization times", w["initialization_times"]),
        ("Forecast steps", w["forecast_steps"]),
        ("Variables", w["variables"]),
        (
            "Batch / ensemble / samples",
            f"{w['batch_size']} / {w['ensemble_size']} / {w['sample_count']}",
        ),
        ("Seed", w["seed"]),
        ("Hardware", r["hardware"]),
        ("Framework stack", r["framework_stack"]),
        ("Precision", r["precision"]),
        ("Distributed", r["distributed"]),
        ("Compile configuration", r["compile_config"]),
        (
            "Startup measurement",
            (
                "enabled by explicit request"
                if p["include_startup"]
                else "disabled (default)"
            ),
        ),
        ("IO configuration", r["io_config"]),
        ("Checkpoint configuration", r["checkpoint_config"]),
        (
            "Warmup / measured / profiled units",
            f"{p['warmup_units']} / {p['measure_units']} / {p['profile_units']}",
        ),
        ("Repetitions", p["repetitions"]),
        ("Correctness check", config["correctness"]["command"]),
        ("Performance goal", config["performance"]["goal"]),
        ("Artifact output", config["artifacts"]["output"]),
    ]
    body = "\n".join(f"| {name} | {display(value)} |" for name, value in rows)
    return (
        "# Test configuration — explicit confirmation required\n\n| Item | Value |\n|---|---|\n"
        + body
        + "\n"
    )


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    workdir = args.workdir.expanduser().resolve()
    if not workdir.is_dir():
        print(f"error: missing workdir: {workdir}", file=sys.stderr)
        return 2
    if (
        not args.baseline_command.strip()
        or not args.data.strip()
        or not args.correctness_command.strip()
    ):
        print(
            "error: command, data, and correctness check must be non-empty",
            file=sys.stderr,
        )
        return 2
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        print(
            f"error: output must be a new or empty directory: {output}", file=sys.stderr
        )
        return 2
    output.mkdir(parents=True, exist_ok=True)
    for name in ("baseline", "traces", "hta", "compile", "ncu", "logs"):
        (output / name).mkdir()

    resolved_name = None
    resolved_hash = None
    if args.resolved_config:
        source = args.resolved_config.expanduser().resolve()
        if not source.is_file():
            print(f"error: missing resolved config: {source}", file=sys.stderr)
            return 2
        resolved_name = "resolved-config" + (source.suffix or ".txt")
        shutil.copyfile(source, output / resolved_name)
        resolved_hash = sha256(output / resolved_name)

    now = dt.datetime.now(dt.timezone.utc)
    run_id = args.run_id or now.strftime("inference_perf_%Y%m%dT%H%M%SZ")
    status = git_value(workdir, "status", "--short")
    config: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "created_at_utc": now.isoformat(),
        "comparison_label": args.comparison_label,
        "source": {
            "git_commit": git_value(workdir, "rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_short": status.splitlines() if status else [],
        },
        "workload": {
            "name": args.workload,
            "workdir": str(workdir),
            "baseline_command": args.baseline_command,
            "candidate_command": args.candidate_command,
            "entry_point": args.entry_point,
            "model_config": args.model_config,
            "resolved_config_artifact": resolved_name,
            "resolved_config_sha256": resolved_hash,
            "initialization_times": args.initialization_times,
            "forecast_steps": args.forecast_steps,
            "variables": args.variables,
            "batch_size": args.batch_size,
            "ensemble_size": args.ensemble_size,
            "sample_count": args.sample_count,
            "seed": args.seed,
        },
        "data": {"identity": args.data, "cache_state": args.cache_state},
        "runtime": {
            "precision": args.precision,
            "hardware": args.hardware,
            "framework_stack": args.framework_stack,
            "distributed": args.distributed,
            "io_config": args.io_config,
            "checkpoint_config": args.checkpoint_config,
            "compile_config": args.compile_config,
        },
        "protocol": {
            "warmup_units": args.warmup_units,
            "measure_units": args.measure_units,
            "profile_units": args.profile_units,
            "repetitions": args.repetitions,
            "default_measurements": ["warm_end_to_end", "steady_state"],
            "include_startup": args.include_startup,
            "startup_measurement": (
                "enabled_by_explicit_request"
                if args.include_startup
                else "disabled_by_default"
            ),
            "profiled_timings_are_not_baselines": True,
            "async_io_completion_required": True,
        },
        "correctness": {"command": args.correctness_command},
        "performance": {"goal": args.performance_goal},
        "artifacts": {"output": str(output)},
    }
    write_json(output / "test-config.json", config)
    (output / "test-config.md").write_text(render_config(config))
    write_json(
        output / "config-confirmation.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "status": "pending",
            "confirmed_config_sha256": None,
        },
    )
    write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "state": "created",
            "required_execution_gate": "explicit user confirmation",
        },
    )
    print(output / "test-config.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
