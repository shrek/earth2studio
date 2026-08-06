#!/usr/bin/env python3
"""Validate an Earth2Studio inference performance-analysis artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def load(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON {path.name}: {exc}")
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_files(bundle: Path, paths: tuple[str, ...], errors: list[str]) -> None:
    for relative in paths:
        path = bundle / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty required artifact: {relative}")


def validate_confirmation(bundle: Path, errors: list[str]) -> None:
    config = load(bundle / "test-config.json", errors)
    confirmation = load(bundle / "config-confirmation.json", errors)
    if not isinstance(config, dict) or not isinstance(confirmation, dict):
        return
    if confirmation.get("status") != "confirmed":
        errors.append("configuration is not explicitly confirmed")
    elif confirmation.get("confirmed_config_sha256") != sha256(
        bundle / "test-config.json"
    ):
        errors.append("configuration confirmation fingerprint is stale")
    workload = config.get("workload", {})
    resolved = workload.get("resolved_config_artifact")
    expected = workload.get("resolved_config_sha256")
    if resolved:
        path = bundle / resolved
        if not path.is_file() or sha256(path) != expected:
            errors.append("resolved configuration snapshot is missing or changed")


def validate_ready(bundle: Path, errors: list[str]) -> None:
    require_files(
        bundle,
        (
            "correctness.json",
            "baseline/summary.json",
            "hta/pipeline.json",
            "hta/pipeline.svg",
            "hta/dominant-kernels.json",
            "hta/dominant-kernels.csv",
            "hta/dominant-kernels.md",
            "hta/dominant-kernels.svg",
            "phase-source-map.json",
            "source-analysis.json",
            "findings.json",
            "ncu/decision.json",
            "report.md",
        ),
        errors,
    )
    health_paths = (
        sorted((bundle / "traces").glob("annotation-health-*.json"))
        if (bundle / "traces").is_dir()
        else []
    )
    if not health_paths:
        errors.append("no trace annotation-health artifact found")
    for path in health_paths:
        health = load(path, errors)
        if isinstance(health, dict) and not health.get("healthy"):
            errors.append(f"trace health is not usable: {path.relative_to(bundle)}")
    correctness = load(bundle / "correctness.json", errors)
    if isinstance(correctness, dict) and correctness.get("status") != "pass":
        errors.append("correctness status must be pass")
    findings = load(bundle / "findings.json", errors)
    rows = findings.get("findings") if isinstance(findings, dict) else findings
    if not isinstance(rows, list):
        errors.append("findings.json must be an array or contain a findings array")
    else:
        for index, finding in enumerate(rows):
            if (
                not isinstance(finding, dict)
                or finding.get("status") != "recommendation_only"
            ):
                errors.append(f"finding {index} must have status recommendation_only")
            if isinstance(finding, dict) and not finding.get("source_analysis_ids"):
                errors.append(f"finding {index} must link source_analysis_ids")
    decision = load(bundle / "ncu/decision.json", errors)
    if isinstance(decision, dict):
        if decision.get("needed") is True and decision.get("status") != "captured":
            errors.append("NCU was needed but capture status is not captured")
        if decision.get("needed") is False and not decision.get("reason"):
            errors.append("NCU skip decision needs a reason")
    kernel_data = load(bundle / "hta/dominant-kernels.json", errors)
    if isinstance(kernel_data, dict) and not kernel_data.get("rows"):
        errors.append("dominant-kernels.json must contain detailed rows")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--confirmed", action="store_true")
    parser.add_argument("--ready", action="store_true")
    args = parser.parse_args()
    bundle = args.bundle.expanduser().resolve()
    errors: list[str] = []
    require_files(
        bundle,
        (
            "manifest.json",
            "test-config.json",
            "test-config.md",
            "config-confirmation.json",
        ),
        errors,
    )
    if not errors and (args.confirmed or args.ready):
        validate_confirmation(bundle, errors)
    if args.ready:
        validate_ready(bundle, errors)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("bundle validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
