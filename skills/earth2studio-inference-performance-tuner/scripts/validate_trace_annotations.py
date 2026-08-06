#!/usr/bin/env python3
"""Validate logical-unit and canonical phase ranges in a Chrome/Kineto trace."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

BOUNDARIES = (
    "ProfilerStep#",
    "forecast_step",
    "inference_step",
    "sample",
    "ensemble_batch",
    "work_item",
)
COMPUTE_PHASES = (
    "forecast_step",
    "diagnostic",
    "sampler_or_denoising_step",
    "assimilation",
)


def load(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def complete_events(document: Any) -> list[dict[str, Any]]:
    events = document.get("traceEvents") if isinstance(document, dict) else document
    if not isinstance(events, list):
        raise ValueError("trace must contain a traceEvents array")
    return [
        event for event in events if isinstance(event, dict) and event.get("ph") == "X"
    ]


def finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def matches(name: str, pattern: str) -> bool:
    return (
        name == pattern
        or name.startswith(pattern + ":")
        or name.startswith(pattern + "#")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--required-phase", action="append", default=[])
    args = parser.parse_args()
    try:
        events = complete_events(load(args.trace))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    invalid = [
        event.get("name", "")
        for event in events
        if not finite_positive(event.get("dur"))
    ]
    boundaries = [
        event
        for event in events
        if isinstance(event.get("name"), str)
        and any(event["name"].startswith(prefix) for prefix in BOUNDARIES)
        and finite_positive(event.get("dur"))
    ]
    names = [str(event.get("name", "")) for event in events]
    required = list(dict.fromkeys(args.required_phase))
    missing = [
        phase for phase in required if not any(matches(name, phase) for name in names)
    ]
    compute_present = [
        phase for phase in COMPUTE_PHASES if any(matches(name, phase) for name in names)
    ]
    native = any(str(event["name"]).startswith("ProfilerStep#") for event in boundaries)
    provenance = "native" if native else ("explicit" if boundaries else "missing")
    healthy = bool(boundaries) and not invalid and not missing and bool(compute_present)
    result = {
        "schema_version": "0.1",
        "variant": args.variant,
        "trace": str(args.trace),
        "healthy": healthy,
        "logical_boundary": {
            "provenance": provenance,
            "count": len(boundaries),
            "names": sorted({str(event["name"]) for event in boundaries}),
        },
        "required_phases": required,
        "missing_required_phases": missing,
        "compute_phases_present": compute_present,
        "invalid_complete_event_count": len(invalid),
        "recapture_required": not healthy,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not healthy:
        print("trace annotation health check failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
