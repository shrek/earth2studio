#!/usr/bin/env python3
"""Summarize raw inference timing samples without third-party dependencies."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(
            "samples must be a non-empty list of finite non-negative values"
        )
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p5": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="JSON object mapping metric names to sample arrays"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        raw: Any = json.loads(args.input.read_text())
        if not isinstance(raw, dict):
            raise ValueError("input must be an object")
        summary = {
            "schema_version": "0.1",
            "source": str(args.input),
            "metrics": {
                name: describe([float(value) for value in values])
                for name, values in raw.items()
                if isinstance(name, str) and isinstance(values, list)
            },
        }
        if len(summary["metrics"]) != len(raw):
            raise ValueError("every metric must map to a sample array")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
