#!/usr/bin/env python3
"""Render inference pipeline and detailed forward dominant-kernel artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1"
WIDTH = 1500
LEFT = 230
RIGHT = 220
PLOT = WIDTH - LEFT - RIGHT
MIN_DISPLAY_WIDTH = 1.2
COLORS = (
    "#4e79a7",
    "#59a14f",
    "#f28e2b",
    "#af7aa1",
    "#76b7b2",
    "#edc948",
    "#e15759",
    "#7b35de",
    "#0b996e",
    "#66788a",
)
FIELDS = (
    "step_id",
    "rank",
    "kernel_rank",
    "name",
    "family",
    "call_count",
    "total_gpu_ms",
    "self_gpu_ms",
    "mean_gpu_ms",
    "median_gpu_ms",
    "p95_gpu_ms",
    "max_gpu_ms",
    "pct_forward_gpu_time",
    "pct_step_wall_time",
    "source_range",
    "shapes",
    "provenance",
)


class InputError(ValueError):
    pass


def number(value: Any, label: str, *, nonnegative: bool = True) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise InputError(f"{label} must be finite and numeric")
    result = float(value)
    if nonnegative and result < 0:
        raise InputError(f"{label} must be non-negative")
    return result


def validate_span(span: Any, duration: float, label: str) -> None:
    if not isinstance(span, dict) or not isinstance(
        span.get("label", span.get("name")), str
    ):
        raise InputError(f"{label} must contain a label/name")
    start = number(span.get("start_ms"), label + ".start_ms")
    end = number(span.get("end_ms"), label + ".end_ms")
    if end <= start or end > duration + 1e-9:
        raise InputError(f"{label} must satisfy 0 <= start < end <= duration")


def validate(document: Any) -> None:
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise InputError("schema_version must equal '0.1'")
    pipeline = document.get("pipeline")
    forward = document.get("forward")
    if not isinstance(pipeline, dict) or not isinstance(forward, dict):
        raise InputError("pipeline and forward objects are required")
    duration = number(pipeline.get("duration_ms"), "pipeline.duration_ms")
    if duration <= 0:
        raise InputError("pipeline duration must be positive")
    lanes = pipeline.get("lanes")
    if not isinstance(lanes, list) or len(lanes) < 2:
        raise InputError("pipeline needs CPU and GPU lanes")
    lane_names = set()
    for i, lane in enumerate(lanes):
        if not isinstance(lane, dict) or not isinstance(lane.get("name"), str):
            raise InputError(f"pipeline lane {i} is invalid")
        lane_names.add(lane["name"].upper())
        if not isinstance(lane.get("spans"), list):
            raise InputError(f"pipeline lane {i} spans must be a list")
        for j, span in enumerate(lane["spans"]):
            validate_span(span, duration, f"pipeline lane {i} span {j}")
    if not {"CPU", "GPU"}.issubset(lane_names):
        raise InputError("pipeline must include CPU and GPU lanes")
    steps = forward.get("steps")
    if not isinstance(steps, list) or not steps:
        raise InputError("forward.steps must be non-empty")
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise InputError(f"forward step {i} is invalid")
        step_duration = number(step.get("duration_ms"), f"forward step {i} duration")
        if step_duration <= 0:
            raise InputError(f"forward step {i} duration must be positive")
        kernels = step.get("kernels")
        rows = step.get("dominant_kernels")
        if (
            not isinstance(kernels, list)
            or not kernels
            or not isinstance(rows, list)
            or not rows
        ):
            raise InputError(f"forward step {i} needs kernels and dominant_kernels")
        for j, kernel in enumerate(kernels):
            validate_span(kernel, step_duration, f"forward step {i} kernel {j}")
            if not isinstance(kernel.get("family"), str) or not isinstance(
                kernel.get("name"), str
            ):
                raise InputError(f"forward step {i} kernel {j} needs name and family")
        for j, row in enumerate(rows):
            for field in (
                "name",
                "family",
                "call_count",
                "total_gpu_ms",
                "pct_forward_gpu_time",
                "pct_step_wall_time",
                "provenance",
            ):
                if field not in row:
                    raise InputError(f"forward step {i} row {j} missing {field}")
            number(row["call_count"], f"row {j} call_count")
            number(row["total_gpu_ms"], f"row {j} total_gpu_ms")


def xscale(value: float, duration: float) -> float:
    return LEFT + value / duration * PLOT


def svg_start(height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:25px;font-weight:700}.sub{font-size:13px;fill:#53657a}.lane{font-size:15px;font-weight:700}.small{font-size:11px}.grid{stroke:#dce3ec;stroke-width:1}</style>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="28" y="38" class="title">{escape(title)}</text>',
        f'<text x="28" y="62" class="sub">{escape(subtitle)}</text>',
    ]


def ticks(lines: list[str], duration: float, top: int, bottom: int) -> None:
    for index in range(6):
        value = duration * index / 5
        x = xscale(value, duration)
        lines.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="middle" class="small">{value:.2f} ms</text>'
        )


def write_svg(path: Path, lines: list[str]) -> None:
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def render_pipeline(document: dict[str, Any], path: Path) -> None:
    pipeline = document["pipeline"]
    duration = float(pipeline["duration_ms"])
    lanes = pipeline["lanes"]
    height = 130 + len(lanes) * 92
    lines = svg_start(
        height,
        pipeline.get("title", "CPU/GPU inference pipeline"),
        f"{pipeline.get('step_id', '')} · {duration:.2f} ms",
    )
    ticks(lines, duration, 84, height - 45)
    for index, lane in enumerate(lanes):
        y = 90 + index * 92
        lines.append(
            f'<text x="{LEFT - 14}" y="{y + 24}" text-anchor="end" class="lane">{escape(lane["name"])}</text>'
        )
        lines.append(
            f'<rect x="{LEFT}" y="{y}" width="{PLOT}" height="40" fill="#edf2f7" stroke="#94a3b8"/>'
        )
        for span_index, span in enumerate(lane["spans"]):
            start, end = float(span["start_ms"]), float(span["end_ms"])
            x, width = xscale(start, duration), max(
                1.2, xscale(end, duration) - xscale(start, duration)
            )
            color = (
                COLORS[span_index % len(COLORS)]
                if span.get("category") not in ("idle", "bubble")
                else "#d9d9d9"
            )
            lines.append(
                f'<rect x="{x:.1f}" y="{y}" width="{width:.1f}" height="40" fill="{color}" stroke="#fff"><title>{escape(str(span.get("label")))}: {end-start:.3f} ms</title></rect>'
            )
            if width > 75:
                lines.append(
                    f'<text x="{x + width / 2:.1f}" y="{y + 25}" text-anchor="middle" class="small" style="fill:#fff">{escape(str(span.get("label")))}</text>'
                )
    write_svg(path, lines)


def normalize_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    provenance = document.get("provenance", {})
    for step in document["forward"]["steps"]:
        ordered = sorted(
            step["dominant_kernels"],
            key=lambda row: (-float(row["total_gpu_ms"]), str(row["name"])),
        )
        for index, raw in enumerate(ordered, 1):
            row = {field: raw.get(field) for field in FIELDS}
            row.update(
                {
                    "step_id": step.get("step_id", ""),
                    "rank": step.get("rank", 0),
                    "kernel_rank": index,
                }
            )
            row["source_range"] = raw.get("source_range", "")
            shapes = raw.get("shapes", [])
            row["shapes"] = (
                json.dumps(shapes, separators=(",", ":"))
                if isinstance(shapes, (list, dict))
                else shapes
            )
            row["provenance"] = raw.get("provenance") or provenance.get(
                "step_boundary", "unknown"
            )
            output.append(row)
    return output


def write_tables(document: dict[str, Any], output: Path) -> None:
    rows = normalize_rows(document)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "provenance": document.get("provenance", {}),
        "rows": rows,
    }
    (output / "dominant-kernels.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with (output / "dominant-kernels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    headings = (
        "Rank",
        "Step",
        "GPU rank",
        "Exact kernel",
        "Family",
        "Calls",
        "Total ms",
        "Self ms",
        "Mean ms",
        "Median ms",
        "p95 ms",
        "Max ms",
        "% forward GPU",
        "% step wall",
        "Launch source/range",
        "Shapes",
        "Provenance",
    )
    keys = (
        "kernel_rank",
        "step_id",
        "rank",
        "name",
        "family",
        "call_count",
        "total_gpu_ms",
        "self_gpu_ms",
        "mean_gpu_ms",
        "median_gpu_ms",
        "p95_gpu_ms",
        "max_gpu_ms",
        "pct_forward_gpu_time",
        "pct_step_wall_time",
        "source_range",
        "shapes",
        "provenance",
    )
    lines = ["| " + " | ".join(headings) + " |", "|" + "---|" * len(headings)]
    for row in rows:
        values = [
            "—" if row[key] in (None, "") else str(row[key]).replace("|", "\\|")
            for key in keys
        ]
        lines.append("| " + " | ".join(values) + " |")
    (output / "dominant-kernels.md").write_text(
        "# Forward-pass dominant kernels\n\n" + "\n".join(lines) + "\n"
    )


def compact_display_intervals(
    kernels: list[dict[str, Any]], duration: float
) -> list[dict[str, Any]]:
    """Merge launches that overlap at the SVG's display-pixel resolution."""
    intervals: list[dict[str, Any]] = []
    for kernel in sorted(
        kernels, key=lambda item: (float(item["start_ms"]), float(item["end_ms"]))
    ):
        start_ms = float(kernel["start_ms"])
        end_ms = float(kernel["end_ms"])
        start_x = xscale(start_ms, duration)
        end_x = max(start_x + MIN_DISPLAY_WIDTH, xscale(end_ms, duration))
        if intervals and start_x <= float(intervals[-1]["end_x"]):
            current = intervals[-1]
            current["end_x"] = max(float(current["end_x"]), end_x)
            current["end_ms"] = max(float(current["end_ms"]), end_ms)
            current["launch_count"] = int(current["launch_count"]) + 1
        else:
            intervals.append(
                {
                    "start_x": start_x,
                    "end_x": end_x,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "launch_count": 1,
                }
            )
    return intervals


def render_kernels(document: dict[str, Any], path: Path) -> None:
    steps = document["forward"]["steps"]
    panels: list[tuple[dict[str, Any], list[str]]] = []
    total_height = 90
    for step in steps:
        families = sorted({kernel["family"] for kernel in step["kernels"]})
        panels.append((step, families))
        total_height += 105 + 34 * len(families)
    lines = svg_start(
        total_height,
        document["forward"].get("title", "Forward-pass dominant kernels"),
        "One lane per family; launch order and gaps preserved at display-pixel resolution",
    )
    top = 88
    for step, families in panels:
        duration = float(step["duration_ms"])
        gpu_time = float(
            step.get(
                "forward_gpu_time_ms",
                sum(float(k["end_ms"]) - float(k["start_ms"]) for k in step["kernels"]),
            )
        )
        panel_height = 82 + 34 * len(families)
        lines.append(
            f'<rect x="18" y="{top - 14}" width="{WIDTH - 36}" height="{panel_height}" rx="8" fill="#fff" stroke="#d8e0ea"/>'
        )
        lines.append(
            f'<text x="34" y="{top + 8}" class="lane">{escape(str(step.get("step_id", "forward")))} · rank {escape(str(step.get("rank", 0)))} · forward {duration:.2f} ms · GPU {gpu_time:.2f} ms</text>'
        )
        axis_top = top + 26
        ticks(lines, duration, axis_top, axis_top + 34 * len(families))
        totals: dict[str, float] = defaultdict(float)
        for kernel in step["kernels"]:
            totals[kernel["family"]] += float(kernel["end_ms"]) - float(
                kernel["start_ms"]
            )
        for family_index, family in enumerate(families):
            y = axis_top + family_index * 34
            share = 100 * totals[family] / gpu_time if gpu_time else 0
            lines.append(
                f'<text x="{LEFT - 12}" y="{y + 18}" text-anchor="end" class="small">{escape(family)} · {totals[family]:.2f} ms · {share:.1f}%</text>'
            )
            family_kernels = [
                item for item in step["kernels"] if item["family"] == family
            ]
            for interval in compact_display_intervals(family_kernels, duration):
                x = float(interval["start_x"])
                width = float(interval["end_x"]) - x
                color = COLORS[family_index % len(COLORS)]
                launch_count = int(interval["launch_count"])
                title = (
                    f"{family} | {launch_count} launch{'es' if launch_count != 1 else ''} | "
                    f"{float(interval['start_ms']):.4f}-{float(interval['end_ms']):.4f} ms | "
                    "compacted at display resolution; exact launches remain in pipeline.json"
                )
                lines.append(
                    f'<rect x="{x:.1f}" y="{y + 4}" width="{width:.1f}" height="20" fill="{color}" stroke="#fff"><title>{escape(title)}</title></rect>'
                )
        top += panel_height + 20
    write_svg(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        document = json.loads(args.input.read_text())
        validate(document)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        render_pipeline(document, args.output_dir / "pipeline.svg")
        render_kernels(document, args.output_dir / "dominant-kernels.svg")
        write_tables(document, args.output_dir)
    except (OSError, json.JSONDecodeError, InputError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
