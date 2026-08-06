#!/usr/bin/env python3
"""Bind explicit user confirmation to the current inference test configuration."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--confirmation-source", default="explicit user confirmation in conversation"
    )
    args = parser.parse_args()
    bundle = args.bundle.expanduser().resolve()
    config_path = bundle / "test-config.json"
    confirmation_path = bundle / "config-confirmation.json"
    try:
        config = load(config_path)
        confirmation = load(confirmation_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if config.get("schema_version") != "0.1" or confirmation.get(
        "run_id"
    ) != config.get("run_id"):
        print("error: incompatible config/confirmation", file=sys.stderr)
        return 2
    resolved = config.get("workload", {}).get("resolved_config_artifact")
    expected = config.get("workload", {}).get("resolved_config_sha256")
    if resolved and sha256(bundle / resolved) != expected:
        print(
            "error: resolved configuration changed; regenerate test configuration",
            file=sys.stderr,
        )
        return 2
    confirmation.update(
        {
            "status": "confirmed",
            "confirmed_config_sha256": sha256(config_path),
            "confirmed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "confirmation_source": args.confirmation_source.strip(),
        }
    )
    confirmation_path.write_text(
        json.dumps(confirmation, indent=2, sort_keys=True) + "\n"
    )
    print(confirmation["confirmed_config_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
