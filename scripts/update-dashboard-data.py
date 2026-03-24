#!/usr/bin/env python3
"""Aggregate bench results into bench-results/history.jsonl for dashboard consumption.

Reads perf-baseline.json and bench-results/full-* directories, emitting one
JSONL record per model per run. Designed to run in CI after Tier 3/4 complete.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_baseline(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_full_run(run_dir: Path) -> list[dict]:
    """Parse a full-* benchmark directory into records."""
    records = []
    run_name = run_dir.name
    # Extract timestamp from directory name: full-YYYYMMDD-HHMMSS
    parts = run_name.split("-")
    if len(parts) >= 3:
        try:
            ts = datetime.strptime(f"{parts[1]}-{parts[2]}", "%Y%m%d-%H%M%S")
            ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            ts = datetime.now(timezone.utc)
    else:
        ts = datetime.now(timezone.utc)

    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        record = {
            "run": run_name,
            "timestamp": ts.isoformat(),
            "model": model_name,
        }

        # Parse modelai bench CSV
        modelai_csv = model_dir / "bench-modelai.csv"
        if modelai_csv.exists():
            record["modelai"] = _parse_bench_csv(modelai_csv)

        # Parse upstream bench CSV
        upstream_csv = model_dir / "bench-upstream.csv"
        if upstream_csv.exists():
            record["upstream"] = _parse_bench_csv(upstream_csv)

        records.append(record)

    return records


def _parse_bench_csv(path: Path) -> dict:
    """Extract key metrics from a bench CSV."""
    metrics = {}
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last = rows[-1]
                for key in ["t/s", "tokens", "time_ms", "n_batch", "n_ubatch"]:
                    if key in last and last[key]:
                        try:
                            metrics[key] = float(last[key])
                        except (ValueError, TypeError):
                            pass
    except Exception:
        pass
    return metrics


def collect_history(bench_dir: Path, baseline_path: Path) -> list[dict]:
    """Collect all full-* runs into history records."""
    records = []
    baseline = parse_baseline(baseline_path)

    for entry in sorted(bench_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("full-"):
            run_records = parse_full_run(entry)
            for r in run_records:
                r["baseline"] = baseline
            records.extend(run_records)

    return records


def main():
    parser = argparse.ArgumentParser(description="Update dashboard history data")
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=Path("bench-results"),
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("bench-results/perf-baseline.json"),
        help="Path to perf-baseline.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bench-results/history.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing history instead of overwriting",
    )
    args = parser.parse_args()

    if not args.bench_dir.exists():
        print(f"Bench directory not found: {args.bench_dir}", file=sys.stderr)
        sys.exit(1)

    records = collect_history(args.bench_dir, args.baseline)

    mode = "a" if args.append else "w"
    with open(args.output, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
