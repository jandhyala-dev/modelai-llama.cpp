#!/usr/bin/env python3
"""Merge and analyze KV compaction benchmark results across run directories.

Usage:
    python3 scripts/analyze-bench-results.py [bench-results/]
    python3 scripts/analyze-bench-results.py bench-results/ --csv merged.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_all_results(base_dir: str) -> list[dict]:
    """Load and merge all results.csv files from run directories."""
    rows = []
    for run_dir in sorted(Path(base_dir).iterdir()):
        csv_path = run_dir / "results.csv"
        if not csv_path.is_file():
            continue
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_source_dir"] = run_dir.name
                rows.append(row)
    return rows


def dedup_rows(rows: list[dict]) -> list[dict]:
    """Keep latest run for each (model, pipeline, n_ctx, ratio, workload) tuple."""
    seen: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row.get("model_name", ""),
            row.get("pipeline", ""),
            row.get("n_ctx", ""),
            row.get("compression_ratio", ""),
            row.get("workload_id", ""),
        )
        # Later directories (sorted by timestamp) overwrite earlier ones.
        seen[key] = row
    return list(seen.values())


def safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def print_quality_table(rows: list[dict]) -> None:
    """Print pipeline x context x ratio -> cosine table."""
    print("\n=== Quality Table: logit_cosine (select pipeline) ===\n")

    # Collect select rows grouped by model.
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("pipeline") == "select":
            by_model[r.get("model_name", "unknown")].append(r)

    for model, model_rows in sorted(by_model.items()):
        print(f"Model: {model}")

        # Collect all contexts and ratios.
        contexts = sorted(set(int(r["n_ctx"]) for r in model_rows))
        ratios = sorted(set(int(r["compression_ratio"]) for r in model_rows))

        # Header.
        hdr = f"{'ctx':>8}"
        for ratio in ratios:
            hdr += f"  {ratio:>6}x"
        print(hdr)

        for ctx in contexts:
            line = f"{ctx:>8}"
            for ratio in ratios:
                matches = [
                    r for r in model_rows
                    if int(r["n_ctx"]) == ctx and int(r["compression_ratio"]) == ratio
                ]
                if matches:
                    cos = safe_float(matches[0].get("logit_cosine", ""))
                    passed = matches[0].get("pass", "false") == "true"
                    marker = " " if passed else "*"
                    line += f"  {cos:>5.3f}{marker}"
                else:
                    line += f"  {'—':>6}"
            print(line)
        print("  (* = below threshold)\n")


def print_throughput_table(rows: list[dict]) -> None:
    """Print throughput delta table for select pipeline."""
    print("\n=== Throughput Table: delta % vs in-run baseline (select) ===\n")

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("pipeline") == "select":
            by_model[r.get("model_name", "unknown")].append(r)

    for model, model_rows in sorted(by_model.items()):
        print(f"Model: {model}")

        contexts = sorted(set(int(r["n_ctx"]) for r in model_rows))
        ratios = sorted(set(int(r["compression_ratio"]) for r in model_rows))

        hdr = f"{'ctx':>8}"
        for ratio in ratios:
            hdr += f"  {ratio:>7}x"
        print(hdr)

        for ctx in contexts:
            line = f"{ctx:>8}"
            for ratio in ratios:
                matches = [
                    r for r in model_rows
                    if int(r["n_ctx"]) == ctx and int(r["compression_ratio"]) == ratio
                ]
                if matches:
                    delta = safe_float(matches[0].get("throughput_delta_pct", ""))
                    line += f"  {delta:>+7.1f}%"
                else:
                    line += f"  {'—':>8}"
            print(line)
        print()


def print_pipeline_comparison(rows: list[dict]) -> None:
    """Compare quality across pipelines at 4K context."""
    print("\n=== Pipeline Comparison at 4K (logit_cosine) ===\n")

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("pipeline") not in ("baseline",) and int(r.get("n_ctx", "0")) == 4096:
            by_model[r.get("model_name", "unknown")].append(r)

    for model, model_rows in sorted(by_model.items()):
        print(f"Model: {model}")
        pipelines = sorted(set(r["pipeline"] for r in model_rows))
        ratios = sorted(set(int(r["compression_ratio"]) for r in model_rows))

        hdr = f"{'pipeline':>12}"
        for ratio in ratios:
            hdr += f"  {ratio:>6}x"
        print(hdr)

        for pipe in pipelines:
            line = f"{pipe:>12}"
            for ratio in ratios:
                matches = [
                    r for r in model_rows
                    if r["pipeline"] == pipe and int(r["compression_ratio"]) == ratio
                ]
                if matches:
                    cos = safe_float(matches[0].get("logit_cosine", ""))
                    line += f"  {cos:>5.3f} "
                else:
                    line += f"  {'—':>6}"
            print(line)
        print()


def print_quality_eval(rows: list[dict]) -> None:
    """Print QuALITY MC evaluation results."""
    quality_rows = [
        r for r in rows
        if r.get("quality_total", "") and int(r.get("quality_total", "0")) > 0
    ]

    if not quality_rows:
        print("\n=== QuALITY MC Evaluation: no data ===\n")
        return

    print("\n=== QuALITY MC Evaluation ===\n")
    print(f"{'model':>30}  {'pipeline':>8}  {'ctx':>6}  {'ratio':>5}  {'correct':>7}  {'total':>5}  {'accuracy':>8}  {'baseline_acc':>12}")

    for r in sorted(quality_rows, key=lambda x: (x["model_name"], x["pipeline"], int(x["n_ctx"]), int(x["compression_ratio"]))):
        print(
            f"{r['model_name']:>30}  {r['pipeline']:>8}  {r['n_ctx']:>6}  "
            f"{r['compression_ratio']:>5}  {r.get('quality_correct',''):>7}  "
            f"{r.get('quality_total',''):>5}  {r.get('quality_accuracy',''):>8}  "
            f"{r.get('quality_baseline_accuracy',''):>12}"
        )
    print()


def print_summary_stats(rows: list[dict]) -> None:
    """Print overall summary statistics."""
    total = len(rows)
    passed = sum(1 for r in rows if r.get("pass") == "true")
    failed = sum(1 for r in rows if r.get("pass") == "false")
    crashed = sum(1 for r in rows if r.get("crash") == "true")
    models = len(set(r.get("model_name", "") for r in rows))
    pipelines = sorted(set(r.get("pipeline", "") for r in rows))

    print("\n=== Summary ===\n")
    print(f"Total rows:    {total}")
    print(f"Models:        {models}")
    print(f"Pipelines:     {', '.join(pipelines)}")
    print(f"Passed:        {passed}")
    print(f"Failed:        {failed}")
    print(f"Crashed:       {crashed}")
    print()


def write_merged_csv(rows: list[dict], output_path: str) -> None:
    """Write merged and deduped rows to a single CSV."""
    if not rows:
        return
    # Use field order from first row, excluding internal fields.
    fields = [k for k in rows[0].keys() if not k.startswith("_")]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged CSV written: {output_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Analyze KV compaction benchmark results")
    parser.add_argument("base_dir", nargs="?", default="bench-results",
                        help="Base directory containing run directories (default: bench-results)")
    parser.add_argument("--csv", metavar="PATH", help="Write merged CSV to this path")
    parser.add_argument("--no-tables", action="store_true", help="Skip table output")
    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        print(f"ERROR: {args.base_dir} not found", file=sys.stderr)
        sys.exit(1)

    rows = load_all_results(args.base_dir)
    if not rows:
        print(f"No results found in {args.base_dir}/*/results.csv")
        sys.exit(0)

    print(f"Loaded {len(rows)} rows from {args.base_dir}")

    rows = dedup_rows(rows)
    print(f"After dedup: {len(rows)} unique (model, pipeline, ctx, ratio, workload) rows")

    if args.csv:
        write_merged_csv(rows, args.csv)

    if not args.no_tables:
        print_summary_stats(rows)
        print_quality_table(rows)
        print_throughput_table(rows)
        print_pipeline_comparison(rows)
        print_quality_eval(rows)


if __name__ == "__main__":
    main()
