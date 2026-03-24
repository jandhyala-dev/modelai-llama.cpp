#!/usr/bin/env python3
"""
Benchmark Regression Check — Automated Comparison Against Baseline

Compares benchmark results from an upstream-sync run against the most recent
modelai-main baseline. Outputs PASS/FAIL per tier with specific regression
percentages.

Supports both results.csv (v1 schema) and results.json (v3 schema).

Usage:
  python3 scripts/bench-regression-check.py \
    --baseline bench-results/20260312-215927-c354fecf \
    --current  bench-results/upstream-sync/tier2-throughput \
    --out-dir  bench-results/upstream-sync

Or auto-detect baseline:
  python3 scripts/bench-regression-check.py \
    --current bench-results/upstream-sync/tier2-throughput
"""

import argparse
import csv
import glob
import json
import os
import sys
import time


# ── Regression thresholds (from plan section 7.1.4) ─────────────────

THRESHOLDS = {
    # Tier 1: Correctness (absolute thresholds)
    "recall_2x": {"min": 0.90, "label": "Fact recall @ 2x"},
    "recall_4x": {"min": 0.80, "label": "Fact recall @ 4x"},
    "recall_10x": {"min": 0.60, "label": "Fact recall @ 10x"},
    "recall_25x": {"min": 0.40, "label": "Fact recall @ 25x"},
    "recall_50x": {"min": 0.20, "label": "Fact recall @ 50x"},
    "needle_retrieval": {"min": 0.80, "label": "Needle retrieval rate"},

    # Tier 2: Performance (relative to baseline — max regression %)
    "prefill_tok_s": {"max_regression_pct": 10, "label": "Prefill throughput"},
    "decode_tok_s": {"max_regression_pct": 10, "label": "Decode throughput"},
    "compaction_time_ms": {"max_regression_pct": 10, "label": "Compaction latency"},
    "peak_rss_mb": {"max_regression_pct": 15, "label": "Peak memory"},
}


def find_latest_baseline(bench_dir):
    """Find the most recent modelai-main benchmark run directory."""
    candidates = []
    for d in glob.glob(os.path.join(bench_dir, "20*")):
        if os.path.isdir(d) and not d.endswith("upstream-sync"):
            manifest = os.path.join(d, "manifest.json")
            if os.path.exists(manifest):
                candidates.append(d)
    if not candidates:
        return None
    # Sort by directory name (timestamp-based)
    candidates.sort(reverse=True)
    return candidates[0]


def load_csv_results(csv_path):
    """Load results from a v1 schema results.csv.

    Validates column count consistency — skips rows where the number of values
    doesn't match the header. This catches the bench-results sanitization
    corruption where 6 fields were eaten by the PII replacement.
    """
    results = []
    skipped = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        n_fields = len(reader.fieldnames) if reader.fieldnames else 0
        for row in reader:
            # DictReader sets missing fields to None — detect misalignment
            none_count = sum(1 for v in row.values() if v is None)
            if none_count > 0:
                skipped += 1
                continue
            results.append(row)
    if skipped > 0:
        sys.stdout.write(f"  WARNING: {csv_path}: skipped {skipped} rows with column misalignment\n")
        sys.stdout.flush()
    return results


def load_json_results(json_path):
    """Load results from a v3 schema results.json."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])


def load_results(directory):
    """Auto-detect and load benchmark results from a directory."""
    # Try results.json first (v3 schema)
    json_path = os.path.join(directory, "results.json")
    if os.path.exists(json_path):
        return load_json_results(json_path), "json"

    # Try results.csv (v1 schema)
    csv_path = os.path.join(directory, "results.csv")
    if os.path.exists(csv_path):
        return load_csv_results(csv_path), "csv"

    # Try nested directories (tier results)
    all_results = []
    for subdir in sorted(glob.glob(os.path.join(directory, "*"))):
        if os.path.isdir(subdir):
            sub_results, _ = load_results(subdir)
            all_results.extend(sub_results)

    # Try JSON files directly in directory
    for jf in sorted(glob.glob(os.path.join(directory, "*.json"))):
        if os.path.basename(jf) != "manifest.json":
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "results" in data:
                all_results.extend(data["results"])
            elif isinstance(data, list):
                all_results.extend(data)

    return all_results, "mixed"


def extract_metrics(results, schema_type):
    """Extract comparable metrics from results."""
    metrics = {}
    for row in results:
        model = row.get("model_name") or row.get("model", "unknown")
        pipeline = row.get("pipeline") or row.get("engine", "unknown")
        ratio_raw = row.get("compression_ratio", 1)
        try:
            ratio = float(ratio_raw)
        except (ValueError, TypeError):
            ratio = 1.0

        key_prefix = f"{model}|{pipeline}|{ratio:.0f}x"

        for field in ["prefill_tok_s", "decode_tok_s", "compaction_time_ms",
                       "peak_rss_mb", "quality_accuracy"]:
            val = row.get(field)
            if val is not None:
                try:
                    metrics[f"{key_prefix}|{field}"] = float(val)
                except (ValueError, TypeError):
                    pass

    return metrics


def compare_metrics(baseline_metrics, current_metrics):
    """Compare current metrics against baseline. Return list of findings."""
    findings = []

    for key, current_val in current_metrics.items():
        if key not in baseline_metrics:
            continue
        baseline_val = baseline_metrics[key]
        if baseline_val == 0:
            continue

        field = key.split("|")[-1]
        delta_pct = ((current_val - baseline_val) / abs(baseline_val)) * 100

        # For throughput: regression = current < baseline (negative delta is bad)
        # For latency/memory: regression = current > baseline (positive delta is bad)
        is_higher_better = field in ("prefill_tok_s", "decode_tok_s", "quality_accuracy")
        regression_pct = -delta_pct if is_higher_better else delta_pct

        threshold_key = field
        threshold = THRESHOLDS.get(threshold_key)
        if threshold and "max_regression_pct" in threshold:
            passed = regression_pct <= threshold["max_regression_pct"]
        else:
            passed = True  # No threshold defined — informational only

        findings.append({
            "key": key,
            "field": field,
            "baseline": round(baseline_val, 2),
            "current": round(current_val, 2),
            "delta_pct": round(delta_pct, 1),
            "regression_pct": round(regression_pct, 1),
            "passed": passed,
            "label": threshold["label"] if threshold else field,
        })

    return findings


def load_tier1_results(directory):
    """Load Tier 1 results (correctness) from needle/recall JSON files."""
    results = {}

    # Needle-in-haystack
    for jf in glob.glob(os.path.join(directory, "**/needle-haystack-results.json"), recursive=True):
        with open(jf) as f:
            data = json.load(f)
        summary = data.get("summary", {})
        results["needle_retrieval"] = summary.get("retrieval_pct", 0) / 100.0

    # Compaction recall (from bench-compaction-1m.py output)
    for jf in glob.glob(os.path.join(directory, "**/compaction-quality-benchmark.json"), recursive=True):
        with open(jf) as f:
            data = json.load(f)
        for phase1 in data.get("phase1", []):
            ratio = phase1.get("ratio")
            post = phase1.get("post_recall", {})
            pct = post.get("pct", 0) / 100.0
            results[f"recall_{ratio}x"] = pct

    return results


def check_tier1(results):
    """Check Tier 1 correctness thresholds."""
    findings = []
    for key, value in results.items():
        threshold = THRESHOLDS.get(key)
        if threshold and "min" in threshold:
            passed = value >= threshold["min"]
            findings.append({
                "key": key,
                "label": threshold["label"],
                "value": round(value, 3),
                "threshold": threshold["min"],
                "passed": passed,
            })
    return findings


def main():
    parser = argparse.ArgumentParser(description="Benchmark regression checker")
    parser.add_argument("--baseline", default=None, help="Baseline results directory (auto-detect if omitted)")
    parser.add_argument("--current", required=True, help="Current results directory")
    parser.add_argument("--out-dir", default=None, help="Output directory for verdict")
    parser.add_argument("--bench-dir", default="bench-results", help="Root bench-results directory")
    parser.add_argument("--branch", default="upstream-sync", help="Branch name for verdict metadata")
    parser.add_argument("--commit", default="unknown", help="Commit SHA for verdict metadata")
    args = parser.parse_args()

    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("BENCHMARK REGRESSION CHECK\n")
    sys.stdout.write("=" * 60 + "\n\n")
    sys.stdout.flush()

    # Find baseline
    baseline_dir = args.baseline
    if not baseline_dir:
        baseline_dir = find_latest_baseline(args.bench_dir)
        if not baseline_dir:
            sys.stdout.write("ERROR: No baseline found in bench-results/\n")
            sys.exit(1)
    sys.stdout.write(f"Baseline: {baseline_dir}\n")
    sys.stdout.write(f"Current:  {args.current}\n\n")
    sys.stdout.flush()

    # ── Tier 1: Correctness ──
    sys.stdout.write("─── Tier 1: Compaction Correctness ───\n")
    tier1_results = load_tier1_results(args.current)
    tier1_findings = check_tier1(tier1_results)
    tier1_pass = True

    if tier1_findings:
        for f in tier1_findings:
            status = "PASS" if f["passed"] else "FAIL"
            sys.stdout.write(f"  {status}: {f['label']} = {f['value']:.1%} (threshold: {f['threshold']:.0%})\n")
            if not f["passed"]:
                tier1_pass = False
    else:
        sys.stdout.write("  No Tier 1 results found (skipping)\n")
        tier1_pass = None

    sys.stdout.write("\n")

    # ── Tier 2: Performance Regression ──
    sys.stdout.write("─── Tier 2: Performance Regression ───\n")
    baseline_results, baseline_type = load_results(baseline_dir)
    current_results, current_type = load_results(args.current)

    baseline_metrics = extract_metrics(baseline_results, baseline_type)
    current_metrics = extract_metrics(current_results, current_type)

    tier2_findings = compare_metrics(baseline_metrics, current_metrics)
    tier2_pass = True

    if tier2_findings:
        for f in tier2_findings:
            status = "PASS" if f["passed"] else "FAIL"
            direction = "+" if f["delta_pct"] > 0 else ""
            sys.stdout.write(
                f"  {status}: {f['key']}: "
                f"{f['baseline']} -> {f['current']} ({direction}{f['delta_pct']}%)\n"
            )
            if not f["passed"]:
                tier2_pass = False
    else:
        sys.stdout.write("  No comparable metrics found (skipping)\n")
        tier2_pass = None

    sys.stdout.write("\n")

    # ── Overall Verdict ──
    # Both tiers must pass (or be skipped) for overall PASS
    blocking_fail = (tier1_pass is False) or (tier2_pass is False)
    overall = "FAIL" if blocking_fail else "PASS"

    sys.stdout.write("─── VERDICT ───\n")
    sys.stdout.write(f"Tier 1 (Correctness):  {tier1_pass if tier1_pass is not None else 'SKIPPED'}\n")
    sys.stdout.write(f"Tier 2 (Performance):  {tier2_pass if tier2_pass is not None else 'SKIPPED'}\n")
    sys.stdout.write(f"Overall: {overall}\n")
    sys.stdout.flush()

    # Save verdict
    verdict = {
        "benchmark": "regression-check",
        "branch": args.branch,
        "commit": args.commit,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_dir": baseline_dir,
        "current_dir": args.current,
        "tier1": {
            "passed": tier1_pass,
            "findings": tier1_findings,
        },
        "tier2": {
            "passed": tier2_pass,
            "findings": tier2_findings,
        },
        "overall": overall,
    }

    out_dir = args.out_dir or args.current
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, "BENCHMARK-VERDICT.json")
    with open(outpath, "w") as f:
        json.dump(verdict, f, indent=2)

    # Also write markdown verdict
    md_path = os.path.join(out_dir, "BENCHMARK-VERDICT.md")
    with open(md_path, "w") as f:
        f.write(f"# Benchmark Verdict: {args.branch}\n\n")
        f.write(f"**Verdict:** {overall}\n")
        f.write(f"**Branch:** `{args.branch}`\n")
        f.write(f"**Commit:** `{args.commit}`\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d')}\n")
        f.write(f"**Baseline:** `{os.path.basename(baseline_dir)}`\n\n")

        f.write(f"## Tier 1: Compaction Correctness — {'PASS' if tier1_pass else 'FAIL' if tier1_pass is False else 'SKIPPED'}\n\n")
        for finding in tier1_findings:
            status = "PASS" if finding["passed"] else "FAIL"
            f.write(f"- {status}: {finding['label']} = {finding['value']:.1%} (threshold: {finding['threshold']:.0%})\n")
        if not tier1_findings:
            f.write("- No Tier 1 results found\n")

        f.write(f"\n## Tier 2: Performance Regression — {'PASS' if tier2_pass else 'FAIL' if tier2_pass is False else 'SKIPPED'}\n\n")
        for finding in tier2_findings:
            status = "PASS" if finding["passed"] else "FAIL"
            direction = "+" if finding["delta_pct"] > 0 else ""
            f.write(f"- {status}: {finding['label']}: {direction}{finding['delta_pct']}%\n")
        if not tier2_findings:
            f.write("- No comparable metrics found\n")

        f.write(f"\n## Tier 3: Cross-Model Validation — informational\n\n")
        f.write("- See individual tier3-* result files\n")

    sys.stdout.write(f"\nVerdict saved: {outpath}\n")
    sys.stdout.write(f"Markdown:     {md_path}\n")
    sys.stdout.flush()

    sys.exit(0 if not blocking_fail else 1)


if __name__ == "__main__":
    main()
