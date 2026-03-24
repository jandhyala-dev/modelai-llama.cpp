#!/usr/bin/env python3
"""Aggregate Phase D results into a summary table.

Reads per-model JSON results from a Phase D run directory, produces:
- Console summary table
- summary.json for Supabase import

Usage:
    python3 scripts/aggregate-phase-d-results.py bench-results/phase-d-YYYYMMDD-HHMMSS/
"""

import json
import sys
from pathlib import Path


def load_model_results(results_dir):
    """Load all per-model baseline + compaction results."""
    models = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name.split("-", 1)[1] if "-" in model_dir.name else model_dir.name

        baselines = list(model_dir.glob("*-baseline.json"))
        compacted = list(model_dir.glob("*-compacted-*.json"))
        skipped = list(model_dir.glob("*-compaction-skipped.json"))

        for bf in baselines:
            try:
                baseline = json.loads(bf.read_text())
            except Exception:
                continue

            ctx_size = baseline.get("context_size", 0)
            uc_results = baseline.get("use_case_results", [])
            recall = baseline.get("recall", {})

            # Find matching compaction result
            compact_data = None
            for cf in compacted:
                try:
                    cd = json.loads(cf.read_text())
                    if cd.get("context_size") == ctx_size:
                        compact_data = cd
                        break
                except Exception:
                    continue

            # Compute metrics
            uc_scores = [r.get("quality_score", 0) for r in uc_results if "quality_score" in r]
            avg_quality = sum(uc_scores) / len(uc_scores) if uc_scores else 0
            uc_errors = sum(1 for r in uc_results if "error" in r)
            recall_score = recall.get("recall_quality_score", 0)
            recall_words = recall.get("response_word_count", 0)
            gen_speeds = [r.get("generation_tok_s", 0) for r in uc_results if r.get("generation_tok_s", 0) > 0]
            avg_gen_speed = sum(gen_speeds) / len(gen_speeds) if gen_speeds else 0

            # Compaction metrics
            compact_time = compact_data.get("compaction_time_ms", 0) if compact_data else None
            compact_ratio = compact_data.get("compaction_ratio", 0) if compact_data else None
            compact_recall = compact_data.get("recall", {}).get("recall_quality_score", 0) if compact_data else None
            compact_recall_words = compact_data.get("recall", {}).get("response_word_count", 0) if compact_data else None
            compact_error = compact_data.get("error") if compact_data else None

            # Gates
            baseline_pass = avg_quality >= 3.0
            compact_pass = compact_recall is not None and compact_recall >= 2.0 if compact_data else None
            recall_pass = recall_score >= 3

            models.append({
                "model": model_name,
                "context_k": ctx_size // 1024,
                "use_cases": len(uc_results),
                "errors": uc_errors,
                "avg_quality": round(avg_quality, 1),
                "recall_score": recall_score,
                "recall_words": recall_words,
                "avg_tok_s": round(avg_gen_speed, 1),
                "compact_time_ms": round(compact_time) if compact_time else None,
                "compact_ratio": round(compact_ratio, 1) if compact_ratio else None,
                "compact_recall": compact_recall,
                "compact_recall_words": compact_recall_words,
                "compact_error": compact_error,
                "baseline_pass": baseline_pass,
                "compact_pass": compact_pass,
                "recall_pass": recall_pass,
                "skipped_compaction": len(skipped) > 0 and compact_data is None,
            })

    return models


def print_summary(models, run_dir):
    """Print console summary table."""
    print(f"\n{'=' * 100}")
    print(f"Phase D Summary — {run_dir.name}")
    print(f"{'=' * 100}")

    # Header
    print(f"{'Model':<30} {'Ctx':>4} {'UCs':>3} {'Err':>3} {'Qual':>5} {'Rcl':>4} "
          f"{'tok/s':>6} {'CmpMs':>6} {'CmpR':>5} {'CmpRcl':>6} {'Base':>5} {'Cmp':>5}")
    print("-" * 100)

    total_pass = 0
    total = len(models)
    for m in models:
        base_icon = "PASS" if m["baseline_pass"] else "FAIL"
        if m["compact_pass"] is None:
            cmp_icon = "SKIP" if m["skipped_compaction"] else "N/A"
        elif m["compact_pass"]:
            cmp_icon = "PASS"
        else:
            cmp_icon = "FAIL"

        if m["baseline_pass"] and (m["compact_pass"] is True or m["compact_pass"] is None):
            total_pass += 1

        compact_time = f"{m['compact_time_ms']}" if m["compact_time_ms"] else "-"
        compact_ratio = f"{m['compact_ratio']}x" if m["compact_ratio"] else "-"
        compact_recall = f"{m['compact_recall']}/5" if m["compact_recall"] is not None else "-"

        print(f"{m['model']:<30} {m['context_k']:>3}K {m['use_cases']:>3} {m['errors']:>3} "
              f"{m['avg_quality']:>4}/5 {m['recall_score']:>3}/5 "
              f"{m['avg_tok_s']:>5.1f} {compact_time:>6} {compact_ratio:>5} {compact_recall:>6} "
              f"{base_icon:>5} {cmp_icon:>5}")

    print("-" * 100)
    print(f"Total: {total_pass}/{total} passed all gates")

    # Flag quality concerns
    print(f"\n{'Flags:':}")
    for m in models:
        if not m["baseline_pass"]:
            print(f"  [FAIL] {m['model']} — baseline quality {m['avg_quality']}/5 < 3.0 threshold")
        if m["errors"] > 0:
            print(f"  [WARN] {m['model']} — {m['errors']} use case errors")
        if m["compact_error"]:
            print(f"  [FAIL] {m['model']} — compaction error: {m['compact_error']}")
        if m["compact_recall"] is not None and m["compact_recall"] < 2:
            print(f"  [FAIL] {m['model']} — post-compact recall {m['compact_recall']}/5 < 2.0 threshold")


def write_summary_json(models, run_dir):
    """Write summary.json for Supabase import."""
    summary = {
        "run_dir": run_dir.name,
        "total_models": len(models),
        "passed": sum(1 for m in models if m["baseline_pass"] and (m.get("compact_pass") is True or m.get("compact_pass") is None)),
        "failed": sum(1 for m in models if not (m["baseline_pass"] and (m.get("compact_pass") is True or m.get("compact_pass") is None))),
        "models": models,
    }
    out_path = run_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/aggregate-phase-d-results.py <phase-d-run-dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    results_dir = run_dir / "results"

    if not results_dir.is_dir():
        print(f"ERROR: No results/ directory in {run_dir}")
        sys.exit(1)

    models = load_model_results(results_dir)
    if not models:
        print("No model results found.")
        sys.exit(1)

    print_summary(models, run_dir)
    write_summary_json(models, run_dir)


if __name__ == "__main__":
    main()
