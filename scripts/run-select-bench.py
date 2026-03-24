#!/usr/bin/env python3
"""Lightweight Benchmark: modelai-llama.cpp vs upstream llama.cpp (select pipeline only)

Skips Ollama and the broken nonuniform pipeline.
Skips gemma3-12b (broken SWA decode).
Output format matches run-3way-bench.py for Supabase import compatibility.

Usage:
    python3 scripts/run-select-bench.py
    python3 scripts/run-select-bench.py --models qwen3-14b deepseek-r1-14b
    python3 scripts/run-select-bench.py --skip-inference   # compaction only
"""

import json
import subprocess
import sys
import os
import csv
from datetime import datetime, timezone
from pathlib import Path

# --- Configuration ---
MODELAI_DIR = Path(os.environ.get("MODELAI_DIR", "."))
UPSTREAM_DIR = Path(os.environ["UPSTREAM_DIR"])

MODELS_DIR = os.environ["MODELAI_MODELS_DIR"]

MODELS = {
    "qwen3-14b": {
        "gguf": f"{MODELS_DIR}/Qwen3-14B-Q4_K_M.gguf",
        "params_b": 14.0,
        "arch": "qwen3",
        "quant": "Q4_K_M",
    },
    "qwen3-8b": {
        "gguf": f"{MODELS_DIR}/Qwen3-8B-Q4_K_M.gguf",
        "params_b": 8.0,
        "arch": "qwen3",
        "quant": "Q4_K_M",
    },
    "qwen3-30b-a3b": {
        "gguf": f"{MODELS_DIR}/Qwen3-30B-A3B-Instruct-Q4_K_M.gguf",
        "params_b": 30.0,
        "arch": "qwen3-moe",
        "quant": "Q4_K_M",
    },
    "deepseek-r1-14b": {
        "gguf": f"{MODELS_DIR}/deepseek-r1-distill-qwen-14b-q4_k_m.gguf",
        "params_b": 14.0,
        "arch": "deepseek-r1",
        "quant": "Q4_K_M",
    },
    # gemma3-12b excluded: SWA decode broken (0.7 tok/s vs 16.5 on Ollama)
}

PROMPT_SIZES = [512, 2048, 4096]
GEN_TOKENS = 128
COMPRESSION_RATIOS = [2, 4, 8]
KV_COMPACT_CONTEXTS = [4096, 8192, 16384]


def get_commit_sha():
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, cwd=MODELAI_DIR)
    return r.stdout.strip()


def get_branch():
    r = subprocess.run(["git", "branch", "--show-current"],
                       capture_output=True, text=True, cwd=MODELAI_DIR)
    return r.stdout.strip()


def get_machine_info():
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                         capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True).stdout.strip())
    return {
        "hostname": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
        "os": f"{os.uname().sysname} {os.uname().release}",
        "arch": os.uname().machine,
        "cpu": cpu,
        "memory_gb": mem // (1024**3),
    }


def run_llama_bench(bench_bin, model_path, prompt_size, gen_tokens, engine_name, model_name):
    """Run llama-bench and return parsed results."""
    print(f"  [{engine_name}] pp={prompt_size} n={gen_tokens} ... ", end="", flush=True)
    try:
        r = subprocess.run(
            [str(bench_bin), "-m", str(model_path), "-ngl", "99",
             "-p", str(prompt_size), "-n", str(gen_tokens), "-r", "1", "-o", "json"],
            capture_output=True, text=True, timeout=600, cwd=str(MODELAI_DIR)
        )
        if r.returncode != 0:
            print(f"FAILED (exit {r.returncode})")
            return None

        data = json.loads(r.stdout)
        pp_ts = data[0].get("avg_ts", 0) if len(data) > 0 else 0
        tg_ts = data[1].get("avg_ts", 0) if len(data) > 1 else 0
        print(f"pp={pp_ts:.1f} tg={tg_ts:.1f} tok/s")
        return {"prefill_tok_s": round(pp_ts, 2), "decode_tok_s": round(tg_ts, 2)}
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        print(f"FAILED ({e})")
        return None


def run_kv_compaction(model_path, model_name, n_ctx, ratio):
    """Run modelai KV compaction benchmark (select pipeline only)."""
    print(f"  [modelai-compact] ctx={n_ctx} {ratio}x select ... ", end="", flush=True)
    longctx_bin = MODELAI_DIR / "build" / "bin" / "test-kv-compact-longctx"

    env = os.environ.copy()
    env["PIPELINE"] = "select"
    env["RATIO"] = str(ratio)

    try:
        r = subprocess.run(
            [str(longctx_bin), "-m", str(model_path), "-c", str(n_ctx), "-ngl", "99"],
            capture_output=True, text=True, timeout=600, env=env, cwd=str(MODELAI_DIR)
        )
        output = r.stdout + r.stderr

        result = {"pipeline": "select", "ratio": ratio, "n_ctx": n_ctx}

        for line in output.split("\n"):
            if "cosine=" in line:
                try:
                    cosine = float(line.split("cosine=")[1].split()[0].rstrip(")"))
                    result["logit_cosine"] = round(cosine, 6)
                except (ValueError, IndexError):
                    result["logit_cosine"] = None
            if "compact=" in line:
                try:
                    result["compaction_time_ms"] = float(line.split("compact=")[1].split("ms")[0])
                except (ValueError, IndexError):
                    pass
            if "baseline=" in line and "tok/s" in line:
                try:
                    result["baseline_decode_tok_s"] = float(line.split("baseline=")[1].split()[0])
                except (ValueError, IndexError):
                    pass
            if "compacted=" in line and "tok/s" in line:
                try:
                    result["compacted_decode_tok_s"] = float(line.split("compacted=")[1].split()[0])
                except (ValueError, IndexError):
                    pass
            if "active_n_kv=" in line:
                try:
                    result["active_n_kv"] = int(line.split("active_n_kv=")[1].split()[0])
                except (ValueError, IndexError):
                    pass
            if "PASS" in line:
                result["pass"] = True
            elif "FAIL" in line and "pass" not in result:
                result["pass"] = False

        cosine_str = f"{result.get('logit_cosine', 'nan')}"
        pass_str = "PASS" if result.get("pass") else "FAIL"
        print(f"cos={cosine_str} [{pass_str}]")
        return result

    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return {"pipeline": "select", "ratio": ratio, "n_ctx": n_ctx, "pass": False, "error": "timeout"}
    except Exception as e:
        print(f"FAILED ({e})")
        return None


def generate_summary(results, timestamp, commit_sha):
    """Generate aggregate summary."""
    engine_stats = {}
    model_stats = {}

    for r in results:
        eng = r.get("engine", "unknown")
        model = r.get("model", "unknown")

        if eng not in engine_stats:
            engine_stats[eng] = {"total": 0, "passed": 0, "failed": 0}
        engine_stats[eng]["total"] += 1
        if r.get("pass", True):
            engine_stats[eng]["passed"] += 1
        else:
            engine_stats[eng]["failed"] += 1

        if model not in model_stats:
            model_stats[model] = {"total": 0, "passed": 0, "failed": 0, "cosines": []}
        model_stats[model]["total"] += 1
        if r.get("pass", True):
            model_stats[model]["passed"] += 1
        else:
            model_stats[model]["failed"] += 1
        if r.get("logit_cosine") is not None:
            model_stats[model]["cosines"].append(r["logit_cosine"])

    # Compute averages
    for m in model_stats.values():
        cosines = m.pop("cosines")
        m["avg_cosine"] = round(sum(cosines) / len(cosines), 6) if cosines else None
        m["min_cosine"] = round(min(cosines), 6) if cosines else None

    return {
        "benchmark_type": "select-only",
        "timestamp": timestamp,
        "commit": commit_sha,
        "engines": ["modelai-llama.cpp", "llama.cpp"],
        "pipeline": "select",
        "engine_stats": engine_stats,
        "model_stats": model_stats,
        "key_findings": {
            "supported_models": [m for m, s in model_stats.items() if s["failed"] == 0],
            "failed_models": [m for m, s in model_stats.items() if s["failed"] > 0],
        },
        "verdict": "PASS" if all(s["failed"] == 0 for s in model_stats.values()) else "MIXED",
    }


def main():
    # Parse args
    args = sys.argv[1:]
    skip_inference = "--skip-inference" in args
    selected_models = None
    if "--models" in args:
        idx = args.index("--models")
        selected_models = []
        for a in args[idx + 1:]:
            if a.startswith("--"):
                break
            selected_models.append(a)

    models = {k: v for k, v in MODELS.items() if selected_models is None or k in selected_models}
    if not models:
        print(f"ERROR: No matching models. Available: {list(MODELS.keys())}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    commit_sha = get_commit_sha()
    branch = get_branch()
    machine = get_machine_info()

    out_dir = MODELAI_DIR / "bench-results" / f"3way-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Select-Only Benchmark (modelai vs upstream) ===")
    print(f"Output: {out_dir}")
    print(f"Commit: {commit_sha} ({branch})")
    print(f"Machine: {machine['cpu']} ({machine['memory_gb']}GB)")
    print(f"Models: {list(models.keys())}")
    print(f"Skip inference: {skip_inference}")
    print()

    # Manifest (compatible with import script)
    manifest = {
        "schema_version": 3,
        "benchmark_type": "3way-comparison",
        "run_id": f"{timestamp}-{commit_sha}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit_sha": commit_sha,
        "branch": branch,
        "machine": machine,
        "engines": ["modelai-llama.cpp", "llama.cpp"],
        "models": list(models.keys()),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    all_results = []

    for model_name, model_info in sorted(models.items()):
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name} ({model_info['params_b']}B {model_info['quant']})")
        print(f"{'='*60}")

        gguf_path = Path(model_info["gguf"])
        if not gguf_path.exists():
            print(f"  SKIP: {gguf_path} not found")
            continue

        # --- 1. Baseline inference benchmarks (modelai vs upstream) ---
        if not skip_inference:
            for pp in PROMPT_SIZES:
                print(f"\n  --- Prompt: {pp} tokens ---")

                # Upstream llama.cpp
                upstream_bench = UPSTREAM_DIR / "build" / "bin" / "llama-bench"
                if upstream_bench.exists():
                    upstream_result = run_llama_bench(
                        upstream_bench, gguf_path, pp, GEN_TOKENS, "llama.cpp", model_name)
                    if upstream_result:
                        all_results.append({
                            "engine": "llama.cpp",
                            "model": model_name,
                            "model_params_b": model_info["params_b"],
                            "architecture": model_info["arch"],
                            "quantization": model_info["quant"],
                            "test_type": "inference",
                            "prompt_tokens": pp,
                            "decode_tokens": GEN_TOKENS,
                            "prefill_tok_s": upstream_result["prefill_tok_s"],
                            "decode_tok_s": upstream_result["decode_tok_s"],
                            "compression_ratio": 1,
                            "pass": True,
                        })
                else:
                    print(f"  [llama.cpp] SKIP: {upstream_bench} not found")

                # modelai-llama.cpp (baseline, no compaction)
                modelai_bench = MODELAI_DIR / "build" / "bin" / "llama-bench"
                modelai_result = run_llama_bench(
                    modelai_bench, gguf_path, pp, GEN_TOKENS, "modelai", model_name)
                if modelai_result:
                    all_results.append({
                        "engine": "modelai-llama.cpp",
                        "model": model_name,
                        "model_params_b": model_info["params_b"],
                        "architecture": model_info["arch"],
                        "quantization": model_info["quant"],
                        "test_type": "inference",
                        "prompt_tokens": pp,
                        "decode_tokens": GEN_TOKENS,
                        "prefill_tok_s": modelai_result["prefill_tok_s"],
                        "decode_tok_s": modelai_result["decode_tok_s"],
                        "compression_ratio": 1,
                        "pass": True,
                    })

        # --- 2. KV Compaction benchmarks (select pipeline only) ---
        print(f"\n  --- KV Compaction (select pipeline only) ---")
        for n_ctx in KV_COMPACT_CONTEXTS:
            for ratio in COMPRESSION_RATIOS:
                compact_result = run_kv_compaction(
                    model_info["gguf"], model_name, n_ctx, ratio)
                if compact_result:
                    all_results.append({
                        "engine": "modelai-compact",
                        "model": model_name,
                        "model_params_b": model_info["params_b"],
                        "architecture": model_info["arch"],
                        "quantization": model_info["quant"],
                        "test_type": "compaction",
                        "pipeline": "select",
                        "n_ctx": n_ctx,
                        "compression_ratio": ratio,
                        "logit_cosine": compact_result.get("logit_cosine"),
                        "compaction_time_ms": compact_result.get("compaction_time_ms"),
                        "baseline_decode_tok_s": compact_result.get("baseline_decode_tok_s"),
                        "compacted_decode_tok_s": compact_result.get("compacted_decode_tok_s"),
                        "active_n_kv": compact_result.get("active_n_kv"),
                        "pass": compact_result.get("pass", False),
                    })

    # --- Write results ---
    results_json = {
        "schema_version": 3,
        "run_id": f"{timestamp}-{commit_sha}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": all_results,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # CSV
    if all_results:
        csv_keys = sorted(set(k for r in all_results for k in r.keys()))
        with open(out_dir / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)

    # Summary
    summary = generate_summary(all_results, timestamp, commit_sha)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Results:  {out_dir}/results.json ({len(all_results)} tests)")
    print(f"CSV:      {out_dir}/results.csv")
    print(f"Summary:  {out_dir}/summary.json")
    print(f"Manifest: {out_dir}/manifest.json")

    # Quick quality summary
    compaction_results = [r for r in all_results if r.get("test_type") == "compaction"]
    if compaction_results:
        passed = sum(1 for r in compaction_results if r.get("pass"))
        cosines = [r["logit_cosine"] for r in compaction_results if r.get("logit_cosine") is not None]
        avg_cos = sum(cosines) / len(cosines) if cosines else 0
        min_cos = min(cosines) if cosines else 0
        print(f"\nCompaction: {passed}/{len(compaction_results)} passed | avg cosine={avg_cos:.4f} | min cosine={min_cos:.4f}")

    return out_dir


if __name__ == "__main__":
    main()
