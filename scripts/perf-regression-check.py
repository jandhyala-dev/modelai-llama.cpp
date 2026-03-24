#!/usr/bin/env python3
"""Performance Regression Checker — modelai-llama.cpp

Runs llama-bench and /compact, compares against a stored baseline, and
reports PASS/FAIL with per-metric regression percentages.

Usage:
    # Local mode — uses MODELAI_MODELS_DIR env var
    python3 scripts/perf-regression-check.py --model-path ~/models/Qwen3-8B-Q4_K_M.gguf

    # CI mode — downloads a small model from HuggingFace
    python3 scripts/perf-regression-check.py --ci

    # Update the baseline with current measurements
    python3 scripts/perf-regression-check.py --model-path ~/models/Qwen3-8B-Q4_K_M.gguf --update-baseline

    # Custom baseline location
    python3 scripts/perf-regression-check.py --model-path ~/models/Qwen3-8B-Q4_K_M.gguf --baseline-path /tmp/bl.json
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..")
BUILD_BIN = os.path.join(REPO_ROOT, "build", "bin")
BENCH_BIN = os.path.join(BUILD_BIN, "llama-bench")
SERVER_BIN = os.path.join(BUILD_BIN, "llama-server")
DEFAULT_BASELINE = os.path.join(REPO_ROOT, "bench-results", "perf-baseline.json")

# CI fallback model (small enough for ephemeral runners)
CI_HF_REPO = "ggml-org/gemma-1.1-2b-it-Q4_K_M-GGUF"
CI_HF_FILE = "gemma-1.1-2b-it.Q4_K_M.gguf"

PORT = 8091  # avoid collision with dev server on 8090
SERVER_URL = f"http://localhost:{PORT}"
SLOT = 0
CTX = 4096  # small context — regression check, not a full benchmark

# ── Output helper ────────────────────────────────────────────────────

def out(msg, end="\n"):
    sys.stdout.write(msg + end)
    sys.stdout.flush()


# ── API helpers ──────────────────────────────────────────────────────

def api(method, path, data=None, timeout=120):
    url = f"{SERVER_URL}{path}"
    body = json.dumps(data).encode() if data else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return {"error": json.loads(e.read())}
        except Exception:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def health():
    try:
        return api("GET", "/health").get("status") == "ok"
    except Exception:
        return False


def chat(messages, max_tokens=256):
    return api("POST", "/v1/chat/completions", {
        "model": "test",
        "id_slot": SLOT,
        "cache_prompt": True,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    })


def compact(target_tokens, method="select"):
    return api("POST", "/compact", {
        "id_slot": SLOT,
        "seq_id": 0,
        "target_tokens": target_tokens,
        "method": method,
    })


# ── Server management ───────────────────────────────────────────────
server_proc = None


def kill_server():
    global server_proc
    if server_proc:
        try:
            server_proc.terminate()
            server_proc.wait(timeout=5)
        except Exception:
            try:
                server_proc.kill()
            except Exception:
                pass
        server_proc = None
    time.sleep(1)


def start_server(model_path):
    global server_proc
    kill_server()

    cmd = [
        SERVER_BIN,
        "-m", model_path,
        "--port", str(PORT),
        "-c", str(CTX),
        "-ngl", "99",
        "--no-mmap",
        "-ctk", "f16",
        "-ctv", "f16",
        "--slots",
        "--metrics",
    ]

    server_proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    for _ in range(90):
        time.sleep(1)
        if health():
            return True
    return False


# ── Measurement: llama-bench ─────────────────────────────────────────

def run_llama_bench(model_path):
    """Run llama-bench and return prompt_eval tok/s and generation tok/s."""
    out("  Running llama-bench (pp=512, tg=128, r=3)...")
    try:
        r = subprocess.run(
            [BENCH_BIN, "-m", model_path, "-ngl", "99",
             "-p", "512", "-n", "128", "-r", "3", "-o", "json"],
            capture_output=True, text=True, timeout=600,
            cwd=REPO_ROOT,
        )
        if r.returncode != 0:
            out(f"    llama-bench FAILED (exit {r.returncode})")
            return None, None

        data = json.loads(r.stdout)
        pp_ts = data[0].get("avg_ts", 0) if len(data) > 0 else 0
        tg_ts = data[1].get("avg_ts", 0) if len(data) > 1 else 0
        out(f"    prompt_eval: {pp_ts:.2f} tok/s")
        out(f"    generation:  {tg_ts:.2f} tok/s")
        return round(pp_ts, 2), round(tg_ts, 2)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        out(f"    llama-bench FAILED: {e}")
        return None, None


# ── Measurement: peak RSS ────────────────────────────────────────────

def measure_peak_rss_mb(model_path):
    """Run a short inference and capture peak RSS via /usr/bin/time or ps."""
    out("  Measuring peak RSS...")
    try:
        # Use a short llama-bench run and measure via GNU time on macOS
        if sys.platform == "darwin":
            # macOS: /usr/bin/time -l reports "maximum resident set size" in bytes
            r = subprocess.run(
                ["/usr/bin/time", "-l",
                 BENCH_BIN, "-m", model_path, "-ngl", "99",
                 "-p", "64", "-n", "16", "-r", "1", "-o", "json"],
                capture_output=True, text=True, timeout=300,
                cwd=REPO_ROOT,
            )
            # macOS time output goes to stderr
            for line in r.stderr.split("\n"):
                if "maximum resident set size" in line:
                    # Format: "  12345678  maximum resident set size"
                    rss_bytes = int(line.strip().split()[0])
                    rss_mb = round(rss_bytes / (1024 * 1024), 1)
                    out(f"    peak RSS: {rss_mb} MB")
                    return rss_mb
        else:
            # Linux: /usr/bin/time -v reports "Maximum resident set size (kbytes)"
            r = subprocess.run(
                ["/usr/bin/time", "-v",
                 BENCH_BIN, "-m", model_path, "-ngl", "99",
                 "-p", "64", "-n", "16", "-r", "1", "-o", "json"],
                capture_output=True, text=True, timeout=300,
                cwd=REPO_ROOT,
            )
            for line in r.stderr.split("\n"):
                if "Maximum resident set size" in line:
                    rss_kb = int(line.strip().split()[-1])
                    rss_mb = round(rss_kb / 1024, 1)
                    out(f"    peak RSS: {rss_mb} MB")
                    return rss_mb

        out("    peak RSS: could not parse")
        return None
    except Exception as e:
        out(f"    peak RSS measurement FAILED: {e}")
        return None


# ── Measurement: compaction at 2x ────────────────────────────────────

def measure_compaction_2x(model_path):
    """Start server, fill KV, compact at 2x, return compaction_time_ms."""
    out("  Measuring /compact at 2x ratio...")

    out("    Starting server...", end="")
    if not start_server(model_path):
        out(" FAILED")
        kill_server()
        return None
    out(" ready")

    try:
        # Warmup
        out("    Warmup...", end="")
        resp = chat([{"role": "user", "content": "Hello"}], max_tokens=5)
        if "error" in resp:
            out(f" ERROR: {resp.get('error')}")
            return None
        out(" ok")

        # Fill KV with a multi-turn conversation
        out("    Filling KV cache...", end="")
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (
                "Explain the complete history of the Internet, from ARPANET to "
                "modern cloud computing. Cover key protocols, organizations, "
                "and technological milestones in detail."
            )},
        ]
        fill_resp = chat(msgs, max_tokens=512)
        if "error" in fill_resp:
            out(f" ERROR: {fill_resp.get('error')}")
            return None

        content = fill_resp["choices"][0]["message"]["content"]
        msgs.append({"role": "assistant", "content": content})
        total_tokens = fill_resp.get("usage", {}).get("total_tokens", 0)
        out(f" {total_tokens} tokens")

        # Add another turn for more KV content
        msgs.append({"role": "user", "content": (
            "Now explain how DNS, BGP, and TLS work together to enable "
            "secure web browsing. Include technical details about each protocol."
        )})
        fill_resp2 = chat(msgs, max_tokens=512)
        if "error" in fill_resp2:
            out(f"    Second fill ERROR: {fill_resp2.get('error')}")
            return None

        content2 = fill_resp2["choices"][0]["message"]["content"]
        msgs.append({"role": "assistant", "content": content2})
        total_tokens = fill_resp2.get("usage", {}).get("total_tokens", 0)
        out(f"    Total KV: {total_tokens} tokens")

        # Re-prime KV before compaction (discard recall question tokens)
        out("    Re-priming KV...", end="")
        prime_resp = chat(msgs, max_tokens=1)
        if "error" in prime_resp:
            out(f" ERROR: {prime_resp.get('error')}")
            return None
        kv_tokens = prime_resp.get("usage", {}).get("total_tokens", total_tokens)
        out(f" {kv_tokens} tokens")

        # Compact at 2x
        target = max(int(kv_tokens / 2), 64)
        out(f"    Compacting {kv_tokens} -> {target} (2x)...", end="")
        cr = compact(target)
        if not cr.get("success"):
            out(f" FAILED: {cr}")
            return None

        compact_ms = cr.get("compaction_time_ms", 0)
        actual_ratio = cr.get("compression_ratio", 0)
        out(f" {actual_ratio:.1f}x in {compact_ms:.0f} ms")
        return round(compact_ms, 1)

    finally:
        kill_server()


# ── Comparison logic ─────────────────────────────────────────────────

def load_baseline(path):
    with open(path, "r") as f:
        return json.load(f)


def save_baseline(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    out(f"\nBaseline saved: {path}")


def compare(measured, baseline):
    """Compare measured metrics against baseline. Returns list of (metric, pct_change, threshold, passed)."""
    bl_metrics = baseline.get("metrics", {})
    thresholds = baseline.get("thresholds", {})
    default_max = thresholds.get("regression_max_pct", 10)
    memory_max = thresholds.get("memory_regression_max_pct", 15)

    checks = []
    metric_configs = [
        # (key, display_name, direction, threshold)
        # direction: "higher_better" means regression = value went DOWN
        # direction: "lower_better"  means regression = value went UP
        ("prompt_eval_tok_s", "Prompt eval (tok/s)", "higher_better", default_max),
        ("generation_tok_s",  "Generation (tok/s)",  "higher_better", default_max),
        ("compact_2x_ms",     "Compact 2x (ms)",     "lower_better",  default_max),
        ("peak_rss_mb",       "Peak RSS (MB)",        "lower_better",  memory_max),
    ]

    for key, display, direction, threshold in metric_configs:
        bl_val = bl_metrics.get(key)
        m_val = measured.get(key)

        if bl_val is None or m_val is None:
            checks.append((display, None, threshold, True, bl_val, m_val))
            continue

        if bl_val == 0:
            checks.append((display, 0.0, threshold, True, bl_val, m_val))
            continue

        if direction == "higher_better":
            # Regression: measured < baseline  =>  pct_change is negative
            pct_change = ((m_val - bl_val) / bl_val) * 100
            passed = pct_change >= -threshold
        else:
            # Regression: measured > baseline  =>  pct_change is positive
            pct_change = ((m_val - bl_val) / bl_val) * 100
            passed = pct_change <= threshold

        checks.append((display, round(pct_change, 1), threshold, passed, bl_val, m_val))

    return checks


# ── Table output ─────────────────────────────────────────────────────

def print_results_table(checks):
    """Print a bordered results table."""
    col_metric = 24
    col_base = 14
    col_curr = 14
    col_delta = 12
    col_thresh = 12
    col_verdict = 8
    total = col_metric + col_base + col_curr + col_delta + col_thresh + col_verdict + 7  # 7 = separators

    hdr = (f"{'Metric':<{col_metric}}"
           f" {'Baseline':>{col_base}}"
           f" {'Current':>{col_curr}}"
           f" {'Delta':>{col_delta}}"
           f" {'Threshold':>{col_thresh}}"
           f" {'Result':>{col_verdict}}")

    out("")
    out("+" + "-" * (total) + "+")
    out("| " + hdr + " |")
    out("+" + "-" * (total) + "+")

    for entry in checks:
        if len(entry) == 6:
            display, pct, threshold, passed, bl_val, m_val = entry
        else:
            display, pct, threshold, passed = entry
            bl_val, m_val = None, None

        bl_str = f"{bl_val}" if bl_val is not None else "---"
        m_str = f"{m_val}" if m_val is not None else "---"

        if pct is None:
            delta_str = "N/A"
            verdict = "SKIP"
        elif passed:
            delta_str = f"{pct:+.1f}%"
            verdict = "PASS"
        else:
            delta_str = f"{pct:+.1f}%"
            verdict = "FAIL"

        thresh_str = f"+/-{threshold}%"

        line = (f"{display:<{col_metric}}"
                f" {bl_str:>{col_base}}"
                f" {m_str:>{col_curr}}"
                f" {delta_str:>{col_delta}}"
                f" {thresh_str:>{col_thresh}}"
                f" {verdict:>{col_verdict}}")
        out("| " + line + " |")

    out("+" + "-" * (total) + "+")


# ── Main ─────────────────────────────────────────────────────────────

def resolve_model_path(args):
    """Resolve model path from args or environment."""
    if args.ci:
        out("CI mode: using HuggingFace model reference")
        return None  # signal to use --hf-repo for llama-bench

    if args.model_path:
        path = os.path.expanduser(args.model_path)
        if not os.path.isfile(path):
            out(f"ERROR: model not found: {path}")
            sys.exit(1)
        return path

    model_dir = os.environ.get("MODELAI_MODELS_DIR")
    if not model_dir:
        out("ERROR: --model-path not given and MODELAI_MODELS_DIR not set")
        sys.exit(1)

    # Default to Qwen3-8B in model dir
    candidates = [
        "Qwen3-8B-Q4_K_M.gguf",
        "Qwen3-14B-Q4_K_M.gguf",
    ]
    for name in candidates:
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            out(f"Using model: {path}")
            return path

    out(f"ERROR: no suitable model found in {model_dir}")
    sys.exit(1)


def run_llama_bench_ci():
    """Run llama-bench in CI mode using a HuggingFace model reference."""
    out("  Running llama-bench via --hf-repo (CI mode)...")
    try:
        r = subprocess.run(
            [BENCH_BIN, "--hf-repo", CI_HF_REPO, "--hf-file", CI_HF_FILE,
             "-ngl", "99", "-p", "512", "-n", "128", "-r", "3", "-o", "json"],
            capture_output=True, text=True, timeout=600,
            cwd=REPO_ROOT,
        )
        if r.returncode != 0:
            out(f"    llama-bench FAILED (exit {r.returncode})")
            return None, None

        data = json.loads(r.stdout)
        pp_ts = data[0].get("avg_ts", 0) if len(data) > 0 else 0
        tg_ts = data[1].get("avg_ts", 0) if len(data) > 1 else 0
        out(f"    prompt_eval: {pp_ts:.2f} tok/s")
        out(f"    generation:  {tg_ts:.2f} tok/s")
        return round(pp_ts, 2), round(tg_ts, 2)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        out(f"    llama-bench FAILED: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Performance regression checker for modelai-llama.cpp",
    )
    parser.add_argument(
        "--model-path",
        help="Path to GGUF model file (local mode)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: use small HuggingFace model (Gemma 2B Q4_K_M)",
    )
    parser.add_argument(
        "--baseline-path",
        default=DEFAULT_BASELINE,
        help=f"Path to baseline JSON (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write current measurements as the new baseline (no comparison)",
    )
    args = parser.parse_args()

    baseline_path = os.path.expanduser(args.baseline_path)

    # Validate binaries exist
    if not os.path.isfile(BENCH_BIN):
        out(f"ERROR: llama-bench not found at {BENCH_BIN}")
        out("Run: cmake --build build --config Release -j")
        sys.exit(1)

    if not os.path.isfile(SERVER_BIN):
        out(f"ERROR: llama-server not found at {SERVER_BIN}")
        out("Run: cmake --build build --config Release -j")
        sys.exit(1)

    model_path = resolve_model_path(args)
    is_ci = args.ci

    git_commit = "unknown"
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=REPO_ROOT,
        )
        if r.returncode == 0:
            git_commit = r.stdout.strip()
    except Exception:
        pass

    model_name = os.path.basename(model_path) if model_path else f"HF:{CI_HF_REPO}"

    out("=" * 72)
    out("PERFORMANCE REGRESSION CHECK — modelai-llama.cpp")
    out(f"  commit:   {git_commit}")
    out(f"  model:    {model_name}")
    out(f"  baseline: {baseline_path}")
    out(f"  date:     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out("=" * 72)

    # ── Collect metrics ──────────────────────────────────────────────
    measured = {}

    # 1. llama-bench: prompt eval and generation speed
    out("\n[1/3] llama-bench")
    if is_ci:
        pp_ts, tg_ts = run_llama_bench_ci()
    else:
        pp_ts, tg_ts = run_llama_bench(model_path)
    measured["prompt_eval_tok_s"] = pp_ts
    measured["generation_tok_s"] = tg_ts

    # 2. Peak RSS
    out("\n[2/3] Peak RSS")
    if is_ci:
        out("  Skipping RSS measurement in CI mode (no local model file)")
        measured["peak_rss_mb"] = None
    else:
        measured["peak_rss_mb"] = measure_peak_rss_mb(model_path)

    # 3. Compaction at 2x
    out("\n[3/3] KV Compaction (2x)")
    if is_ci:
        out("  Skipping compaction in CI mode (requires local model + server)")
        measured["compact_2x_ms"] = None
    else:
        measured["compact_2x_ms"] = measure_compaction_2x(model_path)

    # ── Update baseline mode ─────────────────────────────────────────
    if args.update_baseline:
        new_baseline = {
            "model": model_name,
            "commit": git_commit,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {
                "prompt_eval_tok_s": measured["prompt_eval_tok_s"],
                "generation_tok_s": measured["generation_tok_s"],
                "compact_2x_ms": measured["compact_2x_ms"],
                "peak_rss_mb": measured["peak_rss_mb"],
            },
            "thresholds": {
                "regression_max_pct": 10,
                "memory_regression_max_pct": 15,
            },
        }
        save_baseline(baseline_path, new_baseline)
        out("\nBaseline updated. Run without --update-baseline to check regressions.")
        sys.exit(0)

    # ── Compare against baseline ─────────────────────────────────────
    if not os.path.isfile(baseline_path):
        out(f"\nWARNING: No baseline file found at {baseline_path}")
        out("Run with --update-baseline to create one.")
        out("\nMeasured values (no comparison):")
        for k, v in measured.items():
            out(f"  {k}: {v}")
        sys.exit(0)

    baseline = load_baseline(baseline_path)
    bl_metrics = baseline.get("metrics", {})

    # Check if baseline has any non-null values
    has_values = any(v is not None for v in bl_metrics.values())
    if not has_values:
        out(f"\nBaseline exists but all metrics are null (template).")
        out("Run with --update-baseline to populate it first.")
        out("\nMeasured values (no comparison):")
        for k, v in measured.items():
            out(f"  {k}: {v}")
        sys.exit(0)

    out(f"\nBaseline from: {baseline.get('date', '?')} (commit {baseline.get('commit', '?')})")
    checks = compare(measured, baseline)

    print_results_table(checks)

    # ── Verdict ──────────────────────────────────────────────────────
    any_fail = any(not passed for _, _, _, passed, *_ in checks if _ is not None)

    out("")
    if any_fail:
        out("VERDICT: FAIL — performance regression detected")
        for display, pct, threshold, passed, *rest in checks:
            if pct is not None and not passed:
                out(f"  REGRESSION: {display}: {pct:+.1f}% (threshold: +/-{threshold}%)")
        sys.exit(1)
    else:
        out("VERDICT: PASS — no regressions detected")
        sys.exit(0)


if __name__ == "__main__":
    # Ensure server cleanup on Ctrl+C
    def _cleanup(sig, frame):
        kill_server()
        sys.exit(130)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    main()
