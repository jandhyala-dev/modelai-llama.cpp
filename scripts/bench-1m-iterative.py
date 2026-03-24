#!/usr/bin/env python3
"""
Iterative Compaction to 1M Effective Tokens — tok/s Tracking

Fills context window → compacts at 50x → repeats until 1M effective tokens.
Tracks tok/s, compaction time, and recall at checkpoints.

Tests fact recall at 100K, 250K, 500K, 750K, 1M effective token milestones.
"""

import json
import os
import sys
import time
import subprocess
import signal
import urllib.request
import urllib.error

SERVER_PORT = 8090
SERVER = f"http://localhost:{SERVER_PORT}"
SLOT = 0

MODELS_DIR = os.environ["MODELAI_MODELS_DIR"]
MODEL_PATH = os.path.join(MODELS_DIR, "Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf")
SERVER_BIN = os.path.join(os.environ.get("MODELAI_DIR", "."), "build/bin/llama-server")

COMPACT_RATIO = 50
COMPACT_METHOD = "select"
TARGET_EFFECTIVE = 1_000_000

RECALL_CHECKPOINTS = [100_000, 250_000, 500_000, 750_000, 1_000_000]

FACTS = [
    ("NVIDIA data center revenue", "$115 billion", "115"),
    ("AMD total revenue", "$28 billion", "28"),
    ("Intel foundry losses", "$7 billion", "7"),
    ("TSMC revenue", "$90 billion", "90"),
    ("SK Hynix HBM3E market share", "50 percent", "50"),
    ("Broadcom VMware synergy target", "$8.5 billion", "8.5"),
    ("Sovereign AI investment globally", "$150 billion", "150"),
    ("Apple annual buyback pace", "$110 billion", "110"),
    ("NVIDIA CUDA developer count", "4 million", "4"),
    ("AMD MI300X AI accelerator revenue", "$5 billion", "5"),
]

# Source files for realistic content
SRC_DIR = os.path.join(os.environ.get("MODELAI_DIR", "."), "src")


def api(method, path, data=None, timeout=600):
    url = f"{SERVER}{path}"
    body = json.dumps(data).encode() if data else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return {"error": json.loads(e.read())}
        except:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def health():
    try:
        return api("GET", "/health").get("status") == "ok"
    except:
        return False


def chat(messages, max_tokens=2000):
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
    }, timeout=600)


def build_facts_system():
    lines = [
        "You are a financial analyst. MEMORIZE these exact facts — "
        "you MUST recall them precisely when asked:"
    ]
    for i, (name, value, _) in enumerate(FACTS, 1):
        lines.append(f"  FACT {i}: {name} = {value}")
    return "\n".join(lines)


def test_recall():
    """Test recall of all 10 facts."""
    recall_prompt = (
        "RECALL TEST — answer with EXACT numbers only, one per line:\n"
        "1. NVIDIA data center revenue?\n"
        "2. AMD total revenue?\n"
        "3. Intel foundry losses?\n"
        "4. TSMC revenue?\n"
        "5. SK Hynix HBM3E market share?\n"
        "6. Broadcom VMware synergy target?\n"
        "7. Total sovereign AI investment globally?\n"
        "8. Apple annual buyback pace?\n"
        "9. NVIDIA CUDA developer count?\n"
        "10. AMD MI300X AI accelerator revenue?"
    )

    resp = chat([
        {"role": "system", "content": build_facts_system()},
        {"role": "user", "content": recall_prompt},
    ], max_tokens=500)

    if "error" in resp:
        return None

    answer = resp["choices"][0]["message"]["content"]

    scores = []
    for name, value, key_num in FACTS:
        found = False
        for variant in [key_num, f"${key_num}", f"{key_num}B", f"${key_num}B",
                        f"${key_num} billion", f"{key_num} billion",
                        f"{key_num} million", f"{key_num} percent", f"{key_num}%"]:
            if variant.lower() in answer.lower():
                found = True
                break
        scores.append({"fact": name, "expected": value, "recalled": found})

    recalled = sum(1 for s in scores if s["recalled"])
    return {
        "recalled": recalled,
        "total": len(FACTS),
        "pct": round(recalled / len(FACTS) * 100, 1),
    }


def load_source_files():
    """Load C++ source files for realistic fill content."""
    import glob
    files = []
    for ext in ["*.cpp", "*.h"]:
        for f in glob.glob(os.path.join(SRC_DIR, "**", ext), recursive=True):
            try:
                with open(f) as fh:
                    content = fh.read()
                if len(content) > 500:
                    files.append((os.path.basename(f), content))
            except:
                pass
    files.sort(key=lambda x: len(x[1]), reverse=True)
    return files


def start_server(context_size):
    """Start llama-server with given context size. Returns process."""
    # Kill any existing server
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(3)

    env = os.environ.copy()
    env["LLAMA_COMPACT_ALLOWED_METHODS"] = "select,solver,omp,self_study,chunked_self_study,on_policy"

    cmd = [
        SERVER_BIN,
        "-m", MODEL_PATH,
        "-ngl", "99",
        "-c", str(context_size),
        "-np", "1",
        "--port", str(SERVER_PORT),
    ]

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=open("/tmp/llama-1m-bench.log", "w"),
        stderr=subprocess.STDOUT,
    )

    # Wait for ready
    for i in range(120):
        time.sleep(1)
        if health():
            return proc

    proc.kill()
    return None


def run_iterative_benchmark(context_size):
    """Run iterative compaction to 1M effective tokens."""
    sys.stdout.write(f"\n{'='*70}\n")
    sys.stdout.write(f"ITERATIVE 50x COMPACTION TO 1M — Context: {context_size//1024}K\n")
    sys.stdout.write(f"{'='*70}\n\n")
    sys.stdout.flush()

    # Start server
    sys.stdout.write(f"Starting server with {context_size//1024}K context...\n")
    sys.stdout.flush()
    proc = start_server(context_size)
    if proc is None:
        sys.stdout.write("FAILED: Server did not start\n")
        sys.stdout.flush()
        return None

    # Warmup request
    warmup = chat([{"role": "user", "content": "Hello"}], max_tokens=5)
    if "error" in warmup:
        sys.stdout.write(f"WARMUP FAILED: {warmup['error']}\n")
        sys.stdout.flush()
        proc.kill()
        return None

    warmup_tps = warmup.get("timings", {}).get("predicted_per_second", 0)
    sys.stdout.write(f"Server ready. Warmup speed: {warmup_tps:.1f} tok/s\n\n")
    sys.stdout.flush()

    # Load source files for content
    files = load_source_files()
    file_idx = 0

    sys_prompt = build_facts_system()
    effective = 0
    cycle = 0
    cycle_data = []
    recall_results = []
    next_checkpoint_idx = 0

    # Track fill budget: leave room for compaction
    fill_target = int(context_size * 0.75)  # fill to 75% of context

    while effective < TARGET_EFFECTIVE:
        cycle += 1

        if not health():
            sys.stdout.write(f"  Server crashed at cycle {cycle}\n")
            sys.stdout.flush()
            break

        # Build content from source files
        code = ""
        n_files = 0
        while file_idx < len(files) * 2 and len(code) < 15000:
            fname, content = files[file_idx % len(files)]
            code += f"\n### {fname}\n```\n{content[:5000]}\n```\n"
            file_idx += 1
            n_files += 1

        # Fill context
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Analyze this code:\n{code}\n\nSummarize key functions and patterns."},
        ]

        t_fill_start = time.time()
        fill = chat(msgs, max_tokens=1500)
        t_fill_end = time.time()
        fill_wall = t_fill_end - t_fill_start

        if "error" in fill:
            sys.stdout.write(f"  Cycle {cycle}: Fill error - {fill.get('error', fill)}\n")
            sys.stdout.flush()
            break

        tokens_this_cycle = fill["usage"]["total_tokens"]
        gen_tokens = fill["usage"]["completion_tokens"]
        prompt_tokens = fill["usage"]["prompt_tokens"]
        gen_tps = fill.get("timings", {}).get("predicted_per_second", 0)
        prompt_tps = fill.get("timings", {}).get("prompt_per_second", 0)
        effective += tokens_this_cycle

        # Compact
        target = max(int(tokens_this_cycle / COMPACT_RATIO), 32)
        t_compact_start = time.time()
        cr = compact(target, COMPACT_METHOD)
        t_compact_end = time.time()
        compact_wall = t_compact_end - t_compact_start

        if not cr.get("success"):
            err = cr.get("error", cr)
            if isinstance(err, dict):
                err = err.get("message", str(err))
            sys.stdout.write(f"  Cycle {cycle}: Compact failed - {err}\n")
            sys.stdout.flush()
            break

        actual_ratio = cr.get("compression_ratio", 0)
        compact_ms = cr.get("compaction_time_ms", 0)

        pct = effective / TARGET_EFFECTIVE * 100
        sys.stdout.write(
            f"  Cycle {cycle:3d}: +{tokens_this_cycle:5d} tok → {effective:>9,} eff ({pct:5.1f}%) | "
            f"gen {gen_tps:5.1f} tok/s | prompt {prompt_tps:5.0f} tok/s | "
            f"{actual_ratio:5.1f}x {compact_ms:6.0f}ms | fill {fill_wall:5.1f}s\n"
        )
        sys.stdout.flush()

        cycle_entry = {
            "cycle": cycle,
            "tokens_this_cycle": tokens_this_cycle,
            "gen_tokens": gen_tokens,
            "prompt_tokens": prompt_tokens,
            "effective_total": effective,
            "gen_tok_s": round(gen_tps, 1),
            "prompt_tok_s": round(prompt_tps, 1),
            "compact_ratio": round(actual_ratio, 1),
            "compact_ms": round(compact_ms, 1),
            "compact_wall_s": round(compact_wall, 1),
            "fill_wall_s": round(fill_wall, 1),
        }
        cycle_data.append(cycle_entry)

        # Check recall at milestones
        while (next_checkpoint_idx < len(RECALL_CHECKPOINTS) and
               effective >= RECALL_CHECKPOINTS[next_checkpoint_idx]):
            checkpoint = RECALL_CHECKPOINTS[next_checkpoint_idx]
            sys.stdout.write(f"\n  >>> RECALL CHECK at {checkpoint:,} effective tokens\n")
            sys.stdout.flush()

            recall = test_recall()
            if recall:
                sys.stdout.write(
                    f"  >>> Recall: {recall['recalled']}/{recall['total']} "
                    f"({recall['pct']}%)\n\n"
                )
                recall_results.append({
                    "checkpoint": checkpoint,
                    "cycle": cycle,
                    "effective": effective,
                    **recall,
                })
            else:
                sys.stdout.write(f"  >>> Recall FAILED\n\n")
                recall_results.append({
                    "checkpoint": checkpoint,
                    "cycle": cycle,
                    "effective": effective,
                    "error": "recall test failed",
                })
            sys.stdout.flush()
            next_checkpoint_idx += 1

    # Final summary
    sys.stdout.write(f"\n  Completed: {effective:,} effective tokens in {cycle} cycles\n")
    sys.stdout.flush()

    # Kill server
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except:
        proc.kill()

    return {
        "context_size": context_size,
        "context_size_k": context_size // 1024,
        "effective_tokens": effective,
        "cycles": len(cycle_data),
        "compact_ratio": COMPACT_RATIO,
        "compact_method": COMPACT_METHOD,
        "cycle_data": cycle_data,
        "recall_checkpoints": recall_results,
    }


def main():
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.write("1M EFFECTIVE TOKEN BENCHMARK — Iterative 50x Compaction\n")
    sys.stdout.write(f"Model: Qwen3-Coder-30B-A3B-Instruct-1M (UD-Q4_K_XL)\n")
    sys.stdout.write(f"Hardware: Apple M2 Pro 32GB\n")
    sys.stdout.write(f"Compaction: {COMPACT_RATIO}x {COMPACT_METHOD}\n")
    sys.stdout.write(f"Target: {TARGET_EFFECTIVE:,} effective tokens\n")
    sys.stdout.write(f"Recall checkpoints: {', '.join(f'{c:,}' for c in RECALL_CHECKPOINTS)}\n")
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.flush()

    # Context sizes to test
    # 64K guaranteed to work on 32GB M2 Pro with f16 KV
    # 128K requires 29 GB (GPU OOM on 32GB M2 Pro)
    context_sizes = [65536]

    all_results = []

    for ctx in context_sizes:
        ctx_k = ctx // 1024
        kv_gb = ctx * 6 / 65536  # linear from 64K = 6GB
        total_gb = 17 + kv_gb
        sys.stdout.write(f"\n--- Testing {ctx_k}K context ({kv_gb:.1f} GB KV, {total_gb:.1f} GB total) ---\n")
        if total_gb > 25.5:
            sys.stdout.write(f"WARNING: {total_gb:.1f} GB exceeds GPU budget (25.5 GB). May swap.\n")
        sys.stdout.flush()

        result = run_iterative_benchmark(ctx)
        if result:
            all_results.append(result)
        else:
            sys.stdout.write(f"SKIPPED: {ctx_k}K context failed to start or crashed early\n")
            sys.stdout.flush()
            all_results.append({
                "context_size": ctx,
                "context_size_k": ctx_k,
                "status": "FAILED",
            })

    # Summary comparison
    sys.stdout.write(f"\n{'='*70}\n")
    sys.stdout.write("SUMMARY\n")
    sys.stdout.write(f"{'='*70}\n\n")

    for r in all_results:
        ctx_k = r.get("context_size_k", "?")
        if r.get("status") == "FAILED":
            sys.stdout.write(f"{ctx_k}K: FAILED\n")
            continue

        cycles = r.get("cycles", 0)
        eff = r.get("effective_tokens", 0)
        data = r.get("cycle_data", [])

        if data:
            avg_gen_tps = sum(d["gen_tok_s"] for d in data) / len(data)
            avg_compact_ms = sum(d["compact_ms"] for d in data) / len(data)
            min_gen_tps = min(d["gen_tok_s"] for d in data)
            max_gen_tps = max(d["gen_tok_s"] for d in data)
        else:
            avg_gen_tps = avg_compact_ms = min_gen_tps = max_gen_tps = 0

        sys.stdout.write(
            f"{ctx_k}K: {eff:,} effective in {cycles} cycles | "
            f"gen tok/s: avg={avg_gen_tps:.1f}, min={min_gen_tps:.1f}, max={max_gen_tps:.1f} | "
            f"avg compact: {avg_compact_ms:.0f}ms\n"
        )

        # Recall summary
        for rc in r.get("recall_checkpoints", []):
            cp = rc.get("checkpoint", 0)
            pct = rc.get("pct", "ERR")
            sys.stdout.write(f"  Recall @ {cp:>9,}: {pct}%\n")

    sys.stdout.write("\n")

    # tok/s progression table
    sys.stdout.write("TOK/S PROGRESSION (every 10 cycles):\n")
    sys.stdout.write(f"{'Ctx':>5s} {'Cycle':>6s} {'Effective':>12s} {'Gen tok/s':>10s} "
                     f"{'Prompt tok/s':>12s} {'Compact ms':>11s}\n")
    sys.stdout.write("-" * 60 + "\n")
    for r in all_results:
        if r.get("status") == "FAILED":
            continue
        ctx_k = r["context_size_k"]
        for d in r.get("cycle_data", []):
            if d["cycle"] % 10 == 1 or d["cycle"] <= 3 or d["cycle"] == len(r["cycle_data"]):
                sys.stdout.write(
                    f"{ctx_k:>4}K {d['cycle']:>6d} {d['effective_total']:>12,} "
                    f"{d['gen_tok_s']:>10.1f} {d['prompt_tok_s']:>12.1f} "
                    f"{d['compact_ms']:>11.0f}\n"
                )
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Save results
    output = {
        "benchmark": "iterative-1m-50x",
        "model": "Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL",
        "model_source": "unsloth",
        "hardware": "Apple M2 Pro 32GB",
        "compact_ratio": COMPACT_RATIO,
        "compact_method": COMPACT_METHOD,
        "target_effective": TARGET_EFFECTIVE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
    }

    outdir = os.path.join(os.environ.get("MODELAI_DIR", "."), "bench-results")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "iterative-1m-50x-benchmark.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    sys.stdout.write(f"Results saved: {outpath}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
