#!/usr/bin/env python3
"""
256K Effective Context via 4x Compaction — Coder-1M vs Instruct-2507

Tests iterative 4x compaction to achieve 256K effective tokens from a 64K window.
Tracks: tok/s, compaction time, recall at checkpoints, degradation over cycles.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

SERVER_BIN = os.path.join(os.path.dirname(__file__), "..", "build", "bin", "llama-server")
MODEL_DIR = os.environ["MODELAI_MODELS_DIR"]
PORT = 8090
SERVER = f"http://localhost:{PORT}"
CTX = 65536
SLOT = 0

MODELS = [
    {"name": "Instruct-2507", "file": "Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf"},
    {"name": "Coder-1M",      "file": "Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf"},
]

TARGET_EFFECTIVE = 256_000
COMPACT_RATIO = 4
RECALL_CHECKPOINTS = [50_000, 100_000, 150_000, 200_000, 256_000]

# 10 facts injected at the start — must survive through all compaction cycles
FACTS = [
    ("Project Chimera budget", "$47.3 million"),
    ("Vault encryption key length", "4096 bits"),
    ("Server cluster node count", "2,847 nodes"),
    ("API rate limit per minute", "12,500 requests"),
    ("Database replication factor", "5 replicas"),
    ("CDN edge locations worldwide", "389 locations"),
    ("Annual cloud infrastructure cost", "$18.6 million"),
    ("Peak concurrent users record", "3.2 million"),
    ("Mean time to recovery target", "4.5 minutes"),
    ("Data retention policy duration", "7 years"),
]

# Source files for realistic fill content
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")


def api(method, path, data=None, timeout=300):
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
        except Exception:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def health():
    try:
        return api("GET", "/health").get("status") == "ok"
    except Exception:
        return False


def chat(messages, max_tokens=1500):
    t0 = time.time()
    resp = api("POST", "/v1/chat/completions", {
        "model": "test",
        "id_slot": SLOT,
        "cache_prompt": True,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    wall = time.time() - t0
    resp["_wall"] = wall
    return resp


def compact(target_tokens):
    return api("POST", "/compact", {
        "id_slot": SLOT,
        "seq_id": 0,
        "target_tokens": target_tokens,
        "method": "select",
    })


def out(msg, end="\n"):
    sys.stdout.write(msg + end)
    sys.stdout.flush()


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
            except Exception:
                pass
    files.sort(key=lambda x: len(x[1]), reverse=True)
    return files


def build_facts_system():
    lines = ["You are a senior infrastructure engineer. MEMORIZE these exact facts about our systems:"]
    for i, (name, value) in enumerate(FACTS, 1):
        lines.append(f"  FACT {i}: {name} = {value}")
    lines.append("")
    lines.append("You MUST recall these exact values when asked. These are internal metrics — do not guess.")
    return "\n".join(lines)


def test_recall(system_prompt):
    """Test recall of all 10 facts. Returns (recalled, total, details)."""
    recall_q = "Answer each question with ONLY the specific number/value. One per line.\n\n"
    for i, (name, _) in enumerate(FACTS, 1):
        recall_q += f"{i}. What is {name}?\n"

    resp = chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": recall_q},
    ], max_tokens=500)

    if "error" in resp:
        return 0, 10, f"ERROR: {resp.get('error')}"

    answer = resp["choices"][0]["message"]["content"]

    recalled = 0
    details = []
    for name, value in FACTS:
        num = value.replace("$", "").replace(",", "").replace(" million", "").replace(" billion", "")
        num = num.replace(" bits", "").replace(" nodes", "").replace(" requests", "")
        num = num.replace(" replicas", "").replace(" locations", "").replace(" minutes", "")
        num = num.replace(" years", "").replace(" million", "")

        found = any(
            v.lower() in answer.lower().replace(",", "")
            for v in [num, value, value.replace("$", ""), value.replace(",", "")]
        )
        if found:
            recalled += 1
        details.append({"fact": name, "expected": value, "recalled": found})

    return recalled, 10, details


server_proc = None

def kill_server():
    global server_proc
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
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
    time.sleep(3)

def start_server(model_file):
    global server_proc
    kill_server()
    model_path = os.path.join(MODEL_DIR, model_file)
    server_proc = subprocess.Popen([
        SERVER_BIN, "-m", model_path, "--port", str(PORT),
        "-c", str(CTX), "-ngl", "99", "--no-mmap",
        "-ctk", "f16", "-ctv", "f16", "--slots", "--metrics",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        time.sleep(1)
        if health():
            return True
    return False


def run_iterative_compaction(model_name):
    """Run iterative 4x compaction cycles to reach 256K effective tokens."""
    out(f"\n    Iterative 4x Compaction → {TARGET_EFFECTIVE//1000}K effective tokens")

    system_prompt = build_facts_system()
    src_files = load_source_files()
    file_idx = 0
    effective = 0
    cycle = 0
    cycles = []
    recall_results = {}
    next_checkpoint_idx = 0

    while effective < TARGET_EFFECTIVE:
        cycle += 1

        if not health():
            out(f"    Cycle {cycle}: SERVER CRASHED")
            break

        # Build fill content from source files (10K chars per cycle)
        code_block = ""
        n_files = 0
        while file_idx < len(src_files) * 3 and len(code_block) < 10000:
            fname, content = src_files[file_idx % len(src_files)]
            code_block += f"\n### {fname}\n```cpp\n{content[:3000]}\n```\n"
            file_idx += 1
            n_files += 1

        t0 = time.time()
        fill = chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this code and summarize key functions:\n{code_block}"},
        ], max_tokens=1500)
        fill_time = time.time() - t0

        if "error" in fill:
            out(f"    Cycle {cycle}: FILL ERROR — {fill.get('error')}")
            break

        usage = fill.get("usage", {})
        tokens_this_cycle = usage.get("total_tokens", 0)
        comp_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        effective += tokens_this_cycle

        gen_tok_s = comp_tokens / fill_time if fill_time > 0 else 0
        prompt_tok_s = prompt_tokens / fill_time if fill_time > 0 else 0

        # Compact
        target = max(int(tokens_this_cycle / COMPACT_RATIO), 64)
        cr = compact(target)
        if not cr.get("success"):
            out(f"    Cycle {cycle}: COMPACT FAILED — {cr}")
            break

        actual_ratio = cr.get("compression_ratio", 0)
        compact_ms = cr.get("compaction_time_ms", 0)

        pct = effective / TARGET_EFFECTIVE * 100
        out(f"    Cycle {cycle:3d}: +{tokens_this_cycle:5d} tok → {effective:>9,} eff ({pct:5.1f}%) | "
            f"gen {gen_tok_s:5.1f} t/s | prompt {prompt_tok_s:5.0f} t/s | "
            f"{actual_ratio:4.1f}x {compact_ms:5.0f}ms | fill {fill_time:5.1f}s")

        cycles.append({
            "cycle": cycle,
            "tokens": tokens_this_cycle,
            "effective": effective,
            "gen_tok_s": round(gen_tok_s, 1),
            "prompt_tok_s": round(prompt_tok_s, 1),
            "ratio": round(actual_ratio, 1),
            "compact_ms": round(compact_ms, 1),
            "fill_s": round(fill_time, 1),
        })

        # Recall check at checkpoints
        if (next_checkpoint_idx < len(RECALL_CHECKPOINTS) and
                effective >= RECALL_CHECKPOINTS[next_checkpoint_idx]):
            checkpoint = RECALL_CHECKPOINTS[next_checkpoint_idx]
            recalled, total, details = test_recall(system_prompt)
            out(f"    >>> RECALL CHECK at {checkpoint:,} effective: {recalled}/{total}")
            if recalled < total:
                lost = [d["fact"] for d in details if isinstance(details, list) and not d.get("recalled")]
                if isinstance(details, list):
                    lost = [d["fact"] for d in details if not d["recalled"]]
                    if lost:
                        out(f"        LOST: {', '.join(lost[:3])}")
            recall_results[str(checkpoint)] = {
                "recalled": recalled,
                "total": total,
                "pct": round(recalled / total * 100, 1),
            }
            next_checkpoint_idx += 1

    return {
        "effective_tokens": effective,
        "cycles": len(cycles),
        "target": TARGET_EFFECTIVE,
        "compact_ratio": COMPACT_RATIO,
        "reached_target": effective >= TARGET_EFFECTIVE,
        "cycle_data": cycles,
        "recall_checkpoints": recall_results,
        "avg_gen_tok_s": round(sum(c["gen_tok_s"] for c in cycles) / max(len(cycles), 1), 1),
        "avg_compact_ms": round(sum(c["compact_ms"] for c in cycles) / max(len(cycles), 1), 1),
        "total_wall_s": round(sum(c["fill_s"] for c in cycles), 1),
    }


def main():
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    out("=" * 76)
    out("256K EFFECTIVE CONTEXT — 4x Iterative Compaction")
    out(f"Window: 64K | Target: 256K effective | Ratio: 4x | Method: select")
    out(f"Hardware: Apple M2 Pro 32GB | KV: f16 | enable_thinking: false")
    out(f"Recall checkpoints: {', '.join(f'{c//1000}K' for c in RECALL_CHECKPOINTS)}")
    out(f"Timestamp: {timestamp}")
    out("=" * 76)

    all_results = {}

    for mi, model in enumerate(MODELS):
        out(f"\n{'='*76}")
        out(f"[{mi+1}/2] {model['name']}")
        out(f"{'='*76}")

        out(f"  Starting server...", end="")
        if not start_server(model["file"]):
            out(" FAILED")
            all_results[model["name"]] = {"error": "Server failed"}
            continue
        out(" ready")

        # Warmup
        chat([{"role": "user", "content": "Hello"}], max_tokens=5)

        result = run_iterative_compaction(model["name"])
        all_results[model["name"]] = result

        kill_server()
        out(f"\n  Done: {result['effective_tokens']:,} effective in {result['cycles']} cycles "
            f"({result['total_wall_s']:.0f}s)")

    # ── COMPARISON ────────────────────────────────────────────────────
    out(f"\n{'='*76}")
    out("COMPARISON: 256K Effective Context")
    out(f"{'='*76}\n")

    out(f"{'Metric':<30} {'Instruct-2507':>16} {'Coder-1M':>16}")
    out("-" * 64)

    for metric, key in [
        ("Effective tokens reached", "effective_tokens"),
        ("Cycles completed", "cycles"),
        ("Reached 256K target?", "reached_target"),
        ("Avg generation tok/s", "avg_gen_tok_s"),
        ("Avg compaction time (ms)", "avg_compact_ms"),
        ("Total wall time (s)", "total_wall_s"),
    ]:
        vals = []
        for m in MODELS:
            r = all_results.get(m["name"], {})
            v = r.get(key, "?")
            if isinstance(v, bool):
                v = "YES" if v else "NO"
            elif isinstance(v, (int, float)) and key == "effective_tokens":
                v = f"{v:,}"
            vals.append(str(v))
        out(f"{metric:<30} {vals[0]:>16} {vals[1]:>16}")

    # Recall comparison
    out(f"\n{'Recall Checkpoint':<30} {'Instruct-2507':>16} {'Coder-1M':>16}")
    out("-" * 64)
    for cp in RECALL_CHECKPOINTS:
        vals = []
        for m in MODELS:
            r = all_results.get(m["name"], {})
            rc = r.get("recall_checkpoints", {}).get(str(cp), {})
            if rc:
                vals.append(f"{rc['recalled']}/10 ({rc['pct']}%)")
            else:
                vals.append("not reached")
        out(f"{cp//1000}K effective{'':<19} {vals[0]:>16} {vals[1]:>16}")

    # Tok/s progression
    out(f"\n── TOK/S PROGRESSION (every 5 cycles) ──")
    for m in MODELS:
        r = all_results.get(m["name"], {})
        data = r.get("cycle_data", [])
        if not data:
            continue
        out(f"\n  {m['name']}:")
        out(f"  {'Cycle':>6} {'Effective':>12} {'Gen tok/s':>10} {'Prompt tok/s':>12} {'Compact ms':>11}")
        for c in data:
            if c["cycle"] == 1 or c["cycle"] % 5 == 0 or c["cycle"] == len(data):
                out(f"  {c['cycle']:>6} {c['effective']:>12,} {c['gen_tok_s']:>10.1f} "
                    f"{c['prompt_tok_s']:>12.0f} {c['compact_ms']:>11.0f}")

    # Save
    outdir = os.path.join(os.path.dirname(__file__), "..", "bench-results")
    outpath = os.path.join(outdir, f"256k-effective-{timestamp}.json")
    os.makedirs(outdir, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump({
            "benchmark": "256k-effective-context",
            "timestamp": timestamp,
            "hardware": "Apple M2 Pro 32GB",
            "context_window": CTX,
            "target_effective": TARGET_EFFECTIVE,
            "compact_ratio": COMPACT_RATIO,
            "method": "select",
            "results": all_results,
        }, f, indent=2, default=str)

    out(f"\nResults: {outpath}")
    out("BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
