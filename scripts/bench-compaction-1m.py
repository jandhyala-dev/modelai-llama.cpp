#!/usr/bin/env python3
"""
Benchmark: KV Compaction Quality — What data is lost at each ratio?

Injects 10 specific, verifiable facts into context, then compacts at 2x/4x/10x/25x/50x
and tests recall of each fact. Scores: how many facts survive each compression level.

Phase 2: Iterative compaction cycles to simulate 1M effective context.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

SERVER = "http://localhost:8090"
SLOT = 0

# 10 specific facts the model must remember — easy to verify
FACTS = [
    ("NVIDIA data center revenue", "$115 billion"),
    ("AMD total revenue", "$28 billion"),
    ("Intel foundry losses", "$7 billion"),
    ("TSMC revenue", "$90 billion"),
    ("SK Hynix HBM3E market share", "50 percent"),
    ("Broadcom VMware synergy target", "$8.5 billion"),
    ("Total sovereign AI investment globally", "$150 billion"),
    ("Apple annual buyback pace", "$110 billion"),
    ("NVIDIA CUDA developer count", "4 million"),
    ("AMD MI300X AI accelerator revenue", "$5 billion"),
]

COMPACT_RATIOS = [2, 4, 10, 25, 50]


def api(method, path, data=None, timeout=300):
    url = f"{SERVER}{path}"
    body = json.dumps(data).encode() if data else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())
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
        "temperature": 0.1,  # low temp for deterministic recall
    })


def compact(target_tokens):
    return api("POST", "/compact", {
        "id_slot": SLOT,
        "seq_id": 0,
        "target_tokens": target_tokens,
        "method": "select",
    })


def build_facts_prompt():
    """Build a system prompt with all 10 facts clearly stated."""
    lines = ["You are a financial analyst. MEMORIZE these exact facts:"]
    for i, (name, value) in enumerate(FACTS, 1):
        lines.append(f"  FACT {i}: {name} = {value}")
    lines.append("")
    lines.append("You MUST recall these exact numbers when asked. Do not use general knowledge.")
    return "\n".join(lines)


def build_filler_prompt():
    """Generate a dense prompt to fill context with real analysis content."""
    return (
        "Provide an exhaustive analysis of the semiconductor industry covering: "
        "1) NVIDIA competitive positioning with Blackwell architecture, CUDA ecosystem moat, "
        "InfiniBand networking revenue, and data center margin expansion. "
        "2) AMD MI300X traction vs NVIDIA, EPYC server CPU market share gains, Xilinx FPGA synergies. "
        "3) Intel foundry strategy viability, 18A process timeline, Gaudi AI accelerator challenges. "
        "4) TSMC advanced packaging CoWoS capacity expansion, N2 node timeline, geopolitical risks. "
        "5) SK Hynix HBM3E technology lead over Samsung, capacity figures, pricing trends. "
        "6) Broadcom VMware integration progress, synergy realization targets. "
        "7) Sovereign AI investments by UAE, Saudi Arabia, India, Japan, and EU with specific figures. "
        "8) Apple capital return program, services revenue trajectory, Vision Pro TAM. "
        "9) The training-to-inference transition and its margin implications. "
        "10) Advanced packaging as the new bottleneck replacing lithography. "
        "Be extremely detailed with specific revenue figures, market share percentages, and growth rates."
    )


def test_recall():
    """Ask the model to recall each fact and score accuracy."""
    questions = []
    for name, value in FACTS:
        questions.append(f"What is {name}? Give ONLY the number.")

    # Ask all questions in one request
    prompt = "Answer each question with ONLY the specific number. One answer per line.\n\n"
    for i, q in enumerate(questions, 1):
        prompt += f"{i}. {q}\n"

    resp = chat([{"role": "user", "content": prompt}], max_tokens=500)
    if "error" in resp:
        return None, f"ERROR: {resp['error']}"

    answer = resp["choices"][0]["message"]["content"]
    tokens = resp["usage"]["completion_tokens"]

    # Score: check if each fact's value appears in the answer
    scores = []
    for i, (name, value) in enumerate(FACTS):
        # Extract the key number from value
        number = value.replace("$", "").replace(" billion", "").replace(" percent", "").replace(" million", "")
        # Check multiple formats
        found = False
        for variant in [number, value, value.replace("$", ""), f"${number}", f"{number}B", f"{number} billion"]:
            if variant.lower() in answer.lower():
                found = True
                break
        scores.append({"fact": name, "expected": value, "recalled": found})

    recalled = sum(1 for s in scores if s["recalled"])
    return {
        "recalled": recalled,
        "total": len(FACTS),
        "pct": round(recalled / len(FACTS) * 100, 1),
        "details": scores,
        "raw_answer": answer[:500],
        "tokens": tokens,
    }, answer


def run_single_ratio_test(ratio):
    """Run a complete fill → compact → recall test for one ratio."""
    sys.stdout.write(f"\n{'='*60}\n")
    sys.stdout.write(f"Testing {ratio}x compaction\n")
    sys.stdout.write(f"{'='*60}\n")
    sys.stdout.flush()

    # Step 1: Inject facts + fill context
    sys.stdout.write("  Step 1: Injecting facts + filling context...\n")
    sys.stdout.flush()

    facts_prompt = build_facts_prompt()
    filler = build_filler_prompt()

    fill_resp = chat([
        {"role": "system", "content": facts_prompt},
        {"role": "user", "content": filler},
    ], max_tokens=4000)

    if "error" in fill_resp:
        sys.stdout.write(f"  FILL ERROR: {fill_resp['error']}\n")
        sys.stdout.flush()
        return {"ratio": ratio, "status": "FILL_ERROR", "error": str(fill_resp["error"])}

    total_tokens = fill_resp["usage"]["total_tokens"]
    sys.stdout.write(f"  Filled: {total_tokens} tokens\n")
    sys.stdout.flush()

    # Step 2: Pre-compaction recall
    sys.stdout.write("  Step 2: Pre-compaction recall test...\n")
    sys.stdout.flush()

    pre_recall, pre_raw = test_recall()
    if pre_recall is None:
        sys.stdout.write(f"  PRE-RECALL ERROR: {pre_raw}\n")
        sys.stdout.flush()
        return {"ratio": ratio, "status": "PRE_RECALL_ERROR"}

    sys.stdout.write(f"  Pre-compact recall: {pre_recall['recalled']}/{pre_recall['total']} ({pre_recall['pct']}%)\n")
    sys.stdout.flush()

    # Step 3: Compact
    target = max(int(total_tokens / ratio), 32)
    sys.stdout.write(f"  Step 3: Compacting {total_tokens} → {target} tokens ({ratio}x)...\n")
    sys.stdout.flush()

    cr = compact(target)
    if not cr.get("success"):
        sys.stdout.write(f"  COMPACT FAILED: {cr}\n")
        sys.stdout.flush()
        return {"ratio": ratio, "status": "COMPACT_FAILED", "error": str(cr)}

    actual_ratio = cr.get("compression_ratio", 0)
    compact_ms = cr.get("compaction_time_ms", 0)
    sys.stdout.write(f"  Compacted: {actual_ratio:.1f}x in {compact_ms:.0f}ms\n")
    sys.stdout.flush()

    # Step 4: Post-compaction recall
    sys.stdout.write("  Step 4: Post-compaction recall test...\n")
    sys.stdout.flush()

    if not health():
        sys.stdout.write("  SERVER CRASHED after compaction!\n")
        sys.stdout.flush()
        return {"ratio": ratio, "status": "CRASHED"}

    post_recall, post_raw = test_recall()
    if post_recall is None:
        if not health():
            sys.stdout.write("  SERVER CRASHED during recall!\n")
            sys.stdout.flush()
            return {"ratio": ratio, "status": "CRASHED"}
        sys.stdout.write(f"  POST-RECALL ERROR: {post_raw}\n")
        sys.stdout.flush()
        return {"ratio": ratio, "status": "POST_RECALL_ERROR"}

    sys.stdout.write(f"  Post-compact recall: {post_recall['recalled']}/{post_recall['total']} ({post_recall['pct']}%)\n")
    sys.stdout.flush()

    # Detail which facts were lost
    lost = [s["fact"] for s in post_recall["details"] if not s["recalled"]]
    kept = [s["fact"] for s in post_recall["details"] if s["recalled"]]
    if lost:
        sys.stdout.write(f"  LOST: {', '.join(lost)}\n")
    sys.stdout.write(f"  KEPT: {', '.join(kept)}\n")
    sys.stdout.flush()

    return {
        "ratio": ratio,
        "actual_ratio": round(actual_ratio, 1),
        "status": "OK",
        "kv_before": total_tokens,
        "kv_after": cr.get("compacted_tokens", 0),
        "compact_ms": round(compact_ms, 1),
        "pre_recall": pre_recall,
        "post_recall": post_recall,
        "facts_lost": lost,
        "facts_kept": kept,
    }


def run_iterative_test():
    """Simulate reaching 1M effective context with 10x compaction cycles."""
    sys.stdout.write(f"\n{'='*60}\n")
    sys.stdout.write(f"ITERATIVE TEST: 10x compaction → 1M effective tokens\n")
    sys.stdout.write(f"{'='*60}\n")
    sys.stdout.flush()

    CYCLE_RATIO = 10
    TARGET = 1_000_000
    effective = 0
    cycle = 0
    cycle_results = []

    # Load source files for real content
    import glob
    src_dir = os.path.join(os.environ.get("MODELAI_DIR", "."), "src")
    files = []
    for ext in ["*.cpp", "*.h"]:
        for f in glob.glob(os.path.join(src_dir, "**", ext), recursive=True):
            try:
                with open(f) as fh:
                    content = fh.read()
                if len(content) > 500:
                    files.append((os.path.basename(f), content))
            except:
                pass
    files.sort(key=lambda x: len(x[1]), reverse=True)
    file_idx = 0

    while effective < TARGET:
        cycle += 1

        if not health():
            sys.stdout.write(f"  Server crashed at cycle {cycle}\n")
            sys.stdout.flush()
            break

        # Build content from source files
        code = ""
        n_files = 0
        while file_idx < len(files) and len(code) < 40000:
            fname, content = files[file_idx % len(files)]
            code += f"\n### {fname}\n```\n{content[:5000]}\n```\n"
            file_idx = (file_idx + 1) % len(files)
            n_files += 1

        t0 = time.time()
        fill = chat([
            {"role": "system", "content": "You are analyzing C++ source code."},
            {"role": "user", "content": f"Analyze:\n{code}\n\nSummarize key functions."},
        ], max_tokens=1500)
        t1 = time.time()

        if "error" in fill:
            sys.stdout.write(f"  Cycle {cycle}: Fill error - {fill.get('error', fill)}\n")
            sys.stdout.flush()
            break

        tokens = fill["usage"]["total_tokens"]
        effective += tokens

        # Compact
        target = max(int(tokens / CYCLE_RATIO), 64)
        cr = compact(target)

        if not cr.get("success"):
            sys.stdout.write(f"  Cycle {cycle}: Compact failed - {cr}\n")
            sys.stdout.flush()
            break

        pct = effective / TARGET * 100
        sys.stdout.write(
            f"  Cycle {cycle:3d}: +{tokens:5d} tok → {effective:>9,} effective ({pct:5.1f}%) | "
            f"{cr.get('compression_ratio',0):5.1f}x {cr.get('compaction_time_ms',0):6.0f}ms | "
            f"fill {t1-t0:5.1f}s\n"
        )
        sys.stdout.flush()

        cycle_results.append({
            "cycle": cycle,
            "tokens": tokens,
            "effective": effective,
            "ratio": round(cr.get("compression_ratio", 0), 1),
            "compact_ms": round(cr.get("compaction_time_ms", 0), 1),
            "fill_s": round(t1 - t0, 1),
        })

    return {
        "effective_tokens": effective,
        "cycles": len(cycle_results),
        "data": cycle_results,
    }


def main():
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("COMPACTION QUALITY BENCHMARK\n")
    sys.stdout.write(f"Model: Qwen3-Coder-30B-A3B-Instruct-1M (UD-Q4_K_XL)\n")
    sys.stdout.write(f"Context: 64K, KV: f16, Hardware: M2 Pro 32GB\n")
    sys.stdout.write("=" * 60 + "\n\n")
    sys.stdout.flush()

    if not health():
        sys.stdout.write("ERROR: Server not healthy\n")
        sys.exit(1)

    sys.stdout.write(f"Facts to inject and track: {len(FACTS)}\n")
    for i, (name, val) in enumerate(FACTS, 1):
        sys.stdout.write(f"  {i:2d}. {name} = {val}\n")
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Phase 1: Test each compaction ratio
    results = []
    for ratio in COMPACT_RATIOS:
        result = run_single_ratio_test(ratio)
        results.append(result)

        # Check if server crashed
        if result.get("status") == "CRASHED":
            sys.stdout.write("Server crashed — stopping Phase 1\n")
            break

    # Phase 2: Iterative compaction to 1M
    sys.stdout.write("\n")
    iterative = run_iterative_test()

    # Summary
    sys.stdout.write(f"\n{'='*60}\n")
    sys.stdout.write("SUMMARY: Data Loss by Compaction Ratio\n")
    sys.stdout.write(f"{'='*60}\n\n")

    sys.stdout.write(f"{'Ratio':>6} {'Status':>10} {'Pre':>8} {'Post':>8} {'Lost':>6} {'Facts Lost':>40}\n")
    sys.stdout.write("-" * 80 + "\n")
    for r in results:
        pre = r.get("pre_recall", {}).get("pct", "?")
        post = r.get("post_recall", {}).get("pct", "?")
        lost = ", ".join(r.get("facts_lost", [])[:3])
        if len(r.get("facts_lost", [])) > 3:
            lost += "..."
        n_lost = len(r.get("facts_lost", []))
        sys.stdout.write(
            f"{r.get('ratio','?'):>5}x {r.get('status','?'):>10} "
            f"{pre:>7}% {post:>7}% {n_lost:>5} {lost:>40}\n"
        )
    sys.stdout.flush()

    sys.stdout.write(f"\nIterative test: {iterative['effective_tokens']:,} effective tokens in {iterative['cycles']} cycles\n")
    sys.stdout.flush()

    # Save
    output = {
        "benchmark": "compaction-quality",
        "model": "Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL",
        "context_window": 65536,
        "kv_type": "f16",
        "hardware": "Apple M2 Pro 32GB",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "facts": [{"name": n, "value": v} for n, v in FACTS],
        "phase1": results,
        "phase2_iterative": iterative,
    }

    outpath = os.path.join(os.environ.get("MODELAI_DIR", "."), "bench-results/compaction-quality-benchmark.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    sys.stdout.write(f"\nResults saved: {outpath}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
