#!/usr/bin/env python3
"""
Comprehensive KV Compaction Benchmark — All Methods × All Ratios
Model: Qwen3-Coder-30B-A3B-Instruct-1M (UD-Q4_K_XL)
Hardware: Apple M2 Pro 32GB, 64K context, f16 KV

Tests each compaction method at 2x, 4x, 10x, 25x, 50x.
Captures: compaction time, recall quality, facts lost/kept.

Key design: multi-turn conversation builds ~8K tokens in KV cache.
Recall tests CONTINUE the same conversation (no KV cache replacement).
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

SERVER = "http://localhost:8090"
SLOT = 0

# Fast methods get all ratios; slow methods get representative ratios
FAST_METHODS = ["select", "solver", "omp"]
SLOW_METHODS = ["self_study", "on_policy"]
ALL_RATIOS = [2, 4, 10, 25, 50]
SLOW_RATIOS = [4, 10, 25]  # representative subset for expensive methods

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


def api(method, path, data=None, timeout=600):
    url = f"{SERVER}{path}"
    body = json.dumps(data).encode() if data else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": json.loads(e.read())}
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


FILL_PROMPTS = [
    (
        "Provide an exhaustive analysis of NVIDIA and AMD in the semiconductor industry. "
        "Cover NVIDIA's data center revenue of $115 billion, Blackwell architecture, CUDA ecosystem "
        "with 4 million developers, and competitive moat. For AMD, cover total revenue of $28 billion, "
        "MI300X AI accelerator revenue of $5 billion, EPYC server gains, and Xilinx synergies. "
        "Include ALL specific numbers. Be extremely detailed with growth rates and margins."
    ),
    (
        "Now analyze Intel, TSMC, and SK Hynix in detail. Intel has $7 billion in foundry losses — "
        "cover 18A process timeline, Gaudi AI accelerator challenges, and restructuring. "
        "TSMC has $90 billion revenue — cover CoWoS advanced packaging, N2 node timeline, "
        "and geopolitical risks. SK Hynix has 50% HBM3E market share — cover technology lead "
        "over Samsung, capacity expansion, and pricing. Include ALL specific numbers."
    ),
    (
        "Finally, cover Broadcom's VMware synergy target of $8.5 billion, sovereign AI investment "
        "of $150 billion globally (UAE, Saudi Arabia, India, Japan, EU with specific figures), "
        "Apple's annual buyback pace of $110 billion and services trajectory, and the "
        "training-to-inference transition. Include ALL specific dollar amounts and percentages."
    ),
]


def fill_context_multi_turn():
    """Build ~8K tokens via multi-turn conversation. Returns (total_tokens, messages)."""
    sys_prompt = build_facts_system()
    msgs = [{"role": "system", "content": sys_prompt}]
    total = 0

    for i, prompt in enumerate(FILL_PROMPTS):
        msgs.append({"role": "user", "content": prompt})
        resp = chat(msgs, max_tokens=2000)

        if "error" in resp:
            return None, msgs, str(resp["error"])

        content = resp["choices"][0]["message"]["content"]
        msgs.append({"role": "assistant", "content": content})
        total = resp["usage"]["total_tokens"]
        gen = resp["usage"]["completion_tokens"]
        sys.stdout.write(f"    Turn {i+1}: +{gen} gen, {total} total\n")
        sys.stdout.flush()

    return total, msgs, None


def test_recall(context_msgs):
    """Test recall of injected facts. CONTINUES the conversation (preserves KV)."""
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

    msgs = list(context_msgs)
    msgs.append({"role": "user", "content": recall_prompt})

    resp = chat(msgs, max_tokens=500)
    if "error" in resp:
        return None, str(resp["error"])

    answer = resp["choices"][0]["message"]["content"]
    tokens = resp["usage"]["completion_tokens"]

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
        "details": scores,
        "raw_answer": answer[:600],
        "tokens": tokens,
    }, answer


def run_test(method, ratio, fill_msgs_cache=None):
    """Run one test: fill → pre-recall → compact → post-recall."""
    P = f"  [{method:>12s} @ {ratio:>2d}x]"

    # Step 1: Fill context (multi-turn conversation)
    sys.stdout.write(f"{P} Filling context (multi-turn)...\n")
    sys.stdout.flush()
    total, msgs, err = fill_context_multi_turn()
    if total is None:
        sys.stdout.write(f"{P} FILL ERROR: {err}\n")
        sys.stdout.flush()
        return {"method": method, "ratio": ratio, "status": "FILL_ERROR", "error": err}

    sys.stdout.write(f"{P} Filled {total} tokens in KV\n")
    sys.stdout.flush()

    # Step 2: Pre-compaction recall (CONTINUES same conversation)
    sys.stdout.write(f"{P} Pre-compact recall...\n")
    sys.stdout.flush()
    pre, _ = test_recall(msgs)
    pre_score = pre["pct"] if pre else "ERR"
    sys.stdout.write(f"{P} Pre-compact: {pre_score}%\n")
    sys.stdout.flush()

    # Step 3: Re-prime KV with original conversation before compacting
    # (recall test added tokens; re-send original to restore KV state)
    sys.stdout.write(f"{P} Re-priming KV for compaction...\n")
    sys.stdout.flush()
    prime_resp = chat(msgs, max_tokens=1)
    if "error" in prime_resp:
        sys.stdout.write(f"{P} RE-PRIME ERROR: {prime_resp['error']}\n")
        sys.stdout.flush()
        return {"method": method, "ratio": ratio, "status": "PRIME_ERROR",
                "error": str(prime_resp["error"])}
    kv_tokens = prime_resp["usage"]["total_tokens"]
    sys.stdout.write(f"{P} KV primed: {kv_tokens} tokens\n")
    sys.stdout.flush()

    # Step 4: Compact
    target = max(int(kv_tokens / ratio), 32)
    sys.stdout.write(f"{P} Compacting {kv_tokens}→{target} ({method})...\n")
    sys.stdout.flush()

    t0 = time.time()
    cr = compact(target, method)
    t1 = time.time()
    wall_ms = round((t1 - t0) * 1000, 1)

    if not cr.get("success"):
        err = cr.get("error", cr)
        if isinstance(err, dict):
            err = err.get("message", str(err))
        sys.stdout.write(f"{P} COMPACT FAILED: {err}\n")
        sys.stdout.flush()
        return {"method": method, "ratio": ratio, "status": "COMPACT_FAILED",
                "error": str(err), "kv_before": kv_tokens, "pre_recall_pct": pre_score}

    actual_ratio = cr.get("compression_ratio", 0)
    compact_ms = cr.get("compaction_time_ms", 0)
    cosine = cr.get("cosine_similarity", None)
    compacted_tokens = cr.get("compacted_tokens", 0)
    sys.stdout.write(
        f"{P} Compacted: {actual_ratio:.1f}x in {compact_ms:.0f}ms "
        f"(wall {wall_ms:.0f}ms)"
    )
    if cosine is not None:
        sys.stdout.write(f" cosine={cosine:.4f}")
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Step 5: Post-compaction recall (CONTINUES same conversation)
    if not health():
        sys.stdout.write(f"{P} SERVER CRASHED\n")
        sys.stdout.flush()
        return {"method": method, "ratio": ratio, "status": "CRASHED",
                "kv_before": kv_tokens, "actual_ratio": actual_ratio,
                "compact_ms": compact_ms, "pre_recall_pct": pre_score}

    sys.stdout.write(f"{P} Post-compact recall...\n")
    sys.stdout.flush()
    post, _ = test_recall(msgs)

    if post is None:
        if not health():
            sys.stdout.write(f"{P} CRASHED during recall\n")
            sys.stdout.flush()
            return {"method": method, "ratio": ratio, "status": "CRASHED",
                    "kv_before": kv_tokens, "actual_ratio": actual_ratio,
                    "compact_ms": compact_ms, "pre_recall_pct": pre_score}
        post_score = "ERR"
    else:
        post_score = post["pct"]

    sys.stdout.write(f"{P} Post-compact: {post_score}%\n")
    sys.stdout.flush()

    lost = [s["fact"] for s in (post or {}).get("details", []) if not s["recalled"]]
    kept = [s["fact"] for s in (post or {}).get("details", []) if s["recalled"]]

    result = {
        "method": method,
        "ratio": ratio,
        "actual_ratio": round(actual_ratio, 1),
        "status": "OK",
        "kv_before": kv_tokens,
        "kv_after": compacted_tokens,
        "compact_ms": round(compact_ms, 1),
        "wall_ms": wall_ms,
        "cosine_similarity": cosine,
        "pre_recall_pct": pre["pct"] if pre else None,
        "post_recall_pct": post["pct"] if post else None,
        "pre_recall_count": f"{pre['recalled']}/{pre['total']}" if pre else None,
        "post_recall_count": f"{post['recalled']}/{post['total']}" if post else None,
        "facts_lost": lost,
        "facts_kept": kept,
        "pre_recall_raw": (pre or {}).get("raw_answer", "")[:300],
        "post_recall_raw": (post or {}).get("raw_answer", "")[:300],
    }

    sys.stdout.write(
        f"{P} DONE — {actual_ratio:.1f}x, {compact_ms:.0f}ms, "
        f"recall {pre_score}%→{post_score}%"
    )
    if cosine is not None:
        sys.stdout.write(f", cosine={cosine:.4f}")
    sys.stdout.write("\n\n")
    sys.stdout.flush()
    return result


def main():
    all_methods = FAST_METHODS + SLOW_METHODS
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.write("COMPREHENSIVE KV COMPACTION BENCHMARK\n")
    sys.stdout.write(f"Model: Qwen3-Coder-30B-A3B-Instruct-1M (UD-Q4_K_XL, Unsloth)\n")
    sys.stdout.write(f"Context: 64K | KV: f16 | Hardware: Apple M2 Pro 32GB\n")
    sys.stdout.write(f"Fast methods ({', '.join(FAST_METHODS)}): ratios {ALL_RATIOS}\n")
    sys.stdout.write(f"Slow methods ({', '.join(SLOW_METHODS)}): ratios {SLOW_RATIOS}\n")
    sys.stdout.write(f"Facts tracked: {len(FACTS)}\n")
    sys.stdout.write(f"Fill strategy: 3-turn conversation (~8K tokens)\n")
    sys.stdout.write("=" * 70 + "\n\n")
    sys.stdout.flush()

    if not health():
        sys.stdout.write("ERROR: Server not healthy\n")
        sys.exit(1)

    all_results = []

    for method in all_methods:
        ratios = ALL_RATIOS if method in FAST_METHODS else SLOW_RATIOS
        sys.stdout.write(f"\n{'='*50}\n")
        sys.stdout.write(f"METHOD: {method} (ratios: {ratios})\n")
        sys.stdout.write(f"{'='*50}\n\n")
        sys.stdout.flush()

        for ratio in ratios:
            result = run_test(method, ratio)
            all_results.append(result)

            if result.get("status") == "CRASHED":
                sys.stdout.write(
                    f"  Server crashed — skipping remaining ratios for {method}\n"
                )
                sys.stdout.flush()
                time.sleep(5)
                if not health():
                    sys.stdout.write("  Server did not recover — stopping\n")
                    sys.stdout.flush()
                    break
                continue

    # Summary table
    sys.stdout.write(f"\n{'='*90}\n")
    sys.stdout.write("RESULTS SUMMARY\n")
    sys.stdout.write(f"{'='*90}\n\n")

    sys.stdout.write(
        f"{'Method':>14s} {'Ratio':>6s} {'Status':>12s} {'Actual':>7s} "
        f"{'Time(ms)':>9s} {'Cosine':>8s} {'Pre':>6s} {'Post':>6s} "
        f"{'Lost':>5s}\n"
    )
    sys.stdout.write("-" * 90 + "\n")

    for r in all_results:
        method = r.get("method", "?")
        ratio = r.get("ratio", "?")
        status = r.get("status", "?")
        actual = r.get("actual_ratio", "")
        if actual:
            actual = f"{actual:.1f}x"
        t = r.get("compact_ms", "")
        if t:
            t = f"{t:.0f}"
        cosine = r.get("cosine_similarity")
        cos_s = f"{cosine:.4f}" if cosine is not None else "—"
        pre = r.get("pre_recall_pct", "")
        if pre is not None and pre != "":
            pre = f"{pre:.0f}%"
        post = r.get("post_recall_pct", "")
        if post is not None and post != "":
            post = f"{post:.0f}%"
        lost = len(r.get("facts_lost", []))

        sys.stdout.write(
            f"{method:>14s} {str(ratio)+'x':>6s} {status:>12s} {str(actual):>7s} "
            f"{str(t):>9s} {cos_s:>8s} {str(pre):>6s} {str(post):>6s} "
            f"{lost:>5d}\n"
        )

    sys.stdout.write("\n")

    # Detail: facts lost per test
    sys.stdout.write("FACTS LOST DETAIL:\n")
    for r in all_results:
        if r.get("status") == "OK" and r.get("facts_lost"):
            sys.stdout.write(
                f"  {r['method']:>12s} @ {r['ratio']}x: "
                f"{', '.join(r['facts_lost'])}\n"
            )
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Save results
    output = {
        "benchmark": "compaction-all-methods-v2",
        "model": "Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL",
        "model_source": "unsloth",
        "context_window": 65536,
        "kv_type": "f16",
        "hardware": "Apple M2 Pro 32GB",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fill_strategy": "3-turn multi-turn conversation",
        "fast_methods": FAST_METHODS,
        "slow_methods": SLOW_METHODS,
        "all_ratios": ALL_RATIOS,
        "slow_ratios": SLOW_RATIOS,
        "facts": [{"name": n, "value": v} for n, v, _ in FACTS],
        "results": all_results,
    }

    outdir = "/Users/ajayjandhyala/dev/whippet/modelai-llama.cpp/bench-results"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "compaction-all-methods-benchmark.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    sys.stdout.write(f"Results saved: {outpath}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
