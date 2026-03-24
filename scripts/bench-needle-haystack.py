#!/usr/bin/env python3
"""
Benchmark Tier 1c: Needle-in-Haystack with KV Compaction

Injects a unique fact ("needle") at a specific position in a long context
("haystack"), compacts at a target ratio, then queries for the needle.

Tests positional recall: does compaction preserve information regardless of
where it appears in the context?

Positions: start (5%), middle (50%), end (95%)
Fill sizes: 8192, 32768, 65536 tokens
Compact ratio: 4x (default)

Requires: llama-server running with --endpoint-compact
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

SERVER = os.environ.get("BENCH_SERVER", "http://localhost:8090")
SLOT = 0

# Unique, verifiable needles — unlikely to appear in filler text
NEEDLES = [
    {"id": "N1", "fact": "The Krakatoa volcanic eruption of 1883 produced a sound measured at 172 decibels.", "query": "How many decibels was the Krakatoa 1883 eruption sound?", "answer": "172"},
    {"id": "N2", "fact": "Lake Baikal contains approximately 23,615 cubic kilometers of fresh water.", "query": "How many cubic kilometers of fresh water does Lake Baikal contain?", "answer": "23615"},
    {"id": "N3", "fact": "The Voyager 1 spacecraft is traveling at 17.06 kilometers per second.", "query": "What is Voyager 1's speed in kilometers per second?", "answer": "17.06"},
]

# Filler topics to generate haystack content
FILLER_PROMPTS = [
    "Explain the complete history of Roman aqueduct engineering, covering materials, gradient calculations, siphon systems, and maintenance schedules across all provinces.",
    "Describe the biochemistry of photosynthesis in C3 and C4 plants, including every enzyme in the Calvin cycle, photorespiration pathways, and quantum yield differences.",
    "Analyze the monetary policy tools used by the Federal Reserve from 2008 to 2024, including quantitative easing mechanics, reverse repo operations, and yield curve control debates.",
    "Detail the aerodynamics of supersonic flight, covering shock wave formation, area rule design, thermal protection systems, and scramjet propulsion principles.",
    "Explain the mathematics of elliptic curve cryptography from first principles, including group law, discrete logarithm hardness, and NIST curve parameter selection.",
]


def api(method, path, data=None, timeout=300):
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
    except Exception:
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


def compact(target_tokens):
    return api("POST", "/compact", {
        "id_slot": SLOT,
        "seq_id": 0,
        "target_tokens": target_tokens,
        "method": "select",
    })


def build_haystack(needle_text, position_pct, fill_tokens):
    """Build a multi-turn conversation that places a needle at a target position.

    Generates real assistant responses iteratively to actually fill the KV cache
    to the target token count. The needle is injected at the turn closest to the
    target position percentage.
    """
    # Estimate tokens per turn: ~200 prompt + ~1500 response = ~1700
    estimated_turns = max(2, fill_tokens // 1700)
    needle_turn = max(0, round(estimated_turns * position_pct) - 1)

    messages = [
        {"role": "system", "content": "You are a knowledgeable assistant. Answer in detail."},
    ]

    total_tokens = 0
    turn = 0

    while total_tokens < fill_tokens and turn < estimated_turns + 10:
        filler = FILLER_PROMPTS[turn % len(FILLER_PROMPTS)]
        if turn == needle_turn:
            messages.append({
                "role": "user",
                "content": f"Before answering, note this critical fact: {needle_text}\n\nNow, {filler}",
            })
        else:
            messages.append({"role": "user", "content": filler})

        # Generate a real assistant response to fill the KV cache
        resp = chat(messages, max_tokens=1500)
        if "error" in resp:
            sys.stdout.write(f"    Fill turn {turn}: error — {resp['error']}\n")
            sys.stdout.flush()
            break

        assistant_text = resp["choices"][0]["message"]["content"]
        total_tokens = resp["usage"]["total_tokens"]
        messages.append({"role": "assistant", "content": assistant_text})

        sys.stdout.write(f"    Fill turn {turn}: {total_tokens} tokens\n")
        sys.stdout.flush()
        turn += 1

    return messages, needle_turn, total_tokens


def run_test(needle, position_pct, fill_tokens, compact_ratio):
    """Run a single needle-in-haystack test."""
    pos_label = f"{int(position_pct * 100)}%"
    sys.stdout.write(f"  Needle {needle['id']} @ {pos_label}, fill={fill_tokens}, ratio={compact_ratio}x\n")
    sys.stdout.flush()

    # Step 1: Build and fill context with real generated responses
    messages, needle_turn, total_tokens = build_haystack(needle["fact"], position_pct, fill_tokens)

    if total_tokens == 0:
        return {"status": "FILL_ERROR", "error": "no tokens generated"}

    sys.stdout.write(f"    Filled: {total_tokens} tokens (needle at turn {needle_turn})\n")
    sys.stdout.flush()

    # Step 2: Pre-compaction retrieval
    # messages already ends with an assistant response from the fill phase
    query_messages = messages + [
        {"role": "user", "content": f"{needle['query']} Answer with ONLY the number."},
    ]
    pre_resp = chat(query_messages, max_tokens=100)
    if "error" in pre_resp:
        pre_found = False
    else:
        pre_answer = pre_resp["choices"][0]["message"]["content"]
        pre_found = needle["answer"] in pre_answer.replace(",", "")

    sys.stdout.write(f"    Pre-compact: {'FOUND' if pre_found else 'MISSED'}\n")
    sys.stdout.flush()

    # Step 3: Compact
    target = max(int(total_tokens / compact_ratio), 64)
    cr = compact(target)
    if not cr.get("success"):
        return {
            "status": "COMPACT_FAILED",
            "error": str(cr),
            "pre_found": pre_found,
            "total_tokens": total_tokens,
        }

    actual_ratio = cr.get("compression_ratio", 0)
    compact_ms = cr.get("compaction_time_ms", 0)
    sys.stdout.write(f"    Compacted: {actual_ratio:.1f}x in {compact_ms:.0f}ms\n")
    sys.stdout.flush()

    # Step 4: Post-compaction retrieval
    if not health():
        return {"status": "CRASHED", "pre_found": pre_found, "total_tokens": total_tokens}

    post_resp = chat(query_messages, max_tokens=100)
    if "error" in post_resp:
        post_found = False
        post_answer = str(post_resp["error"])
    else:
        post_answer = post_resp["choices"][0]["message"]["content"]
        post_found = needle["answer"] in post_answer.replace(",", "")

    sys.stdout.write(f"    Post-compact: {'FOUND' if post_found else 'MISSED'}\n")
    sys.stdout.flush()

    return {
        "status": "OK",
        "needle_id": needle["id"],
        "position_pct": position_pct,
        "fill_tokens": fill_tokens,
        "total_tokens": total_tokens,
        "compact_ratio": compact_ratio,
        "actual_ratio": round(actual_ratio, 1),
        "compact_ms": round(compact_ms, 1),
        "pre_found": pre_found,
        "post_found": post_found,
        "post_answer": post_answer[:200],
    }


def main():
    parser = argparse.ArgumentParser(description="Needle-in-haystack compaction benchmark")
    parser.add_argument("--model", default="unknown", help="Model name (for results metadata)")
    parser.add_argument("--fill-sizes", default="8192,32768,65536", help="Comma-separated fill token targets")
    parser.add_argument("--compact-ratio", type=int, default=4, help="Compaction ratio")
    parser.add_argument("--needle-positions", default="0.05,0.50,0.95", help="Comma-separated needle positions (0-1)")
    parser.add_argument("--out-dir", default=None, help="Output directory for results")
    parser.add_argument("--server", default=None, help="Server URL (default: $BENCH_SERVER or localhost:8090)")
    args = parser.parse_args()

    global SERVER
    if args.server:
        SERVER = args.server

    fill_sizes = [int(x) for x in args.fill_sizes.split(",")]
    positions = [float(x) for x in args.needle_positions.split(",")]

    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("NEEDLE-IN-HAYSTACK COMPACTION BENCHMARK (Tier 1c)\n")
    sys.stdout.write(f"Model: {args.model}\n")
    sys.stdout.write(f"Fill sizes: {fill_sizes}\n")
    sys.stdout.write(f"Positions: {[f'{p:.0%}' for p in positions]}\n")
    sys.stdout.write(f"Compact ratio: {args.compact_ratio}x\n")
    sys.stdout.write("=" * 60 + "\n\n")
    sys.stdout.flush()

    if not health():
        sys.stdout.write("ERROR: Server not healthy\n")
        sys.exit(1)

    results = []
    for fill in fill_sizes:
        for pos in positions:
            for needle in NEEDLES:
                result = run_test(needle, pos, fill, args.compact_ratio)
                results.append(result)
                if result.get("status") == "CRASHED":
                    sys.stdout.write("Server crashed — aborting\n")
                    break
            else:
                continue
            break
        else:
            continue
        break

    # Summary
    ok_results = [r for r in results if r.get("status") == "OK"]
    total = len(ok_results)
    pre_found = sum(1 for r in ok_results if r["pre_found"])
    post_found = sum(1 for r in ok_results if r["post_found"])

    sys.stdout.write(f"\n{'='*60}\n")
    sys.stdout.write("SUMMARY\n")
    sys.stdout.write(f"{'='*60}\n\n")
    sys.stdout.write(f"Tests completed: {total}/{len(results)}\n")
    sys.stdout.write(f"Pre-compaction retrieval:  {pre_found}/{total} ({pre_found/total*100:.0f}%)\n" if total else "")
    sys.stdout.write(f"Post-compaction retrieval: {post_found}/{total} ({post_found/total*100:.0f}%)\n" if total else "")

    # Breakdown by position
    for pos in positions:
        pos_results = [r for r in ok_results if r.get("position_pct") == pos]
        if pos_results:
            found = sum(1 for r in pos_results if r["post_found"])
            sys.stdout.write(f"  Position {pos:.0%}: {found}/{len(pos_results)} retrieved\n")

    # Pass criteria: >= 80% retrieval at all positions
    passed = total > 0 and (post_found / total) >= 0.80
    sys.stdout.write(f"\nVerdict: {'PASS' if passed else 'FAIL'} (threshold: 80%)\n")
    sys.stdout.flush()

    # Save results
    output = {
        "benchmark": "needle-in-haystack",
        "model": args.model,
        "compact_ratio": args.compact_ratio,
        "fill_sizes": fill_sizes,
        "positions": positions,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_tests": total,
            "pre_found": pre_found,
            "post_found": post_found,
            "retrieval_pct": round(post_found / total * 100, 1) if total else 0,
            "passed": passed,
        },
        "results": results,
    }

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        outpath = os.path.join(args.out_dir, "needle-haystack-results.json")
    else:
        outpath = "needle-haystack-results.json"

    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    sys.stdout.write(f"\nResults saved: {outpath}\n")
    sys.stdout.flush()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
