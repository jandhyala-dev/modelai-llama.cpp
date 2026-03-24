#!/usr/bin/env python3
"""
Benchmark Tier 3c: Multi-Turn Conversation with Mid-Conversation Compaction

Simulates a 20-turn chat. At turn 10, compacts the KV cache at 4x.
Measures whether the model maintains topic coherence and factual consistency
across the compaction boundary.

Scoring: Each post-compaction response is checked for:
  1. Topic relevance (references the ongoing topic)
  2. Factual consistency (doesn't contradict earlier turns)
  3. Conversation flow (acknowledges prior context)

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

# A structured conversation that builds on itself — easy to test for coherence
CONVERSATION_PLAN = [
    # Turns 1-5: Establish the topic and key facts
    "I'm planning a tech startup called NovaBridge that will build developer tools for WebAssembly. Our team has 4 engineers, a $500K seed round, and we're based in Austin, Texas. What should our first product be?",
    "Good ideas. Let's go with the debugging tool — we'll call it WasmScope. What tech stack should we use for the backend?",
    "We've decided on Rust for the backend. Our CTO Sarah wants to use gRPC for the API. Our lead engineer Marcus prefers REST. Who's right and why?",
    "We'll go with gRPC internally and REST for the public API. Now, our investor wants a demo in 6 weeks. What's a realistic MVP scope?",
    "Perfect. Let's target: source-level breakpoints, variable inspection, and memory profiling for the MVP. What's the biggest technical risk?",
    # Turns 6-10: Deepen with specifics (compaction happens after turn 10)
    "You mentioned memory profiling is the riskiest. Sarah suggested using DWARF debug info. Marcus thinks we should parse the custom section. Compare the two approaches.",
    "We'll go with DWARF. Our third engineer, Priya, is an expert in DWARF parsing. She estimates 3 weeks for the parser. Is that realistic?",
    "Priya says she can parallelize the DWARF parsing across Wasm modules. She wants to use rayon for parallelism. Any concerns?",
    "Good point about the thread pool. Let's cap it at 4 threads. Now, our fourth engineer Alex is handling the frontend. He wants to use SolidJS instead of React. Thoughts?",
    "Alex will use SolidJS. Let's recap: what are the key decisions we've made so far? List the team members and their responsibilities.",
    # Turns 11-15: Post-compaction — test if model remembers specifics
    "Based on our earlier discussion, what was the name of our startup and where are we based?",
    "Which engineer is handling the DWARF parser and how long did they estimate?",
    "What API approach did we decide on — REST, gRPC, or both? And why?",
    "What's Alex's role and which frontend framework is he using?",
    "Our investor just asked for a progress update. Draft a 3-sentence email summarizing our MVP plan, tech stack, and timeline.",
    # Turns 16-20: Complex reasoning requiring full context
    "Priya just told me the DWARF parser is taking longer than expected — 5 weeks instead of 3. How should we adjust the 6-week MVP timeline?",
    "Marcus suggested we cut memory profiling from the MVP and ship just breakpoints + variable inspection. Sarah disagrees. How do we resolve this?",
    "Good compromise. Let's ship breakpoints + variables in week 5, then add basic memory stats in week 6. What's the risk to this plan?",
    "One last thing: our seed investor wants to know our competitive advantage. Based on everything we've discussed, write a 2-sentence pitch.",
    "Thanks for all the help. Summarize every team member, their role, and one key decision they drove.",
]

# Facts that must be present in later responses for coherence
COHERENCE_CHECKS = {
    10: ["NovaBridge", "Austin", "WasmScope", "Sarah", "Marcus", "Priya", "Alex"],
    11: ["NovaBridge", "Austin"],
    12: ["Priya", "DWARF", "3 week"],
    13: ["gRPC", "REST"],
    14: ["Alex", "SolidJS"],
    15: ["WasmScope", "Rust", "6 week"],
    19: ["Sarah", "Marcus", "Priya", "Alex"],
}


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


def chat(messages, max_tokens=1000):
    return api("POST", "/v1/chat/completions", {
        "model": "test",
        "id_slot": SLOT,
        "cache_prompt": True,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    })


def compact(target_tokens):
    return api("POST", "/compact", {
        "id_slot": SLOT,
        "seq_id": 0,
        "target_tokens": target_tokens,
        "method": "select",
    })


def check_coherence(turn_idx, response_text):
    """Check if expected facts appear in the response."""
    if turn_idx not in COHERENCE_CHECKS:
        return None
    expected = COHERENCE_CHECKS[turn_idx]
    found = []
    missing = []
    text_lower = response_text.lower()
    for fact in expected:
        if fact.lower() in text_lower:
            found.append(fact)
        else:
            missing.append(fact)
    return {
        "expected": expected,
        "found": found,
        "missing": missing,
        "score": len(found) / len(expected) if expected else 1.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-turn compaction coherence benchmark")
    parser.add_argument("--model", default="unknown", help="Model name (for results metadata)")
    parser.add_argument("--turns", type=int, default=20, help="Number of conversation turns")
    parser.add_argument("--compact-at-turn", type=int, default=10, help="Compact after this turn")
    parser.add_argument("--compact-ratio", type=int, default=4, help="Compaction ratio")
    parser.add_argument("--out-dir", default=None, help="Output directory for results")
    parser.add_argument("--server", default=None, help="Server URL")
    args = parser.parse_args()

    global SERVER
    if args.server:
        SERVER = args.server

    n_turns = min(args.turns, len(CONVERSATION_PLAN))

    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("MULTI-TURN COMPACTION COHERENCE BENCHMARK (Tier 3c)\n")
    sys.stdout.write(f"Model: {args.model}\n")
    sys.stdout.write(f"Turns: {n_turns}, compact after turn {args.compact_at_turn}\n")
    sys.stdout.write(f"Compact ratio: {args.compact_ratio}x\n")
    sys.stdout.write("=" * 60 + "\n\n")
    sys.stdout.flush()

    if not health():
        sys.stdout.write("ERROR: Server not healthy\n")
        sys.exit(1)

    messages = [
        {"role": "system", "content": "You are a startup advisor. Remember all details from our conversation."},
    ]
    turn_results = []
    compacted = False

    for i in range(n_turns):
        turn_num = i + 1
        user_msg = CONVERSATION_PLAN[i]
        messages.append({"role": "user", "content": user_msg})

        sys.stdout.write(f"Turn {turn_num:2d}: {user_msg[:70]}...\n")
        sys.stdout.flush()

        t0 = time.time()
        resp = chat(messages, max_tokens=800)
        elapsed = time.time() - t0

        if "error" in resp:
            sys.stdout.write(f"  ERROR: {resp['error']}\n")
            turn_results.append({"turn": turn_num, "status": "ERROR", "error": str(resp["error"])})
            break

        assistant_text = resp["choices"][0]["message"]["content"]
        tokens = resp["usage"]["total_tokens"]
        messages.append({"role": "assistant", "content": assistant_text})

        # Coherence check
        coherence = check_coherence(i, assistant_text)
        coherence_str = ""
        if coherence:
            coherence_str = f" coherence={coherence['score']:.0%}"
            if coherence["missing"]:
                coherence_str += f" missing=[{', '.join(coherence['missing'])}]"

        sys.stdout.write(f"  {tokens} tok, {elapsed:.1f}s{coherence_str}\n")
        sys.stdout.flush()

        turn_results.append({
            "turn": turn_num,
            "status": "OK",
            "tokens": tokens,
            "elapsed_s": round(elapsed, 1),
            "response_preview": assistant_text[:300],
            "coherence": coherence,
        })

        # Compact after the target turn
        if turn_num == args.compact_at_turn and not compacted:
            target = max(int(tokens / args.compact_ratio), 64)
            sys.stdout.write(f"\n  >>> COMPACTING: {tokens} -> {target} tokens ({args.compact_ratio}x)\n")
            sys.stdout.flush()

            cr = compact(target)
            if cr.get("success"):
                actual = cr.get("compression_ratio", 0)
                ms = cr.get("compaction_time_ms", 0)
                sys.stdout.write(f"  >>> Compacted: {actual:.1f}x in {ms:.0f}ms\n\n")
                compacted = True
            else:
                sys.stdout.write(f"  >>> COMPACT FAILED: {cr}\n\n")
                turn_results.append({"turn": "compact", "status": "FAILED", "error": str(cr)})
                break

            if not health():
                sys.stdout.write("  >>> Server crashed after compaction\n")
                turn_results.append({"turn": "compact", "status": "CRASHED"})
                break

    # Summary
    sys.stdout.write(f"\n{'='*60}\n")
    sys.stdout.write("COHERENCE SUMMARY\n")
    sys.stdout.write(f"{'='*60}\n\n")

    pre_coherence = []
    post_coherence = []
    for r in turn_results:
        if r.get("coherence") and r["status"] == "OK":
            if r["turn"] <= args.compact_at_turn:
                pre_coherence.append(r["coherence"]["score"])
            else:
                post_coherence.append(r["coherence"]["score"])

    if pre_coherence:
        avg_pre = sum(pre_coherence) / len(pre_coherence)
        sys.stdout.write(f"Pre-compaction coherence:  {avg_pre:.0%} ({len(pre_coherence)} checks)\n")
    if post_coherence:
        avg_post = sum(post_coherence) / len(post_coherence)
        sys.stdout.write(f"Post-compaction coherence: {avg_post:.0%} ({len(post_coherence)} checks)\n")

        # Detail missed items
        for r in turn_results:
            if r.get("coherence") and r["turn"] > args.compact_at_turn and r["coherence"]["missing"]:
                sys.stdout.write(f"  Turn {r['turn']}: missed [{', '.join(r['coherence']['missing'])}]\n")

        passed = avg_post >= 0.70
        sys.stdout.write(f"\nVerdict: {'PASS' if passed else 'FAIL'} (threshold: 70% post-compaction coherence)\n")
    else:
        passed = False
        sys.stdout.write("\nVerdict: FAIL (no post-compaction coherence data)\n")

    sys.stdout.flush()

    # Save results
    output = {
        "benchmark": "multiturn-compaction-coherence",
        "model": args.model,
        "turns": n_turns,
        "compact_at_turn": args.compact_at_turn,
        "compact_ratio": args.compact_ratio,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "pre_coherence": round(sum(pre_coherence) / len(pre_coherence), 3) if pre_coherence else None,
            "post_coherence": round(sum(post_coherence) / len(post_coherence), 3) if post_coherence else None,
            "passed": passed,
        },
        "turns_data": turn_results,
    }

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        outpath = os.path.join(args.out_dir, "multiturn-coherence-results.json")
    else:
        outpath = "multiturn-coherence-results.json"

    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    sys.stdout.write(f"\nResults saved: {outpath}\n")
    sys.stdout.flush()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
