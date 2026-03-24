#!/usr/bin/env python3
"""
Master Benchmark: 6 Qwen3 Models — Identical Tests

Top 3 (30B MoE): Speed, Coding, Research, Compaction Quality (2x/4x)
Bottom 3 (misc): Speed, Coding, Research only (no compaction)
Hardware: Apple M2 Pro 32GB, 64K context, f16 KV
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ── Configuration ────────────────────────────────────────────────────
SERVER_BIN = os.path.join(os.path.dirname(__file__), "..", "build", "bin", "llama-server")
MODEL_DIR = os.environ["MODELAI_MODELS_DIR"]
PORT = 8090
SERVER = f"http://localhost:{PORT}"
CTX = 65536
SLOT = 0

MODELS = [
    # ── Top 3: 30B MoE — full tests + compaction ──
    {"name": "Instruct-2507", "file": "Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf", "type": "instruct", "compact": True},
    {"name": "Thinking-2507", "file": "Qwen3-30B-A3B-Thinking-2507-UD-Q4_K_XL.gguf", "type": "thinking", "compact": True},
    {"name": "Coder-1M",      "file": "Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf", "type": "coder", "compact": True},
    # ── Bottom 3: no compaction ──
    {"name": "Qwen3.5-35B-A3B", "file": "Qwen3.5-35B-A3B-Q4_K_M.gguf", "type": "instruct", "compact": False, "ctx": 32768},
    {"name": "Qwen3-14B",       "file": "Qwen3-14B-Q4_K_M.gguf",       "type": "instruct", "compact": False},
    {"name": "Qwen3-8B",        "file": "Qwen3-8B-Q4_K_M.gguf",        "type": "instruct", "compact": False},
]

# ── Financial Facts for Compaction Quality ───────────────────────────
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

# ── Test Prompts (IDENTICAL across all models) ───────────────────────

SPEED_TESTS = [
    {
        "name": "Trivial",
        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
        "max_tokens": 100,
    },
    {
        "name": "Short",
        "messages": [{"role": "user", "content": "Explain the difference between TCP and UDP in 3-4 sentences."}],
        "max_tokens": 300,
    },
    {
        "name": "Medium",
        "messages": [
            {"role": "system", "content": "You are a helpful technical writer."},
            {"role": "user", "content": (
                "Explain how a modern CPU executes instructions, covering: "
                "pipeline stages, branch prediction, out-of-order execution, "
                "cache hierarchy, and superscalar design. Be thorough."
            )},
        ],
        "max_tokens": 1000,
    },
    {
        "name": "Long",
        "messages": [
            {"role": "system", "content": "You are a computer science professor."},
            {"role": "user", "content": (
                "Write a comprehensive tutorial on distributed systems consensus algorithms. "
                "Cover Paxos, Raft, PBFT, and their tradeoffs. Include: "
                "1) The FLP impossibility result and why it matters. "
                "2) How Raft simplifies Paxos with leader election and log replication. "
                "3) Byzantine fault tolerance vs crash fault tolerance. "
                "4) Real-world deployments (etcd, ZooKeeper, CockroachDB). "
                "5) Performance characteristics and when to use each. "
                "Be detailed with examples."
            )},
        ],
        "max_tokens": 2000,
    },
]

CODING_TASKS = [
    {
        "name": "Algorithm",
        "messages": [
            {"role": "system", "content": "You are a software engineer. Write clean, efficient code."},
            {"role": "user", "content": (
                "Implement a function in Python that finds the longest increasing subsequence "
                "in an array of integers using dynamic programming with binary search (O(n log n)). "
                "Include: the function, time/space complexity analysis, and 3 test cases."
            )},
        ],
        "max_tokens": 1500,
    },
    {
        "name": "Code Review",
        "messages": [
            {"role": "system", "content": "You are a senior code reviewer. Find ALL bugs and issues."},
            {"role": "user", "content": """Review this Python code for bugs and issues:

```python
def merge_sorted_lists(list1, list2):
    result = []
    i = j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
    while i < len(list1):
        result.append(list1[i])
        i += 1
    return result

def binary_search(arr, target):
    low, high = 0, len(arr)
    while low < high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid
        else:
            high = mid
    return -1

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
```

List every bug with explanation and fix."""},
        ],
        "max_tokens": 1500,
    },
    {
        "name": "Refactoring",
        "messages": [
            {"role": "system", "content": "You are a software engineer specializing in clean code."},
            {"role": "user", "content": """Refactor this code to be more maintainable. Explain your changes.

```python
def process_orders(orders):
    total = 0
    discounted = 0
    errors = []
    for order in orders:
        if order['status'] == 'active':
            if order['type'] == 'standard':
                price = order['quantity'] * order['unit_price']
                if order['quantity'] > 100:
                    price = price * 0.9
                elif order['quantity'] > 50:
                    price = price * 0.95
                total += price
            elif order['type'] == 'premium':
                price = order['quantity'] * order['unit_price'] * 1.2
                if order['quantity'] > 100:
                    price = price * 0.85
                elif order['quantity'] > 50:
                    price = price * 0.9
                total += price
                discounted += price * 0.05
            elif order['type'] == 'wholesale':
                price = order['quantity'] * order['unit_price'] * 0.7
                total += price
            else:
                errors.append(f"Unknown type: {order['type']}")
        elif order['status'] == 'cancelled':
            pass
        else:
            errors.append(f"Unknown status: {order['status']}")
    return {'total': total, 'discounted': discounted, 'errors': errors}
```"""},
        ],
        "max_tokens": 2000,
    },
]

RESEARCH_TASKS = [
    {
        "name": "Company Analysis",
        "messages": [
            {"role": "system", "content": "You are a senior financial analyst at a top-tier investment bank."},
            {"role": "user", "content": (
                "Provide a comprehensive analysis of NVIDIA's competitive positioning in the AI "
                "accelerator market as of 2025. Cover:\n"
                "1. Revenue breakdown by segment (data center, gaming, automotive, pro viz)\n"
                "2. Competitive moat (CUDA ecosystem, InfiniBand networking, software stack)\n"
                "3. Key risks (AMD MI300X, custom ASICs from Google/Amazon/Microsoft, China restrictions)\n"
                "4. Valuation framework and growth trajectory\n"
                "Include specific financial metrics and market share data."
            )},
        ],
        "max_tokens": 2000,
    },
    {
        "name": "Sector Comparison",
        "messages": [
            {"role": "system", "content": "You are a semiconductor industry analyst."},
            {"role": "user", "content": (
                "Compare AMD and Intel's strategic positioning in server CPUs and AI accelerators:\n"
                "1. EPYC vs Xeon market share trends\n"
                "2. MI300X vs Gaudi vs NVIDIA competitive positioning\n"
                "3. Intel foundry strategy viability vs TSMC dependency\n"
                "4. Financial health (margins, R&D, debt)\n"
                "5. Investment thesis for each\n"
                "Be specific with numbers."
            )},
        ],
        "max_tokens": 2000,
    },
    {
        "name": "Risk Assessment",
        "messages": [
            {"role": "system", "content": "You are a geopolitical risk analyst for technology supply chains."},
            {"role": "user", "content": (
                "Assess geopolitical risks to the global semiconductor supply chain:\n"
                "1. Taiwan Strait scenario — impact on TSMC and global chip supply\n"
                "2. US-China tech decoupling — effects on equipment makers\n"
                "3. Sovereign AI initiatives and chip demand allocation\n"
                "4. Advanced packaging bottlenecks and geographic concentration\n"
                "5. Mitigation strategies for institutional investors\n"
                "Provide probability-weighted scenarios."
            )},
        ],
        "max_tokens": 2000,
    },
]


# ── API Helpers ──────────────────────────────────────────────────────

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


def chat(messages, max_tokens=2000, temperature=0.1):
    t0 = time.time()
    resp = api("POST", "/v1/chat/completions", {
        "model": "test",
        "id_slot": SLOT,
        "cache_prompt": True,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    })
    wall_time = time.time() - t0
    resp["_wall_time"] = wall_time
    return resp


def compact(target_tokens, method="select"):
    return api("POST", "/compact", {
        "id_slot": SLOT,
        "seq_id": 0,
        "target_tokens": target_tokens,
        "method": method,
    })


def strip_thinking(text):
    """Remove <think>...</think> blocks. Return (visible, thinking)."""
    thinking_parts = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    thinking = "\n".join(thinking_parts)
    visible = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return visible, thinking


def extract_metrics(resp, model_type="instruct"):
    """Extract standardized metrics from a chat response."""
    if "error" in resp:
        return {"status": "ERROR", "error": str(resp["error"])}

    usage = resp.get("usage", {})
    content = resp["choices"][0]["message"]["content"]
    wall_time = resp.get("_wall_time", 0)

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    gen_tok_s = completion_tokens / wall_time if wall_time > 0 else 0

    metrics = {
        "status": "OK",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_time_s": round(wall_time, 2),
        "gen_tok_s": round(gen_tok_s, 1),
        "content_preview": content[:200],
    }

    if model_type == "thinking":
        visible, thinking = strip_thinking(content)
        total_chars = len(content)
        think_chars = len(thinking)
        metrics["thinking_pct"] = round(think_chars / max(total_chars, 1) * 100, 1)
        metrics["visible_chars"] = len(visible)
        metrics["visible_preview"] = visible[:200]
        # Estimate visible tok/s (rough: 1 token ~ 4 chars)
        est_visible_tokens = max(len(visible) // 4, 1)
        metrics["est_visible_tok_s"] = round(est_visible_tokens / wall_time, 1) if wall_time > 0 else 0

    return metrics


# ── Server Management ────────────────────────────────────────────────
server_proc = None


def kill_server():
    global server_proc
    subprocess.run(["pkill", "-f", f"llama-server"], capture_output=True)
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


def start_server(model_file, ctx_override=None):
    global server_proc
    kill_server()

    model_path = os.path.join(MODEL_DIR, model_file)
    if not os.path.exists(model_path):
        return False

    ctx = ctx_override or CTX
    cmd = [
        SERVER_BIN,
        "-m", model_path,
        "--port", str(PORT),
        "-c", str(ctx),
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

    for _ in range(120):
        time.sleep(1)
        if health():
            return True
    return False


# ── Test Runners ─────────────────────────────────────────────────────

def run_speed_tests(model_type):
    """Run speed benchmarks with identical prompts."""
    results = {}

    # Warmup (first request is always slow — model loading)
    out("    Warmup...", end="")
    warmup = chat([{"role": "user", "content": "Hello"}], max_tokens=5)
    if "error" not in warmup:
        warmup_metrics = extract_metrics(warmup, model_type)
        out(f" {warmup_metrics.get('gen_tok_s', '?')} tok/s (cold)")
        results["warmup"] = warmup_metrics
    else:
        out(f" ERROR: {warmup.get('error', '?')}")
        results["warmup"] = {"status": "ERROR"}

    for test in SPEED_TESTS:
        out(f"    {test['name']}...", end="")
        resp = chat(test["messages"], max_tokens=test["max_tokens"])
        m = extract_metrics(resp, model_type)
        results[test["name"]] = m

        tok_s = m.get("gen_tok_s", "?")
        toks = m.get("completion_tokens", "?")
        wall = m.get("wall_time_s", "?")
        extra = ""
        if model_type == "thinking" and "thinking_pct" in m:
            extra = f" | think={m['thinking_pct']}%"
        out(f" {tok_s} tok/s | {toks} tokens | {wall}s{extra}")

    return results


def run_task_suite(tasks, suite_name, model_type):
    """Run a named suite of tasks."""
    results = {}
    out(f"    {suite_name}:")

    for task in tasks:
        out(f"      {task['name']}...", end="")
        # Thinking model gets more tokens to allow for thinking overhead
        max_tok = task["max_tokens"]
        if model_type == "thinking":
            max_tok = min(max_tok * 3, 6000)

        resp = chat(task["messages"], max_tokens=max_tok)
        m = extract_metrics(resp, model_type)
        results[task["name"]] = m

        tok_s = m.get("gen_tok_s", "?")
        toks = m.get("completion_tokens", "?")
        wall = m.get("wall_time_s", "?")
        extra = ""
        if model_type == "thinking" and "thinking_pct" in m:
            extra = f" | think={m['thinking_pct']}% | visible={m.get('visible_chars', '?')} chars"
        out(f" {tok_s} tok/s | {toks} tokens | {wall}s{extra}")

    return results


def run_compaction_quality(model_type):
    """Test compaction quality with fact recall at 2x and 4x."""
    results = {}
    out("    Compaction Quality:")

    # Build system prompt with facts
    facts_lines = ["You are a financial analyst. MEMORIZE these exact facts:"]
    for i, (name, value) in enumerate(FACTS, 1):
        facts_lines.append(f"  FACT {i}: {name} = {value}")
    facts_lines.append("")
    facts_lines.append("You MUST recall these exact numbers when asked.")
    system_prompt = "\n".join(facts_lines)

    # ── Fill context with multi-turn conversation ────────────────────
    out("      Fill context...", end="")
    msgs = [{"role": "system", "content": system_prompt}]

    filler1 = (
        "Provide an exhaustive analysis of the semiconductor industry covering: "
        "NVIDIA Blackwell architecture, CUDA ecosystem moat, InfiniBand networking, "
        "data center margins. AMD MI300X, EPYC gains, Xilinx synergies. "
        "Intel foundry strategy, 18A timeline, Gaudi challenges. "
        "TSMC CoWoS expansion, N2 timeline, geopolitical risks. "
        "SK Hynix HBM3E technology lead. Be extremely detailed."
    )
    msgs.append({"role": "user", "content": filler1})
    fill_max = 2000 if model_type != "thinking" else 4000
    resp1 = chat(msgs, max_tokens=fill_max)
    if "error" in resp1:
        out(f" FILL ERROR: {resp1.get('error')}")
        return {"error": str(resp1.get("error"))}

    content1 = resp1["choices"][0]["message"]["content"]
    msgs.append({"role": "assistant", "content": content1})

    filler2 = (
        "Now analyze: memory market (SK Hynix vs Samsung HBM), sovereign AI investments "
        "(UAE, Saudi, India, Japan, EU — specific figures), training-to-inference transition, "
        "advanced packaging bottlenecks. Also cover Broadcom VMware synergies and "
        "Apple's capital return program trajectory. Be very detailed."
    )
    msgs.append({"role": "user", "content": filler2})
    resp2 = chat(msgs, max_tokens=fill_max)
    if "error" in resp2:
        out(f" FILL ERROR")
        return {"error": str(resp2.get("error"))}

    content2 = resp2["choices"][0]["message"]["content"]
    msgs.append({"role": "assistant", "content": content2})

    total_tokens = resp2.get("usage", {}).get("total_tokens", 0)
    out(f" {total_tokens} tokens filled")
    results["fill_tokens"] = total_tokens

    context_msgs = list(msgs)

    # ── Recall test function ─────────────────────────────────────────
    def test_recall():
        recall_prompt = "Answer each question with ONLY the specific number. One answer per line.\n\n"
        for i, (name, _) in enumerate(FACTS, 1):
            recall_prompt += f"{i}. What is {name}? Give ONLY the number.\n"

        recall_msgs = context_msgs + [{"role": "user", "content": recall_prompt}]
        resp = chat(recall_msgs, max_tokens=500)
        if "error" in resp:
            return {"recalled": 0, "total": 10, "pct": 0, "error": str(resp.get("error"))}

        answer = resp["choices"][0]["message"]["content"]
        if model_type == "thinking":
            answer, _ = strip_thinking(answer)

        scores = []
        for name, value in FACTS:
            num = value.replace("$", "").replace(" billion", "").replace(" percent", "").replace(" million", "")
            found = any(
                v.lower() in answer.lower()
                for v in [num, value, value.replace("$", ""), f"${num}", f"{num}B", f"{num} billion"]
            )
            scores.append({"fact": name, "expected": value, "recalled": found})

        recalled = sum(1 for s in scores if s["recalled"])
        return {
            "recalled": recalled,
            "total": len(FACTS),
            "pct": round(recalled / len(FACTS) * 100, 1),
            "details": scores,
            "raw_answer": answer[:500],
        }

    # ── Baseline recall ──────────────────────────────────────────────
    out("      Baseline recall...", end="")
    baseline = test_recall()
    out(f" {baseline['recalled']}/{baseline['total']} ({baseline['pct']}%)")
    results["baseline"] = baseline

    # ── Compaction at 2x and 4x ──────────────────────────────────────
    for ratio in [2, 4]:
        out(f"      {ratio}x: re-prime...", end="")

        # Re-prime KV with original conversation
        reprime = chat(context_msgs, max_tokens=1)
        if "error" in reprime:
            out(f" REPRIME ERROR: {reprime.get('error')}")
            results[f"{ratio}x"] = {"error": "reprime failed"}
            continue

        reprime_tokens = reprime.get("usage", {}).get("total_tokens", total_tokens)
        target = max(int(reprime_tokens / ratio), 64)
        out(f" compact {reprime_tokens}→{target}...", end="")

        cr = compact(target)
        if not cr.get("success"):
            out(f" COMPACT FAILED: {cr}")
            results[f"{ratio}x"] = {"error": str(cr)}
            continue

        actual_ratio = cr.get("compression_ratio", 0)
        compact_ms = cr.get("compaction_time_ms", 0)
        out(f" {actual_ratio:.1f}x {compact_ms:.0f}ms...", end="")

        # Recall from compacted state
        recall = test_recall()
        lost = [s["fact"] for s in recall.get("details", []) if not s["recalled"]]
        out(f" recall {recall['recalled']}/{recall['total']} ({recall['pct']}%)")
        if lost:
            out(f"        LOST: {', '.join(lost)}")

        results[f"{ratio}x"] = {
            "recall": recall,
            "compact_ms": round(compact_ms, 1),
            "actual_ratio": round(actual_ratio, 1),
            "target_tokens": target,
            "facts_lost": lost,
        }

    return results


# ── Output Helper ────────────────────────────────────────────────────

def out(msg, end="\n"):
    sys.stdout.write(msg + end)
    sys.stdout.flush()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    n_models = len(MODELS)
    out("=" * 72)
    out(f"MASTER BENCHMARK: {n_models} Qwen3 Models — Identical Tests")
    out(f"Hardware: Apple M2 Pro 32GB | Context: 64K (32K for 35B) | KV: f16")
    out(f"Top 3: 30B MoE — speed + coding + research + compaction (2x/4x)")
    out(f"Bottom 3: dense/3.5 — speed + coding + research only")
    out(f"Timestamp: {timestamp}")
    out("=" * 72)

    all_results = {}

    for mi, model in enumerate(MODELS):
        out(f"\n{'='*72}")
        compact_tag = " [+compaction]" if model.get("compact") else " [no compaction]"
        out(f"[{mi+1}/{n_models}] MODEL: {model['name']} ({model['file']}){compact_tag}")
        out(f"{'='*72}")

        ctx = model.get("ctx", CTX)
        out(f"  Starting server (ctx={ctx//1024}K)...", end="")
        t_start = time.time()
        if not start_server(model["file"], ctx_override=ctx):
            out(" FAILED — skipping model")
            all_results[model["name"]] = {"error": "Server failed to start"}
            continue
        out(f" ready ({time.time()-t_start:.0f}s)")

        model_results = {"model": model, "started_at": time.strftime("%H:%M:%S")}

        try:
            out(f"\n  ── Speed Tests ──")
            model_results["speed"] = run_speed_tests(model["type"])

            out(f"\n  ── Coding Tasks ──")
            model_results["coding"] = run_task_suite(CODING_TASKS, "Coding", model["type"])

            out(f"\n  ── Financial Research ──")
            model_results["research"] = run_task_suite(RESEARCH_TASKS, "Research", model["type"])

            if model.get("compact"):
                out(f"\n  ── Compaction Quality ──")
                model_results["compaction"] = run_compaction_quality(model["type"])
            else:
                out(f"\n  ── Compaction: SKIPPED (not supported) ──")
                model_results["compaction"] = {"skipped": True}

        except Exception as e:
            out(f"\n  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            model_results["error"] = str(e)

        model_results["finished_at"] = time.strftime("%H:%M:%S")
        all_results[model["name"]] = model_results

        kill_server()
        out(f"\n  Server stopped. Model complete.\n")

    # ── SUMMARY ──────────────────────────────────────────────────────
    out("\n" + "=" * 72)
    out("COMPARISON SUMMARY")
    out("=" * 72)

    # Speed comparison
    out("\n── GENERATION SPEED (tok/s) ──")
    out(f"{'Test':<12} ", end="")
    for name in all_results:
        out(f"{'  ' + name:>20}", end="")
    out("")
    out("-" * 72)

    for test_name in ["warmup", "Trivial", "Short", "Medium", "Long"]:
        out(f"{test_name:<12} ", end="")
        for name, r in all_results.items():
            if isinstance(r.get("error"), str):
                out(f"{'FAIL':>20}", end="")
                continue
            speed = r.get("speed", {}).get(test_name, {})
            tok_s = speed.get("gen_tok_s", "-")
            toks = speed.get("completion_tokens", "-")
            out(f"{tok_s:>8} ({toks:>4}t)", end="")
        out("")

    # Coding comparison
    out("\n── CODING TASKS ──")
    out(f"{'Task':<14} ", end="")
    for name in all_results:
        out(f"{'  ' + name:>20}", end="")
    out("")
    out("-" * 72)

    for task_name in ["Algorithm", "Code Review", "Refactoring"]:
        out(f"{task_name:<14} ", end="")
        for name, r in all_results.items():
            if isinstance(r.get("error"), str):
                out(f"{'FAIL':>20}", end="")
                continue
            t = r.get("coding", {}).get(task_name, {})
            tok_s = t.get("gen_tok_s", "-")
            wall = t.get("wall_time_s", "-")
            out(f"{tok_s:>6} t/s {wall:>5}s", end="")
        out("")

    # Research comparison
    out("\n── FINANCIAL RESEARCH ──")
    out(f"{'Task':<18} ", end="")
    for name in all_results:
        out(f"{'  ' + name:>18}", end="")
    out("")
    out("-" * 72)

    for task_name in ["Company Analysis", "Sector Comparison", "Risk Assessment"]:
        out(f"{task_name:<18} ", end="")
        for name, r in all_results.items():
            if isinstance(r.get("error"), str):
                out(f"{'FAIL':>18}", end="")
                continue
            t = r.get("research", {}).get(task_name, {})
            tok_s = t.get("gen_tok_s", "-")
            wall = t.get("wall_time_s", "-")
            out(f"{tok_s:>5} t/s {wall:>5}s", end="")
        out("")

    # Thinking model overhead
    out("\n── THINKING MODEL OVERHEAD ──")
    thinking_data = all_results.get("Thinking-2507", {})
    if not isinstance(thinking_data.get("error"), str):
        for suite_name, suite_key in [("Speed", "speed"), ("Coding", "coding"), ("Research", "research")]:
            suite = thinking_data.get(suite_key, {})
            for task_name, task_data in suite.items():
                if isinstance(task_data, dict) and "thinking_pct" in task_data:
                    out(f"  {suite_name}/{task_name}: {task_data['thinking_pct']}% thinking | "
                        f"~{task_data.get('est_visible_tok_s', '?')} visible tok/s")

    # Compaction comparison
    out("\n── COMPACTION QUALITY (select method, 64K context) ──")
    out(f"{'Metric':<20} ", end="")
    for name in all_results:
        out(f"{'  ' + name:>18}", end="")
    out("")
    out("-" * 72)

    out(f"{'Fill tokens':<20} ", end="")
    for name, r in all_results.items():
        comp = r.get("compaction", {})
        fill = comp.get("fill_tokens", "-")
        out(f"{fill:>18}", end="")
    out("")

    out(f"{'Baseline recall':<20} ", end="")
    for name, r in all_results.items():
        comp = r.get("compaction", {})
        b = comp.get("baseline", {})
        out(f"{b.get('recalled', '-')}/10 ({b.get('pct', '-')}%){'':<5}", end="")
    out("")

    for ratio in ["2x", "4x"]:
        out(f"{ratio + ' recall':<20} ", end="")
        for name, r in all_results.items():
            comp = r.get("compaction", {})
            rx = comp.get(ratio, {})
            if "error" in rx:
                out(f"{'ERROR':>18}", end="")
            else:
                recall = rx.get("recall", {})
                ms = rx.get("compact_ms", "-")
                out(f"{recall.get('recalled', '-')}/10 {ms}ms{'':<3}", end="")
        out("")

        # Show lost facts for this ratio
        for name, r in all_results.items():
            comp = r.get("compaction", {})
            rx = comp.get(ratio, {})
            lost = rx.get("facts_lost", [])
            if lost:
                out(f"  {name} LOST: {', '.join(lost)}")

    # Save results
    outdir = os.path.join(os.path.dirname(__file__), "..", "bench-results")
    outpath = os.path.join(outdir, f"master-3model-{timestamp}.json")
    os.makedirs(outdir, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump({
            "benchmark": "master-3model",
            "timestamp": timestamp,
            "hardware": "Apple M2 Pro 32GB",
            "context": CTX,
            "kv_type": "f16",
            "models": [m["name"] for m in MODELS],
            "compact_models": [m["name"] for m in MODELS if m.get("compact")],
            "results": all_results,
        }, f, indent=2, default=str)

    out(f"\nResults saved: {outpath}")
    out("BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
