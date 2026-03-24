#!/usr/bin/env python3
"""Phase D: Deep Research & Long Context Testing Harness

Tests one model at one context size. Designed to be called sequentially.

Usage:
    python3 scripts/run-phase-d-test.py --model-path /path/to/model.gguf \
        --model-name "Qwen3.5-35B-A3B" --context-size 32768 \
        --use-cases D-1-1,D-2-1,D-3-1,D-4-1 \
        --out-dir bench-results/phase-d-YYYYMMDD-HHMMSS
"""

import json
import subprocess
import sys
import os
import time
import signal
import argparse
from datetime import datetime, timezone
from pathlib import Path

MODELAI_DIR = Path(os.environ.get("MODELAI_DIR", "."))
SERVER_BIN = MODELAI_DIR / "build" / "bin" / "llama-server"
SERVER_PORT = 8080
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

# Timeouts (seconds) — set by main() based on context size and CLI overrides
COMPACTION_TIMEOUT = 60
COMPLETION_TIMEOUT = 300
STARTUP_TIMEOUT = 120


def compute_timeouts(context_size, compaction_override=0, completion_override=0, startup_override=0):
    """Compute timeouts that scale with context size. Override with non-zero values."""
    global COMPACTION_TIMEOUT, COMPLETION_TIMEOUT, STARTUP_TIMEOUT
    scale = max(1.0, context_size / 32768)
    COMPACTION_TIMEOUT = compaction_override if compaction_override > 0 else int(60 * scale)
    COMPLETION_TIMEOUT = completion_override if completion_override > 0 else int(300 * scale)
    STARTUP_TIMEOUT = startup_override if startup_override > 0 else int(120 * scale)


def get_process_memory(pid):
    """Get RSS and virtual memory for a process (macOS). Returns dict with MB values."""
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=,vsz=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split()
            rss_kb, vsz_kb = int(parts[0]), int(parts[1])
            return {"rss_mb": round(rss_kb / 1024, 1), "vsz_mb": round(vsz_kb / 1024, 1)}
    except Exception:
        pass
    return {"rss_mb": 0, "vsz_mb": 0}


def get_system_memory():
    """Get system memory pressure (macOS). Returns dict with GB values."""
    try:
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            stats = {}
            for line in lines[1:]:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().rstrip(".")
                    try:
                        stats[key] = int(val)
                    except ValueError:
                        pass
            page_size = 16384  # Apple Silicon default
            free_pages = stats.get("Pages free", 0)
            active_pages = stats.get("Pages active", 0)
            inactive_pages = stats.get("Pages inactive", 0)
            wired_pages = stats.get("Pages wired down", 0)
            compressed_pages = stats.get("Pages occupied by compressor", 0)
            return {
                "free_gb": round(free_pages * page_size / (1024**3), 2),
                "active_gb": round(active_pages * page_size / (1024**3), 2),
                "inactive_gb": round(inactive_pages * page_size / (1024**3), 2),
                "wired_gb": round(wired_pages * page_size / (1024**3), 2),
                "compressed_gb": round(compressed_pages * page_size / (1024**3), 2),
            }
    except Exception:
        pass
    return {}


def get_memory_snapshot(proc):
    """Combined process + system memory snapshot."""
    snapshot = {"timestamp": datetime.now(timezone.utc).isoformat()}
    if proc and proc.poll() is None:
        snapshot["process"] = get_process_memory(proc.pid)
    snapshot["system"] = get_system_memory()
    # Also grab allocations reported by /props
    try:
        import urllib.request
        resp = urllib.request.urlopen(
            urllib.request.Request(f"{SERVER_URL}/props"), timeout=5
        )
        props = json.loads(resp.read())
        modelai = props.get("modelai", {})
        mem_info = modelai.get("runtime", {}).get("memory", {})
        kv_info = modelai.get("runtime", {}).get("kv", {})
        snapshot["server_reported"] = {
            "model_mb": round(mem_info.get("allocated_model_bytes", 0) / (1024**2), 1),
            "context_mb": round(mem_info.get("allocated_context_bytes", 0) / (1024**2), 1),
            "compute_mb": round(mem_info.get("allocated_compute_bytes", 0) / (1024**2), 1),
            "kv_active_total": kv_info.get("active_n_kv_total", 0),
            "kv_active_max": kv_info.get("active_n_kv_max", 0),
        }
    except Exception:
        pass
    return snapshot


# Load use cases
def load_use_cases(out_dir):
    with open(out_dir / "use-cases.json") as f:
        data = json.load(f)
    return {uc["id"]: uc for uc in data["use_cases"]}


def start_server(model_path, context_size):
    """Start llama-server and wait for it to be ready. Returns (proc, startup_info)."""
    print(f"  Starting llama-server (ctx={context_size})...", flush=True)
    proc = subprocess.Popen(
        [str(SERVER_BIN), "-m", str(model_path), "-c", str(context_size),
         "-ngl", "99", "-np", "1", "--port", str(SERVER_PORT), "--host", "127.0.0.1"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )

    # Wait for server to be ready (scales with context size)
    for i in range(STARTUP_TIMEOUT):
        time.sleep(1)
        try:
            import urllib.request
            req = urllib.request.Request(f"{SERVER_URL}/health")
            resp = urllib.request.urlopen(req, timeout=2)
            if resp.status == 200:
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    print(f"  Server ready after {i+1}s", flush=True)
                    mem = get_memory_snapshot(proc)
                    rss = mem.get("process", {}).get("rss_mb", 0)
                    model_mb = mem.get("server_reported", {}).get("model_mb", 0)
                    ctx_mb = mem.get("server_reported", {}).get("context_mb", 0)
                    print(f"  Memory at startup: RSS={rss}MB, model={model_mb}MB, ctx={ctx_mb}MB", flush=True)
                    # Check compaction support
                    compaction_available = False
                    try:
                        presp = urllib.request.urlopen(urllib.request.Request(f"{SERVER_URL}/props"), timeout=5)
                        props = json.loads(presp.read())
                        cap = props.get("modelai", {}).get("capabilities", {}).get("compacted_prefix", {})
                        compaction_available = cap.get("available", False)
                        fallback = cap.get("last_fallback_reason", "none")
                        print(f"  Compaction: available={compaction_available}, fallback_reason={fallback}", flush=True)
                    except Exception:
                        pass
                    startup_info = {
                        "startup_time_s": i + 1,
                        "memory_at_startup": mem,
                        "compaction_available": compaction_available,
                    }
                    return proc, startup_info
        except Exception:
            pass
        if proc.poll() is not None:
            print(f"  Server exited with code {proc.returncode}", flush=True)
            return None, {}

    print(f"  Server failed to start within {STARTUP_TIMEOUT}s", flush=True)
    kill_server(proc)
    return None, {}


def kill_server(proc):
    """Kill llama-server process group."""
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    # Extra cleanup
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(3)


def completion(prompt, max_tokens=2048, temperature=0.7):
    """Send a completion request and measure timing."""
    import urllib.request
    import urllib.error
    payload = json.dumps({
        "prompt": prompt,
        "id_slot": 0,
        "n_predict": max_tokens,
        "temperature": temperature,
        "stop": ["<|im_end|>", "<|endoftext|>", "</s>"],
        "cache_prompt": True,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER_URL}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=COMPLETION_TIMEOUT)
        t_total = time.time() - t0
        data = json.loads(resp.read())

        content = data.get("content", "")
        tokens_predicted = data.get("tokens_predicted", 0)
        tokens_evaluated = data.get("tokens_evaluated", 0)
        timings = data.get("timings", {})

        return {
            "content": content,
            "tokens_predicted": tokens_predicted,
            "tokens_evaluated": tokens_evaluated,
            "total_time_s": round(t_total, 3),
            "prompt_eval_time_ms": timings.get("prompt_ms", 0),
            "generation_time_ms": timings.get("predicted_ms", 0),
            "prompt_tok_s": round(timings.get("prompt_per_second", 0), 2),
            "generation_tok_s": round(timings.get("predicted_per_second", 0), 2),
            "time_to_first_token_ms": round(timings.get("prompt_ms", 0), 1),
        }
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()
        except Exception:
            pass
        return {"error": f"HTTP {e.code}: {body or e.reason}", "total_time_s": round(time.time() - t0, 3)}
    except Exception as e:
        return {"error": str(e), "total_time_s": round(time.time() - t0, 3)}


def get_server_props():
    """Get server props including KV cache utilization."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{SERVER_URL}/props")
        resp = urllib.request.urlopen(req, timeout=5)
        return json.loads(resp.read())
    except Exception:
        return {}


def get_server_metrics():
    """Get Prometheus metrics from server."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{SERVER_URL}/metrics")
        resp = urllib.request.urlopen(req, timeout=5)
        text = resp.read().decode()
        metrics = {}
        for line in text.split("\n"):
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    metrics[parts[0]] = float(parts[1])
        return metrics
    except Exception:
        return {}


def trigger_compaction(ratio=2.0, method="select"):
    """Trigger KV cache compaction."""
    import urllib.request
    import urllib.error
    payload = json.dumps({
        "id_slot": 0,
        "method": method,
        "ratio": ratio,
        "reclaim": True,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER_URL}/compact",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=COMPACTION_TIMEOUT)
        t_total = time.time() - t0
        data = json.loads(resp.read())
        data["compaction_time_ms"] = round(t_total * 1000, 1)
        return data
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()
        except Exception:
            pass
        return {"error": f"HTTP {e.code}: {body or e.reason}", "compaction_time_ms": round((time.time() - t0) * 1000, 1)}
    except Exception as e:
        return {"error": str(e), "compaction_time_ms": round((time.time() - t0) * 1000, 1)}


def score_response_quality(response_text, use_case):
    """Score response quality on 1-5 scale based on content."""
    if not response_text or len(response_text) < 50:
        return 1, "Empty or too short"

    score = 3  # baseline
    notes = []

    word_count = len(response_text.split())
    min_words = use_case.get("expected_depth", {}).get("min_words", 500)

    # Length check
    if word_count >= min_words:
        score += 1
        notes.append(f"Good length ({word_count} words)")
    elif word_count < min_words * 0.3:
        score -= 1
        notes.append(f"Too short ({word_count}/{min_words} words)")

    # Structure check (headers, numbered lists, etc.)
    has_structure = any(c in response_text for c in ["1.", "2.", "- ", "**", "##"])
    if has_structure:
        score = min(score + 1, 5)
        notes.append("Well-structured")

    # Check for hedging/refusal
    refusal_phrases = ["I cannot", "I don't have access", "as an AI", "I'm unable to"]
    if any(p.lower() in response_text.lower() for p in refusal_phrases):
        score = max(score - 1, 1)
        notes.append("Contains refusal/hedging language")

    return min(max(score, 1), 5), "; ".join(notes)


def run_baseline_test(use_case_ids, all_use_cases, model_name, context_size, proc=None):
    """Run baseline test: fill context progressively, test recall."""
    results = []
    conversation = ""
    memory_timeline = []

    # Capture memory before first request
    if proc:
        memory_timeline.append({"event": "pre_baseline", **get_memory_snapshot(proc)})

    # Build conversation using assigned use cases
    for i, uc_id in enumerate(use_case_ids):
        uc = all_use_cases.get(uc_id)
        if not uc:
            continue

        # Format as chat
        turn_prompt = f"<|im_start|>user\n{uc['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        full_prompt = conversation + turn_prompt

        print(f"    [{uc_id}] Sending research prompt ({len(full_prompt)} chars)...", flush=True)
        resp = completion(full_prompt, max_tokens=1024)

        if "error" in resp:
            print(f"    [{uc_id}] ERROR: {resp['error']}", flush=True)
            results.append({
                "use_case_id": uc_id,
                "error": resp["error"],
            })
            continue

        content = resp.get("content", "")
        quality_score, quality_notes = score_response_quality(content, uc)

        # Memory snapshot after each completion
        mem_snap = {}
        if proc:
            mem_snap = get_memory_snapshot(proc)
            rss = mem_snap.get("process", {}).get("rss_mb", 0)
            kv_active = mem_snap.get("server_reported", {}).get("kv_active_total", 0)
            memory_timeline.append({"event": f"after_{uc_id}", **mem_snap})
        else:
            rss = 0
            kv_active = 0

        result = {
            "use_case_id": uc_id,
            "category": uc["category"],
            "title": uc["title"],
            "tokens_evaluated": resp.get("tokens_evaluated", 0),
            "tokens_predicted": resp.get("tokens_predicted", 0),
            "prompt_tok_s": resp.get("prompt_tok_s", 0),
            "generation_tok_s": resp.get("generation_tok_s", 0),
            "time_to_first_token_ms": resp.get("time_to_first_token_ms", 0),
            "total_time_s": resp.get("total_time_s", 0),
            "response_word_count": len(content.split()),
            "quality_score": quality_score,
            "quality_notes": quality_notes,
            "memory_rss_mb": mem_snap.get("process", {}).get("rss_mb", 0),
            "memory_kv_active": mem_snap.get("server_reported", {}).get("kv_active_total", 0),
        }
        results.append(result)

        # Append to conversation for progressive context fill
        conversation += turn_prompt + content + "<|im_end|>\n"

        print(f"    [{uc_id}] {resp.get('generation_tok_s', 0)} tok/s, "
              f"quality={quality_score}/5, {len(content.split())} words, "
              f"RSS={rss}MB, KV={kv_active}", flush=True)

    # Recall test
    recall_prompt = conversation + "<|im_start|>user\nSummarize the key findings from our entire conversation so far. Be specific about facts, figures, and conclusions from each topic we discussed.<|im_end|>\n<|im_start|>assistant\n"

    print(f"    [RECALL] Testing recall ({len(recall_prompt)} chars context)...", flush=True)
    recall_resp = completion(recall_prompt, max_tokens=1024)

    recall_content = recall_resp.get("content", "")
    recall_score = 3
    if recall_content:
        # Check how many use case topics are referenced in recall
        topics_recalled = sum(1 for uc_id in use_case_ids
                            if any(word in recall_content.lower()
                                  for word in all_use_cases.get(uc_id, {}).get("title", "").lower().split()[:3]))
        recall_score = min(1 + topics_recalled, 5)

    # Final memory snapshot after recall
    if proc:
        final_mem = get_memory_snapshot(proc)
        memory_timeline.append({"event": "post_recall", **final_mem})
        peak_rss = max(m.get("process", {}).get("rss_mb", 0) for m in memory_timeline)
        print(f"    [MEMORY] Peak RSS: {peak_rss}MB", flush=True)
    else:
        peak_rss = 0

    return {
        "test_type": "baseline",
        "use_case_results": results,
        "recall": {
            "tokens_evaluated": recall_resp.get("tokens_evaluated", 0),
            "generation_tok_s": recall_resp.get("generation_tok_s", 0),
            "time_to_first_token_ms": recall_resp.get("time_to_first_token_ms", 0),
            "total_time_s": recall_resp.get("total_time_s", 0),
            "recall_quality_score": recall_score,
            "response_word_count": len(recall_content.split()),
        },
        "memory": {
            "peak_rss_mb": peak_rss,
            "timeline": memory_timeline,
        },
        "final_context_chars": len(recall_prompt),
        "conversation_for_compaction": recall_prompt,
        "recall_response": recall_content,
    }


def check_health():
    """Check server health. Returns True if healthy, False otherwise."""
    import urllib.request
    try:
        resp = urllib.request.urlopen(
            urllib.request.Request(f"{SERVER_URL}/health"), timeout=10
        )
        data = json.loads(resp.read())
        return data.get("status") == "ok"
    except Exception:
        return False


def log_error(model_dir, endpoint, status, body, use_case=""):
    """Append structured error to errors.jsonl in the model's output directory."""
    if model_dir is None:
        return
    error_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "status": status,
        "body": body,
        "use_case": use_case,
    }
    errors_path = Path(model_dir) / "errors.jsonl"
    try:
        with open(errors_path, "a") as f:
            f.write(json.dumps(error_entry) + "\n")
    except Exception:
        pass


def run_compaction_test(conversation_prompt, recall_response_baseline, ratio=2.0, proc=None):
    """Run compaction test on existing conversation."""
    # Memory delta: capture RSS before compaction (F5 + F6: note that RSS is
    # unreliable on Apple Silicon unified memory — /props KV stats are the
    # primary signal for compaction effectiveness)
    mem_before = get_memory_snapshot(proc) if proc else {}

    print(f"    [COMPACT] Triggering {ratio}x compaction...", flush=True)
    compact_result = trigger_compaction(ratio=ratio)

    if "error" in compact_result:
        print(f"    [COMPACT] ERROR: {compact_result['error']}", flush=True)
        return {"test_type": f"compacted_{ratio}x", "error": compact_result["error"]}

    print(f"    [COMPACT] Done in {compact_result.get('compaction_time_ms', 0)}ms", flush=True)

    # Health check after compaction (Fix 4) — catches delayed crashes
    if not check_health():
        print(f"    [HEALTH] Server unresponsive after compaction!", flush=True)
        compact_result["error"] = "Server unresponsive after compaction"
        return {"test_type": f"compacted_{ratio}x", "error": compact_result["error"],
                "compaction_time_ms": compact_result.get("compaction_time_ms", 0)}

    # Memory delta after compaction
    mem_after = get_memory_snapshot(proc) if proc else {}
    rss_before = mem_before.get("process", {}).get("rss_mb", 0)
    rss_after = mem_after.get("process", {}).get("rss_mb", 0)
    kv_before = mem_before.get("server_reported", {}).get("kv_active_total", 0)
    kv_after = mem_after.get("server_reported", {}).get("kv_active_total", 0)
    compact_result["memory_delta"] = {
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_freed_mb": round(rss_before - rss_after, 1),
        "kv_before": kv_before,
        "kv_after": kv_after,
    }
    print(f"    [MEMORY] RSS: {rss_before}→{rss_after}MB, KV: {kv_before}→{kv_after}", flush=True)

    # Idempotency check (Fix 6) — second compact should be near-instant or no-op
    # Expected: engine either returns quickly with same token count, or returns
    # an error indicating nothing to compact. Behavior is engine-defined.
    idempotency_result = trigger_compaction(ratio=ratio)
    compact_result["idempotency"] = {
        "second_call_time_ms": idempotency_result.get("compaction_time_ms", 0),
        "second_call_error": idempotency_result.get("error"),
        "is_noop": idempotency_result.get("tokens_before") == idempotency_result.get("tokens_after"),
    }
    print(f"    [IDEMPOTENCY] 2nd compact: {idempotency_result.get('compaction_time_ms', 0)}ms, "
          f"error={idempotency_result.get('error', 'none')}", flush=True)

    # Re-ask recall question — send full conversation + recall question so the
    # model has context to recall from.  If KV cache retained the compacted
    # prefix, cache_prompt=true will reuse it and only evaluate the new tokens.
    # If not, the server re-evaluates from scratch (which is Issue 1 in the
    # engine handoff prompt).
    recall_question = "<|im_start|>user\nSummarize the key findings from our entire conversation so far. Be specific about facts, figures, and conclusions from each topic we discussed.<|im_end|>\n<|im_start|>assistant\n"
    recall_prompt_full = conversation_prompt + recall_question

    print(f"    [RECALL-POST] Testing post-compaction recall ({len(recall_prompt_full)} chars)...", flush=True)
    recall_resp = completion(recall_prompt_full, max_tokens=1024)

    recall_content = recall_resp.get("content", "")

    return {
        "test_type": f"compacted_{ratio}x",
        "compaction_ratio": ratio,
        "compaction_time_ms": compact_result.get("compaction_time_ms", 0),
        "cosine_similarity": compact_result.get("cosine_similarity"),
        "tokens_before": compact_result.get("tokens_before"),
        "tokens_after": compact_result.get("tokens_after"),
        "memory_delta": compact_result.get("memory_delta", {}),
        "idempotency": compact_result.get("idempotency", {}),
        "recall": {
            "generation_tok_s": recall_resp.get("generation_tok_s", 0),
            "time_to_first_token_ms": recall_resp.get("time_to_first_token_ms", 0),
            "total_time_s": recall_resp.get("total_time_s", 0),
            "recall_quality_score": min(max(len(recall_content.split()) // 50, 1), 5),
            "response_word_count": len(recall_content.split()),
        },
        "recall_response": recall_content,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase D test runner")
    parser.add_argument("--model-path", help="Path to GGUF model (not needed with --server-url)")
    parser.add_argument("--model-name", required=True, help="Human-readable model name")
    parser.add_argument("--model-order", type=int, default=0, help="Model test order (1-17)")
    parser.add_argument("--context-size", type=int, required=True, help="Context size in tokens")
    parser.add_argument("--use-cases", required=True, help="Comma-separated use case IDs")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--compaction-ratios", default="2", help="Comma-separated compaction ratios")
    parser.add_argument("--compaction-timeout", type=int, default=0, help="Compaction timeout in seconds (0=auto-scale with context)")
    parser.add_argument("--completion-timeout", type=int, default=0, help="Completion timeout in seconds (0=auto-scale with context)")
    parser.add_argument("--startup-timeout", type=int, default=0, help="Server startup timeout in seconds (0=auto-scale with context)")
    parser.add_argument("--server-url", default="", help="Connect to existing server (skip startup/shutdown). E.g. http://127.0.0.1:8090")
    args = parser.parse_args()

    # When using an existing server, model-path is optional
    external_server = bool(args.server_url)
    if not external_server and not args.model_path:
        parser.error("--model-path is required unless --server-url is provided")

    # Point all requests at the right URL
    if external_server:
        global SERVER_URL
        SERVER_URL = args.server_url.rstrip("/")
        print(f"  Using existing server: {SERVER_URL}")

    compute_timeouts(args.context_size, args.compaction_timeout, args.completion_timeout, args.startup_timeout)
    print(f"  Timeouts: startup={STARTUP_TIMEOUT}s, completion={COMPLETION_TIMEOUT}s, compaction={COMPACTION_TIMEOUT}s")

    out_dir = Path(args.out_dir)
    model_dir = out_dir / "results" / f"{args.model_order:02d}-{args.model_name.replace(' ', '-')}"
    model_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = model_dir / "research-responses"
    responses_dir.mkdir(exist_ok=True)

    use_case_ids = [x.strip() for x in args.use_cases.split(",")]
    compaction_ratios = [float(x.strip()) for x in args.compaction_ratios.split(",")]
    all_use_cases = load_use_cases(out_dir)

    ctx_k = args.context_size // 1024
    print(f"\n{'='*70}")
    print(f"MODEL: {args.model_name} | CONTEXT: {ctx_k}K | USE CASES: {use_case_ids}")
    print(f"{'='*70}")

    # Start or connect to server
    proc = None
    startup_info = {}
    if external_server:
        # Verify external server is healthy
        import urllib.request
        try:
            resp = urllib.request.urlopen(urllib.request.Request(f"{SERVER_URL}/health"), timeout=10)
            data = json.loads(resp.read())
            if data.get("status") != "ok":
                print(f"  ERROR: Server not healthy: {data}")
                return
            print(f"  External server healthy", flush=True)
            # Get compaction support
            try:
                presp = urllib.request.urlopen(urllib.request.Request(f"{SERVER_URL}/props"), timeout=5)
                props = json.loads(presp.read())
                cap = props.get("modelai", {}).get("capabilities", {}).get("compacted_prefix", {})
                startup_info["compaction_available"] = cap.get("available", False)
                print(f"  Compaction: available={startup_info['compaction_available']}", flush=True)
            except Exception:
                startup_info["compaction_available"] = False
            startup_info["external_server"] = True
        except Exception as e:
            print(f"  ERROR: Cannot reach server at {SERVER_URL}: {e}")
            return
    else:
        proc, startup_info = start_server(args.model_path, args.context_size)
        if not proc:
            error_result = {
                "model": args.model_name,
                "context_size": args.context_size,
                "error": "Server failed to start",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(model_dir / f"{ctx_k}k-baseline.json", "w") as f:
                json.dump(error_result, f, indent=2)
            return

    compaction_available = startup_info.get("compaction_available", False)

    try:
        # Step 1: Baseline test
        print(f"\n  --- Baseline Test ({ctx_k}K) ---", flush=True)
        baseline = run_baseline_test(use_case_ids, all_use_cases, args.model_name, args.context_size, proc=proc)
        baseline["model"] = args.model_name
        baseline["context_size"] = args.context_size
        baseline["timestamp"] = datetime.now(timezone.utc).isoformat()
        baseline["startup_info"] = startup_info

        with open(model_dir / f"{ctx_k}k-baseline.json", "w") as f:
            json.dump(baseline, f, indent=2, default=str)

        if baseline.get("recall_response"):
            with open(responses_dir / f"{ctx_k}k-recall-baseline.md", "w") as f:
                f.write(f"# Recall Test — {args.model_name} @ {ctx_k}K (Baseline)\n\n")
                f.write(baseline["recall_response"])

        # Step 2: Compaction tests (only if model supports it)
        if compaction_available:
            conversation = baseline.get("conversation_for_compaction", "")
            for ratio in compaction_ratios:
                print(f"\n  --- Compaction Test ({ratio}x @ {ctx_k}K) ---", flush=True)
                compact_result = run_compaction_test(conversation, baseline.get("recall_response", ""), ratio, proc=proc)
                compact_result["model"] = args.model_name
                compact_result["context_size"] = args.context_size
                compact_result["timestamp"] = datetime.now(timezone.utc).isoformat()

                ratio_str = str(ratio).replace(".", "p")
                with open(model_dir / f"{ctx_k}k-compacted-{ratio_str}x.json", "w") as f:
                    json.dump(compact_result, f, indent=2, default=str)

                if compact_result.get("recall_response"):
                    with open(responses_dir / f"{ctx_k}k-recall-compacted-{ratio_str}x.md", "w") as f:
                        f.write(f"# Recall Test — {args.model_name} @ {ctx_k}K (Compacted {ratio}x)\n\n")
                        f.write(compact_result["recall_response"])
        else:
            print(f"\n  --- Compaction SKIPPED (model reports: not supported) ---", flush=True)
            skip_result = {
                "model": args.model_name,
                "context_size": args.context_size,
                "test_type": "compaction_skipped",
                "reason": startup_info.get("memory_at_startup", {}).get("server_reported", {}).get("last_fallback_reason", "model_unsupported"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(model_dir / f"{ctx_k}k-compaction-skipped.json", "w") as f:
                json.dump(skip_result, f, indent=2)

        # Final memory snapshot + server metrics
        final_mem = get_memory_snapshot(proc)
        with open(model_dir / f"{ctx_k}k-memory-final.json", "w") as f:
            json.dump(final_mem, f, indent=2)

        metrics = get_server_metrics()
        if metrics:
            with open(model_dir / f"{ctx_k}k-metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

    finally:
        if not external_server:
            kill_server(proc)

    print(f"\n  Results saved to {model_dir}/")
    print(f"  DONE: {args.model_name} @ {ctx_k}K\n")


if __name__ == "__main__":
    main()
