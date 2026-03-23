#!/usr/bin/env python3
"""End-to-end multi-cycle KV compaction test.

Simulates real ModelAI usage: fill KV window → compact → fill again → compact,
up to the model's compact_cap. Validates recall quality after each cycle.

Usage:
    python3 scripts/run-e2e-compaction-test.py \
        --model-path ~/dev/whippet/models/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf \
        --model-name "Qwen3-Coder-30B-A3B" \
        --kv-alloc 65536 --compact-cap 262144 --compact-ratio 4 \
        --out-dir bench-results/e2e-TIMESTAMP
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

MODELAI_DIR = Path("/Users/ajayjandhyala/dev/whippet/modelai-llama.cpp")
SERVER_BIN = MODELAI_DIR / "build-release" / "bin" / "llama-server"
if not SERVER_BIN.exists():
    SERVER_BIN = MODELAI_DIR / "build" / "bin" / "llama-server"
SERVER_PORT = 8080
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

# Load prompts from the Phase D use-cases.json if available, otherwise use built-in set.
# The script will cycle through all prompts across compaction cycles.
def load_research_prompts(out_dir):
    """Load prompts from use-cases.json or fall back to built-in prompts."""
    uc_path = out_dir / "use-cases.json"
    if uc_path.exists():
        with open(uc_path) as f:
            data = json.load(f)
        prompts = []
        for uc in data["use_cases"]:
            prompts.append({
                "id": uc["id"],
                "topic": uc["category"],
                "prompt": uc["prompt"],
            })
        return prompts

    # Fallback: built-in prompts covering key financial domains
    return [
        {"id": "E2E-MA-1", "topic": "M&A Activity", "prompt": "Provide an extremely detailed analysis of major technology M&A activity in the last three months. Cover every significant deal: acquirer, target, deal value, strategic rationale, premium paid, regulatory status, expected synergies, and market reaction. Include specific stock price movements, analyst commentary, and compare deal multiples across transactions. Be thorough — cover at least 8-10 deals with full details."},
        {"id": "E2E-BB-1", "topic": "Buybacks", "prompt": "Provide an extremely detailed analysis of recent stock buyback announcements in the last 3 months across the S&P 500. For every major buyback program announced: company name, buyback authorization amount, percentage of market cap, timeline, whether it's a new program or expansion, current buyback yield, total shareholder yield including dividends. Rank the top 15 programs by size and analyze which represent good capital allocation vs. potentially value-destructive repurchases. Include specific EPS accretion estimates."},
        {"id": "E2E-EARN-1", "topic": "Earnings Q4 2025", "prompt": "Provide a comprehensive analysis of Q4 2025 earnings season across all 11 S&P 500 sectors. For each sector: aggregate revenue growth, EPS beat/miss rates, margin trends, forward guidance tone. Then deep-dive into the 10 most notable individual earnings reports: revenue breakdown by segment, operating leverage, management commentary on AI investment, capital allocation plans, and stock reaction. Compare consensus revisions pre vs post earnings."},
        {"id": "E2E-MACRO-1", "topic": "Macro & Rates", "prompt": "Provide an exhaustive analysis of the current macroeconomic environment and its implications for equity markets. Cover: Federal Reserve policy trajectory with specific rate expectations, inflation data trends across CPI/PCE/PPI, labor market conditions including NFP/ADP/JOLTS details, GDP growth decomposition, yield curve analysis, credit spreads, dollar strength, and commodity prices. For each data point, provide the specific number and directional trend. Then build a framework for how these factors interact to determine sector allocation."},
        {"id": "E2E-SEMI-1", "topic": "Semiconductor Deep Dive", "prompt": "Provide the most detailed possible analysis of the semiconductor industry supply chain and competitive landscape. Cover every major player: NVIDIA (datacenter, gaming, auto), AMD (MI300X, EPYC, Ryzen), Intel (foundry strategy, Gaudi), TSMC (capacity, pricing, geopolitics), Broadcom (VMware integration, AI networking), Qualcomm (Snapdragon, auto), TI (analog moat), Marvell (custom silicon). For each: latest quarter financials, forward guidance, capex plans, competitive positioning. Then analyze AI chip demand vs supply dynamics through 2027."},
        {"id": "E2E-LEGAL-1", "topic": "Legal & Antitrust", "prompt": "Provide comprehensive analysis of all active antitrust and regulatory proceedings against major technology companies. Cover: DOJ vs Google (search + ad tech — both cases), FTC vs Amazon, FTC vs Meta, Apple App Store litigation, EU Digital Markets Act enforcement actions, EU AI Act implementation. For each: current status, key rulings, proposed remedies, timeline to resolution, probability of material impact, and quantified financial exposure. Include recent SEC enforcement actions and their precedent value."},
        {"id": "E2E-AI-1", "topic": "AI Industry Analysis", "prompt": "Provide the most comprehensive possible analysis of the AI industry as of early 2026. Cover: foundation model landscape (OpenAI, Anthropic, Google, Meta, Mistral, xAI — capabilities, pricing, market share), AI infrastructure (GPU demand, cloud AI revenue by provider), enterprise AI adoption rates and ROI data, AI startup funding and valuations, open-source AI ecosystem (llama.cpp, vLLM, Hugging Face), AI regulation across jurisdictions, and the economic impact of AI on productivity. Provide specific revenue figures, growth rates, and market sizing for each segment."},
        {"id": "E2E-PORT-1", "topic": "Portfolio Construction", "prompt": "Build a detailed quantitative portfolio construction framework for a retail investor with $500K. Cover: strategic asset allocation across US large cap, US small cap, international developed, emerging markets, fixed income (duration breakdown), alternatives, and cash. For each: specific ETF recommendations with expense ratios, historical risk/return characteristics, current valuation metrics, correlation matrix. Then construct 3 model portfolios (conservative, moderate, aggressive) with specific weights, rebalancing rules, tax-loss harvesting triggers, and expected Sharpe ratios. Include Monte Carlo simulation parameters."},
        {"id": "E2E-ENERGY-1", "topic": "Energy Transition", "prompt": "Provide exhaustive analysis of the energy sector transition. Cover: oil & gas supermajor capital allocation (ExxonMobil, Chevron, Shell, BP, TotalEnergies — capex split between fossil and renewables, dividend sustainability, reserve replacement), renewable energy growth (solar, wind, storage — installed capacity, cost curves, subsidy landscape), nuclear renaissance (SMR companies, regulatory pathway, economics), EV adoption rates and impact on gasoline demand, carbon credit markets, and hydrogen economy development. For each company and technology, provide specific financial metrics and growth projections."},
        {"id": "E2E-HEALTH-1", "topic": "Healthcare & Biotech", "prompt": "Provide comprehensive analysis of the healthcare sector. Cover: Big Pharma pipeline analysis (top 10 companies by R&D spend — key drugs in Phase 3, patent cliffs, biosimilar competition), GLP-1/obesity drug market (Novo Nordisk vs Eli Lilly — market sizing, capacity constraints, pricing dynamics, competitive moats), healthcare AI (diagnostics, drug discovery, clinical operations — specific companies and valuations), managed care landscape (UNH, CVS, CI, HUM, ELV — medical loss ratios, Stars ratings, MA enrollment trends), and biotech funding environment. Include specific revenue forecasts and valuation comparisons."},
        {"id": "E2E-FINTECH-1", "topic": "Fintech & Banking", "prompt": "Provide detailed analysis of the fintech and banking landscape. Cover: digital payments (Visa, Mastercard, PayPal, Block, Adyen — transaction volumes, take rates, cross-border trends), neobanks (Chime, Revolut, Nubank — customer acquisition costs, unit economics, profitability timelines), embedded finance (Stripe, Plaid, Marqeta — B2B adoption), traditional banks vs fintechs (JPM, BAC, WFC — technology investment, digital adoption metrics), crypto/blockchain (institutional adoption, stablecoin regulation, DeFi volumes), and AI in financial services (robo-advisory AUM, algorithmic trading, credit scoring). Specific metrics for each company."},
        {"id": "E2E-CHINA-1", "topic": "China & EM Markets", "prompt": "Provide comprehensive analysis of China and emerging market equities. Cover: China macro (GDP growth decomposition, property sector status, consumer confidence, PMI trends, policy stimulus measures), China tech (Alibaba, Tencent, PDD, ByteDance, BYD — latest financials, regulatory environment, US listing risks), China-US decoupling impact on supply chains, India growth story (Sensex valuation, sector breakdown, foreign investment flows), ASEAN opportunities (Vietnam, Indonesia, Thailand — manufacturing shift beneficiaries), and EM debt dynamics (dollar-denominated debt, central bank reserves, current account balances). Specific valuation comparisons vs developed markets."},
    ]


def completion(prompt, max_tokens=4096, temperature=0.7):
    """Send a completion request."""
    import urllib.request
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
        resp = urllib.request.urlopen(req, timeout=600)
        t_total = time.time() - t0
        data = json.loads(resp.read())
        data["total_time_s"] = round(t_total, 3)
        return data
    except Exception as e:
        return {"error": str(e), "total_time_s": round(time.time() - t0, 3)}


def trigger_compaction(ratio=4.0, method="select"):
    """Trigger KV cache compaction."""
    import urllib.request
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
        resp = urllib.request.urlopen(req, timeout=120)
        t_total = time.time() - t0
        data = json.loads(resp.read())
        data["compaction_time_ms"] = round(t_total * 1000, 1)
        return data
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return {
            "error": f"{e}",
            "error_body": body,
            "compaction_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        return {"error": str(e), "compaction_time_ms": round((time.time() - t0) * 1000, 1)}


def get_kv_stats():
    """Get current KV cache stats from /props."""
    import urllib.request
    try:
        resp = urllib.request.urlopen(
            urllib.request.Request(f"{SERVER_URL}/props"), timeout=5
        )
        props = json.loads(resp.read())
        kv = props.get("modelai", {}).get("runtime", {}).get("kv", {})
        return {
            "active_total": kv.get("active_n_kv_total", 0),
            "active_max": kv.get("active_n_kv_max", 0),
        }
    except Exception:
        return {}


def get_slots():
    """Get slot info from /slots."""
    import urllib.request
    try:
        resp = urllib.request.urlopen(
            urllib.request.Request(f"{SERVER_URL}/slots"), timeout=5
        )
        return json.loads(resp.read())
    except Exception:
        return []


def start_server(model_path, context_size):
    """Start llama-server."""
    print(f"  Starting llama-server (ctx={context_size})...", flush=True)
    proc = subprocess.Popen(
        [str(SERVER_BIN), "-m", str(model_path), "-c", str(context_size),
         "-ngl", "99", "-np", "1", "--port", str(SERVER_PORT), "--host", "127.0.0.1"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )

    for i in range(120):
        time.sleep(1)
        try:
            import urllib.request
            req = urllib.request.Request(f"{SERVER_URL}/health")
            resp = urllib.request.urlopen(req, timeout=2)
            if resp.status == 200:
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    print(f"  Server ready after {i+1}s", flush=True)
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            print(f"  Server exited with code {proc.returncode}", flush=True)
            return None
    print("  Server failed to start within 120s", flush=True)
    return None


def kill_server(proc):
    """Kill llama-server."""
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(3)


def recall_test(conversation, cycle_num, total_topics):
    """Test recall of all prior conversation topics."""
    recall_prompt = conversation + (
        "<|im_start|>user\n"
        f"We have discussed {total_topics} financial research topics across {cycle_num} conversation segments. "
        "Summarize ALL the key findings from our ENTIRE conversation so far. "
        "Be specific about facts, figures, company names, and conclusions from EVERY topic we discussed. "
        "Do not skip any topic.\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    return completion(recall_prompt, max_tokens=2048)


def main():
    parser = argparse.ArgumentParser(description="E2E multi-cycle compaction test")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--kv-alloc", type=int, required=True, help="KV window size (e.g. 65536)")
    parser.add_argument("--compact-cap", type=int, required=True, help="Max effective context (e.g. 262144)")
    parser.add_argument("--compact-ratio", type=float, default=4.0, help="Compaction ratio (default: 4)")
    parser.add_argument("--compact-method", default="select", help="Compaction method")
    parser.add_argument("--max-tokens", type=int, default=-1, help="Max tokens per response (-1 = fill until stop token)")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cycles = args.compact_cap // args.kv_alloc
    fill_target = int(args.kv_alloc * 0.90)  # Fill to 90% before compacting

    # Load prompts — prefer use-cases.json (28 prompts), fall back to built-in (12)
    research_prompts = load_research_prompts(out_dir)

    print("=" * 70)
    print(f"E2E Multi-Cycle Compaction Test")
    print(f"=" * 70)
    print(f"Model: {args.model_name}")
    print(f"KV alloc: {args.kv_alloc} | Compact cap: {args.compact_cap} | Ratio: {args.compact_ratio}x")
    print(f"Cycles: {n_cycles} | Fill target: {fill_target} tokens per cycle")
    print(f"Prompts available: {len(research_prompts)}")
    print(f"Max tokens per response: {'unlimited (stop token)' if args.max_tokens <= 0 else args.max_tokens}")
    print(f"Output: {out_dir}")
    print()

    proc = start_server(args.model_path, args.kv_alloc)
    if not proc:
        print("ERROR: Server failed to start")
        sys.exit(1)

    results = {
        "model": args.model_name,
        "kv_alloc": args.kv_alloc,
        "compact_cap": args.compact_cap,
        "compact_ratio": args.compact_ratio,
        "compact_method": args.compact_method,
        "n_cycles": n_cycles,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cycles": [],
    }

    conversation = "<|im_start|>system\nYou are a senior financial analyst. Provide extremely detailed, data-rich analysis. Always include specific numbers, company names, and actionable insights.<|im_end|>\n"
    prompt_idx = 0
    total_topics = 0
    total_effective_tokens = 0

    try:
        for cycle in range(1, n_cycles + 1):
            print(f"\n{'='*70}")
            print(f"CYCLE {cycle}/{n_cycles} — effective tokens so far: {total_effective_tokens}")
            print(f"{'='*70}")

            cycle_result = {
                "cycle": cycle,
                "prompts": [],
                "pre_compact_kv": None,
                "compaction": None,
                "post_compact_kv": None,
                "recall": None,
            }

            # Fill KV window
            cycle_start_kv = get_kv_stats().get("active_total", 0)
            prompts_this_cycle = 0

            while True:
                kv = get_kv_stats()
                current_kv = kv.get("active_total", 0)
                print(f"  KV: {current_kv}/{args.kv_alloc} ({100*current_kv/args.kv_alloc:.0f}%)", flush=True)

                if current_kv >= fill_target:
                    print(f"  KV window filled ({current_kv} >= {fill_target})", flush=True)
                    break

                # Pick next prompt (cycle through available prompts)
                p = research_prompts[prompt_idx % len(research_prompts)]
                prompt_idx += 1
                total_topics += 1
                prompts_this_cycle += 1

                turn = f"<|im_start|>user\n{p['prompt']}<|im_end|>\n<|im_start|>assistant\n"
                full_prompt = conversation + turn

                print(f"  [{p['id']}] {p['topic']} ({len(full_prompt)} chars)...", flush=True)
                resp = completion(full_prompt, max_tokens=args.max_tokens if args.max_tokens > 0 else 65536)

                if "error" in resp:
                    print(f"  ERROR: {resp['error']}", flush=True)
                    cycle_result["prompts"].append({"id": p["id"], "error": resp["error"]})
                    break

                content = resp.get("content", "")
                tok_s = resp.get("generation_tok_s", 0)
                words = len(content.split())
                tokens_predicted = resp.get("tokens_predicted", 0)

                conversation += turn + content + "<|im_end|>\n"

                print(f"    {tok_s:.1f} tok/s, {words} words, {tokens_predicted} tokens generated", flush=True)
                cycle_result["prompts"].append({
                    "id": p["id"],
                    "topic": p["topic"],
                    "generation_tok_s": tok_s,
                    "words": words,
                    "tokens_predicted": tokens_predicted,
                })

            # Record pre-compact state
            pre_kv = get_kv_stats()
            cycle_result["pre_compact_kv"] = pre_kv
            print(f"\n  Pre-compact KV: {pre_kv.get('active_total', 0)}", flush=True)

            # Recall test BEFORE compaction
            print(f"  [RECALL-PRE] Testing recall before compaction...", flush=True)
            pre_recall = recall_test(conversation, cycle, total_topics)
            pre_recall_words = len(pre_recall.get("content", "").split())
            pre_recall_has_content = "don't have" not in pre_recall.get("content", "").lower()
            print(f"    Pre-compact recall: {pre_recall_words} words, has_content={pre_recall_has_content}", flush=True)

            # Compact
            print(f"\n  [COMPACT] Triggering {args.compact_ratio}x {args.compact_method} compaction...", flush=True)
            compact_result = trigger_compaction(ratio=args.compact_ratio, method=args.compact_method)

            if "error" in compact_result:
                print(f"  COMPACT ERROR: {compact_result['error']}", flush=True)
                if compact_result.get("error_body"):
                    print(f"  Error body: {compact_result['error_body'][:500]}", flush=True)
                cycle_result["compaction"] = compact_result
                results["cycles"].append(cycle_result)
                break
            else:
                print(f"  Compacted in {compact_result.get('compaction_time_ms', 0)}ms", flush=True)
                cycle_result["compaction"] = compact_result

            # Post-compact KV
            post_kv = get_kv_stats()
            cycle_result["post_compact_kv"] = post_kv
            actual_ratio = pre_kv.get("active_total", 1) / max(post_kv.get("active_total", 1), 1)
            print(f"  Post-compact KV: {post_kv.get('active_total', 0)} (actual ratio: {actual_ratio:.1f}x)", flush=True)

            total_effective_tokens += pre_kv.get("active_total", 0)

            # Recall test AFTER compaction — send full conversation to reuse compacted prefix
            print(f"  [RECALL-POST] Testing recall after compaction...", flush=True)
            post_recall = recall_test(conversation, cycle, total_topics)
            post_recall_words = len(post_recall.get("content", "").split())
            post_recall_has_content = "don't have" not in post_recall.get("content", "").lower()
            print(f"    Post-compact recall: {post_recall_words} words, has_content={post_recall_has_content}", flush=True)

            cycle_result["recall"] = {
                "pre_compact": {
                    "words": pre_recall_words,
                    "has_content": pre_recall_has_content,
                    "tok_s": pre_recall.get("generation_tok_s", 0),
                },
                "post_compact": {
                    "words": post_recall_words,
                    "has_content": post_recall_has_content,
                    "tok_s": post_recall.get("generation_tok_s", 0),
                },
            }

            results["cycles"].append(cycle_result)

        # Final summary
        results["total_effective_tokens"] = total_effective_tokens
        results["total_topics"] = total_topics
        results["total_prompts_sent"] = prompt_idx

        # Write results
        with open(out_dir / "e2e-results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Print summary
        print(f"\n{'='*70}")
        print("E2E COMPACTION TEST SUMMARY")
        print(f"{'='*70}")
        print(f"Model: {args.model_name}")
        print(f"Cycles completed: {len(results['cycles'])}/{n_cycles}")
        print(f"Total topics: {total_topics}")
        print(f"Total effective tokens: {total_effective_tokens}")
        for c in results["cycles"]:
            compact = c.get("compaction", {})
            recall = c.get("recall", {})
            pre = recall.get("pre_compact", {})
            post = recall.get("post_compact", {})
            pre_kv_n = c.get("pre_compact_kv", {}).get("active_total", "?")
            post_kv_n = c.get("post_compact_kv", {}).get("active_total", "?")
            compact_ms = compact.get("compaction_time_ms", "ERR")
            err = "ERROR" if "error" in compact else "OK"
            print(f"  Cycle {c['cycle']}: KV {pre_kv_n}→{post_kv_n} | compact={compact_ms}ms [{err}] | recall: pre={pre.get('words',0)}w post={post.get('words',0)}w")
        print(f"\nResults: {out_dir / 'e2e-results.json'}")

    finally:
        kill_server(proc)


if __name__ == "__main__":
    main()
