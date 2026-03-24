#!/usr/bin/env python3
"""
Head-to-Head: Coder-1M vs Instruct-2507
Focus: Excel/formulas, 10-K research, complex analysis
Goal: Which model is faster for real product workloads?
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

# ── Excel / Spreadsheet Tasks ────────────────────────────────────────

EXCEL_TASKS = [
    {
        "name": "XLOOKUP + Nested IF",
        "messages": [
            {"role": "system", "content": "You are an Excel expert. Write formulas only — no explanation unless asked."},
            {"role": "user", "content": (
                "I have a sales spreadsheet:\n"
                "- Column A: Employee Name\n"
                "- Column B: Region (North, South, East, West)\n"
                "- Column C: Q1 Sales\n"
                "- Column D: Q2 Sales\n"
                "- Column E: Q3 Sales\n"
                "- Column F: Q4 Sales\n"
                "- Sheet2!A:B has Region → Target lookup table\n\n"
                "Write formulas for:\n"
                "1. G2: Total annual sales for each employee\n"
                "2. H2: Their region's target (XLOOKUP from Sheet2)\n"
                "3. I2: Performance rating — 'Exceeds' if >120% of target, 'Meets' if 90-120%, "
                "'Below' if 70-90%, 'Critical' if <70%\n"
                "4. J2: Commission — 5% of sales above target if 'Exceeds', 2% if 'Meets', 0 otherwise\n"
                "5. K2: Dynamic rank of all employees by total sales (no ties)"
            )},
        ],
        "max_tokens": 800,
    },
    {
        "name": "VBA Macro",
        "messages": [
            {"role": "system", "content": "You are a VBA expert. Write production-ready code."},
            {"role": "user", "content": (
                "Write a VBA macro that:\n"
                "1. Loops through all rows in 'RawData' sheet (A2:Z10000)\n"
                "2. For each row, if Column C (Status) = 'Active' and Column F (Amount) > 10000:\n"
                "   - Copy the row to 'HighValue' sheet\n"
                "   - Apply conditional formatting: green if Amount > 50000, yellow if > 25000, red otherwise\n"
                "3. Create a pivot summary on 'Summary' sheet: Region (Col B) × Category (Col D), sum of Amount\n"
                "4. Add a chart (stacked bar) from the pivot summary\n"
                "5. Save a PDF of the Summary sheet to the same folder as the workbook\n"
                "Include error handling and a progress bar in the status bar."
            )},
        ],
        "max_tokens": 2000,
    },
    {
        "name": "Python openpyxl Report",
        "messages": [
            {"role": "system", "content": "You are a Python developer specializing in Excel automation."},
            {"role": "user", "content": (
                "Write a Python script using openpyxl that:\n"
                "1. Reads 'sales_data.xlsx' with columns: Date, Product, Region, Units, Revenue, Cost\n"
                "2. Creates a new workbook 'monthly_report.xlsx' with these sheets:\n"
                "   a. 'Summary': Monthly revenue/cost/profit pivot by Region\n"
                "   b. 'Top Products': Top 10 products by revenue with sparkline-style conditional formatting\n"
                "   c. 'Trends': Month-over-month growth rates with red/green color coding\n"
                "3. Apply professional formatting: headers bold, currency format, alternating row colors\n"
                "4. Add data validation dropdowns for Region filter on Summary sheet\n"
                "5. Protect sheets with a password but allow filtering"
            )},
        ],
        "max_tokens": 2000,
    },
    {
        "name": "Complex Array Formulas",
        "messages": [
            {"role": "system", "content": "You are an Excel power user. Use modern dynamic array formulas (365/2021+)."},
            {"role": "user", "content": (
                "Data in A1:F500 with headers: Date, Ticker, Open, High, Low, Close\n\n"
                "Write formulas for:\n"
                "1. H2: UNIQUE list of all tickers, sorted alphabetically\n"
                "2. I2: For each ticker in H, calculate 20-day moving average of Close (spill down)\n"
                "3. J2: For each ticker, calculate daily return % = (Close - prev Close) / prev Close\n"
                "4. K2: Rolling 30-day volatility (std dev of daily returns) for each ticker\n"
                "5. L2: SUMPRODUCT formula to calculate correlation between first two tickers' daily returns\n"
                "6. M2: A single formula that returns the ticker with the highest Sharpe ratio "
                "(assuming 5% risk-free rate, annualized)"
            )},
        ],
        "max_tokens": 1500,
    },
    {
        "name": "Google Sheets Apps Script",
        "messages": [
            {"role": "system", "content": "You are a Google Workspace developer."},
            {"role": "user", "content": (
                "Write a Google Apps Script that:\n"
                "1. Runs on a schedule (daily at 9 AM)\n"
                "2. Reads data from 'Inventory' sheet\n"
                "3. For items where Stock < Reorder_Point, sends an email to purchasing@company.com\n"
                "   with an HTML table of items to reorder (Item, Current Stock, Reorder Point, Supplier)\n"
                "4. Logs the notification to a 'Notification Log' sheet with timestamp\n"
                "5. Updates a 'Dashboard' sheet with: total items, items below reorder, total inventory value\n"
                "6. Creates a time-driven trigger if one doesn't exist\n"
                "Include proper error handling and rate limit awareness."
            )},
        ],
        "max_tokens": 2000,
    },
]

# ── 10-K / SEC Filing Research Tasks ──────────────────────────────────

RESEARCH_10K = [
    {
        "name": "Revenue Segment Analysis",
        "messages": [
            {"role": "system", "content": "You are a senior equity research analyst at Goldman Sachs."},
            {"role": "user", "content": (
                "Analyze NVIDIA's revenue segments as if reading their 10-K filing:\n\n"
                "FY2025 Revenue Breakdown:\n"
                "- Data Center: $115.2B (prev year: $47.5B) — includes H100/H200/B200, InfiniBand, DGX\n"
                "- Gaming: $12.8B (prev: $10.3B) — GeForce RTX 50-series, cloud gaming\n"
                "- Professional Visualization: $2.1B (prev: $1.5B) — RTX for enterprise, Omniverse\n"
                "- Automotive: $1.7B (prev: $1.1B) — DRIVE platform, AV partnerships\n"
                "- OEM & Other: $0.4B (prev: $0.5B)\n\n"
                "Total: $132.2B (prev: $60.9B)\n"
                "Gross Margin: 73.5% (prev: 72.7%)\n"
                "R&D: $12.8B (9.7% of revenue)\n"
                "Operating Income: $82.1B (62.1% margin)\n\n"
                "Provide:\n"
                "1. Segment growth rates and revenue concentration risk\n"
                "2. Margin trajectory analysis\n"
                "3. R&D efficiency ratio vs peers\n"
                "4. Three key risks from the 10-K risk factors section\n"
                "5. Forward revenue estimate for FY2026 with assumptions"
            )},
        ],
        "max_tokens": 2000,
    },
    {
        "name": "Balance Sheet Deep Dive",
        "messages": [
            {"role": "system", "content": "You are a credit analyst evaluating financial health."},
            {"role": "user", "content": (
                "Analyze this balance sheet data (in $B):\n\n"
                "COMPANY A (Tech, $130B revenue):\n"
                "- Cash & equivalents: $31.4B\n"
                "- Short-term investments: $12.2B\n"
                "- Accounts receivable: $17.8B (DSO: 49 days)\n"
                "- Inventory: $5.3B (DIO: 78 days)\n"
                "- Total current assets: $69.2B\n"
                "- PP&E: $8.2B\n"
                "- Goodwill: $24.1B\n"
                "- Total assets: $112.4B\n"
                "- Current liabilities: $18.7B\n"
                "- Long-term debt: $8.5B\n"
                "- Total liabilities: $32.6B\n"
                "- Shareholders' equity: $79.8B\n\n"
                "COMPANY B (Tech, $28B revenue):\n"
                "- Cash: $5.8B\n"
                "- Short-term investments: $1.2B\n"
                "- AR: $6.1B (DSO: 79 days)\n"
                "- Inventory: $4.9B (DIO: 142 days)\n"
                "- Total current assets: $20.1B\n"
                "- PP&E: $6.2B\n"
                "- Goodwill: $24.2B\n"
                "- Total assets: $67.9B\n"
                "- Current liabilities: $12.3B\n"
                "- Long-term debt: $2.4B\n"
                "- Total liabilities: $22.1B\n"
                "- Shareholders' equity: $45.8B\n\n"
                "Calculate and compare:\n"
                "1. Current ratio, quick ratio, cash ratio\n"
                "2. Debt-to-equity, interest coverage (assume 4% coupon)\n"
                "3. Working capital efficiency (CCC = DSO + DIO - DPO, estimate DPO)\n"
                "4. Goodwill-to-equity ratio and impairment risk\n"
                "5. Overall credit assessment (investment grade? which is stronger?)"
            )},
        ],
        "max_tokens": 2000,
    },
    {
        "name": "Risk Factor Extraction",
        "messages": [
            {"role": "system", "content": "You are an SEC filing analyst specializing in risk factor assessment."},
            {"role": "user", "content": (
                "Based on typical semiconductor company 10-K risk factors, generate a structured "
                "risk assessment for a company with these characteristics:\n"
                "- $130B revenue, 87% from one segment (AI data center)\n"
                "- Top 5 customers = 45% of revenue (hyperscalers)\n"
                "- Manufacturing outsourced to TSMC (100% of advanced chips)\n"
                "- 35% of revenue from China-based customers\n"
                "- $24B goodwill from acquisitions (Mellanox, Arm attempt)\n"
                "- Stock-based compensation: $4.2B/year\n"
                "- 29,000 employees, competing for talent with Big Tech\n\n"
                "Structure as:\n"
                "1. TOP 5 risks ranked by severity (1-10) and probability (1-10)\n"
                "2. For each: Risk description, quantified impact, mitigation, monitoring KPI\n"
                "3. Risk correlation matrix (which risks amplify each other)\n"
                "4. Black swan scenario: What single event could cut revenue by >50%?\n"
                "5. ESG risks specific to this profile"
            )},
        ],
        "max_tokens": 2500,
    },
]

# ── Complex Analysis Tasks ────────────────────────────────────────────

COMPLEX_ANALYSIS = [
    {
        "name": "DCF Valuation Model",
        "messages": [
            {"role": "system", "content": "You are a senior investment banker building financial models."},
            {"role": "user", "content": (
                "Build a DCF valuation for a company with:\n"
                "- Current revenue: $132B, growing 40% (decelerating 5pp/year)\n"
                "- Gross margin: 73.5%, expanding 50bp/year to cap at 76%\n"
                "- OpEx as % of revenue: 28%, declining 100bp/year to floor at 22%\n"
                "- CapEx: 5% of revenue, increasing to 8% as builds fabs\n"
                "- Tax rate: 12% (effective, offshore structure)\n"
                "- D&A: 3% of revenue\n"
                "- Working capital: 10% of incremental revenue\n"
                "- WACC: 10.5%\n"
                "- Terminal growth: 3%\n\n"
                "Show:\n"
                "1. 5-year projection table (Revenue through FCF)\n"
                "2. Terminal value calculation (both perpetuity and exit multiple)\n"
                "3. Enterprise value and equity value (net debt: -$35B, i.e., net cash)\n"
                "4. Per-share value (24.5B diluted shares)\n"
                "5. Sensitivity table: WACC (8-13%) × Terminal growth (2-4%)\n"
                "6. Football field summary with bear/base/bull cases"
            )},
        ],
        "max_tokens": 3000,
    },
    {
        "name": "Portfolio Construction",
        "messages": [
            {"role": "system", "content": "You are a quantitative portfolio manager."},
            {"role": "user", "content": (
                "Given these asset returns and correlations, construct an optimal portfolio:\n\n"
                "Expected Returns (annual):\n"
                "- US Large Cap (SPY): 9.2%\n"
                "- US Small Cap (IWM): 11.5%\n"
                "- Int'l Developed (EFA): 7.8%\n"
                "- Emerging Markets (EEM): 10.2%\n"
                "- US Bonds (AGG): 4.5%\n"
                "- TIPS: 3.8%\n"
                "- Gold (GLD): 6.5%\n"
                "- REITs (VNQ): 8.1%\n\n"
                "Volatilities: 15.2%, 19.8%, 16.5%, 21.3%, 4.2%, 5.1%, 15.8%, 18.2%\n\n"
                "Correlations (key pairs):\n"
                "SPY-IWM: 0.88, SPY-EFA: 0.72, SPY-AGG: -0.15, SPY-GLD: 0.05\n"
                "AGG-TIPS: 0.75, EEM-EFA: 0.68, VNQ-SPY: 0.62\n\n"
                "Constraints: No single asset >35%, bonds floor 15%, no leverage, no shorting\n\n"
                "Provide:\n"
                "1. Minimum variance portfolio weights\n"
                "2. Maximum Sharpe ratio portfolio (risk-free rate: 4.8%)\n"
                "3. Risk parity portfolio\n"
                "4. Efficient frontier: 5 portfolios from min-var to max-return\n"
                "5. For each: expected return, volatility, Sharpe ratio, max drawdown estimate"
            )},
        ],
        "max_tokens": 3000,
    },
    {
        "name": "M&A Accretion/Dilution",
        "messages": [
            {"role": "system", "content": "You are an M&A advisor at a bulge bracket bank."},
            {"role": "user", "content": (
                "Analyze this acquisition:\n\n"
                "ACQUIRER:\n"
                "- Share price: $142, Shares: 24.5B, Market cap: $3.48T\n"
                "- EPS: $2.94, P/E: 48.3x\n"
                "- Revenue: $132B, Net Income: $72B\n"
                "- Cash on hand: $43B, Debt capacity: $30B\n\n"
                "TARGET:\n"
                "- Share price: $185, Shares: 1.6B, Market cap: $296B\n"
                "- EPS: $4.12, P/E: 44.9x\n"
                "- Revenue: $28B, Net Income: $6.6B\n"
                "- Expected synergies: $3B/year (50% cost, 50% revenue)\n\n"
                "Deal terms: 30% premium, 60% stock / 40% cash\n\n"
                "Calculate:\n"
                "1. Offer price and total deal value\n"
                "2. Shares issued and new share count\n"
                "3. Pro forma EPS — accretive or dilutive? By how much?\n"
                "4. Break-even synergies needed for accretion\n"
                "5. Pro forma leverage ratios\n"
                "6. IRR at different exit multiples (3-year hold, 35-55x P/E exit)"
            )},
        ],
        "max_tokens": 3000,
    },
]

# ── API / Server ──────────────────────────────────────────────────────

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

def chat(messages, max_tokens=2000):
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

def out(msg, end="\n"):
    sys.stdout.write(msg + end)
    sys.stdout.flush()

# ── Run ───────────────────────────────────────────────────────────────

def run_suite(tasks, suite_name):
    results = {}
    out(f"\n    {suite_name}:")
    for task in tasks:
        out(f"      {task['name']}...", end="")
        resp = chat(task["messages"], max_tokens=task["max_tokens"])
        if "error" in resp:
            out(f" ERROR: {resp['error']}")
            results[task["name"]] = {"status": "ERROR", "error": str(resp["error"])}
            continue

        usage = resp.get("usage", {})
        wall = resp["_wall"]
        prompt_tok = usage.get("prompt_tokens", 0)
        comp_tok = usage.get("completion_tokens", 0)
        tok_s = comp_tok / wall if wall > 0 else 0
        content = resp["choices"][0]["message"]["content"]

        # Count lines of code/formulas in response
        code_lines = 0
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and not stripped.startswith("//") and not stripped.startswith("*"):
                code_lines += 1

        results[task["name"]] = {
            "status": "OK",
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "wall_time_s": round(wall, 2),
            "gen_tok_s": round(tok_s, 1),
            "content_chars": len(content),
            "content_lines": code_lines,
            "hit_limit": comp_tok >= task["max_tokens"] - 5,
            "content_preview": content[:300],
        }

        limit_tag = " [HIT LIMIT]" if comp_tok >= task["max_tokens"] - 5 else ""
        out(f" {tok_s:.1f} tok/s | {comp_tok}t | {wall:.1f}s | {len(content)} chars{limit_tag}")

    return results


def main():
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    out("=" * 76)
    out("HEAD-TO-HEAD: Coder-1M vs Instruct-2507")
    out("Excel/Formulas, 10-K Research, Complex Financial Analysis")
    out(f"Hardware: Apple M2 Pro 32GB | Context: 64K | KV: f16 | enable_thinking: false")
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

        r = {}
        r["excel"] = run_suite(EXCEL_TASKS, "Excel / Spreadsheet")
        r["10k"] = run_suite(RESEARCH_10K, "10-K / SEC Research")
        r["analysis"] = run_suite(COMPLEX_ANALYSIS, "Complex Financial Analysis")

        all_results[model["name"]] = r
        kill_server()
        out(f"\n  Done.\n")

    # ── COMPARISON TABLE ──────────────────────────────────────────────
    out("\n" + "=" * 76)
    out("COMPREHENSIVE COMPARISON: Instruct-2507 vs Coder-1M")
    out("=" * 76)

    suites = [
        ("EXCEL / SPREADSHEET", "excel", EXCEL_TASKS),
        ("10-K / SEC RESEARCH", "10k", RESEARCH_10K),
        ("COMPLEX FINANCIAL ANALYSIS", "analysis", COMPLEX_ANALYSIS),
    ]

    # Aggregates
    totals = {m["name"]: {"tokens": 0, "wall": 0, "tasks": 0, "hit_limit": 0} for m in MODELS}

    for suite_label, suite_key, tasks in suites:
        out(f"\n── {suite_label} ──")
        out(f"{'Task':<28} {'Instruct tok/s':>14} {'Instruct time':>13} {'Instruct tok':>12}  "
            f"{'Coder tok/s':>11} {'Coder time':>10} {'Coder tok':>9}  {'Winner':>8}")
        out("-" * 120)

        for task in tasks:
            name = task["name"]
            cols = []
            for m in MODELS:
                r = all_results.get(m["name"], {}).get(suite_key, {}).get(name, {})
                tok_s = r.get("gen_tok_s", 0)
                wall = r.get("wall_time_s", 0)
                toks = r.get("completion_tokens", 0)
                hit = r.get("hit_limit", False)
                cols.append((tok_s, wall, toks, hit))

                totals[m["name"]]["tokens"] += toks
                totals[m["name"]]["wall"] += wall
                totals[m["name"]]["tasks"] += 1
                if hit:
                    totals[m["name"]]["hit_limit"] += 1

            # Winner by wall time
            if cols[0][1] > 0 and cols[1][1] > 0:
                if cols[1][1] < cols[0][1]:
                    winner = "CODER"
                    pct = (cols[0][1] - cols[1][1]) / cols[0][1] * 100
                    winner_str = f"Coder {pct:.0f}%"
                elif cols[0][1] < cols[1][1]:
                    winner = "INSTRUCT"
                    pct = (cols[1][1] - cols[0][1]) / cols[1][1] * 100
                    winner_str = f"Inst {pct:.0f}%"
                else:
                    winner_str = "TIE"
            else:
                winner_str = "?"

            lim0 = "*" if cols[0][3] else " "
            lim1 = "*" if cols[1][3] else " "

            out(f"{name:<28} {cols[0][0]:>10.1f} t/s {cols[0][1]:>9.1f}s {cols[0][2]:>8}t{lim0}  "
                f"{cols[1][0]:>9.1f} t/s {cols[1][1]:>7.1f}s {cols[1][2]:>7}t{lim1}  {winner_str:>8}")

    # ── AGGREGATE SUMMARY ─────────────────────────────────────────────
    out(f"\n{'='*76}")
    out("AGGREGATE SUMMARY (all 13 tasks)")
    out(f"{'='*76}\n")

    for m in MODELS:
        t = totals[m["name"]]
        avg_tok_s = t["tokens"] / t["wall"] if t["wall"] > 0 else 0
        out(f"  {m['name']}:")
        out(f"    Total tokens generated: {t['tokens']:,}")
        out(f"    Total wall time:        {t['wall']:.1f}s ({t['wall']/60:.1f} min)")
        out(f"    Average tok/s:          {avg_tok_s:.1f}")
        out(f"    Tasks hitting limit:    {t['hit_limit']}/{t['tasks']}")
        out("")

    i_t = totals["Instruct-2507"]
    c_t = totals["Coder-1M"]
    if i_t["wall"] > 0 and c_t["wall"] > 0:
        time_saved = i_t["wall"] - c_t["wall"]
        pct_faster = time_saved / i_t["wall"] * 100
        tok_diff = i_t["tokens"] - c_t["tokens"]
        out(f"  VERDICT:")
        if time_saved > 0:
            out(f"    Coder-1M finishes {time_saved:.0f}s faster ({pct_faster:.0f}% less wall time)")
            out(f"    Coder generates {tok_diff:,} fewer tokens ({tok_diff/max(i_t['tokens'],1)*100:.0f}% more concise)")
        else:
            out(f"    Instruct-2507 finishes {-time_saved:.0f}s faster ({-pct_faster:.0f}% less wall time)")
        out(f"    Instruct hits max_tokens: {i_t['hit_limit']}/{i_t['tasks']} tasks")
        out(f"    Coder hits max_tokens:    {c_t['hit_limit']}/{c_t['tasks']} tasks")

    # Legend
    out(f"\n  * = hit max_tokens limit (response may be truncated)")
    out(f"  enable_thinking: false for both models (BR-1)")

    # Save
    outdir = os.path.join(os.path.dirname(__file__), "..", "bench-results")
    outpath = os.path.join(outdir, f"coder-vs-instruct-{timestamp}.json")
    os.makedirs(outdir, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump({
            "benchmark": "coder-vs-instruct",
            "timestamp": timestamp,
            "hardware": "Apple M2 Pro 32GB",
            "context": CTX,
            "enable_thinking": False,
            "results": all_results,
            "totals": totals,
        }, f, indent=2, default=str)
    out(f"\n  Results: {outpath}")


if __name__ == "__main__":
    main()
