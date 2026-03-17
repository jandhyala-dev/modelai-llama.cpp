# ModelAI Benchmark Results

Structured benchmark results for the modelai-llama.cpp KV cache compaction fork.
All results are designed for automated import into the ModelAI admin portal (Supabase).

## Directory Structure

```
bench-results/
  3way-YYYYMMDD-HHMMSS/           # 3-way comparison runs
    manifest.json                  # Run metadata (machine, commit, models)
    results.json                   # Full results (Supabase import source)
    results.csv                    # Tabular results (spreadsheet-friendly)
    summary.json                   # Aggregate comparison with verdict
  YYYYMMDD-HHMMSS-<sha>-<host>/   # Long-context benchmark runs
    manifest.json
    results.csv
    stdout.log
    stderr.log
    env.txt
  README.md                        # This file
```

## Supabase Import Guide

### Schema: `benchmark_results`

Create this table in Supabase for the 3-way comparison results:

```sql
CREATE TABLE benchmark_results (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  run_id TEXT NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  engine TEXT NOT NULL,                    -- 'modelai-llama.cpp', 'llama.cpp', 'ollama', 'modelai-compact'
  model TEXT NOT NULL,                     -- e.g. 'qwen3-14b'
  model_params_b REAL,                    -- e.g. 14.0
  architecture TEXT,                       -- e.g. 'qwen3', 'gemma3'
  quantization TEXT,                       -- e.g. 'Q4_K_M'
  test_type TEXT NOT NULL,                 -- 'inference' or 'compaction'
  pipeline TEXT,                           -- 'select', 'nonuniform', etc. (compaction only)
  n_ctx INTEGER,                           -- context size (compaction only)
  prompt_tokens INTEGER,
  decode_tokens INTEGER,
  compression_ratio INTEGER DEFAULT 1,
  prefill_tok_s REAL,
  decode_tok_s REAL,
  baseline_decode_tok_s REAL,             -- uncompacted baseline (compaction only)
  compacted_decode_tok_s REAL,            -- post-compaction (compaction only)
  logit_cosine REAL,                       -- quality metric (compaction only)
  compaction_time_ms REAL,                 -- time to compact (compaction only)
  active_n_kv INTEGER,                     -- effective KV after compaction
  pass BOOLEAN DEFAULT TRUE,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for dashboard queries
CREATE INDEX idx_benchmark_engine ON benchmark_results(engine);
CREATE INDEX idx_benchmark_model ON benchmark_results(model);
CREATE INDEX idx_benchmark_run ON benchmark_results(run_id);
CREATE INDEX idx_benchmark_test ON benchmark_results(test_type);
```

### Schema: `benchmark_runs`

Metadata for each benchmark run:

```sql
CREATE TABLE benchmark_runs (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  run_id TEXT UNIQUE NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  benchmark_type TEXT NOT NULL,            -- '3way-comparison', 'longctx', 'campaign'
  commit_sha TEXT NOT NULL,
  branch TEXT NOT NULL,
  hostname TEXT,
  os TEXT,
  arch TEXT,
  cpu TEXT,
  memory_gb INTEGER,
  models JSONB,                            -- array of model names
  engines JSONB,                           -- array of engine names
  verdict TEXT,                            -- summary verdict
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Schema: `benchmark_longctx_results`

For the detailed long-context KV compaction results (49-column schema):

```sql
CREATE TABLE benchmark_longctx_results (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  schema_version INTEGER DEFAULT 2,
  run_id TEXT NOT NULL,
  workload_id TEXT,
  workload_name TEXT,
  dataset_id TEXT,
  model_name TEXT NOT NULL,
  model_params_b REAL,
  architecture TEXT,
  quantization TEXT,
  backend TEXT,
  flash_mode TEXT,
  pipeline TEXT NOT NULL,
  n_ctx INTEGER NOT NULL,
  prefix_tokens INTEGER,
  compactable_tokens INTEGER,
  live_suffix_tokens INTEGER,
  compression_ratio INTEGER,
  compacted_tokens INTEGER,
  continuation_tokens INTEGER,
  prefill_ms REAL,
  prefill_tok_s REAL,
  compaction_time_ms REAL,
  query_generation_time_ms REAL,
  solver_time_ms REAL,
  first_token_ms REAL,
  baseline_decode_tok_s REAL,
  compacted_decode_tok_s REAL,
  in_run_throughput_delta_pct REAL,
  active_n_kv INTEGER,
  allocated_kv_bytes BIGINT,
  reclaimed_kv_bytes BIGINT,
  logit_cosine REAL,
  task_metric_name TEXT,
  task_metric_value TEXT,
  quality_correct INTEGER,
  quality_total INTEGER,
  quality_accuracy TEXT,
  quality_baseline_accuracy TEXT,
  longhealth_correct INTEGER,
  longhealth_total INTEGER,
  longhealth_accuracy TEXT,
  longhealth_baseline_accuracy TEXT,
  threshold_name TEXT,
  threshold_value REAL,
  pass BOOLEAN,
  fallback_used BOOLEAN,
  fallback_reason TEXT,
  crash BOOLEAN DEFAULT FALSE,
  error_text TEXT,
  support_level TEXT,
  support_reason TEXT,
  throughput_pass BOOLEAN,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Import Script (Python)

Use this to import `results.json` into Supabase:

```python
import json
from supabase import create_client

SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Import 3-way comparison results
with open("bench-results/3way-YYYYMMDD-HHMMSS/results.json") as f:
    data = json.load(f)

# Insert run metadata
with open("bench-results/3way-YYYYMMDD-HHMMSS/manifest.json") as f:
    manifest = json.load(f)

supabase.table("benchmark_runs").insert({
    "run_id": data["run_id"],
    "timestamp": data["timestamp"],
    "benchmark_type": manifest["benchmark_type"],
    "commit_sha": manifest["commit_sha"],
    "branch": manifest["branch"],
    "hostname": manifest["machine"]["hostname"],
    "os": manifest["machine"]["os"],
    "arch": manifest["machine"]["arch"],
    "cpu": manifest["machine"]["cpu"],
    "memory_gb": manifest["machine"]["memory_gb"],
    "models": manifest["models"],
    "engines": manifest["engines"],
}).execute()

# Insert individual results
for result in data["results"]:
    result["run_id"] = data["run_id"]
    result["timestamp"] = data["timestamp"]
    supabase.table("benchmark_results").insert(result).execute()
```

### Import Long-Context CSV Results

```python
import csv
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

with open("bench-results/YYYYMMDD-HHMMSS-sha-host/results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Convert numeric fields
        for key in ['prefill_ms', 'prefill_tok_s', 'compaction_time_ms',
                     'baseline_decode_tok_s', 'compacted_decode_tok_s',
                     'logit_cosine', 'threshold_value']:
            if row.get(key):
                try:
                    row[key] = float(row[key])
                except ValueError:
                    row[key] = None
        for key in ['n_ctx', 'prefix_tokens', 'compression_ratio',
                     'compacted_tokens', 'active_n_kv']:
            if row.get(key):
                try:
                    row[key] = int(row[key])
                except ValueError:
                    row[key] = None
        row['pass'] = row.get('pass', '').lower() == 'true'
        row['crash'] = row.get('crash', '').lower() == 'true'

        supabase.table("benchmark_longctx_results").insert(row).execute()
```

## Admin Portal Display

### Recommended Sub-Tab: "Benchmark Tests > modelai-llama.cpp Fork"

Display these views:

1. **3-Way Comparison** — Bar chart comparing engines by model:
   - X: model name, Y: tok/s, grouped by engine
   - Separate charts for prefill and decode

2. **KV Compaction Quality** — Line chart:
   - X: compression ratio, Y: logit cosine similarity
   - Lines per model, threshold line at 0.95

3. **Memory Efficiency** — Table:
   - Model | Context | Baseline KV (MB) | Compacted KV (MB) | Savings %

4. **Long-Context Sweep** — Heatmap:
   - X: context size, Y: compression ratio
   - Cell color: logit cosine (green > 0.95, yellow 0.90-0.95, red < 0.90)

## Result File Paths

All results are relative to the repository root:

| Type | Path Pattern | Format |
|------|-------------|--------|
| 3-way comparison | `bench-results/3way-*/results.json` | JSON |
| 3-way CSV | `bench-results/3way-*/results.csv` | CSV |
| Long-context | `bench-results/YYYYMMDD-*/results.csv` | CSV (schema v2) |
| Manifests | `bench-results/*/manifest.json` | JSON |
| Summaries | `bench-results/3way-*/summary.json` | JSON |

## Benchmark Rules

### BR-1: Disable `enable_thinking` for Qwen3 compaction benchmarks

**Affected models:** All Qwen3 `-2507` (July 2025 refresh) models — both Instruct and Thinking variants.
**Not affected:** Qwen3-Coder-30B-A3B-Instruct-1M (older chat template without thinking feature).

Qwen3 `-2507` models ship with `enable_thinking: true` baked into their chat templates.
This causes two benchmark failures:

1. **Re-prime step rejected.** Compaction quality tests replay assistant messages to restore
   KV cache state before compacting. The `enable_thinking` chat template rejects assistant
   response prefill with: `"Assistant response prefill is incompatible with enable_thinking."`.
   Without re-prime, the compaction test cannot run.

2. **Recall scoring fails.** With thinking enabled, model answers are either inside unclosed
   `<think>` tags (if max_tokens is hit before thinking finishes) or in a separate
   `reasoning_content` API field instead of `content`. The scoring regex searches `content`
   only, so it finds nothing and reports 0/10 recall even when the model knows the answer.

**Fix — per-request (preferred):**

```python
# In the chat() call, add enable_thinking: false
resp = api("POST", "/v1/chat/completions", {
    "model": "test",
    "messages": messages,
    "max_tokens": max_tokens,
    "chat_template_kwargs": {"enable_thinking": False},  # BR-1
})
```

**Fix — server-wide (alternative):**

```bash
llama-server -m model.gguf --chat-template-kwargs '{"enable_thinking": false}'
```

**Why this is safe:** Disabling thinking does not affect compaction quality. Compaction operates
on the KV cache, not on the model's generation behavior. The thinking feature only changes what
tokens the model generates — it does not change how the KV cache represents prior context.
Compaction quality is identical whether thinking is on or off.

**Discovery:** 2026-03-16 master benchmark (`ac87d9ab`). Instruct-2507 scored 0/10 baseline
recall and both models failed re-prime. Coder-1M (same 30B/3B MoE architecture, different chat
template) scored 10/10 at baseline, 2x, and 4x — proving the compaction engine works and the
issue is purely chat template incompatibility.

### BR-2: Budget 3-5x max_tokens for thinking models

Thinking models generate hidden reasoning tokens (`<think>...</think>`) before visible output.
At the 2026-03-16 benchmark:

| Task type | Thinking overhead | Visible output |
|-----------|-------------------|----------------|
| Coding    | >99% of tokens    | 0 chars (never reached answer) |
| Research  | 35-45% of tokens  | 7800-9000 chars visible |

For coding tasks, 4500 max_tokens produced zero visible output — all tokens were spent
on reasoning. For fair comparison, either:

- Set `max_tokens` to 5x the instruct baseline (e.g., 7500 for a 1500-token coding task), or
- Disable thinking via BR-1 for benchmarks that compare output quality across model types, or
- Report raw tok/s and wall time separately (raw throughput is ~50 tok/s for all 30B MoE
  variants; wall time differs due to thinking overhead).

### BR-3: Score recall from both `content` and `reasoning_content`

When `enable_thinking` is active, model responses may appear in either:
- `resp["choices"][0]["message"]["content"]` — visible answer
- `resp["choices"][0]["message"]["reasoning_content"]` — thinking content

Recall scoring functions must check both fields. If only `content` is checked and the model
put numbers inside thinking blocks, recall scores will be artificially zero.

```python
# Correct scoring approach
content = resp["choices"][0]["message"].get("content", "") or ""
reasoning = resp["choices"][0]["message"].get("reasoning_content", "") or ""
search_text = content + " " + reasoning

# Also strip <think> tags from content (some models inline thinking)
import re
search_text += " " + re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
```

## Running Benchmarks

```bash
# Full 3-way comparison (5 models, ~2-3 hours)
python3 scripts/run-3way-bench.py

# Long-context sweep (single model)
./scripts/bench-kv-compact-longctx.sh models/test/Qwen3-14B-Q4_K_M.gguf

# Full campaign (6 models, all pipelines, ~8-12 hours)
./scripts/bench-kv-compact-campaign2.sh

# 6-model master benchmark (speed + coding + research + compaction)
python3 scripts/bench-3model-master.py
```
