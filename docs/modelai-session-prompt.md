# ModelAI Session Prompt — modelai-llama.cpp Integration

## Context

The modelai-llama.cpp fork (jandhyala-dev/modelai-llama.cpp, branch: modelai-main, commit: d5f2d811) has achieved a complete V0 implementation of KV cache compaction via Attention Matching (arXiv:2602.16284). All benchmark results are committed, structured for Supabase import, and ready for integration into the ModelAI admin portal.

## Two Addendums Required

### Addendum 1: Benchmark Results Tab

**Task:** Add a new sub-tab "modelai-llama.cpp Fork" under the "Benchmark Tests" screen in the ModelAI admin portal.

**Data Sources (committed in modelai-llama.cpp repo):**

1. **3-way comparison results:** `bench-results/3way-20260314-052302/results.json` (39KB, 33 test points across 4 models x 3 engines)
2. **Financial analysis benchmark:** `bench-results/qwen3-30b-financial-20260314/results.json` (Qwen3-30B-A3B deep benchmark, 12 compaction points + throughput)
3. **Summary:** `bench-results/3way-20260314-052302/summary.json` (aggregate analysis with per-model quality stats)

**Supabase SQL schemas** are in `bench-results/README.md`. Create these tables:
- `benchmark_results` — individual test results (engine, model, test_type, prefill_tok_s, decode_tok_s, logit_cosine, etc.)
- `benchmark_runs` — run metadata (machine, commit, models, engines, verdict)
- `benchmark_longctx_results` — detailed 49-column long-context results

**Python import scripts** are also in `bench-results/README.md`.

**Display Views:**

1. **3-Way Engine Comparison** — Bar chart:
   - X: model name, Y: tok/s, grouped by engine (modelai-llama.cpp, llama.cpp, Ollama)
   - Separate charts for prefill and decode throughput
   - Highlight: modelai matches upstream performance while adding compaction

2. **KV Compaction Quality Heatmap** —
   - X: compression ratio (2x, 4x, 8x), Y: model name
   - Cell color: logit cosine (green > 0.99, yellow 0.95-0.99, red < 0.95)
   - All cells should be green (>0.97 across the board)

3. **Qwen3-30B-A3B Financial Deep Dive** — Table + line chart:
   - Context sizes: 4K, 8K, 16K, 32K
   - Compression: 2x, 4x, 8x
   - All 12 points with cosine, compaction time, and throughput delta

4. **Memory Efficiency** — Table:
   - Model | Context | Baseline KV (MB) | Compacted KV (MB) | Savings %

5. **Model Recommendation Card** — Highlight card:
   - Qwen3-30B-A3B as recommended financial analysis model
   - Key stats: 504 tok/s prefill, 42+ tok/s decode, >0.999 cosine at all compressions

**Key benchmark numbers to display prominently:**
- 33 test points across 4 models — ALL PASS
- Average logit cosine: 0.9967 (range: 0.973 to 0.999)
- Qwen3-30B-A3B: 0.9995 average cosine across 12 test points
- Baseline parity: modelai-llama.cpp matches upstream llama.cpp within noise
- Unique advantage: KV compaction is NOT available in Ollama or upstream llama.cpp

### Addendum 2: HTML Brief from Overview

**Task:** Create a clean, formatted HTML brief from the committed Markdown overview at `docs/modelai-llama-cpp-overview.md` in the modelai-llama.cpp repo (commit d5f2d811).

**Source file:** `https://github.com/jandhyala-dev/modelai-llama.cpp/blob/modelai-main/docs/modelai-llama-cpp-overview.md`

**Brief contents (section by section):**

1. **Executive Summary** — What modelai-llama.cpp is, what KV compaction does, why it matters for ModelAI
2. **The Science** — arXiv:2602.16284 explanation with link to paper, modified attention formula
3. **Gaps Filled** — 5 specific gaps in llama.cpp that this fork addresses:
   - No KV cache compaction → 7 compaction pipelines
   - No attention-aware token selection → attention-score-weighted selection
   - No memory efficiency signals → active_n_kv, reclaimed bytes
   - No solver infrastructure → Cholesky, NNLS, least-squares, power iteration
   - No long-context memory management → compaction cycles extend usable context
4. **Benchmark Results** — Key comparison tables (3-way, Qwen3-30B deep dive, cross-model quality)
5. **Architecture** — Pipeline diagram, source file reference, key commits
6. **V0 Support Matrix** — What works, what doesn't
7. **Quality Assurance** — Test suite, adversarial review process
8. **Advantages Summary** — vs Ollama table, vs upstream llama.cpp table

**Display location:** Admin console briefs section. This brief should be linked from the benchmark results tab.

**Integration with existing brief:** The overall ModelAI brief (from the previous Claude Code session) should be updated to reference this modelai-llama.cpp brief. Add a section or link pointing to the detailed fork overview. The existing ModelAI brief should mention:
- "ModelAI uses a custom fork of llama.cpp with KV cache compaction — see the modelai-llama.cpp technical brief for details"
- Link to the benchmark results tab

## Review Standards

**CRITICAL:** Both addendums MUST follow the review standards at `docs/review-standards/hostile-review-protocol.md` (committed in both modelai-llama.cpp and should be synced to modelai).

**New mandatory rule (just added to the protocol):** The testing-review loop is mandatory:
1. Tests must attempt to break the code
2. Every bug found must be fixed, committed, and pushed
3. Full adversarial review runs again after fixes
4. Cycle repeats until PASS with zero Critical/Major findings

**Update review standards in modelai and COT repos:** The hostile review protocol has been updated in modelai-llama.cpp with the "MANDATORY TESTING-REVIEW LOOP" section. This same section must be added to the review standards in:
- modelai repo (`docs/review-standards/hostile-review-protocol.md`)
- COT Labs repo (`docs/review-standards/hostile-review-protocol.md`)

## Phase D: Complete Testing

After the two addendums are coded, tested, committed, and pushed in modelai, proceed to Phase D testing:

### Phase D-1: Complete Model Testing

Run ALL models through modelai-llama.cpp, llama.cpp, and Ollama with side-by-side benchmark tables:

**Models to test:**
- Qwen3-30B-A3B (MoE, recommended)
- Qwen3-14B (dense)
- Qwen3-8B (dense)
- DeepSeek-R1-14B (dense, reasoning)
- Any other models available in Ollama

**Tests per model:**
- Baseline inference (pp512, pp2048, pp4096)
- Decode throughput (tg128)
- KV compaction quality at 2x, 4x, 8x (4K and 8K context)

### Phase D-2: ModelAI 28 Use-Case Testing

Test ALL 28 ModelAI use cases end-to-end using modelai-llama.cpp as the inference engine:
- Each use case run with Qwen3-30B-A3B and Qwen3-14B
- Measure: latency, quality, token counts
- Compare with Ollama baseline

### Phase D-3: Financial Analysis Deep Testing

Using Qwen3-30B-A3B:
- Last 3 months M&A activity analysis
- Last 12 months public company stock buybacks
- Earnings report summarization (real SEC filings)
- Financial metric extraction and comparison
- All at 8K-32K context with compaction

### Phase D-4: 128K Long-Context Testing

Using Qwen3-30B-A3B and models that support 128K context:
- Test at 16K, 32K, 64K, 128K token contexts
- KV compaction at 4x, 8x, 16x compression
- Quality measurement (logit cosine) at each point
- Document memory requirements vs available hardware

### Phase D-5: Record All Results

- All results in Supabase-ready JSON/CSV format
- Import into benchmark_results tables
- Display in modelai admin portal benchmark tab
- Commit and push result files to modelai-llama.cpp repo

### Phase D-6: Adversarial Review

- Full adversarial review of ALL Phase D results
- Follow hostile review protocol (no exceptions)
- Break-fix-review loop until clean PASS

### Phase D-7: Improvement Prompt

After all testing and reviews are complete, write a comprehensive prompt back to the modelai-llama.cpp Claude Code session with:
- All issues found during Phase D testing
- Performance bottlenecks identified
- Quality gaps at specific model/context/compression combinations
- Recommended improvements and fixes
- Priority ordering (Critical → Major → Minor)

## File References

All files are in the modelai-llama.cpp repo (jandhyala-dev/modelai-llama.cpp, branch: modelai-main):

| File | Purpose |
|------|---------|
| `docs/modelai-llama-cpp-overview.md` | Complete technical overview (source for HTML brief) |
| `docs/modelai-fork-summary.md` | Fork executive summary and goals |
| `docs/review-standards/hostile-review-protocol.md` | Review standards (updated with testing-review loop) |
| `bench-results/3way-20260314-052302/results.json` | 3-way benchmark results (Supabase import) |
| `bench-results/3way-20260314-052302/summary.json` | Aggregate analysis with verdict |
| `bench-results/3way-20260314-052302/manifest.json` | Run metadata |
| `bench-results/3way-20260314-052302/results.csv` | Tabular results |
| `bench-results/qwen3-30b-financial-20260314/results.json` | Qwen3-30B deep benchmark |
| `bench-results/README.md` | Supabase schemas, import scripts, display guide |
| `CLAUDE.md` | Build commands, conventions, branch model |
