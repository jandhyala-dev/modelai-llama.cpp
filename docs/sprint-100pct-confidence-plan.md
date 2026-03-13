# Sprint Plan: 100% Confidence on KV Compaction Goals

## Fork Goals (from docs/modelai-kv-compaction-plan.md)

1. **Long-session efficiency** — Store old immutable context as a compacted prefix, keep recent mutable context as a live suffix, preserve chat-template/BOS tokens outside the compacted block.
2. **Post-prefill speed / decode throughput** — Lower the effective active KV range on repeated turns, achieve real compute reduction.

## Current Reality (Measured, Not Claimed)

**Working envelope:** select pipeline, 8K-16K context, 8B+ models.

| Context | Goal 1 (quality at 50x) | Goal 2 (throughput) |
|---------|------------------------|---------------------|
| 4K | FAIL (0.825 < 0.85) | PASS (+54%) |
| 8K | PASS (0.965) | PASS (+89%) |
| 16K | PASS (0.998) | PASS (+42%) |
| 32K | PASS (0.962) | FAIL (-22%) |

Self-study pipeline: ALL FAIL (0.11-0.71 cosine). Broken.

## Objective

Close the gap between "works in a narrow envelope" and "100% confidence across supported configurations" by:
1. Encoding the supported envelope truthfully in code and artifacts
2. Gating unsupported configurations so they cannot silently pass
3. Root-causing and fixing the self-study pipeline
4. Investigating and mitigating the 32K throughput regression

## Findings Being Addressed

| ID | Severity | Summary |
|----|----------|---------|
| F1 | Critical | Self-study produces garbage quality |
| F2 | Critical | 32K throughput regresses vs baseline |
| F3 | Critical | 4K/50x quality below threshold |
| F4 | Major | CSV throughput reporting semantics unclear |
| F6 | Critical | No supported-envelope policy encoded |
| F7 | Major | Self-study visible without support gate |
| F8 | Major | No throughput regression gate in benchmarks |
| F9 | Major | Insufficient self-study diagnostics |
| F10 | Major | Benchmark schema lacks support classification |

---

## Sprint Order

```
Sprint 1 (envelope + CSV) ─── prerequisite for everything
  └── Sprint 2 (self-study diagnostics + root cause)
  └── Sprint 3 (32K investigation + policy guard)
```

Sprints 2 and 3 are independent of each other but both depend on Sprint 1.

---

## Sprint 1 — Truthful Envelope + Benchmark Hardening

**Addresses:** F3, F4, F6, F7, F8, F10
**Effort:** 2-3 days
**Branch:** `kv-compact-envelope` from `modelai-main`

### Slice 1a — Support classification in benchmark CSV

**File:** `tests/test-kv-compact-longctx.cpp`

**Add to `longctx_result` struct (after line 489, before `// safety`):**
```cpp
    // support classification
    std::string support_level;    // "supported", "experimental", "blocked"
    std::string support_reason;   // human-readable reason for classification
    bool        throughput_pass;  // true if compacted >= baseline throughput
```

**Add classification function (after `lookup_threshold`, ~line 116):**
```cpp
struct support_classification {
    std::string level;   // "supported" | "experimental" | "blocked"
    std::string reason;
};

static support_classification classify_support(
        const std::string & pipeline,
        int n_ctx,
        int ratio) {
    // Self-study: blocked until quality is proven.
    if (pipeline == "self_study") {
        return {"blocked", "self_study_quality_unproven"};
    }

    // OMP: experimental (insufficient benchmark evidence).
    if (pipeline == "omp") {
        return {"experimental", "insufficient_benchmark_evidence"};
    }

    // Solver: experimental (insufficient benchmark evidence).
    if (pipeline == "solver") {
        return {"experimental", "insufficient_benchmark_evidence"};
    }

    // Baseline: always supported (it's the reference).
    if (pipeline == "baseline") {
        return {"supported", ""};
    }

    // Select pipeline: context + ratio dependent.
    if (pipeline == "select") {
        // 4K at high ratios: experimental (quality fails at 50x).
        if (n_ctx <= 4096 && ratio > 8) {
            return {"experimental", "4k_high_ratio_quality_unproven"};
        }
        // 32K: experimental (throughput regresses).
        if (n_ctx >= 32768) {
            return {"experimental", "32k_throughput_regression"};
        }
        // 8K-16K at any ratio: supported.
        return {"supported", ""};
    }

    return {"experimental", "unknown_pipeline"};
}
```

**Set classification in run logic (~line 933, after threshold lookup):**
```cpp
    const auto sc = classify_support(pipeline, n_ctx, ratio);
    // (these will be written into result later)
```

**Set fields in result (both baseline and compacted paths):**

For baseline path (~line 1036):
```cpp
        result.support_level = "supported";
        result.support_reason = "";
        result.throughput_pass = true;
```

For compacted paths (~line 1252, after throughput_delta_pct):
```cpp
        result.support_level = sc.level;
        result.support_reason = sc.reason;
        result.throughput_pass = (result.compacted_decode_tok_s >= result.baseline_decode_tok_s);
```

**Add throughput gate for supported rows (~line 1275, after quality threshold check):**
```cpp
        result.pass = result.logit_cosine >= threshold_value;

        // Throughput gate: supported rows must not regress.
        if (result.support_level == "supported" && !result.throughput_pass) {
            result.pass = false;
        }
```

**Update CSV header (line ~501):**
Add after `artifact_path`:
```
"support_level,support_reason,throughput_pass\n"
```

**Update CSV format string (line ~525):**
Add after `artifact_path`:
```
",%s,%s,%s"
```
with args:
```
r.support_level.c_str(), r.support_reason.c_str(),
r.throughput_pass ? "true" : "false"
```

**Update `schema_version` to `2` (wherever initialized, ~line 935).**

### Slice 1b — Rename throughput delta column

**File:** `tests/test-kv-compact-longctx.cpp`

Rename in struct, header, format, and all references:
- `throughput_delta_pct` → `in_run_throughput_delta_pct`

This is a search-and-replace across the file. ~8 occurrences.

### Slice 1c — Server capability reporting

**File:** `tools/server/server-context.cpp`

**Update `build_modelai_server_capabilities` compacted_prefix section (line 134):**
```cpp
        { "compacted_prefix", {
            { "available",                  meta.compaction_supported },
            { "enabled",                    compaction_enabled },
            { "zero_beta_flash_compatible", true },
            { "flash_attn_overridden",      compaction_flash_overridden },
            { "last_fallback_reason",       meta.compaction_supported ? "" : "model_unsupported" },
            { "supported_envelope", {
                { "pipelines",    json::array({"select"}) },
                { "min_context",  8192 },
                { "max_context",  16384 },
                { "max_ratio_4k", 8 },
                { "note",         "50x supported at 8K-16K; 4K capped at 8x; 32K experimental" },
            } },
        } },
```

### Slice 1d — Support envelope documentation

**File:** `docs/modelai-compaction-support-envelope.md` (NEW)

Content:
```markdown
# KV Compaction Support Envelope

## Classification Tiers

| Tier | Meaning | Benchmark Gate |
|------|---------|---------------|
| supported | Quality AND throughput pass at measured ratios | logit_cosine >= threshold AND compacted_tok/s >= baseline_tok/s |
| experimental | Quality or throughput may fail; not production-safe | Recorded but not gated |
| blocked | Known catastrophic failure; must not be offered | Must not appear in supported routing |

## Current Envelope (select pipeline, Qwen3-8B+)

| Context | Max Supported Ratio | Tier | Reason |
|---------|-------------------|------|--------|
| 4K | 8x | supported up to 8x, experimental above | Quality fails at 50x (0.825) |
| 8K | 50x | supported | Quality 0.965, throughput +89% |
| 16K | 50x | supported | Quality 0.998, throughput +42% |
| 32K | — | experimental | Throughput regresses (-22% at 50x) |

## Pipeline Classification

| Pipeline | Tier | Reason |
|----------|------|--------|
| select | supported (within context envelope) | Strong quality, proven throughput |
| solver | experimental | Insufficient benchmark evidence |
| omp | experimental | Insufficient benchmark evidence |
| self_study | blocked | Catastrophic quality failure (0.11-0.71 cosine) |

## Larger Models

| Model | Ratio | Quality | Tier |
|-------|-------|---------|------|
| Qwen3-14B-Q4_K_M | 8x | 0.990 | supported |
| Qwen3-30B-A3B-Q4_K_M | 8x | 0.999 | supported |
```

### Slice 1e — Tests

**File:** `tools/server/tests/unit/test_basic.py`

Update the `/props` test to verify the new `supported_envelope` section exists:
```python
    assert "supported_envelope" in caps["compacted_prefix"]
    assert caps["compacted_prefix"]["supported_envelope"]["min_context"] == 8192
```

**Build + test:**
```bash
cmake --build build --target test-kv-compact-longctx llama-server -j$(sysctl -n hw.ncpu)
# Verify CSV schema version 2 output
# Verify /props includes supported_envelope
```

---

## Sprint 2 — Self-Study Diagnostics + Root Cause

**Addresses:** F1, F9
**Effort:** 3-5 days
**Branch:** `kv-compact-self-study-fix` from `modelai-main` (after Sprint 1 merge)

### Slice 2a — Diagnostic instrumentation in Q capture

**File:** `src/llama-kv-compact-self-study.cpp`

**Add warning on dimension mismatch skip (line 53):**
```cpp
    if (d0 != lq.n_embd_head || d1 != lq.n_head_q) {
        LLAMA_LOG_WARN("q_capture: skipping tensor '%s' — dims [%d,%d] != expected [%d,%d]\n",
                       t->name, (int)d0, (int)d1, (int)lq.n_embd_head, (int)lq.n_head_q);
        return;
    }
```

**Add tensor name tracking to capture state. In the layer struct (wherever `llama_q_capture_layer` is defined):**
```cpp
    std::string last_accepted_tensor_name;  // name of last tensor that passed dim check
```

**Set it after successful capture (~line 57):**
```cpp
    lq.last_accepted_tensor_name = t->name;
```

### Slice 2b — Diagnostic stats extension

**File:** `src/llama-kv-compact-self-study.h`

**Extend `llama_kv_compact_self_study_stats`:**
```cpp
struct llama_kv_compact_self_study_stats {
    double   generation_time_ms   = 0.0;
    double   q_capture_time_ms    = 0.0;
    double   solver_time_ms       = 0.0;
    uint32_t n_tokens_generated   = 0;
    uint32_t n_queries_per_head   = 0;
    uint32_t n_prefix_tokens      = 0;
    uint32_t n_selected_tokens    = 0;

    // diagnostics (Sprint 2)
    uint32_t n_layers_with_q      = 0;  // layers where Q was captured
    uint32_t n_dim_mismatches     = 0;  // tensors skipped due to dim mismatch
    float    q_norm_mean          = 0.0f;  // mean L2 norm of captured Q rows
    float    k_norm_mean          = 0.0f;  // mean L2 norm of extracted K rows
    float    beta_norm_mean       = 0.0f;  // mean L2 norm of fitted beta vectors
    float    beta_sparsity        = 0.0f;  // fraction of beta values < 1e-6
    float    fit_residual_mean    = 0.0f;  // mean relative error from fit_beta
};
```

### Slice 2c — Populate diagnostics in self-study pipeline

**File:** `src/llama-kv-compact-self-study.cpp`

In the main pipeline function (after Q capture, after solver), compute and populate:

```cpp
// After Q capture finalize:
stats->n_layers_with_q = 0;
stats->n_dim_mismatches = 0;
for (int il = 0; il < q_state.n_layers; ++il) {
    if (q_state.layers[il].n_tokens > 0) {
        stats->n_layers_with_q++;
    }
}

// After GQA regroup + solver for each head, accumulate:
//   q_norm_mean: compute L2 norm of each Q row, average
//   k_norm_mean: compute L2 norm of each K row, average
//   beta_norm_mean: L2 norm of fitted beta vector
//   beta_sparsity: count(|beta[i]| < 1e-6) / len(beta)
//   fit_residual_mean: from solver output (relative error)
```

The exact code for norm computation:
```cpp
static float compute_row_norm_mean(const llama_kv_compact_matrix & m) {
    if (m.rows == 0 || m.cols == 0) return 0.0f;
    double sum = 0.0;
    for (uint32_t r = 0; r < m.rows; ++r) {
        const float * row = m.row(r);
        float norm_sq = 0.0f;
        for (uint32_t c = 0; c < m.cols; ++c) {
            norm_sq += row[c] * row[c];
        }
        sum += std::sqrt(norm_sq);
    }
    return (float)(sum / m.rows);
}
```

### Slice 2d — Log diagnostics after self-study run

**File:** `src/llama-kv-compact-self-study.cpp`

At the end of the pipeline function, after stats are populated:
```cpp
LLAMA_LOG_INFO("self_study diagnostics: layers_with_q=%u dim_mismatches=%u "
               "q_norm=%.4f k_norm=%.4f beta_norm=%.4f beta_sparsity=%.4f "
               "fit_residual=%.6f\n",
               stats->n_layers_with_q, stats->n_dim_mismatches,
               stats->q_norm_mean, stats->k_norm_mean,
               stats->beta_norm_mean, stats->beta_sparsity,
               stats->fit_residual_mean);
```

### Slice 2e — Solver per-head residual output

**File:** `src/llama-kv-compact-solver.cpp`

In `compute_exp_scores` or `fit_beta`, after NNLS completes, compute and store:
```cpp
// After NNLS solve for beta:
float residual = 0.0f;
// ... compute ||A*beta - b||^2 / ||b||^2 ...
// Store in pipeline_stats or return via output parameter
```

The exact residual computation:
```cpp
// After NNLS: beta is fitted, compute reconstruction error
double sum_sq_err = 0.0;
double sum_sq_target = 0.0;
for (uint32_t qi = 0; qi < queries.rows; ++qi) {
    // target = exp_scores[qi] (original attention distribution)
    // reconstructed = compacted_exp_scores[qi] * beta
    // ... dot product ...
    double err = target - reconstructed;
    sum_sq_err += err * err;
    sum_sq_target += target * target;
}
float relative_residual = (sum_sq_target > 0) ? (float)(sum_sq_err / sum_sq_target) : 0.0f;
```

### Slice 2f — Root cause analysis

After Slices 2a-2e, run self-study on Qwen3-8B at 4K:
```bash
PIPELINE=self_study RATIO=2 ./build/bin/test-kv-compact-longctx \
    -m models/test/Qwen3-8B-Q4_K_M.gguf -c 4096
```

Read the diagnostic output. Expected findings:
1. If `n_dim_mismatches > 0`: Q capture is silently falling back to wrong tensor variant
2. If `q_norm_mean` >> `k_norm_mean` or vice versa: Q/K space mismatch (scale or RoPE)
3. If `beta_sparsity > 0.9`: solver is producing degenerate solutions
4. If `fit_residual_mean > 0.5`: solver cannot reconstruct attention from self-study queries

**Fix based on diagnosis:**
- If tensor variant mismatch: fix dimension filter to accept the correct post-RoPE tensor
- If Q/K scale mismatch: add normalization before solver input
- If solver instability: increase lambda, reduce max_queries_per_kv_head, or add condition number check

### Slice 2g — Update support classification

If self-study quality is fixed (logit_cosine >= threshold):
- Promote from `blocked` to `experimental`
- Run full benchmark campaign to verify

If not fixed:
- Keep `blocked`
- Document root cause in `docs/modelai-compaction-support-envelope.md`

**Build + test:**
```bash
cmake --build build --target test-kv-compact-longctx -j$(sysctl -n hw.ncpu)
ctest --test-dir build -R kv-compact-self-study --output-on-failure
```

---

## Sprint 3 — 32K Throughput Investigation + Policy Guard

**Addresses:** F2, F8
**Effort:** 5-8 days
**Branch:** `kv-compact-32k-throughput` from `modelai-main` (after Sprint 1 merge)

### Slice 3a — Profiling instrumentation

**File:** `src/llama-kv-compacted-prefix-exec.cpp`

Add timing around the three materialization stages:

```cpp
// In the materialization function, wrap each stage:
const int64_t t_k_start = ggml_time_us();
// ... K extraction loop ...
const int64_t t_k_end = ggml_time_us();

const int64_t t_mask_start = ggml_time_us();
// ... mask building loop ...
const int64_t t_mask_end = ggml_time_us();

const int64_t t_beta_start = ggml_time_us();
// ... beta expansion loop ...
const int64_t t_beta_end = ggml_time_us();

LLAMA_LOG_DEBUG("compact_exec: K=%.1fms mask=%.1fms beta=%.1fms (prefix=%u)\n",
                (t_k_end - t_k_start) / 1000.0,
                (t_mask_end - t_mask_start) / 1000.0,
                (t_beta_end - t_beta_start) / 1000.0,
                n_prefix);
```

### Slice 3b — Graph tensor caching (B5 from performance roadmap)

**File:** `src/llama-kv-compacted-prefix-exec.cpp`

The key optimization: cache materialized K/V/mask/beta tensors across decode batches. Only invalidate when:
- Compacted prefix state changes (new compaction applied)
- Live suffix grows (new tokens decoded — but only mask needs updating)

Add a cache struct:
```cpp
struct compacted_prefix_tensor_cache {
    uint32_t cached_n_prefix = 0;
    uint32_t cached_seq_id = UINT32_MAX;
    uint64_t cached_state_version = 0;  // bumped on each compaction

    // Cached K/V data per layer (host-side, ready to set_data)
    std::vector<std::vector<float>> k_cache;  // [n_layers][n_head_kv * n_prefix * n_embd_head]
    std::vector<std::vector<float>> v_cache;  // same layout

    bool valid(uint32_t seq_id, uint64_t state_version) const {
        return cached_seq_id == seq_id && cached_state_version == state_version;
    }

    void invalidate() { cached_state_version = 0; }
};
```

On first decode after compaction: materialize K/V and store in cache.
On subsequent decodes: reuse cached K/V, only rebuild mask + beta (which depend on current token positions).

**File:** `src/llama-kv-cache.h`

Add `compacted_prefix_state_version()` method to track when compacted state changes:
```cpp
    uint64_t compacted_prefix_state_version() const;
```

Increment the version counter in `compacted_prefix_set_execution`, `compacted_prefix_reclaim_live_kv`, and any other mutation.

### Slice 3c — Re-measure 32K throughput

After implementing B5 tensor caching:
```bash
PIPELINE=select RATIO=50 CONTEXTS=32768 \
    ./build/bin/test-kv-compact-longctx \
    -m models/test/Qwen3-8B-Q4_K_M.gguf -c 32768
```

Compare `compacted_decode_tok_s` against baseline. If throughput now meets or exceeds baseline:
- Promote 32K from `experimental` to `supported`
- Update `classify_support()` in `test-kv-compact-longctx.cpp`
- Update `docs/modelai-compaction-support-envelope.md`

If still regressive:
- Keep 32K as `experimental`
- Document the measured improvement from B5
- File the remaining gap as Part 8 work (GPU-native mask/beta)

### Slice 3d — Runtime policy guard in /compact

**File:** `tools/server/server-context.cpp`

In the `SERVER_TASK_TYPE_COMPACT` handler, after method validation but before running compaction, add an optional support-level check:

```cpp
                    // Support envelope check (warn, do not block — experimental
                    // configurations are allowed but logged).
                    const int effective_ctx = prompt_tokens;  // approximate
                    if (method == "self_study") {
                        SRV_WRN("compaction method 'self_study' is currently blocked "
                                "(quality unproven) — proceeding at caller's risk\n");
                    }
                    if (effective_ctx >= 32768) {
                        SRV_WRN("compaction at context >= 32K is experimental "
                                "(throughput may regress) — proceeding at caller's risk\n");
                    }
```

This warns but does not block — the `/compact` endpoint is a power-user surface. The blocking happens at the ModelAI Phase D routing layer, not in the engine.

**Build + test:**
```bash
cmake --build build --target test-kv-compact-longctx llama-server -j$(sysctl -n hw.ncpu)
```

---

## Commit Strategy

Each slice is one commit. Commit message format:
```
kv-compact: <sprint>-<slice> <description>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

Examples:
```
kv-compact: 1a add support classification to benchmark CSV (F6, F10)
kv-compact: 1b rename throughput_delta_pct → in_run_throughput_delta_pct (F4)
kv-compact: 2a add diagnostic logging for Q capture dimension mismatch (F9)
kv-compact: 3b implement graph tensor caching for compacted prefix (F2)
```

## Success Criteria

After all three sprints:

1. Every benchmark row has `support_level` and `support_reason`
2. `supported` rows pass both quality AND throughput gates
3. Self-study is explicitly `blocked` with diagnostic evidence
4. 4K/50x is explicitly `experimental` with clear reason
5. 32K is either fixed (promoted to `supported`) or explicitly `experimental`
6. CSV column semantics are unambiguous
7. `/props` exposes the supported envelope
8. Diagnostics exist to efficiently debug self-study failures

## What This Sprint Does NOT Cover

- Part 7 (server hardening) — independent, parallel workstream
- Part 8 beyond B5 (GPU-native mask/beta, autotuning) — deferred
- Solver/OMP pipeline promotion — needs separate benchmark campaign
- Public API guarantees — still private product-fork only
