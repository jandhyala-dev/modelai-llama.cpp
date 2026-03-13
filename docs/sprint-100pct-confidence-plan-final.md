# Sprint Plan: 100% Confidence on KV Compaction Goals (Final)

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

Goal 2 throughput comparison is **cross-run**: `compacted_decode_tok_s` from the
compacted row versus `baseline_decode_tok_s` from the `pipeline=baseline` row.
The in-run baseline is inflated by state-restore overhead and is NOT suitable
for gating — all in-run deltas are negative (-30% to -80%).

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

**Add to `longctx_result` struct (before line 489, i.e., before the `// safety` comment):**
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
        // 4K at high ratios: experimental.
        // Measured: 4K/2x PASS (0.973), 4K/4x PASS (0.928),
        //           4K/8x FAIL (0.838 < 0.85), 4K/16x PASS (0.903),
        //           4K/50x FAIL (0.825 < 0.85).
        // Non-monotonic quality means the safe boundary is ratio <= 4.
        if (n_ctx <= 4096 && ratio > 4) {
            return {"experimental", "4k_high_ratio_quality_unproven"};
        }
        // 32K: experimental (throughput regresses).
        if (n_ctx >= 32768) {
            return {"experimental", "32k_throughput_regression"};
        }
        // 8K-16K at any ratio, or 4K at ratio <= 4: supported.
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
        // INFORMATIONAL ONLY — do NOT use to gate result.pass.
        // The in-run throughput delta compares state-restore baseline vs
        // compacted decode, which systematically shows overhead (-30% to -80%).
        // The real Goal 2 comparison is cross-run: compacted_decode_tok_s
        // vs the pipeline=baseline row's baseline_decode_tok_s.
        // That cross-run comparison shows +54% at 4K, +89% at 8K, +42% at 16K.
        result.throughput_pass = (result.in_run_throughput_delta_pct > -60.0);
```

**Throughput gate (~line 1275, after quality threshold check):**
```cpp
        result.pass = result.logit_cosine >= threshold_value;
        // NOTE: throughput_pass is tracked but NOT gated here.
        // In-run delta is an apples-to-oranges comparison (state-restore
        // baseline vs compacted). Goal 2 throughput is verified cross-run
        // by comparing compacted_decode_tok_s against the baseline row.
        // ModelAI Phase D routing uses cross-run comparison for policy.
```

**Update CSV header (line ~501):**
Replace the trailing `\n` of the existing header with the new columns:
```
"...,artifact_path,support_level,support_reason,throughput_pass\n"
```
i.e., the existing `"crash,error_text,artifact_path\n"` becomes
`"crash,error_text,artifact_path,support_level,support_reason,throughput_pass\n"`.

**Update CSV format string (line ~525):**
Replace the trailing `\n` of the last format chunk. The existing
`"%s,%s,%s\n"` (crash, error_text, artifact_path) becomes:
```
"%s,%s,%s,%s,%s,%s\n"
```
with the additional args appended:
```
r.support_level.c_str(), r.support_reason.c_str(),
r.throughput_pass ? "true" : "false"
```
Do NOT add a separate format string — modify the existing final line to
avoid double-newline or missing-comma bugs.

**Update `schema_version` to `2` (wherever initialized, ~line 935).**

### Slice 1b — Rename throughput delta column

**File:** `tests/test-kv-compact-longctx.cpp`

Rename in struct, header, format, and all references:
- `throughput_delta_pct` -> `in_run_throughput_delta_pct`

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
                { "min_context",  4096 },
                { "max_context",  16384 },
                { "max_ratio_4k", 4 },
                { "note",         "50x supported at 8K-16K; 4K supported at 2x-4x only (8x fails 0.838 < 0.85); 32K experimental" },
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
| supported | Quality passes at measured ratios | logit_cosine >= threshold |
| experimental | Quality or throughput may fail; not production-safe | Recorded but not gated |
| blocked | Known catastrophic failure; must not be offered | Must not appear in supported routing |

Note: throughput is tracked via `throughput_pass` but is NOT used to gate
`result.pass`. The in-run throughput delta compares state-restore baseline
vs compacted decode, which is an apples-to-oranges comparison. Goal 2
throughput is verified cross-run by comparing `compacted_decode_tok_s`
against the `pipeline=baseline` row's `baseline_decode_tok_s`.

## Current Envelope (select pipeline, Qwen3-8B+)

| Context | Max Supported Ratio | Tier | Reason |
|---------|-------------------|------|--------|
| 4K | 4x | supported up to 4x, experimental above | Quality fails at 8x (0.838 < 0.85) and 50x (0.825 < 0.85); non-monotonic quality (16x passes at 0.903) |
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
    assert caps["compacted_prefix"]["supported_envelope"]["min_context"] == 4096
    assert caps["compacted_prefix"]["supported_envelope"]["max_ratio_4k"] == 4
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

**Add warning and dim-mismatch counter on dimension mismatch skip (line 53):**
```cpp
    if (d0 != lq.n_embd_head || d1 != lq.n_head_q) {
        LLAMA_LOG_WARN("q_capture: skipping tensor '%s' — dims [%d,%d] != expected [%d,%d]\n",
                       t->name, (int)d0, (int)d1, (int)lq.n_embd_head, (int)lq.n_head_q);
        lq.n_dim_mismatches++;
        return;
    }
```

**File:** `src/llama-kv-compact-self-study.h`

**Add dim-mismatch counter and tensor name tracking to `layer_q` (nested struct inside `llama_q_capture_state`, line 43):**
```cpp
    uint32_t    n_dim_mismatches = 0;             // tensors skipped due to dim mismatch
    std::string last_accepted_tensor_name;        // name of last tensor that passed all checks
```

Also reset the counter in `llama_q_capture_state::reset()` (self-study.cpp line 26, inside the loop):
```cpp
        lq.n_dim_mismatches = 0;
        lq.last_accepted_tensor_name.clear();
```

**Set tensor name after ALL checks pass (after the F32 type check at line 62, before the pending/append block at line 74):**
```cpp
    lq.last_accepted_tensor_name = t->name;
```
Note: this must be AFTER the `t->type != GGML_TYPE_F32` check (line 59-62),
not between the dim check and the type check, to avoid recording tensors that
are rejected for being non-F32.

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
    float    beta_sparsity        = 0.0f;  // fraction of log-beta values near zero (weight ≈ 1.0)
    float    fit_residual_mean    = 0.0f;  // mean relative error from fit_beta
};
```

### Slice 2c — Populate diagnostics in self-study pipeline

**File:** `src/llama-kv-compact-self-study.cpp`

**Helper function (add as static in `src/llama-kv-compact-self-study.cpp`, before the pipeline function):**
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

**Populate diagnostics in the pipeline function `llama_kv_compact_self_study_from_live_kv`.**

**Step 1: After Q capture (after line 446, before Phase 2), inside `if (stats)`:**
```cpp
    if (stats) {
        stats->n_layers_with_q = 0;
        stats->n_dim_mismatches = 0;
        for (int il = 0; il < q_state.n_layers; ++il) {
            if (q_state.layers[il].n_tokens > 0) {
                stats->n_layers_with_q++;
            }
            stats->n_dim_mismatches += q_state.layers[il].n_dim_mismatches;
        }
    }
```
Note: `n_dim_mismatches` is summed from the per-layer counters added in Slice 2a.

**Step 2: Inside the Phase 2 loop (after `llama_kv_compact_accumulate_attention_scores` call, line ~488), accumulate Q and K norms:**
```cpp
            // Accumulate Q/K norms for diagnostics
            if (stats) {
                q_norm_sum += compute_row_norm_mean(entry.queries);
                k_norm_sum += compute_row_norm_mean(entry.k);
                n_heads_seen++;
            }
```
Declare accumulators before the Phase 2 loop:
```cpp
    double q_norm_sum = 0.0, k_norm_sum = 0.0;
    double beta_norm_sum = 0.0, beta_sparsity_sum = 0.0;
    double fit_residual_sum = 0.0;
    uint32_t n_heads_seen = 0;
    uint32_t n_beta_heads_seen = 0;
```

**Step 3: Inside the Phase 3 solver loop (after `fit_beta` succeeds, line ~566), accumulate beta diagnostics:**
```cpp
            // Beta diagnostics
            if (stats) {
                // Beta norm (L2 of the log-space beta vector)
                float beta_norm_sq = 0.0f;
                uint32_t beta_zero_count = 0;
                for (uint32_t bi = 0; bi < (uint32_t)beta.size(); ++bi) {
                    beta_norm_sq += beta[bi] * beta[bi];
                    if (std::fabs(beta[bi]) < 1e-6f) beta_zero_count++;
                }
                beta_norm_sum += std::sqrt(beta_norm_sq);
                beta_sparsity_sum += (float)beta_zero_count / std::max<uint32_t>(1, (uint32_t)beta.size());
                n_beta_heads_seen++;
            }
```

**Step 4: For `fit_residual_mean`, use the existing `partition_sum_relative_error` output parameter from `fit_beta`.**
The pipeline currently passes `nullptr` (line 562). Change to:
```cpp
            float head_residual = 0.0f;
            if (!llama_kv_compact_fit_beta(entry.queries, entry.k,
                                            compacted_k, solver_opts,
                                            beta, stats ? &head_residual : nullptr)) {
                solver_ok = false;
                break;
            }
            if (stats) {
                fit_residual_sum += head_residual;
            }
```
This reuses the existing residual computation inside `fit_beta`
(lines 302-313 of `llama-kv-compact-solver.cpp`) — do NOT reimplement it.

**Step 5: After both Phase 2 and Phase 3 loops complete, finalize averages (inside the existing `if (stats)` block at line ~601):**
```cpp
        stats->q_norm_mean       = (n_heads_seen > 0) ? (float)(q_norm_sum / n_heads_seen) : 0.0f;
        stats->k_norm_mean       = (n_heads_seen > 0) ? (float)(k_norm_sum / n_heads_seen) : 0.0f;
        stats->beta_norm_mean    = (n_beta_heads_seen > 0) ? (float)(beta_norm_sum / n_beta_heads_seen) : 0.0f;
        stats->beta_sparsity     = (n_beta_heads_seen > 0) ? (float)(beta_sparsity_sum / n_beta_heads_seen) : 0.0f;
        stats->fit_residual_mean = (n_beta_heads_seen > 0) ? (float)(fit_residual_sum / n_beta_heads_seen) : 0.0f;
```

Note: `n_heads_seen` counts Phase 2 iterations (Q/K norm accumulation).
`n_beta_heads_seen` counts Phase 3 iterations (beta norm/sparsity/residual
accumulation). These may differ if Phase 3 exits early on solver failure.
Using separate counters prevents division by a count from the wrong phase.

### Slice 2d — Log diagnostics after self-study run

**File:** `src/llama-kv-compact-self-study.cpp`

At the end of the pipeline function, **inside the existing `if (stats)` block**
(line ~601). This MUST be inside the null guard — `stats` can be `nullptr` per
the function signature's default parameter:
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

**No new code required in `src/llama-kv-compact-solver.cpp`.**

`llama_kv_compact_fit_beta` already computes the partition-sum relative error
and exposes it via the optional `float * partition_sum_relative_error` parameter
(solver.cpp lines 302-313). The metric is `mean(|pred - target| / max(target, 1e-6))`
across all query rows — this is exactly the `fit_residual_mean` diagnostic.

The only change is in the caller (`src/llama-kv-compact-self-study.cpp`):
pass a non-null `float*` instead of `nullptr`. This is already specified in
Slice 2c Step 4 above. No solver modifications needed.

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

The materialization code is split across four separate functions:
`set_input_k`, `set_input_v`, `set_input_mask`, `set_input_beta`.
These are called independently from graph construction in `llama-graph.cpp`.

Add per-function timing inside each `set_input_*` function:

```cpp
// At the top of llama_compacted_prefix_set_input_k:
const int64_t t0 = ggml_time_us();

// At the bottom, before return:
const int64_t t1 = ggml_time_us();
LLAMA_LOG_DEBUG("compact_exec: K materialization %.1fms (prefix=%u)\n",
                (t1 - t0) / 1000.0, (unsigned)layer.n_compacted_tokens);

// Repeat the same pattern for set_input_v, set_input_mask, set_input_beta.
```

Additionally, add aggregate timing in `src/llama-graph.cpp` at the call
sites where all four functions are invoked for each layer. This gives
per-layer and total materialization cost:

```cpp
// In build_attn or the compacted-prefix input construction block:
const int64_t t_mat_start = ggml_time_us();
// ... all set_input_* calls for this layer ...
const int64_t t_mat_end = ggml_time_us();
// Accumulate total_materialization_us += (t_mat_end - t_mat_start);
// Log aggregate at the end of the build pass.
```

### Slice 3b — Graph tensor caching (B5 from performance roadmap)

**File:** `src/llama-kv-compacted-prefix-exec.cpp`

The key optimization: cache materialized K/V/beta tensors across decode
batches. Only invalidate when compacted prefix state changes (new
compaction applied). Mask must always be rebuilt because it depends on
`ubatch.pos` (current query positions).

**Why beta IS cacheable:** `set_input_beta` expands per-KV-head beta
values to per-Q-head via GQA broadcast. The source data (`layer.beta_data`)
does not change between decodes. The output shape varies with `n_tps`
(tokens per stream), but for single-token decode batches `n_tps == 1` is
constant. The cache stores the expanded beta data and skips re-expansion
when shape matches.

Add a cache struct:
```cpp
struct compacted_prefix_tensor_cache {
    uint32_t cached_n_prefix = 0;
    uint32_t cached_seq_id = UINT32_MAX;
    uint64_t cached_state_version = 0;  // bumped on each compaction

    // Cached K/V/beta data per layer (host-side, ready to set_data)
    std::vector<std::vector<float>> k_cache;     // [n_layers][n_head_kv * n_prefix * n_embd_head]
    std::vector<std::vector<float>> v_cache;     // same layout
    std::vector<std::vector<float>> beta_cache;  // [n_layers][n_head * n_tps * n_prefix]
    uint32_t cached_beta_n_tps = 0;              // shape guard for beta

    bool valid(uint32_t seq_id, uint64_t state_version) const {
        return cached_seq_id == seq_id && cached_state_version == state_version;
    }

    void invalidate() { cached_state_version = 0; }
};
```

**Implementation location:** The caching logic must be in the
`llama_kv_cache::set_input_compacted_prefix_*` WRAPPER methods
(llama-kv-cache.h lines 278-281), NOT in the free functions in exec.cpp.
These wrappers have access to `this` (and thus the cache member). The free
functions do not have a `llama_kv_cache*` parameter.

**Storage location:** The cache instance and version counter must be
declared as members of `llama_kv_cache` (private section). The cache
member must be declared `mutable` because the wrapper methods are `const`:
```cpp
    // in llama_kv_cache private section:
    mutable compacted_prefix_tensor_cache cp_tensor_cache;
    uint64_t compacted_prefix_version_counter = 0;
```

Public accessor:
```cpp
    uint64_t compacted_prefix_state_version() const { return compacted_prefix_version_counter; }
```

**Wrapper pattern (example for K):**
```cpp
void llama_kv_cache::set_input_compacted_prefix_k(ggml_tensor * dst, int32_t il, llama_seq_id seq_id) const {
    const auto * seq = compacted_prefix.get_seq(seq_id);
    if (!seq || !seq->enabled) return;

    const auto & layer = seq->layers[map_layer_ids.at(il)];
    const uint64_t ver = compacted_prefix_version_counter;

    if (cp_tensor_cache.valid(seq_id, ver) && il < (int32_t)cp_tensor_cache.k_cache.size()) {
        // Cache hit: copy from cache to tensor
        std::memcpy(dst->data, cp_tensor_cache.k_cache[map_layer_ids.at(il)].data(),
                     cp_tensor_cache.k_cache[map_layer_ids.at(il)].size() * sizeof(float));
        return;
    }

    // Cache miss: materialize via free function
    llama_compacted_prefix_set_input_k(dst, layer);

    // Fill cache (on first layer, allocate; otherwise just store)
    // ... store dst tensor data into cp_tensor_cache.k_cache[map_layer_ids.at(il)] ...
    cp_tensor_cache.cached_seq_id = seq_id;
    cp_tensor_cache.cached_state_version = ver;
}
```

Note: the above is a PATTERN, not verbatim — the implementer must adapt to
actual tensor memory layout (the free functions write to dst via memcpy with
stride calculations, not flat floats). The implementer should read the dst
tensor data back after materialization.

On first decode after compaction: materialize K/V/beta and store in cache.
On subsequent decodes: reuse cached K/V/beta, only rebuild mask (which
depends on current query token positions via `ubatch.pos`).

**File:** `src/llama-kv-cache.h`

Add `compacted_prefix_state_version()` method to track when compacted state changes:
```cpp
    uint64_t compacted_prefix_state_version() const { return compacted_prefix_version_counter; }
```

**Version bump location:** Increment the version counter in
`compacted_prefix_configure()` — this is where the actual payload data
(K, V, beta, logical positions) is written. Do NOT bump in
`set_execution` or `reclaim_live_kv` alone, because the data mutation
happens in `configure` (called internally by the pipeline functions).
If `configure` succeeds but `set_execution` fails, a bump in
`set_execution` would miss the stale-data window. Bumping in
`configure` covers all mutation paths:
- `compacted_prefix_configure()` -> bump
- `compacted_prefix_clear()` -> bump (data cleared)
- `state_read()` (restore) -> bump (data overwritten)

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
- **Cross-sprint dependency:** also update the hardcoded `max_context` in
  `tools/server/server-context.cpp` Slice 1c (`supported_envelope`
  section) from `16384` to `32768`

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
2. `supported` rows pass quality gate (`logit_cosine >= threshold`)
3. `throughput_pass` is tracked but informational only (not gated)
4. Self-study is explicitly `blocked` with diagnostic evidence
5. 4K/50x is explicitly `experimental` with clear reason
6. 32K is either fixed (promoted to `supported`) or explicitly `experimental`
7. CSV column semantics are unambiguous
8. `/props` exposes the supported envelope
9. Diagnostics exist to efficiently debug self-study failures
10. Phase 2 and Phase 3 stats use independent head counters

## What This Sprint Does NOT Cover

- Part 7 (server hardening) — independent, parallel workstream
- Part 8 beyond B5 (GPU-native mask/beta, autotuning) — deferred
- Solver/OMP pipeline promotion — needs separate benchmark campaign
- Public API guarantees — still private product-fork only
