# PR-6b: KV Compaction Completion — Full Implementation Plan

## Owner
Ajay Jandhyala — ajay@model-ai.app (COT Labs / ModelAI)

## Repository
`jandhyala-dev/modelai-llama.cpp` — branch `modelai-main`

## Scope
This plan covers ALL remaining modelai-llama.cpp work to bring KV compaction and server runtime to production readiness. It addresses eight work areas:

**KV Compaction Core (Parts 1-5):**
1. **Self-study query generation** — highest-quality query path from the paper
2. **Model/backend coverage expansion** — every supported model/backend path
3. **Flash attention + non-zero beta** — final flash-attention story
4. **Production validation** — hardened proof on ModelAI workloads
5. **Upstream-ready sanitization** — clean API surface, docs, naming

**Runtime Integration & Hardening (Parts 6-8):**
6. **Compaction runtime integration** — capability reporting, server wiring, spec drift fixes
7. **Server/runtime hardening** — structured output, tool-call safety, stability, prompt-cache, performance
8. **Product optimization pipeline** — backend capability detection, autotuning, prefix fingerprint caching

ModelAI application-side changes (agent swarm flow, `/compact` endpoint, orchestrator logic) are tracked separately and will be confirmed after all modelai-llama.cpp work is complete.

---

# Review Conditions Resolution

This section documents how each condition from the adversarial reviews was resolved. Conditions were raised by Reviewer-1 (20 findings, all incorporated in commit `3efe995b`) and Reviewer-2 (5 conditions requiring plan updates before implementation).

## Reviewer-2 Conditions (resolved 2026-03-11)

| # | Condition | Resolution | Location in Plan |
|---|-----------|------------|-----------------|
| R2-1 | **[#8] Generation loop underspecified** — No pseudocode for seed token, API calls, EOS handling, KV capacity pre-check | Added complete pseudocode in Part 1 Section "Autoregressive Generation Loop — Complete Specification" with all API calls, EOS-ignore, capacity pre-check, cleanup, and error handling | Part 1, after "GQA Regrouping Detail" |
| R2-2 | **[#19] No quantitative pass/fail criteria** — Validation matrix lists metrics but no thresholds | Added per-test per-compression-ratio acceptance thresholds (ROUGE-L, F1, exact match), self-study vs surrogate delta, and paper reference baselines | Part 4, "Acceptance Criteria" subsection |
| R2-3 | **[#27] Tool-call parser conflict** — #18183 (tighten parser) and #14697 (check content) contradict | Added explicit disambiguation rule: only parse `content` as tool calls when matching `{"name":..., "arguments":...}` structure or `<tool_call>` XML tags; bare JSON never triggers tool-call parsing | Workstream 7b, item 8-9 |
| R2-4 | **[#31] Compacted prefix cache matching** — Token-level LCP fails against compacted prefixes | Added hash-based two-tier matching design: SHA-256 of original prefix token IDs at compaction time, two-tier lookup (token-LCP for uncompacted, hash for compacted), server API specification | Workstream 7e, item 6 |
| R2-5 | **[#34] modelai- prefix in src/** — Part 8 files violate Part 5 upstream-ready convention | Renamed all remaining `modelai-*` files to `llama-*` prefix throughout Part 8 (8a already fixed, now 8c and 8d fixed), updated schedule table | Part 8c, 8d, Implementation Schedule |

## Reviewer-1 Conditions (resolved in commit 3efe995b)

All 20 findings from Reviewer-1's adversarial review were incorporated into the plan with exact code locations and specific fixes. See the individual workstream sections for details.

## Deferred MAJOR Findings — Must Address During Implementation

**[Reviewer-2 Minor Note]** Two MAJOR findings from the adversarial reviews were approved for deferral to implementation time, but must not be forgotten:

1. **[Reviewer-2 #29] #11970 root cause (KV truncation):** The plan acknowledges the issue but defers root cause identification to the Workstream 7d audit. During implementation, the audit MUST document the specific root cause (likely prompt serialization instability across multi-turn `/v1/chat/completions`), add diagnostic logging for `n_past` drops, and verify the fix before the 7d merge gate.

2. **[Reviewer-2 #33] Merge conflict ordering (Parts 6 vs 7):** Both parts modify `server-context.cpp`. The plan specifies Phase 3a/3b split and "Part 6 merges first." During implementation, the branch strategy MUST be: create Part 7 branches AFTER Part 6 merges, or rebase Part 7 on Part 6 before merge. Track conflicts proactively — do not discover them at merge time.

---

# Part 1: Self-Study Query Generation

## What Self-Study Is

Self-study replaces the ~256 cache-key surrogate queries (K vectors used as proxy Q vectors) with real Q tensors captured from forward passes. The MIT paper (arXiv:2602.16284, Figure 4) identifies this as the single largest quality factor for KV cache compaction — worth more than all other optimizations combined.

Current state: `llama_kv_compact_extract_cache_key_queries()` in `src/llama-kv-compact-query.cpp` extracts K vectors as surrogate queries, uniformly sampled down to `max_queries` (default 256). These fail on GQA architectures because K and Q live in different spaces.

Self-study: Generate ~256 continuation tokens from the prefix, capture the real post-RoPE Q tensors via the existing `cb_eval` callback mechanism, regroup across GQA heads, and feed into the existing solver pipeline (`fit_beta` + `fit_values`).

## Key Technical Decisions

### 1. Q-Capture Mechanism: `cb_eval` callback (NOT `ggml_build_forward_expand`)

**Why NOT `ggml_build_forward_expand`:**
- It only adds nodes to the graph — does NOT set `GGML_TENSOR_FLAG_OUTPUT`
- Without the OUTPUT flag, the ggml allocator frees intermediate tensor buffers immediately after their last consumer (reference counting via `n_children` in `ggml/src/ggml-alloc.c:798-816`)
- Qcur's last consumer is the `kq` matmul — after that, its buffer is recycled
- Setting OUTPUT flags on intermediate tensors between graph build and alloc is fragile and unprecedented

**Why `cb_eval`:**
- Production-tested: imatrix tool (`tools/imatrix/imatrix.cpp:228-249`) already captures intermediate tensor data during forward passes using this exact mechanism
- Public API: exposed as `cb_eval` and `cb_eval_user_data` in `llama_context_params` (`include/llama.h:351-352`)
- Two-phase protocol: `ask=true` (filter), `ask=false` (receive with GPU-synced data)
- Scheduler batches non-interested nodes for efficient GPU dispatch (`ggml/src/ggml-backend.cpp:1585-1622`)
- Zero risk to existing graph allocation/scheduling behavior

### 2. Tensor Disambiguation: "Last 3D Qcur-prefixed tensor per layer"

In each layer, `cb(Qcur, "Qcur", il)` fires on 2-3 different ggml tensor nodes:

| Call | Location | ggml op | Shape | Name in graph |
|------|----------|---------|-------|---------------|
| 1 | After `build_lora_mm(wq, cur)` | `GGML_OP_MUL_MAT` | `[n_embd, n_tokens]` (2D) | `"Qcur-{il}"` |
| 2 | After `ggml_add(Qcur, bq)` (if bias) | `GGML_OP_ADD` | `[n_embd, n_tokens]` (2D) | `"Qcur-{il}"` |
| 3 | After `ggml_rope_ext(Qcur, ...)` | `GGML_OP_ROPE` | `[n_embd_head, n_head_q, n_tokens]` (3D) | `"Qcur-{il}"` |

Additional variants per model:
- `"Qcur_normed-{il}"` — after `ggml_rms_norm` (Llama4 kq_norm, Qwen3 q_norm)
- `"Qcur_scaled-{il}"` — after `ggml_scale` (Gemma)
- `"Qcur_attn_temp_scaled-{il}"` — after temperature scaling (Mistral3, DeepSeek2)

**Disambiguation strategy:**
1. Filter: `strncmp(t->name, "Qcur", 4) == 0` — catches all variants
2. Require: `ggml_n_dims(t) >= 3` — skips 2D pre-reshape projections
3. Overwrite per layer: last 3D Qcur-prefixed tensor wins — this is always the tensor passed to `build_attn()`

This handles all 85+ standard attention architectures. The cb_eval fires in topological (execution) order, so the last match is always correct.

### 3. RoPE Constraint: Already Satisfied

The captured Q tensor already has RoPE applied (the callback fires after `ggml_rope_ext`). From `docs/modelai-kv-compaction-plan.md`: "any later pre-RoPE or self-study query source must apply matching RoPE and GQA regrouping before fitting." The RoPE requirement is automatically satisfied. GQA regrouping is handled explicitly in step 6b-3.

### 4. GQA Regrouping

Q tensors have shape `[n_embd_head, n_head_q, n_tokens]`. The solver works per KV-head. For Qwen3-14B (n_head_q=40, n_head_kv=8, n_rep=5):
- Each KV head `h_kv` maps to Q heads `h_kv*5` through `h_kv*5+4`
- 256 generated tokens × 5 Q heads = 1,280 real queries per KV head
- Subsampled to `max_queries_per_kv_head` (default 1024)

### 5. KV Cache Cleanup

During Q-capture generation, `llama_decode()` writes K/V to the cache for the generated continuation tokens. After capture completes:
- Call `llama_kv_cache_seq_rm(ctx, seq_id, prefix_end, gen_pos)` to remove generated tokens
- This restores the cache to its pre-generation state
- Q vectors are already captured in host memory
- Memory overhead during generation: ~256 extra KV slots (negligible)

## File-by-File Changes (Self-Study)

### NEW FILES

#### `src/llama-kv-compact-self-study.h`

New header defining:

```cpp
// Configuration for self-study Q-capture generation
struct llama_kv_compact_self_study_config {
    uint32_t n_generate              = 256;    // continuation tokens to generate
    uint32_t max_queries_per_kv_head = 1024;   // subsample limit after GQA regrouping
    int      nnls_iters              = 64;     // solver iterations
    float    lambda                  = 1e-6f;  // solver regularization
    float    temperature             = 0.0f;   // sampling temp (0 = greedy)
};

// Statistics output
struct llama_kv_compact_self_study_stats {
    double   generation_time_ms   = 0.0;
    double   q_capture_time_ms    = 0.0;
    double   solver_time_ms       = 0.0;
    uint32_t n_tokens_generated   = 0;
    uint32_t n_queries_per_head   = 0;
    uint32_t n_prefix_tokens      = 0;
    uint32_t n_selected_tokens    = 0;
};

// Q-capture state (user_data for cb_eval callback)
struct llama_q_capture_state {
    bool    active   = false;
    int32_t n_layers = 0;

    struct layer_q {
        uint32_t n_embd_head = 0;
        uint32_t n_head_q    = 0;
        uint32_t n_tokens    = 0;
        std::vector<float> data;  // head-major: [head0_tok0..tokN, head1_tok0..tokN, ...]
    };
    std::vector<layer_q> layers;  // indexed by il

    void reset(int32_t n_layers, uint32_t n_embd_head, uint32_t n_head_q);
    void append_from_tensor(int32_t il, const ggml_tensor * t);
};

// cb_eval callback function
bool llama_q_capture_eval_callback(struct ggml_tensor * t, bool ask, void * user_data);

// Pipeline entry point
bool llama_kv_compact_self_study_from_live_kv(
        struct llama_context * ctx,
        llama_kv_cache       & kv,
        llama_seq_id           seq_id,
        uint32_t               target_tokens,
        llama_pos              live_suffix_pos0,
        const llama_kv_compact_self_study_config & config,
        llama_kv_compact_self_study_stats * stats = nullptr,
        llama_pos p0 = 0);
```

#### `src/llama-kv-compact-self-study.cpp`

Implementation containing:

1. **`llama_q_capture_state::reset()`** — Initialize per-layer storage vectors
2. **`llama_q_capture_state::append_from_tensor()`** — GPU→host copy via `ggml_backend_tensor_get()`, append to layer's float vector
3. **`llama_q_capture_eval_callback()`** — The cb_eval callback:
   - Ask phase: return true for 3D tensors with name starting "Qcur"
   - Receive phase: parse layer index from name suffix, call `append_from_tensor()`
   - Overwrite strategy: last 3D Qcur-prefixed tensor per layer wins
4. **`regroup_q_for_kv_head()`** — GQA regrouping: for KV head `h_kv`, collect Q vectors from Q heads `[h_kv*n_rep .. (h_kv+1)*n_rep)`, output as `llama_kv_compact_matrix`
5. **`subsample_queries()`** — Uniform stride subsampling to max_queries_per_kv_head
6. **`llama_kv_compact_self_study_from_live_kv()`** — Main pipeline:
   - Save existing cb_eval, install Q-capture callback
   - Autoregressive generation loop (n_generate tokens)
   - Disable capture, restore previous cb_eval
   - Remove generated tokens from KV cache
   - For each layer, for each KV head: regroup Q, subsample, extract K, run selection + fit_beta + fit_values
   - Store results in compacted prefix store

#### `tests/test-kv-compact-self-study.cpp`

Quality comparison test:
- Load a test model (stories15M or Qwen3-8B)
- Prefill a context
- Run compaction with cache-key surrogates (existing)
- Run compaction with self-study queries (new)
- Compare attention score reconstruction error
- Verify self-study produces lower error

### MODIFIED FILES (Self-Study)

#### `include/llama.h`

**Add** (after `llama_set_abort_callback`, ~line 971):
```cpp
LLAMA_API void llama_set_eval_callback(
        struct llama_context * ctx,
        ggml_backend_sched_eval_callback callback,
        void * user_data);
```

#### `src/llama-context.h`

**Add** method declaration in `llama_context` class:
```cpp
void set_eval_callback(ggml_backend_sched_eval_callback callback, void * user_data);
```

#### `src/llama-context.cpp`

**Add** implementation (after `set_abort_callback`):
```cpp
void llama_context::set_eval_callback(
        ggml_backend_sched_eval_callback callback,
        void * user_data) {
    cparams.cb_eval           = callback;
    cparams.cb_eval_user_data = user_data;
}
```

**Add** C API wrapper (near other `llama_set_*` functions):
```cpp
void llama_set_eval_callback(
        struct llama_context * ctx,
        ggml_backend_sched_eval_callback callback,
        void * user_data) {
    ctx->set_eval_callback(callback, user_data);
}
```

#### `src/llama-kv-compact-pipeline.h`

**Add** at end of file:
```cpp
struct llama_kv_compact_self_study_config;
struct llama_kv_compact_self_study_stats;

bool llama_kv_compact_self_study_from_live_kv(
        struct llama_context * ctx,
        llama_kv_cache       & kv,
        llama_seq_id           seq_id,
        uint32_t               target_tokens,
        llama_pos              live_suffix_pos0,
        const llama_kv_compact_self_study_config & config,
        llama_kv_compact_self_study_stats * stats = nullptr,
        llama_pos p0 = 0);
```

#### `src/llama-kv-cache.h`, `src/llama-kv-cache.cpp`

**Add** `compacted_prefix_self_study_from_live_kv()` wrapper method delegating to pipeline function.

#### `src/llama-kv-cache-iswa.h`, `src/llama-kv-cache-iswa.cpp`

**Add** delegation to `kv_base->compacted_prefix_self_study_from_live_kv(...)`.

#### `CMakeLists.txt`

**Add** `src/llama-kv-compact-self-study.cpp` to llama sources.

## cb_eval Callback Detailed Behavior

### Scheduler interaction (ggml/src/ggml-backend.cpp:1585-1622)

When cb_eval is set, the scheduler switches from batch-all to node-by-node mode:

```
for each node in graph split:
    ask callback: interested in this node?
    if NO: batch with next nodes until one IS interesting
    compute the batch
    GPU synchronize
    if last node was interesting: pass to callback (ask=false)
```

### Performance impact

For Qwen3-14B (28 layers), each decode step has ~2-3 "Qcur" graph nodes per layer that pass the ask filter. The scheduler batches all non-Qcur nodes (hundreds of ops) into single GPU dispatches. Cost per decode step:
- ~84 GPU syncs (3 Qcur variants × 28 layers) in worst case
- ~28 GPU syncs with the v1 optimization (track last pointer, skip non-last)
- Each sync: ~0.01ms + ~20KB GPU→host copy
- Total overhead: ~1-2ms per decode step

For 256 generated tokens: ~256-512ms total callback overhead on top of ~5s generation time (~5-10% overhead).

### When cb_eval is installed

`ggml_backend_sched_set_eval_callback()` is called in `process_ubatch()` at line 1136, but ONLY when the graph is rebuilt (not when reused). Since we set `cparams.cb_eval` before calling `llama_decode()`, it takes effect on the next graph rebuild. During autoregressive generation, single-token batches may reuse the graph — in that case, the callback from the previous rebuild persists on the scheduler.

## Overwrite Strategy for Multiple Qcur Nodes

### Problem
The cb_eval callback fires for ALL graph nodes named "Qcur-{il}", including pre-reshape 2D projections and the post-RoPE 3D tensor. We only want the last one per layer.

### Solution
During the receive phase (ask=false), we overwrite the captured data for each layer. The last write wins. Since graph nodes execute in topological order, the last "Qcur-{il}" is always the post-RoPE (or post-norm) version.

### Why overwriting is acceptable
Each Qcur tensor for a single decode token is small:
- 128 (n_embd_head) × 40 (n_head_q) × 1 (n_tokens) × 4 bytes = ~20KB
- Copying 20KB from GPU 2-3 times per layer is negligible vs. the GPU sync cost

### v1 optimization
Track the tensor pointer during ask phase (store last seen per layer). In receive phase, only copy if the pointer matches the last-seen one. This eliminates redundant copies but is not needed for v0 correctness.

## GQA Regrouping Detail

### Input
Per-layer Q data from capture: float vector of `[n_embd_head × n_head_q × n_tokens]`, laid out head-major.

### Regrouping for KV head `h_kv`
```
n_rep = n_head_q / n_head_kv  (e.g., 40/8 = 5 for Qwen3-14B)

For Q heads h_kv*n_rep through (h_kv+1)*n_rep - 1:
    For each captured token:
        Copy the [n_embd_head] float row into the output matrix

Output matrix: [n_rep * n_tokens, n_embd_head] = [5 * 256, 128] = [1280, 128]
```

### After regrouping
- Subsample to `max_queries_per_kv_head` (default 1024) via uniform stride
- Feed as `llama_kv_compact_matrix` into existing `fit_beta()` and `fit_values()`
- The solver sees 1024 real Q vectors instead of 256 K surrogates

## Autoregressive Generation Loop — Complete Specification

**[Reviewer-2 Condition R2-1]** Complete generation loop with all API calls, edge cases, and error handling.

### Seed Token

The first generated token comes from the **last prefilled token's logits** (argmax). If logits are unavailable from prefill, decode the last prefix token first:
```
seed_token = input_tokens[prefix_end - 1]
llama_decode(ctx, llama_batch_get_one(&seed_token, 1, prefix_end - 1, seq_id))
```

### KV Capacity Pre-Check

```cpp
if (kv_used + n_generate > kv_size) {
    LLAMA_LOG_WARN("self-study: insufficient KV capacity (%u used + %u needed > %u total)\n",
                   kv_used, n_generate, kv_size);
    return false;
}
```

### Complete Pseudocode

```cpp
bool llama_kv_compact_self_study_generate(
        llama_context * ctx, llama_q_capture_state & q_state,
        const llama_kv_compact_self_study_config & config,
        llama_seq_id seq_id, llama_pos prefix_end) {

    const uint32_t n_generate = config.n_generate;  // default 256

    // 1. KV capacity pre-check
    if (llama_kv_cache_used_cells(ctx) + n_generate > llama_kv_cache_size(ctx)) {
        LLAMA_LOG_WARN("self-study: insufficient KV capacity\n");
        return false;
    }

    // 2. Save existing cb_eval and install Q-capture callback
    auto prev_cb = ctx->cparams.cb_eval;
    auto prev_ud = ctx->cparams.cb_eval_user_data;
    q_state.active = true;
    ctx->set_eval_callback(llama_q_capture_eval_callback, &q_state);

    // 3. Seed token from last prefill logits
    float * logits = llama_get_logits_ith(ctx, -1);
    llama_token token = std::distance(logits,
        std::max_element(logits, logits + llama_n_vocab(llama_get_model(ctx))));

    // 4. Autoregressive generation loop
    llama_pos pos = prefix_end;
    uint32_t n_generated = 0;
    for (uint32_t i = 0; i < n_generate; i++) {
        llama_batch batch = llama_batch_get_one(&token, 1, pos, seq_id);
        if (llama_decode(ctx, batch) != 0) {
            LLAMA_LOG_ERROR("self-study: decode failed at step %u\n", i);
            break;
        }
        // cb_eval fires during decode, capturing Q tensors
        n_generated++;
        pos++;
        logits = llama_get_logits_ith(ctx, 0);
        token = std::distance(logits,
            std::max_element(logits, logits + llama_n_vocab(llama_get_model(ctx))));
        // Do NOT stop on EOS — continue for Q diversity (text is discarded)
    }

    // 5. Restore previous cb_eval
    q_state.active = false;
    ctx->set_eval_callback(prev_cb, prev_ud);

    // 6. Remove generated tokens from KV cache
    llama_kv_cache_seq_rm(ctx, seq_id, prefix_end, prefix_end + n_generated);

    LLAMA_LOG_INFO("self-study: captured Q from %u tokens\n", n_generated);
    return n_generated > 0;
}
```

### Key Design Decisions

1. **EOS handling:** Continue past EOS for Q diversity. EOS fires early (~50 tokens); stopping yields too few Q vectors (50 x 5 GQA = 250, below 1024 target). Text is discarded; only Q tensors matter.
2. **Greedy sampling (v0):** Argmax for reproducibility. v1: multiple continuations with different temperatures.
3. **Graph rebuild:** Switching from multi-token prefill to single-token decode triggers a rebuild, picking up the new `cb_eval`.
4. **Error recovery:** Partial Q capture is usable — fewer vectors means slightly lower quality, not failure.

---

# Part 2: Model/Backend Coverage Expansion

## Current Coverage Matrix

| Component | Status | Guard Location |
|-----------|--------|----------------|
| Standard causal attention | Supported | — |
| Flash attention + zero beta | Supported | `llama-graph.cpp:2091-2094` |
| Flash attention + non-zero beta | Blocked | `llama-graph.cpp:2093` (assertion) |
| iSWA (base layers) | Supported | `llama-graph.cpp:2386` |
| iSWA (SWA layers) | N/A — naturally discards | `llama-kv-cache.cpp:1024` |
| Quantized K cache | Supported | Block-alignment check in `llama-kv-compacted-prefix.cpp:201-205` |
| Quantized V (non-transposed) | Supported | — |
| Quantized V (transposed) | Blocked | `llama-kv-cache.cpp:977-981` |
| Multiple attention streams | Blocked | `llama-graph.cpp:2089` (P3 single-stream assertion) |
| MLA (DeepSeek2) | Not supported | Non-standard Q/K decomposition |
| Hybrid recurrent (Mamba/RWKV) | N/A — no KV attention | — |
| M-RoPE (Qwen2-VL) | Not tested | Uses `ggml_rope_multi` instead of `ggml_rope_ext` |
| Metal backend | Implicit — CPU-allocated, GPU-computed | `llama-kv-compacted-prefix.cpp:775` |
| CUDA backend | Implicit — CPU-allocated, GPU-computed | Same |

## Changes Required

### 2a. Quantized Transposed V Support

**Problem:** `llama-kv-cache.cpp:977-981` rejects quantized V when `v_trans=true` because quantization blocks span the embedding dimension, incompatible with column layout extraction.

**Solution:** De-quantize V to F32 during extraction, then re-quantize into the compacted prefix store. This mirrors what the solver already does for K extraction.

**Files:**
- `src/llama-kv-cache.cpp` — Replace early return with dequant-on-extract path in `compacted_prefix_copy_v_head_f32()`
- `tests/test-kv-compacted-prefix-pack.cpp` — Add test case for quantized transposed V

**Effort:** 1 day

### 2b. Multiple Attention Streams

**Problem:** `llama-graph.cpp:2089` asserts `n_stream == 1` for compacted prefix execution.

**Analysis (corrected):** The original analysis was incomplete. The set_input
(data-fill) functions (`set_input_mask`, `set_input_beta`) already handle
`n_stream > 1` via `dst->ne[3]`. However, the graph-building code in
`ensure_compacted_prefix_layer()` hardcodes `ne[3]=1` for K/V/kq_b tensors.
When `n_stream > 1`, `ggml_concat` asserts matching `ne[3]` on non-concat
dimensions, causing a fatal crash. Additionally, `can_execute` rejects
`n_seqs_unq != 1`, making multi-stream compacted prefix unreachable in
production. The assertions are correct guards, not conservative.

**6b-9 (completed):** Clarified guardrails + corrected plan assumptions.
- Added explanatory comments to both assertions documenting graph-path constraints
- Added `test_multi_stream_mask_and_beta()` proving set_input helpers are
  multi-stream-ready (data-fill layer is not the blocker)

**Remaining work for full multi-stream support (deferred):**
1. Thread `n_stream` through `ensure_compacted_prefix_layer()`
2. Create compacted K/V tensors with `ne[3]=n_stream` (not hardcoded 1)
3. Fix compacted `kq_b` to use `n_tps = n_tokens/n_stream` for `ne[1]`
4. Relax `can_execute` to allow `n_seqs_unq > 1` only after graph path is ready
5. Remove the `n_stream == 1` assertions in both standard and iSWA paths

**Files:**
- `src/llama-graph.cpp` — Clarified assertion comments (both standard and iSWA paths)
- `tests/test-kv-compacted-prefix-exec.cpp` — Multi-stream set_input test case

**Effort:** 0.5 day (completed); full multi-stream: 1-2 days (deferred)

### 2c. M-RoPE Model Testing

**Problem:** Models using `ggml_rope_multi()` (Qwen2-VL, Qwen3-VL, GLM4) have not been tested with compaction.

**6b-10 validation (completed):** Tested Qwen3-VL-2B-Instruct (Q4_K_M) with both
selection-only and full solver pipelines. Results:

| Pipeline | Cosine similarity | Threshold | Result |
|----------|------------------|-----------|--------|
| Selection-only | -0.13 | 0.95 | FAIL |
| Full solver | -0.04 | 0.95 | FAIL |

**Root cause analysis:**
1. `can_execute()` at `llama-kv-compacted-prefix-exec.cpp:54` correctly rejects
   M-RoPE batches via `ubatch.is_pos_2d()` — compacted prefix execution is never
   activated for M-RoPE models.
2. However, the pipeline still reclaims live KV cells for the prefix range via
   `compacted_prefix_reclaim_live_kv()`, deleting 256 of 320 context tokens.
3. Continuation decode runs without compacted prefix AND without prefix KV →
   80% context loss → catastrophic quality degradation.

**Code-level findings (all confirm V0 "unsupported" status):**
- `llama-kv-compacted-prefix-exec.cpp:54`: `is_pos_2d()` guard rejects M-RoPE
- `llama-kv-compacted-prefix.h:49`: `logical_positions` stores scalar `llama_pos`
  only, loses M-RoPE spatial coordinates (x, y)
- `llama-kv-compacted-prefix-exec.cpp:83-131`: mask computation uses scalar
  position comparisons, ignoring M-RoPE dimensions
- `llama-kv-cache.cpp:838-870`: position extraction ignores `llama_kv_cell_ext`

**Positive finding:** Q-capture callback and tensor naming are M-RoPE-compatible.
Both `ggml_rope_ext` and `ggml_rope_multi` produce identically named "Qcur-{il}"
tensors with the same 3D shape [n_embd_head, n_head_q, n_tokens].

**Safety fix applied:** Added M-RoPE guard to `compacted_prefix_runtime_supported()`
in `llama-kv-cache.cpp`. When `hparams.n_pos_per_embd() > 1`, the function returns
false, which blocks execution enablement (`set_execution`), execution resolution
(`resolve_compacted_prefix_exec`), and reclaim (`reclaim_live_kv`) — preventing the
catastrophic context loss chain. Note: pipeline entry points (`select_from_live_kv`,
`fit_from_live_kv`, `omp_from_live_kv`) are not guarded and may still build compacted
prefix state that cannot be executed; this wastes CPU but does not cause data loss.
By inspection and local validation (`./build/bin/test-kv-compact-quality -m
Qwen3VL-2B-Instruct-Q4_K_M.gguf`), Qwen3-VL now fails at `set_execution` (safe)
instead of at decode time after context loss (catastrophic).

**Remaining work for M-RoPE support (deferred):**
1. Store M-RoPE extended positions (`llama_kv_cell_ext`) in `logical_positions`
2. Update mask computation to use multi-dimensional position comparisons
3. Remove `is_pos_2d()` guard from `can_execute()` after fixing positions/mask

**Files:** `src/llama-kv-cache.cpp` — M-RoPE guard in `compacted_prefix_runtime_supported()`
**Effort:** 0.5 day (completed)

### 2d. Backend Validation

**Problem:** Compacted prefix data is always CPU-allocated (`ggml_backend_cpu_buffer_type()` in `llama-kv-compacted-prefix.cpp:775`). The set_input functions assume host-backed tensors (`require_host_or_direct_data` in `llama-kv-compacted-prefix-exec.cpp:20-27`). This works on Metal and CUDA because the ggml scheduler copies data to GPU as needed.

**6b-11 manual exploratory validation (completed):** Ran `test-kv-compact-quality` with
stories15M (Q4_0) on Metal and CPU-only backends, all three pipelines (selection, solver,
OMP), plus flash attention on Metal. Model: stories15M-q4_0.gguf, 24M params, 6 layers,
n_ctx=512. Artifact not yet archived; values below are from manual runs.

Commands used:
```bash
# Metal (default build: -DGGML_METAL=ON -DLLAMA_FATAL_WARNINGS=ON)
./build/bin/test-kv-compact-quality -m build/tinyllamas/stories15M-q4_0.gguf
USE_SOLVER=1 ./build/bin/test-kv-compact-quality -m build/tinyllamas/stories15M-q4_0.gguf
USE_OMP=1 ./build/bin/test-kv-compact-quality -m build/tinyllamas/stories15M-q4_0.gguf
USE_FLASH=1 ./build/bin/test-kv-compact-quality -m build/tinyllamas/stories15M-q4_0.gguf

# CPU-only (separate build: -DGGML_METAL=OFF -DGGML_CUDA=OFF -DLLAMA_FATAL_WARNINGS=ON)
./build-cpu/bin/test-kv-compact-quality -m build/tinyllamas/stories15M-q4_0.gguf
USE_SOLVER=1 ./build-cpu/bin/test-kv-compact-quality -m build/tinyllamas/stories15M-q4_0.gguf
USE_OMP=1 ./build-cpu/bin/test-kv-compact-quality -m build/tinyllamas/stories15M-q4_0.gguf
```

| Backend | Pipeline | 2x cos | 4x cos | 8x cos | Result |
|---------|----------|--------|--------|--------|--------|
| Metal | Selection | 0.9998 | 0.9996 | 0.9995 | PASS |
| Metal | Solver | 0.9991 | 0.9998 | 0.9997 | PASS |
| Metal | OMP | 0.9999 | 0.9986 | 0.9998 | PASS |
| Metal+Flash | Selection | 0.9998 | 0.9996 | 0.9995 | PASS |
| CPU-only | Selection | 0.9997 | 0.9996 | 0.9996 | PASS |
| CPU-only | Solver | 0.9996 | 0.9998 | 0.9995 | PASS |
| CPU-only | OMP | 0.9994 | 0.9933 | 0.9996 | PASS |

**Key findings:**
- Provides a preliminary backend sanity check on Metal vs CPU-only for a small model.
  Stories15M (24M params, 6 layers) validates data flow correctness but does not prove
  broad backend equivalence for production-class workloads — larger model validation is
  deferred to Part 4 (production workload tests).
- All pipelines exceed quality thresholds (2x≥0.95, 4x≥0.90, 8x≥0.85) by wide margins.
- CPU-only OMP at 4x compression (0.9933) is the lowest observed datapoint across all
  backend/pipeline combinations. Still above threshold but should be rechecked on larger
  models during production validation.
- Flash attention (zero-beta selection path) works identically to non-flash on Metal.
- CUDA validation deferred: no CUDA hardware available. CI policy (`modelai-ci-policy.md`)
  documents CUDA as Phase 3+ requirement.

**Files:** No code changes — validation only.
**Effort:** 0.5 day (completed)

---

# Part 3: Flash Attention + Non-Zero Beta

## The Problem

Flash attention (`ggml_flash_attn_ext`) does not accept a separate additive bias tensor. Its signature (`ggml/include/ggml.h:2323-2331`) only takes a mask parameter. The mask supports limited broadcasting (`n_head % ne32 == 0`) but cannot represent per-head per-layer beta values.

Non-zero beta (from the solver's `fit_beta`) requires adding a `[n_kv, n_tokens, n_head, 1]` tensor to the attention logits before softmax. In the standard (non-flash) attention path, this is done via `ggml_add(kq, kq_b)` at `llama-graph.cpp:1956`. Flash attention fuses the softmax internally, making external bias addition impossible.

## Current State

PR-6 solved this for the selection-only pipeline (zero beta) by:
- Adding `is_zero_beta()` check on sequence state
- Skipping `kq_b` allocation when beta is all zeros
- Using F16 compacted mask compatible with flash attention
- This is the production default path

## Path Forward

### Option A: Wait for FlashBias (Recommended)

The `modelai-performance-roadmap.md` already documents FlashBias (NeurIPS 2025, arXiv:2505.12044) as the solution. When ggml adds native flash attention bias support (or equivalent API/graph change):
1. The `kq_b == nullptr` guard at `llama-graph.cpp:1883` can be relaxed
2. Beta tensor gets passed to the new flash-bias API
3. Full solver pipeline works with flash attention

**Effort:** 0 (wait for upstream ggml)

### Option B: Encode Beta into Mask (Deferred — Invasive)

Fold per-head beta into the attention mask. This requires per-layer per-head masks instead of broadcast masks. The mask shape would change from `[n_kv, n_tokens, 1, n_stream]` to `[n_kv, n_tokens, n_head, n_stream]` per layer. This is a significant change to the graph construction and memory layout.

**Not recommended for v0.** Document as future option.

### Option C: Require Non-Flash for Non-Zero Beta (Current Behavior)

The current code enforces this requirement at two levels:
- Generic: `use_flash_attn = cparams.flash_attn && kq_b == nullptr` at `llama-graph.cpp:1883`
  disables flash whenever a beta bias tensor exists.
- Compacted-prefix: explicit `GGML_ASSERT(!cparams.flash_attn)` guards on non-zero-beta
  compacted paths at `llama-graph.cpp:2097`, `2175`, and `2393`.

**No code changes needed.** Document the tradeoff: flash is faster but selection-only; standard is slower but supports full solver with non-zero beta.

### Decision

**Adopt Option A (wait) + Option C (current non-flash requirement).**

**6b-12 (completed):** This section IS the deliverable. The flash+beta tradeoff, all three
options, and the decision are documented above. No code changes needed — current code
requires the non-flash path for non-zero beta; compacted-prefix flash+non-zero-beta is
guarded by assertions and is not a supported runtime mode.

**Files:** `docs/pr6b-self-study-implementation-plan.md` — this section
**Effort:** 0 (documentation only, completed)

---

# Part 4: Production Validation on ModelAI Workloads

## Validation Matrix

| Test | Model | Context | Target | Pipeline | Metric |
|------|-------|---------|--------|----------|--------|
| **4a. SEC Filing Extraction** | 14B-class¹ | 8K-32K token filing | 512 tokens | select, self-study | Answer accuracy vs full context |
| **4b. Long Document QA** | 14B-class¹ | 32K token document | 1024 tokens | select, self-study, OMP | ROUGE/F1 on extraction tasks |
| **4c. Multi-Filing Batch** | 14B-class¹ | 200 filings × 8K each | 512 tokens each | select (speed), self-study (quality) | Throughput (filings/min), accuracy |
| **4e. Small Model Smoke** | stories15M | 512 tokens | 64 tokens | all pipelines | No crashes, basic quality |

¹ Approved 14B-class models: Qwen2.5-14B-Instruct (preferred), DeepSeek-R1-14B (qwen2
arch, acceptable with caution — reasoning-heavy training may affect compaction quality
differently). Both available locally via ollama blobs.

### Deferred Workloads (blocked by unsupported model architectures)

| Test | Model | Blocker | Resume When |
|------|-------|---------|-------------|
| **4d. Vision Document** | Qwen3-VL-8B | M-RoPE unsupported (6b-10) | M-RoPE support implemented |

## Implementation (6b-13, completed)

### `tests/test-kv-compact-workload.cpp`

End-to-end workload test measuring logit cosine similarity and decode tok/s:
1. Load model, prefill SEC-filing-style text (batched in chunks of `n_batch`)
2. Save baseline state, capture baseline logits and decode tok/s
3. For each compression ratio (2x, 4x, 8x): restore state, run compaction
   pipeline, measure compacted logit cosine vs baseline, measure compacted
   decode tok/s
4. Auto-detect workload class from model size: W1 (<500M), W2 (1-10B), W3 (>10B)
5. Write CSV artifact when `ARTIFACT` env is set

Environment variables: `PIPELINE` (select/solver/omp), `ARTIFACT` (CSV path)

### `scripts/bench-kv-compact-workload.sh`

Benchmark script: runs all three pipelines sequentially, merges CSV outputs,
tolerates threshold misses (non-zero exit) without aborting.

```bash
# Full benchmark:
./scripts/bench-kv-compact-workload.sh models/test/Qwen3-14B-Q4_K_M.gguf
# Custom context:
N_CTX=8192 ./scripts/bench-kv-compact-workload.sh models/test/Qwen3-14B-Q4_K_M.gguf
```

### Qwen3-14B Results (Apple M2 Pro, Metal, n_ctx=4096)

Configuration: 3276 prefix tokens, live_suffix_pos0=2620 (656 live suffix tokens),
16 continuation tokens for tok/s measurement.

**Select pipeline (zero beta, earliest positions):**

| Ratio | Cosine | Threshold | Pass | Compact (ms) | Baseline tok/s | Compacted tok/s |
|-------|--------|-----------|------|-------------|----------------|-----------------|
| 2x    | 0.9988 | >= 0.95   | PASS | 481         | 13.6           | 7.2             |
| 4x    | 0.9963 | >= 0.90   | PASS | 387         | 13.6           | 8.5             |
| 8x    | 0.9896 | >= 0.85   | PASS | 341         | 13.6           | 9.4             |

**Solver pipeline (NNLS beta + V fitting, surrogate queries):**

| Ratio | Cosine | Threshold | Pass | Compact (ms) | Baseline tok/s | Compacted tok/s |
|-------|--------|-----------|------|-------------|----------------|-----------------|
| 2x    | 0.9591 | >= 0.95   | PASS | 319,332     | 14.8           | 7.1             |
| 4x    | 0.7501 | >= 0.90   | FAIL | 88,046      | 14.8           | 8.5             |
| 8x    | 0.9118 | >= 0.85   | PASS | 53,315      | 14.8           | 9.4             |

Solver is both slower (5 min vs 0.5s at 2x) and lower quality than select on
this GQA model.  Known issue: surrogate cache-key queries produce poor fits on
GQA architectures (n_head=40, n_head_kv=8, GQA ratio 5:1).

**Solver quality inversion (4x worse than 8x):** The solver produces 0.7501 at
4x but 0.9118 at 8x — non-monotonic.  Hypothesis: at 4x the NNLS selects 655
tokens and overfits the surrogate query scores, producing beta weights that
distort the attention distribution.  At 8x, fewer tokens (327) give the solver
less room to overfit.  This inversion does not occur in the select pipeline
(monotonic: 0.9988 > 0.9963 > 0.9896).  Investigation deferred — solver on GQA
is not the production path.

**OMP pipeline: NOT FEASIBLE at production scale.**

OMP was killed after >23 minutes of CPU time without completing the first (2x)
compression ratio.  Root cause: `llama_kv_compact_select_omp` runs per-head
(320 heads = 40 layers × 8 n_head_kv).  With default `nnls_interval=1`, each
greedy step solves NNLS via Cholesky decomposition on a growing matrix.
At iteration i, NNLS cost is O(n_q × i^2 + i^3).  For t=1310 (2x) across 320
heads, total work ≈ 7×10^13 FLOPs — estimated ~2 hours on M2 Pro single-thread.

OMP passes on W1 smoke tests (stories15M, 204 prefix tokens, t=102) where
per-head cost is negligible.  Production use requires either:
1. Increased `nnls_interval` (e.g. 8 or 16) to amortize NNLS cost
2. Multi-threaded per-head parallelism
3. Reduced `max_queries` to limit n_q

### Qwen3-30B-A3B-Instruct Results (Apple M2 Pro, Metal, n_ctx=4096)

MoE model: 30.5B total params, 128 experts, 8 active, GQA 8:1 (n_head=32, n_head_kv=4).

**Select pipeline:**

| Ratio | Cosine | Threshold | Pass | Compact (ms) | Baseline tok/s | Compacted tok/s |
|-------|--------|-----------|------|-------------|----------------|-----------------|
| 2x    | 0.9998 | >= 0.95   | PASS | 298         | 31.6           | 13.8            |
| 4x    | 0.9995 | >= 0.90   | PASS | 229         | 31.6           | 18.0            |
| 8x    | 0.9990 | >= 0.85   | PASS | 208         | 31.6           | 20.6            |

MoE compacts better than dense models: 0.9990 at 8x (vs 0.9896 for dense 14B).

### Compacted decode is slower at 4K context

Both models show compacted decode slower than baseline (Qwen3-14B: 13.6 → 9.4
at 8x; Qwen3-30B-A3B: 31.6 → 20.6 at 8x).  This is expected at small context
(4096 tokens):
- The compacted prefix execution path adds overhead: dual-pass attention
  (compacted + live), tensor concatenation, extra mask construction.
- At 4K context the KV cache is small enough that memory bandwidth is not the
  bottleneck — the dual-pass overhead dominates.
- P5b benchmarks showed compaction speedup materializes at larger contexts
  (16K+) where KV memory bandwidth becomes the bottleneck.
- Validation at 16K/32K context is deferred to production integration testing.

**Files:**
- `tests/test-kv-compact-workload.cpp` — Workload test
- `scripts/bench-kv-compact-workload.sh` — Benchmark script
- `tests/CMakeLists.txt` — Register workload test (no auto-run; requires model)

**Effort:** 2-3 days

## Acceptance Criteria

**[Reviewer-2 Condition R2-2]** Quantitative pass/fail thresholds per test case and compression ratio.

### Per-Test Thresholds

| Test | Compression | Metric | Pass Threshold | Source |
|------|-------------|--------|----------------|--------|
| **4a/4b. Logit quality** | 2x | Logit cosine similarity | >= 0.95 | Calibrated from Qwen3-14B, 30B-A3B data |
| **4a/4b. Logit quality** | 4x | Logit cosine similarity | >= 0.90 | Calibrated from Qwen3-14B, 30B-A3B data |
| **4a/4b. Logit quality** | 8x | Logit cosine similarity | >= 0.85 | Calibrated from Qwen3-14B, 30B-A3B data |
| **4c. Multi-Filing** | 2x | Throughput vs full context | >= 1.5x speedup | Minimum viable benefit |
| **4c. Multi-Filing** | 5x | Accuracy vs full context | >= 0.88 | Must not regress |
| **4e. Small Model Smoke** | any | No crashes | 0 crashes | Hard requirement |

**Metric change:** The plan originally specified ROUGE-L and F1, which require
autoregressive text generation and task-specific evaluation.  The implemented
test uses logit cosine similarity — a single-token metric that directly measures
attention distribution preservation without generation overhead.  Thresholds
were calibrated from observed select pipeline results across two architectures
(dense Qwen3-14B, MoE Qwen3-30B-A3B).  ROUGE-L/F1 validation is deferred to
production integration testing where full generation is available.

### Self-Study vs Surrogate Quality Delta

Self-study queries (Part 1) must outperform cache-key surrogates (existing K-as-Q baseline) to justify the additional generation cost:

| Compression | Metric | Required Delta |
|-------------|--------|---------------|
| 2x | ROUGE-L (4a) | self-study >= surrogate (no regression) |
| 5x | ROUGE-L (4a) | self-study > surrogate by >= 2 percentage points |
| 10x | ROUGE-L (4a) | self-study > surrogate by >= 5 percentage points |
| 5x | Logit cosine similarity | self-study > surrogate by >= 0.02 |

### Reference Baselines from Paper

From arXiv:2602.16284, Table 2 (QuALITY benchmark, Llama-3-8B-Instruct):
- 2x compression: ~71.5% accuracy
- 5x compression: ~70% accuracy
- 10x compression: ~67% accuracy

Our implementation targets may differ due to model differences (Qwen3-14B vs Llama-3-8B) and task differences (SEC filing extraction vs QuALITY multiple-choice). These paper numbers serve as sanity checks, not direct targets.

### Workload Classification

Each benchmark must specify its workload class to ensure reproducibility:

| Class | Model Size | Prefix Length | Description |
|-------|-----------|--------------|-------------|
| **W1** | < 100M params | < 512 tokens | Unit/smoke tests (stories15M) |
| **W2** | 1-10B params | 2K-8K tokens | Standard workloads (Qwen3-8B) |
| **W3** | 10-30B params | 8K-32K tokens | Production workloads (Qwen3-14B) |

### Failure Handling

If a test fails its threshold:
1. Log the actual metric value and the threshold
2. Mark the test as FAIL (not SKIP)
3. Continue running remaining tests (do not abort the suite)
4. Aggregate results into a summary table with PASS/FAIL per test per compression ratio

**[Reviewer-2 Minor Note]** The thresholds above are conservative initial estimates. After the first benchmark pass on real SEC filings with Qwen3-14B, expect to tighten thresholds based on actual performance. The self-study vs surrogate delta thresholds (>= 2pp at 5x) are calibrated to be achievable given the paper's results but should increase if self-study proves stronger than expected.

---

# Part 5: Upstream-Ready Sanitization

## Current State

- **Source code:** Clean. Zero "modelai" references in `src/` or `include/`. All ModelAI references confined to `docs/modelai-*.md`.
- **Naming:** 95% consistent. `llama_kv_compact_*` prefix for solver/pipeline, `llama_compacted_prefix_*` for storage/execution. One duplicate struct: `llama_kv_compact_omp_opts` declared in both `llama-kv-compact-select.h` and `llama-kv-compact-pipeline.h`.
- **Public API:** All compaction functions are internal (no `LLAMA_API`). Not exposed in `include/llama.h`.
- **Tests:** 6 test files using standard llama.cpp framework. Properly labeled.
- **Build:** No conditional compilation flags. Always compiled into `llama` library.
- **Docs:** Entirely ModelAI-specific. Zero upstream-ready documentation.

## Changes Required

### 5a. Fix Duplicate Struct

**Problem:** `llama_kv_compact_omp_opts` is declared in both `src/llama-kv-compact-select.h:19` and forward-declared in `src/llama-kv-compact-pipeline.h:40`.

**6b-14 fix (completed):** Removed the dead forward declaration from `pipeline.h`.
No function signature in `pipeline.h` references `omp_opts`, so neither a forward
declaration nor an include is needed — the type is only used internally in `pipeline.cpp`
which already includes `llama-kv-compact-select.h`.

**Files:** `src/llama-kv-compact-pipeline.h` — removed dead forward declaration
**Effort:** 15 minutes (completed)

### 5b. Long-Context Benchmark Campaign (6b-15)

**Problem:** All existing benchmarks run at 4K context where compacted decode is
slower than baseline (dual-pass attention overhead dominates at small KV sizes).
The throughput crossover point — where KV memory bandwidth savings exceed the
dual-pass overhead — has not been measured.  Without this data we cannot claim
decode speedup on the resume or in product positioning.

**Goal:**
1. Find the context length at which compaction delivers net positive decode
   throughput for each supported architecture
2. Measure quality degradation at longer contexts via logit cosine AND
   QuALITY multiple-choice accuracy (paper-aligned)
3. Produce self-describing artifacts (`manifest.json` + `results.csv`) that
   ModelAI can ingest downstream without re-running benchmarks

**Out of scope for this repo:** Supabase migrations, admin API routes, admin
console UI, upload services.  Those belong in the ModelAI repo and will consume
the artifact contract defined here.

---

#### A. Test Matrix

**Models (6 total — all compaction-compatible):**

| Model | Params | Architecture | KV Path | Rationale |
|-------|--------|-------------|---------|-----------|
| Qwen3-30B-A3B-Instruct | 30B (3B active) | MoE, GQA 8:1 | standard | Production model |
| Qwen3-14B | 14B | Dense, GQA 5:1 | standard | Dense baseline |
| Qwen3-8B | 8B | Dense, GQA | standard | Small/fast iteration |
| Qwen2.5-14B-Instruct | 14B | Dense, GQA | standard | Prior-gen comparison |
| DeepSeek-R1-Distill-Qwen-14B | 14B | Dense, GQA | standard | Reasoning variant |
| Gemma-3-12B-IT | 12B | iSWA (mixed) | iSWA base | Tests iSWA compaction path |

**Support note:** only **base-cache compaction** on iSWA layouts is active in
this matrix. SWA-layer compaction remains unsupported in V0.

**Blocked models (cannot test):**
- Vision (Qwen3-VL) — M-RoPE guard, no fix planned
- Qwen3.5-35B-A3B — hybrid recurrent, needs `llama_memory_hybrid`
- bge-m3 — embedding model (no generation)

**Feasibility matrix (Apple M2 Pro, 32GB unified memory):**

| Model | 4K | 8K | 16K | 32K |
|-------|----|----|-----|-----|
| Qwen3-30B-A3B (19GB weights) | active | active | active | stretch (~22GB total) |
| Qwen3-14B (8.6GB weights) | active | active | active | active (~14GB total) |
| Qwen3-8B (4.9GB weights) | active | active | active | active (~10GB total) |
| Qwen2.5-14B (8.4GB weights) | active | active | active | active (~14GB total) |
| DeepSeek-R1-14B (8.4GB weights) | active | active | active | active (~14GB total) |
| Gemma-3-12B (7.6GB weights) | active | active | active | stretch (~16GB total) |

- **active** = must produce a valid row
- **stretch** = attempt, emit row with `error_text` if OOM

**Context sizes:** 4096, 8192, 16384, 32768
**Compression ratios:** 2x, 4x, 8x
**Pipelines:**
- `baseline` — no compaction, full-context decode (reference measurement)
- `select` — primary compaction pipeline
- `solver` — NNLS beta + V fitting
- `omp` — smoke only (W1); known infeasible at production scale
- `self_study` — **deferred**; API exists (`llama-kv-cache.h:204`,
  `compacted_prefix_self_study_from_live_kv`) but is not wired into the
  workload test dispatch or pipeline dispatcher. Requires `llama_context *`
  parameter threading. Will be implemented as **6b-15b** in the same phase
  immediately after the baseline/select/solver benchmark contract is stable.

**Total active data points (minimum before W1 OMP smoke rows):**
- baseline rows: `6 models × 4 contexts = 24`
- select rows: `6 models × 4 contexts × 3 ratios = 72`
- solver rows: `6 models × 4 contexts × 3 ratios = 72`
- **total active minimum = 168 rows**

`72` is the minimum for a **single compacted pipeline**. The full active
campaign includes baseline + select + solver. OMP W1 smoke rows are additional.

---

#### B. Pipeline Definitions

**`baseline`:** Run prefill + decode burst at full context with NO compaction.
Emit a CSV row with `pipeline=baseline`, `compression_ratio=1`,
`compacted_tokens=0`, all compaction/solver/query time fields = `0`,
`throughput_delta_pct=0`. This row provides the reference decode tok/s for
computing throughput delta in compacted rows.

**`select`:** Top-k position selection (earliest positions by aggregate
attention score). Zero beta weights. Calls
`compacted_prefix_select_from_live_kv()`.

**`solver`:** NNLS beta fitting + least-squares V fitting with surrogate
cache-key queries. Calls `compacted_prefix_fit_from_live_kv()`.

**`omp`:** Orthogonal Matching Pursuit selection + solver fitting. Calls
`compacted_prefix_omp_from_live_kv()`. Known infeasible at production scale
(>23 min for 2x on 14B). Run only in W1 smoke tests.

---

#### C. Required Metrics Per CSV Row

**Identity:** `schema_version`, `run_id`, `workload_id`, `workload_name`,
`dataset_id`, `model_name`, `model_params_b`, `architecture`, `quantization`,
`backend`, `flash_mode`, `pipeline`, `n_ctx`, `compression_ratio`

**Latency / throughput:** `prefill_ms`, `prefill_tok_s`, `compaction_time_ms`,
`query_generation_time_ms`, `solver_time_ms`, `first_token_ms`,
`baseline_decode_tok_s`, `compacted_decode_tok_s`, `throughput_delta_pct`

**KV / memory:** `prefix_tokens`, `compactable_tokens`, `live_suffix_tokens`,
`compacted_tokens`, `continuation_tokens`, `active_n_kv`

**KV byte metrics (nullable — emit `""` when unavailable):**
`allocated_kv_bytes`, `reclaimed_kv_bytes` — No `llama_kv_cache` API currently
exposes byte-level allocation metrics. These columns are reserved for future
use. Emit empty string `""` until a byte-reporting API is added. Downstream
ingestion must treat empty values as null.

**Quality / safety:** `logit_cosine`, `task_metric_name`, `task_metric_value`,
`quality_correct`, `quality_total`, `quality_accuracy`,
`quality_baseline_accuracy`, `threshold_name`, `threshold_value`, `pass`,
`fallback_used`, `fallback_reason`, `crash`, `error_text`, `artifact_path`

**Nullable quality columns:** `quality_*` columns are empty when QuALITY
evaluation is not run for this row (e.g., synthetic text prefill at 16K-32K).

**Throughput formula:**
`throughput_delta_pct = ((compacted_decode_tok_s - baseline_decode_tok_s) / baseline_decode_tok_s) * 100`

Full CSV column contract: see `MODELAI_LLAMA_CPP_LONGCTX_CSV_WRITER_CONTRACT.md`

**Column source mapping (engine-side contract):**

| Column | Source rule |
|--------|-------------|
| `workload_id` | Static workload registry in `test-kv-compact-longctx.cpp` (`W1`, `W2`, `W3`) |
| `workload_name` | Static workload registry display name keyed by `workload_id` |
| `dataset_id` | Static workload registry dataset key (`quality-validation`, `synthetic-sec`, `quality-concat`, etc.) |
| `model_params_b` | Static model registry keyed by `model_name` |
| `flash_mode` | Runtime mode emitted by the engine: `off`, `on`, or `forced_off_by_beta` |
| `threshold_name` | Deterministic threshold table key (`<workload_id>:<metric>:<ratio>`) |
| `threshold_value` | Threshold table value resolved from `threshold_name` |
| `artifact_path` | Row-local path under `bench-results/<run_id>/...` |
| `compactable_tokens` | `prefix_tokens - live_suffix_tokens` |
| `live_suffix_tokens` | Explicit runner configuration for the preserved live suffix |
| `prefill_ms` / `prefill_tok_s` | Measured from the baseline or compacted prefill step in the benchmark runner |
| `first_token_ms` | Time from decode start to first emitted continuation token |
| `allocated_kv_bytes` / `reclaimed_kv_bytes` | Emit concrete values when available, otherwise empty string `\"\"` |
| `quality_*` fields | Computed by the benchmark runner from QuALITY MC scoring when enabled, otherwise empty string `\"\"` |
| `fallback_*` fields | Populated from the engine return path; explicit `false` / empty string when no fallback occurred |
| `baseline_decode_tok_s` | Self-measured by the benchmark runner within the same test invocation, before compaction is applied |
| `compacted_decode_tok_s` | Self-measured by the benchmark runner after compaction, using the same continuation burst length |
| `throughput_delta_pct` | Computed by the runner: `((compacted_decode_tok_s - baseline_decode_tok_s) / baseline_decode_tok_s) * 100` |
| `error_text` | Explicit engine/runtime failure text; empty string on success |

---

#### D. QuALITY Multiple-Choice Evaluation Protocol (Paper-Aligned)

The MIT paper (arXiv:2602.16284, Table 2) evaluates compaction quality on the
QuALITY benchmark by measuring multiple-choice answer accuracy.

**Data format:** Each entry in `quality-validation.jsonl` contains:
- `article` (string) — full article text (approximately 2.7K-8.5K tokens,
  tokenizer-dependent)
- `question` (string) — a comprehension question
- `options` (array of 4 strings) — answer choices
- `answer` (int, 0-3) — index of the correct option
- `hard` (bool) — whether this is a "hard" question

**Prompt template (verbatim):**

```
{article}

Question: {question}

A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

Answer:
```

No chat template is applied. The prompt is raw text. The article is prefilled
as the context; the question + options are appended as a suffix after compaction.

**Scoring method:** Logit comparison over exactly 4 answer tokens.
1. After decoding the prompt (ending with `"Answer:"`), extract logits at the
   last position.
2. Look up token IDs for `" A"`, `" B"`, `" C"`, `" D"` (space-prefixed
   single letters) in the model vocabulary. If a model's tokenizer does not
   produce single-token results for these strings, fall back to `"A"`, `"B"`,
   `"C"`, `"D"` (no space).
3. The predicted answer is `argmax(logits[tok_A], logits[tok_B], logits[tok_C],
   logits[tok_D])`.
4. Map argmax index: 0→A, 1→B, 2→C, 3→D.
5. Compare to `answer` field: correct if `argmax_index == answer`.

**No generation is performed.** This is a single forward pass after the prompt,
reading logits at the final position. This avoids tokenizer-dependent generation
artifacts and is deterministic.

**Baseline comparison:** Run the same prompt at full context (no compaction)
using identical scoring. This produces `quality_baseline_accuracy`.

**CSV encoding:**
- `quality_correct`: count of questions answered correctly (compacted)
- `quality_total`: count of questions evaluated
- `quality_accuracy`: `quality_correct / quality_total` (emitted by engine)
- `quality_baseline_accuracy`: accuracy without compaction (same articles)
- `task_metric_name`: `"quality_accuracy"` for QuALITY rows
- `task_metric_value`: same as `quality_accuracy`

Paper reference baselines (Llama-3-8B-Instruct, Table 2):
- 2x compression: ~71.5% accuracy
- 5x compression: ~70% accuracy
- 10x compression: ~67% accuracy
- Full context (no compaction): ~73% accuracy

**Data sources:**
- QuALITY validation articles (`tests/data/quality-validation.jsonl`, 115
  articles, approximately 2.7K-8.5K tokens depending on tokenizer) for 4K-8K
  natural-text prefill
- Concatenated QuALITY articles or synthetic SEC-filing text for 16K-32K tests
  (QuALITY MC evaluation runs only on single-article prefills where the
  question is meaningful)

---

#### E. Artifact Structure

Each benchmark campaign emits one self-describing directory:

```text
bench-results/<run_id>/
  manifest.json       — versioned run metadata (branch, commit, hardware, models)
  results.csv         — stable CSV contract (all metrics above)
  stdout.log          — full stdout capture
  stderr.log          — full stderr capture
  env.txt             — environment snapshot
```

`run_id` format: `YYYYMMDD-HHMMSS-<shortsha>-<machine>`
Example: `20260312-233015-c985e85a-m2pro32`

Full manifest contract: see `MODELAI_LLAMA_CPP_BENCHMARK_SPEC.md`

---

#### F. C++ Test Binary

New file: `tests/test-kv-compact-longctx.cpp`
- Accepts model path, context size, pipeline, compression ratio via CLI/env
- `pipeline=baseline`: prefill + decode only, no compaction, emit reference row
- `pipeline=select|solver|omp`: prefill, compact, measure delta vs baseline
- QuALITY evaluation mode (`QUALITY_EVAL=1`): after prefill/compaction, append
  question+options suffix, score via logit comparison (section D protocol)
- Emits one CSV row per run following the stable column contract
- Loads QuALITY articles from `tests/data/quality-validation.jsonl`
- Build-only in CMake (no auto-run — requires model files)

---

#### G. Benchmark Driver Script

New file: `scripts/bench-kv-compact-longctx.sh`
- Iterates: models × context sizes × ratios × pipelines (baseline + select +
  solver; omp only for W1)
- Skips stretch cells that OOM; emits row with `crash=true`, `error_text`
- Emits `manifest.json` + `results.csv` into `bench-results/<run_id>/`
- Captures stdout/stderr logs and env snapshot
- Summary table at end with crossover annotation per model
- Unsupported rows are explicit (emitted with `pass=false`, `error_text`),
  not silent skips

---

#### H. Definition of Done — Closure Gate

6b-15 is complete only when ALL of the following are true:

1. `test-kv-compact-longctx.cpp` builds and runs
2. `bench-kv-compact-longctx.sh` produces a valid run directory with
   `manifest.json` + `results.csv`
3. Every run emits CSV rows in the documented stable column contract
4. **At least one W2 or W3 row satisfies ALL of:**
   - Real model >= 1B parameters
   - Real long-context input (n_ctx >= 4096)
   - `pass=true` (quality threshold met)
   - `crash=false`
   - Throughput delta recorded (may be negative — the benchmark's purpose is
     to find the crossover, not to assert it exists)
5. Baseline rows exist for each (model, n_ctx) combination
6. Artifact outputs are self-describing enough for downstream ingestion
7. Unsupported rows are explicit, not silent skips
8. No required downstream field depends on scraping stdout

---

#### I. Expected Outcomes

**Throughput crossover hypothesis:** 8K-16K for dense 14B models, possibly
lower for 30B MoE (larger KV per layer reduces memory bandwidth headroom).

**Quality at long context:** Logit cosine may degrade at 32K if the compactable
prefix contains information the model needs to attend to.  QuALITY articles are
designed to require full-context comprehension — this is a stress test.

---

#### J. Handoff to ModelAI

ModelAI will later:
1. Parse `manifest.json` and validate `schema_version`
2. Ingest `results.csv` into Supabase
3. Expose admin API routes for stats, runs, crossover, regressions
4. Render KV Compaction tab on admin benchmarks page

The engine repo treats `manifest.json` + `results.csv` as the source of truth.
ModelAI implementation details are specified in a separate plan
(`MODELAI_ADMIN_CONSOLE_KV_BENCHMARK_IMPLEMENTATION_PLAN.md`).

---

#### K. Files

- `tests/test-kv-compact-longctx.cpp` — Long-context benchmark test
- `scripts/bench-kv-compact-longctx.sh` — Multi-model benchmark driver
- `tests/CMakeLists.txt` — Register (build-only, no auto-run)
- `docs/pr6b-self-study-implementation-plan.md` — Results table (post-run)

**Effort:** 2-3 days

### 5b-1. Self-Study Benchmark Extension (6b-15b)

**Problem:** `6b-15` intentionally stabilizes the benchmark contract with
`baseline`, `select`, and `solver` first. The benchmark harness does not yet
dispatch `self_study`, so the strongest query path is not represented in the
active matrix.

**Goal:** Promote `self_study` from deferred to active once the long-context
runner can execute it and emit rows in the same artifact contract.

**Deliverables:**
1. Wire `self_study` into `tests/test-kv-compact-longctx.cpp`
2. Wire `self_study` into `scripts/bench-kv-compact-longctx.sh`
3. Update `manifest.json` and CSV contracts so `self_study` is an active
   pipeline, not a deferred note
4. Run the same `W1/W2/W3` closure gates for `self_study`
5. Record `self_study` rows in the same `bench-results/<run_id>/results.csv`
   schema as `baseline`, `select`, and `solver`

**Dependencies:** `6b-15` contract and artifact layout must be stable first.

**Non-goals:** No new admin-console/Supabase work; this remains engine-side
artifact generation only.

**Effort:** 1-2 days

### 5c. Public API Decision (deferred, no slice)

**Decision for v0:** Keep compaction as internal API. Do NOT add `LLAMA_API` markers. Rationale:
- API is still evolving (self-study adds `llama_context *` parameter)
- Upstream has not reviewed the design
- Internal API can change freely between releases

**For upstream submission (future):** Add a minimal public API:
```cpp
LLAMA_API bool llama_kv_cache_compact(
        struct llama_context * ctx,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        const char * method);  // "select", "omp", "self-study"
```

**No code changes in this PR.** Document the API strategy.

### 5c. Upstream-Ready Documentation

Create algorithm and integration documentation suitable for upstream review.

**New files:**
- `docs/kv-compaction-algorithm.md` — Algorithm overview referencing arXiv:2602.16284. Covers: selection, beta fitting (NNLS), V fitting (least-squares), query generation (cache-key surrogates and self-study), compacted prefix execution. No ModelAI product references.
- `docs/kv-compaction-integration.md` — Internal integration guide: how the compacted prefix store connects to llama_kv_cache, how graph construction injects compacted K/V/mask/beta, how set_input populates data. File map of all compaction source files.

**Effort:** 1-2 days

### 5d. Test Hardening

**Problem:** No explicit backend-variant tests. No negative tests for unsupported configurations.

**Solution:**
- Add negative test: attempt compaction on SWA-only cache, verify graceful failure
- Add negative test: attempt compaction with empty prefix, verify no crash
- Add test annotation for Metal vs CUDA vs CPU backends

**Files:**
- `tests/test-kv-compacted-prefix.cpp` — Add negative test cases
- `tests/CMakeLists.txt` — Add backend labels if needed

**Effort:** 0.5 day

---

# Part 6: Compaction Runtime Integration

## Why This Is Needed

**[P1 Finding]** Internal compaction code exists and is functionally complete, but the product layer cannot consume it as a supported feature. The server's `/props` endpoint and `/metrics` endpoint both hardcode `compacted_prefix.available = false` and `modelai_compacted_prefix_available = 0`. This means the ModelAI product orchestrator cannot detect or enable compaction at runtime — the feature is invisible to the product.

**[P1 Finding]** Flash-attention compatibility for compacted KV is only captured for the zero-beta special case (`build_attn()` at `llama-graph.cpp:1883` only uses FA when `kq_b == nullptr`). The documentation and the server's `supports_additive_kq_b: false` flag both hardcode this limitation, but there is no runtime path that automatically selects the right attention mode when compaction is active.

**[P2 Finding]** The compacted-prefix store now accepts aligned quantized K types (`llama-kv-compacted-prefix.cpp:15-24` — `is_supported_compacted_type` accepts Q8_0/Q4_0 when head_dim is block-aligned), but the unit test at `test-kv-compacted-prefix.cpp:346` still expects quantized K to be rejected. This spec drift will cause test failures once compacted quantized K is exercised.

## Changes Required

### 6a. Wire Real Compaction State into Server Capability Reporting

**Problem:** `server-context.cpp` lines 109-114 hardcode:
```cpp
{ "compacted_prefix", {
    { "available",            false },
    { "enabled",              false },
    { "requires_non_flash",   true },
    { "last_fallback_reason", "feature_unavailable" },
} },
```
And lines 145-151 hardcode metrics to `false`/`nullptr`. The `/metrics` endpoint at line 3513 hardcodes `modelai_compacted_prefix_available = 0`.

**Solution:**
1. Add a `bool compaction_available()` method to `server_context` that checks whether the loaded model supports compaction. **Required checks:** non-SWA primary cache, standard attention, non-MLA architecture, non-hybrid-recurrent (Mamba layers have no KV), non-lfm2 (non-standard Q names would cause Q-capture to silently fail). For iSWA: report available=true with a `compaction_supported_layers` count (base layers only).
2. Add a `bool compaction_enabled()` method that checks whether compaction is currently active on any slot.
3. Wire these into `build_modelai_server_capabilities()` and `build_modelai_runtime_summary_from_metrics()`.
4. Update the Prometheus metric to reflect real state.
5. Add `compaction_method` field reporting current pipeline mode: `"none"`, `"select"`, `"omp"`, `"self-study"`.

**Files:**
- `tools/server/server-context.cpp` — Replace hardcoded false with runtime queries (lines 109-114, 145-151, 3513-3519)
- `tools/server/server-context.h` — Add `compaction_available()`, `compaction_enabled()` methods
- `src/llama-kv-cache.h` — Add `has_compacted_prefix()` query method
- `src/llama-kv-cache.cpp` — Implement `has_compacted_prefix()`

**Merge gate:** `/props` returns `compacted_prefix.available = true` when model supports it. After compaction runs, `compacted_prefix.enabled = true`. Verified in integration test.

**Effort:** 1 day

### 6b. Flash Attention + Compaction Automatic Mode Selection

**Problem:** `build_attn()` at `llama-graph.cpp:1883` uses the rule `use_flash_attn = cparams.flash_attn && kq_b == nullptr`. When compaction is active with non-zero beta, `kq_b != nullptr` and flash attention is silently disabled. There is no observability for this — the user requested flash attention but doesn't know it's not being used.

**Solution:**
1. Add a `compaction_flash_attn_override` counter to metrics: how many layers fell back from flash to standard attention due to non-zero beta.
2. Log at INFO level when flash attention is disabled due to compacted beta: `"flash_attn: disabled for layer %d (non-zero compacted beta)"`.
3. Surface the override state in `/props` under `flash_attention.compaction_override_count`.
4. Document the tradeoff in the capability response: when `zero_beta_only = true`, flash attention works; when `zero_beta_only = false`, standard attention is used with full solver quality.

**Files:**
- `src/llama-graph.cpp` — Add logging when FA is overridden by compaction beta (near line 1883)
- `tools/server/server-context.cpp` — Surface override count in capabilities
- `docs/kv-compaction-algorithm.md` — Document the flash/beta tradeoff

**Effort:** 0.5 day

### 6c. Fix Quantized K Compaction Spec Drift

**Problem:** The compacted-prefix store was updated in PR-6 coverage expansion to accept block-aligned quantized types (`llama-kv-compacted-prefix.cpp:15-24`):
```cpp
bool is_supported_compacted_type(ggml_type type, uint32_t head_dim_k, uint32_t head_dim_v) {
    if (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F32) return true;
    const int64_t blk = ggml_blck_size(type);
    if (blk <= 0) return false;
    return (head_dim_k % blk == 0) && (head_dim_v % blk == 0);
}
```

But the unit test at `test-kv-compacted-prefix.cpp:346` still asserts rejection:
```cpp
if (!check(expect_throw([&]() { quantized_store.configure_seq(1, 8, { 0, 4 }, -1); }),
    "quantized K types should be rejected in P2", rc)) return rc;
```

This test will fail as soon as Q8_0 with block-aligned head dims is exercised, since the store now accepts it.

**Solution:**
1. Update the test to reflect the new reality: quantized K with block-aligned heads should SUCCEED.
2. Add a test case for quantized K with NON-block-aligned heads (e.g., head_dim=17 with Q8_0 block_size=32) — this should still fail.
3. Update `docs/modelai-fork-summary.md` line 186 to change "Quantized K compaction | Unsupported" to "Quantized K compaction | Supported (block-aligned head dims)".
4. **[MAJOR] Verify full K extraction → solver → store → exec pipeline for Q8_0 K end-to-end.** The store accepts Q8_0 (`is_supported_compacted_type`), but does `compacted_prefix_copy_k_head_f32()` in `llama-kv-cache.cpp` correctly dequantize quantized K? It calls `ggml_backend_tensor_get()` which returns raw bytes — explicit dequantization is needed. If the solver receives raw quantized bytes interpreted as F32, results will be garbage. **Must trace the full path before claiming "Supported."**
5. Add quality test: compaction with Q8_0 K cache produces acceptable reconstruction error (end-to-end, not just store acceptance).

**Files:**
- `tests/test-kv-compacted-prefix.cpp` — Fix quantized K test (line 346), add non-aligned negative test
- `docs/modelai-fork-summary.md` — Update support matrix (line 186)
- `tests/test-kv-compact-quality.cpp` — Add quantized K quality test

**Effort:** 0.5 day

### 6d. Update Fork Summary Support Matrix

**Problem:** `docs/modelai-fork-summary.md:186-193` lists several items as "Unsupported" that are now supported or partially supported after PR-5b and PR-6:

| Item | Listed Status | Actual Status |
|------|--------------|---------------|
| Quantized K compaction | Unsupported | Supported (block-aligned) |
| Flash-attention compaction path | Unsupported | Supported (zero-beta) |
| Quantized V compaction | Unsupported | Supported (non-transposed) |
| Public/server compacted-prefix enablement | Unsupported | Will be supported after 6a |

**Solution:** Update the matrix to reflect reality. Add a "Conditions" column.

**Files:**
- `docs/modelai-fork-summary.md` — Rewrite support matrix (lines 180-193)

**Effort:** 0.5 hour

---

# Part 7: Server/Runtime Hardening

## Why This Is Needed

KV compaction is the core value proposition, but the compaction pipeline runs inside `llama-server`. If the server itself is unreliable — crashes on malformed tool calls, produces incorrect structured output, leaks memory under repeated schema requests, or has poor prompt-eval performance — then the compaction feature is unusable in production.

This is a separate but complementary track to KV compaction. It addresses llama.cpp server-side bugs and gaps that directly affect ModelAI's ability to ship a reliable local runtime.

## Upstream Issue Status

These upstream issues were verified against the current GitHub state as of 2026-03-11:

| Issue | Title | Upstream Status | Still Relevant? | Action |
|-------|-------|-----------------|-----------------|--------|
| **#10732** | `json_schema` response_format not working | **OPEN** | **YES — HIGH** | Fix in fork: `json_schema` type broken while `json_object` works |
| **#17391** | Segfault under repeated structured output | **CLOSED** (stale-bot) | **YES — MEDIUM** | No confirmed fix; validate under load |
| **#16710** | Server crash on faulty tool call | **CLOSED** (stale-bot) | **YES — HIGH** | No fix merged; defensive parsing needed |
| **#4989** | `cache_prompt` fills KV / batch decode failures | **CLOSED** (stale-bot) | **LOW** | Architecture changed; keep awareness |
| **#12237** | Prompt eval 5x slower than Ollama | **CLOSED** (stale-bot) | **LOW** (CUDA-specific) | Not relevant for Metal builds |
| **#14697** | Tool calls in `content` instead of `tool_calls` | **CLOSED** (stale-bot) | **YES — HIGH** | Dual-path parsing or normalization |
| **#15389** | Performance drop when tools enabled | **CLOSED** | **YES — MEDIUM** | Benchmark validation needed |
| **#13197** | Poor prompt-processing thread utilization | **CLOSED** | **NO** | Misunderstanding by reporter |

### Additional High-Impact Upstream Issues Discovered

Independent GitHub scan found these additional issues worth fixing in the fork:

**Tier 1 — Fix in Fork (High Impact):**

| Issue | Title | Why It Matters for ModelAI |
|-------|-------|---------------------------|
| **#11970** | KV cache truncated incorrectly on /v1/chat/completions | Silent context loss mid-conversation directly collides with compaction — solver makes bad decisions on truncated KV |
| **#19716** | JSON schema conversion fails on typeless nodes | Real-world tool schemas (LangChain, Pydantic) use typeless properties; server rejects them with 400 |
| **#19391** | `ensure_ascii=true` garbles CJK tool arguments | Unicode corruption on second+ tool-call turn; root cause in `common/chat-parser.cpp` |
| **#18183** | Hermes 2 Pro parser accepts JSON text as tool call | Normal assistant text containing JSON triggers phantom tool invocations; safety issue for agents |
| **#19758** | Reverse port not closed after SSE stream | TCP TIME_WAIT blocks server restart; critical for local runtime users start/stop frequently |
| **#17387** | `/slots/0?action=erase` hangs indefinitely | Slot management API broken; HTTP response never sent. Critical for programmatic context management |
| **#19794** | Hybrid model prompt cache forces full re-processing | SWA/hybrid models invalidate context checkpoint on every turn; destroys prompt cache effectiveness |

**Tier 2 — Fix in Fork (Medium-High Impact):**

| Issue | Title | Why It Matters for ModelAI |
|-------|-------|---------------------------|
| **#20260** | Parser fails when model outputs text before `<tool_call>` | Thinking models (Qwen3.5) + tool calling is the dominant agent pattern; parser must handle pre-tag content |
| **#20281** | Server 500 on custom chat template (regression) | Custom templates for compaction-aware prompting would break; recent regression with clear bisect data |
| **#19068** | Grammar trigger loop causes memory leak + gibberish | Long-running daemon enters corrupted state requiring restart; unacceptable for product runtime |
| **#12171** | llama-server inference 3x slower than llama-cli | Server overhead erodes user confidence; HTTP/slot dispatch bottleneck |

**Tier 3 — Monitor Upstream:**

| Issue | Title | Why It Matters |
|-------|-------|---------------|
| **#18310** | Race condition: missing synchronize() after async copy | Concurrency correctness under parallel load |
| **#6685** | Parallel server crash with defrag enabled | Defrag is related to compaction; crash path may be reachable |
| **#9492** | First query extremely slow | Bad first impression for local runtime |
| **#4218** | Grammar sampling performance (roadmap) | Structured output latency degrades all schema/tool responses |

### Additional Issues from March 2026 GitHub Scan

**Tier 1 — Fix in Fork (CRITICAL, discovered 2026-03-11):**

| Issue | Title | Why It Matters for ModelAI |
|-------|-------|---------------------------|
| **#19679** | Random crash on Apple Metal — grammar stack empty | `llama_grammar_accept_token` throws on empty grammar stack; M4 Mac Mini + Qwen3-Coder-Next + FA + jinja. Random crashes during agentic tool-call generation. **Affects PRIMARY deployment target.** |
| **#19304** | Crash at 86K context / 50+ tool calls (grammar stack) | Same root cause as #19679 — grammar state machine fails on certain token pieces at grammar rule boundaries. Long agentic sessions will crash. |
| **#19051** | Server fails open when JSON schema grammar parsing fails | When grammar parsing fails, server logs error but continues generation **unconstrained** (HTTP 200). Silent loss of structured output guarantees. Plan's 7a item 4 already covers intent — add issue # for traceability. |
| **#19010** | Stack overflow in std::regex from crafted JSON Schema `pattern` | `_visit_pattern()` in `json-schema-to-grammar.cpp` uses `std::regex` which has unbounded stack depth. Crafted patterns cause server crash via stack overflow. DoS vector. |
| **#19292** | Context shifting broken after PR #18986 (regression) | `llama_memory_seq_rm + llama_memory_seq_add` produces incoherent output. Directly relevant to compaction — if fork baseline includes this regression, compaction output will be garbage. **Verify fork baseline.** |
| **#20093** | Heap buffer OOB in KV cache with M-RoPE (Qwen 3.5) | OOB read in `apply_ubatch()` during prompt cache restore — `pos.resize(n_tokens)` but M-RoPE needs `n_tokens * n_pos_per_embd`. Crashes on M-RoPE models. |

**Tier 1 — Fix in Fork (HIGH):**

| Issue | Title | Why It Matters for ModelAI |
|-------|-------|---------------------------|
| **#18591** | Tool calling broken in streaming mode (index regression) | Multiple tool calls all get index 0 instead of sequential. OpenAI API contract violation. **Verify fix in fork baseline.** |
| **#19513** | Premature EOS instead of tool call after 10-20 calls | Model generates EOS instead of `<tool_call>` — agentic workflows terminate prematurely. Affects Qwen3-Coder-Next, Minimax, GLM. |
| **#19869** | PEG parser crash with thinking models (Qwen3.5) | `common_chat_peg_parse` throws on degenerate repetition in thinking output. Parser code shared with llama-server. |
| **#19872** | Qwen3.5 template error 500 on tool calls | Jinja `items` filter error on tool call arguments — HTTP 500. Tool calling broken for Qwen3.5. |

**Tier 2 — Fix in Fork (MEDIUM):**

| Issue | Title | Why It Matters for ModelAI |
|-------|-------|---------------------------|
| **#19858** | Qwen3.5 full prompt reprocessing every turn | SWA/hybrid models invalidate checkpoints on every turn — no prompt caching benefit. Duplicate of #19794 pattern. |
| **#19760** | Stop signals not honored + context leakage between conversations | Multi-client correctness/privacy issue. |
| **#19520** | gpt-oss template double-escapes tool arguments | Malformed prompts from double JSON escaping. |
| **#19217** | Memory leak / infinite graph rebuild with LoRA (regression) | Server OOMs when LoRA adapters are used. Bisected to narrow commit range. |
| **#20140** | KV cache offload corruption with --cpu-moe + GPU | Silent data corruption in KV cache during split CPU/GPU inference. |

**Tier 3 — Monitor Upstream:**

| Issue | Title | Why It Matters |
|-------|-------|---------------|
| **#19345** | llama.cpp 40% slower than vLLM on MoE models | Performance gap with dedicated serving frameworks. SWA prompt reprocessing contributes. |
| **#19366** | llama.cpp 1/3 speed of MLX on Apple Silicon | M4 Pro: 24 tok/s llama.cpp vs 60 tok/s MLX for Qwen3-Coder-Next Q4_K_M |
| **#19647** | Chat template inflexibility for fine-tuned model variants | Hard-coded tag parsing conflicts with model-specific thinking/tool tags |

### Upstream Architectural Changes Affecting Compaction (merged Jan-Mar 2026)

These merged upstream PRs introduce architectural changes that the compaction implementation must account for:

| PR | Title | Impact on Compaction |
|----|-------|---------------------|
| **#19067** | kv-cache: V-less cache for MLA models | MLA models (DeepSeek, GLM 4.7) now store K-only with V as a view. Compaction solver assumes separate K/V — must detect MLA and skip V compaction or handle V-as-K-view. |
| **#18601** | memory: `llama_memory_hybrid_iswa` abstraction | New hybrid memory interface for attention+recurrent models. Compaction hooks into `llama_kv_cache` must account for this abstraction layer. |
| **#20301** | Dynamic `head_dim` and `n_rot` for SWA layers | SWA layers can now have per-layer dimensions. iSWA compaction graph (PR-6) assumed uniform dimensions — must validate. |
| **#19408, #20288** | Server checkpoint overhaul (2-checkpoint strategy) | Server now maintains dual checkpoints near prompt end. Compacted prefix must produce valid checkpoints or server will invalidate and force full reprocessing. |
| **#18675** | Autoparser: PEG-based tool-call parser rewrite | All legacy per-model parsers replaced. Fork's Workstream 7b tool-call fixes must target the NEW autoparser, not the old per-model parsers. **Major implementation surface change.** |
| **#19928, #20132** | M-RoPE shift and checkpoint fixes | KV cache save/load now handles `llama_kv_cell_ext` for M-RoPE. Compaction state serialization must match. |

### Strategic Risk: Upstream KV Compaction RFC (#20037)

**[#20037](https://github.com/ggml-org/llama.cpp/issues/20037)** — Filed 2026-03-02 by Gavin0725, titled "Research: Implement Fast KV Compaction via Attention Matching." References the same arXiv:2602.16284 paper that this fork implements. Links to the MIT reference code at `github.com/adamzweiger/compaction`. No core team response yet, no PR attached.

**Risk assessment:** If upstream begins implementing the same algorithm, our fork faces merge conflicts and potential design divergence. **Mitigation:** Our implementation is significantly ahead (Parts 1-6 designed, PR-5b shipped). Maintain upstream-ready naming and API conventions (Part 5) so our work can be contributed upstream if/when the RFC progresses.

## Workstream 7a: Structured JSON Schema Correctness (#10732, #19716, #19051, #19010)

### Objective
Make `response_format: { type: "json_schema", json_schema: ... }` behave deterministically with no silent fallback to unconstrained output. Accept all valid JSON Schema drafts.

### Why ModelAI Cares
ModelAI relies on structured output for report objects, table extraction, financial facts, tool-safe intermediate results, and Excel-safe payloads. If schema enforcement is inconsistent, downstream automation breaks silently.

### Implementation Changes

1. **Normalize request parsing** for both `response_format.type == "json_schema"` and legacy `json_schema` field
2. **[CRITICAL] Reject requests that provide both `json_schema` AND `grammar` fields** — currently (`server-task.cpp:376`) the `if (!data.contains("grammar"))` guard means providing both fields silently ignores `json_schema` and uses the raw grammar with zero validation. This is a bypass path. Fix: return 400 if both fields present.
3. **Add explicit server-side validation:**
   - Malformed schema → structured 4xx error
   - Unsupported schema feature → structured 4xx/5xx with explanation
   - Empty schema → explicit error
   - **Validate `json_schema` is actually a JSON object** — `json_value(data, "json_schema", json::object())` at line 377 silently coerces non-object types to empty object. Add explicit type check.
4. **Remove silent fallback:** if grammar compilation fails, return structured error, never continue as unconstrained text
5. **Fix typeless schema node rejection** (#19716): JSON Schema properties with no `type` field (e.g. `{"description": "..."}`) are valid per drafts 4-2020-12. The `json_schema_to_grammar()` converter at `json-schema-to-grammar.cpp:984-987` already handles this (falls back to `value` primitive), but document this as intentional behavior. Consider adding a strict mode.
6. **[CRITICAL] Add resource limits on grammar compilation** — the `_rules` map in `json-schema-to-grammar.cpp:318-336` has no size limit. Nested `allOf`/`oneOf` generates exponential rules. Recursive `visit()` has no depth limit (`json-schema-to-grammar.cpp:616-620`). Fix: add max depth limit (64), max rule count (10000), and compilation timeout. **Additionally (#19010):** `_visit_pattern()` uses `std::regex` which has unbounded stack recursion on crafted patterns like `(a+)+b`. Fix: limit regex pattern length to 1024 characters, wrap `std::regex` construction in try-catch for `std::regex_error`, and consider replacing with RE2 for bounded execution.
7. **Fix error accumulation** — `visit()` at `json-schema-to-grammar.cpp:989-991` returns empty string on unrecognized schema type instead of throwing immediately. Empty rules in the grammar map can produce unexpected behavior before `check_errors()` runs. Fix: throw immediately on unrecognized type.
8. **Make grammar-generation path observable:** log schema compilation success/failure, surface state in task metadata
9. **Add request-level metric hooks:** schema request count, compile failure count

### Files
- `tools/server/server-task.cpp` (lines 376-387) — request parsing, schema-to-grammar, dual-field rejection
- `tools/server/server-http.cpp` — error responses
- `common/json-schema-to-grammar.cpp` — schema compilation, typeless node handling, depth/rule limits, circular ref protection
- `tests/test-json-schema-to-grammar.cpp` — schema edge cases, DoS schemas
- `tools/server/tests/unit/test_chat_completion.py` — streaming/non-streaming parity

### Tests Required
1. Valid schema enforcement (basic, nested, array-of-objects)
2. Malformed schema rejection (missing required fields, invalid $ref)
3. Typeless schema node acceptance (#19716)
4. Streaming and non-streaming parity
5. `json_object` vs `json_schema` are not conflated
6. Repeated schema requests — no residual grammar contamination
7. **Dual-field rejection** — request with both `json_schema` + `grammar` returns 400
8. **DoS schema** — deeply nested allOf/oneOf (100+ depth) hits limit, returns error
9. **Circular $ref** — schema with circular references hits depth limit

### Merge Gate
- Schema requests enforced or explicitly rejected (no silent fallback)
- Dual-field requests rejected
- Grammar compilation has enforced resource limits
- Streaming and non-streaming tests both pass
- Typeless node schemas accepted

**Effort:** 3-4 days

## Workstream 7b: Tool-Call Crash Resistance (#16710, #14697, #18183, #19391, #20260)

### Objective
Make malformed, hallucinated, misformatted, and edge-case tool calls non-fatal to the server.

### Why ModelAI Cares
ModelAI is an agentic runtime. The server must never crash or hang because a model hallucinated a bad tool call. Current gaps:
- **#16710:** Malformed tool calls freeze/crash the server
- **#14697:** Tool calls appear in `content` field instead of `tool_calls` array — downstream parsers miss them
- **#18183:** Normal assistant JSON text triggers phantom tool invocations
- **#19391:** CJK characters in tool arguments corrupted on re-encoding
- **#20260:** Thinking models outputting text before `<tool_call>` cause 500 errors

### Implementation Changes

**Crash Fixes (must be first commits — verified crash paths from code review):**

1. **[CRITICAL] Fix `safe_args_parse()` crash on empty string** (`chat.cpp:40`): `.at(0)` and `.at(to_parse.length() - 1)` called without checking if `to_parse` is empty. `.at()` throws `std::out_of_range` which propagates as unhandled exception → server crash. **Fix:** add `if (to_parse.empty()) return json(to_parse);` at top of function.

2. **[CRITICAL] Fix tool-call diff array bounds** (`chat.cpp:183-185`): `msg_new.tool_calls[idx]` accessed using index from `msg_prv.tool_calls.size() - 1` without verifying `msg_new.tool_calls` has enough entries. If streaming delta has fewer tool calls than previous, this is out-of-bounds. **Fix:** check `idx < msg_new.tool_calls.size()` before access.

3. **[CRITICAL] Replace GGML_ASSERT in production JSON partial parser** (`json-partial.cpp:94, 108`): `GGML_ASSERT(!stack.empty() && ...)` in SAX parser callbacks. GGML_ASSERT aborts the process. Malformed partial JSON can desynchronize the stack and trigger these assertions, killing the server. **Fix:** replace with `if (!stack.empty() && ...) { ... } else { throw std::runtime_error(...); }`.

4. **[CRITICAL] Fix empty string buffer overread in JSON healing** (`json-partial.cpp:228, 250, 278, 307`): `str[str.length() - 1]` accessed without checking if `str` is empty. On empty string, `str.length() - 1` wraps to `SIZE_MAX` → buffer overread/crash. **Fix:** add `if (str.empty())` guard before all such accesses.

5. **[HIGH] Fix missing "arguments" field check in tool-call parsing** (`chat.cpp:318`): `fc.at("arguments")` called without `fc.contains("arguments")` check (unlike "name" which IS checked at line 314). **Fix:** add `if (!fc.contains("arguments"))` with default empty object.

6. **[HIGH] Fix uncaught `json::parse()` exceptions** (`chat.cpp:1462, 1466, 1595`): Several `json::parse()` calls in `common_chat_templates_apply_jinja()` have no try-catch. **Fix:** wrap in try-catch with structured error.

**Important: Autoparser Rewrite (upstream PR #18675, merged 2026-03-06)**

Upstream merged a complete rewrite of all tool-call parsers, replacing legacy per-model parsers with a unified PEG-based autoparser. This changes the implementation surface for items 7-13 below. The crash-path fixes (items 1-6) target lower-level code (`json-partial.cpp`, `chat.cpp` argument parsing) that is NOT replaced by the autoparser. But feature fixes 7-13 MUST be validated against the new PEG parser infrastructure. Before starting 7b feature work, sync the fork's `upstream-master` to include PR #18675 and verify which of items 7-13 are already addressed by the autoparser.

**Feature Fixes:**

7. **Harden tool-call parsing path** (now PEG-based via `common/chat.cpp` and `common/chat-parser.cpp`):
   - Malformed JSON → structured parse error, not crash
   - Unknown tool → structured validation error, not crash
   - Invalid arguments → structured error, not crash
   - Partial tool-call in streaming → incremental safe parse, not crash
8. **Dual-path tool-call detection with disambiguation** (#14697 + #18183):

   **[Reviewer-2 Condition R2-3]** These two fixes conflict: #14697 says "check `content` for tool calls" while #18183 says "don't parse `content` as tool calls." The disambiguation rule resolves this:

   **Rule:** Only parse `content` as tool calls when the content matches a **recognized tool-call structure**:
   - **XML format:** Content contains `<tool_call>` or `<function_call>` XML tags (Hermes 2 Pro, Mistral formats)
   - **JSON format:** Content is a JSON object (or array of objects) where each object has BOTH `"name"` (string) AND `"arguments"` (string or object) fields — i.e., matches `{"name": <string>, "arguments": <string|object>, ...}`
   - **All other JSON:** Bare JSON objects, arrays, or JSON embedded in natural language text MUST NOT trigger tool-call parsing. This includes: `{"key": "value"}` without name+arguments, JSON code blocks in explanations, JSON examples in assistant text.

   **Implementation:** In `common/chat.cpp` tool-call detection path:
   ```cpp
   // Only parse content as tool call if it matches a recognized structure
   bool is_tool_call_json(const json & j) {
       if (!j.is_object()) return false;
       return j.contains("name") && j["name"].is_string()
           && j.contains("arguments");
   }
   bool is_tool_call_xml(const std::string & text) {
       return text.find("<tool_call>") != std::string::npos
           || text.find("<function_call>") != std::string::npos;
   }
   ```

   **Normalization:** When tool calls are found in `content` (via either format), move them to the `tool_calls` array. Strip the tool-call portion from `content`. If `content` contained ONLY tool calls, set `content` to null.

9. **Fix phantom tool-call detection** (#18183): tighten Hermes 2 Pro parser to require proper `<tool_call>` tags per the disambiguation rule above. The parser in `common/chat-parser.cpp` must NOT interpret bare JSON as tool calls even if the JSON happens to be valid.
10. **Fix CJK re-encoding** (#19391): fix `ensure_ascii` handling in `common/chat-parser.cpp` for multi-turn tool arguments
11. **Handle pre-tag content** (#20260): parser must tolerate `<think>...<tool_call>` sequences from thinking models
12. **Explicit "bad tool call" response path:** return structured error object, never leave slot/task in corrupt state
13. **Add metrics:** malformed/rejected/parse-failure tool-call counts

### Files
- `common/chat.cpp` — tool-call parsing, `safe_args_parse()` (lines 38-48), diff computation (lines 183-185), tool-call field check (line 318), json::parse wrappers (lines 1462, 1466, 1595)
- `common/json-partial.cpp` — SAX parser assertions (lines 94, 108), empty string guards (lines 228, 250, 278, 307)
- `common/chat-parser.cpp` — Hermes/generic tag parsing, CJK encoding
- `tools/server/server-task.cpp` — task lifecycle on tool-call failure
- `tools/server/server-context.cpp` — slot cleanup after tool-call error
- `tools/server/tests/unit/test_tool_call.py` — expanded edge case tests

### Tests Required
1. **Empty tool arguments** → `safe_args_parse("")` does not crash
2. **Streaming delta with fewer tool calls** → no array out-of-bounds
3. **Malformed partial JSON** → no GGML_ASSERT abort; exception caught
4. **Empty string in JSON healing** → no buffer overread
5. **Tool call with missing "arguments" field** → structured error, not crash
6. Unknown tool name → structured error, next request succeeds
7. Malformed arguments JSON → error, not crash
8. Partial/truncated tool-call → safe degradation
9. Streaming tool-call delta corruption → no hang
10. CJK tool arguments round-trip correctly across turns
11. JSON in assistant explanation text → NOT phantom tool call
12. Thinking model text before `<tool_call>` → correctly parsed
13. Tool calls in `content` field → normalized to `tool_calls`
14. Multi-request repeated bad-tool-call stress test

### Merge Gate
- All 6 crash-path fixes verified (items 1-6 above)
- Malformed tool calls cannot kill or wedge the server
- Request state clean after failure; next request succeeds
- CJK arguments preserved across turns
- No phantom tool calls from assistant JSON text

**Effort:** 4-5 days

## Workstream 7c: Structured-Output Stability Under Repeated Load (#17391, #19068, #19679, #19304)

### Objective
Make repeated structured-output workloads stable under sustained load with no memory leaks, state corruption, or grammar stack crashes.

### Why ModelAI Cares
ModelAI workloads repeat the same extraction schema many times: many report sections, many company docs, repeated batch analysis. One-off correctness is insufficient — the runtime must survive hours of repetition.

### Implementation Changes

1. **Audit structured-output request lifecycle** in `server-context.cpp`:
   - Per-request grammar ownership verified
   - Task teardown on completion, cancellation, parse failure, timeout
   - No grammar/parser state leaks across requests
2. **Fix grammar trigger loop** (#19068): add error recovery when grammar sampler enters infinite trigger state. **Detection mechanism:** max iterations = 1000 per token; if exceeded, break with structured error "grammar loop detected at rule [rule_name]". Log the triggering grammar rule for debugging. **Recovery:** reset sampler state, return error to client, ensure slot is clean for next request.
2b. **[CRITICAL] Fix grammar stack corruption crash** (#19679, #19304): `llama_grammar_accept_token` throws `std::runtime_error("Unexpected empty grammar stack after accepting piece")` during generation. Reproducible on Apple Metal (M4 Mac Mini) with Qwen3-Coder-Next + flash attention + jinja, and at 86K context after 50+ tool calls. Root cause: grammar state machine doesn't handle certain multi-byte token pieces at grammar rule boundaries — the stack becomes empty mid-acceptance. **Fix:** add defensive empty-stack check before accessing `stack.back()` in `llama_grammar_accept_token` (similar to the GGML_ASSERT → exception conversion already planned for `json-partial.cpp`). **Test:** long agentic session (50+ tool calls, 86K+ context) on Apple Metal with flash attention — must not crash.
3. **Explicit cleanup on all non-happy paths:** timeout, cancel, parse failure, slot reuse, server sleep/wake
4. **Add stress-safe structured request path:** avoid accumulating parser/grammar state in shared objects; ensure thread-safe lifecycle boundaries
5. **Add metrics:** active structured requests, structured failures, structured cancellations

### Files
- `tools/server/server-context.cpp` — slot lifecycle, cleanup
- `tools/server/server-task.cpp` — task teardown
- `tools/server/server-queue.cpp` — deferred task aging/cleanup
- Grammar sampler state in `common/` — loop detection

### Tests Required
1. Repeated non-streaming structured-output loop (100+ iterations)
2. Repeated streaming structured-output loop (100+ iterations)
3. Parallel structured-output with multiple slots
4. Large system prompt + medium user prompt + fixed schema
5. Failure-path cleanup after forced malformed schema or cancellation
6. Long-run soak test (CI smoke tier + manual heavy tier)
7. Grammar trigger loop detection and recovery test

### Merge Gate
- Repeated structured requests do not crash, hang, or leak memory
- Cleanup verified after failure and cancellation
- Soak test passes on realistic workload

**Effort:** 2-3 days

## Workstream 7d: Server Lifecycle & Slot Management (#17387, #19758, #11970)

### Objective
Make server start/stop/restart clean, slot management programmatic, and KV cache lifecycle correct.

### Why ModelAI Cares
ModelAI's local runtime is frequently started/stopped during development. Slots must be manageable via API for context lifecycle control. Silent KV truncation destroys compaction quality.

### Implementation Changes

1. **Fix port reuse on restart** (#19758): Socket lifecycle is managed by cpp-httplib. The `chunked_content_provider` lambda (`server-http.cpp:356-388`) does NOT check `is_connection_closed()` — server continues generating after client disconnects. **Fix:** add `req.is_connection_closed()` check inside the lambda; also investigate SO_REUSEADDR in httplib configuration.

2. **[CRITICAL] Fix /slots/0?action=erase hang** (#17387): Root cause confirmed — when slot is processing, erase task is deferred (`server-context.cpp:2039`) but `rd.next()` (`server-context.cpp:4297`) blocks the HTTP handler forever. Deferred task only runs via `callback_on_release()` when slot calls `release()`. If slot is stuck, client hangs indefinitely. **Fix:** for erase operations, return 409 Conflict immediately when slot is processing, instead of deferring. Alternative: add timeout to `rd.next()` for erase operations.

3. **Fix KV cache truncation** (#11970): audit KV cache state transitions during multi-turn `/v1/chat/completions`; prevent silent prefix truncation. **Additional finding:** context shift at `server-context.cpp:2176-2177` clears and rebuilds token list AFTER KV cache ops — if token insertion fails (OOM), slot has shifted KV but empty token list (inconsistent state). **Fix:** make token list update transactional with KV ops.

4. **[CRITICAL] Fix child slots stuck in WAIT_OTHER on parent failure** — if parent slot fails/cancels before child slots are found (`server-context.cpp:2880-2897, 1793-1796`), children remain in SLOT_STATE_WAIT_OTHER forever. No timeout or forced release mechanism exists. **Fix:** add timeout on WAIT_OTHER state (e.g., 30s). After timeout, release child slot with error "parent slot timed out".

5. **[HIGH] Fix orphaned KV on LoRA cache skip** — `server-context.cpp:1219` calls `slot.prompt.tokens.clear()` but NOT `prompt_clear()` (which calls `llama_memory_seq_rm()`). This leaves orphaned KV data from the previous LoRA adapter. **Fix:** call `slot.prompt_clear(false)` instead of just `tokens.clear()`.

6. **Add slot state invariant checks:** validate state machine transitions, detect corrupt states early. Use logged warnings (not assertions) in production.

### Files
- `tools/server/server-http.cpp` — socket lifecycle, SSE cleanup, connection close check (line 356-388)
- `tools/server/server-context.cpp` — slot erase handler (lines 2025-2053, 4287-4312), child slot lifecycle (lines 2880-2897), LoRA cache (line 1219), context shift (lines 2147-2181)
- `tools/server/server-task.cpp` — slot state machine
- `src/llama-kv-cache.cpp` — KV sequence management

### Tests Required
1. Server restart after SSE stream — port available immediately
2. Slot erase on processing slot — returns 409, not hang
3. Slot erase on idle slot — returns 200, slot usable afterward
4. Multi-turn chat — KV cache not silently truncated
5. Parent slot failure — child slots released within timeout
6. LoRA adapter change — no orphaned KV data
7. Slot state after error/cancel — invariants hold
8. Client disconnect mid-stream — server stops generating

### Merge Gate
- Server restarts cleanly after streaming connections
- Slot erase API never hangs (returns 409 or 200)
- Child slots cannot be stuck in WAIT_OTHER forever
- KV cache integrity maintained across turns
- No orphaned KV on LoRA switch

**Effort:** 3-4 days

## Workstream 7e: Prompt-Cache Hygiene (#4989, #19794)

### Objective
Make prompt-cache behavior explicit, bounded, and safe for long-session ModelAI workflows.

### Why ModelAI Cares
ModelAI uses long sessions, repeated prompts, and save/restore. Generic prompt-cache behavior can produce pathological KV growth, and hybrid/SWA models currently invalidate the context checkpoint on every turn (#19794), forcing full re-processing.

### Implementation Changes

1. **Audit `cache_prompt` behavior** for `/completion` and `/chat/completions` and repeated requests

2. **[HIGH] Fix silent cache save failure** — `prompt_save()` at `server-context.cpp:234-235` returns silently when `prompt_cache.alloc()` returns nullptr. No log, no metric, no notification. Future cache-hit expectations will miss silently. **Fix:** log at WARN level, increment `cache_save_failures` metric.

3. **[MEDIUM] Fix overly aggressive cache load failure handling** — when `prompt_load()` fails (`server-context.cpp:1153-1154`), ALL KV state is cleared via `prompt_clear(false)`, even though partial prefix reuse might have worked. **Fix:** on cache load failure, fall back to LCP matching instead of full clear.

4. **Fix hybrid model checkpoint invalidation** (#19794): checkpoint validation at `server-context.cpp:2549-2557` erases checkpoints when `pos_min > pos_min_thold` but doesn't validate that checkpoint data pointers aren't stale after context shift. Prevent SWA models from clearing the prompt cache on every turn.

5. **Add hard guardrails:** bounded reuse, explicit failure on KV exhaustion, no uncontrolled auto-generation

6. **Align with compaction state model — hash-based compacted prefix matching:**

   **[Reviewer-2 Condition R2-4]** The server's existing LCP (Longest Common Prefix) matching at `server-context.cpp:2374-2441` compares token IDs:
   ```cpp
   n_past = slot.prompt.tokens.get_common_prefix(input_tokens);
   ```
   A compacted prefix is NOT a token sequence — it's a compressed KV representation of N tokens in M slots (M << N). Token-level LCP matching will always fail against compacted prefixes because the token IDs don't match the compacted representation. This is a fundamental architectural mismatch.

   **Design: Two-Tier Cache Matching**

   Tier 1 — **Token-LCP matching** (existing, for uncompacted prefixes):
   - Works as today: compare incoming prompt token IDs against cached slot tokens
   - Used when the slot has not been compacted

   Tier 2 — **Hash-based matching** (new, for compacted prefixes):
   - At compaction time, compute SHA-256 of the ORIGINAL prefix token IDs (before compaction)
   - Store the hash alongside the compacted prefix metadata in `llama_kv_compacted_prefix_store`
   - On new request, compute SHA-256 of the incoming prompt's prefix portion
   - If hash matches a compacted prefix, reuse the compacted KV data

   **Data structures:**
   ```cpp
   // Added to llama_kv_compacted_prefix_store (or server-level cache)
   struct compacted_prefix_cache_entry {
       uint8_t   prefix_hash[32];     // SHA-256 of original token IDs
       uint32_t  n_original_tokens;   // length of original prefix
       llama_seq_id seq_id;           // sequence owning this prefix
       int64_t   compacted_at_ns;     // timestamp for LRU eviction
   };
   ```

   **Matching algorithm:**
   ```
   on_new_request(input_tokens):
       // Tier 1: try token-LCP matching (existing path)
       n_past = slot.prompt.tokens.get_common_prefix(input_tokens)
       if n_past >= slot.prompt.tokens.size() * 0.9:
           return use_lcp_match(n_past)

       // Tier 2: try compacted prefix hash matching
       candidate_hash = sha256(input_tokens[0..expected_prefix_len])
       for each compacted_entry in cache:
           if candidate_hash == compacted_entry.prefix_hash
              && len(input_tokens) >= compacted_entry.n_original_tokens:
               // Compacted prefix covers the incoming prompt's prefix
               return use_compacted_prefix(compacted_entry, input_tokens)

       // No match — full prefill required
       return prefill_from_scratch(input_tokens)
   ```

   **[Reviewer-2 Minor Note]** The Tier 2 hash matching iterates linearly over compacted entries. For expected scale (1-4 entries per slot), this is fine. If the compacted prefix cache grows significantly (e.g., many concurrent compacted sequences), replace linear scan with a hash table keyed on the first 8 bytes of SHA-256.

   **Interaction with Part 8b (prefix fingerprint caching):** The `prefix_fingerprint` API from Part 8b can be implemented ON TOP of this hash-based matching. The product computes a fingerprint (e.g., from application-specific context structure), the server uses it as an additional matching key alongside the token-hash. The two mechanisms are complementary: token-hash matches exact prompts, fingerprint matches semantically equivalent prompts.

### Files
- `tools/server/server-context.cpp` — prompt save/load (lines 225-248, 1138-1158, 2374-2441), checkpoint validation (lines 2549-2557), new hash-based matching
- `tools/server/server.cpp` — `cache_prompt` configuration
- `src/llama-kv-cache.cpp` — sequence management interaction with cache
- `src/llama-kv-compacted-prefix.h` — add `prefix_hash` to store metadata

### Tests Required
1. Repeated `cache_prompt` request — stable, no growth
2. KV exhaustion → graceful error, not crash
3. Cached prompt does not trigger unintended extra generation
4. **Cache save failure** → logged, metric incremented
5. **Cache load failure** → falls back to LCP, not full clear
6. Slot state coherent after cache failure
7. SWA/hybrid model prompt cache not invalidated per-turn

### Merge Gate
- Prompt-cache behavior bounded and predictable
- Cache failures logged and metriced (no silent failures)
- KV exhaustion graceful
- Hybrid model prompt cache survives across turns

**Effort:** 2-3 days

## Workstream 7f: Server Performance & Observability (#12171, #15389)

### Objective
Close server-side performance gaps and add telemetry for diagnosis.

### Why ModelAI Cares
If `llama-server` is 3x slower than `llama-cli` (#12171), or if merely enabling tools degrades performance (#15389), the product experience suffers. Diagnosis requires telemetry.

### Implementation Changes

1. **Profile server vs CLI path:** identify HTTP/slot dispatch overhead causing #12171
2. **Benchmark tool-enabled vs tool-disabled:** quantify #15389 on target hardware
3. **Add runtime telemetry:** prompt-processing time, generation time, slot queueing delay, grammar overhead
4. **Measure and document schema/tool parsing overhead** — ensure it's not hidden latency
5. **Benchmark harnesses** for server prompt-eval: different `n_threads_batch`, `ubatch`, `batch`, cache types

### Files
- `tools/server/server-context.cpp` — timing instrumentation
- `tools/server/server-task.cpp` — per-request timing
- `tools/server/bench/` — benchmark scripts
- `scripts/bench-server-perf.sh` — new benchmark script

### Tests / Benchmarks Required
1. Server vs CLI prompt-eval benchmark on real-text prompt
2. Tool-enabled vs tool-disabled throughput comparison
3. Structured-output overhead measurement
4. ModelAI workload benchmark: W2 or W3 workload using a model >= 1B params, a >= 2048-token real-text prefix, and explicit quality thresholds

### Merge Gate
- Measured improvement on at least one real workload meeting the minimum benchmark contract:
  - workload class: W2 or W3
  - model size: >= 1B params
  - prefix length: >= 2048 real-text tokens
  - quality threshold: cosine similarity >= 0.95 versus the full-context baseline
- Telemetry clearly shows where gains come from
- No regression in correctness or structured/tool behavior

**Effort:** 2-3 days

---

# Part 8: Product Optimization Pipeline

## Why This Is Needed

**[P2 Finding]** The backend capability matrix from the performance roadmap is still docs-only. There is no real `llama_backend_caps`-style runtime probe for Metal/CUDA/Vulkan/CPU. Current server reporting mostly echoes configured params.

**[P2 Finding]** Prefix caching is only generic llama.cpp prompt reuse (`cache_prompt`, LCP matching), not the generic stable prefix-fingerprint cache described in the performance roadmap. No stable fingerprint persistence, no fingerprint index, and no explicit product/server invalidation contract.

**[P2 Finding]** Batch/ubatch autotuning and laptop-aware thread scheduling are not implemented. Only manual `--threads`, `--threads-batch`, `--batch-size`, `--ubatch-size` knobs exist. No first-run benchmarking, per-device persistence, or P-core/E-core style policy.

**Important positioning:** These are PR-7/PR-8 items from the performance roadmap. They are included in this plan for completeness and to define implementation surfaces, but they should be prioritized AFTER Parts 1-7 are complete. Parts 1-5 (compaction core) and Part 6 (runtime integration) must ship before optimization work begins.

## 8a. Backend Capability Detection

### Objective
Build a runtime `llama_backend_caps` struct populated at model load, so every optimization policy is backend-aware rather than hardcoded.

### Why It's Needed
The Excel product ships on Metal (Mac), CUDA (NVIDIA Windows), Vulkan (AMD/Intel Windows), and CPU fallback. Not all optimizations are available everywhere. Without runtime detection, the product either ships the lowest-common-denominator or crashes on unsupported backends.

### Implementation

```cpp
// NOTE: Uses llama_ prefix (not modelai_) to maintain upstream-ready naming convention.
// Placed in common/ (not src/) to avoid polluting internal llama library.
struct llama_backend_caps {
    bool supports_flash_attn;
    bool supports_kv_quant_q8;
    bool supports_kv_quant_q4;
    bool supports_additive_kq_b;
    bool compacted_prefix_fast_path;   // can stay on FA?
    uint64_t vram_bytes;
    uint64_t max_context_tokens;       // estimated for this model
    int recommended_n_gpu_layers;
    int recommended_batch_size;
    int recommended_ubatch_size;
    int recommended_threads;
    int recommended_threads_batch;
};
```

Populate by querying backend feature flags at model load. Gate all optimization policies through this struct.

### Expected Capability Matrix

| Backend | FA | KV q8_0 | KV q4_0 | Additive kq_b in FA | Notes |
|---------|-------|---------|---------|---------------------|-------|
| Metal | Yes | Yes | Test | No | Block-skip optimization |
| CUDA | Yes | Yes | Yes | No (upstream) | CUDA Graphs boost |
| Vulkan | Yes | Yes (FA only) | No | No | Issue #9551 |
| CPU | Chunked | Yes | Yes | Yes (non-FA path) | Most compatible |

### Files
- `common/llama-backend-caps.h` — New header with struct + probe function (uses `llama_` prefix, not `modelai_`, for upstream-ready naming)
- `common/llama-backend-caps.cpp` — Implementation: query ggml backends
- `tools/server/server-context.cpp` — Populate at model load, expose in `/props`
- `src/llama-context.cpp` — Use for default parameter selection

**Effort:** 2-3 days (PR-7)

## 8b. Prefix Fingerprint Caching

### Objective
Replace generic token-prefix matching with product-supplied stable prefix fingerprints.

### Why It's Needed
Generic LCP matching only reuses tokens that are byte-identical at the start of the prompt. For ModelAI:
- Same workbook + different question = same system prefix + different user turn → token-LCP may work
- Same workbook + structural change (new sheet, new named range) → token-LCP often fails even when the logical prefix remains reusable
- Compacted prefix cannot be matched by token-LCP because compacted KV is not token-aligned

### Implementation

**Architecture note:** The server should expose a **generic prefix fingerprint API** — the product (ModelAI Excel add-in) computes the fingerprint from workbook structure and passes it to the server. Workbook-specific knowledge (named ranges, tabs, column types, invalidation policy) stays in the product layer, not the inference engine. This avoids a layering violation.

1. **Server-side: generic fingerprint-based prefix matching** — accept a `prefix_fingerprint` field in request. Match against stored prefixes by fingerprint instead of token-LCP. Return cache hit/miss status.
2. **Product-side: stable prefix hashing** — compute hash from system prompt + tool definitions + workbook schema. Pass as `prefix_fingerprint` to server.
3. **KV cache persistence per fingerprint:** use `--slot-save-path` or equivalent API
4. **Delta-only appending:** on each user turn, only process user question + changed cells
5. **Partial invalidation:** structural workbook changes invalidate only the workbook portion (product-side decision, server sees new fingerprint)
6. **Interaction with compaction:** compacted prefix IS a valid cache entry; new requests match against it via fingerprint, not token-LCP

### Files
- `tools/server/server-context.cpp` — prefix matching, cache save/load (lines 2374-2441)
- `tools/server/server.cpp` — slot management, `cache_prompt` logic
- `src/llama-kv-cache.cpp` — `seq_pos_min/max`, hash-based matching
- New: `tools/server/server-prefix-cache.h` — generic prefix fingerprint matching and cache indexing

**Effort:** 3-4 days (PR-8)

## 8c. Batch/Ubatch Autotuning

### Objective
Automatically select optimal batch_size, ubatch_size, and thread counts per device.

### Why It's Needed
Current behavior is "use configured/default values" (`llama-context.cpp:156-159`, `common/common.h:69-76`). There is no first-run benchmarking, no per-device persistence, no distinction between P-core and E-core scheduling. For a consumer product shipping on diverse hardware, manual knobs are not viable.

### Implementation

**Note on startup delay:** 48 combinations at ~5s each = ~4 minutes, which is unacceptable for Excel add-in cold start. Benchmark grid MUST run asynchronously in background. Use conservative defaults until tuning completes. Show progress indicator in product UI.

1. **First-run benchmark grid** on model load (**async, background**):
   - `batch_size`: [256, 512, 1024, 2048]
   - `ubatch_size`: [128, 256, 512]
   - `threads_batch`: [2, 4, physical_cores/2, physical_cores]
2. **Sane defaults while tuning runs** — use platform-specific conservative defaults immediately (Metal: ubatch=256, threads=physical_cores/2)
3. **Persist best settings** per tuple: `(model_id, quant_type, backend, device_id)`
4. **Separate tuning for prefill vs decode**
5. **Re-tune triggers:** model change, backend change, major app version

### Files
- New: `src/llama-autotune.h/.cpp` — benchmark grid, persistence
- `src/llama-context.cpp` — consume autotuned values
- `common/common.cpp` — fallback defaults

**Effort:** 3-4 days (PR-8)

## 8d. Laptop-Aware Thread Scheduling

### Objective
Prevent thermal throttling on sustained inference sessions by adapting thread scheduling to consumer hardware.

### Why It's Needed
Current `cpu_params` (`common/common.h:69-76`) has affinity mask and priority, but no policy for:
- P-core vs E-core distinction (Apple Silicon, Intel 12th+ gen)
- Thermal throttling detection and response
- Background mode when app is minimized

### Implementation

1. **P-core/E-core detection** on Apple Silicon and Intel Alder Lake+
2. **Default policy:** use P-cores for prefill (latency-sensitive), E-cores for decode (throughput-sensitive)
3. **Thermal throttling response:** monitor CPU frequency; reduce threads if throttling detected
4. **Background mode:** reduce to E-cores only when app loses focus

### Files
- New: `src/llama-thread-policy.h/.cpp` — core detection, thermal monitoring
- `common/common.cpp` — integrate with `cpu_params`
- `ggml/src/ggml-cpu/ggml-cpu.cpp` — thread pool, affinity

**Effort:** 2-3 days (PR-8)

---

# Implementation Schedule

## Phase 1: KV Compaction Core (Parts 1-5)

| Step | Part | What | Files | Risk | Effort |
|------|------|------|-------|------|--------|
| **6b-1** | 1 | `llama_set_eval_callback` public API | llama.h, llama-context.h/.cpp | None | 1 hour |
| **6b-2** | 1 | `llama_q_capture_state` + callback impl | llama-kv-compact-self-study.h/.cpp | Low | 1 day |
| **6b-3** | 1 | GQA regrouping + subsampling | llama-kv-compact-self-study.cpp | None | 1 day |
| **6b-4** | 1 | Generation loop + Q-capture + KV cleanup | llama-kv-compact-self-study.cpp | Medium | 2 days |
| **6b-5** | 1 | Solver integration (reuse fit_beta/fit_values) | llama-kv-compact-self-study.cpp | Low | 1 day |
| **6b-6** | 1 | Pipeline wiring (cache, iSWA, CMake) | llama-kv-cache.h/.cpp, iswa, pipeline.h, CMakeLists | Low | 0.5 day |
| **6b-7** | 1 | Self-study tests + quality benchmarks | tests/test-kv-compact-self-study.cpp | Low | 1-2 days |
| **6b-8** | 2a | Quantized transposed V extraction | llama-kv-cache.cpp, tests | Low | 1 day |
| **6b-9** | 2b | Remove multi-stream assertion + test | llama-graph.cpp, tests | Low | 0.5 day |
| **6b-10** | 2c | M-RoPE model validation | tests/test-kv-compact-quality.cpp | Low | 0.5 day |
| **6b-11** | 2d | Backend validation (Metal/CUDA/CPU) | tests, docs | None | 0.5 day |
| **6b-12** | 3 | Flash + beta documentation (no code) | docs | None | 0.5 day |
| **6b-13** | 4 | Production workload tests | tests/test-kv-compact-workload.cpp, scripts/ | Medium | 2-3 days |
| **6b-14** | 5a | Fix duplicate OMP opts struct | llama-kv-compact-pipeline.h | None | 15 min |
| **6b-15** | 5b | Long-context benchmark (throughput crossover) | tests/test-kv-compact-longctx.cpp, scripts/ | Medium | 2-3 days |
| **6b-15b** | 5b-1 | Self-study benchmark extension | tests/test-kv-compact-longctx.cpp, scripts/ | Medium | 1-2 days |
| **6b-16** | 5c | Upstream algorithm + integration docs | docs/ | None | 1-2 days |
| **6b-17** | 5d | Test hardening (negative cases) | tests/ | None | 0.5 day |

**Phase 1 Total: ~17-23 days**

## Phase 2: Compaction Runtime Integration (Part 6)

| Step | Part | What | Files | Risk | Effort |
|------|------|------|-------|------|--------|
| **6b-18** | 6a | Wire real compaction state into /props, /metrics | server-context.cpp/.h, llama-kv-cache.h/.cpp | Low | 1 day |
| **6b-19** | 6b | FA + compaction mode selection + observability | llama-graph.cpp, server-context.cpp | Low | 0.5 day |
| **6b-20** | 6c | Fix quantized K compaction spec drift | test-kv-compacted-prefix.cpp, test-kv-compact-quality.cpp | Low | 0.5 day |
| **6b-21** | 6d | Update fork summary support matrix | modelai-fork-summary.md | None | 0.5 hour |

**Phase 2 Total: ~2-3 days**

## Phase 3: Server/Runtime Hardening (Part 7)

| Step | Part | What | Files | Risk | Effort |
|------|------|------|-------|------|--------|
| **7-1** | 7a | Structured JSON schema correctness + resource limits | server-task.cpp, json-schema-to-grammar.cpp | Medium | 3-4 days |
| **7-2** | 7b | Tool-call crash resistance (6 crash fixes + 5 features) | chat.cpp, json-partial.cpp, chat-parser.cpp | **High** | 4-5 days |
| **7-3** | 7c | Structured-output stability + memory leaks | server-context.cpp, server-task.cpp, server-queue.cpp | Medium | 2-3 days |
| **7-4** | 7d | Server lifecycle + slot management (6 fixes) | server-http.cpp, server-context.cpp, llama-kv-cache.cpp | **High** | 3-4 days |
| **7-5** | 7e | Prompt-cache hygiene + hybrid model fix | server-context.cpp, server.cpp | Medium | 2-3 days |
| **7-6** | 7f | Server performance + observability | server-context.cpp, server-task.cpp, bench/ | Low | 2-3 days |

**Phase 3 Total: ~16-22 days**

## Phase 4: Product Optimization Pipeline (Part 8) — Future PR-7/PR-8

| Step | Part | What | Files | Risk | Effort |
|------|------|------|-------|------|--------|
| **8-1** | 8a | Backend capability detection | common/llama-backend-caps.h/.cpp, server-context.cpp | Medium | 2-3 days |
| **8-2** | 8b | Prefix fingerprint caching | server-context.cpp, server-prefix-cache.h | High | 3-4 days |
| **8-3** | 8c | Batch/ubatch autotuning | llama-autotune.h/.cpp, llama-context.cpp | Medium | 3-4 days |
| **8-4** | 8d | Laptop-aware thread scheduling | llama-thread-policy.h/.cpp, common.cpp | Medium | 2-3 days |

**Phase 4 Total: ~10-14 days**

**Grand Total: ~42-57 days of implementation-focused engineering time across all phases**  
**Practical planning note:** including review cycles, debugging, soak tests, benchmark reruns, and merge friction, expect closer to ~50-70 calendar days for one implementer.

### Recommended Implementation Order

1. **Phase 1 first** (Parts 1-5): KV compaction core is the primary value proposition
2. **Phase 2 immediately after** (Part 6): Wiring compaction into the server makes it consumable by the product
3. **Phase 3a crash-path hardening can start in parallel** (Part 7): prioritize 7a/7b/7d crash and hang fixes early on separate branches
4. **Phase 3b soak/perf hardening follows 3a**: structured-output stability, prompt-cache hygiene, and server performance work after the crash paths are closed
5. **Phase 4 after Phase 1-3** (Part 8): Optimization only makes sense once correctness is proven

**Branch strategy:**
- Phase 1-2: `kv-compact-pr6b-*` branches merged into `modelai-main`
- Phase 3a/3b: `server-hardening-pr*` branches merged into `modelai-main`
- Phase 4: `perf-pr7-*` and `perf-pr8-*` branches

---

# Supported Model Architectures

The "last 3D Qcur-prefixed" disambiguation strategy works for all standard attention models (85+ architectures). Known variants:

| Category | Models | Tensor name captured |
|----------|--------|---------------------|
| Standard post-RoPE | llama, qwen2, falcon, olmo, baichuan, exaone, ... | `"Qcur-{il}"` |
| With kq_norm | llama (use_kq_norm), llama-iswa | `"Qcur_normed-{il}"` |
| With q_norm | qwen3, qwen3next, apertus, step35-iswa | `"Qcur_normed-{il}"` or `"Qcur-{il}"` |
| Temperature-scaled | mistral3, deepseek2 | `"Qcur_attn_temp_scaled-{il}"` |
| Gemma scaled | gemma | `"Qcur_scaled-{il}"` |
| M-RoPE | qwen2vl, qwen3vl, glm4 | `"Qcur-{il}"` (uses `ggml_rope_multi`) |

**Not supported (out of scope):** MLA models (deepseek2 alternate path), non-attention models (Mamba, RWKV), models with non-standard Q names (lfm2).

---

# Risk Register

## KV Compaction Risks (Parts 1-5)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| cb_eval overhead slows generation | Low | Low | Benchmarked at ~5-10% overhead; v1 optimization available |
| Graph reuse skips cb_eval reinstall | Medium | Medium | cb_eval persists on scheduler after first install; verify with test |
| Multiple Qcur copies waste bandwidth | Low | None | ~20KB per copy × 3 per layer × 28 layers = ~1.7MB total per step |
| KV cache full during generation | Low | Medium | Only needs ~256 extra slots; fail gracefully if insufficient |
| Autoregressive sampling complexity | Medium | Medium | Start with greedy (temperature=0); add sampling in v1 |
| Solver quality regression vs surrogates | Very Low | High | Self-study proven superior in MIT paper; test validates |
| Quantized transposed V dequant cost | Low | Low | One-time extraction cost; negligible vs solver time |
| Multi-stream compacted prefix bugs | Low | Medium | Exec functions already handle streams; test validates |
| M-RoPE Q-capture incorrect | Low | Medium | Tensor name convention is same; validate with test |
| Flash + non-zero beta needed urgently | Low | High | Selection-only (zero-beta) is production default; FlashBias upstream solves this |
| Upstream review rejects API design | Medium | Medium | Keep internal for now; gather feedback before exposing public API |
| Production workload reveals quality gap | Medium | High | Early testing with real SEC filings catches issues before deployment |

## Runtime Integration Risks (Part 6)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Capability reporting enables compaction on unsupported models | Medium | High | Runtime checks: verify non-SWA cache, standard attention, supported arch before reporting available |
| Quantized K spec drift breaks existing tests | High | Low | Fix test immediately (6b-20); test now validates acceptance, not rejection |
| FA override not visible to product layer | Medium | Medium | Add counter + log + /props field so orchestrator can detect non-FA fallback |

## Server Hardening Risks (Part 7)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Upstream fixes diverge from fork patches | High | Medium | Track upstream PRs; rebase fork patches on upstream sync |
| Grammar loop fix introduces new regressions | Medium | Medium | Extensive repeated-load soak testing before merge |
| Tool-call hardening changes break working parsers | Medium | High | Per-template test matrix; regression tests for all supported chat templates |
| Server lifecycle fixes expose new concurrency bugs | Medium | Medium | Single-threaded slot access model; add state invariant assertions |
| Prompt-cache + compaction interaction bugs | Medium | High | Explicit test: compact prefix → cache → reload → verify correct |

## Product Optimization Risks (Part 8)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Autotuning first-run benchmark delays startup | Medium | Medium | Async benchmark with sane defaults; persist results |
| Backend capability detection wrong on edge hardware | Medium | Medium | Conservative defaults; manual override knobs |
| Prefix fingerprint invalidation on application context changes | Medium | High | Versioned hash schema; partial invalidation (product-side) |
| P-core/E-core detection unreliable on older hardware | Low | Low | Fall back to uniform thread pool |

---

# Post-Implementation Checkpoint

After ALL modelai-llama.cpp code changes are complete, confirm the following integration points:

## ModelAI Application-Side (verify with product team)

1. **Agent swarm flow** — how agent workers call the compaction endpoint
2. **`/compact` endpoint** — server-side routing for select/omp/self-study methods
3. **Orchestrator logic** — when to compact, which method to use, threshold decisions
4. **Capability consumption** — product reads `/props` `compacted_prefix.available` and `compacted_prefix.enabled` to gate UI features
5. **FA override awareness** — product detects when compaction forces non-flash attention and shows appropriate UX
6. **Backend-aware defaults** — product consumes `llama_backend_caps` for initial configuration on fresh install

## Server-Side Integration (verify in CI)

1. **`/props` reports truth** — compaction availability reflects loaded model capabilities
2. **`/metrics` reports truth** — Prometheus metrics show real compaction state
3. **Structured output reliable** — no crashes or silent fallbacks under sustained schema workloads
4. **Tool calls safe** — malformed tool calls produce errors, not crashes
5. **Prompt cache coherent** — cached prompts survive across turns, hybrid models don't re-prefill

## Definition of Done

This plan is complete when:
1. Self-study query generation produces measurably better compaction quality than cache-key surrogates
2. All supported model/backend paths are covered and tested
3. Server capability reporting reflects real compaction state
4. Structured output is correct and stable under repeated load
5. Tool calls cannot crash the server
6. Prompt-cache behavior is bounded and predictable
7. All tests pass on Metal backend (Apple Silicon primary target)
8. Fork summary support matrix matches reality
9. At least one real ModelAI workload closure gate passes with all of:
   - workload class W2 or W3
   - model size >= 1B params
   - prefix length >= 2048 real-text tokens
   - quality threshold cosine similarity >= 0.95 versus the full-context baseline
