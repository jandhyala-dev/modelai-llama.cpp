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
8. **Product optimization pipeline** — backend capability detection, autotuning, Excel-aware prefix caching

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

**Analysis:** Multiple streams are used for speculative decoding and parallel sampling. The compacted prefix mask already has an `n_stream` dimension (`dst->ne[3]` in `llama-kv-compacted-prefix-exec.cpp:96`), and the beta set_input handles streams (`dst->ne[3]` in `llama-kv-compacted-prefix-exec.cpp:187`). The assertion is conservative.

**Solution:** Remove the assertion and test with n_stream > 1. The exec functions already handle it.

**Files:**
- `src/llama-graph.cpp` — Remove P3 single-stream assertion at line 2089
- `tests/test-kv-compacted-prefix-exec.cpp` — Add multi-stream test case

**Effort:** 0.5 day

### 2c. M-RoPE Model Testing

**Problem:** Models using `ggml_rope_multi()` (Qwen2-VL, Qwen3-VL, GLM4) have not been tested with compaction. The Q-capture callback should still work since the post-RoPE Qcur is still named `"Qcur-{il}"` and is 3D.

**Solution:** Add a test using a Qwen3-VL model to verify Q-capture works with multi-dimensional RoPE. No code changes expected — just validation.

**Files:**
- `tests/test-kv-compact-quality.cpp` — Add M-RoPE model variant test

**Effort:** 0.5 day

### 2d. Backend Validation

**Problem:** Compacted prefix data is always CPU-allocated (`ggml_backend_cpu_buffer_type()` in `llama-kv-compacted-prefix.cpp:775`). The set_input functions assume host-backed tensors (`require_host_or_direct_data` in `llama-kv-compacted-prefix-exec.cpp:20-27`). This works on Metal and CUDA because the ggml scheduler copies data to GPU as needed.

**Solution:** No code changes needed. Add explicit backend validation tests:
- Metal: run quality test on Apple Silicon with `GGML_METAL=ON`
- CUDA: run quality test with `GGML_CUDA=ON` (CI or manual)
- CPU-only: run quality test with no accelerator

**Files:**
- `tests/test-kv-compact-quality.cpp` — Add backend-specific test annotations
- `docs/modelai-ci-policy.md` — Document backend test matrix

**Effort:** 0.5 day

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

The `modelai-performance-roadmap.md` already documents FlashBias (NeurIPS 2025, arXiv:2505.12044) as the solution. When ggml adds native flash attention bias support:
1. The `kq_b == nullptr` guard at `llama-graph.cpp:1883` can be relaxed
2. Beta tensor gets passed to `ggml_flash_attn_ext_bias()` (or equivalent)
3. Full solver pipeline works with flash attention

**Effort:** 0 (wait for upstream ggml)

### Option B: Encode Beta into Mask (Deferred — Invasive)

Fold per-head beta into the attention mask. This requires per-layer per-head masks instead of broadcast masks. The mask shape would change from `[n_kv, n_tokens, 1, n_stream]` to `[n_kv, n_tokens, n_head, n_stream]` per layer. This is a significant change to the graph construction and memory layout.

**Not recommended for v0.** Document as future option.

### Option C: Fallback to Non-Flash for Non-Zero Beta (Current Behavior)

The current assertions already enforce this: when beta is non-zero, flash attention is disabled and the standard path is used. This works correctly today.

**No code changes needed.** Document the tradeoff: flash is faster but selection-only; standard is slower but supports full solver with non-zero beta.

### Decision

**Adopt Option A (wait) + Option C (current fallback).** Add documentation explaining the tradeoff.

**Files:**
- `docs/pr6b-self-study-implementation-plan.md` — This document (already documents the decision)
- No code changes for flash + beta in this PR

**Effort:** 0 (documentation only)

---

# Part 4: Production Validation on ModelAI Workloads

## Validation Matrix

| Test | Model | Context | Target | Pipeline | Metric |
|------|-------|---------|--------|----------|--------|
| **4a. SEC Filing Extraction** | Qwen3-14B | 8K-32K token filing | 512 tokens | select, self-study | Answer accuracy vs full context |
| **4b. Long Document QA** | Qwen3-14B | 32K token document | 1024 tokens | select, self-study, OMP | ROUGE/F1 on extraction tasks |
| **4c. Multi-Filing Batch** | Qwen3-14B | 200 filings × 8K each | 512 tokens each | select (speed), self-study (quality) | Throughput (filings/min), accuracy |
| **4d. Vision Document** | Qwen3-VL-8B | Image + 4K text | 256 tokens | select | Answer accuracy |
| **4e. Small Model Smoke** | stories15M | 512 tokens | 64 tokens | all pipelines | No crashes, basic quality |

## Implementation

### `tests/test-kv-compact-workload.cpp`

End-to-end workload test:
1. Load model from `models/test/` directory
2. Prefill a real SEC filing text (or representative test fixture)
3. Run compaction with each pipeline mode
4. Run extraction query against compacted context
5. Compare output quality against full-context baseline
6. Report metrics: latency, peak memory, answer similarity

### `scripts/bench-kv-compact-workload.sh`

Benchmark script for manual validation:
- Runs all workload tests with timing
- Outputs CSV with metrics per test × pipeline × compression ratio
- Supports `--model` flag for different model paths

**Files:**
- `tests/test-kv-compact-workload.cpp` — New workload test
- `scripts/bench-kv-compact-workload.sh` — Benchmark script
- `tests/CMakeLists.txt` — Register workload test (LABEL "model")

**Effort:** 2-3 days

## Acceptance Criteria

**[Reviewer-2 Condition R2-2]** Quantitative pass/fail thresholds per test case and compression ratio.

### Per-Test Thresholds

| Test | Compression | Metric | Pass Threshold | Source |
|------|-------------|--------|----------------|--------|
| **4a. SEC Filing** | 2x | ROUGE-L vs full context | >= 0.95 | Conservative estimate |
| **4a. SEC Filing** | 5x | ROUGE-L vs full context | >= 0.90 | Conservative estimate |
| **4a. SEC Filing** | 10x | ROUGE-L vs full context | >= 0.85 | Conservative estimate |
| **4b. Long Doc QA** | 2x | F1 on extraction tasks | >= 0.93 | Conservative estimate |
| **4b. Long Doc QA** | 5x | F1 on extraction tasks | >= 0.85 | Conservative estimate |
| **4b. Long Doc QA** | 10x | F1 on extraction tasks | >= 0.75 | Conservative estimate |
| **4c. Multi-Filing** | 2x | Throughput vs full context | >= 1.5x speedup | Minimum viable benefit |
| **4c. Multi-Filing** | 5x | Accuracy vs full context | >= 0.88 | Must not regress |
| **4d. Vision Document** | 2x | Answer accuracy (exact match) | >= 0.90 | Conservative estimate |
| **4e. Small Model Smoke** | any | No crashes | 0 crashes | Hard requirement |

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

**Solution:** Keep the definition in `llama-kv-compact-select.h` only. Replace the forward declaration in `llama-kv-compact-pipeline.h` with an `#include`.

**Files:**
- `src/llama-kv-compact-pipeline.h` — Replace forward declaration with include

**Effort:** 15 minutes

### 5b. Public API Decision

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

## Workstream 7a: Structured JSON Schema Correctness (#10732, #19716)

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
6. **[CRITICAL] Add resource limits on grammar compilation** — the `_rules` map in `json-schema-to-grammar.cpp:318-336` has no size limit. Nested `allOf`/`oneOf` generates exponential rules. Recursive `visit()` has no depth limit (`json-schema-to-grammar.cpp:616-620`). Fix: add max depth limit (64), max rule count (10000), and compilation timeout.
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

**Feature Fixes:**

7. **Harden tool-call parsing path** (`common/chat.cpp`):
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

## Workstream 7c: Structured-Output Stability Under Repeated Load (#17391, #19068)

### Objective
Make repeated structured-output workloads stable under sustained load with no memory leaks or state corruption.

### Why ModelAI Cares
ModelAI workloads repeat the same extraction schema many times: many report sections, many company docs, repeated batch analysis. One-off correctness is insufficient — the runtime must survive hours of repetition.

### Implementation Changes

1. **Audit structured-output request lifecycle** in `server-context.cpp`:
   - Per-request grammar ownership verified
   - Task teardown on completion, cancellation, parse failure, timeout
   - No grammar/parser state leaks across requests
2. **Fix grammar trigger loop** (#19068): add error recovery when grammar sampler enters infinite trigger state. **Detection mechanism:** max iterations = 1000 per token; if exceeded, break with structured error "grammar loop detected at rule [rule_name]". Log the triggering grammar rule for debugging. **Recovery:** reset sampler state, return error to client, ensure slot is clean for next request.
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

   **Interaction with Part 8b (Excel-aware prefix caching):** The `prefix_fingerprint` API from Part 8b can be implemented ON TOP of this hash-based matching. The product computes a fingerprint (e.g., from workbook structure), the server uses it as an additional matching key alongside the token-hash. The two mechanisms are complementary: token-hash matches exact prompts, fingerprint matches semantically equivalent prompts.

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
Prevent thermal throttling on sustained workbook sessions by adapting thread scheduling to consumer hardware.

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
| **6b-15** | 5c | Upstream algorithm + integration docs | docs/ | None | 1-2 days |
| **6b-16** | 5d | Test hardening (negative cases) | tests/ | None | 0.5 day |

**Phase 1 Total: ~14-18 days**

## Phase 2: Compaction Runtime Integration (Part 6)

| Step | Part | What | Files | Risk | Effort |
|------|------|------|-------|------|--------|
| **6b-17** | 6a | Wire real compaction state into /props, /metrics | server-context.cpp/.h, llama-kv-cache.h/.cpp | Low | 1 day |
| **6b-18** | 6b | FA + compaction mode selection + observability | llama-graph.cpp, server-context.cpp | Low | 0.5 day |
| **6b-19** | 6c | Fix quantized K compaction spec drift | test-kv-compacted-prefix.cpp, test-kv-compact-quality.cpp | Low | 0.5 day |
| **6b-20** | 6d | Update fork summary support matrix | modelai-fork-summary.md | None | 0.5 hour |

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
| Quantized K spec drift breaks existing tests | High | Low | Fix test immediately (6b-19); test now validates acceptance, not rejection |
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
| Excel prefix hashing breaks on workbook structure changes | Medium | High | Versioned hash schema; partial invalidation |
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
