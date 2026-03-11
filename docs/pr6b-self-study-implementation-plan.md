# PR-6b: KV Compaction Completion — Full Implementation Plan

## Owner
Ajay Jandhyala — ajay@model-ai.app (COT Labs / ModelAI)

## Repository
`jandhyala-dev/modelai-llama.cpp` — branch `modelai-main`

## Scope
This plan covers ALL remaining modelai-llama.cpp work to bring KV compaction to production readiness. It addresses five gaps identified against the full RFC vision:

1. **Self-study query generation** — highest-quality query path from the paper
2. **Model/backend coverage expansion** — every supported model/backend path
3. **Flash attention + non-zero beta** — final flash-attention story
4. **Production validation** — hardened proof on ModelAI workloads
5. **Upstream-ready sanitization** — clean API surface, docs, naming

ModelAI application-side changes (agent swarm flow, `/compact` endpoint, orchestrator logic) are tracked separately and will be confirmed after all modelai-llama.cpp work is complete.

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

# Implementation Schedule

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

**Total: ~14-18 days**

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

---

# Post-Implementation Checkpoint

After ALL modelai-llama.cpp code changes are complete, confirm the following ModelAI application-side integration points before considering PR-6b done:

1. **Agent swarm flow** — how agent workers call the compaction endpoint
2. **`/compact` endpoint** — server-side routing for select/omp/self-study methods
3. **Orchestrator logic** — when to compact, which method to use, threshold decisions
