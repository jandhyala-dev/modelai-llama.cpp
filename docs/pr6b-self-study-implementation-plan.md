# PR-6b: Self-Study Query Generation — Full Implementation Plan

## Owner
Ajay Jandhyala — ajay@model-ai.app (COT Labs / ModelAI)

## Repository
`jandhyala-dev/modelai-llama.cpp` — branch `modelai-main`

## Context
This plan covers ONLY the modelai-llama.cpp (inference runtime) side of PR-6b.
ModelAI application-side changes (agent swarm flow, `/compact` endpoint, orchestrator logic) are tracked separately and will be confirmed after all modelai-llama.cpp work is complete.

---

## What Self-Study Is

Self-study replaces the ~256 cache-key surrogate queries (K vectors used as proxy Q vectors) with real Q tensors captured from forward passes. The MIT paper (arXiv:2602.16284, Figure 4) identifies this as the single largest quality factor for KV cache compaction — worth more than all other optimizations combined.

Current state: `llama_kv_compact_extract_cache_key_queries()` in `src/llama-kv-compact-query.cpp` extracts K vectors as surrogate queries, uniformly sampled down to `max_queries` (default 256). These fail on GQA architectures because K and Q live in different spaces.

Self-study: Generate ~256 continuation tokens from the prefix, capture the real post-RoPE Q tensors via the existing `cb_eval` callback mechanism, regroup across GQA heads, and feed into the existing solver pipeline (`fit_beta` + `fit_values`).

---

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

---

## File-by-File Changes

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

### MODIFIED FILES

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

#### `src/llama-kv-cache.h`

**Add** method in `llama_kv_cache` class:
```cpp
bool compacted_prefix_self_study_from_live_kv(
        struct llama_context * ctx,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        const llama_kv_compact_self_study_config & config,
        llama_kv_compact_self_study_stats * stats = nullptr,
        llama_pos p0 = 0);
```

#### `src/llama-kv-cache.cpp`

**Add** wrapper implementation delegating to `llama_kv_compact_self_study_from_live_kv()`.

#### `src/llama-kv-cache-iswa.h`

**Add** same method declaration for iSWA cache class.

#### `src/llama-kv-cache-iswa.cpp`

**Add** delegation to `kv_base->compacted_prefix_self_study_from_live_kv(...)`.

#### `CMakeLists.txt`

**Add** to llama sources:
```cmake
src/llama-kv-compact-self-study.cpp
```

---

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

---

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

---

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

## Implementation Steps

| Step | What | Files | Risk | Effort |
|------|------|-------|------|--------|
| 6b-1 | `llama_set_eval_callback` public API | llama.h, llama-context.h, llama-context.cpp | None — trivial setter | 1 hour |
| 6b-2 | `llama_q_capture_state` + callback impl | llama-kv-compact-self-study.h/.cpp | Low — follows imatrix pattern | 1 day |
| 6b-3 | GQA regrouping + subsampling | llama-kv-compact-self-study.cpp | None — pure math | 1 day |
| 6b-4 | Generation loop with Q-capture + KV cleanup | llama-kv-compact-self-study.cpp | Medium — autoregressive sampling | 2 days |
| 6b-5 | Solver integration (reuse fit_beta/fit_values) | llama-kv-compact-self-study.cpp | Low — plug in query source | 1 day |
| 6b-6 | Pipeline wiring (cache class, iSWA, CMake) | llama-kv-cache.h/.cpp, iswa, pipeline.h, CMakeLists | Low — boilerplate | 0.5 day |
| 6b-7 | Tests + quality benchmarks | tests/test-kv-compact-self-study.cpp | Low | 1-2 days |
| 6b-8 | Server endpoint integration | llama-server (TBD) | Low | 1 day |

**Total: ~8-10 days**

---

## Supported Model Architectures

The "last 3D Qcur-prefixed" disambiguation strategy works for all standard attention models (85+ architectures). Known variants:

| Category | Models | Tensor name captured |
|----------|--------|---------------------|
| Standard post-RoPE | llama, qwen2, falcon, olmo, baichuan, exaone, ... | `"Qcur-{il}"` |
| With kq_norm | llama (use_kq_norm), llama-iswa | `"Qcur_normed-{il}"` |
| With q_norm | qwen3, qwen3next, apertus, step35-iswa | `"Qcur_normed-{il}"` or `"Qcur-{il}"` (post-norm then post-RoPE) |
| Temperature-scaled | mistral3, deepseek2 | `"Qcur_attn_temp_scaled-{il}"` |
| Gemma scaled | gemma | `"Qcur_scaled-{il}"` |

**Not supported (out of scope for v0):** MLA models (deepseek2 alternate path), non-attention models (Mamba, RWKV), models with non-standard Q names (lfm2, qwen3next alternate).

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| cb_eval overhead slows generation | Low | Low | Benchmarked at ~5-10% overhead; v1 optimization available |
| Graph reuse skips cb_eval reinstall | Medium | Medium | cb_eval persists on scheduler after first install; verify with test |
| Multiple Qcur copies waste bandwidth | Low | None | ~20KB per copy × 3 per layer × 28 layers = ~1.7MB total per step |
| KV cache full during generation | Low | Medium | Only needs ~256 extra slots; fail gracefully if insufficient |
| Autoregressive sampling complexity | Medium | Medium | Start with greedy (temperature=0); add sampling in v1 |
| Solver quality regression vs surrogates | Very Low | High | Self-study proven superior in MIT paper; test validates |
