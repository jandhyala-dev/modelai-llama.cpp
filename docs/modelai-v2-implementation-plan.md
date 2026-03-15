# ModelAI llama.cpp — V2 Implementation Plan

**Plan commit:** `(this commit)`
**V1 baseline commit:** `0a637086` (V1 complete — 15 models validated, all tests pass)
**Upstream base:** `0cd4f472` (upstream-master)
**Date:** 2026-03-15
**Owner:** Ajay Jandhyala — ajay@model-ai.app (COT Labs / ModelAI)

## Revision History

| Date | Version | Change |
|------|---------|--------|
| 2026-03-15 | v1 | Initial V2 plan: 8 phases, GPU solver + production solver + flash+beta + upstream sync + SWA + 50x + 128K |

---

## Executive Summary

### What V1 Delivered (Complete)

- KV cache compaction via select pipeline — **production-ready at 2-8x compression**
- 15 models validated across 7 architectures, 51 integration tests each, all pass
- Select pipeline cosine >= 0.950 on all models
- Baseline performance parity with upstream llama.cpp (within +/- 3% noise)
- Server integration: `/compact`, `/props`, `/metrics` with pipeline allowlist
- Per-stage timing instrumentation, serialization round-trip, B5 GPU-resident staging
- Solver pipeline functional but **excluded from production** (poor GQA quality: cosine -0.17 to 0.91)

### What V2 Targets

V2 extends compaction from the 2-8x select-only production path to a **10-20x production-quality solver pipeline** with GPU acceleration, flash attention compatibility, and broader model support.

| Goal | V1 State | V2 Target |
|------|----------|-----------|
| Production compression ratio | 2-8x (select only) | **10-20x** (solver + nonuniform) |
| Solver quality on GQA models | -0.17 to 0.91 cosine (broken) | **>= 0.90** cosine (production-grade) |
| Compaction latency (14B, 4K ctx) | ~2-5s on CPU | **< 500ms** (GPU solver) |
| Flash attention with compacted prefix | Blocked (falls back to non-FA) | **Supported** (FlashBias or hybrid) |
| Model coverage | 15 models (3 incompatible) | **18+ models** (upstream sync) |
| SWA compaction | Base cache only | **Full iSWA** (base + SWA sub-cache) |
| 50x compression | Not attempted | **Research capability** (quality-gated) |
| 128K context validation | Not tested | **Planned** (deferred implementation) |

### Honest Quality Assessment at High Compression

From the MIT paper's own benchmarks (Qwen3-4B, QuALITY):

| Ratio | Accuracy | Quality Loss | Production Viable? |
|-------|----------|:------------:|:------------------:|
| 2x | ~71.5% | ~0% | Yes |
| 5x | ~70% | ~2% | Yes |
| 10x | ~67% | ~6% | Yes (with caveats) |
| 20x | ~60% | ~16% | Marginal |
| 50x | ~55% | ~23% | Research only |

**V2 production target: 10x with >= 0.90 cosine. 20-50x as opt-in experimental capability.**

### What Is Explicitly Out of Scope

- KV cache quantization stacking (Q8_0/Q4_0 on compacted entries) — V3
- Speculative decoding integration — V3
- Prompt cache sharing with compacted prefixes — V3
- Public API guarantees — V3
- Multi-slot compaction — V3
- CUDA/Vulkan backend support — V3 (V2 is Metal-only for GPU solver)
- On-policy sequential compaction (compact layer l, use for layer l+1) — research track
- Training-based compression (Cartridges, KV-Distill) — not applicable

---

## Phase 1: Upstream Sync

**Goal:** Resolve 3 model incompatibilities (Gemma3-12B, GPT-OSS-20B, Phi4-14B) and pick up Metal kernel improvements.

**Risk: HIGH** — upstream llama.cpp has diverged significantly. Merge conflicts in KV cache code are expected.

### 1.1 Upstream Merge Strategy

The upstream-master branch tracks `ggml-org/llama.cpp` master. The merge must:

1. **Identify the target upstream commit** — pick a stable point after Gemma3 arch support, GPT-OSS arch support, and Phi4 graph fix land
2. **Cherry-pick vs full merge** — full merge preferred for long-term maintainability, but cherry-pick if merge conflicts in KV cache code are intractable
3. **Preserve all compaction code** — every file in `src/llama-kv-compact-*`, `src/llama-kv-compacted-prefix-*`, and compaction-related changes in `src/llama-kv-cache.*` must survive the merge
4. **Re-run full V1 test suite** — all 15 models × 51 tests must still pass after merge

### 1.2 Target Upstream Features

| Feature | Why | Upstream Reference |
|---------|-----|--------------------|
| Gemma3 architecture support | Unblocks Gemma3-12B model loading | Merged upstream ~Feb 2026 |
| GPT-OSS architecture support | Unblocks GPT-OSS-20B model loading | `gptoss` arch registration |
| Phi4 graph hash set fix | Unblocks Phi4-14B (GGML assertion) | GGML hash set sizing |
| Fused multiply-add for Q4/Q5/Q6_K | 16-28% faster prompt processing | [PR #20032](https://github.com/ggml-org/llama.cpp/pull/20032) |
| Metal mul_mv_ext for BF16/Q2_K/Q3_K | Faster Metal kernels | [PR #20250](https://github.com/ggml-org/llama.cpp/pull/20250) |
| KV cache defrag fixes | Defrag bug can corrupt data | [PR #10873](https://github.com/ggerganov/llama.cpp/pull/10873) |

### 1.3 Merge Verification

After merge:
- [ ] Build succeeds: `cmake -B build -DGGML_METAL=ON && cmake --build build --config Release`
- [ ] ctest passes: `ctest --test-dir build -L main --output-on-failure`
- [ ] All 15 V1 models pass 51/51 integration tests
- [ ] Gemma3-12B loads and passes integration tests
- [ ] GPT-OSS-20B loads and passes integration tests
- [ ] Phi4-14B loads and passes integration tests
- [ ] Select pipeline cosine >= 0.950 on all models
- [ ] Baseline tok/s within 5% of V1 numbers (no regression from merge)

### 1.4 Estimated Effort

**High.** Upstream KV cache code has been refactored (`llama_kv_cells_unified`, SWA changes). Expect 2-5 days of merge conflict resolution and verification.

---

## Phase 2: GPU Solver via Metal Compute Shaders (B4)

**Goal:** Move the attention score computation and matrix operations from CPU to GPU, reducing compaction latency from ~2-5s to <500ms for 14B models at 4K context.

### 2.1 What Moves to GPU

| Operation | Current (CPU) | GPU Target | Speedup |
|-----------|--------------|------------|---------|
| Attention score matrix: `score[q][k] = dot(Q[q], K[k]) / sqrt(d)` | O(n_q × n_k × d) scalar loops with NEON | Metal compute shader, thousands of threads | **10-15x** |
| Exp + softmax normalization | Per-row sequential | Per-row parallel reduction | **5-8x** |
| Matrix multiply (X^T X for normal equations) | Scalar with NEON 4-wide | Metal GEMM | **10-20x** |
| K/V extraction from KV cache | `ggml_backend_tensor_get()` per-head → CPU | Direct GPU buffer read (no host roundtrip) | **Eliminates transfer** |

### 2.2 What Stays on CPU

| Operation | Why |
|-----------|-----|
| Cholesky decomposition | Sequential pivot-dependent — GPU parallelism doesn't help |
| NNLS projected gradient (50 iterations) | Small problem size (t × t where t = compacted tokens), sequential convergence |
| Power iteration for spectral norm (8 iterations) | Sequential, small matrix |
| Pipeline orchestration | Control flow |

### 2.3 Implementation Plan

**2.3.1 Metal Shader: Attention Score Matrix**

New file: `ggml/src/ggml-metal/ggml-kv-compact-solver.metal`

```metal
kernel void kv_compact_attention_scores(
    device const float * queries  [[buffer(0)]],   // [n_q × d]
    device const float * keys     [[buffer(1)]],   // [n_k × d]
    device float       * scores   [[buffer(2)]],   // [n_q × n_k]
    constant uint      & n_q      [[buffer(3)]],
    constant uint      & n_k      [[buffer(4)]],
    constant uint      & d        [[buffer(5)]],
    constant float     & inv_sqrt_d [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Each thread computes one score[qi][ki]
    uint qi = gid.y;
    uint ki = gid.x;
    if (qi >= n_q || ki >= n_k) return;

    float sum = 0.0f;
    for (uint i = 0; i < d; i++) {
        sum += queries[qi * d + i] * keys[ki * d + i];
    }
    scores[qi * n_k + ki] = sum * inv_sqrt_d;
}
```

**2.3.2 Metal Shader: Softmax with Max-Shift**

Per-row softmax with shared-memory max reduction.

**2.3.3 Metal Shader: GEMM for Normal Equations**

X^T X computation for the least-squares system. Can leverage Metal Performance Shaders (MPS) `MPSMatrixMultiplication` for this.

**2.3.4 GPU K/V Extraction**

Replace `ggml_backend_tensor_get()` (GPU→CPU copy) with direct GPU buffer access:
- Allocate Metal buffers for extracted K/V matrices
- Copy within GPU: KV cache tensor → solver input buffer
- Solver shaders read directly from these buffers
- Only final results (beta, compacted V) are read back to CPU for storage

**2.3.5 Host-Side Dispatch**

New file: `src/llama-kv-compact-solver-metal.mm` (Objective-C++ for Metal API)

```
llama_kv_compact_solver_metal_context:
  - init(device, command_queue)
  - compute_attention_scores(queries, keys) → scores_buffer
  - compute_softmax(scores_buffer) → exp_scores_buffer
  - compute_normal_equations(X, Y) → XtX, XtY
  - readback_results() → host memory
```

### 2.4 Performance Budget

For Qwen3-14B at 4K context (40 layers × 8 KV heads = 320 invocations):

| Phase | V1 CPU Time | V2 GPU Time | Notes |
|-------|:-----------:|:-----------:|-------|
| K/V extraction | ~800ms | ~50ms | Eliminate host roundtrip |
| Attention scores | ~1200ms | ~80ms | GPU parallel dot products |
| Softmax | ~200ms | ~20ms | GPU parallel reduction |
| NNLS (50 iters) | ~300ms | ~300ms | Stays on CPU (small problem) |
| Cholesky + V solve | ~200ms | ~200ms | Stays on CPU (sequential) |
| **Total** | **~2700ms** | **~650ms** | **~4x speedup** |

Conservative estimate: **4x speedup** (not 10x — NNLS/Cholesky remain sequential bottlenecks).
At 8K context: CPU ~8-12s → GPU ~2-3s.

### 2.5 Testing

- [ ] Metal shader unit tests: attention scores match CPU reference within fp32 tolerance
- [ ] Full solver pipeline produces identical beta/V to CPU path (cosine >= 0.9999)
- [ ] No memory leaks (Metal buffer lifecycle)
- [ ] Graceful CPU fallback when Metal is unavailable
- [ ] Per-stage timing shows GPU phases faster than CPU phases

### 2.6 Files Created/Modified

| File | Action |
|------|--------|
| `ggml/src/ggml-metal/ggml-kv-compact-solver.metal` | NEW — Metal compute shaders |
| `src/llama-kv-compact-solver-metal.h` | NEW — Metal solver API |
| `src/llama-kv-compact-solver-metal.mm` | NEW — Metal solver implementation |
| `src/llama-kv-compact-pipeline.cpp` | MODIFY — dispatch to GPU solver when available |
| `CMakeLists.txt` | MODIFY — Metal shader compilation |
| `tests/test-kv-compact-solver-metal.cpp` | NEW — GPU solver tests |

### 2.7 Estimated Effort

**High.** Metal compute shader development + Objective-C++ interop + buffer management. Expect 3-5 days for a working prototype, 1-2 more for optimization and testing.

---

## Phase 3: Production Solver Pipeline

**Goal:** Fix the GQA quality degradation that makes the solver pipeline unusable on most production models (cosine -0.17 to 0.91 → target >= 0.90 on all models).

### 3.1 Root Cause Analysis

The V1 solver uses **cache-key-as-query surrogates**: it takes keys from the KV cache and uses them as pseudo-queries to compute attention scores. This fails on GQA models because:

1. **Key space ≠ query space** — GQA models have fewer KV heads than query heads (4:1 ratio on Qwen3). Keys are optimized for shared attention, not for representing individual query patterns.
2. **No GQA regrouping** — the surrogate doesn't account for the query-head-to-KV-head mapping.
3. **Small surrogate count** — V1 uses ~256 surrogates vs the paper's ~50,000 self-study queries.

### 3.2 Fix Path: Improved Query Generation

Three approaches, in order of implementation priority:

**3.2.1 Self-Study Query Generation (Paper Method)**

The paper's recommended approach: run a second forward pass through the model with random input tokens to capture actual query vectors from each attention head.

- File: `src/llama-kv-compact-self-study.cpp` (exists, needs quality fixes)
- Change: Increase query count from ~256 to ~5,000-10,000 per head
- Change: Proper GQA regrouping — map query heads to KV heads before scoring
- Reference: `compaction/query_generation/self_study.py` lines 1-800
- **Cost:** Requires a model forward pass (~1-3s for 14B). This is the dominant cost, not the solver math.
- **Quality:** Paper shows self-study is the single most impactful quality factor (Figure 4 ablation)

**3.2.2 Prefill-Q Query Extraction**

Extract actual queries from the original prefill computation (before compaction). Zero additional cost — queries are a byproduct of the prefill.

- File: `src/llama-kv-compact-query.cpp` (exists, prefill_q pipeline exists)
- Change: Store prefill queries in a buffer during initial prompt processing
- Change: Use these real queries for solver fitting
- **Cost:** Zero additional compute (queries already computed during prefill)
- **Quality:** Limited to queries from the original prompt — doesn't generalize to future queries
- **Best for:** Same-session compaction where future queries are similar to past queries

**3.2.3 Hybrid: Prefill-Q + Self-Study Supplement**

Use prefill queries as the base, supplement with a shorter self-study run (~1,000 queries) for generalization.

- **Cost:** ~500ms additional (shorter self-study)
- **Quality:** Best of both — real query signal + generalization coverage

### 3.3 Nonuniform Per-Head Budgets

Currently all heads get the same compression ratio. The paper shows some heads tolerate 50x while others need 2x.

- File: `src/llama-kv-compact-budget.cpp` (exists, uniform only)
- Change: Implement head sensitivity curve precomputation per Algorithm 4 (Section 3.4)
- Change: Per-head budget allocation based on attention entropy
- Reference: `compaction/compaction_methods/per_layer_head.py`
- **Impact:** Critical for 10x+ compression. Without this, aggressive compression destroys high-sensitivity heads.

### 3.4 Ridge Scaling

Current: fixed `lambda=1e-6` with 10x escalation on failure.
Paper: spectral/Frobenius normalization of the regularization parameter.

- File: `src/llama-kv-compact-solver.cpp` (spectral ridge exists but disabled by default)
- Change: Enable spectral ridge by default, tune lambda per model family
- **Impact:** Minor quality improvement, major numerical stability at extreme ratios.

### 3.5 Quality Targets

After Phase 3, the solver pipeline must achieve:

| Model | 2x Cosine | 4x Cosine | 10x Cosine |
|-------|:---------:|:---------:|:----------:|
| All standard causal | >= 0.95 | >= 0.92 | >= 0.90 |
| GQA models (Qwen, Llama) | >= 0.95 | >= 0.92 | >= 0.88 |
| MoE models (Qwen3-30B-A3B) | >= 0.97 | >= 0.95 | >= 0.92 |

If these targets are not met, the solver remains experimental (not added to production allowlist).

### 3.6 Testing

- [ ] Solver cosine >= 0.90 on all 15 V1 models at 2x (currently as low as -0.17)
- [ ] Solver cosine >= 0.88 on GQA models at 10x
- [ ] Self-study query generation produces >= 5,000 queries per head
- [ ] GQA regrouping correctly maps query heads to KV heads
- [ ] Nonuniform budgets: high-sensitivity heads get more tokens than low-sensitivity heads
- [ ] Timing: total solver pipeline < 5s on 14B at 4K (with GPU solver from Phase 2)

### 3.7 Estimated Effort

**Medium-High.** Self-study quality fixes are the critical path. 3-5 days.

---

## Phase 4: Flash Attention + Beta

**Goal:** Allow flash attention to work with compacted prefixes that have non-zero beta, eliminating the non-FA fallback path.

**Status: BLOCKED on upstream.** The flash-attention path in llama.cpp does not support additive `kq_b` (the beta bias). There are two possible resolution paths.

### 4.1 Path A: FlashBias Implementation (Preferred)

FlashBias ([arXiv:2505.12044](https://arxiv.org/abs/2505.12044), NeurIPS 2025) extends flash attention with efficient additive bias support.

**What's needed:**
1. Implement FlashBias Metal kernel that accepts an additive bias tensor alongside Q, K, V
2. Integrate with llama.cpp's attention graph building (`src/llama-graph.cpp`)
3. The bias tensor is the compacted prefix's beta values, shaped `[n_compacted_tokens]` per head

**Difficulty: VERY HIGH.** This requires writing a custom flash attention Metal kernel from scratch or modifying the existing one. The existing Metal FA kernel is already complex (~500 lines of Metal shader code).

**Alternative:** Wait for upstream llama.cpp to adopt FlashBias. Monitor [flash-attention Issue #1219](https://github.com/Dao-AILab/flash-attention/issues/1219).

### 4.2 Path B: Hybrid Attention (Pragmatic)

Split the attention computation:
- **Live KV suffix:** uses flash attention (no beta needed)
- **Compacted prefix:** uses standard attention with beta (non-FA, as today)
- **Combine:** weighted sum of the two attention outputs

**What's needed:**
1. Modify attention graph to split Q into two paths at the compacted prefix boundary
2. FA path handles live KV (the majority of computation at long contexts)
3. Non-FA path handles compacted prefix (small — typically 128-512 tokens)
4. Merge attention outputs with proper softmax normalization

**Difficulty: MEDIUM-HIGH.** Requires attention graph surgery but no new Metal kernels.

**Performance benefit:** At 16K context with 2x compaction to 8K:
- FA handles 8K live tokens (fast)
- Non-FA handles 128-512 compacted tokens (small, fast enough)
- vs current: non-FA handles all 8K+ tokens (slow)

### 4.3 Recommendation

**Implement Path B (Hybrid) first.** It's achievable without upstream support and gives most of the performance benefit. Path A (FlashBias) is a future optimization.

### 4.4 Testing

- [ ] Hybrid attention produces identical logits to non-FA path (cosine >= 0.9999)
- [ ] Performance: pp512 with hybrid path >= 90% of pure FA performance
- [ ] Performance: tg128 with hybrid path >= 95% of pure FA performance
- [ ] Quality: select pipeline cosine unchanged from V1 values

### 4.5 Estimated Effort

Path B (Hybrid): **Medium-High.** 3-5 days. Attention graph modification is delicate.
Path A (FlashBias): **Very High.** 2-4 weeks. Metal kernel development.

---

## Phase 5: SWA Full Compaction

**Goal:** Extend compaction to the SWA (Sliding Window Attention) sub-cache, currently rejected by `compacted_prefix_runtime_supported()`.

### 5.1 Current State

- V1: `compacted_prefix_runtime_supported()` rejects SWA caches (`n_swa > 0`)
- Gemma2-9B (iSWA): compaction works on base cache, SWA sub-cache rejected
- This is correct behavior — SWA tokens expire naturally (sliding window eviction)

### 5.2 Why Compact SWA?

SWA tokens that are still within the window are actively used for attention. Compacting them could:
1. Allow higher effective compression on iSWA models
2. Reduce active KV count for the SWA portion
3. Enable compaction on models where SWA layers dominate

### 5.3 Challenges

1. **SWA eviction interaction** — the sliding window naturally evicts old tokens. Compacted tokens in the SWA cache must respect the window boundary.
2. **Different attention patterns** — SWA layers attend locally (within window), not globally. Compaction selects tokens based on global attention patterns. Need SWA-specific selection.
3. **Window-aligned compaction** — compact only tokens that are about to exit the window, not the full SWA cache.

### 5.4 Implementation

1. New function: `compacted_prefix_runtime_supported_swa()` — separate guard for SWA compaction
2. SWA-specific selection: compact tokens in the `[window_start, window_start + window_size/2]` range (oldest half of window)
3. SWA-specific quality metric: attention pattern match within window context, not global
4. Integration with iSWA models: compact base cache (existing) + compact SWA sub-cache (new)

### 5.5 Testing

- [ ] Gemma2-9B: SWA sub-cache compaction succeeds (currently rejected)
- [ ] Gemma2-9B: Quality after SWA compaction >= 0.90 cosine
- [ ] SWA window boundary correctly respected
- [ ] iSWA model: both base and SWA caches compacted independently
- [ ] Non-iSWA models: behavior unchanged from V1

### 5.6 Dependencies

- Phase 1 (Upstream Sync) — may need newer SWA code from upstream
- Gemma3-12B availability — need a working Gemma3 load to test full iSWA path

### 5.7 Estimated Effort

**Medium.** 2-3 days. The main complexity is the SWA-specific selection logic.

---

## Phase 6: High Compression (10-50x)

**Goal:** Enable 10-20x production compression and 50x research capability using the full paper algorithm.

**Dependencies:** Phase 2 (GPU solver) and Phase 3 (production solver) must be complete first.

### 6.1 Chunked Compaction

For contexts > 8K tokens, the paper uses KV-based chunking (~12K tokens/chunk):

1. Split the prefix into chunks of ~12K tokens
2. Compact each chunk independently
3. Handle RoPE phase alignment across chunk boundaries
4. Merge compacted chunks into a single compacted prefix

- Reference: Section 3.5 and Appendix C.3 of the paper
- File: `src/llama-kv-compact-pipeline.cpp` — chunked pipeline exists but needs production hardening
- **Impact:** Required for compacting long contexts (16K+) at high ratios

### 6.2 OMP Key Selection

Orthogonal Matching Pursuit produces better key selection than top-k at >10x compression:

1. V1 has OMP implemented but it's slow (~23 minutes for 2x on 14B at 4K)
2. OMP-fast variant (k=4, tau=2) per paper: ~104s on H200, proportionally slower on M3 Pro
3. With GPU solver (Phase 2): OMP scoring moves to GPU, reducing to ~10-30s
4. OMP quality advantage only manifests at >10x — at 2-8x, top-k is adequate

- File: `src/llama-kv-compact-select.cpp` — OMP exists, needs GPU acceleration
- **Decision:** OMP remains experimental/opt-in. top-k + nonuniform budgets is the production path for 10x.

### 6.3 50x Compression Profile

Based on the paper, 50x requires:

| Component | Required | Status |
|-----------|----------|--------|
| Self-study queries (~50K/head) | Yes | Phase 3.2.1 |
| OMP-fast key selection | Yes | Phase 6.2 |
| Nonuniform per-head budgets | Yes | Phase 3.3 |
| Chunked compaction | Yes | Phase 6.1 |
| Ridge scaling | Yes | Phase 3.4 |

**Quality at 50x:** expect 15-25% accuracy loss on benchmarks. This is a **research capability**, not a production default. Exposed via `LLAMA_COMPACT_ALLOWED_METHODS=solver,omp` environment variable.

### 6.4 Production 10x Profile

The default production path at 10x:

| Component | Method |
|-----------|--------|
| Query generation | Prefill-Q + short self-study supplement |
| Key selection | top-k with nonuniform per-head budgets |
| Beta fitting | GPU-accelerated NNLS (Phase 2) |
| V fitting | GPU-accelerated least-squares (Phase 2) |

**Expected quality at 10x:** cosine >= 0.88 on standard causal models, >= 0.90 on MoE.

### 6.5 Testing

- [ ] 10x compression: cosine >= 0.88 on all 15+ models
- [ ] 20x compression: cosine >= 0.85 on Qwen3-8B, Qwen3-30B-A3B
- [ ] 50x compression: cosine measured and documented (no pass/fail threshold)
- [ ] Chunked compaction at 16K context: quality matches single-block at same ratio
- [ ] OMP selection at 10x outperforms top-k at 10x (measured quality delta)

### 6.6 Estimated Effort

**Medium.** Most components exist. 2-3 days to integrate and tune. OMP GPU acceleration is the hardest part.

---

## Phase 7: 128K Context Validation

**Goal:** Implement and validate compaction quality and performance at 128K context. Test on models that fit in 32GB (3B-7B). Document hardware requirements for 14B+ at 128K.

### 7.1 Hardware Constraints

| Model Size | KV @ 128K | Total RAM | Testable on 32GB Mac? |
|:----------:|:---------:|:---------:|:---------------------:|
| 3B | ~2.6 GB | ~4.5 GB | **Yes** |
| 7B | ~5.2 GB | ~9.6 GB | **Yes** |
| 8B | ~6.6 GB | ~11.5 GB | **Tight — test with care** |
| 14B | ~10+ GB | ~19+ GB | No (swap, meaningless results) |
| 30B MoE | ~10+ GB | ~27+ GB | No |

### 7.2 Implementation: 128K Integration Tests

New test file: `tests/test-kv-compact-128k.cpp`

Tests run on 3B and 7B models at 128K context with select and solver pipelines:

**7.2.1 Quality at 128K**

| Model | Context | Ratios | Pipeline | Target Cosine | Memory Risk |
|-------|---------|--------|----------|:-------------:|:-----------:|
| Llama3.2-3B | 128K | 2x, 4x, 8x, 16x | select | >= 0.90 | Safe (~4.5GB) |
| Qwen2.5-7B | 128K | 2x, 4x, 8x, 16x | select | >= 0.90 | Safe (~9.6GB) |
| Qwen3-8B | 128K | 2x, 4x, 8x | select | >= 0.85 | Tight (~11.5GB) — monitor RSS |
| Qwen3-14B | 128K | 2x, 4x | select | >= 0.85 | Very tight (~19GB) — monitor RSS, kill if swap detected |

**7.2.2 Performance at 128K**

| Test | Measurement | Target |
|------|-------------|--------|
| Compaction latency (select) | Time to compact 128K → 64K (2x) | < 2s on 3B, < 5s on 7B |
| Compaction latency (solver) | Time to compact 128K → 64K (2x) | < 5s on 3B with GPU solver |
| Post-compaction decode tok/s | Decode 128 tokens after compaction | Faster than uncompacted baseline |
| Memory savings | Peak RSS with 8x compaction vs none | Measurable reduction in active KV |
| Serialization round-trip | Save/restore compacted 128K state | Cosine >= 0.99 |

**7.2.3 Chunked Compaction at 128K**

128K context requires chunked compaction (single-block won't fit in solver memory). Tests must verify:

- [ ] Chunked compaction at 128K produces quality within 0.02 cosine of single-block at 8K
- [ ] RoPE phase alignment correct across chunk boundaries (no position discontinuities)
- [ ] Chunk merging produces a single contiguous compacted prefix
- [ ] Decode after chunked compaction produces finite, coherent logits

**7.2.4 Stress Tests**

- [ ] Compact 128K → 8K (16x) on Llama3.2-3B: quality and stability
- [ ] Compact 128K → 16K (8x) on Qwen2.5-7B: quality and stability
- [ ] Multiple compaction cycles: compact, decode, compact again — no memory leaks
- [ ] Serialization at 128K: save state, clear, restore, verify cosine

### 7.3 Server Integration at 128K

Test `/compact` endpoint with 128K context via llama-server:

- [ ] `/compact` succeeds at 128K with `method=select`, `ratio=2`
- [ ] `/props` reports correct `active_n_kv` after 128K compaction
- [ ] `/metrics` shows compaction timing at 128K
- [ ] Streaming decode after 128K compaction produces valid SSE chunks
- [ ] Memory stays within 32GB envelope for 3B-7B models

### 7.4 14B+ at 128K (Future Hardware)

When 64GB+ hardware is available (M4 Max, cloud instance, or provisioned server):

| Model | Context | Ratios | Pipeline | Target Cosine |
|-------|---------|--------|----------|:-------------:|
| Qwen3-14B | 128K | 2x, 4x, 8x | select + solver | >= 0.90 |
| DeepSeek-R1-14B | 128K | 2x, 4x | select | >= 0.90 |

These tests are **written and ready to run** but need 64GB+ to run without swap pressure.

Qwen3-30B-A3B at 128K is tested separately in Phase 8 (see below).

### 7.5 Estimated Effort

**Medium.** 2-3 days. Test code + chunked compaction hardening + server integration at 128K.

---

## Phase 8: Final Validation, Qwen3-30B-A3B 128K Stress Test, and Documentation

**Goal:** Stress-test the full V2 stack with the largest feasible model at 128K, update all docs, run comprehensive benchmarks.

### 8.1 Qwen3-30B-A3B at 128K — Capstone Stress Test

This is the final validation after all Phases 1-7 are complete. Qwen3-30B-A3B is the most demanding model that can potentially fit on 32GB (17.3GB weights + KV cache).

**Why this model:** MoE architecture with 30B total / 3B active. It achieved the best V1 select cosine (0.999) and the best solver cosine (0.906). If any model can handle 128K with compaction on 32GB, this is it — because compaction reduces the KV cache that would otherwise make it impossible.

**Protocol — run with extreme care:**

1. Close all other applications. Kill Ollama, browsers, anything consuming memory.
2. Monitor memory continuously: `vm_stat 1` in a separate terminal
3. Start with the smallest test first: 128K context, 8x compaction (reduces KV from ~10GB to ~1.25GB)
4. If RSS stays under 28GB and no swap activity: proceed to 4x, then 2x
5. **Abort immediately** if swap pages start increasing — results under swap are meaningless

| Test | Context | Ratio | Expected KV After Compaction | Total RAM Needed | Feasible? |
|------|---------|:-----:|:----------------------------:|:----------------:|:---------:|
| A | 128K | 8x | ~1.25 GB | ~18.5 GB | **Likely yes** |
| B | 128K | 4x | ~2.5 GB | ~19.8 GB | **Probably** |
| C | 128K | 2x | ~5.0 GB | ~22.3 GB | **Tight** |
| D | 128K | 16x | ~0.6 GB | ~17.9 GB | **Yes (best shot)** |

**Success criteria:**
- [ ] At least one compaction ratio completes without swap
- [ ] Post-compaction decode produces finite, coherent logits
- [ ] Quality: select cosine >= 0.85 (relaxed threshold for extreme conditions)
- [ ] Serialization round-trip works at 128K compacted state
- [ ] Compaction latency documented (GPU solver if available from Phase 2)

**If Qwen3-30B-A3B at 128K works on 32GB Mac with compaction, this is the headline result:**
> "Run a 30B model with 128K context on a 32GB laptop — only possible with KV compaction."

### 8.2 Documentation Updates

| Document | Change |
|----------|--------|
| `docs/modelai-fork-summary.md` | V2 support matrix, GPU solver, production solver quality, SWA compaction |
| `docs/modelai-v2-benchmark-results.md` | NEW — full model matrix at 2x, 4x, 10x, 20x with solver pipeline |
| `docs/modelai-performance-roadmap.md` | Update B4 to DONE, mark Phase 5 items as DONE |
| `docs/modelai-compaction-support-envelope.md` | Expand support tiers: 10-20x supported, 50x experimental |
| `docs/modelai-v1-benchmark-results.md` | Add V1→V2 comparison section |
| `CLAUDE.md` | Update V0 Support Matrix to V2 Support Matrix |

### 8.3 Benchmark Plan

**8.3.1 Solver Quality Matrix**

Run solver pipeline on all 18+ models at 2x, 4x, 10x, 20x. Compare to V1 select-only baseline.

**8.3.2 GPU Solver Performance**

Measure compaction latency (CPU vs GPU) across model sizes and context lengths.

**8.3.3 Flash Attention Hybrid**

Measure decode performance with hybrid FA path vs pure non-FA path.

**8.3.4 SWA Compaction**

Measure quality and performance on Gemma2-9B and Gemma3-12B with full SWA compaction.

**8.3.5 128K Context Results**

Full quality and performance data at 128K for all tested models (3B, 7B, 8B, 14B, 30B MoE).

### 8.4 Estimated Effort

**Medium.** 2-3 days for stress test + docs + benchmarks.

---

## Dependency Graph

```
Phase 1: Upstream Sync
    |
    +--→ Phase 2: GPU Solver (B4)
    |        |
    |        +--→ Phase 3: Production Solver
    |        |        |
    |        |        +--→ Phase 6: High Compression (10-50x)
    |        |
    |        +--→ Phase 4: Flash Attention + Beta
    |
    +--→ Phase 5: SWA Full Compaction
    |
    +--→ Phase 7: 128K Validation (3B-7B local, 14B+ future hardware)
    |
    +--- All above --→ Phase 8: Documentation & Benchmarks
```

Phases 2-5 can partially parallelize after Phase 1 completes:
- Phase 2 (GPU solver) and Phase 5 (SWA) are independent
- Phase 3 (production solver) depends on Phase 2
- Phase 4 (flash+beta) depends on Phase 2
- Phase 6 (high compression) depends on Phases 2 + 3

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| Upstream merge conflicts in KV cache code | HIGH | HIGH | Cherry-pick strategy as fallback; preserve compaction code in separate files |
| Metal compute shader bugs (Phase 2) | MEDIUM | MEDIUM | CPU fallback always available; extensive unit tests |
| Self-study query quality still poor on GQA | MEDIUM | HIGH | Prefill-Q as alternative; if neither works, solver remains experimental |
| FlashBias kernel too complex (Phase 4A) | HIGH | MEDIUM | Hybrid path (Phase 4B) is the pragmatic alternative |
| SWA compaction degrades quality (Phase 5) | LOW | LOW | SWA compaction is opt-in; base-cache-only remains default |
| 50x quality too poor for any use case | MEDIUM | LOW | 50x is documented as research; 10-20x is the production target |
| 128K testing reveals context-length quality cliff | MEDIUM | MEDIUM | Chunked compaction (Phase 6.1) is the mitigation |

---

## Success Criteria

V2 is complete when:

1. **GPU solver works:** compaction latency < 500ms for 14B at 4K on M3 Pro
2. **Solver quality fixed:** cosine >= 0.90 at 2x on all validated models (currently -0.17 to 0.91)
3. **10x production compression:** cosine >= 0.88 on standard causal models
4. **Flash attention hybrid:** decode performance within 10% of pure FA
5. **18+ models validated:** Gemma3-12B, GPT-OSS-20B, Phi4-14B added via upstream sync
6. **SWA compaction:** base + SWA sub-cache compaction on iSWA models
7. **128K validated:** compaction works at 128K on 3B-7B models with quality >= 0.90

---

## Total Estimated Effort

| Phase | Effort | Calendar |
|-------|:------:|:--------:|
| Phase 1: Upstream Sync | High | 2-5 days |
| Phase 2: GPU Solver | High | 3-5 days |
| Phase 3: Production Solver | Medium-High | 3-5 days |
| Phase 4: Flash + Beta (Hybrid) | Medium-High | 3-5 days |
| Phase 5: SWA Compaction | Medium | 2-3 days |
| Phase 6: High Compression | Medium | 2-3 days |
| Phase 7: 128K Validation (3B-14B) | Medium | 2-3 days |
| Phase 8: 30B-A3B Stress Test + Docs | Medium | 2-3 days |
| **Total** | | **18-32 days** |
