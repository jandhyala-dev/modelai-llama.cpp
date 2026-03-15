# ModelAI llama.cpp — V2 Implementation Plan

**Plan commit:** `(this commit)`
**V1 baseline commit:** `0a637086` (V1 complete — 15 models validated, all tests pass)
**Upstream base:** `0cd4f472` (upstream-master)
**Date:** 2026-03-15
**Owner:** Ajay Jandhyala — ajay@model-ai.app (COT Labs / ModelAI)
**MIT reference repo:** `/Users/ajayjandhyala/dev/whippet/compaction/` (2 commits, up to date)

## Revision History

| Date | Version | Change |
|------|---------|--------|
| 2026-03-15 | v1 | Initial V2 plan: 8 phases, GPU solver + production solver + flash+beta + upstream sync + SWA + 50x + 128K |
| 2026-03-15 | v2 | Full rewrite. Adversarial gap analysis against MIT repo. 13 critical deviations identified. MIT repo as source of truth. |
| 2026-03-15 | v2.1 | **Phase reorder.** Metal GPU solver moved to Phase 3 (early) so all downstream phases benefit from GPU acceleration. SWA bumped to Phase 9. |

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

V2 replaces all low-grade fork solver/NNLS/OMP/budget code with **faithful C++ ports of the MIT reference implementation**, then extends coverage to GPU acceleration, flash attention, SWA, and 128K contexts.

| Goal | V1 State | V2 Target |
|------|----------|-----------|
| Production compression ratio | 2-8x (select only) | **10-20x** (solver + nonuniform) |
| Solver quality on GQA models | -0.17 to 0.91 cosine (broken) | **>= 0.90** cosine (production-grade) |
| Compaction latency (14B, 4K ctx) | ~2-5s on CPU | **< 500ms** (GPU solver) |
| Flash attention with compacted prefix | Blocked (falls back to non-FA) | **Supported** (hybrid split) |
| Model coverage | 15 models (3 incompatible) | **18+ models** (upstream sync) |
| SWA compaction | Base cache only | **Full iSWA** (base + SWA sub-cache) |
| 50x compression | Not attempted | **Research capability** (quality-gated) |
| 128K context validation | Not tested | **Validated** on 3B-14B models |

### Design Principle: MIT Repo Is Source of Truth

The MIT compaction repo (`/Users/ajayjandhyala/dev/whippet/compaction/`) contains the reference implementation for every compaction algorithm. **V2 does not reinvent any algorithm.** Every numerical technique is ported directly from the MIT Python code to C++. Where the fork has diverged, the fork code is replaced.

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
- Training-based compression (Cartridges, KV-Distill) — not applicable

---

## Adversarial Gap Analysis: Fork vs MIT Reference

This analysis compares every numerical algorithm in the fork against the MIT reference. All deviations are tagged with severity and remediation phase.

### GAP-01: NNLS Solver Is Fundamentally Different (CRITICAL)

**MIT** (`algorithms/base.py:471-605`):
- Default: `torch.linalg.lstsq(M, y)` + clamp to `[lower_bound, upper_bound]` (zero iterations)
- Optional PGD: spectral norm via power iteration → step size `η = 1/L` → gradient descent with projection
- Bounds: `lower_bound=1e-12`, `upper_bound=None`
- Fallback: lstsq NaN → Cholesky with `λ=1e-6`

**Fork** (`llama-kv-compact-solver.cpp:316-338`):
- Gradient descent in **log-domain** with bounds `[0.05, 20.0]` (≈ e^±3), 2 iterations
- No lstsq option, no fallback solver
- Bounds are **much tighter** — MIT allows weights as small as 1e-12, fork floors at 0.05

**Impact:** NNLS produces different beta values on every model. Root cause of solver quality degradation on GQA models.

**Remediation:** Phase 2.1 — Replace fork NNLS with direct lstsq + clamp (MIT default). Add PGD as optional path.

### GAP-02: V Fitting Has Only Cholesky Solver (MAJOR)

**MIT** (`algorithms/base.py:240-420`):
- 3 solvers: `lstsq` (default, `torch.linalg.lstsq`), `cholesky`, `pinv` (`torch.linalg.pinv`)
- Ridge scaling: `spectral` (λ × ||X||₂²), `frobenius` (λ × ||X||²_F / t), `fixed` (λ raw)
- Fallback: lstsq NaN → Cholesky with `λ=1e-6`

**Fork** (`llama-kv-compact-solver.cpp:360-405`):
- Only Cholesky via `solve_least_squares_normal_eq` with lambda escalation (×10, 5 attempts)
- Ridge scaling: `min(λ × spectral_norm, 1.0)` — **caps at 1.0**, which is incorrect for ill-conditioned matrices where spectral norm >> 1
- No lstsq or pinv fallback

**Impact:** V fitting fails silently on ill-conditioned matrices. Ridge cap at 1.0 under-regularizes when it matters most.

**Remediation:** Phase 2.2 — Port MIT's 3-solver cascade. Remove ridge lambda cap. Add Frobenius scaling mode.

### GAP-03: Self-Study Query Generation Is Minimal (CRITICAL)

**MIT** (`query_generation/self_study.py:1-800`):
- Two-phase: Model A generates questions, Model B generates answers
- vLLM batch inference for speed
- Hook-based Q extraction via `q_proj(hidden_states)` → `q_norm` → RoPE → reshape
- ~50,000 queries per KV head
- Chunked prefill for memory efficiency
- Multiple conversation specs with extraction functions

**Fork** (`llama-kv-compact-self-study.cpp:141-240`):
- Single-model autoregressive generation, ~256 tokens
- Q-capture via `cb_eval` callback on `Qcur` tensor
- ~256 × n_rep queries per KV head (at most ~1024 for GQA 4:1)
- No batch mode, no conversation specs

**Impact:** 50x fewer queries than MIT. Query diversity is the single most impactful quality factor (paper Figure 4 ablation). This is the primary root cause of solver quality failure.

**Remediation:** Phase 4 — Increase generation to ~2000 tokens. Add prefill-Q capture as zero-cost supplement. Target ~5,000-10,000 queries per KV head combined.

### GAP-04: No Progressive OMP Schedule (MAJOR)

**MIT** (`algorithms/omp.py:50-55`, DEFAULT_PROGRESSIVE_SCHEDULE):
```python
[(300, k_choice=1, nnls_interval=1),
 (1500, k_choice=2, nnls_interval=2),
 (None, k_choice=4, nnls_interval=2)]
```

**Fork** (`llama-kv-compact-select.cpp:169-345`):
- Fixed `k_choice=1`, `nnls_interval=1` (no progressive schedule)
- Always solves NNLS every iteration

**Impact:** OMP is ~4x slower than it could be at high token counts (>1500 keys). At 10x+ compression with large contexts, this makes OMP infeasible.

**Remediation:** Phase 5.1 — Port MIT's progressive schedule. Add schedule parameter to OMP options.

### GAP-05: No Drop-Key Refinement (MAJOR)

**MIT** (`algorithms/omp.py:629-702`):
- After initial selection: check β < `drop_key_beta_cutoff`
- Drop low-weight keys, mask permanently, re-enter selection loop
- Max 3 refinement entries to prevent infinite loops

**Fork** (`llama-kv-compact-select.cpp:264-301`):
- Beta pruning during OMP loop (removes mid-iteration), different mechanism
- Threshold: `log(-7.0)` ≈ e^-7 ≈ 0.0009 (in log-domain)
- No post-selection refinement pass

**Impact:** Fork's mid-loop pruning is less effective than MIT's post-selection refinement. Key quality at >10x compression is lower.

**Remediation:** Phase 5.2 — Add MIT's post-selection drop-key refinement phase after OMP loop.

### GAP-06: Head Budget Uses Entropy, Not Influence Curves (MAJOR)

**MIT** (`head_budget_optimization/solver.py:220-355`):
- Greedy budget solver with pre-computed influence curves (ratio → δ log perplexity)
- Per-model budget JSONs in `head_budgets/` directory (Gemma3, Qwen3, Llama3.1)
- Step size 0.001 for fine-grained allocation
- Marginal benefit = δ(current_ratio) - δ(current_ratio + step_size)

**Fork** (`llama-kv-compact-budget.cpp:12-167`):
- Entropy-proportional: `weight = 1/max(entropy, 1e-3)`
- Uniform sensitivity curve assumed
- No influence curve precomputation
- No per-model budget files

**Impact:** Entropy is a weak proxy for head sensitivity. Paper Figure 5 shows some heads tolerate 50x while others need 2x. Without influence curves, nonuniform budgets are suboptimal.

**Remediation:** Phase 5.3 — Port MIT's greedy solver. Ship pre-computed budgets from MIT repo for supported models.

### GAP-07: No On-Policy Sequential Compaction (RESEARCH)

**MIT** (`compaction_methods/per_layer_head_on_policy.py`):
- Compact layer 0 → generate on-policy queries using compacted Layer 0 → compact Layer 1 → ...
- Each layer sees compacted earlier layers during query generation
- More accurate than off-policy (all layers compacted with same queries)

**Fork:** Not implemented.

**Impact:** Quality improvement at >10x compression. Not critical for 2-8x where off-policy is sufficient.

**Remediation:** Phase 8 (experimental) — Port MIT's on-policy loop after core solver is fixed.

### GAP-08: Chunked Compaction Lacks KV-Based Mode (MODERATE)

**MIT** (`compaction_methods/chunked.py`):
- **KV-based** (default): Prefill full sequence, construct virtual [prefix + chunk + suffix] in KV space, compact chunk
- **Text-based**: Split text, re-tokenize chunks independently
- No RoPE correction needed for KV-based mode
- Multiple chunking strategies (fixed-size, per-document, per-note)

**Fork** (`llama-kv-compact-pipeline.cpp:932-1241`):
- Basic chunked pipeline: split prefix, per-chunk top-k, merge, single solver pass
- No KV-based chunking (operates on token positions, not KV states)
- No RoPE phase handling across boundaries

**Impact:** Quality loss at chunk boundaries for long contexts (>16K). KV-based chunking preserves cross-chunk attention information.

**Remediation:** Phase 6 — Port MIT's KV-based chunking strategy.

### GAP-09: Missing Prefill-Q Query Extraction (MODERATE)

**MIT** (`query_generation/self_study.py:560-800`):
- Forward pass hook extracts Q vectors from actual prefill computation
- `q = q_proj(hidden_states)` → q_norm → RoPE → reshape
- Zero additional cost — queries are a byproduct of prefill

**Fork:** Not implemented. Prefill-Q pipeline exists in concept but only extracts cache keys as surrogates.

**Impact:** Free queries from actual prefill are the highest-quality surrogates.

**Remediation:** Phase 4.2 — Extend Q-capture callback to fire during initial prefill.

### GAP-10: No Attention Bias Pass-Through in Selection (MINOR)

**MIT** (`algorithms/omp.py:63-74`):
- Accepts `attention_bias` tensor, adds to scores before softmax
- Used for sliding window masking, causal masking, context prefix offsets

**Fork:** Selection functions don't accept attention bias.

**Remediation:** Phase 9 (SWA) — Add attention_bias parameter to selection and OMP functions.

### GAP-11: Missing Query Mixing Configuration (MINOR)

**MIT** (`query_generation/config.py`):
- `QueryConfig` supports multiple `QueryMethodConfig` with fractional mixing
- Methods: `self_study`, `random_vectors`, `cache_keys`, `context_prefill`

**Fork:** Fixed pipeline: either cache-key surrogates or self-study. No mixing.

**Remediation:** Phase 4 — Add query source mixing in pipeline orchestration.

### GAP-12: Ridge Lambda Cap at 1.0 Is Incorrect (MAJOR)

**Fork** (`llama-kv-compact-solver.cpp:218-223`):
```cpp
return std::min(lambda * sn, 1.0f);  // CAP AT 1.0
```

**MIT** (`algorithms/base.py`):
```python
lambda_scaled = ridge_lambda * (spectral_norm ** 2)  // NO CAP
```

**Impact:** When spectral norm is large, the cap prevents adequate regularization. The solver produces NaN or garbage.

**Remediation:** Phase 2.2 — Remove the 1.0 cap. Scale by spectral_norm², not spectral_norm.

### GAP-13: MIT Budget JSONs Not Shipped (MODERATE)

**MIT** (`head_budget_optimization/head_budgets/`):
- Pre-computed budgets for: `gemma-3-12b-it`, `gemma-3-4b-it`, `Llama-3.1-8B-Instruct`, `Qwen3-4B`, `Qwen3-4B-Instruct-2507`
- Each has: `optimized_agnostic.json`, `pyramidkv_beta20.json`, `uniform.json`

**Fork:** No pre-computed budget files.

**Remediation:** Phase 5.3 — Copy MIT budget JSONs into fork. Generate budgets for additional models.

---

## MIT Source-of-Truth File Mapping

Every V2 solver/algorithm change maps directly to MIT reference code.

| Fork File | MIT Source File | What to Port |
|-----------|----------------|-------------|
| `llama-kv-compact-solver.cpp` (NNLS) | `algorithms/base.py:471-605` | lstsq + clamp default, PGD with spectral step, bounds [1e-12, None] |
| `llama-kv-compact-solver.cpp` (V fit) | `algorithms/base.py:240-420` | 3-solver cascade (lstsq→cholesky→pinv), spectral/frobenius/fixed ridge, no lambda cap |
| `llama-kv-compact-solver.cpp` (ridge) | `algorithms/base.py:156-188` | spectral = λ × ‖X‖₂², frobenius = λ × ‖X‖²_F / t, fixed = λ |
| `llama-kv-compact-select.cpp` (OMP) | `algorithms/omp.py:478-718` | Progressive schedule, lazy NNLS, drop-key refinement (max 3 entries) |
| `llama-kv-compact-budget.cpp` | `head_budget_optimization/solver.py:220-355` | Greedy solver with influence curves, step_size=0.001 |
| `llama-kv-compact-self-study.cpp` | `query_generation/self_study.py:1-800` | Increase to ~2000 gen tokens, prefill-Q extraction, Q/K norm handling |
| `llama-kv-compact-pipeline.cpp` (chunked) | `compaction_methods/chunked.py` | KV-based chunking, no RoPE correction needed |
| `llama-kv-compact-pipeline.cpp` (nonuniform) | `compaction_methods/per_layer_head.py:288-306` | Budget JSON loading, `max_ratio_per_head` cap |
| `data/head_budgets/` (NEW) | `head_budget_optimization/head_budgets/` | Copy pre-computed budgets for shipped models |
| `llama-kv-compact-pipeline.cpp` (on-policy) | `compaction_methods/per_layer_head_on_policy.py` | On-policy sequential compaction (experimental) |

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

**High.** Upstream KV cache code has been refactored. Expect 2-5 days of merge conflict resolution and verification.

---

## Phase 2: Solver Core — Port MIT Numerical Algorithms (GAP-01, GAP-02, GAP-12)

**Goal:** Replace all fork solver/NNLS code with faithful C++ ports of MIT reference algorithms. This is the foundation for every subsequent phase.

**MIT source files:** `algorithms/base.py` (lines 1-605)

### 2.1 Replace NNLS Implementation (GAP-01)

**Current fork** (`llama-kv-compact-solver.cpp:316-338`): Log-domain gradient descent with bounds [0.05, 20.0], 2 iterations.

**Replace with MIT algorithm** (`algorithms/base.py:471-605`):

**Mode 1 — Direct Solve (default, `nnls_iters=0`):**
```
B = lstsq(M, y)                    // normal equations via Cholesky
B = clamp(B, lower_bound, upper_bound)  // lower=1e-12, upper=None
```

**Mode 2 — Projected Gradient Descent (optional, `nnls_iters > 0`):**
```
L = spectral_norm(M)²              // power iteration, 3 iterations per MIT
η = 1/L                            // step size
FOR t = 1 to nnls_iters:
    grad = M^T @ (M @ B - y)
    B = clamp(B - η * grad, lower_bound, upper_bound)
```

**Key changes from current fork:**
- Remove log-domain NNLS entirely
- Change bounds from [0.05, 20.0] to [1e-12, None] (match MIT defaults)
- Default to lstsq + clamp (0 iterations), not gradient descent
- Power iteration uses 3 steps for spectral norm (MIT), not 8 (fork)
- Step size is `1/L` (spectral norm²), not `1/(spectral_norm + lambda)` (fork)

### 2.2 Replace V Fitting (C2 Solver) (GAP-02, GAP-12)

**Replace with MIT 3-solver cascade** (`algorithms/base.py:240-420`):

```
TRY lstsq:
    C2 = lstsq(X, Y)               // normal equations via Cholesky
    IF C2 has NaN → GOTO cholesky

TRY cholesky:
    C2 = cholesky_solve(X^T X + λI, X^T Y)
    IF fails → GOTO pinv

TRY pinv:
    C2 = pinv(X^T X + λI) @ X^T Y  // pseudoinverse
```

**Ridge scaling modes** (`algorithms/base.py:156-188`):

| Mode | Formula | When to use |
|------|---------|-------------|
| `spectral` | λ_eff = λ × σ_max(X)² | Default. Adapts to matrix condition |
| `frobenius` | λ_eff = λ × (‖X‖²_F / t) | Robust to outlier singular values |
| `fixed` | λ_eff = λ | Direct control |

**Critical fix:** Remove `std::min(..., 1.0f)` cap. Scale by `spectral_norm²`, not `spectral_norm`.

**C++ pinv implementation:** Use Cholesky with aggressive regularization as practical equivalent (no LAPACK SVD available).

### 2.3 Max-Shift Rescaling

The fork's rescaling (`llama-kv-compact-solver.cpp:283-293`) is a valid numerical correction for C++ fp32. MIT doesn't need it (PyTorch handles precision). **Keep the fork's rescaling.**

### 2.4 Testing

- [ ] NNLS lstsq mode: beta values match MIT reference within fp32 tolerance on stories15M
- [ ] NNLS PGD mode: converges in `nnls_iters` iterations with correct step size
- [ ] V fitting lstsq → cholesky → pinv cascade: inject NaN to verify fallback chain
- [ ] Ridge scaling: spectral/frobenius/fixed modes produce correct λ_eff values
- [ ] No lambda cap: verify λ_eff > 1.0 is allowed
- [ ] Solver cosine on stories15M >= 0.99 (currently 0.996)
- [ ] Solver cosine on Qwen3-8B > 0.87 (currently 0.874)
- [ ] Power iteration with 3 iterations matches MIT spectral norm within 1%

### 2.5 Files Created/Modified

| File | Action |
|------|--------|
| `src/llama-kv-compact-solver.cpp` | MAJOR REWRITE — new NNLS, 3-solver V fit, ridge scaling modes |
| `src/llama-kv-compact-solver.h` | MODIFY — new solver options (solver_mode, ridge_scale_mode) |
| `tests/test-kv-compact-solver-reference.cpp` | NEW — MIT reference comparison tests |

### 2.6 Estimated Effort

**Medium.** 2-3 days.

---

## Phase 3: GPU Solver via Metal Compute Shaders (B4)

**Goal:** Move attention score computation and matrix operations from CPU to GPU, reducing compaction latency from ~2-5s to <500ms for 14B at 4K context.

**Rationale for early placement:** GPU acceleration benefits every subsequent phase — query scoring (Phase 4), OMP selection (Phase 5), chunked compaction (Phase 6), and high compression (Phase 8) all involve heavy attention score computation. Building GPU support early means all downstream work runs faster.

### 3.1 What Moves to GPU

| Operation | Current (CPU) | GPU Target | Speedup |
|-----------|--------------|------------|---------|
| Attention score matrix: `score[q][k] = dot(Q[q], K[k]) / sqrt(d)` | O(n_q × n_k × d) scalar loops with NEON | Metal compute shader, thousands of threads | **10-15x** |
| Exp + softmax normalization | Per-row sequential | Per-row parallel reduction | **5-8x** |
| Matrix multiply (X^T X for normal equations) | Scalar with NEON 4-wide | Metal GEMM or MPS MatrixMultiplication | **10-20x** |
| K/V extraction from KV cache | `ggml_backend_tensor_get()` per-head → CPU | Direct GPU buffer read (no host roundtrip) | **Eliminates transfer** |

### 3.2 What Stays on CPU

| Operation | Why |
|-----------|-----|
| Cholesky decomposition | Sequential pivot-dependent — GPU parallelism doesn't help |
| NNLS (lstsq + clamp) | Small problem size (t × t), sequential |
| Power iteration for spectral norm (3 iters) | Sequential, small matrix |
| Pipeline orchestration | Control flow |
| OMP greedy loop | Sequential selection with dependencies |

### 3.3 Metal Shader Implementation

New file: `ggml/src/ggml-metal/ggml-kv-compact-solver.metal`

**Shader 1: Attention Score Matrix**
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
```

**Shader 2: Softmax with Max-Shift** — Per-row softmax with threadgroup max reduction.

**Shader 3: GEMM for Normal Equations** — X^T X computation. Can leverage Metal Performance Shaders `MPSMatrixMultiplication`.

### 3.4 Host-Side Dispatch

New file: `src/llama-kv-compact-solver-metal.mm` (Objective-C++ for Metal API)

```
llama_kv_compact_solver_metal_context:
  - init(device, command_queue)
  - compute_attention_scores(queries, keys) → scores_buffer
  - compute_softmax(scores_buffer) → exp_scores_buffer
  - compute_normal_equations(X, Y) → XtX, XtY
  - readback_results() → host memory
```

### 3.5 GPU K/V Extraction

Replace `ggml_backend_tensor_get()` (GPU→CPU copy) with direct GPU buffer access:
- Allocate Metal buffers for extracted K/V matrices
- Copy within GPU: KV cache tensor → solver input buffer
- Solver shaders read directly from these buffers
- Only final results (beta, compacted V) are read back to CPU for storage

### 3.6 Performance Budget

For Qwen3-14B at 4K context (40 layers × 8 KV heads = 320 invocations):

| Phase | V1 CPU Time | V2 GPU Time | Notes |
|-------|:-----------:|:-----------:|-------|
| K/V extraction | ~800ms | ~50ms | Eliminate host roundtrip |
| Attention scores | ~1200ms | ~80ms | GPU parallel dot products |
| Softmax | ~200ms | ~20ms | GPU parallel reduction |
| NNLS (lstsq + clamp) | ~300ms | ~300ms | Stays on CPU (small problem) |
| Cholesky + V solve | ~200ms | ~200ms | Stays on CPU (sequential) |
| **Total** | **~2700ms** | **~650ms** | **~4x speedup** |

Conservative estimate: **4x speedup** (NNLS/Cholesky remain sequential bottlenecks).

### 3.7 Testing

- [ ] Metal shader unit tests: attention scores match CPU reference within fp32 tolerance
- [ ] Full solver pipeline produces identical beta/V to CPU path (cosine >= 0.9999)
- [ ] No memory leaks (Metal buffer lifecycle)
- [ ] Graceful CPU fallback when Metal is unavailable
- [ ] Per-stage timing shows GPU phases faster than CPU phases

### 3.8 Files Created/Modified

| File | Action |
|------|--------|
| `ggml/src/ggml-metal/ggml-kv-compact-solver.metal` | NEW — Metal compute shaders |
| `src/llama-kv-compact-solver-metal.h` | NEW — Metal solver API |
| `src/llama-kv-compact-solver-metal.mm` | NEW — Metal solver implementation |
| `src/llama-kv-compact-pipeline.cpp` | MODIFY — dispatch to GPU solver when available |
| `CMakeLists.txt` | MODIFY — Metal shader compilation |
| `tests/test-kv-compact-solver-metal.cpp` | NEW — GPU solver tests |

### 3.9 Estimated Effort

**High.** Metal compute shader development + Objective-C++ interop + buffer management. 3-5 days for working prototype, 1-2 more for optimization.

---

## Phase 4: Query Generation — Self-Study + Prefill-Q (GAP-03, GAP-09, GAP-11)

**Goal:** Increase query count from ~256 to ~5,000-10,000 per KV head by extending self-study and adding prefill-Q extraction. GPU acceleration from Phase 3 speeds up query scoring.

**MIT source files:** `query_generation/self_study.py`, `query_generation/config.py`

### 4.1 Extend Self-Study Generation (GAP-03)

**Current fork** (`llama-kv-compact-self-study.cpp:141-240`): ~256 generated tokens.

**Port from MIT:**
1. Increase `n_generate` from 256 to **2000** (config parameter, not hardcoded)
2. Multiple generation rounds: run self-study 3-5 times with different seed tokens for diversity
3. Keep existing Q-capture callback mechanism (it's correct)
4. After generation: subsample to `max_queries_per_kv_head=10000` with uniform stride

**Why not 50,000 like MIT?** MIT uses vLLM batch inference with GPUs. We run single-model autoregressive on a Mac. 2000 tokens × 3 rounds × n_rep = ~24,000 queries for GQA 4:1. This is 25x more than V1.

**Q/K Normalization:** The fork's existing `scale = k_norm_mean / q_norm_mean` fix is correct. Keep it.

### 4.2 Add Prefill-Q Capture (GAP-09)

**New capability:** Capture Q vectors during the initial prefill forward pass.

1. Install Q-capture callback before initial `llama_decode()` for the input prompt
2. Extract Q vectors at every layer as a byproduct of normal inference
3. Store in the same `llama_q_capture_state` structure
4. After prefill: combine prefill Q's with self-study Q's (concatenate, then subsample)

**MIT reference:** `query_generation/self_study.py:560-800` — hook-based Q extraction.

**Cost:** Zero additional compute. Q vectors are computed during prefill anyway.

### 4.3 Query Source Mixing (GAP-11)

```cpp
struct llama_kv_compact_query_config {
    bool use_prefill_q    = true;   // capture Q during prefill
    bool use_self_study   = true;   // generate Q via autoregressive continuation
    bool use_cache_keys   = false;  // legacy: use K as surrogate Q
    uint32_t n_generate   = 2000;   // self-study generation tokens
    uint32_t n_rounds     = 3;      // self-study repetitions
    uint32_t max_per_head = 10000;  // subsample limit per KV head
};
```

### 4.4 Testing

- [ ] Self-study generates >= 2000 tokens per round (up from 256)
- [ ] Prefill-Q captures Q vectors for every layer during initial prompt processing
- [ ] Combined query count >= 5000 per KV head for GQA 4:1 model
- [ ] Solver cosine with improved queries: >= 0.90 on Qwen3-8B at 2x (currently 0.874)
- [ ] Solver cosine with improved queries: >= 0.85 on DeepSeek-R1-8B at 2x (currently -0.025)
- [ ] GQA regrouping still correct after increasing query count

### 4.5 Files Created/Modified

| File | Action |
|------|--------|
| `src/llama-kv-compact-self-study.cpp` | MODIFY — increase n_generate, multi-round, prefill-Q |
| `src/llama-kv-compact-self-study.h` | MODIFY — new config fields |
| `src/llama-kv-compact-query.cpp` | MODIFY — prefill-Q extraction mode |
| `src/llama-kv-compact-pipeline.cpp` | MODIFY — query mixing in solver pipeline |

### 4.6 Estimated Effort

**Medium.** Most infrastructure exists. 2-3 days.

---

## Phase 5: OMP + Budget — Port MIT Selection and Allocation (GAP-04, GAP-05, GAP-06, GAP-13)

**Goal:** Replace fork's OMP and budget allocation with MIT reference implementations. GPU from Phase 3 accelerates OMP scoring.

**MIT source files:** `algorithms/omp.py`, `head_budget_optimization/solver.py`

### 5.1 Progressive OMP Schedule (GAP-04)

**Port MIT's DEFAULT_PROGRESSIVE_SCHEDULE:**

```cpp
struct omp_schedule_entry {
    uint32_t threshold;     // switch after selecting this many keys
    uint32_t k_choice;      // keys per iteration
    uint32_t nnls_interval; // NNLS refit frequency
};

static const omp_schedule_entry DEFAULT_OMP_SCHEDULE[] = {
    { 300,       1, 1 },   // 0-300:   standard OMP (exact)
    { 1500,      2, 2 },   // 301-1500: batch 2, refit every 2
    { UINT32_MAX, 4, 2 },  // 1501+:   batch 4, refit every 2
};
```

**Speed impact:** At >1500 keys, ~8x faster OMP iteration rate.

### 5.2 Drop-Key Refinement (GAP-05)

**Port MIT's post-selection refinement** (`algorithms/omp.py:629-702`):

```
AFTER initial OMP selection of t keys:
  FOR refinement_pass = 0 to 2:  // max 3 entries
    dropped = [i for i in selected if beta[i] < drop_key_beta_cutoff]
    IF len(dropped) == 0: BREAK
    mask[dropped] = permanent
    Re-enter OMP loop to fill len(dropped) slots
    Refit NNLS on new selection
```

**Replace** fork's mid-loop beta pruning with MIT's post-selection approach.

### 5.3 Greedy Budget Solver (GAP-06, GAP-13)

**Port from MIT** (`head_budget_optimization/solver.py:220-355`):

```
allocate_all_heads(min_ratio_per_head)
WHILE sum(budgets) < target_total:
    best_head = argmax_h marginal_benefit(h, step_size)
    budgets[best_head] += step_size
```

**Ship pre-computed budgets:** Copy MIT repo's `head_budgets/` into `data/head_budgets/`.

**Fallback for unknown models:** Use existing entropy-proportional allocation.

### 5.4 Testing

- [ ] Progressive OMP: phases transition at 300 and 1500 keys
- [ ] OMP with schedule >= 3x faster than fixed k=1 for 2000+ target keys
- [ ] Drop-key refinement: max 3 passes (no infinite loop)
- [ ] Greedy budget matches MIT output on Qwen3-4B with `optimized_agnostic.json`
- [ ] Pre-computed budget JSONs load correctly
- [ ] Nonuniform with greedy budgets: cosine >= 0.92 at 4x on Qwen3-8B

### 5.5 Files Created/Modified

| File | Action |
|------|--------|
| `src/llama-kv-compact-select.cpp` | MAJOR REWRITE — progressive schedule, drop-key refinement |
| `src/llama-kv-compact-select.h` | MODIFY — schedule struct, refinement options |
| `src/llama-kv-compact-budget.cpp` | MAJOR REWRITE — greedy solver, JSON loading |
| `src/llama-kv-compact-budget.h` | MODIFY — influence curve types |
| `data/head_budgets/` | NEW — copy from MIT repo |

### 5.6 Estimated Effort

**Medium-High.** 3-5 days.

---

## Phase 6: Chunked Compaction + Long Context (GAP-08)

**Goal:** Port MIT's KV-based chunking for long contexts (>8K). Harden for 128K.

**MIT source files:** `compaction_methods/chunked.py`

### 6.1 KV-Based Chunking

**Port MIT's KV-based approach:**

1. Run full prefill over entire context (captures all KV states)
2. For each chunk `[start, end)`:
   - Extract KV states for `[start, end)` from full prefill cache
   - Construct virtual context: `[compacted_so_far + chunk_KV + suffix_KV]`
   - Compact chunk using queries generated over full context
   - Merge compacted chunk into growing compacted prefix
3. No RoPE correction needed — KV states have correct positional encoding from original prefill

### 6.2 Chunk Size Selection

```cpp
struct llama_kv_compact_chunk_opts {
    uint32_t chunk_size    = 12288;  // ~12K tokens (MIT default)
    bool     kv_based      = true;   // KV-based vs position-based
};
```

### 6.3 128K with Chunking

128K at 12K chunks = ~11 chunks. Each chunk compacted independently → merge into single compacted prefix. No additional KV memory needed beyond what's already loaded.

### 6.4 Testing

- [ ] KV-based chunking matches single-block quality within 0.02 cosine at 8K
- [ ] No RoPE discontinuities across chunk boundaries
- [ ] Chunked compaction at 128K: 11 chunks merge correctly
- [ ] Decode after chunked compaction produces finite logits

### 6.5 Estimated Effort

**Medium.** 2-3 days.

---

## Phase 7: Flash Attention + Beta Hybrid

**Goal:** Allow flash attention to work with compacted prefixes that have non-zero beta.

### 7.1 Hybrid Attention (Pragmatic Path)

Split the attention computation:
- **Live KV suffix:** uses flash attention (no beta needed)
- **Compacted prefix:** uses standard attention with beta (non-FA, as today)
- **Combine:** weighted sum with proper softmax normalization

```
out = (sum_prefix × out_prefix + sum_live × out_live) / (sum_prefix + sum_live)
```

**Performance:** At 16K with 2x compaction: FA handles 8K live tokens (fast), non-FA handles 128-512 compacted tokens (small).

### 7.2 FlashBias (Future, V3)

Custom flash attention Metal kernel with additive bias. Defer to V3 or upstream adoption.

### 7.3 Testing

- [ ] Hybrid attention cosine >= 0.9999 vs non-FA path
- [ ] pp512 with hybrid >= 90% of pure FA
- [ ] tg128 with hybrid >= 95% of pure FA

### 7.4 Estimated Effort

**Medium-High.** 3-5 days.

---

## Phase 8: High Compression (10-50x) + On-Policy (GAP-07)

**Goal:** Enable 10-20x production compression and 50x research capability using the full MIT algorithm stack from Phases 2-6.

**Dependencies:** Phases 2-6 (solver core, GPU, queries, OMP, chunking) must be complete.

### 8.1 Production 10x Profile

| Component | Method | MIT Source |
|-----------|--------|-----------|
| Query generation | Prefill-Q + self-study (2000 × 3 rounds) | `self_study.py` |
| Key selection | top-k with nonuniform per-head budgets (greedy solver) | `solver.py` |
| Beta fitting | lstsq + clamp (MIT default) | `base.py:471-605` |
| V fitting | lstsq → cholesky cascade with spectral ridge | `base.py:240-420` |
| GPU acceleration | Metal shaders for attention scores + softmax | Phase 3 |

**Expected quality at 10x:** cosine >= 0.88 on standard causal models, >= 0.90 on MoE.

### 8.2 Research 50x Profile

| Component | Method | MIT Source |
|-----------|--------|-----------|
| Query generation | Self-study (2000 × 5 rounds) | `self_study.py` |
| Key selection | OMP with progressive schedule | `omp.py:478-718` |
| Budget allocation | Greedy solver with influence curves | `solver.py:220-355` |
| Refinement | Drop-key post-selection (cutoff=1e-4) | `omp.py:629-702` |

**Quality at 50x:** ~15-25% accuracy loss. Research-only.

### 8.3 On-Policy Sequential Compaction (GAP-07, Experimental)

**Port from MIT** (`compaction_methods/per_layer_head_on_policy.py`):

1. Compact layer 0 with original queries
2. Materialize compacted layer 0 into KV cache
3. Forward pass with compacted layer 0 → on-policy queries for layer 1
4. Compact layer 1 with on-policy queries
5. Repeat for remaining layers

**Cost:** n_layers × forward_pass_time. ~60-120s for 40-layer 14B. Opt-in only.

### 8.4 Testing

- [ ] 10x compression: cosine >= 0.88 on all 15+ models
- [ ] 20x compression: cosine >= 0.85 on Qwen3-8B, Qwen3-30B-A3B
- [ ] 50x compression: cosine measured and documented
- [ ] On-policy at 10x: cosine delta vs off-policy measured on 3 models

### 8.5 Estimated Effort

**Medium.** Integration and tuning: 2-3 days. On-policy: +2 days.

---

## Phase 9: SWA Full Compaction + Attention Bias (GAP-10)

**Goal:** Extend compaction to the SWA sub-cache and add attention bias pass-through.

### 9.1 SWA Sub-Cache Compaction

- V1: `compacted_prefix_runtime_supported()` rejects SWA caches
- V2: Compact SWA tokens about to exit the window

**SWA-specific selection:** Compact tokens in `[window_start, window_start + window_size/2]` (oldest half). Use attention scores within window context, not global.

### 9.2 Attention Bias in Selection (GAP-10)

Port MIT's `attention_bias` parameter:
```cpp
void llama_kv_compact_select_topk(
    const float * scores, uint32_t n,
    const float * attention_bias,  // NEW: [n] or nullptr
    uint32_t t,
    std::vector<uint32_t> & selected);
```

**Use cases:** SWA windowed masking, causal masking for out-of-order sequences.

### 9.3 Testing

- [ ] Gemma2-9B: SWA sub-cache compaction succeeds
- [ ] Gemma2-9B: quality >= 0.90 cosine after SWA compaction
- [ ] Attention bias correctly masks positions outside window
- [ ] iSWA: both base and SWA caches compacted independently
- [ ] Non-iSWA models: behavior unchanged from V1

### 9.4 Estimated Effort

**Medium.** 2-3 days.

---

## Phase 10: 128K Context Validation

**Goal:** Validate compaction quality and performance at 128K context on models that fit in 32GB.

### 10.1 Hardware Constraints

| Model Size | KV @ 128K | Total RAM | Testable on 32GB Mac? |
|:----------:|:---------:|:---------:|:---------------------:|
| 3B | ~2.6 GB | ~4.5 GB | **Yes** |
| 7B | ~5.2 GB | ~9.6 GB | **Yes** |
| 8B | ~6.6 GB | ~11.5 GB | **Tight** |
| 14B | ~10+ GB | ~19+ GB | No (swap) |
| 30B MoE | ~10+ GB | ~27+ GB | No (see Phase 11) |

### 10.2 128K Integration Tests

| Model | Context | Ratios | Pipeline | Target Cosine |
|-------|---------|--------|----------|:-------------:|
| Llama3.2-3B | 128K | 2x, 4x, 8x, 16x | select + solver | >= 0.90 |
| Qwen2.5-7B | 128K | 2x, 4x, 8x, 16x | select + solver | >= 0.90 |
| Qwen3-8B | 128K | 2x, 4x, 8x | select + solver | >= 0.85 |

**Performance targets:**
- Compaction latency (select): < 2s on 3B, < 5s on 7B
- Post-compaction decode: faster than uncompacted baseline
- Serialization round-trip: cosine >= 0.99

**Chunked compaction at 128K:**
- [ ] 11 chunks (~12K each) merge correctly
- [ ] Quality within 0.02 cosine of single-block at 8K
- [ ] Decode produces finite, coherent logits

**Server integration:**
- [ ] `/compact` succeeds at 128K
- [ ] `/props` reports correct `active_n_kv`
- [ ] Memory stays within 32GB for 3B-7B models

### 10.3 Estimated Effort

**Medium.** 2-3 days.

---

## Phase 11: Capstone — Qwen3-30B-A3B at 128K + Final Documentation

**Goal:** Stress-test full V2 stack with largest feasible model at 128K. Update all docs.

### 11.1 Qwen3-30B-A3B at 128K — Capstone Stress Test

**Why this model:** MoE with 30B total / 3B active. Best V1 cosines (select 0.999, solver 0.906). With 8-16x compaction, KV drops from ~10GB to ~0.6-1.25GB.

**Protocol — run with extreme care:**

1. Close all other applications. Kill Ollama, browsers.
2. Monitor memory: `vm_stat 1` in separate terminal
3. Start smallest: 128K context, 16x compaction (KV → ~0.6GB, total ~17.9GB)
4. If RSS < 28GB and no swap: proceed to 8x, 4x, 2x
5. **Abort immediately** if swap pages increase

| Test | Context | Ratio | Expected KV After | Total RAM | Feasible? |
|------|---------|:-----:|:-----------------:|:---------:|:---------:|
| D | 128K | 16x | ~0.6 GB | ~17.9 GB | **Yes** |
| A | 128K | 8x | ~1.25 GB | ~18.5 GB | **Likely** |
| B | 128K | 4x | ~2.5 GB | ~19.8 GB | **Probably** |
| C | 128K | 2x | ~5.0 GB | ~22.3 GB | **Tight** |

**Success criteria:**
- [ ] At least one ratio completes without swap
- [ ] Post-compaction decode produces finite logits
- [ ] Quality: select cosine >= 0.85
- [ ] Serialization round-trip works at 128K

**Headline result if successful:**
> "Run a 30B model with 128K context on a 32GB laptop — only possible with KV compaction."

### 11.2 Documentation Updates

| Document | Change |
|----------|--------|
| `docs/modelai-fork-summary.md` | V2 support matrix, GPU solver, solver quality, SWA |
| `docs/modelai-v2-benchmark-results.md` | NEW — full model matrix at 2x-50x |
| `docs/modelai-performance-roadmap.md` | Update B4 to DONE |
| `CLAUDE.md` | Update support matrix to V2 |

### 11.3 Estimated Effort

**Medium.** 2-3 days.

---

## Dependency Graph

```
Phase 1: Upstream Sync
    |
    +--→ Phase 2: Solver Core (MIT port — NNLS, V fit, ridge)
    |        |
    |        +--→ Phase 3: GPU Solver (Metal) ← EARLY: accelerates all downstream
    |        |        |
    |        |        +--→ Phase 4: Query Generation (self-study + prefill-Q)
    |        |        |        |
    |        |        |        +--→ Phase 5: OMP + Budget (progressive, greedy solver)
    |        |        |                 |
    |        |        |                 +--→ Phase 8: High Compression (10-50x) + On-Policy
    |        |        |
    |        |        +--→ Phase 6: Chunked Compaction (KV-based)
    |        |                 |
    |        |                 +--→ Phase 10: 128K Validation
    |        |
    |        +--→ Phase 7: Flash Attention Hybrid (independent of GPU solver)
    |
    +--→ Phase 9: SWA + Attention Bias (independent of solver)
    |
    +--- All above --→ Phase 11: Capstone + Docs
```

**Critical path:** Phases 1 → 2 → 3 → 4 → 5 → 8 (solver quality + GPU acceleration).

**Parallelizable after Phase 1:**
- Phase 7 (flash hybrid) is independent of Phases 3-6
- Phase 9 (SWA) is independent of Phases 2-8

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| Upstream merge conflicts in KV cache code | HIGH | HIGH | Cherry-pick strategy as fallback |
| MIT NNLS port quality doesn't match Python | MEDIUM | HIGH | Reference test: C++ output vs MIT Python on same inputs |
| Self-study query count still insufficient | LOW | MEDIUM | Prefill-Q supplements; more rounds |
| Metal compute shader bugs | MEDIUM | MEDIUM | CPU fallback always available |
| FlashBias kernel too complex | HIGH | MEDIUM | Hybrid path is the pragmatic alternative |
| 128K testing reveals quality cliff | MEDIUM | MEDIUM | Chunked compaction is the mitigation |
| Pre-computed budget JSONs don't cover all models | LOW | LOW | Entropy fallback for unknown models |

---

## Success Criteria

V2 is complete when:

1. **Solver core matches MIT:** NNLS, V fitting, and ridge scaling produce values within fp32 tolerance of MIT Python reference on identical inputs
2. **GPU solver works:** compaction latency < 500ms for 14B at 4K on M3 Pro
3. **Solver quality fixed:** cosine >= 0.90 at 2x on all validated models (currently -0.17 to 0.91)
4. **10x production compression:** cosine >= 0.88 on standard causal models
5. **Flash attention hybrid:** decode performance within 10% of pure FA
6. **18+ models validated:** Gemma3-12B, GPT-OSS-20B, Phi4-14B added via upstream sync
7. **SWA compaction:** base + SWA sub-cache on iSWA models
8. **128K validated:** compaction works at 128K on 3B-8B models with quality >= 0.85
9. **128K 30B capstone:** Qwen3-30B-A3B at 128K completes at least one compaction ratio without swap

---

## Total Estimated Effort

| Phase | Effort | Calendar |
|-------|:------:|:--------:|
| Phase 1: Upstream Sync | High | 2-5 days |
| Phase 2: Solver Core (MIT port) | Medium | 2-3 days |
| Phase 3: GPU Solver (Metal) | High | 3-5 days |
| Phase 4: Query Generation | Medium | 2-3 days |
| Phase 5: OMP + Budget | Medium-High | 3-5 days |
| Phase 6: Chunked Compaction | Medium | 2-3 days |
| Phase 7: Flash Attention Hybrid | Medium-High | 3-5 days |
| Phase 8: High Compression + On-Policy | Medium | 4-5 days |
| Phase 9: SWA + Attention Bias | Medium | 2-3 days |
| Phase 10: 128K Validation | Medium | 2-3 days |
| Phase 11: Capstone + Docs | Medium | 2-3 days |
| **Total** | | **27-43 days** |
