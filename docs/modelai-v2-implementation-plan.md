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
| 2026-03-15 | v2.1 | Phase reorder. Metal GPU solver moved to Phase 3 (early). SWA bumped to Phase 9. 11 phases total. |
| 2026-03-15 | v3 | **Review fix.** Address all findings from 2 adversarial reviewers. Add 3 new gaps (GAP-14/15/16). Fix Cholesky/QR trade-off (use Accelerate LAPACK). Fix FA hybrid LSE design. Revise GPU target to <700ms. Add chunk overlap. Port budget swap/annealing solvers. Add select regression gate. Fix DeepSeek target. Acknowledge existing on-policy code. Add query memory budget. Revise timeline to 35-55 days. |

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
| Solver quality on GQA models | -0.17 to 0.91 cosine (broken) | **>= 0.90** cosine on standard models; DeepSeek-R1 excluded (requires investigation) |
| Compaction latency (14B, 4K ctx) | ~2-5s on CPU | **< 700ms** (GPU solver) |
| Flash attention with compacted prefix | Blocked (falls back to non-FA) | **Supported** (hybrid split with LSE extraction) |
| Model coverage | 15 models (3 incompatible) | **18+ models** (upstream sync) |
| SWA compaction | Base cache only | **Full iSWA** (base + SWA sub-cache) |
| 50x compression | Not attempted | **Research capability** (quality-gated) |
| 128K context validation | Not tested | **Validated** on 3B-14B models |

### Design Principle: MIT Repo Is Source of Truth

The MIT compaction repo (`/Users/ajayjandhyala/dev/whippet/compaction/`) contains the reference implementation for every compaction algorithm. **V2 does not reinvent any algorithm.** Every numerical technique is ported directly from the MIT Python code to C++. Where the fork has diverged, the fork code is replaced.

### Numerical Stability: Cholesky vs QR Trade-Off

The MIT Python code uses `torch.linalg.lstsq` (QR decomposition via LAPACK `xGELS`) which has condition number κ for the solve. The fork's C++ code uses Cholesky normal equations which have condition number κ² (squared). This means Cholesky is less numerically stable for ill-conditioned matrices at high compression.

**V2 mitigation strategy:**
1. **Primary:** Use macOS Accelerate framework LAPACK `sgels` for true QR-based least squares (available on all Apple Silicon Macs)
2. **Fallback:** Cholesky with symmetrization (`XtX = 0.5 * (XtX + XtX.T)`) and minimum regularization (λ >= 1e-8) to partially compensate for κ²
3. **Last resort:** Cholesky with aggressive regularization (Tikhonov) — biases toward zero but numerically stable
4. At 50x compression where κ can exceed 10^6, QR via LAPACK is essential — Cholesky will produce garbage

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

This analysis compares every numerical algorithm in the fork against the MIT reference. All deviations are tagged with severity and remediation phase. 16 gaps total (13 original + 3 from reviewer findings).

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

**Impact:** NNLS produces different beta values on every model. One of the primary contributors to solver quality degradation on GQA models (along with GAP-12 and GAP-03).

**Remediation:** Phase 2.1 — Replace fork NNLS with Accelerate LAPACK `sgels` + clamp (matching MIT's QR-based lstsq). Add PGD as optional path.

### GAP-02: V Fitting Has Only Cholesky Solver (MAJOR)

**MIT** (`algorithms/base.py:240-420`):
- Primary: `torch.linalg.lstsq` (QR-based via LAPACK `xGELS`)
- Fallback: Cholesky with `λ=1e-6` and symmetrization `XtX = 0.5 * (XtX + XtX.T)`
- Ridge scaling: `spectral` (λ × ‖X‖₂²), `frobenius` (λ × ‖X‖²_F / t), `fixed` (λ raw)
- Underdetermined case (n < t): Uses `XXᵀ + λI` formulation instead of `XᵀX + λI`

**Fork** (`llama-kv-compact-solver.cpp:360-405`):
- Only Cholesky via `solve_least_squares_normal_eq` with lambda escalation (×10, 5 attempts)
- Ridge scaling: `min(λ × spectral_norm, 1.0)` — **caps at 1.0**, incorrect
- No symmetrization before Cholesky
- No underdetermined formulation

**Impact:** V fitting fails silently on ill-conditioned matrices. Ridge cap at 1.0 under-regularizes when it matters most.

**Remediation:** Phase 2.2 — Use Accelerate LAPACK `sgels` as primary. Cholesky with symmetrization as fallback. Remove ridge cap. Add Frobenius scaling. Handle n < t case.

### GAP-03: Self-Study Query Generation Is Minimal (CRITICAL)

**MIT** (`query_generation/self_study.py:1-800`):
- Two-phase: Model A generates questions, Model B generates answers
- vLLM batch inference for speed
- Hook-based Q extraction via `q_proj(hidden_states)` → `q_norm` → RoPE → reshape
- ~50,000 queries per KV head
- Multiple `ConversationSpec` objects with different question types for semantic diversity

**Fork** (`llama-kv-compact-self-study.cpp:141-240`):
- Single-model autoregressive generation, ~256 tokens
- Q-capture via `cb_eval` callback on `Qcur` tensor
- ~256 × n_rep queries per KV head (at most ~1024 for GQA 4:1)
- No diversity controls (same distribution for all queries)

**Impact:** 50x fewer queries than MIT, all from one distribution. Query diversity (not just count) is a primary quality factor (paper Figure 4). Simply generating more continuation tokens has diminishing returns — all queries are from the same autoregressive distribution.

**Remediation:** Phase 4 — Increase generation to ~2000 tokens with diversity: (a) vary temperature/top_p across rounds, (b) inject question-word seed tokens, (c) add prefill-Q capture for zero-cost high-quality queries. Target ~5,000-10,000 per KV head.

### GAP-04: No Progressive OMP Schedule (MAJOR)

**MIT** (`algorithms/omp.py:50-55`, DEFAULT_PROGRESSIVE_SCHEDULE):
```python
[(300, k_choice=1, nnls_interval=1),
 (1500, k_choice=2, nnls_interval=2),
 (None, k_choice=4, nnls_interval=2)]
```

**Fork** (`llama-kv-compact-select.cpp:169-345`):
- Fixed `k_choice=1`, `nnls_interval=1` (no progressive schedule)

**Impact:** OMP is ~4x slower than it could be at high token counts.

**Remediation:** Phase 5.1 — Port MIT's progressive schedule.

### GAP-05: No Drop-Key Refinement (MAJOR)

**MIT** (`algorithms/omp.py:629-702`):
- After initial selection: check β < `drop_key_beta_cutoff`
- Drop low-weight keys, mask permanently, re-enter selection loop
- Max 3 refinement entries

**Fork** (`llama-kv-compact-select.cpp:264-301`):
- Beta pruning during OMP loop (mid-iteration), different mechanism

**Impact:** Fork's mid-loop pruning is less effective than MIT's post-selection refinement.

**Remediation:** Phase 5.2 — Replace mid-loop pruning with MIT's post-selection drop-key refinement.

### GAP-06: Head Budget Uses Entropy, Not Influence Curves (MAJOR)

**MIT** (`head_budget_optimization/solver.py`):
- **6 solver functions:** `solve_greedy`, `solve_swap`, `solve_annealing`, `solve_for_ratios`, `solve_ratio_agnostic_swap`, `solve_ratio_agnostic`
- The `optimized_agnostic.json` files in the MIT repo were generated by `solve_ratio_agnostic_swap()` (the swap solver), NOT the greedy solver
- Pre-computed budget JSONs for Gemma3-4B, Gemma3-12B, Llama3.1-8B, Qwen3-4B

**Fork** (`llama-kv-compact-budget.cpp:12-167`):
- Entropy-proportional: `weight = 1/max(entropy, 1e-3)`
- No influence curve precomputation
- No budget files

**Impact:** Entropy is a weak proxy. Greedy solver alone cannot reproduce MIT's `optimized_agnostic.json` budgets or generate budgets for new models.

**Remediation:** Phase 5.3 — Port greedy + swap + `solve_ratio_agnostic_swap` solvers. Ship MIT budget JSONs. Provide budget generation script for new models.

### GAP-07: On-Policy Implementation Is Simpler Than MIT (MODERATE)

**MIT** (`compaction_methods/per_layer_head_on_policy.py`):
- Per-layer sequential: compact layer 0 → generate on-policy queries using compacted Layer 0 → compact Layer 1 → ...
- `_extract_on_policy_queries_for_layer()` has 200+ lines of cache-building logic

**Fork** (`llama-kv-compact-on-policy.cpp`, 90 lines):
- Two-pass approach: Pass 1 = standard solver, Pass 2 = generate continuation + re-run solver with captured Q
- Does NOT do per-layer sequential compaction
- Falls back to Pass 1 on Pass 2 failure

**Impact:** Fork's 2-pass approach captures some on-policy benefit but misses per-layer sequential correction that improves quality at >10x.

**Remediation:** Phase 8 — Extend existing 90-line on-policy file to support MIT's per-layer sequential mode. Current 2-pass code is useful and should be kept as fast-path.

### GAP-08: Chunked Compaction Lacks KV-Based Mode (MODERATE)

**MIT** (`compaction_methods/chunked.py`):
- **KV-based** (default): Prefill full sequence, construct virtual [prefix + chunk + suffix] in KV space
- Chunking strategies: `FixedSizeChunking`, `LongHealthChunking`, `LQAChunking`
- No chunk overlap in MIT (verified — `overlap_ratio` does not exist in MIT code)

**Fork** (`llama-kv-compact-pipeline.cpp:932-1241`):
- Basic position-based splitting

**Impact:** Quality loss at chunk boundaries for long contexts.

**Note:** Reviewer 2 cited MIT `overlap_ratio=0.1` but this parameter does not exist in the MIT codebase. However, chunk boundary quality IS a legitimate concern. V2 adds optional overlap as an enhancement beyond MIT.

**Remediation:** Phase 6 — Port MIT's KV-based chunking. Add optional overlap at chunk boundaries (fork enhancement, not MIT port) to mitigate boundary quality loss.

### GAP-09: Missing Prefill-Q Query Extraction (MODERATE)

**MIT** (`query_generation/self_study.py:560-800`):
- Forward pass hook extracts Q vectors from actual prefill computation

**Fork** (`llama-kv-compact-prefill-q.cpp`, 347 lines):
- Prefill-Q infrastructure **exists** but uses cache keys as surrogates, not actual Q extraction from prefill forward pass

**Remediation:** Phase 4.2 — Extend existing prefill-Q code to capture real Q vectors during initial prefill via `cb_eval` callback.

### GAP-10: No Attention Bias Pass-Through in Selection (MINOR)

**MIT** (`algorithms/omp.py:63-74`):
- Accepts `attention_bias` tensor

**Fork:** Selection functions don't accept attention bias.

**Remediation:** Phase 9 (SWA) — Add attention_bias parameter.

### GAP-11: Missing Query Mixing Configuration (MINOR)

**MIT** (`query_generation/config.py`):
- `QueryConfig` supports multiple `QueryMethodConfig` with fractional mixing

**Fork:** Fixed pipeline, no mixing.

**Remediation:** Phase 4 — Add query source mixing config.

### GAP-12: Ridge Lambda Cap at 1.0 Is Incorrect (MAJOR)

**Fork** (`llama-kv-compact-solver.cpp:218-223`):
```cpp
return std::min(lambda * sn, 1.0f);  // CAP AT 1.0
```

**MIT** (`algorithms/base.py`):
```python
lambda_scaled = ridge_lambda * (spectral_norm ** 2)  // NO CAP, squared
```

**Impact:** Cap prevents adequate regularization. Also scales by spectral_norm, not spectral_norm² (MIT squares it).

**Remediation:** Phase 2.2 — Remove cap. Scale by spectral_norm².

### GAP-13: MIT Budget JSONs Not Shipped (MODERATE)

**MIT** (`head_budget_optimization/head_budgets/`):
- Pre-computed budgets for: `gemma-3-12b-it`, `gemma-3-4b-it`, `Llama-3.1-8B-Instruct`, `Qwen3-4B`, `Qwen3-4B-Instruct-2507`
- Each has: `optimized_agnostic.json`, `pyramidkv_beta20.json`, `uniform.json`
- JSON structure: `{"L0H0": 0.0025, "L0H1": 0.0015, ...}` — proportions per head summing to 1.0

**Remediation:** Phase 5.3 — Copy MIT budget JSONs. Generate budgets for Phase 1 new models using ported swap solver.

### GAP-14: Spectral-to-Frobenius Ridge Fallback Missing (MINOR) [NEW — Reviewer 1]

**MIT** (`algorithms/base.py:152-155`):
```python
try:
    lam = ridge_lambda * (torch.linalg.matrix_norm(X, ord=2)**2)
except Exception:
    lam = ridge_lambda * ((torch.linalg.matrix_norm(X, ord='fro')**2) / t)
```

**Fork:** No automatic fallback from spectral to Frobenius on failure.

**Remediation:** Phase 2.2 — Add spectral→Frobenius fallback in ridge scaling.

### GAP-15: OMP Cached Selection Order Missing (MINOR) [NEW — Reviewer 1 + Reviewer 2]

**MIT** (`algorithms/omp.py:243-263`):
- `cached_selection_order` parameter allows reusing a previously computed OMP selection order for efficient multi-ratio evaluation

**Impact:** Without this, evaluating quality at 2x, 4x, 8x, 16x requires running OMP 4 separate times. With it, run OMP once at max ratio, take prefixes.

**Remediation:** Phase 5.1 — Add `cached_selection_order` to OMP options.

### GAP-16: OMP Quality Parameters Missing (MINOR) [NEW — Reviewer 2]

**MIT** (`algorithms/omp.py:43-68`):
- `use_abs_corr`: use |correlation| for key selection (default False)
- `normalize_exp_scores`: L2-normalize columns before correlation (default False)
- `zerobeta`: zero beta before C2 computation (default False)

**Fork:** None of these parameters exist.

**Remediation:** Phase 5.1 — Add as config parameters to OMP options struct. Match MIT defaults.

---

## MIT Source-of-Truth File Mapping

| Fork File | MIT Source File | What to Port |
|-----------|----------------|-------------|
| `llama-kv-compact-solver.cpp` (NNLS) | `algorithms/base.py:471-605` | Accelerate LAPACK `sgels` + clamp (QR-based), PGD, bounds [1e-12, None] |
| `llama-kv-compact-solver.cpp` (V fit) | `algorithms/base.py:240-420` | `sgels` primary, Cholesky+symmetrization fallback, spectral/frobenius/fixed ridge, no cap, n<t handling |
| `llama-kv-compact-solver.cpp` (ridge) | `algorithms/base.py:146-161` | spectral = λ × ‖X‖₂² with spectral→frobenius fallback, frobenius = λ × ‖X‖²_F / t, fixed = λ |
| `llama-kv-compact-select.cpp` (OMP) | `algorithms/omp.py:478-718` | Progressive schedule, lazy NNLS, drop-key refinement, cached_selection_order, use_abs_corr/normalize_exp_scores/zerobeta params |
| `llama-kv-compact-budget.cpp` | `head_budget_optimization/solver.py` | Greedy + swap + ratio_agnostic_swap solvers, influence curves, smoothing_window |
| `llama-kv-compact-self-study.cpp` | `query_generation/self_study.py` | Increase to ~2000 gen tokens, diversity via temperature/seed variation, prefill-Q |
| `llama-kv-compact-pipeline.cpp` (chunked) | `compaction_methods/chunked.py` | KV-based chunking + optional overlap (fork enhancement) |
| `llama-kv-compact-pipeline.cpp` (nonuniform) | `compaction_methods/per_layer_head.py:288-306` | Budget JSON loading, max_ratio_per_head cap with redistribution |
| `data/head_budgets/` (NEW) | `head_budget_optimization/head_budgets/` | Copy pre-computed budgets + generation script for new models |
| `llama-kv-compact-on-policy.cpp` | `compaction_methods/per_layer_head_on_policy.py` | Extend 90-line 2-pass to MIT's per-layer sequential mode |

---

## Phase 1: Upstream Sync

**Goal:** Resolve 3 model incompatibilities (Gemma3-12B, GPT-OSS-20B, Phi4-14B) and pick up Metal kernel improvements.

**Risk: HIGH** — upstream llama.cpp has diverged significantly. Merge conflicts in KV cache code are expected.

### 1.1 Upstream Merge Strategy

1. **Identify the target upstream commit** — pick a stable point after Gemma3 arch, GPT-OSS arch, and Phi4 graph fix
2. **Cherry-pick vs full merge** — full merge preferred; cherry-pick if KV cache conflicts are intractable. **Decision criteria for switching to cherry-pick:** if merge conflict resolution exceeds 3 days with no working build, switch to targeted cherry-picks of the 3 model fixes + Metal kernel improvements
3. **Preserve all compaction code** — every `src/llama-kv-compact-*` and `src/llama-kv-compacted-prefix-*` file must survive
4. **Re-run full V1 test suite** after merge

### 1.2 Target Upstream Features

| Feature | Why |
|---------|-----|
| Gemma3 architecture support | Unblocks Gemma3-12B |
| GPT-OSS architecture support | Unblocks GPT-OSS-20B |
| Phi4 graph hash set fix | Unblocks Phi4-14B |
| Fused multiply-add for Q4/Q5/Q6_K | 16-28% faster pp |
| Metal mul_mv_ext for BF16/Q2_K/Q3_K | Faster Metal kernels |

### 1.3 Merge Verification

- [ ] Build succeeds: `cmake -B build -DGGML_METAL=ON && cmake --build build --config Release`
- [ ] ctest passes: `ctest --test-dir build -L main --output-on-failure`
- [ ] All 15 V1 models pass 51/51 integration tests
- [ ] 3 new models (Gemma3-12B, GPT-OSS-20B, Phi4-14B) load and pass integration tests
- [ ] **Select pipeline regression gate:** cosine >= 0.950 on all models (must not regress from V1)
- [ ] Baseline tok/s within 5% of V1 numbers

### 1.4 Estimated Effort

**High.** 2-5 days.

---

## Phase 2: Solver Core — Port MIT Numerical Algorithms (GAP-01, GAP-02, GAP-12, GAP-14)

**Goal:** Replace all fork solver/NNLS code with faithful C++ ports of MIT reference algorithms using macOS Accelerate LAPACK for numerical parity.

**MIT source files:** `algorithms/base.py` (lines 1-605)

### 2.1 Replace NNLS Implementation (GAP-01)

**Current fork:** Log-domain gradient descent with bounds [0.05, 20.0], 2 iterations.

**Replace with MIT algorithm using Accelerate LAPACK:**

**Mode 1 — QR-based lstsq (default, `nnls_iters=0`):**
```
B = sgels(M, y)                     // Accelerate LAPACK sgels (QR decomposition)
B = clamp(B, lower_bound, upper_bound)  // lower=1e-12, upper=None
```

**CPU fallback (no Accelerate):**
```
B = cholesky_solve(M^T M + 1e-8*I, M^T y)  // normal equations with min regularization
XtX = 0.5 * (XtX + XtX.T)                  // symmetrization before Cholesky
B = clamp(B, lower_bound, upper_bound)
```

**Mode 2 — Projected Gradient Descent (optional, `nnls_iters > 0`):**
```
L = spectral_norm(M)²              // power iteration, 3 iterations
η = 1/L
FOR t = 1 to nnls_iters:
    grad = M^T @ (M @ B - y)
    B = clamp(B - η * grad, lower_bound, upper_bound)
```

**Key changes:**
- Remove log-domain NNLS entirely
- Change bounds from [0.05, 20.0] to [1e-12, None]
- Use LAPACK `sgels` for QR-based solve (condition number κ, not κ²)
- Cholesky fallback adds symmetrization step and minimum regularization

### 2.2 Replace V Fitting (GAP-02, GAP-12, GAP-14)

**Replace with LAPACK-based solver cascade:**

```
TRY sgels (QR):
    C2 = sgels(X, Y)                // Accelerate LAPACK (condition κ)
    IF C2 has NaN → GOTO cholesky

TRY cholesky:
    XtX = X^T @ X + λ_eff * I
    XtX = 0.5 * (XtX + XtX.T)      // SYMMETRIZATION — prevents fp32 asymmetry
    IF n < t:                       // UNDERDETERMINED case
        Use XXᵀ + λI formulation instead
    C2 = cholesky_solve(XtX, X^T Y)
    IF fails → GOTO aggressive_cholesky

TRY aggressive_cholesky:
    C2 = cholesky_solve(XtX + large_λ * I, X^T Y)  // Tikhonov bias toward zero
    // NOTE: This biases the solution. At 50x compression this is expected.
```

**Ridge scaling modes:**

| Mode | Formula | Fallback |
|------|---------|----------|
| `spectral` | λ_eff = λ × σ_max(X)² | On failure → frobenius |
| `frobenius` | λ_eff = λ × (‖X‖²_F / t) | — |
| `fixed` | λ_eff = λ | — |

**Critical fixes:**
- Remove `std::min(..., 1.0f)` cap
- Scale by spectral_norm² (not spectral_norm)
- Add symmetrization before every Cholesky
- Add spectral→frobenius automatic fallback (GAP-14)
- Handle underdetermined case n < t

**LAPACK availability:** macOS Accelerate framework provides `sgels` on all Apple Silicon Macs. Link via `-framework Accelerate`. Add `#include <Accelerate/Accelerate.h>` and use `sgels_()` (Fortran interface).

### 2.3 Max-Shift Rescaling

The fork's rescaling (`llama-kv-compact-solver.cpp:283-293`) is a valid numerical correction for C++ fp32. **Keep it.**

### 2.4 Testing

- [ ] NNLS with LAPACK `sgels`: beta values match MIT reference within fp32 tolerance on stories15M
- [ ] NNLS with Cholesky fallback: symmetrization prevents asymmetry failures
- [ ] V fitting `sgels` → cholesky → aggressive_cholesky cascade: inject NaN to verify each fallback
- [ ] Ridge scaling: spectral/frobenius/fixed modes produce correct λ_eff
- [ ] Spectral→frobenius fallback triggers correctly on spectral norm failure
- [ ] No lambda cap: verify λ_eff > 1.0 is allowed
- [ ] Underdetermined case (n < t): verify correct formulation used
- [ ] **Select pipeline regression:** all 15 V1 models still achieve cosine >= 0.950 at 2x
- [ ] Solver cosine on stories15M >= 0.99
- [ ] Solver cosine on Qwen3-8B >= 0.87 (currently 0.874, should improve)
- [ ] Add degenerate-input reference test: near-duplicate keys producing ill-conditioned XᵀX

### 2.5 Files Created/Modified

| File | Action |
|------|--------|
| `src/llama-kv-compact-solver.cpp` | MAJOR REWRITE — LAPACK sgels, 3-tier cascade, ridge scaling, symmetrization |
| `src/llama-kv-compact-solver.h` | MODIFY — solver_mode, ridge_scale_mode, underdetermined flag |
| `CMakeLists.txt` | MODIFY — link Accelerate framework on macOS |
| `tests/test-kv-compact-solver-reference.cpp` | NEW — MIT reference comparison + degenerate input tests |

### 2.6 Estimated Effort

**Medium-High.** 3-5 days (revised from 2-3 — mathematically dense rewrite with LAPACK integration).

---

## Phase 3: GPU Solver via Metal Compute Shaders (B4)

**Goal:** Move attention score computation and matrix operations from CPU to GPU, reducing compaction latency from ~2-5s to <700ms for 14B at 4K context.

**Note:** GPU acceleration benefits Phases 4-6 and 8 downstream. However, Phases 4-5 are **algorithmically independent** of GPU and can proceed with CPU-only solver if Phase 3 is delayed. This is an explicit schedule-risk mitigation — see Risk Register.

### 3.1 What Moves to GPU

| Operation | Current (CPU) | GPU Target | Speedup |
|-----------|--------------|------------|---------|
| Attention scores: `dot(Q[q], K[k]) / sqrt(d)` | Scalar + NEON | Metal compute shader | **10-15x** |
| Exp + softmax normalization | Sequential | Parallel reduction | **5-8x** |
| Matrix multiply (XᵀX for normal equations) | NEON 4-wide | Metal GEMM / MPS | **10-20x** |
| K/V extraction | `ggml_backend_tensor_get()` → CPU | GPU buffer copy | **Eliminates transfer** |

### 3.2 What Stays on CPU

| Operation | Why |
|-----------|-----|
| LAPACK `sgels` (QR solve) | Sequential, requires Accelerate framework |
| Cholesky decomposition | Sequential pivot-dependent |
| NNLS (lstsq + clamp) | Small problem, sequential |
| Pipeline orchestration | Control flow |
| OMP greedy loop | Sequential with dependencies |

### 3.3 Metal Shader Implementation

New file: `ggml/src/ggml-metal/ggml-kv-compact-solver.metal`

**Integration note:** llama.cpp compiles Metal shaders as a monolithic file. The new shader may need to be included in the existing `ggml-metal.metal` or establish a new compilation unit via `ggml_metal_library_from_source()`.

**Shader 1: Attention Score Matrix** — each thread computes one score[qi][ki].

**Shader 2: Softmax with Max-Shift** — per-row softmax with threadgroup max reduction.

**Shader 3: GEMM** — XᵀX via Metal Performance Shaders `MPSMatrixMultiplication` (validate fp32 support).

### 3.4 GPU K/V Extraction

- Allocate Metal buffers for extracted K/V matrices
- Copy within GPU: KV cache tensor → solver input buffer
- Only final results (beta, compacted V) read back to CPU

### 3.5 Buffer Lifecycle

Metal buffer allocation/deallocation:
- Allocate per-compaction-request (not per-head — amortize allocation)
- Command buffer per batch of heads (reduce kernel launch overhead)
- `waitUntilCompleted` for synchronous dispatch (simplicity > async complexity)

### 3.6 Performance Budget

For Qwen3-14B at 4K context (40 layers × 8 KV heads = 320 invocations):

| Phase | V1 CPU Time | V2 GPU Time | Notes |
|-------|:-----------:|:-----------:|-------|
| K/V extraction | ~800ms | ~50ms | GPU buffer copy |
| Attention scores | ~1200ms | ~80ms | GPU parallel |
| Softmax | ~200ms | ~20ms | GPU parallel |
| LAPACK sgels + NNLS | ~300ms | ~300ms | CPU (LAPACK) |
| Cholesky + V solve | ~200ms | ~200ms | CPU (sequential) |
| **Total** | **~2700ms** | **~650ms** | **~4x speedup** |

**Why <700ms not <500ms:** CPU-bound LAPACK/Cholesky (500ms combined) is the floor. The <700ms target accounts for GPU overhead and provides margin. Future optimization: batch 320 head NNLS problems as a single GPU dispatch (stretch goal, not Phase 3 scope).

### 3.7 CPU Fallback

When Metal is unavailable: all operations run on CPU with NEON. No quality difference, only speed difference (~2700ms vs ~650ms). Fallback performance target: < 3000ms.

### 3.8 Testing

- [ ] Metal shader unit tests: attention scores match CPU within fp32 tolerance
- [ ] Full pipeline cosine >= 0.9999 GPU vs CPU (numerical equivalence)
- [ ] No memory leaks (Metal buffer lifecycle)
- [ ] Graceful CPU fallback when Metal unavailable
- [ ] Per-stage timing: GPU phases faster than CPU phases

### 3.9 Files Created/Modified

| File | Action |
|------|--------|
| `ggml/src/ggml-metal/ggml-kv-compact-solver.metal` | NEW — Metal compute shaders |
| `src/llama-kv-compact-solver-metal.h` | NEW — Metal solver API |
| `src/llama-kv-compact-solver-metal.mm` | NEW — Objective-C++ implementation |
| `src/llama-kv-compact-pipeline.cpp` | MODIFY — dispatch to GPU when available |
| `CMakeLists.txt` | MODIFY — Metal shader compilation |
| `tests/test-kv-compact-solver-metal.cpp` | NEW — GPU solver tests |

### 3.10 Estimated Effort

**High.** 5-8 days (revised from 3-5 — first-time Metal compute + Obj-C++ interop + build integration).

---

## Phase 4: Query Generation — Self-Study + Prefill-Q (GAP-03, GAP-09, GAP-11)

**Goal:** Increase query count from ~256 to ~5,000-10,000 per KV head with improved diversity.

**MIT source files:** `query_generation/self_study.py`, `query_generation/config.py`

### 4.1 Extend Self-Study Generation (GAP-03)

1. Increase `n_generate` from 256 to **2000** (config parameter)
2. Multiple generation rounds (3-5) with **diversity controls:**
   - Vary `temperature` across rounds: [0.6, 0.8, 1.0]
   - Vary seed tokens: use argmax (default), then top-k samples (k=5, k=10)
   - Inject question-word seeds: "What", "How", "Why" tokens to elicit question-like continuations
3. Keep existing Q-capture callback mechanism
4. Subsample to `max_queries_per_kv_head=10000` with uniform stride

**Why not 50,000 like MIT?** MIT uses vLLM batch inference with 2 models. Fork runs single-model autoregressive. 2000 × 3 rounds × GQA n_rep ≈ ~24,000 queries for GQA 4:1.

### 4.2 Add Prefill-Q Capture (GAP-09)

Extend existing `llama-kv-compact-prefill-q.cpp` (347 lines):
1. Install Q-capture callback before initial `llama_decode()`
2. Capture real Q vectors at every layer during prefill
3. Combine with self-study Q's (concatenate, then subsample)

**Cost:** Zero additional compute.

### 4.3 Query Memory Budget

For Qwen3-14B (40 layers × 8 KV heads × 128 head_dim × 10,000 queries × 4 bytes):
- **Per-layer:** 8 heads × 10K × 128 × 4 = ~40 MB
- **All layers simultaneously:** 40 × 40 MB = ~1.6 GB

**Strategy:** Process per-layer, not batch. During Phase 2 solver (per-layer loop), capture queries for current layer only. Peak query memory: ~40 MB (one layer at a time), not 1.6 GB.

### 4.4 Query Source Mixing (GAP-11)

```cpp
struct llama_kv_compact_query_config {
    bool use_prefill_q    = true;
    bool use_self_study   = true;
    bool use_cache_keys   = false;  // legacy
    uint32_t n_generate   = 2000;
    uint32_t n_rounds     = 3;
    float    temperatures[3] = {0.6f, 0.8f, 1.0f};  // per-round diversity
    uint32_t max_per_head = 10000;
};
```

### 4.5 Testing

- [ ] Self-study generates >= 2000 tokens per round
- [ ] Prefill-Q captures Q at every layer during initial prompt
- [ ] Combined query count >= 5000 per KV head for GQA 4:1
- [ ] Temperature variation produces measurably different Q distributions across rounds
- [ ] **Cumulative quality after Phases 2-4:** solver cosine >= 0.90 on Qwen3-8B at 2x
- [ ] **Cumulative quality after Phases 2-4:** solver cosine improvement on DeepSeek-R1-8B (measure, no fixed target — see DeepSeek note below)
- [ ] **Select pipeline regression:** cosine >= 0.950 on all 15 models (unchanged from V1)
- [ ] Query memory stays < 100 MB peak (per-layer processing)

**DeepSeek-R1-8B note:** V1 solver cosine is -0.025 (anti-correlated). The root cause is unclear — possibly NNLS bounds (GAP-01), possibly MLA architecture mismatch. After Phase 2 NNLS fix and Phase 4 query improvements, measure and report actual cosine. Do NOT commit to >= 0.90 until root cause is understood. If quality remains < 0.50, DeepSeek-R1 requires dedicated investigation outside the standard pipeline.

### 4.6 Files Created/Modified

| File | Action |
|------|--------|
| `src/llama-kv-compact-self-study.cpp` | MODIFY — n_generate, multi-round, diversity controls |
| `src/llama-kv-compact-self-study.h` | MODIFY — query config with temperatures |
| `src/llama-kv-compact-prefill-q.cpp` | MODIFY — real Q capture during prefill |
| `src/llama-kv-compact-pipeline.cpp` | MODIFY — query mixing |

### 4.7 Estimated Effort

**Medium.** 2-3 days.

---

## Phase 5: OMP + Budget — Port MIT Selection and Allocation (GAP-04, GAP-05, GAP-06, GAP-13, GAP-15, GAP-16)

**Goal:** Replace fork's OMP and budget allocation with MIT reference implementations.

**MIT source files:** `algorithms/omp.py`, `head_budget_optimization/solver.py`

### 5.1 Progressive OMP Schedule + Parameters (GAP-04, GAP-15, GAP-16)

```cpp
struct omp_schedule_entry {
    uint32_t threshold;
    uint32_t k_choice;
    uint32_t nnls_interval;
};

static const omp_schedule_entry DEFAULT_OMP_SCHEDULE[] = {
    { 300,       1, 1 },
    { 1500,      2, 2 },
    { UINT32_MAX, 4, 2 },
};

struct llama_kv_compact_omp_opts {
    // ... existing fields ...
    const omp_schedule_entry * schedule = DEFAULT_OMP_SCHEDULE;
    uint32_t schedule_len              = 3;
    bool     use_abs_corr              = false;   // GAP-16
    bool     normalize_exp_scores      = false;   // GAP-16
    bool     zerobeta                  = false;   // GAP-16
    std::vector<uint32_t> * cached_selection_order = nullptr;  // GAP-15
};
```

**Cached selection order (GAP-15):** Run OMP once at max target size. Cache the full selection order. For smaller ratios (2x, 4x, 8x), take prefixes of the cached order. Eliminates re-running OMP per ratio.

### 5.2 Drop-Key Refinement (GAP-05)

Replace mid-loop beta pruning with MIT's post-selection refinement:

```
AFTER initial OMP selection of t keys:
  FOR refinement_pass = 0 to 2:  // max 3
    dropped = [i for i in selected if beta[i] < cutoff]
    IF len(dropped) == 0: BREAK
    mask[dropped] = permanent
    Re-enter OMP loop to fill len(dropped) slots
    Refit NNLS on new selection

// Always do final NNLS solve when nnls_interval > 1
IF nnls_interval > 1:
    solve_nnls(selected, target)  // MIT omp.py:706-709
```

### 5.3 Budget Solver — Port Greedy + Swap + Ratio-Agnostic (GAP-06, GAP-13)

**Port 3 of MIT's 6 solvers** (the ones needed for production):

| MIT Solver | Port? | Why |
|-----------|:-----:|-----|
| `solve_greedy()` | **Yes** | Fast allocation for runtime use |
| `solve_swap()` | **Yes** | Handles U-shaped curves, needed for budget generation |
| `solve_ratio_agnostic_swap()` | **Yes** | Generates `optimized_agnostic.json` budgets for new models |
| `solve_annealing()` | No | Marginal improvement over swap; expensive |
| `solve_for_ratios()` | No | Wrapper, trivial to add later |
| `solve_ratio_agnostic()` | No | Wrapper for ratio_agnostic_swap |

**MIT solver features to port:**
- `smoothing_window` for influence curves — sliding average for noisy curves (use smoothed for decisions, original for loss evaluation)
- `max_ratio_per_head` cap with excess redistribution to other heads
- Global-layer detection (infer from curve keys, skip SWA layers)

**Budget JSON loading:**
```cpp
bool llama_kv_compact_load_head_budgets(
    const char * json_path,
    std::vector<float> & proportions  // output: per-head proportions
);
```

**JSON parsing:** Use a minimal hand-rolled JSON parser (no external deps) or embed a single-header parser. The JSON structure is simple: `{"L0H0": 0.0025, ...}`.

**Budget generation for new models:** Ship a Python script (`scripts/generate_head_budgets.py`) that:
1. Loads a model via the fork's `llama-cli`
2. Runs single-head compaction sweeps at multiple ratios
3. Feeds influence curves to the ported swap solver
4. Outputs `optimized_agnostic.json`

This covers Phase 1's 3 new models (Gemma3-12B, GPT-OSS-20B, Phi4-14B).

### 5.4 Testing

- [ ] Progressive OMP: phases transition at 300 and 1500 keys
- [ ] OMP with schedule >= 3x faster than fixed k=1 for 2000+ target keys
- [ ] Cached selection order: multi-ratio evaluation runs OMP once, not per ratio
- [ ] Drop-key refinement: max 3 passes, no infinite loop
- [ ] Final NNLS solve always runs when nnls_interval > 1
- [ ] Greedy budget matches MIT output on Qwen3-4B with `optimized_agnostic.json`
- [ ] Swap solver reproduces MIT's `optimized_agnostic.json` for Qwen3-4B (within tolerance)
- [ ] Budget generation script produces valid JSON for a new model
- [ ] **Select pipeline regression:** cosine >= 0.950 on all models

### 5.5 Files Created/Modified

| File | Action |
|------|--------|
| `src/llama-kv-compact-select.cpp` | MAJOR REWRITE — progressive schedule, drop-key, cached order, new params |
| `src/llama-kv-compact-select.h` | MODIFY — schedule struct, OMP params |
| `src/llama-kv-compact-budget.cpp` | MAJOR REWRITE — greedy + swap + ratio_agnostic_swap, JSON loading, smoothing |
| `src/llama-kv-compact-budget.h` | MODIFY — influence curve types, solver selection |
| `data/head_budgets/` | NEW — copy from MIT + generated budgets for new models |
| `scripts/generate_head_budgets.py` | NEW — budget generation for new models |

### 5.6 Estimated Effort

**High.** 4-6 days (revised from 3-5 — swap/annealing solvers + budget generation script).

---

## Phase 6: Chunked Compaction + Long Context (GAP-08)

**Goal:** Port MIT's KV-based chunking for long contexts (>8K). Add optional overlap for boundary quality.

**MIT source files:** `compaction_methods/chunked.py`, `chunking/strategies.py`

### 6.1 KV-Based Chunking

1. Run full prefill over entire context
2. For each chunk `[start, end)`:
   - Extract KV states from full prefill cache
   - Construct virtual context: `[compacted_so_far + chunk_KV + suffix_KV]`
   - Compact chunk
   - Merge into growing compacted prefix
3. No RoPE correction needed — KV states have correct positional encoding

### 6.2 Chunk Configuration

```cpp
struct llama_kv_compact_chunk_opts {
    uint32_t chunk_size    = 12288;  // ~12K tokens (MIT default)
    bool     kv_based      = true;   // KV-based vs position-based
    float    overlap_ratio = 0.1f;   // 10% overlap (fork enhancement, not MIT)
};
```

**Overlap handling (fork enhancement):** MIT does NOT implement chunk overlap. However, chunk boundaries are a real quality concern. V2 adds optional 10% overlap:
- Overlap region: last 10% of chunk N overlaps first 10% of chunk N+1
- Merge strategy: for positions in overlap, average the compacted representations from both chunks
- When `overlap_ratio=0`: matches MIT behavior exactly

### 6.3 128K with Chunking

128K at 12K chunks with 10% overlap: ~12 chunks. No additional KV memory beyond what's loaded.

### 6.4 Testing

- [ ] KV-based chunking matches single-block quality within 0.02 cosine at 8K
- [ ] Overlap=0.1 improves boundary quality vs overlap=0 (measured on 16K context)
- [ ] No RoPE discontinuities across chunk boundaries
- [ ] 128K: chunks merge correctly into single compacted prefix
- [ ] Decode after chunked compaction produces finite logits

### 6.5 Estimated Effort

**Medium.** 3-4 days (revised from 2-3 — overlap handling adds complexity).

---

## Phase 7: Flash Attention + Beta Hybrid

**Goal:** Allow flash attention to work with compacted prefixes that have non-zero beta.

### 7.1 The LSE Problem

The combination formula `out = (sum_prefix × out_prefix + sum_live × out_live) / (sum_prefix + sum_live)` requires the log-sum-exp (LSE) from both attention blocks. `ggml_flash_attn_ext` (ggml.h:2323) returns only the output tensor — not LSE.

### 7.2 LSE Extraction Approach

**Option A (recommended): Modify Metal FA kernel to output LSE as side-channel**

Add an extra output buffer to `ggml_flash_attn_ext` that stores `LSE[head][query]`. The LSE is computed internally during FA (it's the softmax denominator) — we just need to write it out.

Modification scope: `ggml-metal.metal` flash attention kernel + `ggml-metal.m` dispatch code. The LSE buffer is `[n_head, n_query]` floats — tiny compared to the KV cache.

**Option B (fallback): Compute LSE separately for live suffix**

Run a non-FA attention pass on the live suffix to extract LSE only. Discard the attention output (use FA's output instead). Wastes compute but avoids kernel modification.

**Option C (simplest): Two-pass FlashDecoding-style tiling**

Split QKV into tiles. Each tile produces `(partial_out, partial_LSE)`. Combine across tiles using the online softmax trick. This is architecturally what FlashDecoding already does.

### 7.3 Combination Formula (Log-Domain)

To avoid overflow when partition sums have very different magnitudes:

```
m = max(LSE_prefix, LSE_live)
out = (exp(LSE_prefix - m) × out_prefix + exp(LSE_live - m) × out_live) / (exp(LSE_prefix - m) + exp(LSE_live - m))
```

### 7.4 Testing

- [ ] Hybrid attention cosine >= 0.9999 vs non-FA path
- [ ] Log-domain combination: no overflow for LSE values differing by >100
- [ ] pp512 with hybrid >= 90% of pure FA
- [ ] tg128 with hybrid >= 95% of pure FA

### 7.5 Estimated Effort

**High.** 5-8 days (revised from 3-5 — LSE extraction requires Metal kernel modification or alternative approach).

---

## Phase 8: High Compression (10-50x) + On-Policy (GAP-07)

**Goal:** Enable 10-20x production compression and 50x research capability. Extend on-policy compaction from 2-pass heuristic to iterative refinement and per-layer sequential modes.

**Dependencies:** Phases 2-7 must be complete.

**Novel extensions:** Iterative and sequential on-policy are novel extensions beyond the MIT reference implementation (`omp.py`/`base.py`). There is no external reference to verify against; correctness relies on quality-gated convergence and solver residual monitoring.

### 8.1 Single-Layer Refit Utility + Residual Reporting

**Files:** `src/llama-kv-compact-prefill-q.cpp` (modified), `src/llama-kv-compact-prefill-q.h` (modified), `src/llama-kv-cache.cpp` (modified), `src/llama-kv-cache.h` (modified), `src/llama-kv-compact-pipeline.h` (modified)

**8.1a — Residual output:** Add `float mean_partition_sum_relative_error` to `llama_kv_compact_pipeline_stats` and `llama_kv_compact_prefill_q_stats`. Aggregate per-head `partition_sum_relative_error` from `fit_beta()` across all layers/heads. Add same aggregation to `fit_from_live_kv()` so initial K-as-Q pass produces a baseline residual for quality gate comparison.

**8.1b — Tensor cache invalidation:** Add `compacted_prefix_bump_version()` to `llama_kv_cache` — increments version counter and invalidates tensor cache. Called after `refit_single_layer` modifies layer data in place.

**8.1c — Single-layer refit function:** New `llama_kv_compact_refit_single_layer()` extracted from the solver loop in `prefill_q_with_captured_state()` (lines 286-334). Takes model layer ID (not layout index), resolves internally via `compacted_prefix_layer_layout_for_solver()`. Re-fits beta/V for one layer using new Q, without re-selecting positions. Precondition: live KV cache must not be reclaimed.

### 8.2 Iterative On-Policy Refinement

**Files:** `src/llama-kv-compact-on-policy.cpp` (modified), `src/llama-kv-compact-on-policy.h` (NEW), `src/llama-kv-compact-pipeline.h` (modified), `src/llama-kv-cache.h` (modified), `src/llama-kv-cache.cpp` (modified)

New `llama_kv_compact_iterative_on_policy_from_live_kv()` with config struct containing `n_on_policy_passes` (0 = K-as-Q only, 1 = existing 2-pass, 2+ = iterative) and `quality_min_improvement` threshold (default 0.5%).

**Algorithm:** Initial K-as-Q fit, then for each on-policy pass: enable execution, generate continuation with Q-capture, re-solve all layers, apply quality gate (stop if relative residual improvement < threshold), clear Q-capture state. Quality gate uses mean `partition_sum_relative_error` from `fit_beta()`.

**Chunked interaction:** Initial pass uses `fit_from_live_kv()` on full prefix. Subsequent iterations use non-chunked `prefill_q_with_captured_state()` because compacted prefix is already short.

### 8.3 Per-Layer Sequential On-Policy (Experimental)

**Files:** `src/llama-kv-compact-on-policy.cpp` (modified), `src/llama-kv-compact-on-policy.h` (modified), `src/llama-kv-compact-pipeline.h` (modified), `src/llama-kv-cache.h` (modified), `src/llama-kv-cache.cpp` (modified)

New `llama_kv_compact_sequential_on_policy_from_live_kv()`. Iterates over compacted layouts (not model layers — SWA layers automatically skipped). For each layer: generate continuation with Q-capture, call `refit_single_layer()`, clear Q-capture. Next generation sees refitted layer, exploiting sequential dependency.

**Key invariant:** Mutation during active execution is safe because `can_reuse()` returns false when `compacted_prefix_active` (Phase 7), forcing graph rebuild each decode. `bump_version()` invalidates tensor cache after each refit.

**Cost:** `layouts.size()` × `n_generate_q` full forward passes. Qwen3-14B (40 layers, 64 tokens): ~45s total.

### 8.4 High-Compression Auto-Tuning

**Files:** `tools/server/server-context.cpp` (modified), `tools/server/server-task.h` (modified)

When `ratio >= 10.0`, auto-apply stronger solver parameters for fields at sentinel values (`UINT32_MAX` for integers, negative for floats):

| Parameter | Default (ratio < 10) | Auto (ratio >= 10) |
|---|---|---|
| max_queries | 256 | 512 |
| nnls_iters | 2 | 4 |
| lambda | 1e-6 | 1e-5 |
| n_generate | 256 | 512 |
| n_on_policy_passes | 1 | 2 |
| max_queries_per_kv_head | 1024 | 2048 |

Server timeout scaled by `n_on_policy_passes + 1` (or `layouts.size()` for sequential) to avoid premature abort.

### 8.5 High-Compression Test Scaffolding

**Files:** `tests/test-kv-compact-workload.cpp` (modified), `tests/test-kv-compact-prefill-q.cpp` (new or modified)

Extend ratio/threshold arrays to `{2, 4, 8, 10, 20, 50}` / `{0.95, 0.90, 0.85, 0.80, 0.70, 0.50}`. Add unit tests for `refit_single_layer`, iterative quality gate, sequential loop iteration count, version counter bump.

### 8.6 Server Dispatch

**Files:** `tools/server/server-context.cpp` (modified), `tools/server/server-task.h` (modified)

- `"on_policy"` with `n_on_policy_passes >= 1` dispatches to iterative function (supersedes existing 2-pass call)
- `"sequential_on_policy"` dispatches to sequential function
- `"on_policy"` with `n_on_policy_passes == 0` dispatches to `fit_from_live_kv()` (K-as-Q only)

### 8.7 Hyperparameter Sweep (Deferred to CI)

- Sweep `ridge_lambda` values [1e-8, 1e-6, 1e-4, 1e-2] on 3 models
- Sweep `n_generate` values [500, 1000, 2000, 5000]
- Record cosine at 2x, 4x, 10x for each combination
- Deferred: requires model files not available on dev machine

### 8.8 Testing

- [ ] 10x compression: cosine >= 0.88 on all standard causal models (CI)
- [ ] 20x compression: cosine >= 0.85 on Qwen3-8B, Qwen3-30B-A3B (CI)
- [ ] 50x compression: cosine measured and documented (no pass/fail threshold, CI)
- [ ] Iterative on-policy: quality gate stops non-improving iterations
- [ ] Sequential on-policy: iterates layouts.size() times, not n_layers times
- [ ] On-policy at 10x: cosine delta vs off-policy measured on 3 models (CI)
- [ ] Unit tests pass without model files

### 8.9 Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Stale tensor cache after refit | Resolved | `bump_version()` invalidates cache |
| Quality gate unimplementable | Resolved | Residual aggregation in `fit_beta()` |
| Live KV reclaimed during loop | Medium | Documented precondition: no reclaim until pipeline returns |
| Sequential on-policy cost (~45s) | Medium | Experimental flag; cost transparent in stats |
| Iterative convergence unknown | Medium | Quality gate stops non-improving iterations; CI validation |
| 50x quality insufficient | Low | Informational threshold (0.50); tunable via config |

### 8.10 Commit Structure

1. `kv-compact: Phase 8a — single-layer refit utility + residual reporting` (8.1)
2. `kv-compact: Phase 8b — iterative on-policy with quality gate` (8.2, 8.4, 8.6 iterative dispatch)
3. `kv-compact: Phase 8c — per-layer sequential on-policy` (8.3, 8.6 sequential dispatch)
4. `kv-compact: Phase 8d — high-compression test scaffolding` (8.5)

### 8.11 Estimated Effort

**Medium-High.** 5-7 days (on-policy cache-building complexity + quality gate + sequential mode).

---

## Phase 9: SWA Full Compaction + Attention Bias (GAP-10)

**Goal:** Extend compaction to SWA sub-cache. Add attention bias pass-through.

### 9.1 SWA Sub-Cache Compaction

- V1: `compacted_prefix_runtime_supported()` rejects SWA caches
- V2: Compact tokens about to exit the sliding window

**SWA-specific:** Compact tokens in `[window_start, window_start + window_size/2]`. Use attention scores within window context.

**SWA token lifecycle:** Compacted SWA tokens are consumed by the current window and then discarded as the window slides. They do NOT persist.

### 9.2 Attention Bias in Selection (GAP-10)

```cpp
void llama_kv_compact_select_topk(
    const float * scores, uint32_t n,
    const float * attention_bias,  // [n] or nullptr
    uint32_t t,
    std::vector<uint32_t> & selected);
```

### 9.3 Testing

- [ ] Gemma2-9B: SWA sub-cache compaction succeeds
- [ ] Gemma2-9B: quality >= 0.90 cosine after SWA compaction
- [ ] Attention bias correctly masks positions
- [ ] Non-iSWA models: behavior unchanged

### 9.4 Estimated Effort

**Medium.** 2-3 days.

---

## Phase 10: 128K Context Validation

**Goal:** Validate compaction quality and performance at 128K on models that fit in 32GB.

### 10.1 Hardware Constraints

| Model Size | KV @ 128K | Total RAM | Testable on 32GB Mac? |
|:----------:|:---------:|:---------:|:---------------------:|
| 3B | ~2.6 GB | ~4.5 GB | **Yes** |
| 7B | ~5.2 GB | ~9.6 GB | **Yes** |
| 8B | ~6.6 GB | ~11.5 GB | **Tight** |
| 14B | ~10+ GB | ~19+ GB | No (swap) |

### 10.2 128K Integration Tests

| Model | Context | Ratios | Pipeline | Target Cosine |
|-------|---------|--------|----------|:-------------:|
| Llama3.2-3B | 128K | 2x, 4x, 8x, 16x | select + solver | >= 0.90 |
| Qwen2.5-7B | 128K | 2x, 4x, 8x, 16x | select + solver | >= 0.90 |
| Qwen3-8B | 128K | 2x, 4x, 8x | select + solver | >= 0.85 |

**Performance:** Compaction latency (select) < 2s on 3B, < 5s on 7B.

**Chunked:** 11 chunks merge correctly. Quality within 0.02 of single-block at 8K.

**Server:** `/compact` succeeds at 128K. Memory stays within 32GB for 3B-7B.

### 10.3 Estimated Effort

**Medium.** 2-3 days.

---

## Phase 11: Capstone — Qwen3-30B-A3B at 128K + Final Documentation

**Goal:** Stress-test full V2 stack with largest feasible model at 128K. Update all docs.

### 11.1 Qwen3-30B-A3B at 128K — Capstone Stress Test

**Why:** MoE 30B/3B active. Best V1 cosines (select 0.999, solver 0.906). With 8-16x compaction, KV → ~0.6-1.25GB.

**Protocol:**
1. Close all applications. Monitor `vm_stat 1`.
2. Start smallest: 128K + 16x (KV → ~0.6GB, total ~17.9GB)
3. If no swap: proceed to 8x, 4x, 2x
4. Abort on swap activity

| Test | Ratio | Expected KV | Total RAM | Feasible? |
|------|:-----:|:-----------:|:---------:|:---------:|
| D | 16x | ~0.6 GB | ~17.9 GB | **Yes** |
| A | 8x | ~1.25 GB | ~18.5 GB | **Likely** |
| B | 4x | ~2.5 GB | ~19.8 GB | **Probably** |
| C | 2x | ~5.0 GB | ~22.3 GB | **Tight** |

### 11.2 Server Endpoint Updates

V2 adds new solver capabilities that the `/compact` endpoint should expose:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solver_mode` | string | `"select"` | Pipeline: select, solver, omp, nonuniform |
| `ridge_scale` | string | `"spectral"` | Ridge scaling: spectral, frobenius, fixed |
| `query_source` | string | `"prefill_q+self_study"` | Query generation mode |
| `n_generate` | int | 2000 | Self-study generation tokens |

All parameters are optional with V1 defaults. No breaking changes to existing API contract.

### 11.3 Documentation Updates

| Document | Change |
|----------|--------|
| `docs/modelai-fork-summary.md` | V2 support matrix, GPU solver, solver quality, SWA |
| `docs/modelai-v2-benchmark-results.md` | NEW — full model matrix at 2x-50x |
| `CLAUDE.md` | Update support matrix to V2 |

### 11.4 Estimated Effort

**Medium.** 2-3 days.

---

## Dependency Graph

```
Phase 1: Upstream Sync
    |
    +--→ Phase 2: Solver Core (LAPACK + MIT port)
    |        |
    |        +--→ Phase 3: GPU Solver (Metal) ← accelerates downstream
    |        |        |
    |        |        +--→ Phase 4: Query Generation
    |        |        |        |
    |        |        |        +--→ Phase 5: OMP + Budget
    |        |        |                 |
    |        |        |                 +--→ Phase 8: High Compression + On-Policy
    |        |        |
    |        |        +--→ Phase 6: Chunked Compaction
    |        |                 |
    |        |                 +--→ Phase 10: 128K Validation
    |        |
    |        +--→ Phase 7: Flash Attention Hybrid (independent of GPU)
    |        |
    |        +-- Phases 4, 5 CAN proceed with CPU solver if Phase 3 is delayed
    |
    +--→ Phase 9: SWA + Attention Bias (independent of solver)
    |
    +--- All above --→ Phase 11: Capstone + Docs + Server
```

**Critical path:** 1 → 2 → 3 → 4 → 5 → 8

**Schedule risk mitigation:** If Phase 3 (Metal) slips, Phases 4 and 5 proceed with CPU-only solver. GPU is a speed improvement, not a correctness requirement. This breaks the critical path dependency on Phase 3.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| Upstream merge conflicts in KV cache code | HIGH | HIGH | Cherry-pick fallback; decision criteria defined (3 days) |
| Phase 3 (Metal) slips, blocking critical path | HIGH | MEDIUM | Phases 4-5 proceed with CPU solver (explicit parallel path) |
| LAPACK sgels not available on target platform | LOW | HIGH | Cholesky+symmetrization fallback always available |
| MIT NNLS port quality doesn't match Python | MEDIUM | HIGH | Reference test: C++ vs MIT Python on identical inputs |
| Self-study query diversity insufficient | MEDIUM | MEDIUM | Prefill-Q supplements; temperature variation; more rounds |
| Phase 7 LSE extraction fails | MEDIUM | MEDIUM | Option B (separate non-FA LSE pass) as fallback |
| DeepSeek-R1-8B quality remains negative | HIGH | LOW | Excluded from blanket target; dedicated investigation track |
| Pre-computed budgets don't cover new models | LOW | LOW | Budget generation script; entropy fallback |
| Hyperparameter interaction after gap fixes | MEDIUM | MEDIUM | Explicit sweep step in Phase 8.4 |

---

## Success Criteria

V2 is complete when:

1. **Solver core matches MIT:** NNLS and V fitting use LAPACK `sgels` (QR-based) with Cholesky+symmetrization fallback, producing values within fp32 tolerance of MIT Python on identical inputs
2. **GPU solver works:** compaction latency **< 700ms** for 14B at 4K on M3 Pro (CPU fallback < 3000ms)
3. **Solver quality fixed:** cosine >= 0.90 at 2x on all standard causal models (Llama, Qwen, Mistral); DeepSeek-R1 measured and reported separately
4. **10x production compression:** cosine >= 0.88 on standard causal models
5. **Flash attention hybrid:** decode performance within 10% of pure FA, with valid LSE combination
6. **18+ models validated:** Gemma3-12B, GPT-OSS-20B, Phi4-14B added via upstream sync
7. **SWA compaction:** base + SWA sub-cache on iSWA models
8. **128K validated:** compaction works at 128K on 3B-8B models with quality >= 0.85
9. **128K 30B capstone:** Qwen3-30B-A3B at 128K completes at least one ratio without swap
10. **Select pipeline regression:** V1 select cosine >= 0.950 maintained on all models throughout all phases

---

## Total Estimated Effort

| Phase | Effort | Calendar |
|-------|:------:|:--------:|
| Phase 1: Upstream Sync | High | 2-5 days |
| Phase 2: Solver Core (LAPACK + MIT port) | Medium-High | 3-5 days |
| Phase 3: GPU Solver (Metal) | High | 5-8 days |
| Phase 4: Query Generation | Medium | 2-3 days |
| Phase 5: OMP + Budget (+ swap solver) | High | 4-6 days |
| Phase 6: Chunked Compaction (+ overlap) | Medium | 3-4 days |
| Phase 7: Flash Attention Hybrid (+ LSE) | High | 5-8 days |
| Phase 8: High Compression + On-Policy + Sweep | Medium-High | 5-7 days |
| Phase 9: SWA + Attention Bias | Medium | 2-3 days |
| Phase 10: 128K Validation | Medium | 2-3 days |
| Phase 11: Capstone + Docs + Server | Medium | 2-3 days |
| **Total** | | **35-55 days** |
