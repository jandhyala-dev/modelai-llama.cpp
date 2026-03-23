# Design Decisions

Key architecture and implementation decisions in modelai-llama.cpp, with rationale. Each decision was made deliberately and validated through adversarial review.

---

## 1. Compacted prefix lives inside `llama_kv_cache`, not as a separate system

**Decision:** All compaction state is a sidecar inside the KV cache abstraction, not a standalone module.

**Why:** The KV cache already owns sequence lifecycle (copy, clear, remove), position tracking, and memory allocation. A separate system would need to mirror all of this, creating synchronization bugs. By living inside `llama_kv_cache`, the compacted prefix inherits save/restore, sequence operations, and memory management automatically.

**Trade-off:** The KV cache struct is more complex. But the alternative — a parallel state manager — would be far more error-prone.

---

## 2. `select` pipeline is the production default, not `solver`

**Decision:** The solver pipeline (full beta + V fitting) is excluded from the production allowlist. `select` (top-k attention-score selection, zero beta) is the default.

**Why:** The solver produces cosine similarity of -0.174 to 0.906 on GQA models — below the 0.95 threshold and sometimes negative. `select` achieves 0.946-0.999 across 15 models. The solver is also 200-750x slower (39-48s vs 62-185ms).

**When solver is used:** Self-study and on-policy pipelines use the solver for beta/V fitting, but only after position selection is locked in. The solver is never used for production position selection on GQA models.

---

## 3. All solver math in fp32

**Decision:** Query extraction, attention scoring, NNLS fitting, and V fitting all run in fp32. Results are cast to model/cache dtype only when written to the compacted prefix store.

**Why:** The matrix sizes are small (t ~32-256 selected positions per head). fp32 is fast enough at these sizes, and mixed-precision artifacts in the NNLS solver (which involves divisions, norms, and iterative updates) would be hard to debug. Beta especially must stay fp32 — it's an additive bias to log-space attention scores, so quantization noise would change which tokens get attended to.

---

## 4. No external dependencies for the solver

**Decision:** Pure C++ dense solver. No LAPACK requirement for V0. LAPACK `sgels` added as optional in V2 (via Accelerate framework on macOS).

**Why:** llama.cpp's build system is cmake-only with zero mandatory external dependencies. Adding a LAPACK requirement would break builds on Android, RISC-V, and embedded platforms. The matrix sizes (t × d_head, typically 32-256 × 128) are small enough that dense C++ solvers with Cholesky decomposition perform well.

**V2 addition:** LAPACK `sgels` added as the preferred path for QR-based least-squares (higher numerical stability at high compression), with Cholesky as fallback when LAPACK is unavailable.

---

## 5. Shared position schedule across heads

**Decision:** One logical-position array per sequence (not per-head). Per-layer and per-KV-head fitting uses this shared schedule.

**Why:** Simplifies the compacted prefix store — one position array instead of n_heads position arrays. The `select` pipeline selects positions by aggregate attention score, so positions are shared by design. Per-head position selection (as in nonuniform) still uses the shared array after union truncation.

**Trade-off:** Nonuniform per-head budgets must union-truncate to a shared set, which can cause quality loss when head selections are highly disjoint (BUG-I01 fallback mitigates this).

---

## 6. `ggml_backend_tensor_set()` for GPU upload

**Decision:** Compacted prefix tensors are materialized into a host staging buffer, then uploaded via `ggml_backend_tensor_set()`.

**Why:** The original pattern used host pointer swap (`dst->data = staging_buffer`), which forced ggml to treat tensors as host-backed. When concatenated with GPU-backed live KV tensors, this caused 80+ Metal GPU↔CPU sync stalls per decode batch, making compacted decode 3.4x slower than baseline at 32K (BUG-I02). `ggml_backend_tensor_set()` routes to Metal/CUDA/CPU automatically.

---

## 7. Per-layer flash attention eligibility

**Decision:** Each layer independently decides flash vs standard attention based on whether its compacted prefix has zero beta.

**Why:** Flash attention cannot handle additive `kq_b` (beta) bias. Zero-beta layers (from `select` pipeline or solver layers with near-zero betas) can safely use flash. Non-zero-beta layers must use standard attention. Per-layer eligibility maximizes flash usage — in practice, most layers have zero beta with the `select` pipeline.

---

## 8. M-RoPE blocked, IMROPE text-only allowed

**Decision:** Compaction is blocked on M-RoPE models (Qwen2-VL, GLM4) but allowed on IMROPE text-only models (Qwen3.5, Qwen3.5-MOE).

**Why:** M-RoPE encodes spatial position (x, y, t) into the rotary embedding. Compaction assumes 1D position ordering — merging spatial positions would produce nonsensical attention. IMROPE is a text-only variant that uses standard 1D positions for text tokens; the `is_imrope` flag distinguishes safe text-only batches from spatial batches.

---

## 9. Quality gate for iterative on-policy

**Decision:** Iterative on-policy refinement stops when the solver residual doesn't improve by at least 0.5% between iterations.

**Why:** The solver residual (partition-sum relative error) is a cheap proxy for fit quality. If it doesn't decrease, further iterations are generating Q from a fixed point of the compacted representation — the Q distribution has converged and won't change. This prevents wasting compute on non-improving iterations.

---

## 10. State serialization with version bumps

**Decision:** Compacted prefix state has an explicit version number (currently version 2). New fields require a version bump.

**Why:** Save/restore must work across server restarts and potentially across binary versions. Version 1 stored positions, K, V, beta. Version 2 added `is_imrope` flag. The reader checks the version and knows exactly which fields to expect.

---

## 11. One-shot guard for auto-compaction

**Decision:** Auto-compaction fires once when the threshold is crossed, then disables itself until the next reclaim cycle.

**Why:** Without the guard, compaction could trigger a retry loop: compact → reclaim → fill → compact → reclaim → ... The one-shot guard ensures each fill cycle gets exactly one compaction attempt.

---

## 12. Merge-based history for `modelai-main`

**Decision:** Upstream syncs use merge commits (not rebase) into `modelai-main`.

**Why:** Merge commits preserve the exact point where upstream entered the product branch. This makes `git log --first-parent` show a clean timeline of fork milestones with merge points clearly marked. Rebase would flatten the history, making it impossible to tell which commits came from upstream vs fork.

---

## 13. Adversarial review protocol for every merge

**Decision:** No code enters `modelai-main` without passing a 13-section adversarial review with concrete traces and a disprove-it pass.

**Why:** KV cache compaction touches attention computation — the core of transformer inference. A subtle bug (wrong index, truncated division, stale state) produces silently wrong output, not a crash. The review protocol requires walking real numbers through the code, testing boundaries, and actively trying to break it. This caught 18 Critical/Major bugs that would have shipped otherwise.

See `AGENTS.md` for the full protocol.
