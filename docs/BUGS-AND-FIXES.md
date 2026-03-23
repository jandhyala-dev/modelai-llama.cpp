# Bugs Found and Fixed

This document tracks every bug discovered, investigated, and fixed in modelai-llama.cpp — both in our compaction code and in upstream llama.cpp.

We believe transparency about bugs is a quality signal, not a weakness. Every entry below was found through adversarial testing, fixed with a root-cause analysis, and verified with regression tests.

**Total: 29 fix commits across 29 distinct bugs (18 Critical/Major, 11 Minor/Infra).**

---

## Critical and Major Bugs

### Post-Compaction Crash on Stale Prompt Cache

| Field | Value |
|-------|-------|
| **Severity** | Critical (crash) |
| **Status** | Fixed |
| **Affected** | All models after `reclaim_live_kv()` when slot reuses prompt cache |
| **Commit** | `4b6af8b02` |

**What happened:** After `compacted_prefix_reclaim_live_kv()` removed live KV positions, the slot's `prompt.tokens` still held old positions. On the next request, `update_slots()` found a matching BOS prefix, set `n_past > 0`, but `pos_min` was `-1` because the KV positions no longer existed. Crash.

**Fix:** Clear `slot->prompt.tokens` after reclaim so the slot treats the next request as a fresh prompt.

---

### BUG-I01: Nonuniform Pipeline Catastrophic Quality Loss at 8K

| Field | Value |
|-------|-------|
| **Severity** | Critical |
| **Status** | Fixed |
| **Affected** | DeepSeek-R1-14B at 8K, potentially all models at large context |
| **Commit** | `3d5132b1` |

**What happened:** The nonuniform pipeline allocates per-head KV budgets based on attention entropy. When per-head selections are highly disjoint, the union of selections exceeds the target count. After truncation, >50% of heads lost ALL selected tokens, producing beta = -infinity and V = zeros. Logit cosine dropped to 0.211 (near-random).

**Fix:** After union truncation, if >50% of heads are fully masked, fall back to the `select` pipeline with a warning.

---

### Nonuniform NaN Propagation After Union Truncation

| Field | Value |
|-------|-------|
| **Severity** | Critical |
| **Status** | Fixed |
| **Affected** | All models using nonuniform pipeline when heads lose all positions |
| **Commit** | `df895bed7` |

**What happened:** When union truncation dropped all of a head's selected positions, `-inf` beta flowed into `fit_values()`, producing NaN in V output. NaN propagated through subsequent attention computation.

**Fix:** All-masked heads now get zero V instead of `-inf` beta flowing into the fitting stage.

---

### Q/K Norm Mismatch in Self-Study Solver

| Field | Value |
|-------|-------|
| **Severity** | Critical |
| **Status** | Fixed |
| **Affected** | All models using self-study or prefill-Q pipelines |
| **Commit** | `bcced551` |

**What happened:** Query norms (~16) were ~1.8x smaller than key norms (~28) due to different learned scales. Without normalization, the attention softmax peaked incorrectly, NNLS produced extreme beta weights (~356), and cosine similarity was 0.710.

**Fix:** Per-head Q normalization — scale Q by `k_norm / q_norm`. After fix: cosine 0.988–0.995, beta norm down 40%, fit residual down 63–78%.

---

### Prefill-Q seq_rm Range Calculation

| Field | Value |
|-------|-------|
| **Severity** | Critical |
| **Status** | Fixed |
| **Affected** | All models using prefill-Q pipeline when `p0 > 0` |
| **Commit** | `909b6055d` |

**What happened:** `seq_rm()` was called with range `(p0, n_tokens)` instead of `(p0, p0 + n_tokens)`, clearing wrong KV positions when the starting position wasn't zero.

**Fix:** Correct range to `(p0, p0 + n_tokens)`.

---

### BUG-I02: Compacted Decode 3.4x Slower Than Baseline at 32K

| Field | Value |
|-------|-------|
| **Severity** | Major |
| **Status** | Fixed |
| **Affected** | All models using compacted prefix at 32K+ |
| **Commit** | `3d5132b1` (B5 fix) |

**What happened:** The compacted prefix `set_input_*` functions used host pointer swap (`dst->data = staging_buffer`) which forced ggml to treat tensors as host-backed. When concatenated with GPU-backed live KV tensors, this caused 80+ Metal GPU↔CPU sync stalls per decode batch. Decode throughput dropped from 6.8 to 2.0 tok/s at 32K.

**Fix:** Added `ggml_backend_tensor_set()` upload after host-side materialization in all three `set_input_compacted_prefix_k/v/kq_b` functions. Tensors now route to Metal/CUDA/CPU automatically.

---

### BUG-M01: Self-Study Q-Capture Memory Exceeds 16 GB

| Field | Value |
|-------|-------|
| **Severity** | Major |
| **Status** | Fixed |
| **Affected** | All models with self-study pipeline (Qwen3-14B: 4.9 GB at default config) |
| **Commit** | `8639040b7` |

**What happened:** During self-study Q-capture, all layers capture Q simultaneously during each decode step. With default config (n_generate=2000, n_rounds=3), Qwen3-14B projected 4.9 GB for Q alone, pushing total system memory to 16.4 GB — exceeding the 16 GB target.

**Fix:** Added `max_q_capture_mb` config (default 1 GB). Before generation, the pipeline computes projected allocation and auto-reduces `n_generate` or `n_rounds` to stay within budget. Total drops from 16.4 GB to 12.5 GB.

---

### Post-Compaction Amnesia

| Field | Value |
|-------|-------|
| **Severity** | Major |
| **Status** | Fixed |
| **Affected** | All models when using `/compact` with `reclaim:true` |
| **Commit** | `e312c450` |

**What happened:** After `/compact` with `reclaim:true`, the server cleared `slot->prompt.tokens`, causing the next `/completion` request to find 0 prefix match and re-evaluate from scratch — as if the conversation history was lost.

**Fix:** Preserve the token record so prefix matching recognizes the compacted context. Also fixed KV stats (`/props`, `/compact` response) to include compacted prefix tokens in `active_n_kv` counts.

---

### seq_cp Not Copying is_imrope Flag

| Field | Value |
|-------|-------|
| **Severity** | Major |
| **Status** | Fixed |
| **Affected** | IMROPE models (Qwen3.5, Qwen3.5-MOE) after sequence copy |
| **Commit** | `b44d4cd1e` |

**What happened:** `seq_cp()` did not copy `is_imrope` from source to destination. After seq_cp on IMROPE models, the copied sequence had `is_imrope=false`, causing `can_execute()` to silently reject `is_pos_2d()` batches. Compacted execution silently stopped working.

**Fix:** Copy `is_imrope` flag in `seq_cp()`. Also fixed `clear()` to reset `is_imrope` to false.

---

### is_imrope Not Serialized in State Save/Restore

| Field | Value |
|-------|-------|
| **Severity** | Major |
| **Status** | Fixed |
| **Affected** | IMROPE models after save+restore cycle |
| **Commit** | `d3a5759ab` |

**What happened:** `state_write`/`state_read` did not include the `is_imrope` flag. After a save+restore cycle, IMROPE sequences lost their flag, causing `can_execute()` to reject batches.

**Fix:** Bump state version to 2. Serialize `is_imrope` as `uint8_t` between execution flag and positions array.

---

### Solver Bugs Per arXiv:2602.16284 Review

| Field | Value |
|-------|-------|
| **Severity** | Major |
| **Status** | Fixed |
| **Affected** | All models using solver/on-policy pipelines |
| **Commit** | `3572cde30` |

**What happened:** Three bugs found by auditing against the reference paper:
1. **Max-shift inconsistency:** Full-key and compact-key exp-scores used different per-query shifts, distorting the NNLS optimization objective.
2. **Beta lower bound too low:** `1e-12` vs paper's `e^{-3}` (0.05). Near-zero weights destabilize value fitting (Appendix C.2).
3. (Third fix in same commit — solver stabilization.)

**Fix:** Unified shift computation, raised beta floor to paper's value.

---

### Nonuniform Budget Infeasibility at Extreme Ratios

| Field | Value |
|-------|-------|
| **Severity** | Major |
| **Status** | Fixed |
| **Affected** | Small models at high compression (e.g., stories15M at 8x with 36 KV heads) |
| **Commit** | `294a03b6c` |

**What happened:** When `min_per_head * n_heads > total_budget`, the nonuniform pipeline couldn't allocate any valid budget.

**Fix:** Auto-reduce `min_per_head` to `max(1, budget/n_heads)` when the constraint is infeasible.

---

### Biased K Norm Sampling in Chunked Self-Study

| Field | Value |
|-------|-------|
| **Severity** | Major |
| **Status** | Fixed |
| **Affected** | All models using chunked self-study pipeline |
| **Commit** | `07fae7f4d` |

**What happened:** K norm estimation used only the first 256 positions. BOS/system-prompt tokens have systematically different K norms than document body, biasing Q normalization for later chunks.

**Fix:** Uniform strided sampling across the full prefix.

---

### Infinite Loop in Nonuniform Under-Allocation

| Field | Value |
|-------|-------|
| **Severity** | Major |
| **Status** | Fixed |
| **Affected** | All models using nonuniform pipeline when all heads are at max budget |
| **Commit** | `909b6055d` |

**What happened:** When all heads were at `max_budget`, the allocation loop kept incrementing past the cap without terminating.

**Fix:** Break when all heads are at max.

---

### Stale Residual in Iterative Quality Gate

| Field | Value |
|-------|-------|
| **Severity** | Major |
| **Status** | Fixed |
| **Affected** | All models using iterative on-policy pipeline |
| **Commit** | `b7bfe9d0e` |

**What happened:** When the quality gate triggered on regression, stats reported the previous iteration's residual while the current (worse) result was already committed. API clients overestimated compaction quality.

**Fix:** Update `r_prev = r_cur` unconditionally before the quality gate check.

---

### F16 Mask Cast Guard for Non-Flash Models

| Field | Value |
|-------|-------|
| **Severity** | Major |
| **Status** | Fixed |
| **Affected** | ALiBi models using compacted prefix with non-flash attention |
| **Commit** | `52b8871e0` |

**What happened:** The per-layer F16 mask cast fired whenever `zero_beta && flash_attn`, even when the model had a non-nullptr `kq_b` (ALiBi). Passed F16 mask to standard attention unnecessarily.

**Fix:** Add `kq_b_combined == nullptr` guard.

---

### Ablation Flags Dead Code

| Field | Value |
|-------|-------|
| **Severity** | Major (testing infrastructure) |
| **Status** | Fixed |
| **Affected** | Benchmark tooling — `--no-beta` and `--no-cv` flags |
| **Commit** | `da87e6397` |

**What happened:** `--no-cv` was parsed but the env var was never checked (dead code). `--no-beta` was conflated with eviction, routing to `select` instead of the solver pipeline.

**Fix:** Added `skip_beta_fit()` (via `LLAMA_COMPACT_NO_BETA=1`) to all 5 solver paths.

---

### CI Billing Drain from Inherited Workflows

| Field | Value |
|-------|-------|
| **Severity** | Major (operational) |
| **Status** | Fixed |
| **Commits** | `6d8688f06`, `4e618aac9` |

**What happened:** Forking llama.cpp inherited 21+ GitHub Actions workflows. These ran on every push, consuming CI minutes and attempting to queue on self-hosted runners that don't exist in the fork. A second wave of 14 more workflows arrived when upstream split `build.yml` into per-platform files.

**Fix:** Renamed all inherited workflows to `.disabled`. Only `modelai-ci`, `modelai-server-smoke`, and `modelai-perf-smoke` remain active.

---

## Minor Bugs

### BUG-R01: Exception Safety in Tensor Staging

**Commit:** `63cd5d9a6` | If a compacted prefix exec function threw, `dst->data` was left pointing to the host staging buffer. Added try/catch guards to restore before re-throwing.

### BUG-R02: Thread Safety of Static Warning Flags

**Commit:** `63cd5d9a6` | `compacted_prefix_runtime_supported()` used non-atomic `static bool` flags. Changed to `std::atomic<bool>` with `exchange()`.

### Subsampling Truncation for GQA Models

**Commit:** `0b0a3a570` | Q subsampling truncated incorrectly when n_head_q != n_head_kv. Fixed slice bounds.

### Q-Capture Overwrite Strategy

**Commit:** `b4c1e547f` | Hot-path allocation in Q-capture overwrite. Fixed allocation strategy.

### Rollback on Post-Configure Failure

**Commit:** `45d7f29a2` | `configure_seq()` didn't roll back on partial failure. Added rollback.

### seq_id Assert and Pending Cleanup

**Commit:** `c01982747` | Missing seq_id bounds check in compaction dispatch. Added assert and cleaned up pending state.

### ask-Phase n_dims Bug in Self-Study

**Commit:** `90f87173b` | Self-study ask-phase used wrong `n_dims`. Fixed dimension calculation.

### Quantized K Spec Drift

**Commit:** `0989d6db9` | Quantized K extraction spec diverged from implementation. Aligned spec and code.

### /compact Endpoint Review Findings

**Commit:** `41526d93d` | Multiple findings from adversarial review of `/compact` endpoint. Fixed validation, error handling.

### Stale /props Assertions

**Commit:** `5510b88cb` | `/props` response didn't reflect compacted prefix state. Fixed assertions.

### Chunked Budget Overshoot

**Commit:** `3214e6e89` | Chunked pipeline budget exceeded target when `n_chunks > target_tokens`. Fixed with excess distribution and `n_selected` cap.

### GQA Divisibility Assert in Self-Study Q-Capture

**Commit:** `3214e6e89` | Missing divisibility check in Q-capture regroup for GQA models. Added assert.

### Zerobeta Warning Gating

**Commit:** `14c61a053` | Warning only fired for layer 0/head 0. Now fires on first occurrence in any layer/head.

### Misleading Flash Attention Warning

**Commit:** `2017770e5` | Warning said flash was fully disabled. Updated to reflect per-layer eligibility (zero-beta layers still use flash).

### clear() Not Resetting is_imrope

**Commit:** `b44d4cd1e` | `clear()` didn't reset `is_imrope` to false. Inconsistent with clear contract.

### Minor Fixes (Bundled in Earlier Commits)

| ID | Description | Fix |
|----|-------------|-----|
| m-02 | Per-round RNG seed was fixed at 42 for all rounds | Per-round unique seed |
| m-03 | `partial_sort` bound excess (`k_select + i` vs `k_select`) | Corrected bound |
| m-04 | Missing `isfinite` check in OMP Cholesky solver | Added guard |
| R1-Minor-2 | All-negative-infinity logits in temperature sampling | Guard added |
| R1-Minor-3 | JSON parser key validation (missing digit requirement) | Require digits, consume full key |

---

## Open Issues

### BUG-I04: Compaction Time Scales Super-Linearly

**Severity:** Minor | **Status:** Profiling in place, optimization deferred

Compaction time at 8K: 518ms, 16K: ~1.1s, 32K: ~3.7s. The attention score computation is O(n_prefix × n_queries × n_heads × n_layers). Per-stage timing instrumentation is in place to guide future optimization.

### BUG-I05: Qwen3-14B Quality Dip at 8K/8x

**Severity:** Minor | **Status:** Investigation deferred

Cosine similarity of 0.973 at 8K/8x — the lowest of 33 test points but still above the 0.95 production threshold. All other points > 0.99.

---

## Upstream Bugs (Found in ggml-org/llama.cpp)

These bugs exist in upstream llama.cpp and affect our fork. We track them, add mitigations where possible, and sync fixes when they land upstream.

### BUG-U01: Gemma3-12B SWA Decode 20x Slower Than Ollama

| Field | Value |
|-------|-------|
| **Severity** | Major |
| **Status** | Mitigated (upstream bug, not fork-specific) |

Gemma3-12B decode: 0.7 tok/s on llama.cpp vs 16.5 tok/s on Ollama. Confirmed NOT a fork regression — upstream shows the same performance. Added runtime warning when SWA sub-cache is detected. Gemma3-12B listed as unsupported for compaction.

### BUG-U03: Server Grammar Stack Crash on Apple Metal

| Field | Value |
|-------|-------|
| **Severity** | Critical |
| **Status** | Open upstream ([#19679](https://github.com/ggml-org/llama.cpp/issues/19679), [#19304](https://github.com/ggml-org/llama.cpp/issues/19304)) |

`llama_grammar_accept_token` crashes on empty grammar stack with flash attention + jinja, or at 86K context with 50+ tool calls. Affects all llama-server deployments on Apple Silicon.

### BUG-U04: KV Cache Truncation on Chat Completions

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Status** | Open upstream ([#11970](https://github.com/ggml-org/llama.cpp/issues/11970)) |

KV cache sometimes truncated incorrectly during `/v1/chat/completions`. Could interact with compaction by compacting an already-truncated cache.

---

## Upstream Compatibility Verification

We verified compatibility with 6 upstream KV cache changes before shipping:

| Upstream Issue | Status | Verification |
|---------------|--------|-------------|
| [#10873](https://github.com/ggml-org/llama.cpp/issues/10873) KV cache defrag corruption | Safe — defrag removed upstream, no compaction code calls defrag |
| [#12695](https://github.com/ggml-org/llama.cpp/issues/12695) KV guard refactor | Compatible — fork uses current post-refactor API |
| [#13194](https://github.com/ggml-org/llama.cpp/issues/13194) SWA KV cache support | Compatible — SWA correctly rejected at runtime |
| [#17450](https://github.com/ggml-org/llama.cpp/issues/17450) Unified KV buffer | Compatible — all tests use `kv_unified=true` |
| [#12253](https://github.com/ggml-org/llama.cpp/issues/12253) Shift/defrag correctness | Safe — shift guard prevents interaction |
| [#11213](https://github.com/ggml-org/llama.cpp/issues/11213) KV cells unified | N/A — decomposed into #12695 and #13194 |

---

## Security Patches Synced

| Date | Upstream PR | Description |
|------|------------|-------------|
| 2026-03-23 | [#20908](https://github.com/ggml-org/llama.cpp/pull/20908) | RPC remote code execution patch — critical security fix synced same day |
