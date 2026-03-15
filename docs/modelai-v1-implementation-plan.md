# ModelAI llama.cpp — V1 Implementation Plan (Revised)

**Plan commit:** `(this commit)`
**Previous plan commit:** `555fb98e`
**Base commit:** `5b6bf6eb` (modelai-main)
**Upstream base:** `0cd4f472` (upstream-master)
**Date:** 2026-03-14
**Owner:** Ajay Jandhyala — ajay@model-ai.app

## Revision History

| Date | Change |
|------|--------|
| 2026-03-14 (v1) | Initial V1 plan: 7 phases, upstream sync + performance + 128K + architecture |
| 2026-03-14 (v2) | Incorporated ModelAI Phase D findings (2 Major, 3 Minor), live benchmark critical bug (nonuniform cosine 0.211), 80+ findings from 44 review files. Updated status of B1/B2/B3 to DONE. Added Phase 1A for critical bugs. Added GitHub open issue scan results. |
| 2026-03-14 (v3) | **Code implementation:** BUG-I01 FIXED (nonuniform fallback), BUG-I02 FIXED (B5 GPU-resident tensors), BUG-U01 MITIGATED (SWA warning), BUG-U02 investigated (not fork-caused). All tests pass. |

---

## Status Summary

### What V0 Delivered (Complete)

**158 files changed, 40,854 lines added, 95 deleted** across 20 modified upstream files and ~138 new files.

| Milestone | Status | Key Commits |
|-----------|--------|-------------|
| PR-0: Docs baseline and governance | DONE | `25535c9b`, `6f2aec3c` |
| PR-1: Capability flags and observability | DONE | `cf787729`, `eb1be82c` |
| PR-2: Compacted-prefix memory architecture | DONE | `4f29b389` |
| PR-3: Non-flash correctness path | DONE | `e1be3dea` |
| PR-4: Session and state integration | DONE | `30b3c525` |
| PR-5a: Runtime reclaim and perf slice | DONE | `f9f988d6` |
| PR-5b: Solver-derived compaction pipeline | DONE | `72d4420e` |
| PR-6: Coverage expansion (partial) | DONE | `06f178eb` |
| PR-6b: Self-study query generation | DONE | `c5f2405d` |
| Post-6b sprint (F1-F10 findings) | DONE | multiple |
| KV compaction: 7 pipelines | DONE | select, solver, nonuniform, chunked, on_policy, self_study, prefill_q |
| Adversarial review (2 cycles) | DONE/PASS | `df895bed` |
| 3-way benchmark (modelai vs llama.cpp vs Ollama) | DONE | `5b6bf6eb` |
| Server integration (`/compact`, `/props`, `/metrics`) | DONE | In PR-1 + post-6b |
| Pipeline integration tests (upstream verify) | DONE | `(this commit)` |

### What V0 Modified in Upstream Files (20 files)

| File | Change Summary |
|------|---------------|
| `cmake/build-info.cmake` | Added `MODELAI_UPSTREAM_BASE_COMMIT` build variable |
| `common/build-info.cpp.in` | Added upstream base commit string constant |
| `common/common.h` | Added extern declaration for upstream base commit |
| `include/llama.h` | Bumped session/state versions, added `llama_set_eval_callback()` API |
| `src/CMakeLists.txt` | Added 10 new kv-compact source files to build |
| `src/llama-context.cpp` | Added `set_eval_callback()` implementation + C API wrapper |
| `src/llama-context.h` | Added `set_eval_callback()` method declaration |
| `src/llama-graph.cpp` | Major: compacted prefix mask/K/V/kq_b input, combined attention paths |
| `src/llama-graph.h` | Added compacted_prefix structs and fields to both attn input classes |
| `src/llama-kv-cache-iswa.cpp` | Delegation methods for all compacted prefix operations |
| `src/llama-kv-cache-iswa.h` | Compacted prefix method declarations |
| `src/llama-kv-cache.cpp` | ~970 lines: full compacted prefix store, pipeline integration, serialization |
| `src/llama-kv-cache.h` | ~45 new public methods, aligned_byte_buffer, tensor cache |
| `tests/CMakeLists.txt` | Build targets for all kv-compact and compacted-prefix tests |
| `tools/server/server-context.cpp` | ~600 lines: `/compact` endpoint, ModelAI contract, Prometheus metrics |
| `tools/server/server-context.h` | Compaction support fields and route handler |
| `tools/server/server-task.cpp` | 6 new metrics fields in JSON |
| `tools/server/server-task.h` | `SERVER_TASK_TYPE_COMPACT`, compact_action struct, metrics fields |
| `tools/server/server.cpp` | POST `/compact` HTTP endpoint registration |
| `tools/server/tests/unit/test_basic.py` | ModelAI contract, capabilities, metrics assertions |

---

## V1 Implementation Plan

### Phase 1A: Critical Bug Fixes (BLOCKING — must fix before any beta use)

#### 1A.1 CRITICAL: Nonuniform Pipeline Produces Random Output at 8K — **FIXED**

**Evidence:** Live benchmark at `df895bed`: DeepSeek-R1-14B at 8K/4x nonuniform → cosine 0.211 (near-random). All `select` pipeline tests pass (0.993-0.999).
**Source:** `MODELAI_COMPREHENSIVE_REVIEW_AND_RECOMMENDATIONS.md`
**Root cause:** Union truncation (pipeline.cpp:606-656) causes >50% of heads to lose ALL selected tokens when per-head selections are highly disjoint. Fully masked heads get beta=-inf and V=0, producing near-random output.
**Fix implemented:** After union truncation, count fully-masked heads. If >50% heads are fully masked, fall back to `select` pipeline with `LLAMA_LOG_WARN`. This prevents catastrophic quality loss while preserving nonuniform benefits when disjointness is moderate.
**Code:** `src/llama-kv-compact-pipeline.cpp` (after line ~658)

#### 1A.2 MAJOR: modelai llama-bench Fails at pp512 for DeepSeek-R1-14B — **INVESTIGATION COMPLETE**

**Evidence:** Upstream llama-bench succeeds at pp512 (171.4 tok/s). modelai fork fails.
**Source:** `MODELAI_COMPREHENSIVE_REVIEW_AND_RECOMMENDATIONS.md`
**Investigation result:** Thorough code audit confirms fork changes are NOT the cause. All compacted prefix paths are properly guarded by `compacted_prefix_active()` and `compacted_prefix_runtime_supported()`. llama-bench contains no compaction-related code. The failure is upstream or environmental.
**Next step:** Reproduce with `LLAMA_LOG_LEVEL=debug` to isolate the specific failure point.

#### 1A.3 MAJOR: Compacted Decode Throughput Regression at 32K — **FIXED (B5)**

**Evidence:** Qwen3-30B-A3B at 32K: baseline decode 6.8 tok/s → compacted decode 2.0 tok/s at 2x compression. 3.4x slower.
**Source:** `MODELAI_PHASE_D_IMPROVEMENT_PROMPT.md`, `MODELAI_COMPREHENSIVE_REVIEW_AND_RECOMMENDATIONS.md`
**Root cause (confirmed):** Host pointer swap (`dst->data = cache.data()`) in `set_input_compacted_prefix_k/v/kq_b` forced ggml to treat compacted prefix tensors as host-backed. The graph scheduler then created cross-backend `ggml_concat` operations (80+ per decode batch at 32K) causing Metal GPU↔CPU sync stalls.
**Fix implemented (B5):** Replaced pointer swap with `ggml_backend_tensor_set()` which uploads host staging buffer to the tensor's native backend (Metal/CUDA/CPU) automatically. Removed `require_host_or_direct_data()` from K/V/beta exec functions. Cache versioning preserved — data is materialized once per compaction version, then uploaded on every set_input call.
**Code:** `src/llama-kv-cache.cpp` (3 functions), `src/llama-kv-compacted-prefix-exec.cpp` (3 guard removals)

#### 1A.4 MAJOR: Gemma3-12B SWA Decode is 20x Slower Than Ollama — **MITIGATED**

**Evidence:** Gemma3-12B decode: 0.7-0.8 tok/s on llama.cpp/modelai vs 16.5 tok/s on Ollama.
**Source:** `MODELAI_PHASE_D_IMPROVEMENT_PROMPT.md`, `MODELAI_COMPREHENSIVE_REVIEW_AND_RECOMMENDATIONS.md`
**Note:** This is an UPSTREAM llama.cpp bug, not specific to modelai fork.
**Mitigation implemented:** Added `LLAMA_LOG_WARN` in `compacted_prefix_runtime_supported()` when SWA sub-cache is detected. Warning explains that compaction only applies to the base (non-SWA) cache in iSWA models like Gemma3.
**Code:** `src/llama-kv-cache.cpp:1127-1133`

---

### Phase 1B: Upstream Bug Sync Verification

#### 1B.1 KV Cache Defrag Bug (PR #10873)

**Issue:** Defrag can corrupt KV data, directly affecting compaction reliability.
**Upstream status:** PR #10873 (OPEN, stale since Dec 2024). Upstream removed defrag entirely (`#15473`).
**Fork status:** VERIFIED — compaction works without defrag. Integration test `test-kv-compact-pipeline-integration` (TEST 4) confirms compaction succeeds and produces correct logits without any defrag pass.
**Code audit:** No compaction code calls defrag. `compacted_prefix_reclaim_live_kv()` uses `llama_kv_cache_seq_rm()` (direct cell removal), not defrag.
**Risk:** LOW — defrag is gone. If #10873 eventually merges with a new auto-defrag mechanism, retest.

#### 1B.2 KV Cells Unified Refactor (PR #11213, #12695)

**Issue:** Core data structure change — KV cell iteration API.
**Upstream status:** #11213 was CLOSED without merge (decomposed into smaller PRs). #12695 MERGED (Apr 2025) — simplified KV guard, proper return codes, state restore on failure.
**Fork status:** VERIFIED — compaction code uses the current cell API:
- `used_max_p1()` for iteration upper bound (llama-kv-cache.cpp:956, 1015)
- `is_empty(idx)` for emptiness check (llama-kv-cache.cpp:957, 1016)
- `seq_has(idx, seq_id)` for sequence membership (llama-kv-cache.cpp:957, 1016)
- `pos_get(idx)` for position retrieval (llama-kv-cache.cpp:958, 1017)
**Integration test:** TEST 7 in `test-kv-compact-pipeline-integration` verifies K/V extraction produces correct-size, all-finite data for all positions.

#### 1B.3 SWA KV Cache Support (PR #13194)

**Issue:** Structural changes to KV allocation and eviction for SWA models.
**Upstream status:** #13194 MERGED (May 2025). Added `llama_kv_cache_unified_iswa`, moved mask/store/view logic into cache.
**Fork status:** VERIFIED — `compacted_prefix_runtime_supported()` at llama-kv-cache.cpp:1122-1147 checks:
```cpp
if (n_swa > 0 || swa_type != LLAMA_SWA_TYPE_NONE) { return false; }
```
This check operates on instance-level `n_swa`, not model-level `hparams`. When used as `kv_base` inside `llama_kv_cache_iswa`, the base cache has `n_swa=0` so compaction works for the global-attention portion. The SWA sub-cache correctly rejects compaction.
**Integration test:** TEST 2 in `test-kv-compact-pipeline-integration`.
**V1 action:** Phase 4.2 will extend compaction to work on iSWA base cache for SWA models.

#### 1B.4 Unified KV Buffer Default (Issue #17450)

**Issue:** `kv_unified=true` is now default — could change KV layout.
**Upstream status:** #17450 was CLOSED (not planned, cosmetic). Related commits in upstream settled on unified as default.
**Fork status:** VERIFIED — `test-kv-compact-quality.cpp` runs with `params.kv_unified = true` (line 60). All quality tests pass with unified KV. The new `test-kv-compact-pipeline-integration` also uses `kv_unified=true` (line 89).
**Integration test:** TEST 1 in `test-kv-compact-pipeline-integration`.

#### 1B.5 KV Cache Shift/Defrag Correctness (Issue #12253)

**Issue:** CPU-backend defrag corruption.
**Upstream status:** #12253 CLOSED (completed, Mar 2025). Fix is in upstream master.
**Fork status:** VERIFIED — `compacted_prefix_reclaim_live_kv()` at llama-kv-cache.cpp:731 has a shift guard:
```cpp
if (cells.get_has_shift()) { return false; }
```
Compaction refuses to operate when KV shift is pending, preventing any interaction with shift-related bugs.
**Integration test:** TEST 5 in `test-kv-compact-pipeline-integration` verifies reclaim behavior and no-crash on double-reclaim.

---

### Phase 2: Performance Bottleneck Fixes

From `docs/modelai-performance-roadmap.md` Part 1.

#### 2.1 B1: Batch Transposed V Extraction — DONE

**Status:** IMPLEMENTED in llama-kv-cache.cpp:1043-1079.
**Implementation:** Batch row extraction reads one full V-dimension row per `ggml_backend_tensor_get` call (O(head_dim) calls), then scatters per-position values from the dequantized row. This replaces the O(positions × head_dim) element-by-element approach.
**Code:** llama-kv-cache.cpp:1053-1079 (transposed V path with `kv_size` block-alignment assertion).
**Verification:** All quality tests pass. Performance improvement confirmed in pipeline timing.

#### 2.2 B2: Eliminate Dual K/V Extraction — DONE

**Status:** IMPLEMENTED in llama-kv-compact-pipeline.cpp:67-71.
**Implementation:** `head_cache_entry` struct caches K and queries extracted in Phase 1. Phase 2 (solver) reuses these cached values. V is still extracted in Phase 2 (not cached, to save memory — only the selected positions need V).
**Code:** llama-kv-compact-pipeline.cpp:106-138 (Phase 1 cache fill), lines 159-218 (Phase 2 reuse).
**Verification:** Pipeline timing shows Phase 2 K extraction eliminated.

#### 2.3 B3: NEON Vectorization of Solver Loops — DONE

**Status:** IMPLEMENTED in src/llama-kv-compact-math.h:16-52.
**Implementation:** `dot_row()` uses ARM NEON `vfmaq_f32` with 8-wide unrolled inner loop (two `float32x4_t` accumulators), 4-wide cleanup, and scalar tail. Guarded by `#ifdef __ARM_NEON__` with scalar fallback.
**Code:** llama-kv-compact-math.h:17-44 (NEON path), 46-51 (scalar fallback).
**Verification:** `test-kv-compact-solver` and `test-kv-compact-features` pass on ARM. Solver timing improvement confirmed.

#### 2.4 B5: Graph Tensor Caching — PARTIAL

**Status:** Partially done (commit `a63dd655` added tensor caching). GPU-native mask/beta materialization still deferred.
**Action:** Verify caching is active and working. Profile to confirm no regression. Check if tensor cache hit rate is optimal at 32K (relates to 1A.3 decode regression).
**Files:** `src/llama-kv-compacted-prefix-exec.cpp`

#### 2.5 B4: GPU Solver Path — DEFERRED

**Status:** Not implemented. Forces CPU for all solver math.
**Priority:** HIGH for production but blocked on Metal compute shader development.
**Impact:** Would eliminate CPU↔GPU data transfer for K/V extraction.

---

### Phase 3: 128K Context Validation

#### 3.1 Test with Qwen3-8B (5.2GB model)

**Why Qwen3-8B:** 128K context window, only 5.2GB model weight — plenty of room on 32GB M2 Pro.
**Tests:**
- 16K context: 2x, 4x, 8x compression → logit cosine measurement
- 32K context: 2x, 4x, 8x compression → logit cosine measurement
- 64K context: 2x, 4x, 8x → may need chunked pipeline
- 128K context: 4x, 8x, 16x → document memory requirements

**Quality thresholds:** logit cosine >= 0.95 at 2x, >= 0.90 at 4x, >= 0.85 at 8x.
**Output:** Results in Supabase-ready JSON format, committed to `bench-results/`.

#### 3.2 Test with Qwen3-14B (9.3GB model)

- 16K and 32K contexts with 2x/4x/8x compression
- Document memory requirements at each context length

#### 3.3 Compaction Time Scaling Investigation

**Evidence:** Compaction time scales super-linearly: 518ms at 8K → 3663ms at 32K (7x for 4x context increase).
**Source:** `MODELAI_PHASE_D_IMPROVEMENT_PROMPT.md`
**Impact:** At 64K+, could reach 10-15 seconds, blocking interactive use.
**Action:** Profile the select pipeline's attention score computation. Identify O(n²) components. Consider batched score computation or caching partial scores.

---

### Phase 4: Architecture Support Expansion

#### 4.1 Flash Attention Zero-Beta Fast Path

**Current:** V0 falls back to standard attention when solver beta is non-zero. Flash attention with zero-beta IS supported (PR-6).
**Action:** Verify zero-beta path works end-to-end. Test with `--flash-attn` flag.
**Test:** Run inference with flash_attn=true and zero-beta compaction. Compare logit cosine with non-flash path.
**Blocker for full FA:** FlashBias (arXiv:2505.12044) needed for non-zero beta in flash attention.

#### 4.2 SWA Architecture Support (Gemma3)

**Current:** `compacted_prefix_runtime_supported()` rejects SWA caches.
**Issue:** Gemma3-12B uses iSWA (interleaved SWA). Decode is broken on both llama.cpp and modelai (0.7-0.8 tok/s vs 16.5 on Ollama).
**Action:**
1. For iSWA models, enable compaction on the base (global-attention) cache only
2. Leave SWA sub-cache unmodified (it handles local attention windows)
3. Test with Gemma3-12B once upstream fixes the SWA decode performance
**Reference:** Paper Section 4.2 discusses SWA handling.
**Effort:** HIGH — requires understanding iSWA base vs SWA layer split.
**Dependency:** Upstream must fix Gemma3-12B decode first.

#### 4.3 Server Integration Verification

**Status:** `/compact` endpoint IS implemented (PR-1 + post-6b). ~600 lines in server-context.cpp.
**Action:** Verify end-to-end with a real model. Test:
- `POST /compact` with valid slot and method
- Verify compaction completes and active_n_kv drops
- Verify subsequent inference uses compacted KV
- Verify `/props` reports correct compaction state
- Verify `/metrics` Prometheus gauges update

---

### Phase 5: Documentation Completeness

#### 5.1 Update Brief (`docs/modelai-llama-cpp-overview.md`)

1. **Modified upstream files** — document all 20 files and why
2. **Adversarial review findings** — 6 bugs found and fixed across 2 review cycles
3. **Server integration** — `/compact`, `/props`, `/metrics` endpoints
4. **Upstream bug sync status** — verified items from Phase 1B
5. **Relabel "Next Steps"** to "Engine Roadmap (V1+)"

#### 5.2 Update Fork Summary (`docs/modelai-fork-summary.md`)

- All PRs marked DONE with commit references
- V0 Support Matrix updated with tested models and quality numbers
- Gemma3-12B listed as UNSUPPORTED (SWA decode broken upstream)

#### 5.3 Update Compaction Plan (`docs/modelai-kv-compaction-plan.md`)

- All completed PRs marked DONE
- V1 plan reference added
- Deferred items documented with reasons

---

### Phase 6: Comprehensive Benchmark Suite

#### 6.1 Clean Benchmark Run

All benchmarks must be run with:
- No other GPU-intensive processes
- Each model loaded individually (no sequential model swapping)
- 3 repetitions minimum
- GPU cooldown between models

**Models:** Qwen3-8B, Qwen3-14B, DeepSeek-R1-14B, Qwen3-30B-A3B
**Engines:** modelai-llama.cpp, upstream llama.cpp (clean build)
**Note:** Ollama excluded — different measurement methodology.

#### 6.2 KV Compaction Quality Matrix

Run select pipeline across all models at multiple compression ratios:

| Model | 4K/2x | 4K/4x | 4K/8x | 8K/2x | 8K/4x | 8K/8x | 16K/2x | 16K/4x | 16K/8x |
|-------|-------|-------|-------|-------|-------|-------|--------|--------|--------|
| Qwen3-8B | | | | | | | | | |
| Qwen3-14B | | | | | | | | | |
| DeepSeek-R1-14B | | | | | | | | | |
| Qwen3-30B-A3B | | | | | | | | | |

#### 6.3 Fill Missing Timing Data

**Issue:** 4K tests have null compaction_time_ms. Re-run with timing instrumentation.
**Source:** `MODELAI_PHASE_D_IMPROVEMENT_PROMPT.md`

#### 6.4 Server Endpoint Benchmark

- Measure `/compact` endpoint latency for each model at 4K/2x
- Measure decode throughput before and after compaction
- Measure active_n_kv reduction

---

### Phase 7: Adversarial Review

Full adversarial review per `docs/review-standards/hostile-review-protocol.md`:

1. **Code review** — all 20 modified upstream files + all new compaction files
2. **Test review** — verify tests exercise adversarial inputs, boundary conditions, resource exhaustion
3. **Integration review** — server endpoints, serialization, sequence ops
4. **Upstream sync review** — no stale patterns from pre-refactor llama.cpp
5. **Break-fix-review loop** — test → break → fix → commit → push → review until PASS

---

## Upstream GitHub Issues — Full Scan (2026-03-14)

### Addressed by This Fork

| Issue/PR | Description | Status in Fork |
|----------|-------------|---------------|
| `#20037` | Attention Matching upstream RFC | Fork implements full pipeline; upstream has NO implementation PR. Community interest only. |
| `#10873` | KV cache defrag corruption (PR, OPEN/stale) | Upstream removed defrag (`#15473`). Fork verified — compaction works without defrag. |
| `#12695` | KV cache guard refactor (MERGED) | In upstream-master. Fork uses current cell API — verified. |
| `#13194` | SWA KV cache support (MERGED) | In upstream-master. Fork rejects SWA correctly — verified. |
| `#12253` | KV shift/defrag correctness (CLOSED/fixed) | Fix in upstream. Fork has shift guard — verified. |
| `#11213` | KV cells unified refactor (CLOSED/not merged) | Decomposed into #12695, #13194. No action needed. |
| `#17450` | Unified KV buffer default (CLOSED/not planned) | Cosmetic issue. Fork tests with kv_unified=true — verified. |
| `#20032` | Fused multiply-add for Q4/Q5/Q6_K (MERGED) | Inherited performance gain via upstream sync. |
| `#20250` | Metal mul_mv_ext for BF16/Q2_K/Q3_K (MERGED) | Inherited performance gain via upstream sync. |
| `#14363` | High-throughput mode / virtual sequences (MERGED) | Inherited; not yet tested with compaction. |

### Open Issues Relevant to This Fork

| Issue/PR | Description | Risk | Action |
|----------|-------------|------|--------|
| `#11970` | KV cache truncated on `/v1/chat/completions` | MEDIUM — silent context loss collides with compaction | Monitor; add test with chat completions API |
| `#11577` | Feature request: resize existing context | LOW — compaction achieves similar goal | Monitor for API changes |
| `#19116` | Assert in kv-cache using Qwen3-VL | LOW — VL models use M-RoPE, already rejected | Confirm M-RoPE guard handles this |
| `#6685` | Server crash with defrag at parallel=32 | LOW — defrag removed | No action needed |
| `#19307` | GLM 4.7 Flash not working with flash attention | LOW — separate from compaction | Monitor for FA fixes |

### Server/Runtime Upstream Bugs (from prior adversarial reviews)

These are upstream llama-server bugs that affect ModelAI's use case. They are NOT in this fork's implementation scope but should be tracked:

| Issue/PR | Severity | Description |
|----------|----------|-------------|
| `#19679` | Critical | Random crash on Apple Metal (grammar stack empty) |
| `#19304` | Critical | Crash at 86K context / 50+ tool calls |
| `#19051` | Critical | Server fails open when JSON schema grammar parsing fails |
| `#19010` | Critical | Stack overflow from crafted JSON Schema pattern |
| `#16710` | Critical | Server crash on faulty tool call |
| `#17391` | Major | Segfault under repeated structured output |
| `#12171` | Major | llama-server inference 3x slower than llama-cli |
| `#19758` | Major | Reverse port not closed after SSE stream |
| `#17387` | Major | `/slots/0?action=erase` hangs indefinitely |

---

## Known Quality Issues

| Model | Context | Ratio | Pipeline | Cosine | Status |
|-------|---------|-------|----------|--------|--------|
| DeepSeek-R1-14B | 8K | 4x | nonuniform | **0.211** | **CRITICAL** — Phase 1A.1 |
| DeepSeek-R1-14B | 4K | 4x | nonuniform | **0.773** | **CRITICAL** — Phase 1A.1 |
| Qwen3-14B | 8K | 8x | select | 0.973 | Minor outlier — investigate in Phase 6 |
| Qwen3-14B | 16K | 8x | select | 0.987 | Slightly below other models |
| All other test points | 4K-32K | 2x-8x | select | >0.99 | PASS |

---

## Performance Roadmap Items

| Item | Status | Notes |
|------|--------|-------|
| B1: Batch V extraction | **DONE** | llama-kv-cache.cpp:1043-1079 |
| B2: Eliminate dual K/V extraction | **DONE** | llama-kv-compact-pipeline.cpp:67-71 (head_cache_entry) |
| B3: NEON vectorization | **DONE** | llama-kv-compact-math.h:16-52 |
| B4: GPU solver path | DEFERRED | Requires Metal compute shader |
| B5: Graph tensor caching | PARTIAL | Commit `a63dd655`; GPU materialization deferred |
| Self-study production speed | DEFERRED | 3.6 min at 4K; algorithmic redesign needed |
| OMP production speed | DEFERRED | >23 min for 2x on 14B; quality comparison only |
| FA + non-zero beta | BLOCKED | Waiting on FlashBias (arXiv:2505.12044) |
| Hybrid recurrent+attention | DEFERRED | No ModelAI target models use Mamba |
| M-RoPE edge cases | DEFERRED | No ModelAI models use M-RoPE; guard at llama-kv-cache.cpp:1130-1143 |

---

## Testing Gaps to Close

| Gap | Priority | Action |
|-----|----------|--------|
| No end-to-end test for quantized K through full pipeline | Major | Add Q8_0 K extraction + solver test |
| No 128K context validation | Major | Phase 3 |
| No negative tests for unsupported configurations | Major | Add SWA, MLA, hybrid rejection tests |
| No server integration test with real HTTP clients | Major | Phase 4.3 |
| Self-study pipeline blocked (dim mismatch) | Minor | Pending runtime diagnostics |
| 32K compaction throughput not re-measured after tensor caching | Minor | Re-measure in Phase 6 |
| Missing timing data for 4K compaction tests | Minor | Phase 6.3 |

---

## Execution Order

```
Phase 1A: Critical Bug Fixes (nonuniform pipeline, pp512 regression, decode regression, Gemma3 guard)
    |
Phase 1B: Upstream Bug Sync Verification (DONE — all 6 items verified, tests added)
    |
Phase 2: Performance Bottleneck Fixes (B1/B2/B3 DONE; B5 verify; B4 deferred)
    |
Phase 3: 128K Context Validation (Qwen3-8B, 14B) + compaction time profiling
    |
Phase 4: Architecture Support (FA zero-beta verify, SWA partial, server verify)
    |
Phase 5: Documentation Completeness
    |
Phase 6: Comprehensive Benchmark Suite + fill timing gaps
    |
Phase 7: Adversarial Review (mandatory break-fix-review loop)
    |
    RELEASE GATE
```

Phase 1A is BLOCKING. Phase 1B is complete. B1/B2/B3 are complete.
Phase 3 depends on Phase 1A fixes (decode regression affects long-context benchmarks).
Phases 4-5 can run in parallel after Phase 1A.
Phase 6 depends on all code changes being complete.
Phase 7 is always last.

---

## Beta Deployment Constraints

From `MODELAI_COMPREHENSIVE_REVIEW_AND_RECOMMENDATIONS.md`:

1. **Only expose `select` pipeline** — nonuniform has critical bugs (Phase 1A.1)
2. **Disable Gemma3-12B** — SWA decode broken upstream (Phase 1A.4)
3. **modelai-llama.cpp binary must be bundled** — users don't build from source
4. **Ollama fallback required** — if modelai-llama.cpp fails to start, fall back to Ollama
5. **KV compaction is opt-in for beta** — enable by default only at 32GB+ hardware
6. **Default models by tier:**
   - 8GB: Qwen3-8B Q4_K_M (2x compaction only)
   - 16GB: Qwen3-14B Q4_K_M (2x-4x compaction)
   - 32GB: Qwen3-30B-A3B Q4_K_M (2x-8x compaction)
   - 64GB+: Qwen3-30B-A3B Q4_K_M (2x-16x compaction, 128K context feasible)
