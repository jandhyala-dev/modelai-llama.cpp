# ModelAI llama.cpp — V1 Implementation Plan

**Base commit:** `5b6bf6eb` (modelai-main)
**Upstream base:** `0cd4f472` (upstream-master)
**Date:** 2026-03-14
**Owner:** Ajay Jandhyala — ajay@model-ai.app

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

### What V0 Did NOT Complete (Outstanding)

These items were planned or identified but not implemented:

---

## V1 Implementation Plan

### Phase 1: Upstream Bug Sync (Critical)

These are bugs/changes in upstream llama.cpp that directly affect compaction correctness and must be verified or synced.

#### 1.1 KV Cache Defrag Bug (PR #10873)

**Issue:** Defrag can corrupt KV data, directly affecting compaction reliability.
**Status:** PR #10873 was from the old ggerganov/llama.cpp repo. Upstream has since had a major defrag refactor:
- `#13988` — refactored defrag mechanism
- `#14081` — fixed shift and defrag logic
- `#14189` — fixed use-after-move
- `#15473` — removed KV cache defragmentation logic entirely

**Action:** Verify our fork correctly handles the defrag removal at `#15473`. Since we track upstream-master, this should be inherited. **Test:** Confirm compaction works correctly without defrag. Write a test that compacts, then verifies data integrity.

#### 1.2 KV Cells Unified Refactor (PR #11213, #12695)

**Issue:** Core data structure change — all KV cell iteration code must adapt.
**Status:** `#12695` is in our upstream-master (`a10b36c9`). `#11213` may have been superseded.
**Action:** Verify our compaction code iterates KV cells using the refactored API. Audit `src/llama-kv-cache.cpp` compaction methods for any stale cell iteration patterns.
**Risk:** HIGH — if our code uses old cell iteration, it may silently produce wrong results.

#### 1.3 SWA KV Cache Support (PR #13194)

**Issue:** Structural changes to KV allocation and eviction.
**Status:** `#13194` is in our upstream-master (`e298d2fb`). Our V0 explicitly rejects SWA (`n_swa > 0`).
**Action:** For V1, implement SWA-aware compaction (Phase 4). For now, verify the rejection logic still works correctly with the new SWA code.

#### 1.4 Unified KV Buffer Default (Issue #17450)

**Issue:** `kv_unified=true` is now default — could change KV layout assumptions.
**Status:** Related commits in upstream: `#16736`, `#18117`, `#18716`, `#19145`.
**Action:** Test compaction with unified KV enabled (it should be the default now). Verify `compacted_prefix_copy_k_head_f32` and `copy_v_head_f32` work correctly with unified layout.

#### 1.5 KV Cache Shift/Defrag Correctness (Issue #12253)

**Issue:** Prevents data loss during defrag/shift operations.
**Status:** Addressed by upstream refactor and defrag removal (`#15473`).
**Action:** Verify compacted prefix state survives shift operations. Write test: compact → shift → verify cosine similarity.

---

### Phase 2: Performance Bottleneck Fixes

From `docs/modelai-performance-roadmap.md` Part 1 — these are known internal bottlenecks.

#### 2.1 B1: Batch Transposed V Extraction

**Current:** `compacted_prefix_copy_v_head_f32` reads one scalar at a time across non-contiguous strides.
**Fix:** Read full V columns per element dimension in one `ggml_backend_tensor_get` call, then scatter to output buffer.
**Files:** `src/llama-kv-cache.cpp` (lines ~953-969)
**Impact:** Single biggest solver bottleneck. Should reduce compaction time by 30-50%.
**Test:** Existing V extraction tests must produce identical results.

#### 2.2 B2: Eliminate Dual K/V Extraction

**Current:** K/V extracted once during query scoring (Phase 1), again during fitting (Phase 2).
**Fix:** Extract K/V per-head in Phase 1, store in temporary buffers, pass to Phase 2.
**Files:** `src/llama-kv-compact-pipeline.cpp` (lines ~90-111, 131-189)
**Impact:** 2x unnecessary I/O eliminated.
**Test:** Pipeline timing should drop ~40-50%. Quality tests must still pass.

#### 2.3 B3: NEON Vectorization of Solver Loops

**Current:** All dot products, Cholesky decomposition, matrix multiply, and exp() are scalar loops.
**Fix:** Replace scalar `dot_row` with ARM NEON `vfmaq_f32` 4-wide FMA. Add `#ifdef __ARM_NEON__` guard.
**Files:** `src/llama-kv-compact-math.cpp` (dot_row used in solver, select, budget, pipeline)
**Impact:** 4-8x speedup on Apple Silicon for solver math.
**Scope:** Only vectorize `dot_row`. Leave Cholesky pivot logic and exp() as scalar.
**Test:** Solver unit tests must produce identical results within fp32 tolerance.

#### 2.4 B5: Graph Tensor Caching

**Current:** Compacted prefix K/V/beta/mask tensors created fresh each decode batch.
**Status:** PARTIALLY DONE (commit `a63dd655` added tensor caching). GPU-native mask/beta materialization still deferred.
**Action:** Verify caching is active and working. Profile to confirm no regression.

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

---

### Phase 4: Architecture Support Expansion

#### 4.1 Flash Attention Zero-Beta Fast Path

**Current:** V0 falls back to standard attention when solver beta is non-zero. Flash attention with zero-beta IS supported (PR-6).
**Action:** Verify zero-beta path works end-to-end. Test with `--flash-attn` flag.
**Test:** Run inference with flash_attn=true and zero-beta compaction. Compare logit cosine with non-flash path.
**Blocker for full FA:** FlashBias (arXiv:2505.12044) needed for non-zero beta in flash attention.

#### 4.2 SWA Architecture Support (Gemma3)

**Current:** `compacted_prefix_runtime_supported()` rejects SWA caches.
**Issue:** Gemma3-12B uses SWA — currently fails at model load level (iSWA format issue).
**Action:**
1. Fix iSWA model load issue for Gemma3
2. Implement SWA-aware compaction: only compact global-attention portion, leave SWA portion as-is
3. Test with Gemma3-12B at various compression ratios
**Reference:** Paper Section 4.2 discusses SWA handling.
**Effort:** HIGH — requires understanding iSWA base vs SWA layer split.

#### 4.3 Server Integration Verification

**Status:** `/compact` endpoint IS implemented (in PR-1 + post-6b sprint). Server-context.cpp has ~600 lines of ModelAI integration.
**Action:** Verify the endpoint works end-to-end with a real model. Test:
- `POST /compact` with valid slot and method
- Verify compaction completes and active_n_kv drops
- Verify subsequent inference uses compacted KV
- Verify `/props` reports correct compaction state
- Verify `/metrics` Prometheus gauges update

---

### Phase 5: Documentation Completeness

#### 5.1 Update Brief (`docs/modelai-llama-cpp-overview.md`)

Add missing sections:
1. **Modified upstream files** — document all 20 files changed and why
2. **Adversarial review findings** — detail the 6 bugs found and fixed across 2 review cycles:
   - Cycle 1: Chunked pipeline budget overshoot, prefill-Q p0 assertion, GQA divisibility assert
   - Cycle 2: Nonuniform NaN propagation, chunked recursive merge, budget allocator convergence
3. **Server integration** — document `/compact`, `/props`, `/metrics` endpoints
4. **Upstream bug sync status** — which upstream issues are addressed
5. **Relabel "Next Steps"** to "Engine Roadmap (V1+)" with clear categorization

#### 5.2 Update Fork Summary (`docs/modelai-fork-summary.md`)

- PR-5b section (lines 147-159): update to reflect DONE status with measured results
- Add PR-6 and PR-6b completion status
- Add post-6b sprint status
- Update V0 Support Matrix with current tested models and quality numbers

#### 5.3 Update Compaction Plan (`docs/modelai-kv-compaction-plan.md`)

- Mark all completed PRs as DONE with commit references
- Add V1 plan reference
- Document what was deferred and why

#### 5.4 Sync Review Standards

Ensure `docs/review-standards/hostile-review-protocol.md` has the "MANDATORY TESTING-REVIEW LOOP" section (added in prior session).

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
**Note:** Ollama excluded from comparison — it uses different measurement methodology and adds HTTP overhead. Document this decision.

#### 6.2 KV Compaction Quality Matrix

Run select pipeline across all models at multiple compression ratios:

| Model | 4K/2x | 4K/4x | 4K/8x | 8K/2x | 8K/4x | 8K/8x | 16K/2x | 16K/4x | 16K/8x |
|-------|-------|-------|-------|-------|-------|-------|--------|--------|--------|
| Qwen3-8B | | | | | | | | | |
| Qwen3-14B | | | | | | | | | |
| DeepSeek-R1-14B | | | | | | | | | |
| Qwen3-30B-A3B | | | | | | | | | |

#### 6.3 Server Endpoint Benchmark

- Measure `/compact` endpoint latency for each model at 4K/2x
- Measure decode throughput before and after compaction
- Measure active_n_kv reduction

---

### Phase 7: Adversarial Review

Full adversarial review per `docs/review-standards/hostile-review-protocol.md`:

1. **Code review** — all 20 modified upstream files + all new compaction files
2. **Test review** — verify all tests exercise adversarial inputs, boundary conditions, resource exhaustion
3. **Integration review** — verify server endpoints, serialization, sequence ops
4. **Upstream sync review** — verify no stale patterns from pre-refactor llama.cpp
5. **Break-fix-review loop** — mandatory: test → break → fix → commit → push → review again until PASS

---

## Upstream GitHub Issues Addressed by This Fork

| Issue/PR | Description | Status in Fork |
|----------|-------------|---------------|
| `#20037` | Attention Matching upstream RFC | Fork implements full pipeline; upstream tracking only |
| `#10873` | KV cache defrag corruption | Upstream removed defrag entirely (`#15473`); inherited via sync |
| `#12695` | KV cache guard refactor | In upstream-master (`a10b36c9`); verify compaction compatibility |
| `#13194` | SWA KV cache support | In upstream-master (`e298d2fb`); fork rejects SWA (V1 Phase 4) |
| `#12253` | KV shift/defrag correctness | Addressed by defrag removal; verify shift still works |
| `#20032` | Fused multiply-add for Q4/Q5/Q6_K | In upstream-master (`2afcdb97`); inherited performance gain |
| `#20250` | Metal mul_mv_ext for BF16/Q2_K/Q3_K | In upstream-master (`e22cd0aa`); inherited performance gain |
| `#14363` | High-throughput mode (virtual sequences) | In upstream-master (`225e7a14`); not yet tested with compaction |

## Upstream Issues NOT YET Addressed

| Issue/PR | Description | Risk | Action |
|----------|-------------|------|--------|
| `#11213` | `llama_kv_cells_unified` refactor | HIGH — may break KV cell iteration | Audit compaction code |
| `#17450` | Unified KV buffer default | MEDIUM — layout assumptions | Test with unified KV |
| `#14847` / `#15650` | Flash attention Metal stability | LOW — V0 uses non-flash | Monitor for V1 FA work |
| `#9551` | Vulkan KV quantization needs FA | LOW — Apple Silicon only target | N/A for current hardware |

## Performance Roadmap Items NOT Addressed

| Item | Description | Priority | Blocked By |
|------|-------------|----------|-----------|
| B4: GPU solver path | Forces CPU for all solver math | HIGH | Requires Metal compute shader for solver |
| Self-study production speed | 3.6 min at 4K context | MEDIUM | Algorithmic — needs orders-of-magnitude speedup |
| OMP production speed | >23 min for 2x on 14B | LOW | Known infeasible; quality comparison only |
| Flash attention + non-zero beta | Requires FlashBias | BLOCKED | Waiting on upstream FlashBias (arXiv:2505.12044) |
| Hybrid recurrent+attention | Mamba layers have no KV | DEFERRED | Architecture not in ModelAI target models |
| M-RoPE edge cases | Multi-position models | DEFERRED | No ModelAI models use M-RoPE |

## Execution Order

```
Phase 1: Upstream Bug Sync Verification
    |
Phase 2: Performance Bottleneck Fixes (B1, B2, B3)
    |
Phase 3: 128K Context Validation (Qwen3-8B, 14B)
    |
Phase 4: Architecture Support (FA zero-beta verify, SWA, server verify)
    |
Phase 5: Documentation Completeness
    |
Phase 6: Comprehensive Benchmark Suite
    |
Phase 7: Adversarial Review (mandatory break-fix-review loop)
    |
    RELEASE GATE
```

Phases 1-2 can partially overlap. Phase 3 depends on Phase 2 (perf fixes reduce compaction time for long contexts). Phases 4-5 can run in parallel. Phase 6 depends on all code changes being complete. Phase 7 is always last.
