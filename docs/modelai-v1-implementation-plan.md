# ModelAI llama.cpp — V1 Complete Implementation Plan

**Plan commit:** `(this commit)`
**Previous plan commit:** `57606905`
**Code baseline commit:** `3d5132b1` (Phase 1A fixes applied)
**Upstream base:** `0cd4f472` (upstream-master)
**Date:** 2026-03-14
**Owner:** Ajay Jandhyala — ajay@model-ai.app (COT Labs / ModelAI)

## Revision History

| Date | Version | Change |
|------|---------|--------|
| 2026-03-14 | v1 | Initial V1 plan: 7 phases, upstream sync + performance + 128K + architecture |
| 2026-03-14 | v2 | Incorporated Phase D findings, live benchmark critical bug (nonuniform cosine 0.211), 80+ findings from 44 review files. Updated B1/B2/B3 to DONE. Added Phase 1A. |
| 2026-03-14 | v3 | Code implementation: BUG-I01 FIXED (nonuniform fallback), BUG-I02 FIXED (B5 GPU-resident tensors), BUG-U01 MITIGATED (SWA warning), BUG-U02 investigated. |
| 2026-03-14 | v4 | **Complete rewrite.** All phases through Phase 7 with explicit tests for each code fix, benchmark test plan, out-of-scope delineation, risk register. Structured for adversarial plan review. |

---

## Executive Summary

### What V0 Delivered (Complete)

**158 files changed, 40,854 lines added** across 20 modified upstream files and ~138 new files.

- KV cache compaction via Attention Matching (arXiv:2602.16284)
- 7 pipelines: select, solver, nonuniform, chunked, on_policy, self_study, prefill_q
- Server integration: `/compact`, `/props`, `/metrics` endpoints
- 12 test files, 4 benchmark scripts, 9 server API tests
- 2 adversarial review cycles passed
- 3-way benchmark (modelai vs llama.cpp vs Ollama)

### What Phase 1A Fixed (Complete — commit `3d5132b1`)

- BUG-I01: Nonuniform pipeline fallback when >50% heads fully masked
- BUG-I02: GPU-resident compacted prefix tensors via `ggml_backend_tensor_set()`
- BUG-U01: SWA runtime warning in `compacted_prefix_runtime_supported()`
- BUG-U02: Investigation confirmed DeepSeek pp512 failure is NOT fork-caused

### What V1 Covers (Phases 3–7)

All remaining code changes, tests, benchmarks, documentation, and release gate work needed before beta deployment. **No code changes without explicit approval.**

### What Is Explicitly Out of Scope

See [Out of Scope](#explicitly-out-of-scope) section at end of document.

---

## Completed Phases (Reference Only)

### Phase 1A: Critical Bug Fixes — COMPLETE

| ID | Severity | Fix | File(s) | Commit |
|----|----------|-----|---------|--------|
| BUG-I01 | Critical | Nonuniform fallback when >50% heads fully masked | `src/llama-kv-compact-pipeline.cpp` | `3d5132b1` |
| BUG-I02 | Major | `ggml_backend_tensor_set()` replaces host pointer swap | `src/llama-kv-cache.cpp`, `src/llama-kv-compacted-prefix-exec.cpp` | `3d5132b1` |
| BUG-U01 | Major | SWA warning in `compacted_prefix_runtime_supported()` | `src/llama-kv-cache.cpp` | `3d5132b1` |
| BUG-U02 | Major | Investigation: not fork-caused | Documentation only | `3d5132b1` |

### Phase 1B: Upstream Bug Sync — COMPLETE

All 6 upstream issues verified safe/compatible. Integration tests added in `test-kv-compact-pipeline-integration.cpp`.

| Upstream PR/Issue | Status | Integration Test |
|-------------------|--------|-----------------|
| #10873 (defrag corruption) | SAFE — defrag removed | TEST 4 |
| #12695 (KV guard refactor) | COMPATIBLE — using current API | TEST 3, TEST 7 |
| #13194 (SWA KV cache) | COMPATIBLE — SWA correctly rejected | TEST 2 |
| #17450 (unified KV buffer) | COMPATIBLE — all tests use kv_unified=true | TEST 1 |
| #12253 (shift/defrag correctness) | SAFE — shift guard prevents interaction | TEST 5 |
| #11213 (KV cells unified) | N/A — decomposed into #12695, #13194 | covered above |

### Phase 2: Performance Bottlenecks B1/B2/B3 — COMPLETE

| Bottleneck | Status | Code Location |
|------------|--------|---------------|
| B1: Batch transposed V extraction | DONE | `llama-kv-cache.cpp:1043-1079` |
| B2: Eliminate dual K/V extraction | DONE | `llama-kv-compact-pipeline.cpp:67-71` |
| B3: NEON vectorization | DONE | `llama-kv-compact-math.h:16-52` |
| B4: GPU solver path | DEFERRED | Requires Metal compute shader — out of V1 scope |
| B5: GPU-resident tensor upload (K/V/beta) | DONE (Phase 1A) | `llama-kv-cache.cpp:2161-2267` |
| B5: GPU-native mask materialization | BY DESIGN: remains host-backed | Mask tensor is small (n_prefix × 1 float per head) and written directly to `dst->data`. GPU upload overhead would exceed any savings. `require_host_or_direct_data` intentionally retained for mask in `llama-kv-compacted-prefix-exec.cpp`. |

---

## V1 Remaining Work

### Phase 3: Test Suite for Phase 1A Code Fixes

**Goal:** Every code fix from Phase 1A gets explicit regression tests. Every identified testing gap gets closed. All tests must compile with `cmake -B build -DGGML_METAL=ON` and run via `ctest --test-dir build -L main --output-on-failure`.

---

#### 3.1 Tests for Fix 1: B5 GPU-Resident Tensor Upload (BUG-I02)

**What was fixed:** `set_input_compacted_prefix_k/v/kq_b` in `src/llama-kv-cache.cpp` replaced `dst->data = cache.data()` pointer swap with `ggml_backend_tensor_set()` for backend-agnostic upload.

**File to modify:** `tests/test-kv-compact-pipeline-integration.cpp`

**Test 3.1.1 — Compacted decode produces correct logits after B5 upload**

- Setup: Load model, fill KV to 128 tokens, compact with select pipeline at 2x
- Action: Decode 8 continuation tokens using compacted prefix
- Assert: All logits are finite (no NaN/inf from stale or uninitialized staging buffer)
- Assert: Logit cosine similarity between compacted and baseline >= 0.95
- Why: The B5 fix changed HOW data reaches the GPU. This test verifies the data is correct after upload via `ggml_backend_tensor_set()`.

**Test 3.1.2 — Cache version change triggers re-materialization**

- Setup: Load model, fill KV to 128 tokens, compact at 2x, decode 4 tokens
- Action: Re-compact at 4x (different target_tokens → version changes), decode 4 more tokens
- Assert: Second compaction produces different active_n_kv
- Assert: Continuation logits after re-compaction are finite
- Assert: The `cp_tensor_cache_t` version counter increments on each compaction
- Why: B5 uses a version-keyed cache. This test verifies that version changes trigger fresh staging buffer materialization rather than serving stale cached data.

**Test 3.1.3 — K/V/beta staging buffers have correct byte sizes**

- Setup: Load model, compact at 2x
- Assert: `cp_cache.k_bytes[ikv].size() == ggml_nbytes(k_tensor)` for each layer
- Assert: `cp_cache.v_bytes[ikv].size() == ggml_nbytes(v_tensor)` for each layer
- Assert: `cp_cache.kq_b_bytes[ikv].size() == ggml_nbytes(kq_b_tensor)` for each layer
- Why: `aligned_byte_buffer::resize()` does NOT zero-initialize (uses `posix_memalign`). If the byte count is wrong, `ggml_backend_tensor_set` would read past the buffer or write partial data.

---

#### 3.2 Tests for Fix 2: Nonuniform Pipeline Fallback (BUG-I01)

**What was fixed:** `src/llama-kv-compact-pipeline.cpp` added a guard after union truncation that falls back to `select` pipeline when >50% of KV heads are fully masked.

**File to modify:** `tests/test-kv-compact-pipeline-integration.cpp`

**Test 3.2.1 — Nonuniform pipeline produces valid output (no fallback case)**

- Setup: Load model, fill KV to 128 tokens
- Action: Run nonuniform pipeline at 2x compression (low disjointness → few masked heads)
- Assert: Pipeline completes without fallback warning
- Assert: Logit cosine >= 0.90
- Why: Verifies nonuniform pipeline still works when head disjointness is low (the common case).

**Test 3.2.2 — Nonuniform pipeline falls back to select at high compression**

- Setup: Load model, fill KV to 128 tokens
- Action: Run nonuniform pipeline at 8x compression (high disjointness → many masked heads)
- Assert: If fallback triggers, logit cosine >= 0.90 (select pipeline quality)
- Assert: If fallback does NOT trigger, logit cosine >= 0.85 (8x is aggressive)
- Why: At high compression, union truncation masks many heads. This test verifies the fallback produces acceptable quality rather than cosine ~0.2.

**Test 3.2.3 — Fallback threshold boundary: exactly 50% masked**

- Setup: This is a unit-level trace, not a model test. Construct a scenario where `n_fully_masked * 2 == total_kv_heads` (exactly 50%).
- Assert: Fallback does NOT trigger (guard uses strict `>`, not `>=`)
- Why: The guard condition is `n_fully_masked * 2 > total_kv_heads`. At exactly 50%, the guard should NOT fire. This boundary must be correct.
- Implementation note: This test may need to be a code-level assertion rather than a model-driven test, since controlling exact head masking from model input is non-deterministic. Consider adding a dedicated unit test function that directly tests the fallback logic with mock per_head_mask data.

---

#### 3.3 Tests for Fix 3: SWA Warning (BUG-U01)

**What was fixed:** `src/llama-kv-cache.cpp` added `LLAMA_LOG_WARN` in `compacted_prefix_runtime_supported()` when `n_swa > 0 || swa_type != LLAMA_SWA_TYPE_NONE`.

**File to modify:** `tests/test-kv-compact-pipeline-integration.cpp`

**Test 3.3.1 — SWA cache correctly rejected with return value**

- Setup: This test requires either a model with SWA layers or a synthetic KV cache with `n_swa > 0`
- Assert: `compacted_prefix_runtime_supported()` returns `false` for SWA sub-cache
- Assert: `compacted_prefix_runtime_supported()` returns `true` for the base (non-SWA) cache of the same iSWA model
- Why: Verifies the fix correctly rejects only the SWA sub-cache while allowing the base cache.
- Implementation note: If no SWA model is available for automated tests (stories15M is not SWA), this test should be marked as requiring the `model-swa` label and skipped in CI until a suitable model fixture is available. Document the manual verification procedure.

**Test 3.3.2 — Warning fires once only (static guard)**

- Assert: The `static bool warned_swa` guard prevents duplicate warnings
- Implementation note: This is difficult to test in isolation because `static` state persists across test cases in the same process. Verify by code review + manual testing. Do NOT add a test that depends on log output parsing — that is fragile.
- Why: Thread-safe static initialization (C++11 guarantee) means the flag is safe, but verifying it fires exactly once across multiple calls requires log capture which is not reliable in ctest.

---

#### 3.4 Negative Configuration Rejection Tests

**What to test:** `compacted_prefix_runtime_supported()` correctly rejects all unsupported configurations listed in the V0 support matrix.

**File to modify:** `tests/test-kv-compact-pipeline-integration.cpp`

**Test 3.4.1 — M-RoPE / MLA model rejected (head dimension mismatch)**

- Setup: Models using M-RoPE (e.g., Qwen3-VL) or MLA (e.g., DeepSeek-V2/V3) must be rejected by the head dimension guard at `llama-kv-cache.cpp:1130-1143`
- Assert: `compacted_prefix_runtime_supported()` returns `false`
- Why: Both M-RoPE and MLA architectures result in `n_embd_head_v != n_embd_head_k`, which the guard checks. MLA uses latent key/value dimensions that differ from query dimensions.
- Implementation note: Requires a model with `n_embd_head_v != n_embd_head_k` or M-RoPE flag. If no such model fixture is available, document as manual test with specific model name (Qwen3-VL for M-RoPE, DeepSeek-V2 for MLA).

**Test 3.4.2 — Flash attention with non-zero beta rejected**

- Setup: Enable `flash_attn = true` in context params, run compaction that produces beta != 0
- Assert: The non-flash attention path is taken (or compaction is rejected if beta > 0 under flash attention)
- Why: V0 only supports flash attention with zero-beta. Non-zero beta requires FlashBias (arXiv:2505.12044) which is not implemented.
- Implementation note: The rejection happens in the graph build path (`llama-graph.cpp`), not in `compacted_prefix_runtime_supported()`. Trace the exact code path and verify the fallback behavior.

**Test 3.4.3 — Hybrid recurrent+attention model rejected**

- Setup: Models using hybrid recurrent+attention architectures (e.g., Mamba, RWKV, Jamba) must be rejected
- Assert: `compacted_prefix_runtime_supported()` returns `false`
- Why: Hybrid models interleave attention layers with recurrent (SSM) layers. The compaction algorithm assumes all layers use KV-cached attention. Compacting only the attention layers while leaving recurrent state unchanged would produce incorrect output.
- Implementation note: No ModelAI target models use hybrid architectures. If no model fixture is available, verify rejection by code review of the guard condition and document as manual test. The guard checks for recurrent layer presence in model architecture metadata.

---

#### 3.5 Q8_0 K Extraction Pipeline Test

**What to test:** K extraction from Q8_0 quantized KV cache produces correct values through the full compaction pipeline.

**File to modify:** `tests/test-kv-compact-pipeline-integration.cpp`

**Test 3.5.1 — Q8_0 K values are finite and correct**

- Setup: Load model with Q8_0 type_k (set `type_k = GGML_TYPE_Q8_0` in context params)
- Action: Fill KV cache, extract K values via compaction pipeline
- Assert: All extracted K values are finite (not NaN/inf)
- Assert: K tensor shapes match expected `[n_embd_head_k, n_prefix_tokens, n_head_kv]`
- Assert: Logit cosine with baseline >= 0.90
- Why: K extraction uses `ggml_backend_tensor_get` + dequantization. Q8_0 has different block size and dequantization logic than F16/F32. This has never been tested through the full pipeline.
- Implementation note: stories15M model may not support Q8_0 KV. Verify model compatibility before implementing. If incompatible, this test requires a different model fixture.

---

#### 3.6 Test Registration and Build

**File to modify:** `tests/CMakeLists.txt`

All new tests in Phase 3 should be added to existing test files (primarily `test-kv-compact-pipeline-integration.cpp`) to avoid test binary proliferation. New test cases should be guarded by test number constants and documented in the test file header.

**Build verification:**
```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
ctest --test-dir build -L main --output-on-failure
```

**Expected result:** All tests pass (46/47 + new tests, minus 1 pre-existing upstream tokenizer failure).

---

### Phase 4: Outstanding Bug Fixes

**Goal:** Fix BUG-I03 (timing data), investigate BUG-I04 (O(n²) scaling), and investigate BUG-I05 (quality dip). Each fix includes a verification test.

---

#### 4.1 BUG-I03: Missing Compaction Timing Data at Short Contexts

**Bug:** `compaction_time_ms`, `baseline_decode_tok_s`, `compacted_decode_tok_s` are null for Qwen3-30B-A3B at 4K (all ratios) and 8K (4x, 8x).

**Root cause hypothesis:** Timing instrumentation is conditional on context length or token count exceeding a threshold. At short contexts, the condition is not met.

**File to investigate:** `tests/test-kv-compact-workload.cpp`, `tests/test-kv-compact-longctx.cpp`

**Fix approach:**
1. Trace the timing instrumentation code path for a 4K context
2. Identify the condition that gates timing data collection
3. Fix the condition so timing is collected at all context lengths
4. If the issue is that compaction completes too fast for millisecond resolution, switch to microsecond resolution

**Verification test:**
- Run `test-kv-compact-workload` at 4K context with Qwen3-30B-A3B
- Assert: `compaction_time_ms` is not null and > 0
- Assert: `baseline_decode_tok_s` and `compacted_decode_tok_s` are not null and > 0

---

#### 4.2 BUG-I04: Compaction Time Scales Super-Linearly

**Bug:** 8K: 518ms → 16K: 1093-1312ms (~2.5x) → 32K: 2633-3663ms (~3x). This is O(n²) scaling. At 64K+, compaction could reach 10-15 seconds.

**Root cause (known):** Attention score computation in the select pipeline is O(n_prefix × n_queries × n_heads × n_layers). With n_queries proportional to context length, this becomes O(n²).

**File to investigate:** `src/llama-kv-compact-pipeline.cpp`, `src/llama-kv-compact-query.cpp`

**Fix approach (profiling only in V1 — optimization deferred):**
1. Add high-resolution timing instrumentation to each pipeline stage:
   - K extraction time
   - Query generation time
   - Attention score computation time
   - Selection/sorting time
   - Beta solving time (if applicable)
   - V extraction time
   - KV cache write time
2. Run instrumented pipeline at 4K, 8K, 16K, 32K
3. Identify which stage dominates the O(n²) scaling
4. Document the bottleneck for Phase 6 optimization planning

**Files to modify:**
- `src/llama-kv-compact-pipeline.cpp` — add timing around each stage
- `include/llama.h` or `src/llama-kv-compact-pipeline.h` — add timing fields to `llama_kv_compact_stats`

**Verification:**
- Run instrumented pipeline at 4K and 32K
- Assert: per-stage timing data is populated (not zero)
- Assert: sum of per-stage times ≈ total compaction time (within 10%)
- Document: which stage accounts for >50% of the O(n²) growth

**Output:** Profiling results documented in bug tracker with specific bottleneck identification and recommended optimization approach for future phase.

---

#### 4.3 BUG-I05: Qwen3-14B Quality Dip at 8K/8x

**Bug:** Qwen3-14B at 8K/8x select pipeline: logit cosine = 0.973 (lowest of 33 test points). All other points > 0.99. Also 16K/8x: 0.987.

**Severity:** Minor — above 0.95 threshold.

**Investigation approach (model-dependent — requires Qwen3-14B):**
1. Run additional test points: 8K/6x, 8K/10x, 12K/8x to characterize the quality curve
2. Compare attention patterns at 8K for Qwen3-14B vs Qwen3-8B (which doesn't show the dip)
3. Check if the dip is specific to Q4_K_M quantization or also appears at higher quant levels
4. If the dip is consistent: document as a model-specific characteristic, not a bug
5. If the dip is inconsistent: investigate compaction algorithm behavior at this context/ratio combination

**Files:** No code changes expected. Documentation update to bug tracker.

**Verification:**
- Run Qwen3-14B at 8K with ratios 4x, 6x, 8x, 10x
- Plot quality curve to determine if 8x is an outlier or part of a smooth degradation
- Document results in `docs/modelai-v1-bug-tracker.md`

---

### Phase 5: Server Integration & Architecture Verification

**Goal:** Verify server endpoints, flash attention zero-beta path, and B5 decode throughput improvement. Produce end-to-end integration tests for the server API.

---

#### 5.1 Server Integration End-to-End Tests

**Existing tests:** `tools/server/tests/unit/test_compact.py` has 9 test cases covering `/compact` request validation, error handling, and basic state.

**Missing coverage:**
1. Full prompt → compact → continue workflow
2. `/metrics` Prometheus gauge validation after compaction
3. Compacted inference quality verification through HTTP API
4. Multi-slot compaction behavior

**File to modify:** `tools/server/tests/unit/test_compact.py`

**Test 5.1.1 — Prompt, compact, and continue workflow**

- Setup: Start llama-server with model, POST `/v1/completions` with prompt (128+ tokens)
- Action: POST `/compact` with method=select, ratio=2
- Assert: Response includes `active_n_kv < original_n_kv`
- Action: POST `/v1/completions` with continuation prompt
- Assert: Completion returns valid text (not empty, not garbage)
- Assert: No server error/crash

**Test 5.1.2 — Prometheus metrics update after compaction**

- Setup: Start llama-server with `--metrics`
- Action: POST `/compact`, then GET `/metrics`
- Assert: `llamacpp_kv_compaction_active_n_kv` gauge is present and > 0
- Assert: `llamacpp_kv_compaction_ratio` gauge reflects requested ratio
- Assert: `llamacpp_kv_compaction_time_ms` histogram is populated

**Test 5.1.3 — Multiple sequential compactions**

- Setup: Start llama-server, fill KV via prompt
- Action: POST `/compact` at 2x, then POST `/compact` at 4x
- Assert: second compaction succeeds
- Assert: active_n_kv after 4x < active_n_kv after 2x

**Build and run:**
```bash
cd tools/server/tests && python -m pytest unit/test_compact.py -v
```

---

#### 5.2 Chat Completions API Test (GitHub #11970)

**Issue:** KV cache sometimes truncated incorrectly during `/v1/chat/completions`. Could cause compaction to compact an already-truncated cache.

**File to create:** `tools/server/tests/unit/test_compact_chat.py`

**Test 5.2.1 — Compact after chat completions**

- Setup: Start llama-server with chat template
- Action: POST `/v1/chat/completions` with multi-turn conversation (>128 tokens total)
- Action: POST `/compact` with method=select, ratio=2
- Assert: Compaction succeeds (200 response, no crash)
- Assert: active_n_kv is reasonable (> 0, < original)

**Test 5.2.2 — Continue chat after compaction**

- Action: POST `/v1/chat/completions` with next user message after compaction
- Assert: Response is valid (not empty, grammatically coherent)
- Assert: No server crash or assertion failure

---

#### 5.3 Flash Attention Zero-Beta Verification

**Current state:** V0 supports flash attention with zero-beta (PR-6). Non-zero beta under flash attention is NOT supported (requires FlashBias).

**Verification approach:**
1. Build with `GGML_METAL=ON` (flash attention available on Metal)
2. Run inference with `--flash-attn` flag enabled
3. If solver produces beta = 0 for all heads, flash attention path is taken
4. Compare logit cosine between flash and non-flash paths

**File to modify:** `tests/test-kv-compact-pipeline-integration.cpp` (or `test-kv-compact-quality.cpp`)

**Test 5.3.1 — Flash attention with select pipeline (zero-beta)**

- Setup: Load model with `flash_attn = true`, fill KV to 128 tokens
- Action: Run select pipeline at 2x (select pipeline uses zero beta by default)
- Assert: Inference completes without error
- Assert: Logit cosine >= 0.90 (may differ slightly from non-flash due to different attention kernel)
- Implementation note: If the model fixture (stories15M) does not support flash attention on Metal, document as requiring manual verification with a production model. Record the specific model and result.

---

#### 5.4 B5 Decode Throughput Verification

**Goal:** Confirm the B5 fix (GPU-resident tensors) improved decode throughput at 32K context. This is a quantitative verification of the BUG-I02 fix.

**Approach:**
1. Build at commit BEFORE B5 fix (`df895bed`) — measure decode tok/s at 32K with compaction
2. Build at commit AFTER B5 fix (`3d5132b1`) — measure decode tok/s at 32K with compaction
3. Compare

**Model:** Qwen3-30B-A3B at 32K context, 2x compression
**Metric:** `compacted_decode_tok_s`
**Expected:** Before: ~2.0 tok/s. After: >= 5.0 tok/s (approaching baseline of 6.8 tok/s).

**Script:**
```bash
# Before B5
git checkout df895bed
cmake -B build-before -DGGML_METAL=ON && cmake --build build-before -j$(sysctl -n hw.ncpu)
PIPELINE=select RATIO=2 N_CTX=32768 ./scripts/bench-kv-compact-workload.sh /path/to/Qwen3-30B-A3B-Q4_K_M.gguf

# After B5
git checkout 3d5132b1
cmake -B build-after -DGGML_METAL=ON && cmake --build build-after -j$(sysctl -n hw.ncpu)
PIPELINE=select RATIO=2 N_CTX=32768 ./scripts/bench-kv-compact-workload.sh /path/to/Qwen3-30B-A3B-Q4_K_M.gguf
```

**Verification criteria:**
- After/Before ratio > 2.0x improvement
- After throughput > 4.0 tok/s (at minimum, no longer 3.4x regression)

---

### Phase 6: Documentation Updates

**Goal:** All documentation reflects current state. No stale information. No claims about features that don't work.

---

#### 6.1 Update Overview (`docs/modelai-llama-cpp-overview.md`)

| Section | Change |
|---------|--------|
| Modified upstream files | Document all 20 files with change summary (use table from Phase 1A reference) |
| Adversarial review findings | 6 bugs found and fixed across 2 review cycles, plus Phase 1A fixes |
| Server integration | `/compact`, `/props`, `/metrics` endpoints with example curl commands |
| Upstream bug sync | 6 verified items from Phase 1B |
| Next Steps | Relabel to "Engine Roadmap (V1+)" — remove items that are complete |

#### 6.2 Update Fork Summary (`docs/modelai-fork-summary.md`)

| Section | Change |
|---------|--------|
| PR status | All PRs through PR-6 marked DONE with commit references |
| V0 Support Matrix | Add tested models with quality numbers from benchmarks |
| Unsupported models | Gemma3-12B listed as UNSUPPORTED (SWA decode broken upstream) |
| Known limitations | B4 GPU solver deferred, flash attention beta>0 blocked |

#### 6.3 Update Compaction Plan (`docs/modelai-kv-compaction-plan.md`)

| Section | Change |
|---------|--------|
| PR status | All completed PRs marked DONE |
| V1 plan reference | Link to this document |
| Deferred items | Document with reasons: B4, FlashBias, SWA, paper gaps |

#### 6.4 Update Performance Roadmap (`docs/modelai-performance-roadmap.md`)

| Section | Change |
|---------|--------|
| B1/B2/B3 | Update status from OPEN to DONE |
| B5 | Update status to DONE with B5 fix commit reference |
| B4 | Confirm DEFERRED with justification |
| P5b steps | Mark completed steps |

#### 6.5 Update Bug Tracker (`docs/modelai-v1-bug-tracker.md`)

- BUG-I03: Update with timing fix results from Phase 4.1
- BUG-I04: Update with profiling results from Phase 4.2
- BUG-I05: Update with investigation results from Phase 4.3
- All "FIXED" entries: Verify code line references are accurate for current commit

---

### Phase 7: Comprehensive Benchmark Suite

**Goal:** Produce complete quality, performance, and timing data for all supported models. Fill all gaps identified in Phase D testing. Results committed to `bench-results/`.

---

#### 7.1 Quality Matrix — Select Pipeline

Run select pipeline across all 4 models at all context/ratio combinations. 3 repetitions each.

**Hardware:** Apple Silicon (M2 Pro 32GB or equivalent)
**Build:** `cmake -B build -DGGML_METAL=ON && cmake --build build --config Release`
**Script:** `scripts/bench-kv-compact-longctx.sh`

| Model | Context Sizes | Compression Ratios |
|-------|--------------|-------------------|
| Qwen3-8B Q4_K_M | 4K, 8K, 16K, 32K | 2x, 4x, 8x |
| Qwen3-14B Q4_K_M | 4K, 8K, 16K, 32K | 2x, 4x, 8x |
| DeepSeek-R1-14B Q4_K_M | 4K, 8K, 16K | 2x, 4x, 8x |
| Qwen3-30B-A3B Q4_K_M | 4K, 8K, 16K, 32K | 2x, 4x, 8x |

**Total test points:** 4 models × 3–4 contexts × 3 ratios × 3 reps = **108–144 runs**

**Quality thresholds (per `docs/modelai-ci-policy.md`):**
- 2x compression: logit cosine >= 0.99
- 4x compression: logit cosine >= 0.95
- 8x compression: logit cosine >= 0.90

**Metrics per run:**
1. `logit_cosine` — primary quality metric
2. `compaction_time_ms` — must not be null
3. `baseline_decode_tok_s` — must not be null
4. `compacted_decode_tok_s` — must not be null
5. `active_n_kv` — verify matches target
6. `solver_time_ms` — if solver pipeline
7. `query_gen_time_ms` — if applicable

**Pass criteria:** ALL test points meet quality threshold for their compression ratio. No null timing data.

**Failure handling:** If any test point fails quality threshold:
1. Investigate root cause
2. If model-specific: document in bug tracker, add to known quality issues
3. If algorithmic: create BUG-I0x entry, fix in Phase 4, re-benchmark

---

#### 7.2 128K Context Validation

**Model:** Qwen3-8B Q4_K_M (5.2GB — fits in 32GB with 128K context)

| Context | Ratios | Notes |
|---------|--------|-------|
| 32K | 2x, 4x, 8x | Baseline comparison with other models |
| 64K | 2x, 4x, 8x | May need chunked pipeline if context exceeds single compaction limit |
| 128K | 4x, 8x, 16x | Document memory requirements at each ratio |

**Additional 128K metrics:**
- Peak memory usage (via `ggml_backend_buffer_get_size`)
- Compaction time (expected: 10-15s based on O(n²) scaling)
- Whether chunked pipeline is needed

**Quality thresholds:**
- 64K at 2x: >= 0.95
- 64K at 4x: >= 0.90
- 128K at 4x: >= 0.90
- 128K at 8x: >= 0.85
- 128K at 16x: >= 0.80

---

#### 7.3 Timing Data Fill (BUG-I03)

After Phase 4.1 fix, re-run all test points that previously had null timing data:

| Model | Context | Ratios |
|-------|---------|--------|
| Qwen3-30B-A3B | 4K | 2x, 4x, 8x |
| Qwen3-30B-A3B | 8K | 4x, 8x |

**Assert:** All timing fields populated, all > 0.

---

#### 7.4 Server Endpoint Benchmarks

**Model:** Qwen3-14B Q4_K_M

| Test | Metric | Method |
|------|--------|--------|
| `/compact` latency at 4K/2x | Time from request to response (ms) | POST `/compact`, measure wall time |
| `/compact` latency at 16K/2x | Time from request to response (ms) | Same |
| Decode throughput before compaction | tok/s | POST `/v1/completions`, measure throughput |
| Decode throughput after compaction | tok/s | POST `/compact` then `/v1/completions` |
| active_n_kv reduction | Ratio | `original_kv / active_n_kv` |

**3 repetitions each.** Report mean and std dev.

---

#### 7.5 CI Workload Coverage (W4–W6)

Per `docs/modelai-ci-policy.md`, workloads W4-W6 have not been measured:

| Workload | Description | Test Plan |
|----------|-------------|-----------|
| W4 | Multi-turn conversation (3+ turns, 4K total) | Server API: 3 chat completions turns, compact between turns, verify quality |
| W5 | Concurrent slots (parallel=2) | Server API: 2 parallel completions, compact slot 0, verify slot 1 unaffected |
| W6 | Long-running session (>50 decode batches) | CLI: prompt 4K tokens, decode 500 tokens, compact, decode 500 more |

**Pass criteria:** No crashes, no quality degradation below threshold, correct state isolation between slots.

---

#### 7.6 B5 Decode Throughput Before/After

Execute the before/after comparison described in Phase 5.4. Record results in benchmark artifacts.

**Output format:** CSV row with columns:
`commit,model,context,ratio,pipeline,baseline_decode_tok_s,compacted_decode_tok_s,improvement_ratio`

---

### Adversarial Review & Release Gate

**Goal:** Full adversarial review per `docs/review-standards/hostile-review-protocol.md`. Release checklist execution per `docs/modelai-release-checklist.md`. CI policy validation per `docs/modelai-ci-policy.md`.

---

#### Adversarial Review Loop

Per the hostile review protocol's mandatory testing-review loop:

1. **Tests must attempt to break the code.** All Phase 3 tests must include boundary inputs and adversarial cases (not just happy paths).
2. **Every bug found MUST be fixed, committed, and pushed** before the review cycle concludes.
3. **After fixes, a full adversarial review MUST run again** using the hostile review protocol.
4. **The cycle repeats** until the review returns PASS with zero Critical or Major findings.

**Review scope:**
- All 20 modified upstream files
- All new compaction files (~138 files)
- All Phase 3 test additions
- All Phase 4 bug fixes
- Server integration (server-context.cpp, test_compact.py)
- Serialization (state save/restore)

**Mandatory traces (from hostile review protocol):**
- Production trace: Qwen3-14B at 16K/2x select pipeline, full callback sequence
- Boundary trace: 1 token context, maximum context, ratio=1
- Adversarial trace: unusual GQA ratio (n_head_kv=1), partial restore
- Integer arithmetic trace: every division/modulo/stride with substituted values
- State machine trace: compaction state before/after/failure/rollback

---

#### Release Checklist Execution

Per `docs/modelai-release-checklist.md`, all 24 items must be checked:

**Pre-Release (8 items):**
- [ ] All Phase 7 benchmark results committed
- [ ] All tests pass (ctest -L main)
- [ ] Server tests pass (pytest test_compact.py)
- [ ] Bug tracker has no Critical or Major OPEN items
- [ ] Documentation is current (Phase 6 complete)
- [ ] Adversarial review PASS verdict
- [ ] Release notes drafted
- [ ] Binary built and tested

**Release (8 items):**
- [ ] Tag created
- [ ] Binary archived
- [ ] Release notes published
- [ ] Supabase model metadata updated
- [ ] ModelAI server integration tested
- [ ] Rollback procedure documented
- [ ] Monitoring alerts configured
- [ ] Stakeholders notified

**Post-Release (8 items):**
- [ ] Smoke test on production hardware
- [ ] Benchmark results match pre-release
- [ ] Error rate baseline established
- [ ] Metrics archival confirmed
- [ ] Upstream sync planned
- [ ] Next milestone scoped
- [ ] Bug tracker updated
- [ ] Release retrospective scheduled

---

#### CI Policy Validation

Per `docs/modelai-ci-policy.md`:

| Gate | Requirement | Verification |
|------|-------------|-------------|
| Build | Compiles clean on Apple Silicon | `cmake -B build -DGGML_METAL=ON && cmake --build build` |
| Core tests | 46/47 pass (1 known upstream failure) | `ctest --test-dir build -L main` |
| Server tests | All test_compact.py pass | `cd tools/server/tests && python -m pytest unit/test_compact.py -v` |
| Quality regression | No test point drops >1% below previous measurement | Compare Phase 7 results with Phase D results |
| Performance regression | No test point drops >10% in tok/s | Compare with baseline measurements |
| PR-5b closure | All 7 conditions met | Verify against `docs/modelai-ci-policy.md` thresholds |

---

## Explicitly Out of Scope

These items are NOT part of V1. Each has a documented reason for deferral.

| Item | Reason | Future Phase |
|------|--------|-------------|
| B4: GPU solver path | Requires Metal compute shader development — significant new capability | PR-7 |
| Upstream contribution (Track B) | Upstream has no implementation PR for #20037. Premature to contribute before stabilizing fork. | Post-V1 |
| Paper gaps for 50x compression (6 items) | Ridge scaling, on-policy sequential, chunked compaction refinement, etc. — research-grade work | PR-6a/6b/6c |
| Product optimizations P1-P6 | Prefix caching, KV quantization, autotuning, thread scheduling — product features | PR-7/PR-8 |
| Flash attention with non-zero beta | Blocked on FlashBias (arXiv:2505.12044) — external dependency | PR-6+ |
| SWA architecture full support | Blocked on upstream fixing Gemma3-12B decode (0.7 tok/s vs 16.5 Ollama) | Post-upstream-fix |
| Hybrid recurrent+attention | No ModelAI target models use Mamba/RWKV. Guard exists. | Not planned |
| M-RoPE edge cases | No ModelAI models use M-RoPE. Guard exists at `llama-kv-cache.cpp:1130-1143`. | Not planned |
| Self-study production speed | 3.6 min at 4K — needs algorithmic redesign, not incremental fix | PR-7 |
| OMP production speed | >23 min for 2x on 14B — quality-comparison-only pipeline | Not planned |
| CUDA backend testing | No CUDA hardware available for V1 testing | Phase 3+ of CI policy |

---

## Execution Order & Dependencies

```
Phase 1A: Critical Bug Fixes ──────────────────── COMPLETE
    │
Phase 1B: Upstream Bug Sync ───────────────────── COMPLETE
    │
Phase 2: Performance (B1/B2/B3) ───────────────── COMPLETE
    │
    ├── Phase 3: Test Suite (code changes) ──────── Tests for 1A fixes + gaps
    │       │
    │       └── Phase 4: Bug Fixes (code changes) ── BUG-I03, I04, I05
    │               │
    ├── Phase 5: Server & Architecture ──────────── E2E tests, FA verify, B5 verify
    │       │
    │       │      (Phases 3-5 produce all code changes)
    │       │
    ├── Phase 6: Documentation ──────────────────── Docs only, no code
    │       │
    │       └── Phase 7: Benchmarks ─────────────── Requires all code complete
    │               │
    │               └── Release Gate ────────────── Adversarial review + checklist
    │                       │
    │                      RELEASE
```

**Critical path:** Phase 3 → Phase 4 → Phase 7 → Release Gate

**Parallelizable:**
- Phase 6 (docs) can run in parallel with Phases 3-5
- Phase 5 (server/architecture) can run in parallel with Phase 4 (bug fixes) if they touch different files
- Phase 4.3 (BUG-I05 investigation) requires Qwen3-14B model — can be done independently

**Blockers:**
- Phase 7 benchmarks require ALL code changes to be complete (Phases 3-5)
- Release gate requires Phase 7 benchmarks to be complete
- Phase 5.4 (B5 throughput) requires Qwen3-30B-A3B model and checkout of old commit
- 128K validation (Phase 7.2) may require extended test time (>1 hour per model)

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Q8_0 KV not supported by stories15M fixture | Medium | Phase 3.5 test cannot run in CI | Test with production model manually; document result |
| Flash attention not available on stories15M | Medium | Phase 5.3 test cannot run in CI | Test with production model manually; document result |
| 128K OOM on 32GB M2 Pro | Low | Phase 7.2 incomplete | Use Qwen3-8B (5.2GB) which leaves 26GB for KV; reduce batch size |
| BUG-I04 profiling reveals no single dominant bottleneck | Medium | No clear optimization target | Document all stage timings; defer optimization to PR-7 |
| BUG-I05 investigation finds algorithmic issue at 8x | Low | Quality threshold may need adjustment | Document as model-specific; investigate if >1 model affected |
| Adversarial review finds Critical bug in V0 code | Medium | Delays release for break-fix cycle | Budget 2-3 review cycles in timeline; fix immediately |
| Upstream llama.cpp release breaks fork | Low | Requires emergency upstream sync | upstream-master branch isolates risk; test before merge |
| SWA warning static guard not thread-safe on non-C++11 compilers | Very Low | Warning may fire multiple times | C++11 is required by llama.cpp; document compiler requirement |

---

## Known Quality Issues

| Model | Context | Ratio | Pipeline | Cosine | Status |
|-------|---------|-------|----------|--------|--------|
| DeepSeek-R1-14B | 8K | 4x | nonuniform | 0.211 | **FIXED** — Phase 1A.1 fallback to select |
| DeepSeek-R1-14B | 4K | 4x | nonuniform | 0.773 | **FIXED** — Phase 1A.1 fallback to select |
| Qwen3-14B | 8K | 8x | select | 0.973 | Minor — investigate Phase 4.3 |
| Qwen3-14B | 16K | 8x | select | 0.987 | Minor — slightly below other models |
| All other test points | 4K-32K | 2x-8x | select | >0.99 | PASS |

---

## Upstream Issue Tracker

### Addressed by This Fork

| Issue/PR | Description | Status in Fork |
|----------|-------------|---------------|
| `#20037` | Attention Matching upstream RFC | Fork implements full pipeline |
| `#10873` | KV cache defrag corruption | VERIFIED SAFE |
| `#12695` | KV guard refactor | VERIFIED COMPATIBLE |
| `#13194` | SWA KV cache support | VERIFIED COMPATIBLE |
| `#12253` | Shift/defrag correctness | VERIFIED SAFE |
| `#11213` | KV cells unified | N/A — decomposed |
| `#17450` | Unified KV buffer | VERIFIED COMPATIBLE |

### Open Issues Relevant to Fork

| Issue/PR | Risk | Action |
|----------|------|--------|
| `#11970` | KV cache truncated on chat completions | Phase 5.2 test |
| `#19116` | Assert in kv-cache using Qwen3-VL | Confirm M-RoPE guard |
| `#11577` | Resize existing context | Monitor |

### Upstream Server Bugs (Tracked, Not Fork Scope)

| Issue/PR | Severity | Description |
|----------|----------|-------------|
| `#19679` | Critical | Random crash on Apple Metal (grammar stack) |
| `#19304` | Critical | Crash at 86K context / 50+ tool calls |
| `#19051` | Critical | Server fails open on JSON schema grammar parse failure |
| `#19010` | Critical | Stack overflow from crafted JSON Schema |
| `#16710` | Critical | Server crash on faulty tool call |

---

## Beta Deployment Constraints

1. **Only expose `select` pipeline** — nonuniform fallback is a safety net, not a production path
2. **Disable Gemma3-12B** — SWA decode broken upstream
3. **modelai-llama.cpp binary must be bundled** — users don't build from source
4. **Ollama fallback required** — if modelai-llama.cpp fails to start, fall back to Ollama
5. **KV compaction is opt-in for beta** — enable by default only at 32GB+ hardware
6. **Default models by tier:**
   - 8GB: Qwen3-8B Q4_K_M (2x compaction only)
   - 16GB: Qwen3-14B Q4_K_M (2x-4x compaction)
   - 32GB: Qwen3-30B-A3B Q4_K_M (2x-8x compaction)
   - 64GB+: Qwen3-30B-A3B Q4_K_M (2x-16x compaction, 128K context feasible)
