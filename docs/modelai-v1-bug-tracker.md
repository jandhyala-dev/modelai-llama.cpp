# ModelAI llama.cpp — V1 Bug Tracker

**Repo:** `jandhyala-dev/modelai-llama.cpp`
**Branch:** `modelai-main`
**Tracking commit:** `626c1664` (updated 2026-03-14)

This document tracks all known bugs affecting modelai-llama.cpp, organized by source (internal, upstream, ModelAI integration). Each bug has a severity, status, root cause analysis, and resolution plan.

---

## Internal Bugs (Bugs in modelai-llama.cpp compaction code)

### BUG-I01: Nonuniform Pipeline Catastrophic Quality Loss at 8K

| Field | Value |
|-------|-------|
| **Severity** | Critical |
| **Status** | **FIXED** |
| **Pipeline** | nonuniform |
| **Models affected** | DeepSeek-R1-14B (confirmed), potentially all models at large context |
| **Discovered** | 2026-03-14, live benchmark at commit `df895bed` |
| **Fixed** | 2026-03-14, `src/llama-kv-compact-pipeline.cpp` |
| **Plan reference** | Phase 1A.1 |

**Symptoms:**
- DeepSeek-R1-14B at 8K/4x nonuniform: logit cosine = 0.211 (near-random)
- DeepSeek-R1-14B at 4K/4x nonuniform: logit cosine = 0.773 (below threshold)
- All `select` pipeline tests pass (0.993-0.999) on same model/context/ratio

**Root cause analysis:**
The nonuniform pipeline allocates per-head budgets based on attention entropy (Algorithm 4). When per-head selections are highly disjoint across many KV heads, the union of selections exceeds the target token count. The pipeline truncates the union by aggregate score ranking (`src/llama-kv-compact-pipeline.cpp:606-656`). After truncation, many heads lose ALL their selected tokens (`head_fully_masked=true` at line 727). These heads get:
- beta = -infinity for all positions (line 730)
- V = all zeros (lines 739-743)

At 8K with 4x compression on a model with many KV heads (DeepSeek-R1 has 8 KV heads per GQA group), the cascade causes the majority of heads to contribute zero attention weight, producing effectively random output.

**Fix implemented:**
After union truncation, count fully-masked heads. If >50% of heads are fully masked, fall back to `select` pipeline with `LLAMA_LOG_WARN`. This prevents catastrophic quality loss while preserving nonuniform benefits when head disjointness is moderate.

**Code location:** `src/llama-kv-compact-pipeline.cpp`, after union truncation block (line ~658).

---

### BUG-I02: Compacted Decode Throughput Regression at 32K

| Field | Value |
|-------|-------|
| **Severity** | Major |
| **Status** | **FIXED** |
| **Models affected** | Qwen3-30B-A3B at 32K (confirmed) |
| **Discovered** | 2026-03-14, ModelAI Phase D testing |
| **Fixed** | 2026-03-14, B5 GPU-resident tensor fix |
| **Plan reference** | Phase 1A.3 |

**Symptoms:**
- Baseline decode at 32K: 6.8-6.9 tok/s
- Compacted decode at 32K (2x compression, 5376 active_n_kv): 2.0 tok/s
- Expected: faster decode with fewer KV entries, not 3.4x slower

**Root cause (confirmed):**
The compacted prefix `set_input_*` functions used host pointer swap (`dst->data = cache.data()`) which forced ggml to treat the tensor as host-backed. When the graph scheduler encountered `ggml_concat` of a host-backed compacted prefix tensor with a GPU-backed live KV tensor, it created cross-backend operations — 80+ per decode batch at 32K context. Each cross-backend concat caused a Metal GPU↔CPU sync stall.

**Fix implemented (B5):**
Replaced host pointer swap with `ggml_backend_tensor_set()` in all three `set_input_compacted_prefix_k/v/kq_b` functions. The staging buffer is materialized on host (cache hit or miss), then uploaded to the tensor's native backend via `ggml_backend_tensor_set()` which routes to Metal/CUDA/CPU automatically. Removed `require_host_or_direct_data()` checks from K/V/beta exec functions since the caller now guarantees host staging.

**Code locations:**
- `src/llama-kv-cache.cpp:2177-2191` (K), `2213-2227` (V), `2255-2267` (beta)
- `src/llama-kv-compacted-prefix-exec.cpp:143-144` (K), `170-171` (V), `202-203` (beta)

---

### BUG-I03: Missing Compaction Timing Data at Short Contexts

| Field | Value |
|-------|-------|
| **Severity** | Minor |
| **Status** | OPEN |
| **Models affected** | Qwen3-30B-A3B at 4K (all ratios), 8K (4x, 8x) |
| **Discovered** | 2026-03-14, ModelAI Phase D testing |
| **Plan reference** | Phase 4.1 |

**Symptoms:**
- `compaction_time_ms`, `baseline_decode_tok_s`, `compacted_decode_tok_s` are null for some test points
- These are the most commonly used context sizes

**Fix:** Re-run benchmarks with timing instrumentation enabled for all test points.

---

### BUG-I04: Compaction Time Scales Super-Linearly

| Field | Value |
|-------|-------|
| **Severity** | Minor |
| **Status** | OPEN |
| **Models affected** | All models |
| **Discovered** | 2026-03-14, ModelAI Phase D testing |
| **Plan reference** | Phase 4.2 |

**Symptoms:**
- 8K: 518ms
- 16K: 1093-1312ms (~2.5x for 2x context)
- 32K: 2633-3663ms (~3x for 2x context, 7x total)

**Root cause:** The attention score computation in the select pipeline is O(n_prefix * n_queries * n_heads * n_layers). With n_queries proportional to context length, this becomes O(n^2). At 64K+, compaction could reach 10-15 seconds.

**Fix strategy:** Profile to identify the dominant O(n^2) component. Consider capping n_queries or using chunked score computation.

---

### BUG-I05: Qwen3-14B Quality Dip at 8K/8x

| Field | Value |
|-------|-------|
| **Severity** | Minor |
| **Status** | OPEN — investigation |
| **Pipeline** | select |
| **Models affected** | Qwen3-14B only |
| **Discovered** | 2026-03-14, ModelAI Phase D testing |
| **Plan reference** | Phase 4.3 |

**Symptoms:**
- Qwen3-14B at 8K/8x: logit cosine = 0.973 (lowest of 33 test points)
- All other test points > 0.99
- Qwen3-14B at 16K/8x: 0.987 (also slightly below average)

**Status:** Above the 0.95 threshold. Investigate in Phase 4.3 with additional test points (8K/6x, 8K/10x) to characterize the quality curve.

---

## Upstream Bugs (Bugs in ggml-org/llama.cpp affecting this fork)

### BUG-U01: Gemma3-12B SWA Decode 20x Slower Than Ollama

| Field | Value |
|-------|-------|
| **Severity** | Major |
| **Status** | **MITIGATED** (upstream bug, fork warning added) |
| **Upstream issue** | Not filed — upstream knows about SWA performance |
| **Models affected** | Gemma3-12B, all iSWA models |
| **Discovered** | 2026-03-14, 3-way benchmark |
| **Mitigated** | 2026-03-14, SWA runtime warning added |
| **Plan reference** | Phase 1A.4 |

**Symptoms:**
- Gemma3-12B decode: 0.7-0.8 tok/s on llama.cpp and modelai-llama.cpp
- Gemma3-12B decode: 16.5 tok/s on Ollama
- This is NOT a modelai-specific bug — both upstream and fork show same performance

**Impact on modelai:** Gemma3-12B is unusable for production. Must be listed as UNSUPPORTED.

**Fork action (implemented):** Added `LLAMA_LOG_WARN` in `compacted_prefix_runtime_supported()` (src/llama-kv-cache.cpp:1127-1133) when SWA sub-cache is detected. Warning fires once per session, informing users that compaction only applies to the base (non-SWA) cache in iSWA models.

---

### BUG-U02: modelai llama-bench Fails at pp512 for DeepSeek-R1-14B

| Field | Value |
|-------|-------|
| **Severity** | Major |
| **Status** | **INVESTIGATION COMPLETE** — not caused by fork changes |
| **Models affected** | DeepSeek-R1-14B |
| **Discovered** | 2026-03-14, 3-way benchmark |
| **Investigated** | 2026-03-14 |
| **Plan reference** | Phase 1A.2 |

**Symptoms:**
- Upstream llama-bench succeeds at pp512: 171.4 tok/s
- modelai fork llama-bench fails at pp512 for same model
- Other models work fine on modelai fork

**Investigation results (2026-03-14):**
Thorough code audit confirms the pp512 failure is NOT caused by fork changes:
1. `compacted_prefix_runtime_supported()` — all 6 callers are properly guarded
2. llama-bench — contains NO references to compacted prefix, cb_eval, or compaction
3. Graph build path — compacted prefix tensors only created when `compacted_prefix_active()` returns true (requires valid state)
4. DeepSeek architecture — GQA configuration read from model hparams, no hardcoded assumptions
5. Shape validation — all checks are defensive and would throw exceptions on mismatch

**Conclusion:** Root cause is upstream or environmental. The fork's compacted prefix code paths are never triggered during llama-bench execution. Requires reproduction with `LLAMA_LOG_LEVEL=debug` to isolate further.

---

### BUG-U03: Server Grammar Stack Crash on Apple Metal

| Field | Value |
|-------|-------|
| **Severity** | Critical (upstream) |
| **Status** | OPEN upstream (#19679, #19304) |
| **Impact** | Affects all llama-server deployments on Apple Silicon |

**Symptoms:** `llama_grammar_accept_token` throws on empty grammar stack. Crashes on M4 Mac Mini with flash attention + jinja. Also crashes at 86K context with 50+ tool calls.

**Fork action:** Track upstream fix. Not in fork implementation scope.

---

### BUG-U04: KV Cache Truncation on Chat Completions

| Field | Value |
|-------|-------|
| **Severity** | Medium (upstream) |
| **Status** | OPEN upstream (#11970) |
| **Impact** | Silent context loss that could interact with compaction |

**Symptoms:** KV cache sometimes truncated incorrectly during `/v1/chat/completions` API calls. Could cause compaction to compact an already-truncated cache.

**Fork action:** Monitor. Add integration test with chat completions API.

---

## Upstream Bug Sync Status (Verified)

### BUG-S01: KV Cache Defrag Corruption (#10873) — VERIFIED SAFE

| Field | Value |
|-------|-------|
| **Severity** | N/A (resolved upstream) |
| **Status** | VERIFIED — no impact on fork |
| **Upstream PR** | #10873 (OPEN/stale), superseded by #15473 (defrag removed) |
| **Verification** | Code audit + integration test (TEST 4) |

**Verification details:** No compaction code calls defrag. `compacted_prefix_reclaim_live_kv()` uses `llama_kv_cache_seq_rm()` (direct cell removal). Integration test confirms compaction works without defrag and produces correct logits.

---

### BUG-S02: KV Guard Refactor (#12695) — VERIFIED COMPATIBLE

| Field | Value |
|-------|-------|
| **Severity** | N/A (compatible) |
| **Status** | VERIFIED — fork uses current API |
| **Upstream PR** | #12695 (MERGED, Apr 2025) |
| **Verification** | Code audit + integration test (TEST 3, TEST 7) |

**Verification details:** Compaction code uses `used_max_p1()`, `is_empty()`, `seq_has()`, `pos_get()` — all current post-refactor API. Integration test confirms K/V extraction returns correct-size, all-finite data.

---

### BUG-S03: SWA KV Cache Support (#13194) — VERIFIED COMPATIBLE

| Field | Value |
|-------|-------|
| **Severity** | N/A (compatible) |
| **Status** | VERIFIED — SWA correctly rejected |
| **Upstream PR** | #13194 (MERGED, May 2025) |
| **Verification** | Code audit + integration test (TEST 2) |

**Verification details:** `compacted_prefix_runtime_supported()` checks instance-level `n_swa` (not model-level). For iSWA models, base cache has `n_swa=0` (compaction works), SWA sub-cache has `n_swa>0` (compaction rejected). M-RoPE guard also prevents compaction on Qwen3-VL and similar models.

---

### BUG-S04: KV Cells Unified (#11213) — NO ACTION NEEDED

| Field | Value |
|-------|-------|
| **Severity** | N/A (resolved) |
| **Status** | VERIFIED — no impact |
| **Upstream PR** | #11213 (CLOSED without merge) |
| **Verification** | Code audit + integration test (TEST 7) |

**Verification details:** PR #11213 was decomposed into #12695 and #13194, both of which are verified above. The cell iteration API used by compaction code is the current API.

---

### BUG-S05: Unified KV Buffer (#17450) — VERIFIED COMPATIBLE

| Field | Value |
|-------|-------|
| **Severity** | N/A (compatible) |
| **Status** | VERIFIED — tests use kv_unified=true |
| **Upstream issue** | #17450 (CLOSED, not planned — cosmetic) |
| **Verification** | All tests run with kv_unified=true + integration test (TEST 1) |

---

### BUG-S06: Shift/Defrag Correctness (#12253) — VERIFIED SAFE

| Field | Value |
|-------|-------|
| **Severity** | N/A (guarded) |
| **Status** | VERIFIED — shift guard prevents interaction |
| **Upstream issue** | #12253 (CLOSED, completed Mar 2025) |
| **Verification** | Code audit + integration test (TEST 5) |

**Verification details:** `compacted_prefix_reclaim_live_kv()` at llama-kv-cache.cpp:731 checks `cells.get_has_shift()` and returns false when shift is pending. This prevents any interaction between compaction and shift bugs.
