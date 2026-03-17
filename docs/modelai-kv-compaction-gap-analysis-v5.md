# Adversarial Gap Analysis & Implementation Plan V5

**Date:** 2026-03-16
**Supersedes:** V4 (same date) — V5 adds GAP-N (SWA validation), GAP-O (MLA unsupported), Qwen3.5 architecture analysis
**V4 review status:** PASS from both reviewers (3 shared Minor findings, incorporated in V4→V5)
**V5 review status:** PASS from both reviewers (R1: 1 Minor — C_v Tier 3 mislabeled; R2: 1 Minor — GAP-N Gemma-3 load issue). Both findings incorporated below.
**Commit baseline:** 3f357af6 (latest, includes benchmark results + stale-prompt-cache fix)
**Sources audited:**
1. `modelai-llama.cpp` @ `modelai-main` (3f357af6) — 23 compaction source files
2. MIT `adamzweiger/compaction` @ `2e59cc8` — Python reference (DEAD: 2 commits, Feb 19)
3. `fabiantax/llama.cpp` branches `feat/kv-compaction` + `claude/kv-cache-compaction-636FM`
4. arXiv:2602.16284 "Fast KV Compaction via Attention Matching"
5. Upstream `ggml-org/llama.cpp` — issue #20037 (RFC only), commit 4893cc07 (hybrid seq_rm fix)

**V4→V5 changes:**
- Added GAP-N: SWA validation (iSWA base layers, Gemma 3)
- Added GAP-O: MLA listed as unsupported with rationale (DeepSeek V3/R1 full)
- Added Qwen3.5-35B-A3B architecture analysis in GAP-F (hybrid + IMROPE = 2 stacked blockers, 3:1 recurrent:attention layer ratio, specific layer pattern from model loader)
- Gap count: 15 gaps (A-O) — 10 actionable, 3 deferred, 2 validation-only
- Added V4-N to Week 1 timeline alongside V4-C
- Softened DeepSeek R1 distilled claim — no explicit testing in fork, expected based on architecture lineage
- Added MIT exhaustive audit results (5 minor items reviewed, none rise to gap level)
- Fixed C_v Tier 3 label: "pseudoinverse" → "aggressive Cholesky (escalated λ)" (V5 R1 Minor-1)
- Added GAP-N known issue: Gemma-3-12B-IT load failure in prior testing (V5 R2 Minor-1)

**V3→V4 changes (reviewer-driven):**
- GAP-C: Corrected "max/sum/RMS" → "sum/RMS only"; added MAX/MEAN to gap scope (R1 Major-1)
- GAP-E: Corrected "hook into FAILED_PREPARE retry path" → "build retry path" — no retry exists (R1 Minor-3)
- Added GAP-L: Context-prefill server integration (R2 Minor-01)
- Added GAP-M: OMP performance — >600s timeout on 30B models (benchmark finding)
- Fixed "10 methods" → "9 methods" (R1 Minor-2)
- Fixed RMS line reference: select.h:31, select.cpp:55 (R1 Minor-4)
- Fixed layer filter location: filter lambda in memory-hybrid.cpp:45-49, callback in kv-cache.cpp:132 (R2 Minor-05)
- Reordered V4-A (crash guards) before V4-B (beta graduation) (R2 Minor-03)
- Added V4-D→V4-I dependency (R1 Minor-5)
- Documented chunk overlap as intentionally excluded (R1 Minor-1)
- Added model download prerequisite note (R2 Minor-04)
- Downgraded "100% confidence" to ~98% (both reviewers)
- Incorporated benchmark data: select 100% recall at 50x, solver 39-48s, OMP >600s timeout

---

## PART 1: CORRECTED INVENTORY — What modelai Actually Has

After full code audit of all 23 source files, verified by two independent reviewers:

| Feature | Status | Files | Verification |
|---------|--------|-------|-------------|
| Top-k selection | DONE | llama-kv-compact-select.cpp | `select_topk()` at line 67 |
| OMP with progressive schedule | DONE | llama-kv-compact-select.cpp:200-433 | Schedule `[(300,1,1),(1500,2,2),(∞,4,2)]` at line 316-320 |
| Drop-key refinement (beta < -7) | DONE | llama-kv-compact-select.cpp:378-432 | Max 3 passes, `beta_cutoff` logic |
| NNLS solver (PGD + lstsq+clamp) | DONE | llama-kv-compact-solver.cpp | Box-constrained PGD, Cholesky fallback |
| LSQ C_v fitting (3-tier cascade) | DONE | llama-kv-compact-solver.cpp | LAPACK sgels → Cholesky → aggressive Cholesky (escalated λ) |
| Spectral ridge scaling | DONE | llama-kv-compact-solver.cpp | `SPECTRAL` mode at line 505, power iteration at line 423 |
| Score aggregation (SUM + RMS) | DONE | llama-kv-compact-select.h:31, select.cpp:55 | `finalize_rms_scores()` — **NO MAX or MEAN mode** |
| Beta computation (log-weights) | DONE | llama-kv-compact-solver.cpp | `fit_beta()` |
| Beta injection in compute graph | DONE | llama-kv-compacted-prefix-exec.cpp:198 | `set_input_beta()` loads kq_b tensor |
| Flash attn + zero-beta path | DONE | llama-graph.cpp:2210-2238 | Per-layer zero-beta eligibility check |
| Self-study Q capture (multi-round) | DONE | llama-kv-compact-self-study.cpp | `n_rounds`, `temperatures[]`, auto mem budget |
| On-policy iterative (Phase 8) | DONE | llama-kv-compact-on-policy.cpp:54 | Quality gate at line 140-151 |
| On-policy sequential per-layer | DONE | llama-kv-compact-on-policy.cpp:177 | `sequential_on_policy_from_live_kv()` |
| Chunked compaction (>8K) | DONE | llama-kv-compact-pipeline.cpp:940 | `chunked_from_live_kv()` — non-overlapping (see note below) |
| Entropy-based per-head budgets | DONE | llama-kv-compact-budget.cpp | `head_entropy()`, `allocate_budgets()` |
| Metal GPU solver acceleration | DONE | llama-kv-compact-solver-metal.mm | GPU attention scores + XᵀX assembly |
| Compacted prefix state save/restore | DONE | llama-kv-compacted-prefix.h | `state_write/read` |
| Server HTTP endpoints (9 methods) | DONE | server-context.cpp:2224-2228 | select, solver, omp, self_study, chunked_self_study, nonuniform, chunked, on_policy, sequential_on_policy |
| NEON-optimized dot product | DONE | llama-kv-compact-math.h | ARM intrinsics |
| Non-uniform budget JSON import | DONE | llama-kv-compact-budget.cpp | `load_budget_json()` |
| Quality gate (residual improvement) | DONE | llama-kv-compact-on-policy.cpp:140-151 | Relative improvement threshold |
| Q-capture memory budget limiting | DONE | llama-kv-compact-self-study.cpp | `max_q_capture_mb` auto-reduction |
| Cached OMP selection order | DONE | llama-kv-compact-select.h:77-80, select.cpp:248-280 | Reuses greedy order, recomputes beta via NNLS |
| Hybrid model KV cache extraction | DONE | server-context.cpp:112-123 | `get_kv_cache_base()` handles plain, iSWA, hybrid, hybrid-iSWA |
| Hybrid seq_rm rollback safety | DONE | llama-memory-hybrid.cpp:132-139 | Recurrent-first rollback, server prompt_clear(true) |
| Hybrid layer filtering | DONE | llama-memory-hybrid.cpp:45-49 (filter lambda), llama-kv-cache.cpp:132 (callback invocation) | `map_layer_ids` maps only attention layers |

**Note on chunked compaction:** Uses non-overlapping chunks. Chunk overlap is intentionally excluded — overlap is primarily useful for optimization-based compaction (learning synthetic keys across chunk boundaries). For selection-based compaction (choosing original keys to keep), non-overlapping chunks are correct because key selection is position-independent.

**The V1 benchmark uses `select` method (zero-beta) by choice, NOT because beta is unimplemented.**

---

## PART 2: VERIFIED GAPS — What's Actually Missing

### GAP-A: V1 Ships Select-Only (Policy Gap, Conditional on PPL Data)

**What:** The V1 default uses `select` method which produces zero-beta.

**Benchmark data (Qwen3-Coder-30B-A3B, M2 Pro 32GB):**

| Method | Time | Recall (2x-50x) |
|--------|------|-----------------|
| select | 62-185ms | 100% at all ratios |
| solver (beta+C_v) | 39-48s | 100% at all ratios |
| omp | >600s timeout | untestable |

**Why it matters:** The benchmark shows select achieves 100% recall even at 50x (4213→84 tokens). Solver adds beta+C_v but is 200-750x slower with no measurable quality improvement on this 10-fact recall test. However, recall is a coarse metric — PPL would reveal subtle attention quality degradation that recall doesn't capture.

**Paper reference:** Section 3.2 — "The beta correction is essential for maintaining attention output fidelity beyond 5x compression." Our benchmark contradicts this at the recall level but hasn't tested PPL.

**What to do:** CONDITIONAL on V4-D (benchmark tool) PPL results:
- If PPL shows beta improves quality: graduate `solver` to default (not `omp` — see GAP-M)
- If PPL shows no difference: keep `select` as default, document beta as optional quality mode
- In either case, `omp` cannot be default until GAP-M is resolved

**Effort:** Config change + benchmark. No new code.

---

### GAP-B: Non-Uniform Budget Solver (Entropy Proxy vs Influence Curves)

**What we have:** Entropy-based allocation — `H(head) = -sum(p log p)`, lower entropy = more sensitive = bigger budget.

**What the paper has:** Influence curves — compact each head at multiple ratios, measure reconstruction error, use swap-based solver to optimize allocation.

**MIT reference:** `head_budget_optimization/solver.py` — greedy-from-zero, swap-based (recommended), simulated annealing.

**Why it matters:** Entropy is a proxy. The influence curve approach measures actual damage. Paper calls this "most impactful ablation" in Section 3.4.

**What to do:**
1. Add `llama_kv_compact_head_influence_curve()` — compact one head at ratios [0.01, 0.05, 0.1, 0.2, 0.5, 1.0], measure MSE
2. Add `llama_kv_compact_swap_budget_solver()` — swap-based allocation from influence curves
3. Leverage existing `cached_selection_order` to amortize OMP cost across ratios
4. Keep entropy as fast default, influence as quality option

**Effort:** ~400 lines C++. No GPU work.

---

### GAP-C: Score Aggregation Modes + Pooling

**What we have:** SUM and RMS aggregation only (select.h:10-13). No MAX. No MEAN. No pooling.

**What the paper has (Section 3.3):** Three aggregation modes: max, mean, RMS. Optional avgpool/maxpool smoothing with kernel_size=7.

**MIT reference:** `highest_attention_keys.py` — all three aggregation modes, `pooling` parameter with avgpool/maxpool.

**Why it matters:** Paper ablation shows RMS is generally best, but MAX can outperform on specific architectures where a single dominant query determines key importance. MEAN is equivalent to SUM after normalization. Pooling reduces noise in long contexts.

**What to do:**
1. Add `LLAMA_KV_COMPACT_SCORE_AGG_MAX` and `LLAMA_KV_COMPACT_SCORE_AGG_MEAN` to enum
2. Implement MAX (per-key maximum across queries) and MEAN (normalized sum) in accumulate function
3. Add optional 1D avgpool with kernel_size parameter over position-sorted scores
4. Make pooling optional (default off, enable via server param)

**Effort:** ~40 lines for aggregation modes, ~20 lines for pooling. Trivial.

---

### GAP-D: No Public C API

**What we have:** Internal-only compaction. Server exposes HTTP endpoints but no C API in `llama.h`.

**What fabiantax has:**
```c
llama_compact_params llama_compact_params_default(void);
int32_t llama_kv_cache_compact(ctx, seq_id, params);
void llama_kv_cache_set_auto_compact(ctx, threshold, params);
```

**Why it matters:** Blocks COT integration, third-party apps, and upstream contribution.

**What to do:** Export `llama_compact_params` struct + `llama_kv_cache_compact()` in `include/llama.h`. Thin wrapper around existing pipeline methods.

**Effort:** ~100 lines header + ~50 lines implementation.

---

### GAP-E: No Auto-Compaction During Generation

**What we have:** Manual `POST /compact` endpoint only.

**What fabiantax has:** `llama_kv_cache_set_auto_compact(ctx, threshold=0.9, params)`.

**Why it matters:** Conversations exceeding context silently error. Auto-compaction transparently extends effective context.

**What to do:** Build a prepare-retry path in server slot processing (no retry mechanism currently exists — R1 verified this):
1. Add `llama_kv_cache_set_auto_compact()` to public API
2. Store threshold + params in context
3. After slot prepare failure due to full KV cache:
   - Check if auto-compact enabled and KV usage > threshold
   - Set `slot.auto_compact_attempted = true` (one-shot guard)
   - Run compaction with stored params
   - Retry slot prepare
   - If retry fails, report error normally (guard prevents loop)
4. Reset `auto_compact_attempted` at start of each new request
5. Log compaction events with timing

**Concurrency:** Server task processing is single-threaded (all tasks processed sequentially in main loop — R1 adversarial trace verified this). No explicit compaction lock is needed in current architecture, but implicit serialization through the task loop provides safety. If the server ever becomes multi-threaded, an explicit lock must be added.

**Effort:** ~100 lines in server-context.cpp (includes building the retry path).

---

### GAP-F: IMROPE / M-RoPE Model Support

**What we have:** Unsupported (CLAUDE.md V0 matrix). No IMROPE detection in compaction code.

**What fabiantax has:** IMROPE detection via `llama_model_rope_type()`, ext_x/ext_y per cell.

**CRITICAL:** fabiantax's position remapping (`cell.pos = j`) is BROKEN — destroys multi-resolution spatial encoding.

**What to do:**
1. Add `n_pos_per_embd` detection from `llama_model_rope_type()`
2. Preserve ext_x/ext_y in compacted prefix store
3. Do NOT remap positions for IMROPE — keep original positions
4. Requires V4-C (hybrid validation) to succeed first

**Qwen3.5-35B-A3B architecture analysis (stacked blockers):**

Qwen3.5-35B-A3B has TWO stacked blockers — hybrid SSM+attention (GAP-G) AND IMROPE (this gap):

- **Layer pattern:** `hparams.recurrent_layer_arr[i] = ((i + 1) % full_attn_interval != 0)` with `full_attn_interval = 4` (src/llama-model.cpp:2433-2440). Every 4th layer (0-indexed: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39) is full attention. Remaining 30/40 layers are recurrent (Mamba SSM). 3:1 recurrent:attention ratio.
- **Hybrid blocker (GAP-G):** SOLVED — `get_kv_cache_base()` extracts attention-only cache via `get_mem_attn()`. Layer filter excludes recurrent layers from KV cache. Needs validation testing only.
- **IMROPE blocker (this gap):** UNSOLVED — `QWEN35MOE` maps to `LLAMA_ROPE_TYPE_IMROPE` (src/llama-model.cpp:8841). Multi-dimensional positions via `rope_sections` (4 sections). Compacted prefix stores scalar `pos` — must be extended for multi-resolution positions.
- **Dependency:** GAP-G (hybrid validation) must pass before attempting IMROPE work. If hybrid compaction fails on Qwen3-30B-A3B (standard RoPE), IMROPE is moot.

**Effort:** ~200 lines.
**Risk:** HIGH. Open question: should this be BLOCKED until upstream provides IMROPE compaction test vectors?

---

### GAP-G: Hybrid SSM+Attention Validation (Testing Only)

**V3 correction (verified by both reviewers):** Code already exists. `get_kv_cache_base()` handles all 4 memory types. KV cache constructor filters via `!hparams.is_recurrent(il)` lambda. `map_layer_ids` prevents wrong-layer access. Upstream seq_rm fix (4893cc07) is present.

**What's missing:** Nobody has run compaction on a hybrid model.

**What to do:**
1. Load Qwen3-30B-A3B (hybrid SSM+attention, standard RoPE)
2. Prefill 4K tokens, compact via `POST /compact` method=select (fast, proven)
3. Verify: compaction only touches attention layers
4. Verify: generation continues correctly
5. PPL comparison pre/post
6. Update CLAUDE.md V0 support matrix

**Prerequisite:** Model download — Qwen3-30B-A3B is ~17GB Q4_K_M. Allow 1-4 hours for download depending on bandwidth.

**Effort:** 0.5 day testing.
**Risk:** Low.

---

### GAP-H: Optimization-Based Compaction (Deferred)

Joint gradient optimization of C1, beta, C2 via LBFGS/Adam. Research direction, not production.

---

### GAP-I: OMP Full-Attention Evaluation (Deferred)

Marginal quality improvement at enormous cost. Standard OMP is already high quality.

---

### GAP-J: Benchmark Tool with PPL Evaluation

**What fabiantax has:** `kv-compact-bench.cpp` — PPL after compaction, ablation flags.

**What we have:** `scripts/bench-compaction-methods.py` (recall-based, commit 3f357af6), but no integrated PPL measurement.

**What to do:** Build `tools/kv-compact-bench/`:
1. Prefill from file (wikitext-2 or custom)
2. Compact at specified ratio/method
3. Continue eval and compute PPL
4. Ablation: `--no-beta`, `--no-cv`, `--evict-only`
5. Environment variables: `LLAMA_COMPACT_NO_BETA`, `LLAMA_COMPACT_NO_CV`
6. CSV output

**Effort:** ~300 lines.

---

### GAP-K: Extreme Compression Robustness

**What happened:** Server crashed after 114x compaction.

**Root causes:** Underdetermined solver, budget starvation (0 tokens for some heads), NaN cascade in self-study.

**What to do:**
1. Minimum budget floor per head (2 tokens) — prevents starvation
2. NaN guard after solver output — falls back to zero-beta for that head
3. Total budget floor: refuse if `target_tokens < n_head_kv * min_per_head`
4. Warning log with recommended max ratio

**Note (R1 boundary trace):** The 2-token floor is necessary but not sufficient. At 114x on 3277 tokens: target=28, floor=10 (5 heads × 2). Target passes floor but system is still severely underdetermined. The NaN guard is the real safety net.

**Effort:** ~50 lines.

---

### GAP-L: Context-Prefill Server Integration (NEW in V4)

**Source:** R2 Minor-01, confirmed in llama-kv-compact-prefill-q.h:9-12.

**What exists:** `prepare_q_capture()` infrastructure for capturing Q tensors during forward pass. Two modes documented: context-prefill (zero overhead, requires cb_eval hook) and repeat-prefill (doubles prefill cost, self-contained).

**What's missing:** Server integration to install `cb_eval` during the original prefill decode. Currently, all Q-capture paths use either repeat-prefill (self_study) or K-as-surrogate (select/solver). Context-prefill would enable zero-overhead Q capture.

**Why it matters:** At 16K context, repeat-prefill adds ~2s. Context-prefill eliminates the re-decode cost — Q is captured during the prefill that's happening anyway.

**What to do:**
1. In server slot processing, install `cb_eval` before the prefill decode
2. Capture Q tensors per-layer during prefill
3. Store in Q-capture state attached to the slot
4. After prefill, Q-capture state is available for compaction solver without re-decode
5. Restore original cb_eval after prefill
6. For contexts >8K, subsample captured Q positions (reuse `max_q_capture_mb` infrastructure from self-study) to bound memory

**Memory note (R1+R2 finding):** Naive Q-capture at 100K context = 40 layers × n_head_q × 100K × 128 × 4B = tens of GB. Must subsample or use reservoir sampling to stay within budget. The existing `max_q_capture_mb` auto-reduction handles this for self-study; context-prefill must reuse the same infrastructure.

**Effort:** ~80 lines in server-context.cpp (increased from 40 to account for Q subsampling logic).
**Risk:** Low. cb_eval is an existing hook mechanism. Memory bounded by existing infrastructure.

---

### GAP-M: OMP Performance at Scale (NEW in V4)

**Source:** Benchmark finding. OMP timed out at >600s on Qwen3-Coder-30B-A3B with just 5K tokens at 2x compression.

**Root cause:** OMP runs the full greedy loop independently per KV head:
- 384 heads (48 layers × 8 KV heads)
- Per head: ~1150 OMP iterations × 4000 candidates × 256 queries ≈ 1.2 billion ops
- Total: ~460 billion ops. At single-thread CPU: hours.

Compare: `select` does one-pass attention scoring (no iterative loop) → 62-185ms.

**Why it matters:** OMP cannot be the default method if it can't finish on production models. It also blocks `self_study` and `on_policy` benchmarking (server gets stuck after OMP timeout).

**What to do (pick one or combine):**
1. **GPU-accelerate OMP correlation** — the inner loop (select.cpp:300-313) is embarrassingly parallel. Metal kernel for correlation computation. (Deferred to future phase — see V4-H scope note.)
2. **Per-head timeout with fallback** — if one head's OMP exceeds 5s, keep the partial OMP selection (greedy property ensures each partial selection is locally optimal) and fill remaining positions with top-k scored keys. Beta refit runs on the full combined selection. OMP becomes "best effort" quality improvement.
3. **Subset OMP** — run OMP only on the top-N most entropy-sensitive heads, top-k for the rest.
4. **Coarser progressive schedule** — increase k_choice for early iterations (select 4 keys at a time instead of 1).

**Effort:** Option 2 is ~30 lines (simplest). Option 1 is ~200 lines (best long-term). Options can be combined.
**Risk:** Medium for option 1 (Metal kernel). Low for options 2-4.

---

### GAP-N: SWA (Sliding Window Attention) Validation (NEW in V4)

**What we have:** `get_kv_cache_base()` handles `llama_kv_cache_iswa` via `kv_iswa->get_base()`, extracting the base (non-SWA) cache. SWA layers use a sliding window and don't need compaction. Code exists, but is untested.

**Why it matters:** Gemma 3 and similar iSWA models are popular. Base layers are standard attention and should be compactable — SWA layers pass through unchanged.

**What to do:**
1. Load Gemma 3 (iSWA model)
2. Compact via method=select
3. Verify: only base layers compacted, SWA layers untouched
4. Verify: generation continues correctly
5. Update CLAUDE.md support matrix

**Known issue:** Gemma-3-12B-IT failed to load in prior testing (missing hyperparameter key). May require GGUF rebuild or upstream fix before validation can proceed.

**Effort:** 0.5 day testing, 0 lines new code (unless bugs found).
**Risk:** Low (if model loads). Model load issue may require upstream resolution.

---

### GAP-O: MLA (Multi-head Latent Attention) — Unsupported (Deferred)

**Affected models:** DeepSeek V3, DeepSeek R1 (full, not distilled).

**Why it's fundamentally different:** MLA stores compressed latent vectors per position, not standard K/V tensors. K and V are projected on-the-fly from latents during attention computation. Our compaction algorithm (select original keys to keep, compute beta, refit C_v) assumes standard K/V — there are no "original keys" in MLA, only latent codes.

**Theoretical approaches (all problematic):**
1. **Evict latent positions** — loses both K and V information simultaneously (no independent key selection possible)
2. **Project K/V from latents, compact, re-encode** — lossy round-trip, expensive, and the compressed latent space may not preserve the reconstructed K/V faithfully
3. **Latent-space compaction** — select which latent positions to keep based on attention scores. Simpler than (2) but the paper's beta/C_v math doesn't apply to latent representations

**Decision:** List as **unsupported** in V0 matrix. Not a gap to close — fundamentally different architecture that doesn't map to arXiv:2602.16284. Revisit if/when a latent-space compaction paper appears.

**Note:** DeepSeek R1 *distilled* variants (Qwen-based, Llama-based) are expected to use standard KV cache based on their architecture lineage. Verify via model loader (`llama_model_rope_type()` and KV cache type) before claiming support — no explicit testing has been done on distilled variants in this fork.

---

## PART 3: FEATURES WE HAVE THAT NOBODY ELSE DOES

| Feature | modelai | MIT | fabiantax |
|---------|---------|-----|-----------|
| Metal GPU solver | YES | No | No |
| LAPACK solver (QR, numerically stable) | YES | PyTorch lstsq | Hand-rolled Gaussian elimination |
| Multi-round self-study with temp sweep | YES | Single-round | K-proxy only |
| Iterative on-policy with quality gate | YES | Basic on-policy | No |
| Sequential per-layer on-policy refit | YES | Yes (Python) | No |
| OMP + progressive schedule + drop-key | YES | Yes (Python) | No |
| Chunked self-study for long contexts | YES | KV-based chunking | No |
| Compacted prefix store (separate KV) | YES | CompactedPrefixCache (Python) | In-place mutation |
| Per-layer zero-beta flash attn | YES | N/A | No |
| 9 server methods via HTTP | YES | N/A | 1 method |
| NEON dot product | YES | N/A | No |
| Q-capture memory budget limiting | YES | N/A | No |
| Cached OMP selection order | YES | Yes (Python) | No |
| Hybrid model server support (4 types) | YES | N/A | No (same cast limitation) |
| Hybrid seq_rm rollback safety | YES | N/A | No |

---

## PART 4: BENCHMARK DATA

### Compaction Method Comparison (Qwen3-Coder-30B-A3B, M2 Pro 32GB)

| Method | Ratio | KV Before | KV After | Time (ms) | Recall |
|--------|-------|-----------|----------|-----------|--------|
| select | 2x | 3,889 | 1,944 | 185 | 100% |
| select | 4x | 4,125 | 1,031 | 114 | 100% |
| select | 10x | 5,516 | 551 | 105 | 100% |
| select | 25x | 4,242 | 169 | 64 | 100% |
| select | 50x | 4,213 | 84 | 62 | 100% |
| solver | 2x | 4,182 | 2,091 | 47,225 | 100% |
| solver | 4x | 4,386 | 1,096 | 44,000 | 100% |
| solver | 10x | 5,006 | 500 | 47,662 | 100% |
| solver | 25x | 4,278 | 171 | 39,584 | 100% |
| solver | 50x | 4,421 | 88 | 40,387 | 100% |
| omp | 2x | 5,139 | — | >600,000 | timeout |

**Key findings:**
1. `select` is the production winner: 100% recall at all ratios, 62-185ms
2. `solver` adds beta+C_v but is 200-750x slower with no measurable recall improvement
3. `omp` is non-viable on 30B models without GPU acceleration or algorithmic improvement
4. Even at 50x (4213→84 tokens), all 10 injected facts recalled perfectly
5. `self_study` and `on_policy` blocked by OMP server timeout, untested

**Caveat:** 10-fact recall is a coarse metric. PPL testing (V4-D) needed to detect subtle quality differences.

---

## PART 5: COMPARATIVE ANALYSIS — All Repos

### MIT `adamzweiger/compaction` (Python Reference)

**Status:** DEAD. 2 commits total (Feb 19, 2026).

**Gaps from MIT not in our code:**
- Influence curve budget solver (GAP-B)
- MAX score aggregation (GAP-C)
- Attention score pooling (GAP-C)
- Optimization-based compaction (GAP-H, deferred)
- OMP-full attention evaluation (GAP-I, deferred)

**Additional MIT features reviewed (do not rise to gap level):**
- `zerobeta` option — equivalent to our `select` method (zero-beta by design); ablation covered by V4-D `--no-beta` flag
- Direct C2 nearest-neighbor fitting — alternative to LSQ C_v. Minor quality variant; our 3-tier cascade (LAPACK sgels → Cholesky → aggressive Cholesky (escalated λ)) is more robust
- Global selection methods (cross-head budget allocation in selection step) — interesting for MoE architectures but not in the paper's algorithm; revisit if influence-curve budgets (GAP-B) prove insufficient
- Text-based chunking (sentence/paragraph boundaries) — requires tokenizer awareness; paper uses fixed-size chunks; our non-overlapping chunking is correct for selection-based approach
- `normalize_exp_scores` / `use_abs_corr` — already implemented in our score computation pipeline

**No model support** beyond dense Qwen3 (no MoE, SSM, IMROPE).

### fabiantax `llama.cpp` (Third-Party C++ Fork)

**Gaps from fabiantax not in our code:**
- Public C API (GAP-D)
- Auto-compaction threshold (GAP-E)
- IMROPE detection + ext_x/ext_y storage (GAP-F)
- PPL benchmark tool with ablation (GAP-J)
- Environment variable ablation flags

**fabiantax features we intentionally DON'T borrow:**
- `defrag_after_compact()` — our prefix store is better architecture
- State-level save/parse/modify/restore — different architecture choice
- Sensitivity variance weighting — we have entropy (good) and will add influence curves (best)
- IMROPE position remapping (`cell.pos = j`) — **BROKEN**, destroys spatial encoding
- `trailing_data` SSM passthrough — unnecessary, our `get_mem_attn()` extracts attention cache directly

### Upstream `ggml-org/llama.cpp`

- Issue #20037: RFC only, no implementation, stalled
- Commit 4893cc07: hybrid seq_rm fix — **already in our fork** ✓
- Commit bb96bfd3: hybrid memory size fix — **already in our fork** ✓

---

## PART 6: WHAT TO BORROW FROM FABIANTAX

| Item | Source | Borrow? | Rationale |
|------|--------|---------|-----------|
| `n_pos_per_embd` detection | server-kv-compact.h | YES | Right API for IMROPE detection |
| `ext_x`/`ext_y` storage | kv-compact-state.h | YES (concept) | Adapt to our prefix store |
| `llama_compact_params` struct | llama.h | YES (adapt) | Good API surface |
| PPL benchmark concept | kv-compact-bench.cpp | ADAPT | Rewrite with our solver |
| `LLAMA_COMPACT_NO_BETA/CV` env vars | kv-compact-bench.cpp | YES | Useful ablation |
| Auto-compaction concept | llama.h | YES (concept) | Implement in our server path |
| `defrag_after_compact()` | llama-kv-compact.cpp | NO | Prefix store is better |
| State save/parse/restore | kv-compact-state.h | NO | Different architecture |
| Sensitivity variance | kv-compact-math.h | NO | Entropy + influence is better |
| IMROPE `cell.pos = j` | server-kv-compact.h | **NO** | BROKEN |
| `trailing_data` passthrough | kv-compact-state.h | NO | Unnecessary with get_mem_attn() |

---

## PART 7: IMPLEMENTATION PLAN

### Phase V4-A: Extreme Compression Guards + Score Modes (0.5 day)

**Goal:** Prevent 114x-style crashes, add missing aggregation modes.

1. Add minimum budget floor per head (2 tokens)
2. Add NaN guard after solver output
3. Add total budget floor with warning log
4. Add `LLAMA_KV_COMPACT_SCORE_AGG_MAX` and `LLAMA_KV_COMPACT_SCORE_AGG_MEAN` to enum
5. Implement MAX and MEAN accumulation in score computation
6. Add optional 1D avgpool with kernel_size parameter

**Files:** llama-kv-compact-budget.cpp, llama-kv-compact-solver.cpp, llama-kv-compact-select.h, llama-kv-compact-select.cpp, llama-kv-compact-pipeline.cpp
**Risk:** Trivial.
**Testing:** Attempt 100x compaction — must not crash, must log warning. Verify MAX/MEAN/RMS produce different score rankings on a test vector.

---

### Phase V4-B: Graduate Beta (Conditional on PPL) (0.5 day)

**Goal:** Determine whether beta/C_v provides measurable quality improvement.

**CONDITIONAL on V4-D results.** Do not execute until PPL data exists.

**Decision criterion (define during V4-D):**
- PPL difference < 0.5% at target ratio → select ≈ solver → keep `select` as default
- PPL difference 0.5-2% → marginal win → document as "solver recommended for quality-sensitive workloads"
- PPL difference > 2% → solver wins → graduate `solver` to default, document speed cost
- Measure on wikitext-2 with n≥3 runs per configuration for statistical confidence
- Test at ratios: 5x, 10x, 20x, 50x

1. If PPL shows select ≈ solver: keep `select` as default, document solver as quality option
2. If PPL shows solver wins: change default to `solver`, document the 200-750x speed cost
3. In NO case make `omp` default until GAP-M is resolved

**Files:** server-context.cpp (gating), CLAUDE.md
**Risk:** Low.

---

### Phase V4-C: Hybrid Model Validation (0.5 day)

**Goal:** Verify existing hybrid support works end-to-end.

**Prerequisite:** Model download — Qwen3-30B-A3B is ~17GB. Allow 1-4 hours.

1. Load Qwen3-30B-A3B
2. Prefill 4K tokens, compact via method=select
3. Verify attention-only layer compaction (check layer count in logs)
4. Verify generation continues
5. Update CLAUDE.md

**Files:** CLAUDE.md only (unless bugs found)
**Risk:** Low.

---

### Phase V4-D: Benchmark Tool with PPL (1 day)

**Goal:** PPL-based quality measurement. Answers "does beta matter?" and enables all subsequent quality validation.

1. Build `tools/kv-compact-bench/`
2. PPL evaluation: prefill → compact → continue eval
3. CSV output: ratio, method, ppl, ppl_err, time_ms
4. Ablation: `--no-beta`, `--no-cv`, `--evict-only`
5. Env vars: `LLAMA_COMPACT_NO_BETA`, `LLAMA_COMPACT_NO_CV`

**Files:** new tools/kv-compact-bench/
**Risk:** Low.

---

### Phase V4-E: Public C API (1 day)

**Goal:** Export compaction to COT and third-party consumers.

1. `llama_compact_params` struct + `llama_kv_cache_compact()` in include/llama.h
2. Thin wrapper in src/llama-kv-compact-api.cpp
3. Default method: `select` (proven fast, 100% recall)

**Files:** include/llama.h, new src/llama-kv-compact-api.cpp, CMakeLists.txt
**Risk:** Low.

---

### Phase V4-F: Auto-Compaction (1 day)

**Goal:** Transparent context extension when KV cache fills.

1. `llama_kv_cache_set_auto_compact()` in public API
2. Build prepare-retry path in server slot processing (currently NO retry exists)
3. After slot prepare failure:
   - Check auto-compact enabled + KV usage > threshold
   - Set `slot.auto_compact_attempted = true`
   - Compact
   - Retry prepare
   - If retry fails, report error (guard prevents loop)
4. Reset guard per new request

**Concurrency:** Task processing is single-threaded (verified by R1 adversarial trace). No lock needed in current architecture. If server becomes multi-threaded, explicit compaction lock required.

**Files:** server-context.cpp
**Risk:** Medium.

---

### Phase V4-G: Context-Prefill Server Integration (0.5 day)

**Goal:** Zero-overhead Q capture during original prefill.

1. Install `cb_eval` before prefill decode
2. Capture Q tensors per-layer (subsample for contexts >8K using `max_q_capture_mb` budget)
3. Store Q-capture state on slot
4. Available for solver without repeat-prefill or self-study cost
5. Restore original cb_eval after prefill

**Files:** server-context.cpp (~80 lines, includes Q subsampling logic)
**Risk:** Low.

---

### Phase V4-H: OMP Performance Fix (1 day — timeout fallback only)

**Goal:** Make OMP viable on 30B+ models.

Current: 384 heads × 1150 OMP iterations × O(T×n) correlation = hours.

**Week 2 scope (R2 clarification):** Implement timeout fallback only (~30 lines). Metal kernel for OMP correlation deferred to future phase.

**Approach:**
1. Per-head timeout (5s) with top-k fallback — ~30 lines, immediate fix
2. Coarser progressive schedule for large models — adaptive k_choice based on T×n_heads
3. *(Deferred)* Metal kernel for correlation step — ~200 lines, future phase

**Timeout fallback semantics (R1 clarification):** Keep partial OMP selection (greedy property ensures validity); fill remaining positions with top-k scored keys. Beta refit runs on the full combined selection.

**Files:** llama-kv-compact-select.cpp
**Risk:** Medium (Metal kernel). Low (timeout fallback).

---

### Phase V4-I: Influence-Curve Budget Solver (2 days)

**Goal:** Replace entropy proxy with reconstruction error measurement.

**Depends on:** V4-D (benchmark tool for measurement).

1. Per-head reconstruction error at multiple ratios
2. Swap-based budget optimizer
3. Optional behind `use_influence_budgets` flag

**Files:** llama-kv-compact-budget.cpp, llama-kv-compact-pipeline.cpp
**Risk:** Medium. Expensive.

---

### Phase V4-J: IMROPE Support (3 days)

**Goal:** Enable Qwen3.5, Qwen2-VL, GLM4.

**Depends on:** V4-C (hybrid validation succeeds), V4-D (PPL measurement tool).

1. IMROPE detection, ext_x/ext_y preservation
2. No position remapping for IMROPE
3. Test on Qwen3.5-35B-A3B

**Files:** llama-kv-compacted-prefix.h, llama-kv-compacted-prefix-exec.cpp, llama-kv-compact-pipeline.cpp
**Risk:** HIGH. Open question: block until upstream provides test vectors?

---

## PART 8: PRIORITY ORDER AND TIMELINE

```
Week 1:
  V4-A  Crash guards + score modes        [0.5 day] -- safety first
  V4-C  Hybrid model validation            [0.5 day] -- Qwen3-30B-A3B unlock
  V4-N  SWA validation                     [0.5 day] -- Gemma 3 unlock (alongside V4-C)
  V4-D  Benchmark tool (PPL)               [1 day]   -- answers "does beta matter?"
  V4-G  Context-prefill server hook         [0.5 day] -- zero-cost Q capture

Week 2:
  V4-B  Graduate beta (conditional on PPL)  [0.5 day] -- only if V4-D shows win
  V4-E  Public C API                        [1 day]   -- unblocks COT
  V4-F  Auto-compaction                     [1 day]   -- transparent context extension
  V4-H  OMP timeout fallback                [1 day]   -- unblocks self_study/on_policy testing (Metal kernel deferred)

Week 3:
  V4-I  Influence-curve budgets             [2 days]  -- quality at high compression
  V4-J  IMROPE support                      [3 days]  -- Qwen3.5-35B-A3B support (HIGH RISK)

Deferred:
  GAP-H  Optimization-based compaction      -- research, not production
  GAP-I  OMP full-attention evaluation      -- marginal gain
  GAP-O  MLA support                        -- fundamentally different architecture (DeepSeek V3/R1 full)
```

**Dependency chain:**
- V4-A before V4-B (crash guards before beta graduation)
- V4-D before V4-B (PPL data before graduation decision)
- V4-D before V4-I (PPL tool for measurement)
- V4-D before V4-J (PPL tool for IMROPE validation)
- V4-C before V4-J (hybrid must work before adding IMROPE)
- V4-H before self_study/on_policy benchmarking (OMP timeout blocks server)

---

## PART 9: CONFIDENCE ASSESSMENT

| Gap ID | Confidence | Evidence |
|--------|-----------|----------|
| GAP-A (beta not default) | 100% | V1 uses `select`, confirmed in benchmark 3f357af6 |
| GAP-B (entropy vs influence) | 100% | Only entropy in budget.cpp, no reconstruction error |
| GAP-C (no MAX/MEAN/pooling) | 100% | Enum has only SUM+RMS (select.h:10-13, verified by R1) |
| GAP-D (no public API) | 100% | grep `LLAMA_API.*compact` returns nothing |
| GAP-E (no auto-compact) | 100% | No retry path in server, no threshold trigger |
| GAP-F (no IMROPE) | 100% | CLAUDE.md lists unsupported, no detection in compact code |
| GAP-G (hybrid untested) | 100% | Code exists but no test evidence |
| GAP-H (no optimization) | 100% | No gradient-based solver |
| GAP-I (no OMP-full) | 100% | OMP uses partition function, not full attention MSE |
| GAP-J (no PPL bench tool) | 100% | No tools/kv-compact-bench/ directory |
| GAP-K (no crash guards) | 100% | 114x crash diagnosed |
| GAP-L (no context-prefill hook) | 100% | prefill-q.h:9-10 says "server integration needed" |
| GAP-M (OMP >600s timeout) | 100% | Benchmark: >600s on 30B at 2x with 5K tokens |
| GAP-N (SWA untested) | 100% | get_kv_cache_base() handles iSWA but no test evidence |
| GAP-O (MLA unsupported) | 100% | Fundamentally different architecture, not standard KV |

**Previously misidentified gaps (8 items from V1, corrected in V3):**
- Beta, RMS, OMP, self-study, on-policy sequential, chunked, spectral ridge, drop-key — all IMPLEMENTED

**V3→V4 corrections (reviewer-driven):**
- ~~"max/sum/RMS" aggregation~~ → only SUM+RMS (R1 Major-1)
- ~~"10 server methods"~~ → 9 methods (R1 Minor-2)
- ~~"hook into FAILED_PREPARE retry"~~ → retry must be built (R1 Minor-3)
- ~~"100% complete"~~ → see below

**Confidence this is the complete gap list: ~98%.** Two minor unlisted items found by reviewers (context-prefill, MAX aggregation) — both now included. Remaining uncertainty: paper may have additional configuration parameters in Table 1 or appendices not individually audited. Random query generation (paper Section 3.1) intentionally excluded — worst-performing method in paper ablation.

**Review status:**
- V4: PASS from both reviewers. 3 shared Minor findings (GAP-L memory spec, V4-B PPL criterion, V4-H/Week 2 scope) — all incorporated.
- V5: PASS from both reviewers. R1: 1 Minor (C_v Tier 3 mislabeled as "pseudoinverse" — fixed to "aggressive Cholesky"). R2: 1 Minor (GAP-N missing known Gemma-3 load issue — added). 20+ spot-checks verified across all review rounds. Zero misidentifications. All 14 prior findings (11 V3→V4 + 3 V4→V5) confirmed resolved.
