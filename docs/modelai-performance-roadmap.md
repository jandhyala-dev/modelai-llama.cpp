# ModelAI llama.cpp — Performance & Optimization Roadmap

## Part 1: Current Fork Bottlenecks (Internal)

These are performance issues **inside the existing P5b code** that should be fixed before or during PR-6.

### Critical Bottlenecks

| # | Issue | Location | Impact | Fix Effort |
|---|-------|----------|--------|------------|
| **B1** | **Transposed V element-by-element extraction** — non-flash path reads one scalar at a time across non-contiguous strides | `src/llama-kv-cache.cpp:953-969` | O(head_dim) worse memory access pattern, terrible cache locality — single biggest solver bottleneck | Medium |
| **B2** | **K/V extracted twice** — once during query scoring (Phase 1), again during fitting (Phase 2) | `src/llama-kv-compact-pipeline.cpp:90-111, 131-189` | 2x unnecessary I/O and type conversion overhead | Low |
| **B3** | **Zero SIMD vectorization in solver** — all dot products, Cholesky decomposition, matrix multiply, and exp() are scalar loops | `src/llama-kv-compact-solver.cpp` (entire file, especially `dot_row` lines 10-16, normal equations lines 80-92, power iteration lines 154-206) | 4-8x slower than vectorized on Apple Silicon (ARM NEON vfmaq_f32) | Medium |
| **B4** | **No GPU solver path** — forces host-device transfer of full K/V matrices per head per layer via `ggml_backend_tensor_get()` | All solver modules | Blocks pipeline on CPU even when KV cache lives on Metal GPU | High |
| **B5** | **Graph tensors reallocated every batch** — compacted prefix K/V/beta/mask tensors created fresh each decode batch | `src/llama-graph.cpp:464-495` | Unnecessary allocation churn on every decode step | Low |

### Recommended Quick Wins

1. **Cache extracted K/V matrices** — extract once in Phase 1 (query scoring), reuse in Phase 2 (fitting). Saves 50% of extraction time.
2. **Batch V extraction** — read full V columns in one `ggml_backend_tensor_get` call, then transpose in-memory instead of element-by-element.
3. **Replace `dot_row` with NEON intrinsics** — `vfmaq_f32` for 4-wide FMA on Apple Silicon, or at minimum add `#pragma omp simd`.
4. **Cache graph tensors** — reuse compacted prefix tensors across decode batches when the compacted state hasn't changed.

---

## Part 2: Paper Gaps for 50x Compression

The MIT paper **does demonstrate 50x compression** on Qwen3-4B (QuALITY benchmark) but requires techniques the fork doesn't have yet.

**Paper:** "Fast KV Compaction via Attention Matching" — [arXiv:2602.16284](https://arxiv.org/abs/2602.16284)
**Reference implementation:** [github.com/adamzweiger/compaction](https://github.com/adamzweiger/compaction)
**Upstream tracking issue:** [ggml-org/llama.cpp#20037](https://github.com/ggml-org/llama.cpp/issues/20037)

### What 50x Actually Requires (from the paper)

| Component | Paper | Fork (P5b) | Gap |
|-----------|-------|------------|-----|
| Query extraction | Self-study (~50k queries/head, 139s on H200) — `compaction/query_generation/self_study.py` | Cache-keys surrogate (~256 queries/head) — `src/llama-kv-compact-query.cpp` | **Major** — self-study is the single most impactful quality factor (Figure 4 leave-one-out ablation) |
| Key selection | OMP with periodic refits (Algorithm 1, k=4, tau=2, 104s OMP-fast) — `compaction/algorithms/omp.py` | Top-k only (additive softmax scoring) — `src/llama-kv-compact-select.cpp` | **Major** — OMP significantly outperforms top-k at >10x compression |
| Per-head budgets | Nonuniform via precomputed sensitivity curves (Algorithm 4, Section 3.4, Figures 2/7) — `compaction/compaction_methods/per_layer_head.py` | Shared position schedule, uniform budget | **Major** — some heads tolerate 50x while others need 2x |
| Chunked compaction | KV-based chunking (~12k tokens/chunk, Section 3.5, Appendix C.3) | Single-block only | **Moderate** — needed for >8k token contexts |
| Ridge scaling | Spectral/Frobenius normalization | Fixed lambda=1e-6 with 10x escalation | **Minor** — affects numerical stability at extreme ratios |
| On-policy sequential | Compact layer l, then extract queries from modified model for layer l+1 (Section 4.2) | All layers use same pre-compaction queries | **Moderate** — slight but consistent quality improvement per paper |

### Paper Quality at Various Compression Ratios

**Qwen3-4B on QuALITY (894 questions, 50 articles, 5-7k token passages):**

| Ratio | Accuracy | Notes |
|-------|----------|-------|
| 1x (baseline) | 71.5% | Full context |
| 2x | ~71.5% | Near-lossless |
| 5x | ~70% | Minimal degradation |
| 10x | ~67% | Notable drop begins |
| 20x | ~60% | Significant — quality cliff starts |
| 50x | ~55% | ~23% absolute loss from baseline |

**LongHealth (60k-token patient records, information-dense):**

| Ratio | F1 Score | Notes |
|-------|----------|-------|
| 2x | ~68% | Good preservation |
| 10x | ~62% | Moderate degradation |
| 50x | ~35% | Severe — information-dense tasks compress poorly |

**Wall-clock breakdown (60k-token LongHealth, Gamma3-12B, single H200):**

| Phase | Time | Source |
|-------|------|--------|
| Context prefill | 7s | Table 1, Section 4 |
| Repeat-prefill | 8s | Table 1 |
| Self-study query gen | 139s | Table 1 — dominates total time |
| Highest attention key selection | 3s | Table 1 |
| OMP key selection | 565s | Table 1 |
| OMP-fast (k=4, tau=2) | 104s | Table 1 |
| NNLS beta fitting | 2.2s | Table 1 |
| Least-squares V fitting | 1.8s | Table 1 |

Key insight: **the 20x-to-50x cliff is steep**, and that's where OMP + nonuniform budgets + self-study queries make the critical difference vs simpler methods. The fork's current top-k + cache-keys approach is adequate for 2-10x but will not reach 50x at acceptable quality.

---

## Part 3: Complementary Optimizations for the Fork

### Tier 1 — High Impact, Stacks Multiplicatively with Compaction

| Technique | What | Compression | Stacking Effect | Source |
|-----------|------|-------------|-----------------|--------|
| **KV cache quantization (Q8_0/Q4_0)** | Quantize remaining KV entries after compaction | 2-4x | 50x compaction x 4x quant = **200x total** | llama.cpp [Discussion #5932](https://github.com/ggml-org/llama.cpp/discussions/5932), [Issue #6863](https://github.com/ggml-org/llama.cpp/issues/6863) |
| **KVSplit (K8V4)** | Differential quantization: 8-bit keys + 4-bit values (keys are more sensitive to quantization) | 59% memory reduction (~2.4x) | Multiplicative with compaction | [github.com/dipampaul17/KVSplit](https://github.com/dipampaul17/KVSplit) — optimized for M1/M2/M3 Metal, [HN discussion](https://news.ycombinator.com/item?id=44009321) |
| **KIVI (2-bit asymmetric KV quantization)** | Per-channel key quant, per-token value quant | 2.6x peak memory reduction | Multiplicative | [arXiv:2402.02750](https://arxiv.org/abs/2402.02750), [github.com/jy-yuan/KIVI](https://github.com/jy-yuan/KIVI) |
| **KVQuant** | Per-channel key quantization + non-uniform per-layer datatypes + dense-and-sparse quantization | 8x at 2-bit (nuq2), enables 1M context on single A100 | Multiplicative | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079), [github.com/SqueezeAILab/KVQuant](https://github.com/SqueezeAILab/KVQuant) |
| **Speculative decoding** | Draft model proposes tokens, main model verifies in batch | 1.5-2.5x throughput | Orthogonal — faster generation + compaction = smaller KV during verification | llama.cpp [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md), [Discussion #10466](https://github.com/ggml-org/llama.cpp/discussions/10466) |

### Tier 2 — Medium Impact, Improves Quality or Latency

| Technique | What | Benefit | Source |
|-----------|------|---------|--------|
| **StreamingLLM / Attention sinks** | Protect first 4 tokens that attract disproportionate attention regardless of semantics | Quality stability at high compression ratios | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453), [github.com/mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm) |
| **FlashBias** | Flash attention with additive bias support (NeurIPS 2025) | Unblocks flash attention for compacted-prefix path — currently the V0 blocker since beta requires additive `kq_b` | [arXiv:2505.12044](https://arxiv.org/abs/2505.12044), [flash-attention Issue #1219](https://github.com/Dao-AILab/flash-attention/issues/1219) |
| **CacheBlend** | Reuse precomputed KV caches for non-prefix text by selectively recomputing ~15% of critical tokens | 2.2-3.3x TTFT reduction, 2.8-5x throughput improvement for RAG workloads | [arXiv:2405.16444](https://arxiv.org/abs/2405.16444) (EuroSys 2025 Best Paper), [github.com/LMCache/LMCache](https://github.com/LMCache/LMCache) |
| **Prompt cache + compacted prefix sharing** | Compact system prompt once, share across all server slots | Amortizes compaction cost for multi-user serving | llama.cpp `cache_prompt` (default on), [Discussion #10311](https://github.com/ggml-org/llama.cpp/discussions/10311), [Discussion #15530](https://github.com/ggml-org/llama.cpp/discussions/15530) |
| **MiniKV** | Combines 2-bit quantization with adaptive pyramid-shaped KV retention across layers | 86% KV cache compression (~7x), 98.5% accuracy preserved | [arXiv:2411.18077](https://arxiv.org/abs/2411.18077) |
| **LoRC (Low-Rank Compression)** | SVD-based low-rank approximation of KV weight matrices, no retraining | 60% compression with negligible quality loss | [arXiv:2410.03111](https://arxiv.org/abs/2410.03111) |
| **ThinK (Query-Driven Channel Pruning)** | Prunes least significant key channels based on query-dependent importance | 20% additional memory reduction on top of other methods; combined with KIVI achieves 2.8x | [arXiv:2407.21018](https://arxiv.org/html/2407.21018v3) |

### Tier 3 — Research / Monitor

| Technique | What | Compression | Source |
|-----------|------|-------------|--------|
| **KVTC (KV Cache Transform Coding)** | PCA + adaptive quantization + entropy coding (ICLR 2026) | 20-40x alone | [OpenReview](https://openreview.net/forum?id=aNVKROYpLB) |
| **MLA (Multi-Head Latent Attention)** | Low-rank KV projection (DeepSeek V2/V3/R1 architecture) | 7-25x natively | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434), [arXiv:2502.14837](https://arxiv.org/abs/2502.14837), [HuggingFace blog](https://huggingface.co/blog/NormalUhr/mla-explanation) |
| **SALS (Sparse Attention in Latent Space)** | RoPE-free latent space attention (NeurIPS 2025) — insight: RoPE increases key rank, making compression lossy | 6.4x, 5.7x attention speedup | [arXiv:2510.24273](https://arxiv.org/abs/2510.24273) |
| **HCAttention** | Heterogeneous: key quantization on GPU + value offloading to CPU | 8x, extends Llama-3-8B to 4M tokens on single A100 | [arXiv:2507.19823](https://arxiv.org/abs/2507.19823) |
| **ChunkKV** | Semantic-chunk-level compression preserving linguistic structures | 26.5% throughput improvement, 8.7% precision improvement | [arXiv:2502.00299](https://arxiv.org/abs/2502.00299) |
| **Cartridges** | Gradient-based KV compression (hours to train per context) | Up to 50x but orders of magnitude slower than Attention Matching | Referenced in [arXiv:2602.16284](https://arxiv.org/abs/2602.16284) Section 4 |
| **KV-Distill** | KL-divergence distillation of KV caches into shorter representations | Up to 95% context reduction (20x) | [OpenReview](https://openreview.net/forum?id=p7vJ3wsm34) |
| **Training-Free Native Sparse Attention** | Hierarchical block-wise KV cache selection | 16x on 32K sequences, 99% accuracy | [OpenReview](https://openreview.net/forum?id=sQjYtFSEuZ) |

### Eviction-Based Methods (Complementary Baselines)

| Technique | Compression | Quality | Source |
|-----------|-------------|---------|--------|
| **H2O (Heavy-Hitter Oracle)** | 5x (20% retention) + 29x throughput improvement | Near-lossless on standard benchmarks | [arXiv:2306.14048](https://arxiv.org/abs/2306.14048) |
| **SnapKV** | 3-8x (fixed per-head budget) | Strong with adaptive per-head budgets (AdaKV) | [ACL 2024 Findings](https://aclanthology.org/2024.findings-acl.195.pdf) |
| **PyramidKV** | 8x (12% retention) | Matches full-cache accuracy | [arXiv:2406.02069](https://arxiv.org/html/2406.02069v1) |

### Combined/Stacking Results from Literature

| Combination | Result | Source |
|-------------|--------|--------|
| MiniKV (2-bit quant + pyramid eviction) | 86% compression, 98.5% accuracy | [arXiv:2411.18077](https://arxiv.org/abs/2411.18077) |
| ThinK + KIVI (channel pruning + 2-bit quant) | 2.8x memory reduction | [arXiv:2407.21018](https://arxiv.org/html/2407.21018v3) |
| Attention Matching + Summarization | 200x compression (6340->31 tokens) | [arXiv:2602.16284](https://arxiv.org/abs/2602.16284) Table 2 |
| CacheBlend + prefix caching | 100% KV cache hit rate in RAG | [arXiv:2405.16444](https://arxiv.org/abs/2405.16444) |
| KVQuant + disaggregated serving | 10M context on 8-GPU | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) |

---

## Part 4: Upstream llama.cpp Features to Leverage

### Must Sync (Correctness / Compatibility)

| Feature | Why | Reference |
|---------|-----|-----------|
| KV cache defrag fixes (June 2025) | Defrag bug can corrupt data, directly affects compaction reliability | [PR #10873](https://github.com/ggerganov/llama.cpp/pull/10873) |
| `llama_kv_cells_unified` refactor | Core data structure change — all compaction code touching KV cell iteration must adapt | [PR #11213](https://app.semanticdiff.com/gh/ggerganov/llama.cpp/pull/11213/overview), [PR #12695](https://github.com/ggml-org/llama.cpp/pull/12695) |
| SWA KV cache support | Structural changes to KV allocation and eviction logic | [PR #13194](https://github.com/ggml-org/llama.cpp/pull/13194) |
| Unified KV buffer default behavior | `kv_unified=true` now default even without flag — could change KV layout assumptions | [Issue #17450](https://github.com/ggml-org/llama.cpp/issues/17450) |
| KV cache shift/defrag correctness | Prevents data loss during defrag operations | [Issue #12253](https://github.com/ggml-org/llama.cpp/issues/12253) |

### Should Adopt (Performance)

| Feature | Benefit | Reference |
|---------|---------|-----------|
| Fused multiply-add for Q4/Q5/Q6_K | 16-28% faster prompt processing and token generation | [PR #20032](https://github.com/ggml-org/llama.cpp/pull/20032) |
| Metal mul_mv_ext for BF16/Q2_K/Q3_K | Faster Metal kernels for additional quant types | [PR #20250](https://github.com/ggml-org/llama.cpp/pull/20250) |
| High-throughput mode (virtual sequences) | Better multi-user serving with unified/split memory modes | [PR #14363](https://github.com/ggml-org/llama.cpp/pull/14363) |
| CUDA Graphs (if targeting NVIDIA) | 10-15% decode speedup (143->164 tok/s on H100 for llama-2-7b Q4_K_M) | [NVIDIA blog](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/), `GGML_CUDA_GRAPH_OPT=1` |
| Block interleaving for Q5_K (AVX512/AVX2) | Significant prompt processing speed improvement on x86 | Merged Feb 2026 upstream |

### Worth Monitoring

| Feature | Notes | Reference |
|---------|-------|-----------|
| Upstream Issue #20037 (Attention Matching) | The exact paper has been filed as upstream RFC. Potential convergence or contribution opportunity | [Issue #20037](https://github.com/ggml-org/llama.cpp/issues/20037) |
| ik_llama.cpp Q8_KV + FlashMLA | Performance-focused fork with innovations that sometimes get upstreamed (graph split mode, Q8_KV, fused MoE) | [github.com/ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) |
| MLX performance gap | MLX runs ~53% faster than llama.cpp on Apple Silicon for short context (~230 vs ~150 tok/s) | [arXiv:2511.05502](https://arxiv.org/abs/2511.05502), [Discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167) |
| Eagle-3 speculative decoding | Current SOTA speculative method, not yet in llama.cpp | [Discussion #15902](https://github.com/ggml-org/llama.cpp/discussions/15902) |
| Flash attention Metal stability | Known regressions on Metal — monitor for V1+ flash attention support | [Issue #14847](https://github.com/ggml-org/llama.cpp/issues/14847), [Discussion #15650](https://github.com/ggml-org/llama.cpp/discussions/15650) |
| NVIDIA kvpress benchmarking framework | 20+ compression method implementations as interchangeable baselines | [github.com/NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) |

### Alternative Inference Engines (Competitive Context)

| Engine | Apple Silicon Perf | Notes | Source |
|--------|-------------------|-------|--------|
| **MLX** | ~230 tok/s | 53% faster than llama.cpp on short context, native Apple framework | [arXiv:2511.05502](https://arxiv.org/abs/2511.05502) |
| **MLC-LLM** | ~190 tok/s | Good out-of-box experience, lower TTFT | [arXiv:2511.05502](https://arxiv.org/abs/2511.05502) |
| **llama.cpp** | ~150 tok/s | Broader hardware support, GGUF ecosystem, customizability | [Discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167) |
| **vLLM** | N/A (CUDA) | PagedAttention, continuous batching, 2-4x throughput | [arXiv:2309.06180](https://arxiv.org/abs/2309.06180), [vLLM blog](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) |
| **SGLang** | N/A (CUDA) | 16,215 tok/s on H100, 29% faster than vLLM | [sitepoint benchmark](https://www.sitepoint.com/ollama-vs-vllm-performance-benchmark-2026/) |

---

## Part 5: P5b Integration Plan

All items below are scoped to land **before P5b merges**. Items marked POST-P5b are documented here for planning but explicitly deferred.

### Step 1: Solver Performance Fixes (P5b scope)

These fix bottlenecks in existing P5b code. No new features, just making existing code production-viable.

**1a. Eliminate dual K/V extraction**
- File: `src/llama-kv-compact-pipeline.cpp`
- Change: Extract K/V per-head in Phase 1 (query scoring), store in temporary buffers, pass to Phase 2 (fitting) instead of re-extracting
- Why now: 2x unnecessary I/O is unacceptable for the benchmark proof gate
- Effort: Low (refactor pipeline data flow)
- Test: Existing solver + quality tests must still pass; pipeline timing should drop ~40-50%

**1b. Batch transposed V extraction**
- File: `src/llama-kv-cache.cpp`, function `compacted_prefix_copy_v_head_f32` (lines 953-969)
- Change: Read full V column per element dimension in one `ggml_backend_tensor_get` call (stride = kv_size * type_size), then scatter to output buffer — instead of one element at a time
- Why now: Element-by-element extraction is the single largest solver bottleneck
- Effort: Medium (need to handle backend buffer alignment)
- Test: Existing V extraction tests must produce identical results

**1c. NEON vectorization of hot solver loops**
- File: `src/llama-kv-compact-solver.cpp`
- Change: Replace scalar `dot_row` (lines 10-16) with ARM NEON `vfmaq_f32` 4-wide FMA. Add `#ifdef __ARM_NEON__` guard with scalar fallback.
- Why now: 4-8x speedup on Apple Silicon for the solver math that runs per-head per-layer
- Effort: Medium (dot_row is called everywhere — matrix multiply, power iteration, attention scoring, Cholesky)
- Scope limit: Only vectorize `dot_row`. Do not vectorize Cholesky pivot logic or exp() — diminishing returns.
- Test: Solver unit tests must produce identical results within fp32 tolerance

### Step 2: OMP Key Selection (P5b scope)

This is a required paper-aligned deliverable. Top-k alone will not achieve acceptable quality at compression ratios above 10x.

**2a. OMP implementation**
- Files: `src/llama-kv-compact-select.h`, `src/llama-kv-compact-select.cpp`
- Change: Add `llama_kv_compact_select_omp()` alongside existing `llama_kv_compact_select_topk()`
- Algorithm: OMP with periodic NNLS refits per Algorithm 1 in the paper (k=4, tau=2 for fast variant)
- Reference: `compaction/compaction/algorithms/omp.py` lines 1-719 in the MIT reference implementation
- Dependency: Requires `llama_kv_compact_fit_beta()` from solver (already implemented)
- Test: Add OMP unit test in `tests/test-kv-compact-solver.cpp` comparing against top-k at 4x and 8x compression

**2b. Pipeline integration**
- File: `src/llama-kv-compact-pipeline.cpp`
- Change: Add `selection_method` field to pipeline config (enum: `TOPK`, `OMP`, `OMP_FAST`)
- Default: `TOPK` (preserves current behavior; OMP can be selected per-call)
- Test: Quality test at 4x with OMP must show >= top-k cosine similarity

### Step 3: Quality Test Coverage Expansion (P5b scope)

Current tests only validate 2x compression. The P5b merge gate requires proof at higher ratios.

**3a. Multi-ratio quality test**
- File: `tests/test-kv-compact-quality.cpp`
- Change: Add test cases for 4x, 8x, and 10x compression ratios on the fixture model
- Quality thresholds: logit cosine >= 0.95 at 4x, >= 0.90 at 8x, >= 0.85 at 10x (calibrate on first run)
- Why now: P5b merge gate requires quality evidence at meaningful compression ratios

**3b. OMP vs top-k comparison test**
- File: `tests/test-kv-compact-quality.cpp`
- Change: Run same compression ratio with both top-k and OMP, report cosine delta
- Why now: Proves OMP adds value before claiming paper alignment

### Step 4: Documentation Updates (P5b scope)

**4a. Update fork-summary.md**
- File: `docs/modelai-fork-summary.md`
- Change: Lines 147-159 describe P5b deliverables as "must be implemented" — update to reflect implemented status with measured results
- Change: Line 159 "until PR-5b lands" — update to reflect that P5b solver code exists

**4b. Update adversarial-review.md**
- File: `docs/modelai-adversarial-review.md`
- Change: Finding B1 "No solver code exists" — update to reflect that solver, select, query, and pipeline modules now exist

**4c. Update kv-compaction-plan.md**
- File: `docs/modelai-kv-compaction-plan.md`
- Change: P5b section should reflect current implementation status and measured results

**4d. Update ci-policy.md**
- File: `docs/modelai-ci-policy.md`
- Change: Add P5b quality test thresholds at multiple compression ratios (not just the single 2x test)

### POST-P5b: Deferred to PR-6+

The following items are explicitly **out of P5b scope** but documented here for planning:

**PR-6a: Nonuniform Per-Head Budgets**
- Implement head sensitivity curve precomputation per Algorithm 4 (Section 3.4)
- Per-head budget allocation in pipeline orchestrator
- Requires restructuring shared position schedule to per-head schedules
- Reference: `compaction/compaction/compaction_methods/per_layer_head.py`

**PR-6b: Self-Study Query Generation**
- Implement two-phase on-policy query extraction via model forward pass
- GQA regrouping for query-head to KV-head mapping (`_attention_to_kv()`)
- Reference: `compaction/compaction/query_generation/self_study.py` lines 1-800
- Benchmark quality improvement vs cache-keys baseline at 10x and 50x

**PR-6c: Chunked Compaction**
- KV-based chunking for >8k token contexts (Section 3.5)
- RoPE phase alignment across chunks
- Chunk boundary handling in pipeline orchestrator
- Reference: Appendix C.3 of the paper

**PR-7: Stacking Optimizations**
- KV quantization (Q8_0 / Q4_0) on compacted entries — multiplicative savings
- Flash attention support via FlashBias ([arXiv:2505.12044](https://arxiv.org/abs/2505.12044)) or upstream additive bias support
- Speculative decoding integration testing with compacted KV cache
- Prompt cache interaction with compacted prefixes for multi-slot serving
- KVSplit differential quantization ([github.com/dipampaul17/KVSplit](https://github.com/dipampaul17/KVSplit))

### P5b Integration Order and Dependencies

```
Step 1a (dual extraction fix)
    |
Step 1b (batch V extraction)    Step 2a (OMP implementation)
    |                               |
Step 1c (NEON vectorization)    Step 2b (pipeline integration)
    |                               |
    +-------------------------------+
                |
        Step 3a (multi-ratio quality tests)
                |
        Step 3b (OMP vs top-k comparison)
                |
        Step 4 (documentation updates)
                |
          P5b MERGE GATE
```

Steps 1a/1b/1c and 2a/2b can proceed in parallel. Step 3 depends on both being complete. Step 4 depends on measured results from Step 3.

---

## Curated Reference Lists

### KV Cache Compression Surveys and Awesome Lists

- [Awesome-KV-Cache-Compression](https://github.com/October2001/Awesome-KV-Cache-Compression) — comprehensive paper list
- [Awesome-KV-Cache-Management](https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management) — broader management techniques
- [KV Cache Compression Survey (arXiv:2508.06297)](https://arxiv.org/pdf/2508.06297) — systematic survey
- [KV Cache Compression Benchmark (arXiv:2407.01527)](https://arxiv.org/abs/2407.01527) — cross-method comparison
- [Understanding Physics of KV Cache Compression (arXiv:2603.01426)](https://arxiv.org/abs/2603.01426) — theoretical foundations

### llama.cpp Architecture and Performance References

- [DeepWiki: llama.cpp KV Cache](https://deepwiki.com/ggml-org/llama.cpp/3.6-batch-processing-pipeline) — memory management internals
- [llama.cpp Server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) — server configuration
- [Discussion #4130](https://github.com/ggml-org/llama.cpp/discussions/4130) — parallelization/batching architecture
- [Discussion #18308](https://github.com/ggml-org/llama.cpp/discussions/18308) — optimal parallel inference parameters
- [Justine Tunney: Edge AI Just Got Faster](https://justine.lol/mmap/) — advanced mmap for LLM loading
- [Justine Tunney: LLaMA Now Goes Faster on CPUs](https://justine.lol/matmul/) — CPU matrix multiply optimization

### Press Coverage

- [VentureBeat: 50x KV Cache Compression](https://venturebeat.com/orchestration/new-kv-cache-compaction-technique-cuts-llm-memory-50x-without-accuracy-loss/) — Attention Matching coverage
- [Red Hat: vLLM or llama.cpp](https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case) — engine comparison guide
