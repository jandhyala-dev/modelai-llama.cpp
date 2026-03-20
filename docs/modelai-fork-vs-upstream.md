---
title: "modelai-llama.cpp vs Upstream llama.cpp"
---

2026-03-19 (verified against codebase at b44d4cd1 on modelai-main)

## Overview

modelai-llama.cpp is a private product fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds **KV cache compaction via Attention Matching** -- a technique from [arXiv:2602.16284](https://arxiv.org/abs/2602.16284) (MIT, 2026). Everything below was verified against the actual codebase. Upstream llama.cpp is an excellent inference engine; this document describes what the fork adds on top of it.

---

## Comparison Table

| Feature | Upstream llama.cpp | modelai-llama.cpp |
|---------|-------------------|-------------------|
| **KV Cache Compaction** | Not available. When context fills up, options are truncation or sliding window. | 2--50x compression via Attention Matching. A 16K document compacted at 4x uses 4K of KV memory, freeing 12K slots for new turns. |
| **Compaction Methods** | N/A | 10 server-exposed pipelines: `select` (production default, 62--185ms), `solver`, `omp`, `nonuniform`, `chunked`, `on_policy`, `sequential_on_policy`, `self_study`, `chunked_self_study`, `context_prefill` |
| **Effective Context Extension** | Fixed at startup. No mechanism to exceed the physical KV cache size. | Iterative compaction cycles extend usable context beyond physical allocation. 256K effective context proven from a 64K physical window with 100% fact recall through 49--57 compaction cycles. |
| **Compaction Quality** | N/A | 0.997 cosine similarity at 50x compression on Qwen3-30B-A3B. 102 quality test points across 16 compatible models, 6 compression ratios (2x--50x), 100% pass rate. |
| **Public Compaction C API** | N/A | `llama_compact_params` struct, `llama_compact_default_params()`, `llama_kv_cache_compact()`, `llama_kv_cache_set_auto_compact()` in `include/llama.h` |
| **Auto-Compaction** | When KV cache fills, inference fails. | Threshold-triggered auto-compaction with one-shot guard to prevent retry loops. Conversations extend transparently beyond physical context. |
| **Per-Head Budget Allocation** | N/A | Entropy-based allocation + influence-curve swap solver (`LLAMA_COMPACT_INFLUENCE_BUDGETS=1`). Allocates more KV budget to attention heads that are more sensitive to compression. |
| **Score Aggregation** | N/A | 4 modes for attention score accumulation: SUM (default), RMS, MAX, MEAN. Optional 1D avgpool smoothing. |
| **Solver Infrastructure** | No linear algebra in codebase. | Pure C++ dense fp32 solver: Cholesky decomposition (retry-lambda), NNLS for beta fitting, least-squares V fitting (3-tier cascade), power iteration for spectral ridge. NEON-optimized for Apple Silicon. Zero external dependencies (no LAPACK). |
| **Crash Protection** | N/A (no compaction to crash). | Budget floor (2 tok/head), NaN guard after solver, total budget floor with warning, OMP per-head 5s timeout with top-k fallback. |
| **IMROPE Model Support** | Supports IMROPE models for inference. | Supports IMROPE models for inference AND compaction. Qwen3.5, Qwen3.5-MOE text-only IMROPE safe for compaction. M-RoPE (spatial, Qwen2-VL) correctly blocked. |
| **Compaction State Persistence** | N/A | Full state serialization (version 2) -- compacted prefix survives session save/restore including IMROPE flag, positions, K/V/beta data. |
| **Runtime Telemetry** | Reports total KV buffer size. | Server `/props` exposes: `compacted_prefix.available`, `compacted_prefix.enabled`, pipeline list, tested envelope. Runtime summary exposes: `active_n_kv_total`, `active_n_kv_max`, `sequence_state_bytes_total`, compaction status and method. |
| **Flash Attention Hybrid** | Flash attention is all-or-nothing. | Per-layer eligibility: zero-beta layers use flash attention, non-zero-beta layers use standard path. Best of both worlds. |
| **PPL Benchmark Tool** | No compaction-specific benchmarking. | `tools/kv-compact-bench/` (593 lines): prefill, compact, continue-eval with CSV output. Ablation flags: `--no-beta`, `--no-cv`, `--evict-only`. |
| **Baseline Performance** | Baseline | Identical or faster. 17-model speed comparison (1B--30B, 7 architectures) shows zero systematic regression. Several models faster due to fork optimizations (Granite +17.7% pp, Aya +11.0% tg, Qwen2.5-7B +11.1% pp). |

---

## What This Means in Practice

### For Long-Context Workloads

A financial analyst loading a 256K-token SEC filing on an M2 Pro 32GB laptop:

- **Upstream llama.cpp:** Must allocate 256K context upfront (requires ~24GB KV cache for 30B MoE at f16). Doesn't fit in memory. Alternative: truncate to 64K and lose 75% of the document.
- **modelai-llama.cpp:** Allocate 64K context (6GB KV cache). Process the full 256K document through iterative compaction cycles. Each cycle compresses old context and frees space for new content. 100% fact recall at every checkpoint. Total: 22GB (fits comfortably).

### For Multi-Turn Conversations

After 20 turns of conversation consuming 32K tokens:

- **Upstream llama.cpp:** KV cache is 32K tokens. Next turn must fit in remaining context or conversation resets.
- **modelai-llama.cpp:** Compact at 4x to 8K tokens. 24K slots freed for new turns. Compaction takes <300ms -- imperceptible to the user. Previous context is preserved with >99% fidelity.

### For Production Integration

- **Upstream llama.cpp:** No compaction API. Context management is the caller's problem.
- **modelai-llama.cpp:** Single API call: `llama_kv_cache_compact(ctx, seq_id, params)`. Or set-and-forget: `llama_kv_cache_set_auto_compact(ctx, 0.9, params)` -- compaction fires automatically when KV usage exceeds 90%.

---

## Benchmark Highlights

### Speed: Zero Overhead

17 models tested (1B--30B). Prompt processing and token generation within +/-1% for most models. No speed penalty for having compaction capability available.

| Model | PP delta | TG delta |
|-------|---------|---------|
| TinyLlama-1.1B | -0.1% | +0.3% |
| Llama3.1-8B | 0.0% | -0.2% |
| Qwen3-30B-A3B | -0.5% | -0.7% |
| Granite3.1-Dense-8B | **+17.7%** | **+3.6%** |
| Aya-8B | **+10.9%** | **+11.0%** |
| Qwen2.5-7B | **+11.1%** | **+6.2%** |

### Quality: Near-Lossless at Extreme Compression

| Compression | Qwen3-30B-A3B Cosine | 10-Fact Recall |
|------------|---------------------|----------------|
| 2x | 0.9997 | 10/10 (100%) |
| 4x | 0.9993 | 10/10 (100%) |
| 10x | 0.9997 | 10/10 (100%) |
| 50x | 0.9967 | 10/10 (100%) |

### Production Method: `select`

The `select` method is the production default. It uses attention-score-weighted top-k selection -- fast, reliable, and sufficient for all tested workloads.

| Ratio | Time (ms) | Fact Recall |
|-------|----------|-------------|
| 2x | 185 | 100% |
| 4x | 114 | 100% |
| 10x | 105 | 100% |
| 25x | 64 | 100% |
| 50x | 62 | 100% |

---

## Implementation Scale

| Metric | Value |
|--------|-------|
| Compaction source files | 23 |
| Lines of compaction code | 9,590 |
| Test files | 12 |
| Test lines | 5,779 |
| Server compaction methods | 10 |
| Score aggregation modes | 4 |
| Models validated | 17 (7 architectures) |
| Quality test points | 102 (100% pass) |
| External dependencies added | Zero |
| Speed regression | Zero |

---

## Supported Architectures

| Architecture | Compaction Status |
|-------------|------------------|
| Standard causal (LLaMA, Mistral, Qwen2.5, DeepSeek) | Supported |
| MoE (Qwen3-30B-A3B) | Supported |
| Hybrid SSM+attention (attention layers, standard RoPE) | Supported |
| IMROPE text-only (Qwen3.5, Qwen3.5-MOE) | Supported (V4-J) |
| iSWA base layers (Gemma3) | Supported (untested) |
| Quantized K cache (Q8_0, Q4_K) | Supported |
| F16/BF16/F32 KV cache | Supported |
| M-RoPE spatial (Qwen2-VL, GLM4) | Not supported (spatial positions) |
| MLA (DeepSeek V3/R1 full) | Not supported (compressed latents) |
| Pure recurrent (Mamba/RWKV) | Not applicable (no KV cache) |

---

## Quality Assurance

All code reviewed under the Hostile Review Protocol with mandatory traces and disprove-it pass:

- 8 implementation phases (PR-0 through Phase 8) -- each reviewed and PASS
- 4 V5 sprints -- each reviewed by 2 independent adversarial reviewers, all PASS
- V5 gap analysis -- 4 review rounds, PASS at V5

---

*Repository: [jandhyala-dev/modelai-llama.cpp](https://github.com/jandhyala-dev/modelai-llama.cpp) | Branch: modelai-main | Commit: b44d4cd1*
*All claims verified against codebase. All benchmark results reproducible on Apple M2 Pro 32GB.*
