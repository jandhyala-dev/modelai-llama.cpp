# modelai-llama.cpp — Highlights

One-page summary for the GitHub README.

---

## What It Is

Production fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) adding **KV cache compaction** via Attention Matching ([arXiv:2602.16284](https://arxiv.org/abs/2602.16284)).

Instead of truncating or evicting old context, compaction compresses the KV cache into a smaller learned representation that preserves attention behavior. The model produces near-identical outputs after compaction.

---

## Key Numbers

| Metric | Value |
|--------|-------|
| **Quality** | 0.946–0.999 logit cosine similarity across 15 models (select pipeline) |
| **Decode speedup** | Up to +63% at 8K context (Qwen3-8B, 8x compression) |
| **Max effective context** | 256K tokens from 64K physical KV cache (100% fact recall) |
| **Baseline overhead** | Zero — fork matches upstream decode speed within 2% |
| **Compaction speed** | 62–518ms depending on context length and ratio |
| **Models validated** | 17 tested, 15 pass quality gate |
| **Architectures supported** | Standard, iSWA, hybrid SSM+attention, hybrid-iSWA, IMROPE |

---

## How Compaction Works

```
Fill 64K KV cache → Compact at 4x → 16K compacted + 48K free → Fill again → Repeat
Each cycle advances position IDs by ~48K
256K effective context from 64K physical allocation
```

1. **Select** key positions by attention score (62–185ms)
2. **Fit** optional additive bias (beta) and compressed values (V)
3. **Prepend** compacted tensors to live KV during attention
4. **Reclaim** freed KV slots for new tokens

The `select` pipeline (production default) uses zero beta, enabling flash attention compatibility. Quality: 0.997–0.999 for Qwen3-8B across all tested configurations.

---

## Benchmark Highlights

### Decode Speed After Compaction

| Model | Context | Ratio | Cosine | Compacted (t/s) | Baseline (t/s) | Speedup |
|-------|---------|-------|--------|-----------------|----------------|---------|
| Qwen3-8B | 8K | 8x | 0.997 | 9.3 | 5.7 | **+63%** |
| Qwen3-30B-A3B | 4K | 8x | 0.999 | 20.2 | 13.5 | **+50%** |
| Qwen3-8B | 4K | 8x | 0.997 | 13.8 | 10.7 | **+29%** |
| Qwen3-8B | 16K | 8x | 0.999 | 5.3 | 4.8 | **+10%** |

### Quality Heat Map (Select Pipeline)

| Model | 4K/2x | 4K/4x | 4K/8x | 8K/2x | 8K/4x | 8K/8x |
|-------|-------|-------|-------|-------|-------|-------|
| Qwen3-8B | 0.999 | 0.999 | 0.997 | 0.999 | 0.998 | 0.997 |
| Qwen3-30B-A3B | 0.999 | 0.999 | 0.999 | — | — | — |
| Qwen3-14B | 0.995 | 0.996 | 0.992 | 0.999 | 0.997 | 0.973 |
| DeepSeek-R1-14B | 0.999 | 0.998 | 0.993 | — | — | — |

### 256K Iterative Context Extension

64K physical KV cache → 49-57 compaction cycles → 256K effective tokens → **100% 10-fact recall at every checkpoint**

### Fork vs Upstream vs Ollama (Baseline, No Compaction)

| Model | modelai (t/s) | upstream (t/s) | ollama (t/s) |
|-------|--------------|----------------|-------------|
| Qwen3-8B | 30.4 | 30.1 | 29.2 |
| Qwen3-14B | 17.1 | 17.4 | 16.7 |
| Qwen3-30B-A3B | 52.8 | 51.8 | 46.7 |

Zero regression from compaction code. MoE model (30B-A3B) is 13% faster than Ollama at baseline.

---

## What We Built (Implementation Scale)

- **7 compaction pipelines:** select, solver, OMP, self-study, chunked, on-policy, sequential
- **Pure C++ solver:** NNLS beta fitting, least-squares V, Cholesky decomposition, NEON-optimized — zero external dependencies
- **Metal GPU acceleration:** Attention score + XtX assembly on Apple Silicon
- **Public C API:** `llama_kv_cache_compact()`, `llama_kv_cache_set_auto_compact()`
- **Full state persistence:** Compacted prefix survives save/restore (version 2 serialization)
- **Auto-compaction:** Threshold-triggered with one-shot guard
- **Per-layer flash hybrid:** Zero-beta layers use flash attention, non-zero use standard
- **5 architecture support:** Standard, iSWA (base layers), hybrid SSM+attention, hybrid-iSWA, IMROPE text-only
- **Quantized K cache:** Q8_0, Q4_K extraction with block-aligned dequantization
- **Crash protection:** Budget floor, NaN guard, OMP timeout with fallback, memory budget guard

---

## Quality Assurance

- **29 fix commits** addressing 29 distinct bugs (18 Critical/Major)
- **17 models tested** — 15 pass the 0.95 cosine quality gate
- **51 CI-gated C++ tests**, 64 total (48 main-label + 3 model-label + 13 server pytests)
- **8 engine test tiers** — pytests, API snapshots, perf regression, quality gate, Windows CI, stress tests, contract tests, live dashboard
- **6 upstream KV cache changes** verified compatible before shipping
- **RPC RCE security patch** synced same-day from upstream
- **Weekly upstream sync** — automated Saturday 2PM PDT via CI, 3-branch model
- **Every merge gated** by 13-section adversarial review with concrete traces

---

## Supported Models

**Validated (cosine >= 0.950):**
Qwen3-8B, Qwen3-14B, Qwen3-30B-A3B, Qwen3-Coder-30B-A3B-1M, Qwen2.5-7B, Qwen2.5-14B, DeepSeek-R1-14B, DeepSeek-R1-1.5B, Granite3.1-Dense-8B, Aya-Expanse-8B, Llama-3.2-3B, SmolLM2-1.7B, stories15M, Yi-1.5-6B, Phi-3-mini-4k

**Unsupported:**
Gemma3-12B (upstream SWA bug), Qwen2-VL (M-RoPE spatial), GLM4 (M-RoPE), Mamba/RWKV (pure recurrent, no KV cache)

---

## Documentation

| Document | Description |
|----------|-------------|
| [CHANGELOG](CHANGELOG.md) | Full implementation history: V0 → V1 → V2 → V5 → Phase 8 |
| [DESIGN-DECISIONS](DESIGN-DECISIONS.md) | 13 architecture decisions with rationale |
| [BUGS-AND-FIXES](BUGS-AND-FIXES.md) | All 29 bugs: root cause, fix, commit SHA |
| [UPSTREAM-SYNC](UPSTREAM-SYNC.md) | Sync process, CI workflows, test tiers |
| [kv-compaction-algorithm](kv-compaction-algorithm.md) | Algorithm stages: selection, fitting, execution |
| [kv-compaction-integration](kv-compaction-integration.md) | File map and architecture support matrix |
| [benchmark-fork-vs-upstream](benchmark-fork-vs-upstream.md) | 3-way speed comparison |
| [paper-comparison](paper-comparison-2602.16284.md) | Fork vs arXiv:2602.16284 reference |

---

## Links

- **Paper:** [arXiv:2602.16284](https://arxiv.org/abs/2602.16284) — Fast KV Compaction via Attention Matching (Zweiger et al., MIT Han Lab)
- **Upstream:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- **Upstream RFC:** [ggml-org/llama.cpp#20037](https://github.com/ggml-org/llama.cpp/issues/20037)
- **License:** MIT
