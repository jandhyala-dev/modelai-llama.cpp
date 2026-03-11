# P5b Benchmark Results — Model Validation

**Date:** 2026-03-11
**Branch:** `modelai-main` (post P5b merge at `72d4420e`)
**Build:** 8301 (32d1b11c) — AppleClang 17.0.0, Darwin arm64
**Hardware:** Apple M2 Pro, 32 GB unified memory
**Pipeline:** selection-only (position-based, zero beta) — default for GQA models

## Quality: Logit Cosine Similarity

Selection-only pipeline, non-flash attention, 320-token real-text prefix, 256-token compaction window.

| Model | Params | Quant | Arch | 2x (>=0.95) | 4x (>=0.90) | 8x (>=0.85) |
|-------|--------|-------|------|:-----------:|:-----------:|:-----------:|
| Qwen3-14B | 14B | Q4_K_M | qwen3 (GQA 4:1) | **0.956** | 0.895 | — |
| Qwen3-VL-8B | 8B | Q4_K_M | qwen3 (GQA 4:1) | **0.963** | **0.963** | **0.963** |
| Qwen3-8B | 8B | Q4_K_M | qwen3 (GQA 4:1) | **0.996** | **0.993** | **0.990** |
| DeepSeek R1 14B | 14B | Q3_K_M | llama (GQA 4:1) | **0.990** | **0.981** | **0.980** |
| GPT OSS 20B | 20B | Q3_K_M | gpt_oss (iSWA) | **0.987** | **0.987** | **0.987** |
| Qwen 2.5 14B | 14B | Q2_K | qwen2 (GQA 4:1) | **0.988** | **0.958** | **0.956** |
| Qwen 2.5 7B | 7B | Q3_K_M | qwen2 (MHA 1:1) | **0.976** | **0.963** | **0.956** |
| Gemma 3 12B | 12B | Q4_K_M | gemma3 | 0.859 | — | — |
| stories15M | 24M | Q4_0 | llama (MHA 1:1) | **1.000** | **1.000** | **1.000** |

### Notes

- Qwen3-14B narrowly misses the 4x threshold (0.895 vs 0.90); 2x is the production sweet spot.
- GPT OSS 20B and Qwen3-VL-8B show flat cosine across compression ratios — likely due to iSWA / architecture-specific KV behavior.
- Gemma 3 12B does not work well with the selection-only pipeline (0.859 at 2x). Not recommended for KV compaction.
- Full solver pipeline degrades GQA models (Qwen3-14B: -0.106 cosine); selection-only is the correct path.

## Performance: Decode Throughput (tok/s)

Compacted-prefix perf harness, 320-token seed, 32-token decode, 3 iterations, n_kv 512 -> 256.

| Model | Baseline tok/s | After Compaction tok/s | Speedup |
|-------|:-:|:-:|:-:|
| Qwen3-14B | 12.1 | 13.9 | **+15%** |
| Qwen3-VL-8B | 29.2 | 30.2 | +3% |
| Qwen3-8B | 17.6 | 21.2 | **+20%** |
| DeepSeek R1 14B | 10.7 | 12.6 | **+18%** |
| GPT OSS 20B | 49.3 | 49.8 | +1% |
| Gemma 3 12B | 19.2 | 19.2 | 0% |

### Notes

- Standard causal models (Qwen3, DeepSeek R1) show 15-20% speedup from active n_kv reduction.
- iSWA models (GPT OSS 20B) show minimal speedup because compaction only applies to non-SWA layers.
- Gemma 3 shows no speedup (consistent with quality failure).

## Infrastructure Tests

| Model | Pack | State-Restore | Solver Unit |
|-------|:----:|:-------------:|:-----------:|
| Qwen3-14B | PASS | PASS | N/A |
| Qwen3-VL-8B | PASS | PASS | N/A |
| Qwen3-8B | PASS | PASS | N/A |
| DeepSeek R1 14B | PASS | PASS | N/A |
| GPT OSS 20B (iSWA) | PASS | PASS | N/A |
| Gemma 3 12B | PASS | PASS | N/A |
| stories15M | PASS | PASS | PASS |

## Recommended ModelAI Production Stack

| Role | Model | Size | Port |
|------|-------|------|------|
| Primary reasoning | Qwen3-14B Q4_K_M | 8.4 GB | 8090 |
| Embeddings | BGE-M3 F16 | 1.1 GB | 8091 |
| Vision (on-demand) | Qwen3-VL-8B Q4_K_M + mmproj | 5.8 GB | — |

Total memory footprint: ~15.3 GB, leaving ~17 GB for KV cache and context windows on 32 GB Apple Silicon.

## Model Files

All models stored at `models/test/` (git-ignored):

| File | Size | Source |
|------|------|--------|
| Qwen3-14B-Q4_K_M.gguf | 8.4 GB | Qwen/Qwen3-14B-GGUF |
| Qwen3VL-8B-Instruct-Q4_K_M.gguf | 4.7 GB | Qwen/Qwen3-VL-8B-Instruct-GGUF |
| mmproj-Qwen3VL-8B-Instruct-F16.gguf | 1.1 GB | Qwen/Qwen3-VL-8B-Instruct-GGUF |
| Qwen3-8B-Q4_K_M.gguf | 4.7 GB | Qwen/Qwen3-8B-GGUF |
| deepseek-r1-distill-qwen-14b-q3_k_m.gguf | 6.8 GB | — |
| openai-gpt-oss-20b-q3_k_m.gguf | 11 GB | — |
| gemma-3-12b-it-Q4_K_M.gguf | 6.8 GB | ggml-org/gemma-3-12b-it-GGUF |
| qwen2.5-14b-instruct-q2_k (split) | 5.3 GB | — |
| qwen2.5-7b-instruct-q3_k_m.gguf | 3.5 GB | — |
| bge-m3-f16.gguf | 1.1 GB | — |
| stories15M-q4_0.gguf | 18 MB | CI smoke test |
