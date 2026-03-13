# KV Compaction Support Envelope

## Classification Tiers

| Tier | Meaning | Benchmark Gate |
|------|---------|---------------|
| supported | Quality passes at measured ratios | logit_cosine >= threshold |
| experimental | Quality or throughput may fail; not production-safe | Recorded but not gated |
| blocked | Known catastrophic failure; must not be offered | Must not appear in supported routing |

Note: throughput is tracked via `throughput_pass` but is NOT used to gate
`result.pass`. The in-run throughput delta compares state-restore baseline
vs compacted decode, which is an apples-to-oranges comparison. Goal 2
throughput is verified cross-run by comparing `compacted_decode_tok_s`
against the `pipeline=baseline` row's `baseline_decode_tok_s`.

## Current Envelope (select pipeline, Qwen3-8B+)

| Context | Max Supported Ratio | Tier | Reason |
|---------|-------------------|------|--------|
| 4K | 4x | supported up to 4x, experimental above | Quality fails at 8x (0.838 < 0.85) and 50x (0.825 < 0.85); non-monotonic quality (16x passes at 0.903) |
| 8K | 50x | supported | Quality 0.965, throughput +89% |
| 16K | 50x | supported | Quality 0.998, throughput +42% |
| 32K | — | experimental | Throughput regresses (-22% at 50x) |

## Pipeline Classification

| Pipeline | Tier | Reason |
|----------|------|--------|
| select | supported (within context envelope) | Strong quality, proven throughput |
| solver | experimental | Insufficient benchmark evidence |
| omp | experimental | Insufficient benchmark evidence |
| self_study | blocked | Catastrophic quality failure (0.11-0.71 cosine) |

## Self-Study Root Cause (Sprint 2 Diagnostics)

**Status:** blocked. Diagnostic instrumentation added (Sprint 2a-2d) but runtime
analysis pending. The following diagnostics are now available when running
self_study pipeline with stats enabled:

- `n_layers_with_q` / `n_dim_mismatches` — detect silent Q capture failures
- `q_norm_mean` / `k_norm_mean` — detect Q/K space mismatch (scale or RoPE)
- `beta_norm_mean` / `beta_sparsity` — detect degenerate solver solutions
- `fit_residual_mean` — detect solver inability to reconstruct attention

**Hypothesized root causes (to be confirmed by diagnostic run):**
1. Q/K scale mismatch — self-study Q vectors captured post-RoPE may have
   different normalization than K vectors extracted from the live cache
2. Solver instability — NNLS with 1024 queries and high-dimensional inputs
   may produce degenerate beta vectors
3. Tensor variant mismatch — dim filter may be rejecting the correct
   post-RoPE tensor while accepting an intermediate

**Next steps:** Run `PIPELINE=self_study RATIO=2 ./build/bin/test-kv-compact-longctx
-m models/test/Qwen3-8B-Q4_K_M.gguf -c 4096` and read diagnostic output.

## Larger Models

| Model | Ratio | Quality | Tier |
|-------|-------|---------|------|
| Qwen3-14B-Q4_K_M | 8x | 0.990 | supported |
| Qwen3-30B-A3B-Q4_K_M | 8x | 0.999 | supported |
