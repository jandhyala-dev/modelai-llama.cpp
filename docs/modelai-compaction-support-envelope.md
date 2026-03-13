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

## Current Envelope (select pipeline, post solver-bugfix 3572cde3)

| Context | Max Supported Ratio | Tier | Reason |
|---------|-------------------|------|--------|
| 4K | 8x | supported | Quality 0.989-0.999 across all ratios (post bugfix, was 0.838 pre-fix) |
| 8K | 8x | supported | Quality 0.991-0.999, throughput scales with ratio |
| 16K | 8x | supported | Quality 0.973-0.999 |
| 32K | 8x | experimental | Quality 0.998+ (excellent) but decode throughput 0.8-2.6 tok/s (materialization overhead) |

## Pipeline Classification

| Pipeline | Tier | Reason |
|----------|------|--------|
| select | supported (4K-16K all ratios) | Best quality (0.973-0.999), zero solver overhead |
| solver | experimental | GQA quality degradation — cosine 0.818-0.982 on Qwen3-14B (5:1 GQA), compaction 45-283s at 4K |
| omp | experimental | Slow compaction, timeout-prone at low ratios (236s at 8x, >300s at 2x/4x) |
| self_study | experimental | Quality 0.993, compaction 3.6min at 4K (was 22min pre solver fix), decode 12.2 tok/s |

## Self-Study Root Cause (Sprint 2 Diagnostics — Resolved)

**Status:** experimental. Root cause identified and fixed (bcced551).

**Root cause:** Q/K scale mismatch. Self-study captures post-RoPE Q from
autoregressive generation via `cb_eval`. Q norms (~16) are ~1.8x smaller
than K norms (~28) due to different learned scales in W_q vs W_k projections.
Without normalization, the attention softmax peaks incorrectly and the NNLS
solver produces extreme beta weights (beta_norm ~356).

**Fix:** Per-head Q normalization — scale Q rows by `k_norm / q_norm` before
attention score computation. Diagnostics log pre-normalization values.

**Results (Qwen3-8B, 4K/4x):**
| Metric | Before fix | After fix |
|--------|-----------|-----------|
| cosine | -0.458 / 0.710 | **0.988** |
| beta_norm | 356 / 536 | 285 |
| fit_residual | 0.144 / 0.167 | 0.063 |

**Other hypotheses ruled out:**
- Tensor variant mismatch: `layers_with_q=36` (all layers captured Q correctly).
  The 7344 dim mismatches are pre-reshape 2D Qcur projections [4096,1] being
  correctly rejected; post-reshape 3D tensors [128,32,1] are accepted.
- Solver instability: fit_residual dropped from 14-17% to 6.3% after Q
  normalization, confirming the solver works correctly when given properly
  scaled inputs.

**Remaining limitation:** Pipeline overhead (~22 min at 4K) from 256-token
autoregressive generation + NNLS solver across all heads. Decode throughput
after compaction is similar to OMP, but the compaction step itself is slow.

## Cross-Model Validation (Campaign 2026-03-13, select pipeline)

| Model | GQA Ratio | 4K/2x | 4K/4x | 4K/8x | 8K/2x | 8K/4x | 8K/8x | Tier |
|-------|-----------|-------|-------|-------|-------|-------|-------|------|
| Qwen2.5-7B-Q4_K_M | 4:1 | 0.999 | 0.995 | 0.989 | 0.998 | 0.996 | 0.991 | supported |
| Qwen3-8B-Q4_K_M | 4:1 | 0.999 | 0.999 | 0.997 | 0.999 | 0.998 | 0.997 | supported |
| Qwen2.5-14B-Q4_K_M | 4:1 | 0.987 | 0.968 | 0.946 | 0.993 | 0.984 | 0.964 | supported |
| Qwen3-14B-Q4_K_M | 5:1 | 0.995 | 0.996 | 0.992 | 0.999 | 0.997 | 0.995 | supported |
| DeepSeek-R1-14B-Q4_K_M | 4:1 | 0.999 | 0.997 | 0.994 | 0.998 | 0.996 | 0.993 | supported |
| Qwen3-30B-A3B-Q4_K_M | 8:1 | 0.999 | 0.999 | 0.999 | — | — | — | supported |
| Gemma-3-12B-Q4_K_M | iSWA | — | — | — | — | — | — | blocked (model load) |
