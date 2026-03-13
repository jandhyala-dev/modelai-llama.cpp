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
| self_study | experimental | Quality proven (0.988 cosine post Q/K norm fix); pipeline overhead (~22min at 4K) limits throughput |

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

## Larger Models

| Model | Ratio | Quality | Tier |
|-------|-------|---------|------|
| Qwen3-14B-Q4_K_M | 8x | 0.990 | supported |
| Qwen3-30B-A3B-Q4_K_M | 8x | 0.999 | supported |
