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

## Larger Models

| Model | Ratio | Quality | Tier |
|-------|-------|---------|------|
| Qwen3-14B-Q4_K_M | 8x | 0.990 | supported |
| Qwen3-30B-A3B-Q4_K_M | 8x | 0.999 | supported |
