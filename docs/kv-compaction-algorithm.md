# KV Compaction Algorithm

## Overview

KV compaction reduces the active prefix length of an attention cache by replacing a long prefix with a smaller learned representation that approximately preserves attention behavior for future queries. This implementation follows the Attention Matching formulation from arXiv:2602.16284 and supports multiple query-generation and selection modes.

The runtime keeps three logical regions:

1. An uncompacted prefix that must remain verbatim, such as BOS or chat-template tokens.
2. A compacted prefix represented by per-layer compacted K, V, and beta payloads.
3. A live suffix that remains in the ordinary KV cache.

The compacted prefix is executed as an additional attention source. It is never exposed as a public API contract; all interfaces remain internal to the runtime.

## Algorithm Stages

### 1. Query Extraction

The solver needs a matrix of reference queries for each layer and KV head.

Supported query sources:

- Cache-key surrogates: reuse live K vectors as approximate Q vectors.
- Self-study queries: autoregressively generate continuation tokens, capture post-RoPE Q tensors through `cb_eval`, regroup Q heads into KV-head matrices, then subsample.

The self-study path is higher quality because Q and K are both represented in the same RoPE-baked space used by the real attention computation. Cache-key surrogates remain useful as a cheaper fallback.

### 2. Key Selection

For each layer and KV head, the runtime scores compactable tokens and selects a smaller set of positions.

Implemented modes:

- Top-k additive softmax scoring (`select`): fast, assigns zero beta, compatible with flash attention.
- OMP (`omp`): iterative orthogonal matching pursuit with periodic NNLS refits.
- Full solver (`fit`, `self-study`): uses the selected positions for beta and V fitting.

Selection is performed per KV head. For grouped-query attention, Q heads are regrouped into the corresponding KV head before scoring or fitting.

### 3. Beta Fitting

The beta term preserves the additive attention correction introduced by Attention Matching. The runtime fits beta in fp32 using NNLS with a regularized Cholesky solve.

Key properties:

- All solver math is fp32.
- Regularization uses `lambda = 1e-6`.
- Beta is stored in compacted-prefix payloads after fitting.
- The `select` pipeline uses zero beta by design.

### 4. Value Fitting

After beta is fixed, the runtime fits compacted V with a least-squares solve using the selected keys and reference queries. The fitted values are stored per layer and KV head in the compacted-prefix store.

### 5. Compacted Prefix Execution

Execution materializes four inputs for attention:

- `K_compact`
- `V_compact`
- `beta_compact`
- compacted causal mask columns

These tensors are prepended to the live KV path in `build_attn()`. The graph uses the same per-layer compacted payloads that were produced by the fitting pipeline.

## GQA Handling

Captured Q tensors are token-major during collection for efficient appends. Before fitting:

1. Q heads are regrouped into KV-head matrices.
2. Each KV head receives a matrix with shape `[n_queries, n_embd_head]`.
3. Query counts are optionally capped by uniform subsampling.

For non-GQA models, regrouping becomes a pass-through with one Q head per KV head.

## Pipeline Modes

### Select

- Query source: cache-key surrogates
- Selection: top-k
- Beta: zero
- V: copied from selected live rows
- Flash attention: compatible

### Fit

- Query source: cache-key surrogates
- Selection: top-k scores followed by solver fitting
- Beta: NNLS
- V: least-squares fit
- Flash attention: requires non-flash path because non-zero beta uses additive `kq_b`

### OMP

- Query source: cache-key surrogates
- Selection: OMP with periodic NNLS refits
- Beta: NNLS
- V: least-squares fit
- Flash attention: requires non-flash path

### Self-study

- Query source: autoregressive continuation with Q capture
- Selection / fitting: same solver stages as full fit mode
- Flash attention: zero-beta mode remains compatible; non-zero beta requires non-flash path

## Quality Metrics

The runtime uses two primary quality signals:

- Attention-output or continuation-logit cosine similarity for low-noise regression checks.
- Task metrics such as multiple-choice accuracy or downstream answer quality on workload benchmarks.

Benchmark plans define threshold tables per workload and compression ratio. Small-model smoke results are not treated as proof of production long-context quality.

## Flash Attention Tradeoff

The current execution paths support two regimes:

- Zero beta (`select` pipeline): flash attention compatible and fastest.
- Non-zero beta (`fit`, `omp`, `self-study`): requires additive `kq_b`, so the runtime must use the non-flash attention path.

When flash attention is requested but non-zero beta is active, the graph construction automatically falls back to the non-flash path for affected layers. This override is logged at INFO level and surfaced in the `/props` runtime summary as `compaction.flash_attn_overridden`.

A future FlashBias-style kernel, or equivalent upstream graph change, would remove this restriction and allow flash attention with non-zero beta.

## Current Limits

The implementation does not currently provide:

- M-RoPE execution support
- SWA-layer compaction
- hybrid recurrent-attention compaction
- public API guarantees for compacted-prefix internals

Those remain explicitly gated or deferred.
