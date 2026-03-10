# ModelAI llama.cpp Fork — Summary

## Purpose

This is a private product fork of `llama.cpp` maintained by COT Labs / ModelAI.

It exists to:
1. Implement KV cache compaction via Attention Matching for long-session efficiency and post-prefill speed
2. Serve as the primary local inference runtime for ModelAI (replacing Ollama-first assumptions)
3. Enable direct product control over engine releases, CI, and benchmarking

## Product Goals

### Goal 1: Long-Session Efficiency
- Store old immutable context as a compacted prefix
- Keep recent mutable context as a live suffix
- Preserve chat-template / system-prefix tokens outside the compacted block
- Expose allocated KV bytes and active KV length separately

### Goal 2: Post-Prefill Speed / Decode Throughput
- Compact only old immutable prefixes
- Reduce effective active KV length in repeated-turn inference
- Packed layout or equivalent changes so logical reduction becomes real compute reduction

## KV Compaction Algorithm (Attention Matching)

**Paper:** "Fast KV Compaction via Attention Matching" (arXiv:2602.16284)
**Reference code:** https://github.com/adamzweiger/compaction (MIT, Python)

Given a KV prefix of length T, build a compacted representation of length t << T:

1. Generate reference queries from prefill path
2. Select compacted keys (topk by attention score or OMP greedy)
3. Solve for beta via NNLS to preserve attention mass
4. Solve for compacted values via OLS to match attention output
5. Replace KV cache: T entries → t entries

Modified attention: `softmax(q @ C_k^T + beta) @ C_v`

Key facts:
- Approximate, not exact
- beta is essential (per layer / per KV head / per compacted token)
- Nonuniform budgets matter at high compression ratios
- Chat-template/BOS tokens must stay uncompacted

## V0 Support Matrix

| Category | Status |
|---|---|
| Standard causal models with `llama_kv_cache` | Supported |
| Non-flash attention path | Supported |
| Non-quantized V cache | Supported |
| Uncompacted chat-template/BOS prefix | Supported |
| Uniform budgets (default) | Supported |
| Precomputed nonuniform schedules | Supported where available |
| Flash attention path | Unsupported |
| Quantized V compaction | Unsupported |
| SWA / split-memory compaction | Unsupported |
| Hybrid recurrent+attention | Unsupported |
| M-RoPE path | Unsupported |
| Public API guarantees | Unsupported |

## Runtime Strategy

- `llama-server` is the primary runtime target for ModelAI
- Ollama becomes optional compatibility fallback
- ModelAI must own model management and process lifecycle

## Tracks

### Track A — Private Product Fork (current)
- Target repo: `jandhyala-dev/modelai-llama.cpp`
- Optimized for speed of iteration and product value
- AI-assisted implementation is allowed

### Track B — Optional Future Upstreaming
- Begins only after Track A has measured results and stable architecture
- Requires separate upstream-ready review pass
- Subject to upstream AI-contribution policy

## Constraints from llama.cpp

1. `llama_memory_t` is the real memory abstraction boundary
2. KV buffers are fixed-size allocations (logical reduction ≠ physical memory savings automatically)
3. Non-flash supports additive `kq_b`; flash does not
4. Quantized V requires flash attention
5. Position continuity and restore invariants are strict
6. Multiple memory types exist (standard KV, SWA split, hybrid recurrent+attention)
7. Beta cannot live as a flat slot scalar — dimensionality is per layer/head/token

## Governance

- `modelai-main` is the stable shipping branch
- CI is mandatory before merge into `modelai-main`
- Releases are tagged only from `modelai-main`
- ModelAI pins to exact tags or SHAs
- Every release tag records upstream base SHA and benchmark deltas
