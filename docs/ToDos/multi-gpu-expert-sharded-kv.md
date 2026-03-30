# LOW PRIORITY — Multi-GPU Expert-Sharded KV Caching

## Context

The PiKV paper (ICML 2025, arXiv:2508.06526) introduces expert-sharded KV
cache management for Mixture-of-Experts models in distributed multi-GPU
serving environments. It uses hash-based KV sharding across GPUs
(`s(t,e) = (t mod N_tok) XOR (e mod N_exp)`) and a cache-aware routing
penalty (`-lambda * log(1+miss_e)`) that couples expert selection to cache
state. Claims 2.2x inference speedup with 65% memory reduction.

## Why Not Now

This solves a different problem from our current work. Our KV compaction is
single-device attention-matching compression (arXiv:2602.16284). PiKV targets
distributed expert-aware caching for multi-GPU serving where KV entries are
sharded across devices by expert affiliation.

We flagged this as a future Opportunity E during the hybrid-aware compaction
plan (`docs/PLAN-hybrid-compaction-and-bf16.md`) but did not incorporate it
because:

1. ModelAI's current deployment target is single-device (consumer hardware).
2. Expert-sharded KV requires multi-GPU infrastructure we don't have yet.
3. The cache-aware routing modification changes expert selection behavior,
   which is a model-level change beyond the compaction pipeline scope.

## When This Becomes Relevant

- Multi-GPU serving support is added to ModelAI.
- Large MoE models (Qwen3.5-122B, Qwen3.5-397B) are targeted for production.
- KV cache memory becomes the bottleneck in multi-GPU MoE inference.

## References

- Paper: https://arxiv.org/abs/2508.06526
- Code: https://github.com/NoakLiu/PiKV
- License: MIT (compatible)
