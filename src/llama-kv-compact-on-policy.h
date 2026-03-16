#pragma once

#include "llama.h"

#include <cstdint>

class llama_kv_cache;
struct llama_context;
struct llama_kv_compact_pipeline_stats;

// Configuration for iterative on-policy refinement (Phase 8).
//
// n_on_policy_passes counts on-policy re-solve passes beyond the initial
// K-as-Q fit:
//   0: K-as-Q solver only (equivalent to fit_from_live_kv)
//   1: existing 2-pass (K-as-Q + one on-policy pass)
//   2+: iterative refinement with quality gate
struct llama_kv_compact_on_policy_config {
    uint32_t n_on_policy_passes       = 1;
    uint32_t n_generate               = 256;
    uint32_t max_queries              = 256;
    uint32_t max_queries_per_kv_head  = 1024;
    int      nnls_iters               = 2;
    float    lambda                   = 1e-6f;
    float    quality_min_improvement  = 0.005f;  // stop if relative residual improvement < 0.5%
};

// Iterative on-policy compaction (Phase 8).
//
// Pass 0: K-as-Q solver (fit_from_live_kv), records baseline residual.
// For each on-policy pass:
//   1. Enable compacted execution
//   2. Generate continuation with Q-capture
//   3. Re-solve all layers with captured Q
//   4. Quality gate: stop if residual didn't improve by threshold
//
// Does NOT call compacted_prefix_reclaim_live_kv() — caller's responsibility.
bool llama_kv_compact_iterative_on_policy_from_live_kv(
        struct llama_context * ctx,
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats = nullptr,
        llama_pos p0 = 0,
        const llama_kv_compact_on_policy_config & config = {});

// Configuration for per-layer sequential on-policy (Phase 8, experimental).
struct llama_kv_compact_sequential_config {
    uint32_t n_generate_q             = 64;
    uint32_t max_queries_per_kv_head  = 1024;
    int      nnls_iters               = 2;
    float    lambda                   = 1e-6f;
};

// Per-layer sequential on-policy compaction (Phase 8, experimental).
//
// After initial K-as-Q solve, iterates over compacted layouts and refits
// each layer using Q captured from generation with the latest compacted state.
// Each generation sees the previously refitted layers, creating sequential
// dependency.
//
// Does NOT call compacted_prefix_reclaim_live_kv() — caller's responsibility.
bool llama_kv_compact_sequential_on_policy_from_live_kv(
        struct llama_context * ctx,
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats = nullptr,
        llama_pos p0 = 0,
        const llama_kv_compact_sequential_config & config = {});
