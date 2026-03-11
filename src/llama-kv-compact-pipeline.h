#pragma once

#include "llama.h"

class llama_kv_cache;

struct llama_kv_compact_pipeline_stats {
    double query_generation_time_ms = 0.0;
    double solver_time_ms = 0.0;
    uint32_t n_prefix_tokens = 0;
    uint32_t n_selected_tokens = 0;
};

// Full solver pipeline: selection + beta fitting + V fitting.
// Uses cache-keys-as-queries for scoring and solver optimization.
bool llama_kv_compact_fit_from_live_kv(
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats = nullptr,
        llama_pos p0 = 0,
        uint32_t max_queries = 256,
        int nnls_iters = 64,
        float lambda = 1e-6f);

// Selection-only pipeline: keeps earliest target_tokens positions from the
// prefix with original K/V values and zero beta.  No solver fitting is
// performed.  This mode avoids quality degradation caused by surrogate
// cache-key queries on GQA architectures and serves as the default v0
// compaction path until real query extraction is implemented.
bool llama_kv_compact_select_from_live_kv(
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats = nullptr,
        llama_pos p0 = 0);
