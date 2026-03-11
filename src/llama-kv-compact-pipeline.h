#pragma once

#include "llama.h"

class llama_kv_cache;

struct llama_kv_compact_pipeline_stats {
    double query_generation_time_ms = 0.0;
    double solver_time_ms = 0.0;
    uint32_t n_prefix_tokens = 0;
    uint32_t n_selected_tokens = 0;
};

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
