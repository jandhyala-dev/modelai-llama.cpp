#pragma once

#include "llama-kv-compact-solver.h"
#include "llama.h"

#include <cstdint>
#include <vector>

class llama_kv_cache;

struct llama_kv_compact_query_params {
    uint32_t max_queries = 256;
};

bool llama_kv_compact_extract_cache_key_queries(
        const llama_kv_cache & kv,
        llama_seq_id seq_id,
        int32_t il,
        uint32_t head_kv,
        const std::vector<llama_pos> & positions,
        const llama_kv_compact_query_params & params,
        llama_kv_compact_matrix & queries_out);
