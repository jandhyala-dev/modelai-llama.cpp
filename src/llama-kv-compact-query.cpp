#include "llama-kv-compact-query.h"

#include "llama-kv-cache.h"

#include <algorithm>

bool llama_kv_compact_extract_cache_key_queries(
        const llama_kv_cache & kv,
        llama_seq_id seq_id,
        int32_t il,
        uint32_t head_kv,
        const std::vector<llama_pos> & positions,
        const llama_kv_compact_query_params & params,
        llama_kv_compact_matrix & queries_out) {
    if (positions.empty()) {
        queries_out = {};
        return false;
    }

    std::vector<llama_pos> sampled;
    const uint32_t limit = std::min<uint32_t>(params.max_queries, positions.size());
    sampled.reserve(limit);
    if (limit == positions.size()) {
        sampled = positions;
    } else {
        for (uint32_t i = 0; i < limit; ++i) {
            const size_t idx = (size_t(i) * positions.size()) / limit;
            sampled.push_back(positions[std::min(idx, positions.size() - 1)]);
        }
        std::sort(sampled.begin(), sampled.end());
        sampled.erase(std::unique(sampled.begin(), sampled.end()), sampled.end());
    }

    std::vector<float> q;
    if (!kv.compacted_prefix_copy_k_head_f32(il, seq_id, head_kv, sampled, q)) {
        queries_out = {};
        return false;
    }

    llama_compacted_prefix_layer_layout layout;
    if (!kv.compacted_prefix_layer_layout_for_solver(il, layout)) {
        queries_out = {};
        return false;
    }

    queries_out.resize(sampled.size(), layout.n_embd_head_k);
    queries_out.data = std::move(q);
    return true;
}
