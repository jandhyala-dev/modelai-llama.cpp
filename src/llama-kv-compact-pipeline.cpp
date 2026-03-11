#include "llama-kv-compact-pipeline.h"

#include "llama-kv-cache.h"
#include "llama-kv-compact-query.h"
#include "llama-kv-compact-select.h"
#include "llama-kv-compact-solver.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace {

bool gather_matrix_rows(
        const std::vector<float> & src,
        uint32_t src_rows,
        uint32_t cols,
        const std::vector<uint32_t> & row_indices,
        llama_kv_compact_matrix & dst) {
    if (src.size() != size_t(src_rows) * cols) {
        return false;
    }
    dst.resize(row_indices.size(), cols);
    for (size_t i = 0; i < row_indices.size(); ++i) {
        const uint32_t src_row = row_indices[i];
        if (src_row >= src_rows) {
            return false;
        }
        std::memcpy(dst.row(i), src.data() + size_t(src_row) * cols, size_t(cols) * sizeof(float));
    }
    return true;
}

void write_compacted_payload(
        std::vector<uint8_t> & dst,
        ggml_type type,
        uint32_t n_head_kv,
        uint32_t n_tokens,
        uint32_t head,
        uint32_t dim,
        const llama_kv_compact_matrix & rows) {
    GGML_ASSERT(rows.rows == n_tokens);
    GGML_ASSERT(rows.cols == dim);

    auto from_float = ggml_get_type_traits(type)->from_float_ref;
    GGML_ASSERT(from_float != nullptr);

    const size_t token_bytes = ggml_row_size(type, dim);
    for (uint32_t token = 0; token < n_tokens; ++token) {
        void * dst_ptr = dst.data() + (size_t(head) * n_tokens + token) * token_bytes;
        from_float(rows.row(token), dst_ptr, dim);
    }
}

} // namespace

bool llama_kv_compact_fit_from_live_kv(
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats,
        llama_pos p0,
        uint32_t max_queries,
        int nnls_iters,
        float lambda) {
    if (seq_id < 0 || target_tokens == 0 || live_suffix_pos0 <= p0) {
        return false;
    }

    std::vector<llama_pos> prefix_positions;
    if (!kv.compacted_prefix_seq_positions(seq_id, p0, live_suffix_pos0, prefix_positions)) {
        return false;
    }
    if (prefix_positions.empty()) {
        return false;
    }

    const auto & layouts = kv.get_compacted_prefix()->get_layouts();
    if (layouts.empty()) {
        return false;
    }

    const uint32_t n_prefix_tokens = prefix_positions.size();
    const uint32_t n_selected = std::min<uint32_t>(target_tokens, n_prefix_tokens);
    std::vector<float> aggregate_scores(n_prefix_tokens, 0.0f);

    const auto t_query_start = std::chrono::steady_clock::now();
    for (const auto & layout : layouts) {
        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            std::vector<float> full_k_data;
            if (!kv.compacted_prefix_copy_k_head_f32(int32_t(layout.layer_id), seq_id, head, prefix_positions, full_k_data)) {
                return false;
            }

            llama_kv_compact_matrix full_k(n_prefix_tokens, layout.n_embd_head_k);
            full_k.data = std::move(full_k_data);

            llama_kv_compact_matrix queries;
            if (!llama_kv_compact_extract_cache_key_queries(
                        kv, seq_id, int32_t(layout.layer_id), head, prefix_positions,
                        llama_kv_compact_query_params{ max_queries }, queries)) {
                return false;
            }

            llama_kv_compact_accumulate_attention_scores(queries, full_k, aggregate_scores);
        }
    }
    const auto t_query_end = std::chrono::steady_clock::now();

    const std::vector<uint32_t> selected_local = llama_kv_compact_select_topk(aggregate_scores, n_selected);
    std::vector<llama_pos> selected_positions;
    selected_positions.reserve(selected_local.size());
    for (uint32_t idx : selected_local) {
        selected_positions.push_back(prefix_positions[idx]);
    }

    const llama_pos seq_max = kv.seq_pos_max(seq_id);
    const uint32_t logical_token_count = seq_max >= 0 ? uint32_t(seq_max + 1) : uint32_t(live_suffix_pos0);
    if (!kv.compacted_prefix_configure(seq_id, logical_token_count, selected_positions, live_suffix_pos0)) {
        return false;
    }

    auto * seq = kv.get_compacted_prefix()->get_seq(seq_id);
    if (seq == nullptr || !seq->enabled || seq->layers.size() != layouts.size()) {
        return false;
    }

    const auto t_solver_start = std::chrono::steady_clock::now();
    const llama_kv_compact_solver_opts solver_opts = {
        /* lambda          = */ lambda,
        /* nnls_iters      = */ nnls_iters,
        /* nnls_lower_bound= */ 1e-12f,
        /* nnls_upper_bound= */ 20.0f,
    };

    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        auto & dst_layer = seq->layers[li];

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            std::vector<float> full_k_data;
            if (!kv.compacted_prefix_copy_k_head_f32(int32_t(layout.layer_id), seq_id, head, prefix_positions, full_k_data)) {
                return false;
            }
            llama_kv_compact_matrix full_k(n_prefix_tokens, layout.n_embd_head_k);
            full_k.data = std::move(full_k_data);

            std::vector<float> full_v_data;
            if (!kv.compacted_prefix_copy_v_head_f32(int32_t(layout.layer_id), seq_id, head, prefix_positions, full_v_data)) {
                return false;
            }
            llama_kv_compact_matrix full_v(n_prefix_tokens, layout.n_embd_head_v);
            full_v.data = std::move(full_v_data);

            llama_kv_compact_matrix queries;
            if (!llama_kv_compact_extract_cache_key_queries(
                        kv, seq_id, int32_t(layout.layer_id), head, prefix_positions,
                        llama_kv_compact_query_params{ max_queries }, queries)) {
                return false;
            }

            llama_kv_compact_matrix compacted_k;
            if (!gather_matrix_rows(full_k.data, full_k.rows, full_k.cols, selected_local, compacted_k)) {
                return false;
            }

            std::vector<float> beta;
            if (!llama_kv_compact_fit_beta(queries, full_k, compacted_k, solver_opts, beta, nullptr)) {
                return false;
            }

            if (layout.n_embd_head_v > 0) {
                llama_kv_compact_matrix compacted_v;
                if (!llama_kv_compact_fit_values(queries, full_k, full_v, compacted_k, beta, solver_opts, compacted_v)) {
                    return false;
                }
                write_compacted_payload(dst_layer.v_data, layout.type_v, layout.n_head_kv, n_selected, head, layout.n_embd_head_v, compacted_v);
            }

            write_compacted_payload(dst_layer.k_data, layout.type_k, layout.n_head_kv, n_selected, head, layout.n_embd_head_k, compacted_k);
            for (uint32_t token = 0; token < n_selected; ++token) {
                dst_layer.beta_data[size_t(head) * n_selected + token] = beta[token];
            }
        }
    }
    const auto t_solver_end = std::chrono::steady_clock::now();

    if (stats) {
        stats->query_generation_time_ms = std::chrono::duration<double, std::milli>(t_query_end - t_query_start).count();
        stats->solver_time_ms = std::chrono::duration<double, std::milli>(t_solver_end - t_solver_start).count();
        stats->n_prefix_tokens = n_prefix_tokens;
        stats->n_selected_tokens = n_selected;
    }

    return true;
}

bool llama_kv_compact_select_from_live_kv(
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats,
        llama_pos p0) {
    if (seq_id < 0 || target_tokens == 0 || live_suffix_pos0 <= p0) {
        return false;
    }

    std::vector<llama_pos> prefix_positions;
    if (!kv.compacted_prefix_seq_positions(seq_id, p0, live_suffix_pos0, prefix_positions)) {
        return false;
    }
    if (prefix_positions.empty()) {
        return false;
    }

    const auto & layouts = kv.get_compacted_prefix()->get_layouts();
    if (layouts.empty()) {
        return false;
    }

    const uint32_t n_prefix_tokens = prefix_positions.size();
    const uint32_t n_selected = std::min<uint32_t>(target_tokens, n_prefix_tokens);

    // Selection-only: keep the earliest n_selected positions (already sorted).
    std::vector<llama_pos> selected_positions(prefix_positions.begin(),
                                               prefix_positions.begin() + n_selected);

    const llama_pos seq_max = kv.seq_pos_max(seq_id);
    const uint32_t logical_token_count = seq_max >= 0 ? uint32_t(seq_max + 1) : uint32_t(live_suffix_pos0);
    if (!kv.compacted_prefix_configure(seq_id, logical_token_count, selected_positions, live_suffix_pos0)) {
        return false;
    }

    auto * seq = kv.get_compacted_prefix()->get_seq(seq_id);
    if (seq == nullptr || !seq->enabled || seq->layers.size() != layouts.size()) {
        return false;
    }

    // Populate K/V from original cache values with zero beta.
    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        auto & dst_layer = seq->layers[li];

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            // Copy original K
            std::vector<float> k_f32;
            if (!kv.compacted_prefix_copy_k_head_f32(
                        int32_t(layout.layer_id), seq_id, head, selected_positions, k_f32)) {
                return false;
            }
            llama_kv_compact_matrix k_mat(n_selected, layout.n_embd_head_k);
            k_mat.data = std::move(k_f32);
            write_compacted_payload(dst_layer.k_data, layout.type_k, layout.n_head_kv,
                                    n_selected, head, layout.n_embd_head_k, k_mat);

            // Copy original V
            if (layout.n_embd_head_v > 0) {
                std::vector<float> v_f32;
                if (!kv.compacted_prefix_copy_v_head_f32(
                            int32_t(layout.layer_id), seq_id, head, selected_positions, v_f32)) {
                    return false;
                }
                llama_kv_compact_matrix v_mat(n_selected, layout.n_embd_head_v);
                v_mat.data = std::move(v_f32);
                write_compacted_payload(dst_layer.v_data, layout.type_v, layout.n_head_kv,
                                        n_selected, head, layout.n_embd_head_v, v_mat);
            }

            // Zero beta
            for (uint32_t t = 0; t < n_selected; ++t) {
                dst_layer.beta_data[size_t(head) * n_selected + t] = 0.0f;
            }
        }
    }

    if (stats) {
        stats->query_generation_time_ms = 0.0;
        stats->solver_time_ms = 0.0;
        stats->n_prefix_tokens = n_prefix_tokens;
        stats->n_selected_tokens = n_selected;
    }

    return true;
}
