#include "llama-kv-compact-pipeline.h"

#include "llama-kv-cache.h"
#include "llama-kv-compact-budget.h"
#include "llama-kv-compact-query.h"
#include "llama-kv-compact-select.h"
#include "llama-kv-compact-self-study.h"
#include "llama-kv-compact-solver.h"
#include "llama-kv-compacted-prefix.h"
#include "llama-context.h"
#include "llama-model.h"

#include "llama-impl.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <numeric>
#include <cstring>

// Hard minimum target tokens below which compaction is refused.
// For solver/omp pipelines, the system becomes severely underdetermined
// when target_tokens is too small relative to the number of KV heads.
static constexpr uint32_t LLAMA_KV_COMPACT_MIN_TARGET_TOKENS = 2;

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
        uint32_t /*n_head_kv*/,
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

// Cached per-head data from Phase 1, reused in Phase 2.
struct head_cache_entry {
    llama_kv_compact_matrix k;       // [n_prefix x n_embd_head_k]
    llama_kv_compact_matrix queries; // [n_queries x n_embd_head_k]
};

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
    if (target_tokens < LLAMA_KV_COMPACT_MIN_TARGET_TOKENS) {
        LLAMA_LOG_WARN("%s: target_tokens=%u below minimum floor %u — refusing compaction\n",
                       __func__, target_tokens, LLAMA_KV_COMPACT_MIN_TARGET_TOKENS);
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

    // Phase 1: Extract K + queries, score, CACHE for Phase 2.
    // Eliminates the dual extraction that existed before.
    std::vector<std::vector<head_cache_entry>> layer_cache(layouts.size());

    double t_k_extract_ms = 0.0;
    double t_attn_score_ms = 0.0;
    double t_selection_ms = 0.0;
    double t_v_extract_ms = 0.0;
    double t_kv_write_ms = 0.0;

    const auto t_query_start = std::chrono::steady_clock::now();
    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        layer_cache[li].resize(layout.n_head_kv);

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            auto & entry = layer_cache[li][head];

            const auto t_k_start = std::chrono::steady_clock::now();
            std::vector<float> k_data;
            if (!kv.compacted_prefix_copy_k_head_f32(
                        int32_t(layout.layer_id), seq_id, head,
                        prefix_positions, k_data)) {
                return false;
            }
            entry.k.resize(n_prefix_tokens, layout.n_embd_head_k);
            entry.k.data = std::move(k_data);

            if (!llama_kv_compact_extract_cache_key_queries(
                        kv, seq_id, int32_t(layout.layer_id), head,
                        prefix_positions,
                        llama_kv_compact_query_params{ max_queries },
                        entry.queries)) {
                return false;
            }
            const auto t_k_end = std::chrono::steady_clock::now();
            t_k_extract_ms += std::chrono::duration<double, std::milli>(t_k_end - t_k_start).count();

            const auto t_score_start = std::chrono::steady_clock::now();
            llama_kv_compact_accumulate_attention_scores(
                    entry.queries, entry.k, aggregate_scores);
            const auto t_score_end = std::chrono::steady_clock::now();
            t_attn_score_ms += std::chrono::duration<double, std::milli>(t_score_end - t_score_start).count();
        }
    }
    const auto t_query_end = std::chrono::steady_clock::now();

    const auto t_sel_start = std::chrono::steady_clock::now();
    const std::vector<uint32_t> selected_local = llama_kv_compact_select_topk(aggregate_scores, n_selected);
    const auto t_sel_end = std::chrono::steady_clock::now();
    t_selection_ms = std::chrono::duration<double, std::milli>(t_sel_end - t_sel_start).count();
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

    // Phase 2: Solver (reuses cached K + queries from Phase 1).
    const auto t_solver_start = std::chrono::steady_clock::now();
    const llama_kv_compact_solver_opts solver_opts = {
        /* lambda           */ lambda,
        /* nnls_iters       */ nnls_iters,
        /* nnls_lower_bound */ 1e-12f,
        /* nnls_upper_bound */ 0.0f,
    };

    double residual_sum = 0.0;
    uint32_t residual_count = 0;

    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        auto & dst_layer = seq->layers[li];

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            const auto & entry = layer_cache[li][head];

            // V still needs extraction (not cached in Phase 1 to save memory).
            const auto t_v_start = std::chrono::steady_clock::now();
            std::vector<float> full_v_data;
            if (!kv.compacted_prefix_copy_v_head_f32(
                        int32_t(layout.layer_id), seq_id, head,
                        prefix_positions, full_v_data)) {
                return false;
            }
            llama_kv_compact_matrix full_v(n_prefix_tokens, layout.n_embd_head_v);
            full_v.data = std::move(full_v_data);
            const auto t_v_end = std::chrono::steady_clock::now();
            t_v_extract_ms += std::chrono::duration<double, std::milli>(t_v_end - t_v_start).count();

            llama_kv_compact_matrix compacted_k;
            if (!gather_matrix_rows(entry.k.data, entry.k.rows,
                                    entry.k.cols, selected_local,
                                    compacted_k)) {
                return false;
            }

            float head_residual = 0.0f;
            std::vector<float> beta;
            bool beta_ok = llama_kv_compact_fit_beta(entry.queries, entry.k,
                                                      compacted_k, solver_opts,
                                                      beta, &head_residual);

            // NaN guard (GAP-K): if beta fitting fails (numerical instability
            // at extreme compression), fall back to zero-beta for this head.
            if (!beta_ok) {
                LLAMA_LOG_WARN("%s: beta fitting failed for layer %zu head %u — falling back to zero-beta\n",
                               __func__, li, head);
                beta.assign(n_selected, 0.0f);
                head_residual = 0.0f;
            }
            residual_sum += head_residual;
            residual_count++;

            if (layout.n_embd_head_v > 0) {
                llama_kv_compact_matrix compacted_v;
                bool v_ok = beta_ok && llama_kv_compact_fit_values(
                            entry.queries, entry.k, full_v,
                            compacted_k, beta, solver_opts,
                            compacted_v);
                if (!v_ok) {
                    // NaN guard: fall back to original V values at selected positions.
                    if (beta_ok) {
                        LLAMA_LOG_WARN("%s: V fitting failed for layer %zu head %u — using original V\n",
                                       __func__, li, head);
                    }
                    if (!gather_matrix_rows(full_v.data, full_v.rows,
                                            full_v.cols, selected_local,
                                            compacted_v)) {
                        return false;
                    }
                }
                const auto t_w_start = std::chrono::steady_clock::now();
                write_compacted_payload(dst_layer.v_data, layout.type_v,
                                        layout.n_head_kv, n_selected, head,
                                        layout.n_embd_head_v, compacted_v);
                const auto t_w_mid = std::chrono::steady_clock::now();
                t_kv_write_ms += std::chrono::duration<double, std::milli>(t_w_mid - t_w_start).count();
            }

            const auto t_w2_start = std::chrono::steady_clock::now();
            write_compacted_payload(dst_layer.k_data, layout.type_k,
                                    layout.n_head_kv, n_selected, head,
                                    layout.n_embd_head_k, compacted_k);
            for (uint32_t token = 0; token < n_selected; ++token) {
                dst_layer.beta_data[size_t(head) * n_selected + token] = beta[token];
            }
            const auto t_w2_end = std::chrono::steady_clock::now();
            t_kv_write_ms += std::chrono::duration<double, std::milli>(t_w2_end - t_w2_start).count();
        }
    }
    const auto t_solver_end = std::chrono::steady_clock::now();

    if (stats) {
        stats->query_generation_time_ms = std::chrono::duration<double, std::milli>(t_query_end - t_query_start).count();
        stats->solver_time_ms = std::chrono::duration<double, std::milli>(t_solver_end - t_solver_start).count();
        stats->k_extraction_time_ms = t_k_extract_ms;
        stats->attention_score_time_ms = t_attn_score_ms;
        stats->selection_time_ms = t_selection_ms;
        stats->v_extraction_time_ms = t_v_extract_ms;
        stats->kv_write_time_ms = t_kv_write_ms;
        stats->total_time_ms = t_k_extract_ms + t_attn_score_ms + t_selection_ms
                             + t_v_extract_ms + t_kv_write_ms
                             + stats->solver_time_ms;
        stats->n_prefix_tokens = n_prefix_tokens;
        stats->n_selected_tokens = n_selected;
        stats->mean_partition_sum_relative_error = residual_count > 0
            ? (float)(residual_sum / residual_count) : 0.0f;
    }

    return true;
}

bool llama_kv_compact_omp_from_live_kv(
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
    if (target_tokens < LLAMA_KV_COMPACT_MIN_TARGET_TOKENS) {
        LLAMA_LOG_WARN("%s: target_tokens=%u below minimum floor %u — refusing compaction\n",
                       __func__, target_tokens, LLAMA_KV_COMPACT_MIN_TARGET_TOKENS);
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

    // Phase 1: Extract K + queries per head, run OMP for selection voting.
    std::vector<float> vote_scores(n_prefix_tokens, 0.0f);
    std::vector<std::vector<head_cache_entry>> layer_cache(layouts.size());

    double t_k_extract_ms = 0.0;
    double t_attn_score_ms = 0.0;
    double t_selection_ms = 0.0;
    double t_v_extract_ms = 0.0;
    double t_kv_write_ms = 0.0;

    const llama_kv_compact_omp_opts omp_opts = {};

    const auto t_query_start = std::chrono::steady_clock::now();
    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        layer_cache[li].resize(layout.n_head_kv);

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            auto & entry = layer_cache[li][head];

            const auto t_k_start = std::chrono::steady_clock::now();
            std::vector<float> k_data;
            if (!kv.compacted_prefix_copy_k_head_f32(
                        int32_t(layout.layer_id), seq_id, head,
                        prefix_positions, k_data)) {
                return false;
            }
            entry.k.resize(n_prefix_tokens, layout.n_embd_head_k);
            entry.k.data = std::move(k_data);

            if (!llama_kv_compact_extract_cache_key_queries(
                        kv, seq_id, int32_t(layout.layer_id), head,
                        prefix_positions,
                        llama_kv_compact_query_params{ max_queries },
                        entry.queries)) {
                return false;
            }
            const auto t_k_end = std::chrono::steady_clock::now();
            t_k_extract_ms += std::chrono::duration<double, std::milli>(t_k_end - t_k_start).count();

            // Run OMP per head to get greedy residual-based selection.
            const auto t_score_start = std::chrono::steady_clock::now();
            std::vector<float> beta_head;
            const std::vector<uint32_t> selected_head =
                llama_kv_compact_select_omp(entry.queries, entry.k, n_selected, omp_opts, beta_head);

            // Vote: increment score for each index selected by this head.
            for (uint32_t idx : selected_head) {
                vote_scores[idx] += 1.0f;
            }
            const auto t_score_end = std::chrono::steady_clock::now();
            t_attn_score_ms += std::chrono::duration<double, std::milli>(t_score_end - t_score_start).count();
        }
    }
    const auto t_query_end = std::chrono::steady_clock::now();

    // Aggregate votes -> global selection via topk on vote counts.
    const auto t_sel_start = std::chrono::steady_clock::now();
    const std::vector<uint32_t> selected_local = llama_kv_compact_select_topk(vote_scores, n_selected);
    const auto t_sel_end = std::chrono::steady_clock::now();
    t_selection_ms = std::chrono::duration<double, std::milli>(t_sel_end - t_sel_start).count();
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

    // Phase 2: Solver refit (reuses cached K + queries from Phase 1).
    const auto t_solver_start = std::chrono::steady_clock::now();
    const llama_kv_compact_solver_opts solver_opts = {
        /* lambda           */ lambda,
        /* nnls_iters       */ nnls_iters,
        /* nnls_lower_bound */ 1e-12f,
        /* nnls_upper_bound */ 0.0f,
    };

    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        auto & dst_layer = seq->layers[li];

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            const auto & entry = layer_cache[li][head];

            const auto t_v_start = std::chrono::steady_clock::now();
            std::vector<float> full_v_data;
            if (!kv.compacted_prefix_copy_v_head_f32(
                        int32_t(layout.layer_id), seq_id, head,
                        prefix_positions, full_v_data)) {
                return false;
            }
            llama_kv_compact_matrix full_v(n_prefix_tokens, layout.n_embd_head_v);
            full_v.data = std::move(full_v_data);
            const auto t_v_end = std::chrono::steady_clock::now();
            t_v_extract_ms += std::chrono::duration<double, std::milli>(t_v_end - t_v_start).count();

            llama_kv_compact_matrix compacted_k;
            if (!gather_matrix_rows(entry.k.data, entry.k.rows,
                                    entry.k.cols, selected_local,
                                    compacted_k)) {
                return false;
            }

            std::vector<float> beta;
            bool beta_ok = llama_kv_compact_fit_beta(entry.queries, entry.k,
                                                      compacted_k, solver_opts,
                                                      beta, nullptr);

            // NaN guard (GAP-K): fall back to zero-beta on solver failure.
            if (!beta_ok) {
                LLAMA_LOG_WARN("%s: beta fitting failed for layer %zu head %u — falling back to zero-beta\n",
                               __func__, li, head);
                beta.assign(n_selected, 0.0f);
            }

            if (layout.n_embd_head_v > 0) {
                llama_kv_compact_matrix compacted_v;
                bool v_ok = beta_ok && llama_kv_compact_fit_values(
                            entry.queries, entry.k, full_v,
                            compacted_k, beta, solver_opts,
                            compacted_v);
                if (!v_ok) {
                    if (beta_ok) {
                        LLAMA_LOG_WARN("%s: V fitting failed for layer %zu head %u — using original V\n",
                                       __func__, li, head);
                    }
                    if (!gather_matrix_rows(full_v.data, full_v.rows,
                                            full_v.cols, selected_local,
                                            compacted_v)) {
                        return false;
                    }
                }
                const auto t_w_start = std::chrono::steady_clock::now();
                write_compacted_payload(dst_layer.v_data, layout.type_v,
                                        layout.n_head_kv, n_selected, head,
                                        layout.n_embd_head_v, compacted_v);
                const auto t_w_mid = std::chrono::steady_clock::now();
                t_kv_write_ms += std::chrono::duration<double, std::milli>(t_w_mid - t_w_start).count();
            }

            const auto t_w2_start = std::chrono::steady_clock::now();
            write_compacted_payload(dst_layer.k_data, layout.type_k,
                                    layout.n_head_kv, n_selected, head,
                                    layout.n_embd_head_k, compacted_k);
            for (uint32_t token = 0; token < n_selected; ++token) {
                dst_layer.beta_data[size_t(head) * n_selected + token] = beta[token];
            }
            const auto t_w2_end = std::chrono::steady_clock::now();
            t_kv_write_ms += std::chrono::duration<double, std::milli>(t_w2_end - t_w2_start).count();
        }
    }
    const auto t_solver_end = std::chrono::steady_clock::now();

    if (stats) {
        stats->query_generation_time_ms = std::chrono::duration<double, std::milli>(t_query_end - t_query_start).count();
        stats->solver_time_ms = std::chrono::duration<double, std::milli>(t_solver_end - t_solver_start).count();
        stats->k_extraction_time_ms = t_k_extract_ms;
        stats->attention_score_time_ms = t_attn_score_ms;
        stats->selection_time_ms = t_selection_ms;
        stats->v_extraction_time_ms = t_v_extract_ms;
        stats->kv_write_time_ms = t_kv_write_ms;
        stats->total_time_ms = t_k_extract_ms + t_attn_score_ms + t_selection_ms
                             + t_v_extract_ms + t_kv_write_ms
                             + stats->solver_time_ms;
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
    if (target_tokens < LLAMA_KV_COMPACT_MIN_TARGET_TOKENS) {
        LLAMA_LOG_WARN("%s: target_tokens=%u below minimum floor %u — refusing compaction\n",
                       __func__, target_tokens, LLAMA_KV_COMPACT_MIN_TARGET_TOKENS);
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
    double t_k_extract_ms = 0.0;
    double t_v_extract_ms = 0.0;
    double t_kv_write_ms = 0.0;

    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        auto & dst_layer = seq->layers[li];

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            // Copy original K
            const auto t_k_start = std::chrono::steady_clock::now();
            std::vector<float> k_f32;
            if (!kv.compacted_prefix_copy_k_head_f32(
                        int32_t(layout.layer_id), seq_id, head, selected_positions, k_f32)) {
                return false;
            }
            llama_kv_compact_matrix k_mat(n_selected, layout.n_embd_head_k);
            k_mat.data = std::move(k_f32);
            const auto t_k_end = std::chrono::steady_clock::now();
            t_k_extract_ms += std::chrono::duration<double, std::milli>(t_k_end - t_k_start).count();

            // Copy original V
            if (layout.n_embd_head_v > 0) {
                const auto t_v_start = std::chrono::steady_clock::now();
                std::vector<float> v_f32;
                if (!kv.compacted_prefix_copy_v_head_f32(
                            int32_t(layout.layer_id), seq_id, head, selected_positions, v_f32)) {
                    return false;
                }
                llama_kv_compact_matrix v_mat(n_selected, layout.n_embd_head_v);
                v_mat.data = std::move(v_f32);
                const auto t_v_end = std::chrono::steady_clock::now();
                t_v_extract_ms += std::chrono::duration<double, std::milli>(t_v_end - t_v_start).count();

                const auto t_w_start = std::chrono::steady_clock::now();
                write_compacted_payload(dst_layer.v_data, layout.type_v, layout.n_head_kv,
                                        n_selected, head, layout.n_embd_head_v, v_mat);
                const auto t_w_mid = std::chrono::steady_clock::now();
                t_kv_write_ms += std::chrono::duration<double, std::milli>(t_w_mid - t_w_start).count();
            }

            const auto t_w2_start = std::chrono::steady_clock::now();
            write_compacted_payload(dst_layer.k_data, layout.type_k, layout.n_head_kv,
                                    n_selected, head, layout.n_embd_head_k, k_mat);

            // Zero beta
            for (uint32_t t = 0; t < n_selected; ++t) {
                dst_layer.beta_data[size_t(head) * n_selected + t] = 0.0f;
            }
            const auto t_w2_end = std::chrono::steady_clock::now();
            t_kv_write_ms += std::chrono::duration<double, std::milli>(t_w2_end - t_w2_start).count();
        }
    }

    if (stats) {
        stats->query_generation_time_ms = 0.0;
        stats->solver_time_ms = 0.0;
        stats->k_extraction_time_ms = t_k_extract_ms;
        stats->attention_score_time_ms = 0.0;
        stats->selection_time_ms = 0.0;
        stats->v_extraction_time_ms = t_v_extract_ms;
        stats->kv_write_time_ms = t_kv_write_ms;
        stats->total_time_ms = t_k_extract_ms + t_v_extract_ms + t_kv_write_ms;
        stats->n_prefix_tokens = n_prefix_tokens;
        stats->n_selected_tokens = n_selected;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Nonuniform per-head budget pipeline (Algorithm 4)
// ---------------------------------------------------------------------------

bool llama_kv_compact_nonuniform_from_live_kv(
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats,
        llama_pos p0,
        uint32_t max_queries,
        int nnls_iters,
        float lambda,
        uint32_t min_per_head) {
    if (seq_id < 0 || target_tokens == 0 || live_suffix_pos0 <= p0) {
        return false;
    }
    if (target_tokens < LLAMA_KV_COMPACT_MIN_TARGET_TOKENS) {
        LLAMA_LOG_WARN("%s: target_tokens=%u below minimum floor %u — refusing compaction\n",
                       __func__, target_tokens, LLAMA_KV_COMPACT_MIN_TARGET_TOKENS);
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

    const uint32_t n_prefix_tokens = (uint32_t) prefix_positions.size();

    // Phase 1: Extract K + queries per head, compute entropy for budget allocation.
    struct per_head_data {
        llama_kv_compact_matrix k;
        llama_kv_compact_matrix queries;
        float entropy = 0.0f;
    };

    // Flatten all heads across layers for budget computation.
    uint32_t total_kv_heads = 0;
    for (const auto & layout : layouts) {
        total_kv_heads += layout.n_head_kv;
    }

    // Total budget floor (GAP-K): refuse if target can't allocate minimum per head.
    const uint32_t total_floor = total_kv_heads * LLAMA_KV_COMPACT_BUDGET_FLOOR_PER_HEAD;
    if (target_tokens < total_floor) {
        LLAMA_LOG_WARN("%s: target_tokens=%u < %u heads × %u min = %u — "
                       "max recommended ratio: %.1fx. Refusing nonuniform compaction.\n",
                       __func__, target_tokens, total_kv_heads,
                       LLAMA_KV_COMPACT_BUDGET_FLOOR_PER_HEAD, total_floor,
                       (float) n_prefix_tokens / total_floor);
        return false;
    }

    std::vector<std::vector<per_head_data>> layer_data(layouts.size());
    std::vector<float> all_entropies;
    all_entropies.reserve(total_kv_heads);

    // Per-stage timing (populated at end; may be unused on fallback path).
    double t_k_extract_ms  = 0.0;
    double t_attn_score_ms = 0.0;
    double t_selection_ms  = 0.0;
    double t_v_extract_ms  = 0.0;
    double t_kv_write_ms   = 0.0;
    GGML_UNUSED(t_k_extract_ms); GGML_UNUSED(t_attn_score_ms);
    GGML_UNUSED(t_selection_ms); GGML_UNUSED(t_v_extract_ms);
    GGML_UNUSED(t_kv_write_ms);

    const auto t_query_start = std::chrono::steady_clock::now();
    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        layer_data[li].resize(layout.n_head_kv);

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            auto & hd = layer_data[li][head];

            const auto t_k_start = std::chrono::steady_clock::now();
            std::vector<float> k_data;
            if (!kv.compacted_prefix_copy_k_head_f32(
                        int32_t(layout.layer_id), seq_id, head,
                        prefix_positions, k_data)) {
                return false;
            }
            hd.k.resize(n_prefix_tokens, layout.n_embd_head_k);
            hd.k.data = std::move(k_data);

            if (!llama_kv_compact_extract_cache_key_queries(
                        kv, seq_id, int32_t(layout.layer_id), head,
                        prefix_positions,
                        llama_kv_compact_query_params{ max_queries },
                        hd.queries)) {
                return false;
            }
            const auto t_k_end = std::chrono::steady_clock::now();
            t_k_extract_ms += std::chrono::duration<double, std::milli>(t_k_end - t_k_start).count();

            hd.entropy = llama_kv_compact_head_entropy(hd.queries, hd.k);
            all_entropies.push_back(hd.entropy);
        }
    }

    // Compute per-head budgets.
    llama_kv_compact_budget_opts budget_opts;
    budget_opts.total_budget = std::min(target_tokens, n_prefix_tokens);
    // When the total budget is smaller than min_per_head * n_heads, reduce
    // min_per_head so the allocator can actually satisfy the constraint.
    budget_opts.min_per_head = std::min(min_per_head,
                                        std::max(1u, budget_opts.total_budget / total_kv_heads));
    budget_opts.max_per_head = n_prefix_tokens;

    const std::vector<uint32_t> budgets = llama_kv_compact_allocate_budgets(all_entropies, budget_opts);
    if (budgets.size() != total_kv_heads) {
        return false;
    }

    // Phase 2: Per-head top-k selection with individual budgets.
    const auto t_score_start = std::chrono::steady_clock::now();
    std::vector<std::vector<uint32_t>> per_head_selections(total_kv_heads);
    uint32_t head_idx = 0;
    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        for (uint32_t head = 0; head < layout.n_head_kv; ++head, ++head_idx) {
            auto & hd = layer_data[li][head];

            std::vector<float> head_scores(n_prefix_tokens, 0.0f);
            llama_kv_compact_accumulate_attention_scores(
                    hd.queries, hd.k, head_scores);
            per_head_selections[head_idx] = llama_kv_compact_select_topk(
                    head_scores, budgets[head_idx]);
        }
    }
    const auto t_score_end = std::chrono::steady_clock::now();
    t_attn_score_ms += std::chrono::duration<double, std::milli>(t_score_end - t_score_start).count();

    // Build union of all per-head selections.
    const auto t_sel_start = std::chrono::steady_clock::now();
    std::vector<bool> per_head_mask;
    std::vector<uint32_t> union_local = llama_kv_compact_build_union(
            per_head_selections, total_kv_heads, per_head_mask);

    // Cap union size to target_tokens. If the union is larger (disjoint
    // per-head selections), truncate by aggregated score ranking to keep
    // the output within the caller's target.
    const uint32_t target_cap = std::min(target_tokens, n_prefix_tokens);
    if (union_local.size() > target_cap) {
        // Aggregate scores across all heads for union positions.
        std::vector<float> agg_scores(n_prefix_tokens, 0.0f);
        head_idx = 0;
        for (size_t li = 0; li < layouts.size(); ++li) {
            const auto & layout = layouts[li];
            for (uint32_t head = 0; head < layout.n_head_kv; ++head, ++head_idx) {
                auto & hd = layer_data[li][head];
                std::vector<float> head_scores(n_prefix_tokens, 0.0f);
                llama_kv_compact_accumulate_attention_scores(
                        hd.queries, hd.k, head_scores);
                for (uint32_t ki = 0; ki < n_prefix_tokens; ++ki) {
                    agg_scores[ki] += head_scores[ki];
                }
            }
        }

        // Re-select from union positions only, ranked by aggregated score.
        std::sort(union_local.begin(), union_local.end(),
                  [&](uint32_t a, uint32_t b) {
                      return agg_scores[a] > agg_scores[b];
                  });
        union_local.resize(target_cap);
        std::sort(union_local.begin(), union_local.end());

        // Rebuild per-head mask for truncated union.
        per_head_mask = {};  // clear old mask
        // Build manually using the same logic as build_union.
        const uint32_t union_size = (uint32_t) union_local.size();
        per_head_mask.assign(size_t(total_kv_heads) * union_size, false);

        // Build position-to-union-index map.
        std::vector<uint32_t> pos_to_union;
        if (!union_local.empty()) {
            pos_to_union.assign(union_local.back() + 1, UINT32_MAX);
            for (uint32_t j = 0; j < union_size; ++j) {
                pos_to_union[union_local[j]] = j;
            }
        }

        for (uint32_t h = 0; h < total_kv_heads; ++h) {
            for (uint32_t idx : per_head_selections[h]) {
                if (idx < pos_to_union.size()) {
                    uint32_t j = pos_to_union[idx];
                    if (j != UINT32_MAX) {
                        per_head_mask[size_t(h) * union_size + j] = true;
                    }
                }
            }
        }
    }

    const uint32_t n_selected = (uint32_t) union_local.size();
    if (n_selected == 0) {
        return false;
    }

    // BUG-I01 guard: if union truncation caused >50% of heads to lose ALL
    // their selected tokens, the nonuniform pipeline will produce near-random
    // output (cosine ~0.2).  Fall back to the select pipeline which uses a
    // single global selection set and avoids per-head masking entirely.
    {
        uint32_t n_fully_masked = 0;
        for (uint32_t h = 0; h < total_kv_heads; ++h) {
            bool has_any = false;
            for (uint32_t j = 0; j < n_selected; ++j) {
                if (per_head_mask[size_t(h) * n_selected + j]) {
                    has_any = true;
                    break;
                }
            }
            if (!has_any) {
                n_fully_masked++;
            }
        }

        if (n_fully_masked * 2 > total_kv_heads) {
            LLAMA_LOG_WARN("%s: nonuniform pipeline: %u/%u heads fully masked after union truncation — "
                           "falling back to select pipeline\n",
                           __func__, (unsigned)n_fully_masked, (unsigned)total_kv_heads);
            return llama_kv_compact_select_from_live_kv(
                    kv, seq_id, target_tokens, live_suffix_pos0, stats, p0);
        }
    }

    const auto t_sel_end = std::chrono::steady_clock::now();
    t_selection_ms = std::chrono::duration<double, std::milli>(t_sel_end - t_sel_start).count();

    const auto t_query_end = std::chrono::steady_clock::now();

    // Convert union indices to positions.
    std::vector<llama_pos> selected_positions;
    selected_positions.reserve(n_selected);
    for (uint32_t idx : union_local) {
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

    // Phase 3: Solver with per-head masking (beta=-inf for non-selected positions).
    const auto t_solver_start = std::chrono::steady_clock::now();
    const llama_kv_compact_solver_opts solver_opts = {
        /* lambda           */ lambda,
        /* nnls_iters       */ nnls_iters,
        /* nnls_lower_bound */ 1e-12f,
        /* nnls_upper_bound */ 0.0f,
        /* ridge_scale      */ LLAMA_KV_COMPACT_RIDGE_SPECTRAL,
    };

    head_idx = 0;
    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        auto & dst_layer = seq->layers[li];

        for (uint32_t head = 0; head < layout.n_head_kv; ++head, ++head_idx) {
            const auto & hd = layer_data[li][head];

            // V extraction.
            const auto t_v_start = std::chrono::steady_clock::now();
            std::vector<float> full_v_data;
            if (!kv.compacted_prefix_copy_v_head_f32(
                        int32_t(layout.layer_id), seq_id, head,
                        prefix_positions, full_v_data)) {
                return false;
            }
            llama_kv_compact_matrix full_v(n_prefix_tokens, layout.n_embd_head_v);
            full_v.data = std::move(full_v_data);
            const auto t_v_end = std::chrono::steady_clock::now();
            t_v_extract_ms += std::chrono::duration<double, std::milli>(t_v_end - t_v_start).count();

            // Gather union K rows.
            llama_kv_compact_matrix compacted_k;
            if (!gather_matrix_rows(hd.k.data, hd.k.rows, hd.k.cols,
                                    union_local, compacted_k)) {
                return false;
            }

            // Fit beta on this head's selected subset.
            std::vector<float> beta;
            bool beta_ok = llama_kv_compact_fit_beta(hd.queries, hd.k,
                                                      compacted_k, solver_opts,
                                                      beta, nullptr);

            // NaN guard (GAP-K): fall back to zero-beta on solver failure.
            if (!beta_ok) {
                LLAMA_LOG_WARN("%s: beta fitting failed for layer %zu head %u — falling back to zero-beta\n",
                               __func__, li, head);
                beta.assign(n_selected, 0.0f);
            }

            // Apply per-head mask: set beta=-inf for positions NOT selected by this head.
            bool head_fully_masked = true;
            for (uint32_t j = 0; j < n_selected; ++j) {
                if (!per_head_mask[size_t(head_idx) * n_selected + j]) {
                    beta[j] = -std::numeric_limits<float>::infinity();
                } else {
                    head_fully_masked = false;
                }
            }

            // V fitting.
            if (layout.n_embd_head_v > 0) {
                llama_kv_compact_matrix compacted_v;
                if (head_fully_masked) {
                    // All positions were dropped by union truncation — zero out V
                    // to prevent NaN from -inf beta propagating through fit_values.
                    compacted_v.resize(n_selected, layout.n_embd_head_v);
                    std::fill(compacted_v.data.begin(), compacted_v.data.end(), 0.0f);
                } else if (!beta_ok) {
                    // Beta fell back to zero — use original V values.
                    if (!gather_matrix_rows(full_v.data, full_v.rows,
                                            full_v.cols, union_local,
                                            compacted_v)) {
                        return false;
                    }
                } else if (!llama_kv_compact_fit_values(
                            hd.queries, hd.k, full_v,
                            compacted_k, beta, solver_opts,
                            compacted_v)) {
                    LLAMA_LOG_WARN("%s: V fitting failed for layer %zu head %u — using original V\n",
                                   __func__, li, head);
                    if (!gather_matrix_rows(full_v.data, full_v.rows,
                                            full_v.cols, union_local,
                                            compacted_v)) {
                        return false;
                    }
                }
                const auto t_w_start = std::chrono::steady_clock::now();
                write_compacted_payload(dst_layer.v_data, layout.type_v,
                                        layout.n_head_kv, n_selected, head,
                                        layout.n_embd_head_v, compacted_v);
                const auto t_w_mid = std::chrono::steady_clock::now();
                t_kv_write_ms += std::chrono::duration<double, std::milli>(t_w_mid - t_w_start).count();
            }

            const auto t_w2_start = std::chrono::steady_clock::now();
            write_compacted_payload(dst_layer.k_data, layout.type_k,
                                    layout.n_head_kv, n_selected, head,
                                    layout.n_embd_head_k, compacted_k);
            for (uint32_t token = 0; token < n_selected; ++token) {
                dst_layer.beta_data[size_t(head) * n_selected + token] = beta[token];
            }
            const auto t_w2_end = std::chrono::steady_clock::now();
            t_kv_write_ms += std::chrono::duration<double, std::milli>(t_w2_end - t_w2_start).count();
        }
    }
    const auto t_solver_end = std::chrono::steady_clock::now();

    if (stats) {
        stats->query_generation_time_ms = std::chrono::duration<double, std::milli>(t_query_end - t_query_start).count();
        stats->solver_time_ms = std::chrono::duration<double, std::milli>(t_solver_end - t_solver_start).count();
        stats->k_extraction_time_ms = t_k_extract_ms;
        stats->attention_score_time_ms = t_attn_score_ms;
        stats->selection_time_ms = t_selection_ms;
        stats->v_extraction_time_ms = t_v_extract_ms;
        stats->kv_write_time_ms = t_kv_write_ms;
        stats->total_time_ms = t_k_extract_ms + t_attn_score_ms + t_selection_ms
                             + t_v_extract_ms + t_kv_write_ms
                             + stats->solver_time_ms;
        stats->n_prefix_tokens = n_prefix_tokens;
        stats->n_selected_tokens = n_selected;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Chunked compaction pipeline (Section 3.5)
// ---------------------------------------------------------------------------

bool llama_kv_compact_chunked_from_live_kv(
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats,
        llama_pos p0,
        uint32_t max_queries,
        int nnls_iters,
        float lambda,
        uint32_t chunk_size) {
    if (seq_id < 0 || target_tokens == 0 || live_suffix_pos0 <= p0) {
        return false;
    }
    if (target_tokens < LLAMA_KV_COMPACT_MIN_TARGET_TOKENS) {
        LLAMA_LOG_WARN("%s: target_tokens=%u below minimum floor %u — refusing compaction\n",
                       __func__, target_tokens, LLAMA_KV_COMPACT_MIN_TARGET_TOKENS);
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

    const uint32_t n_prefix_tokens = (uint32_t) prefix_positions.size();
    if (n_prefix_tokens <= chunk_size) {
        // Single chunk: delegate to standard solver pipeline.
        return llama_kv_compact_fit_from_live_kv(kv, seq_id, target_tokens,
                                                  live_suffix_pos0, stats, p0,
                                                  max_queries, nnls_iters, lambda);
    }

    // Split prefix into chunks and allocate proportional budgets.
    uint32_t n_chunks = (n_prefix_tokens + chunk_size - 1) / chunk_size;

    // If more chunks than target tokens, merge chunks so each gets budget >= 1.
    // This avoids the infeasible case where n_chunks floors at 1 per chunk > target.
    if (n_chunks > target_tokens) {
        const uint32_t merged_chunk_size = (n_prefix_tokens + target_tokens - 1) / target_tokens;
        // Recurse with the merged chunk size so each chunk gets budget >= 1.
        return llama_kv_compact_chunked_from_live_kv(
            kv, seq_id, target_tokens, live_suffix_pos0, stats, p0,
            max_queries, nnls_iters, lambda, merged_chunk_size);
    }

    std::vector<uint32_t> chunk_starts(n_chunks);
    std::vector<uint32_t> chunk_sizes(n_chunks);
    std::vector<uint32_t> chunk_budgets(n_chunks);

    uint32_t budget_allocated = 0;
    for (uint32_t c = 0; c < n_chunks; ++c) {
        chunk_starts[c] = c * chunk_size;
        chunk_sizes[c] = std::min(chunk_size, n_prefix_tokens - chunk_starts[c]);
        // Proportional budget allocation per chunk.
        chunk_budgets[c] = (uint32_t) std::round(
            float(target_tokens) * float(chunk_sizes[c]) / float(n_prefix_tokens));
        chunk_budgets[c] = std::max(chunk_budgets[c], 1u);
        budget_allocated += chunk_budgets[c];
    }
    // Adjust budgets to hit exact target.
    if (budget_allocated > target_tokens) {
        // Distribute overshoot across chunks, allowing budget=0 for some chunks.
        std::vector<uint32_t> order(n_chunks);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return chunk_budgets[a] < chunk_budgets[b];
        });
        uint32_t excess = budget_allocated - target_tokens;
        for (uint32_t idx : order) {
            if (excess == 0) break;
            uint32_t reduce = std::min(chunk_budgets[idx], excess);
            chunk_budgets[idx] -= reduce;
            excess -= reduce;
        }
    } else if (budget_allocated < target_tokens) {
        chunk_budgets[n_chunks - 1] += target_tokens - budget_allocated;
    }

    // Phase 1: Per-chunk scoring and selection.
    const auto t_query_start = std::chrono::steady_clock::now();

    // Per-stage timing.
    double t_k_extract_ms  = 0.0;
    double t_attn_score_ms = 0.0;
    double t_selection_ms  = 0.0;
    double t_v_extract_ms  = 0.0;
    double t_kv_write_ms   = 0.0;
    GGML_UNUSED(t_k_extract_ms); GGML_UNUSED(t_attn_score_ms);
    GGML_UNUSED(t_selection_ms); GGML_UNUSED(t_v_extract_ms);
    GGML_UNUSED(t_kv_write_ms);

    // Collect selected positions from all chunks.
    std::vector<uint32_t> all_selected_local;
    all_selected_local.reserve(target_tokens);

    // Cache per-chunk per-head data for solver phase.
    struct chunk_head_cache {
        llama_kv_compact_matrix k;
        llama_kv_compact_matrix queries;
    };
    // chunk_caches[chunk][layer][head]
    std::vector<std::vector<std::vector<chunk_head_cache>>> chunk_caches(n_chunks);

    for (uint32_t c = 0; c < n_chunks; ++c) {
        const uint32_t cs = chunk_starts[c];
        const uint32_t cn = chunk_sizes[c];
        const uint32_t cb = std::min(chunk_budgets[c], cn);

        // Extract chunk positions.
        std::vector<llama_pos> chunk_positions(
            prefix_positions.begin() + cs,
            prefix_positions.begin() + cs + cn);

        std::vector<float> chunk_scores(cn, 0.0f);
        chunk_caches[c].resize(layouts.size());

        for (size_t li = 0; li < layouts.size(); ++li) {
            const auto & layout = layouts[li];
            chunk_caches[c][li].resize(layout.n_head_kv);

            for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
                auto & cc = chunk_caches[c][li][head];

                const auto t_k_start = std::chrono::steady_clock::now();
                std::vector<float> k_data;
                if (!kv.compacted_prefix_copy_k_head_f32(
                            int32_t(layout.layer_id), seq_id, head,
                            chunk_positions, k_data)) {
                    return false;
                }
                cc.k.resize(cn, layout.n_embd_head_k);
                cc.k.data = std::move(k_data);

                if (!llama_kv_compact_extract_cache_key_queries(
                            kv, seq_id, int32_t(layout.layer_id), head,
                            chunk_positions,
                            llama_kv_compact_query_params{ max_queries },
                            cc.queries)) {
                    return false;
                }
                const auto t_k_end = std::chrono::steady_clock::now();
                t_k_extract_ms += std::chrono::duration<double, std::milli>(t_k_end - t_k_start).count();

                const auto t_score_start = std::chrono::steady_clock::now();
                llama_kv_compact_accumulate_attention_scores(
                        cc.queries, cc.k, chunk_scores);
                const auto t_score_end = std::chrono::steady_clock::now();
                t_attn_score_ms += std::chrono::duration<double, std::milli>(t_score_end - t_score_start).count();
            }
        }

        // Select top-k within this chunk.
        const auto t_sel_start = std::chrono::steady_clock::now();
        const std::vector<uint32_t> chunk_selected = llama_kv_compact_select_topk(chunk_scores, cb);
        const auto t_sel_end = std::chrono::steady_clock::now();
        t_selection_ms += std::chrono::duration<double, std::milli>(t_sel_end - t_sel_start).count();

        // Map chunk-local indices to global prefix indices.
        for (uint32_t idx : chunk_selected) {
            all_selected_local.push_back(cs + idx);
        }
    }

    // Sort all selected indices globally and cap at target.
    std::sort(all_selected_local.begin(), all_selected_local.end());
    if (all_selected_local.size() > target_tokens) {
        all_selected_local.resize(target_tokens);
    }
    const uint32_t n_selected = (uint32_t) all_selected_local.size();

    const auto t_query_end = std::chrono::steady_clock::now();

    // Convert to positions.
    std::vector<llama_pos> selected_positions;
    selected_positions.reserve(n_selected);
    for (uint32_t idx : all_selected_local) {
        selected_positions.push_back(prefix_positions[idx]);
    }

    // Configure compacted prefix store.
    const llama_pos seq_max = kv.seq_pos_max(seq_id);
    const uint32_t logical_token_count = seq_max >= 0 ? uint32_t(seq_max + 1) : uint32_t(live_suffix_pos0);
    if (!kv.compacted_prefix_configure(seq_id, logical_token_count, selected_positions, live_suffix_pos0)) {
        return false;
    }

    auto * seq_state = kv.get_compacted_prefix()->get_seq(seq_id);
    if (seq_state == nullptr || !seq_state->enabled || seq_state->layers.size() != layouts.size()) {
        return false;
    }

    // Phase 2: Solver pass — extract K/V for selected positions, fit beta/V.
    // We use the full prefix for the solver (not per-chunk) since the selected
    // positions span multiple chunks. K/V are re-extracted from the live cache
    // for the globally selected positions.
    const auto t_solver_start = std::chrono::steady_clock::now();
    const llama_kv_compact_solver_opts solver_opts = {
        /* lambda           */ lambda,
        /* nnls_iters       */ nnls_iters,
        /* nnls_lower_bound */ 1e-12f,
        /* nnls_upper_bound */ 0.0f,
        /* ridge_scale      */ LLAMA_KV_COMPACT_RIDGE_SPECTRAL,
    };

    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        auto & dst_layer = seq_state->layers[li];

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            // Extract full-prefix K for scoring.
            const auto t_k2_start = std::chrono::steady_clock::now();
            std::vector<float> full_k_data;
            if (!kv.compacted_prefix_copy_k_head_f32(
                        int32_t(layout.layer_id), seq_id, head,
                        prefix_positions, full_k_data)) {
                return false;
            }
            llama_kv_compact_matrix full_k(n_prefix_tokens, layout.n_embd_head_k);
            full_k.data = std::move(full_k_data);

            // Build queries from full prefix K (cache-key surrogates).
            llama_kv_compact_matrix queries;
            if (!llama_kv_compact_extract_cache_key_queries(
                        kv, seq_id, int32_t(layout.layer_id), head,
                        prefix_positions,
                        llama_kv_compact_query_params{ max_queries },
                        queries)) {
                return false;
            }
            const auto t_k2_end = std::chrono::steady_clock::now();
            t_k_extract_ms += std::chrono::duration<double, std::milli>(t_k2_end - t_k2_start).count();

            // Gather selected K rows.
            llama_kv_compact_matrix compacted_k;
            if (!gather_matrix_rows(full_k.data, full_k.rows, full_k.cols,
                                    all_selected_local, compacted_k)) {
                return false;
            }

            // Fit beta.
            std::vector<float> beta;
            bool beta_ok = llama_kv_compact_fit_beta(queries, full_k,
                                                      compacted_k, solver_opts,
                                                      beta, nullptr);

            // NaN guard (GAP-K): fall back to zero-beta on solver failure.
            if (!beta_ok) {
                LLAMA_LOG_WARN("%s: beta fitting failed for layer %zu head %u — falling back to zero-beta\n",
                               __func__, li, head);
                beta.assign(n_selected, 0.0f);
            }

            // V extraction and fitting.
            if (layout.n_embd_head_v > 0) {
                const auto t_v_start = std::chrono::steady_clock::now();
                std::vector<float> full_v_data;
                if (!kv.compacted_prefix_copy_v_head_f32(
                            int32_t(layout.layer_id), seq_id, head,
                            prefix_positions, full_v_data)) {
                    return false;
                }
                llama_kv_compact_matrix full_v(n_prefix_tokens, layout.n_embd_head_v);
                full_v.data = std::move(full_v_data);
                const auto t_v_end = std::chrono::steady_clock::now();
                t_v_extract_ms += std::chrono::duration<double, std::milli>(t_v_end - t_v_start).count();

                llama_kv_compact_matrix compacted_v;
                bool v_ok = beta_ok && llama_kv_compact_fit_values(
                            queries, full_k, full_v,
                            compacted_k, beta, solver_opts,
                            compacted_v);
                if (!v_ok) {
                    if (beta_ok) {
                        LLAMA_LOG_WARN("%s: V fitting failed for layer %zu head %u — using original V\n",
                                       __func__, li, head);
                    }
                    if (!gather_matrix_rows(full_v.data, full_v.rows,
                                            full_v.cols, all_selected_local,
                                            compacted_v)) {
                        return false;
                    }
                }
                const auto t_w_start = std::chrono::steady_clock::now();
                write_compacted_payload(dst_layer.v_data, layout.type_v,
                                        layout.n_head_kv, n_selected, head,
                                        layout.n_embd_head_v, compacted_v);
                const auto t_w_mid = std::chrono::steady_clock::now();
                t_kv_write_ms += std::chrono::duration<double, std::milli>(t_w_mid - t_w_start).count();
            }

            const auto t_w2_start = std::chrono::steady_clock::now();
            write_compacted_payload(dst_layer.k_data, layout.type_k,
                                    layout.n_head_kv, n_selected, head,
                                    layout.n_embd_head_k, compacted_k);
            for (uint32_t token = 0; token < n_selected; ++token) {
                dst_layer.beta_data[size_t(head) * n_selected + token] = beta[token];
            }
            const auto t_w2_end = std::chrono::steady_clock::now();
            t_kv_write_ms += std::chrono::duration<double, std::milli>(t_w2_end - t_w2_start).count();
        }
    }
    const auto t_solver_end = std::chrono::steady_clock::now();

    if (stats) {
        stats->query_generation_time_ms = std::chrono::duration<double, std::milli>(t_query_end - t_query_start).count();
        stats->solver_time_ms = std::chrono::duration<double, std::milli>(t_solver_end - t_solver_start).count();
        stats->k_extraction_time_ms = t_k_extract_ms;
        stats->attention_score_time_ms = t_attn_score_ms;
        stats->selection_time_ms = t_selection_ms;
        stats->v_extraction_time_ms = t_v_extract_ms;
        stats->kv_write_time_ms = t_kv_write_ms;
        stats->total_time_ms = t_k_extract_ms + t_attn_score_ms + t_selection_ms
                             + t_v_extract_ms + t_kv_write_ms
                             + stats->solver_time_ms;
        stats->n_prefix_tokens = n_prefix_tokens;
        stats->n_selected_tokens = n_selected;
    }

    return true;
}
