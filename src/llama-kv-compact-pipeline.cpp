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
#include <limits>
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

    const auto t_query_start = std::chrono::steady_clock::now();
    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        layer_cache[li].resize(layout.n_head_kv);

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            auto & entry = layer_cache[li][head];

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

            llama_kv_compact_accumulate_attention_scores(
                    entry.queries, entry.k, aggregate_scores);
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

    // Phase 2: Solver (reuses cached K + queries from Phase 1).
    const auto t_solver_start = std::chrono::steady_clock::now();
    const llama_kv_compact_solver_opts solver_opts = {
        /* lambda           */ lambda,
        /* nnls_iters       */ nnls_iters,
        /* nnls_lower_bound */ 0.05f,
        /* nnls_upper_bound */ 20.0f,
    };

    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        auto & dst_layer = seq->layers[li];

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            const auto & entry = layer_cache[li][head];

            // V still needs extraction (not cached in Phase 1 to save memory).
            std::vector<float> full_v_data;
            if (!kv.compacted_prefix_copy_v_head_f32(
                        int32_t(layout.layer_id), seq_id, head,
                        prefix_positions, full_v_data)) {
                return false;
            }
            llama_kv_compact_matrix full_v(n_prefix_tokens, layout.n_embd_head_v);
            full_v.data = std::move(full_v_data);

            llama_kv_compact_matrix compacted_k;
            if (!gather_matrix_rows(entry.k.data, entry.k.rows,
                                    entry.k.cols, selected_local,
                                    compacted_k)) {
                return false;
            }

            std::vector<float> beta;
            if (!llama_kv_compact_fit_beta(entry.queries, entry.k,
                                            compacted_k, solver_opts,
                                            beta, nullptr)) {
                return false;
            }

            if (layout.n_embd_head_v > 0) {
                llama_kv_compact_matrix compacted_v;
                if (!llama_kv_compact_fit_values(
                            entry.queries, entry.k, full_v,
                            compacted_k, beta, solver_opts,
                            compacted_v)) {
                    return false;
                }
                write_compacted_payload(dst_layer.v_data, layout.type_v,
                                        layout.n_head_kv, n_selected, head,
                                        layout.n_embd_head_v, compacted_v);
            }

            write_compacted_payload(dst_layer.k_data, layout.type_k,
                                    layout.n_head_kv, n_selected, head,
                                    layout.n_embd_head_k, compacted_k);
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

    const llama_kv_compact_omp_opts omp_opts = {};

    const auto t_query_start = std::chrono::steady_clock::now();
    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        layer_cache[li].resize(layout.n_head_kv);

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            auto & entry = layer_cache[li][head];

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

            // Run OMP per head to get greedy residual-based selection.
            std::vector<float> beta_head;
            const std::vector<uint32_t> selected_head =
                llama_kv_compact_select_omp(entry.queries, entry.k, n_selected, omp_opts, beta_head);

            // Vote: increment score for each index selected by this head.
            for (uint32_t idx : selected_head) {
                vote_scores[idx] += 1.0f;
            }
        }
    }
    const auto t_query_end = std::chrono::steady_clock::now();

    // Aggregate votes -> global selection via topk on vote counts.
    const std::vector<uint32_t> selected_local = llama_kv_compact_select_topk(vote_scores, n_selected);
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
        /* nnls_lower_bound */ 0.05f,
        /* nnls_upper_bound */ 20.0f,
    };

    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        auto & dst_layer = seq->layers[li];

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            const auto & entry = layer_cache[li][head];

            std::vector<float> full_v_data;
            if (!kv.compacted_prefix_copy_v_head_f32(
                        int32_t(layout.layer_id), seq_id, head,
                        prefix_positions, full_v_data)) {
                return false;
            }
            llama_kv_compact_matrix full_v(n_prefix_tokens, layout.n_embd_head_v);
            full_v.data = std::move(full_v_data);

            llama_kv_compact_matrix compacted_k;
            if (!gather_matrix_rows(entry.k.data, entry.k.rows,
                                    entry.k.cols, selected_local,
                                    compacted_k)) {
                return false;
            }

            std::vector<float> beta;
            if (!llama_kv_compact_fit_beta(entry.queries, entry.k,
                                            compacted_k, solver_opts,
                                            beta, nullptr)) {
                return false;
            }

            if (layout.n_embd_head_v > 0) {
                llama_kv_compact_matrix compacted_v;
                if (!llama_kv_compact_fit_values(
                            entry.queries, entry.k, full_v,
                            compacted_k, beta, solver_opts,
                            compacted_v)) {
                    return false;
                }
                write_compacted_payload(dst_layer.v_data, layout.type_v,
                                        layout.n_head_kv, n_selected, head,
                                        layout.n_embd_head_v, compacted_v);
            }

            write_compacted_payload(dst_layer.k_data, layout.type_k,
                                    layout.n_head_kv, n_selected, head,
                                    layout.n_embd_head_k, compacted_k);
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

    std::vector<std::vector<per_head_data>> layer_data(layouts.size());
    std::vector<float> all_entropies;
    all_entropies.reserve(total_kv_heads);

    const auto t_query_start = std::chrono::steady_clock::now();
    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        layer_data[li].resize(layout.n_head_kv);

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            auto & hd = layer_data[li][head];

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

            hd.entropy = llama_kv_compact_head_entropy(hd.queries, hd.k);
            all_entropies.push_back(hd.entropy);
        }
    }

    // Compute per-head budgets.
    llama_kv_compact_budget_opts budget_opts;
    budget_opts.total_budget = std::min(target_tokens, n_prefix_tokens);
    budget_opts.min_per_head = std::min(min_per_head, budget_opts.total_budget);
    budget_opts.max_per_head = n_prefix_tokens;

    const std::vector<uint32_t> budgets = llama_kv_compact_allocate_budgets(all_entropies, budget_opts);
    if (budgets.size() != total_kv_heads) {
        return false;
    }

    // Phase 2: Per-head top-k selection with individual budgets.
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

    // Build union of all per-head selections.
    std::vector<bool> per_head_mask;
    std::vector<uint32_t> union_local = llama_kv_compact_build_union(
            per_head_selections, total_kv_heads, per_head_mask);

    const uint32_t n_selected = (uint32_t) union_local.size();
    if (n_selected == 0) {
        return false;
    }

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
        /* nnls_lower_bound */ 0.05f,
        /* nnls_upper_bound */ 20.0f,
        /* spectral_ridge   */ false,
    };

    head_idx = 0;
    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        auto & dst_layer = seq->layers[li];

        for (uint32_t head = 0; head < layout.n_head_kv; ++head, ++head_idx) {
            const auto & hd = layer_data[li][head];

            // V extraction.
            std::vector<float> full_v_data;
            if (!kv.compacted_prefix_copy_v_head_f32(
                        int32_t(layout.layer_id), seq_id, head,
                        prefix_positions, full_v_data)) {
                return false;
            }
            llama_kv_compact_matrix full_v(n_prefix_tokens, layout.n_embd_head_v);
            full_v.data = std::move(full_v_data);

            // Gather union K rows.
            llama_kv_compact_matrix compacted_k;
            if (!gather_matrix_rows(hd.k.data, hd.k.rows, hd.k.cols,
                                    union_local, compacted_k)) {
                return false;
            }

            // Fit beta on this head's selected subset.
            std::vector<float> beta;
            if (!llama_kv_compact_fit_beta(hd.queries, hd.k,
                                            compacted_k, solver_opts,
                                            beta, nullptr)) {
                return false;
            }

            // Apply per-head mask: set beta=-inf for positions NOT selected by this head.
            for (uint32_t j = 0; j < n_selected; ++j) {
                if (!per_head_mask[size_t(head_idx) * n_selected + j]) {
                    beta[j] = -std::numeric_limits<float>::infinity();
                }
            }

            // V fitting.
            if (layout.n_embd_head_v > 0) {
                llama_kv_compact_matrix compacted_v;
                if (!llama_kv_compact_fit_values(
                            hd.queries, hd.k, full_v,
                            compacted_k, beta, solver_opts,
                            compacted_v)) {
                    return false;
                }
                write_compacted_payload(dst_layer.v_data, layout.type_v,
                                        layout.n_head_kv, n_selected, head,
                                        layout.n_embd_head_v, compacted_v);
            }

            write_compacted_payload(dst_layer.k_data, layout.type_k,
                                    layout.n_head_kv, n_selected, head,
                                    layout.n_embd_head_k, compacted_k);
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
    const uint32_t n_chunks = (n_prefix_tokens + chunk_size - 1) / chunk_size;
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
    // Adjust last chunk to hit exact target.
    if (budget_allocated > target_tokens && chunk_budgets[n_chunks - 1] > 1) {
        chunk_budgets[n_chunks - 1] -= std::min(
            chunk_budgets[n_chunks - 1] - 1,
            budget_allocated - target_tokens);
    } else if (budget_allocated < target_tokens) {
        chunk_budgets[n_chunks - 1] += target_tokens - budget_allocated;
    }

    // Phase 1: Per-chunk scoring and selection.
    const auto t_query_start = std::chrono::steady_clock::now();

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

                llama_kv_compact_accumulate_attention_scores(
                        cc.queries, cc.k, chunk_scores);
            }
        }

        // Select top-k within this chunk.
        const std::vector<uint32_t> chunk_selected = llama_kv_compact_select_topk(chunk_scores, cb);

        // Map chunk-local indices to global prefix indices.
        for (uint32_t idx : chunk_selected) {
            all_selected_local.push_back(cs + idx);
        }
    }

    // Sort all selected indices globally.
    std::sort(all_selected_local.begin(), all_selected_local.end());
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
        /* nnls_lower_bound */ 0.05f,
        /* nnls_upper_bound */ 20.0f,
        /* spectral_ridge   */ false,
    };

    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        auto & dst_layer = seq_state->layers[li];

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            // Extract full-prefix K for scoring.
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

            // Gather selected K rows.
            llama_kv_compact_matrix compacted_k;
            if (!gather_matrix_rows(full_k.data, full_k.rows, full_k.cols,
                                    all_selected_local, compacted_k)) {
                return false;
            }

            // Fit beta.
            std::vector<float> beta;
            if (!llama_kv_compact_fit_beta(queries, full_k,
                                            compacted_k, solver_opts,
                                            beta, nullptr)) {
                return false;
            }

            // V extraction and fitting.
            if (layout.n_embd_head_v > 0) {
                std::vector<float> full_v_data;
                if (!kv.compacted_prefix_copy_v_head_f32(
                            int32_t(layout.layer_id), seq_id, head,
                            prefix_positions, full_v_data)) {
                    return false;
                }
                llama_kv_compact_matrix full_v(n_prefix_tokens, layout.n_embd_head_v);
                full_v.data = std::move(full_v_data);

                llama_kv_compact_matrix compacted_v;
                if (!llama_kv_compact_fit_values(
                            queries, full_k, full_v,
                            compacted_k, beta, solver_opts,
                            compacted_v)) {
                    return false;
                }
                write_compacted_payload(dst_layer.v_data, layout.type_v,
                                        layout.n_head_kv, n_selected, head,
                                        layout.n_embd_head_v, compacted_v);
            }

            write_compacted_payload(dst_layer.k_data, layout.type_k,
                                    layout.n_head_kv, n_selected, head,
                                    layout.n_embd_head_k, compacted_k);
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
