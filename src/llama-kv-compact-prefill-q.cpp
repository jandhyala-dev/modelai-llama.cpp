#include "llama-kv-compact-prefill-q.h"
#include "llama-kv-compact-query.h"
#include "llama-kv-compact-select.h"
#include "llama-kv-compact-shared.h"
#include "llama-kv-cache.h"
#include "llama-kv-compacted-prefix.h"
#include "llama-context.h"
#include "llama-model.h"

#include "llama-impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

// Mean L2 row norm for Q/K scale normalization.
static float row_norm_mean(const llama_kv_compact_matrix & m) {
    if (m.rows == 0 || m.cols == 0) return 0.0f;
    double sum = 0.0;
    for (uint32_t r = 0; r < m.rows; ++r) {
        const float * row = m.row(r);
        float norm_sq = 0.0f;
        for (uint32_t c = 0; c < m.cols; ++c) {
            norm_sq += row[c] * row[c];
        }
        sum += std::sqrt(norm_sq);
    }
    return (float)(sum / m.rows);
}

void llama_kv_compact_prepare_q_capture(
        struct llama_context * ctx,
        llama_q_capture_state & q_state_out) {
    const auto & hparams = ctx->get_model().hparams;
    const uint32_t n_layer     = hparams.n_layer_all;
    const uint32_t n_embd_head = hparams.n_embd_head_k(0);
    const uint32_t n_head_q    = hparams.n_head(0);

    q_state_out.reset((int32_t) n_layer, n_embd_head, n_head_q, 0);
}

bool llama_kv_compact_prefill_q_from_live_kv(
        struct llama_context * ctx,
        llama_kv_cache       & kv,
        llama_seq_id           seq_id,
        uint32_t               target_tokens,
        llama_pos              live_suffix_pos0,
        const llama_token    * prefix_tokens,
        uint32_t               n_prefix_tokens_input,
        const llama_kv_compact_prefill_q_config & config,
        llama_kv_compact_prefill_q_stats * stats,
        llama_pos p0) {

    GGML_ASSERT(seq_id == 0 && "prefill-Q pipeline requires seq_id == 0");

    if (n_prefix_tokens_input == 0 || prefix_tokens == nullptr) {
        return false;
    }

    // llama_batch_get_one assigns positions starting from 0; callers must use p0=0.
    GGML_ASSERT(p0 == 0 && "prefill-Q pipeline requires p0 == 0 (batch position limitation)");

    // Phase 1: Repeat-prefill with Q-capture.
    const auto & hparams = ctx->get_model().hparams;
    const uint32_t n_layer     = hparams.n_layer_all;
    const uint32_t n_embd_head = hparams.n_embd_head_k(0);
    const uint32_t n_head_q    = hparams.n_head(0);

    llama_q_capture_state q_state;
    q_state.reset((int32_t) n_layer, n_embd_head, n_head_q, n_prefix_tokens_input);

    // Save and install Q-capture callback.
    const auto & cparams = ctx->get_cparams();
    auto prev_cb = cparams.cb_eval;
    auto prev_ud = cparams.cb_eval_user_data;

    q_state.active = true;
    ctx->set_eval_callback(llama_q_capture_eval_callback, &q_state);

    // Re-decode prefix tokens. Process in batches to avoid exceeding batch size.
    const uint32_t batch_size = std::min(n_prefix_tokens_input, (uint32_t) 512);
    bool prefill_ok = true;

    // First, remove the existing prefix from KV cache so we can re-decode.
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, seq_id, p0, (llama_pos)(p0 + n_prefix_tokens_input));

    for (uint32_t i = 0; i < n_prefix_tokens_input; i += batch_size) {
        const uint32_t cur_batch = std::min(batch_size, n_prefix_tokens_input - i);

        // Build batch with correct positions.
        llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(prefix_tokens + i), (int32_t) cur_batch);

        if (llama_decode(ctx, batch) != 0) {
            LLAMA_LOG_ERROR("prefill-Q: decode failed at offset %u\n", i);
            prefill_ok = false;
            break;
        }

        q_state.finalize_step();
    }

    // Restore callback.
    q_state.active = false;
    ctx->set_eval_callback(prev_cb, prev_ud);

    if (!prefill_ok) {
        return false;
    }

    // Phase 2-3: Use captured Q for compaction (delegates to shared implementation).
    return llama_kv_compact_prefill_q_with_captured_state(
        kv, seq_id, target_tokens, live_suffix_pos0,
        q_state, config, stats, p0);
}

bool llama_kv_compact_prefill_q_with_captured_state(
        llama_kv_cache       & kv,
        llama_seq_id           seq_id,
        uint32_t               target_tokens,
        llama_pos              live_suffix_pos0,
        llama_q_capture_state & q_state,
        const llama_kv_compact_prefill_q_config & config,
        llama_kv_compact_prefill_q_stats * stats,
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

    const uint32_t n_prefix_tokens = (uint32_t) prefix_positions.size();
    const uint32_t n_selected = std::min<uint32_t>(target_tokens, n_prefix_tokens);

    // Phase 2: Score + select using captured Q.
    std::vector<float> aggregate_scores(n_prefix_tokens, 0.0f);

    struct head_entry {
        llama_kv_compact_matrix k;
        llama_kv_compact_matrix queries;
    };
    std::vector<std::vector<head_entry>> layer_cache(layouts.size());

    const auto t_query_start = std::chrono::steady_clock::now();
    uint32_t actual_queries = 0;
    uint32_t layers_with_q = 0;

    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        layer_cache[li].resize(layout.n_head_kv);

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            auto & entry = layer_cache[li][head];

            // Extract K.
            std::vector<float> k_data;
            if (!kv.compacted_prefix_copy_k_head_f32(
                        (int32_t) layout.layer_id, seq_id, head,
                        prefix_positions, k_data)) {
                return false;
            }
            entry.k.resize(n_prefix_tokens, layout.n_embd_head_k);
            entry.k.data = std::move(k_data);

            // Regroup captured Q for this KV head.
            if (!llama_q_capture_regroup_for_kv_head(
                        q_state, (int32_t) layout.layer_id,
                        head, layout.n_head_kv, entry.queries)) {
                // Fallback to cache-key surrogates if Q capture failed for this layer.
                if (!llama_kv_compact_extract_cache_key_queries(
                            kv, seq_id, (int32_t) layout.layer_id, head,
                            prefix_positions,
                            llama_kv_compact_query_params{ config.max_queries_per_kv_head },
                            entry.queries)) {
                    return false;
                }
            } else {
                if (head == 0) {
                    layers_with_q++;
                }
                llama_q_capture_subsample(entry.queries, config.max_queries_per_kv_head);

                // Q/K norm normalization (same as self-study).
                float q_norm = row_norm_mean(entry.queries);
                float k_norm = row_norm_mean(entry.k);
                if (q_norm > 1e-8f && k_norm > 1e-8f) {
                    float scale = k_norm / q_norm;
                    for (size_t i = 0; i < entry.queries.data.size(); ++i) {
                        entry.queries.data[i] *= scale;
                    }
                }
            }

            if (li == 0 && head == 0) {
                actual_queries = entry.queries.rows;
            }

            llama_kv_compact_accumulate_attention_scores(
                    entry.queries, entry.k, aggregate_scores);
        }
    }
    const auto t_query_end = std::chrono::steady_clock::now();

    // Global selection.
    const std::vector<uint32_t> selected_local = llama_kv_compact_select_topk(aggregate_scores, n_selected);
    std::vector<llama_pos> selected_positions;
    selected_positions.reserve(selected_local.size());
    for (uint32_t idx : selected_local) {
        selected_positions.push_back(prefix_positions[idx]);
    }

    const llama_pos seq_max = kv.seq_pos_max(seq_id);
    const uint32_t logical_token_count = seq_max >= 0 ? (uint32_t)(seq_max + 1) : (uint32_t) live_suffix_pos0;
    if (!kv.compacted_prefix_configure(seq_id, logical_token_count, selected_positions, live_suffix_pos0)) {
        return false;
    }

    auto * seq = kv.get_compacted_prefix()->get_seq(seq_id);
    if (seq == nullptr || !seq->enabled || seq->layers.size() != layouts.size()) {
        return false;
    }

    // Phase 3: Solver.
    const auto t_solver_start = std::chrono::steady_clock::now();
    const llama_kv_compact_solver_opts solver_opts = {
        /* lambda           */ config.lambda,
        /* nnls_iters       */ config.nnls_iters,
        /* nnls_lower_bound */ 1e-12f,
        /* nnls_upper_bound */ 0.0f,
        /* ridge_scale      */ LLAMA_KV_COMPACT_RIDGE_SPECTRAL,
    };

    double residual_sum = 0.0;
    uint32_t residual_count = 0;

    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        auto & dst_layer = seq->layers[li];

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            const auto & entry = layer_cache[li][head];

            std::vector<float> full_v_data;
            if (!kv.compacted_prefix_copy_v_head_f32(
                        (int32_t) layout.layer_id, seq_id, head,
                        prefix_positions, full_v_data)) {
                return false;
            }
            llama_kv_compact_matrix full_v(n_prefix_tokens, layout.n_embd_head_v);
            full_v.data = std::move(full_v_data);

            llama_kv_compact_matrix compacted_k;
            if (!gather_matrix_rows(entry.k, selected_local, compacted_k)) {
                return false;
            }

            float head_residual = 0.0f;
            std::vector<float> beta;
            bool beta_ok;
            if (llama_kv_compact_skip_beta_fit()) {
                beta.assign(n_selected, 0.0f);
                beta_ok = true;
            } else {
                beta_ok = llama_kv_compact_fit_beta(entry.queries, entry.k,
                                                      compacted_k, solver_opts,
                                                      beta, &head_residual);
                // NaN guard (GAP-K): fall back to zero-beta on solver failure.
                if (!beta_ok) {
                    LLAMA_LOG_WARN("prefill-Q: beta fitting failed for layout %zu head %u — falling back to zero-beta\n",
                                   li, head);
                    beta.assign(n_selected, 0.0f);
                    head_residual = 0.0f;
                }
            }
            residual_sum += head_residual;
            residual_count++;

            if (layout.n_embd_head_v > 0) {
                llama_kv_compact_matrix compacted_v;
                bool v_ok = beta_ok && !llama_kv_compact_skip_cv_fit() &&
                            llama_kv_compact_fit_values(
                            entry.queries, entry.k, full_v,
                            compacted_k, beta, solver_opts,
                            compacted_v);
                if (!v_ok) {
                    // Fall back to original V values at selected positions.
                    if (beta_ok && !llama_kv_compact_skip_cv_fit()) {
                        LLAMA_LOG_WARN("prefill-Q: V fitting failed for layout %zu head %u — using original V\n",
                                       li, head);
                    }
                    if (!gather_matrix_rows(full_v, selected_local, compacted_v)) {
                        return false;
                    }
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
        stats->prefill_time_ms    = std::chrono::duration<double, std::milli>(t_query_end - t_query_start).count();
        stats->solver_time_ms     = std::chrono::duration<double, std::milli>(t_solver_end - t_solver_start).count();
        stats->n_prefix_tokens    = n_prefix_tokens;
        stats->n_selected_tokens  = n_selected;
        stats->n_queries_per_head = actual_queries;
        stats->n_layers_with_q    = layers_with_q;
        stats->mean_partition_sum_relative_error = residual_count > 0
            ? (float)(residual_sum / residual_count) : 0.0f;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Phase 8: Single-layer refit
// ---------------------------------------------------------------------------

bool llama_kv_compact_refit_single_layer(
        llama_kv_cache       & kv,
        llama_seq_id           seq_id,
        int32_t                il,
        llama_q_capture_state & q_state,
        const llama_kv_compact_prefill_q_config & config) {

    if (seq_id < 0) {
        return false;
    }

    // Precondition: execution must be enabled so that Q-capture generation
    // used the compacted prefix, making the captured Q reflect the current
    // compacted state (sequential dependency).
    GGML_ASSERT(kv.compacted_prefix_execution_enabled(seq_id));

    // Resolve model layer ID → layout.
    llama_compacted_prefix_layer_layout layout;
    if (!kv.compacted_prefix_layer_layout_for_solver(il, layout)) {
        return false;  // unmapped layer (e.g. SWA)
    }

    auto * store = kv.get_compacted_prefix();
    if (!store) {
        return false;
    }
    auto * seq = store->get_seq(seq_id);
    if (!seq || !seq->enabled || seq->layers.empty()) {
        return false;
    }

    // Find the layer storage entry matching this model layer ID.
    const auto & layouts = store->get_layouts();
    size_t ikv = SIZE_MAX;
    for (size_t i = 0; i < layouts.size(); ++i) {
        if (layouts[i].layer_id == (uint32_t) il) {
            ikv = i;
            break;
        }
    }
    if (ikv == SIZE_MAX || ikv >= seq->layers.size()) {
        return false;
    }

    auto & dst_layer = seq->layers[ikv];
    const uint32_t n_selected = dst_layer.n_compacted_tokens;
    if (n_selected == 0) {
        return false;
    }

    // Get the selected positions from initial solve.
    const auto & selected_positions = seq->logical_positions;
    if (selected_positions.size() != n_selected) {
        return false;
    }

    // Get all prefix positions (needed for full K extraction).
    std::vector<llama_pos> all_prefix_positions;
    if (!kv.compacted_prefix_seq_positions(seq_id, 0, seq->live_suffix_pos0, all_prefix_positions)) {
        return false;
    }
    if (all_prefix_positions.empty()) {
        return false;
    }
    const uint32_t n_prefix = (uint32_t) all_prefix_positions.size();

    // Build index mapping: for each selected position, find its index in all_prefix_positions.
    std::vector<uint32_t> selected_local_indices;
    selected_local_indices.reserve(n_selected);
    for (const auto & sel_pos : selected_positions) {
        bool found = false;
        for (uint32_t j = 0; j < n_prefix; ++j) {
            if (all_prefix_positions[j] == sel_pos) {
                selected_local_indices.push_back(j);
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }

    const llama_kv_compact_solver_opts solver_opts = {
        /* lambda           */ config.lambda,
        /* nnls_iters       */ config.nnls_iters,
        /* nnls_lower_bound */ 1e-12f,
        /* nnls_upper_bound */ 0.0f,
        /* ridge_scale      */ LLAMA_KV_COMPACT_RIDGE_SPECTRAL,
    };

    for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
        // Extract full K from live cache.
        std::vector<float> full_k_data;
        if (!kv.compacted_prefix_copy_k_head_f32(il, seq_id, head,
                    all_prefix_positions, full_k_data)) {
            return false;
        }
        llama_kv_compact_matrix full_k(n_prefix, layout.n_embd_head_k);
        full_k.data = std::move(full_k_data);

        // Regroup captured Q for this KV head.
        llama_kv_compact_matrix queries;
        if (!llama_q_capture_regroup_for_kv_head(
                    q_state, il, head, layout.n_head_kv, queries)) {
            // Fallback to cache-key surrogates.
            if (!llama_kv_compact_extract_cache_key_queries(
                        kv, seq_id, il, head,
                        all_prefix_positions,
                        llama_kv_compact_query_params{ config.max_queries_per_kv_head },
                        queries)) {
                return false;
            }
        } else {
            llama_q_capture_subsample(queries, config.max_queries_per_kv_head);

            // Q/K norm normalization.
            float q_norm = row_norm_mean(queries);
            float k_norm = row_norm_mean(full_k);
            if (q_norm > 1e-8f && k_norm > 1e-8f) {
                float scale = k_norm / q_norm;
                for (size_t i = 0; i < queries.data.size(); ++i) {
                    queries.data[i] *= scale;
                }
            }
        }

        // Gather compacted K rows.
        llama_kv_compact_matrix compacted_k;
        if (!gather_matrix_rows(full_k, selected_local_indices, compacted_k)) {
            return false;
        }

        // Fit beta.
        std::vector<float> beta;
        if (!llama_kv_compact_fit_beta(queries, full_k,
                                        compacted_k, solver_opts,
                                        beta, nullptr)) {
            return false;
        }

        // Fit V.
        if (layout.n_embd_head_v > 0) {
            std::vector<float> full_v_data;
            if (!kv.compacted_prefix_copy_v_head_f32(il, seq_id, head,
                        all_prefix_positions, full_v_data)) {
                return false;
            }
            llama_kv_compact_matrix full_v(n_prefix, layout.n_embd_head_v);
            full_v.data = std::move(full_v_data);

            llama_kv_compact_matrix compacted_v;
            if (!llama_kv_compact_fit_values(queries, full_k, full_v,
                                              compacted_k, beta, solver_opts,
                                              compacted_v)) {
                return false;
            }
            write_compacted_payload(dst_layer.v_data, layout.type_v,
                          layout.n_head_kv, n_selected, head,
                          layout.n_embd_head_v, compacted_v);
        }

        // Write K and beta.
        write_compacted_payload(dst_layer.k_data, layout.type_k,
                      layout.n_head_kv, n_selected, head,
                      layout.n_embd_head_k, compacted_k);
        for (uint32_t token = 0; token < n_selected; ++token) {
            dst_layer.beta_data[size_t(head) * n_selected + token] = beta[token];
        }
    }

    // Update zero-beta cache and invalidate tensor cache.
    dst_layer.update_zero_beta_cache();
    kv.compacted_prefix_bump_version();

    return true;
}
