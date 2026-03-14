#include "llama-kv-compact-prefill-q.h"
#include "llama-kv-compact-query.h"
#include "llama-kv-compact-select.h"
#include "llama-kv-cache.h"
#include "llama-kv-compacted-prefix.h"
#include "llama-context.h"
#include "llama-model.h"

#include "llama-impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

// Gather selected rows from a source matrix.
static bool gather_rows(
        const llama_kv_compact_matrix & src,
        const std::vector<uint32_t> & row_indices,
        llama_kv_compact_matrix & dst) {
    dst.resize((uint32_t) row_indices.size(), src.cols);
    for (size_t i = 0; i < row_indices.size(); ++i) {
        const uint32_t r = row_indices[i];
        if (r >= src.rows) {
            return false;
        }
        std::memcpy(dst.row((uint32_t) i), src.row(r), (size_t) src.cols * sizeof(float));
    }
    return true;
}

// Write solver output into quantized payload.
static void write_payload(
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

    (void) n_head_kv;
}

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
    const uint32_t n_layer     = hparams.n_layer;
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

    // Phase 1: Repeat-prefill with Q-capture.
    const auto & hparams = ctx->get_model().hparams;
    const uint32_t n_layer     = hparams.n_layer;
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
    llama_memory_seq_rm(mem, seq_id, p0, (llama_pos) n_prefix_tokens_input);

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
                if (li == 0 && head == 0) {
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
        /* nnls_lower_bound */ 0.05f,
        /* nnls_upper_bound */ 20.0f,
        /* spectral_ridge   */ false,
    };

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
            if (!gather_rows(entry.k, selected_local, compacted_k)) {
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
                write_payload(dst_layer.v_data, layout.type_v,
                              layout.n_head_kv, n_selected, head,
                              layout.n_embd_head_v, compacted_v);
            }

            write_payload(dst_layer.k_data, layout.type_k,
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
    }

    return true;
}
