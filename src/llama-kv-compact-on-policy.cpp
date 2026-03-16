// On-policy compaction pipelines (Section 3.1 / 4.2, Phase 8 extensions).
//
// Original 2-pass:
//   Pass 1: Compact with K-as-Q surrogates (standard solver pipeline)
//   Pass 2: Generate continuation, capture Q via cb_eval, re-run solver
//
// Phase 8 additions:
//   - Iterative on-policy: wraps Pass 2 in a loop with quality gate
//   - Per-layer sequential: refits one layer at a time with layer-dependent Q

#include "llama-kv-compact-on-policy.h"
#include "llama-kv-compact-pipeline.h"
#include "llama-kv-compact-prefill-q.h"
#include "llama-kv-compact-self-study.h"
#include "llama-kv-cache.h"
#include "llama-kv-compacted-prefix.h"
#include "llama-context.h"
#include "llama-model.h"

#include "llama-impl.h"

#include <algorithm>
#include <chrono>

// Legacy 2-pass on-policy (kept for backward compatibility).
bool llama_kv_compact_on_policy_from_live_kv(
        struct llama_context * ctx,
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats,
        llama_pos p0,
        uint32_t max_queries,
        int nnls_iters,
        float lambda,
        uint32_t n_generate_q) {
    llama_kv_compact_on_policy_config config;
    config.n_on_policy_passes      = 1;
    config.n_generate              = n_generate_q;
    config.max_queries             = max_queries;
    config.max_queries_per_kv_head = max_queries;
    config.nnls_iters              = nnls_iters;
    config.lambda                  = lambda;

    return llama_kv_compact_iterative_on_policy_from_live_kv(
            ctx, kv, seq_id, target_tokens, live_suffix_pos0, stats, p0, config);
}

// ---------------------------------------------------------------------------
// Iterative on-policy (Phase 8)
// ---------------------------------------------------------------------------

bool llama_kv_compact_iterative_on_policy_from_live_kv(
        struct llama_context * ctx,
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats,
        llama_pos p0,
        const llama_kv_compact_on_policy_config & config) {

    if (seq_id < 0 || target_tokens == 0 || live_suffix_pos0 <= p0 || ctx == nullptr) {
        return false;
    }

    const auto t_total_start = std::chrono::steady_clock::now();

    // Pass 0 (K-as-Q): initial compaction.
    llama_kv_compact_pipeline_stats pass0_stats;
    if (!llama_kv_compact_fit_from_live_kv(kv, seq_id, target_tokens,
                                            live_suffix_pos0, &pass0_stats,
                                            p0, config.max_queries,
                                            config.nnls_iters, config.lambda)) {
        return false;
    }

    float r_prev = pass0_stats.mean_partition_sum_relative_error;
    double total_query_gen_ms = pass0_stats.query_generation_time_ms;
    double total_solver_ms    = pass0_stats.solver_time_ms;
    uint32_t n_prefix_tokens  = pass0_stats.n_prefix_tokens;
    uint32_t n_selected       = pass0_stats.n_selected_tokens;

    if (config.n_on_policy_passes == 0) {
        // K-as-Q only mode.
        if (stats) {
            *stats = pass0_stats;
        }
        return true;
    }

    // On-policy passes.
    const auto & hparams = ctx->get_model().hparams;
    const uint32_t n_layer     = hparams.n_layer;
    const uint32_t n_embd_head = hparams.n_embd_head_k(0);
    const uint32_t n_head_q    = hparams.n_head(0);

    for (uint32_t pass = 0; pass < config.n_on_policy_passes; ++pass) {
        // Enable compacted execution for Q-capture generation.
        if (!kv.compacted_prefix_set_execution(seq_id, true)) {
            LLAMA_LOG_WARN("on-policy: failed to enable execution at pass %u\n", pass);
            break;
        }

        // Generate continuation with Q-capture.
        llama_q_capture_state q_state;
        q_state.reset((int32_t) n_layer, n_embd_head, n_head_q, config.n_generate);

        const auto t_gen_start = std::chrono::steady_clock::now();
        if (!llama_kv_compact_self_study_generate(ctx, q_state, config.n_generate, seq_id)) {
            LLAMA_LOG_WARN("on-policy: Q-capture generation failed at pass %u, keeping previous result\n", pass);
            break;
        }
        const auto t_gen_end = std::chrono::steady_clock::now();
        total_query_gen_ms += std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();

        // Re-solve all layers with captured Q.
        // Note: prefill_q_with_captured_state() internally calls configure_seq(),
        // which clears execution state.
        llama_kv_compact_prefill_q_config pass_config;
        pass_config.max_queries_per_kv_head = config.max_queries_per_kv_head;
        pass_config.nnls_iters              = config.nnls_iters;
        pass_config.lambda                  = config.lambda;

        llama_kv_compact_prefill_q_stats pass_stats;
        if (!llama_kv_compact_prefill_q_with_captured_state(
                    kv, seq_id, target_tokens, live_suffix_pos0,
                    q_state, pass_config, &pass_stats, p0)) {
            LLAMA_LOG_WARN("on-policy: solver failed at pass %u, keeping previous result\n", pass);
            break;
        }

        total_solver_ms += pass_stats.solver_time_ms;
        n_prefix_tokens = pass_stats.n_prefix_tokens;
        n_selected      = pass_stats.n_selected_tokens;

        float r_cur = pass_stats.mean_partition_sum_relative_error;

        // Quality gate: stop if residual didn't improve enough.
        float denom = std::max(r_prev, 1e-12f);
        float improvement = (r_prev - r_cur) / denom;
        if (improvement < config.quality_min_improvement) {
            LLAMA_LOG_INFO("on-policy: pass %u residual not improved (%.6f -> %.6f, delta %.4f%%), stopping\n",
                           pass, r_prev, r_cur, improvement * 100.0f);
            break;
        }

        LLAMA_LOG_INFO("on-policy: pass %u residual improved %.6f -> %.6f (%.2f%%)\n",
                       pass, r_prev, r_cur, improvement * 100.0f);
        r_prev = r_cur;
    }

    const auto t_total_end = std::chrono::steady_clock::now();

    if (stats) {
        stats->query_generation_time_ms = total_query_gen_ms;
        stats->solver_time_ms           = total_solver_ms;
        stats->n_prefix_tokens          = n_prefix_tokens;
        stats->n_selected_tokens        = n_selected;
        stats->mean_partition_sum_relative_error = r_prev;
        stats->total_time_ms = std::chrono::duration<double, std::milli>(
                t_total_end - t_total_start).count();
    }

    return true;
}

// ---------------------------------------------------------------------------
// Per-layer sequential on-policy (Phase 8, experimental)
// ---------------------------------------------------------------------------

bool llama_kv_compact_sequential_on_policy_from_live_kv(
        struct llama_context * ctx,
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats,
        llama_pos p0,
        const llama_kv_compact_sequential_config & config) {

    if (seq_id < 0 || target_tokens == 0 || live_suffix_pos0 <= p0 || ctx == nullptr) {
        return false;
    }

    const auto t_total_start = std::chrono::steady_clock::now();

    // Initial solve (all layers, K-as-Q).
    llama_kv_compact_pipeline_stats pass0_stats;
    if (!llama_kv_compact_fit_from_live_kv(kv, seq_id, target_tokens,
                                            live_suffix_pos0, &pass0_stats,
                                            p0, 256 /* max_queries */,
                                            config.nnls_iters, config.lambda)) {
        return false;
    }

    double total_query_gen_ms = pass0_stats.query_generation_time_ms;
    double total_solver_ms    = pass0_stats.solver_time_ms;

    // Enable execution for sequential per-layer refinement.
    if (!kv.compacted_prefix_set_execution(seq_id, true)) {
        if (stats) { *stats = pass0_stats; }
        return true;
    }

    const auto * store = kv.get_compacted_prefix();
    const auto & layouts = store->get_layouts();

    const auto & hparams = ctx->get_model().hparams;
    const uint32_t n_layer     = hparams.n_layer;
    const uint32_t n_embd_head = hparams.n_embd_head_k(0);
    const uint32_t n_head_q    = hparams.n_head(0);

    uint32_t layers_refit = 0;

    for (size_t idx = 0; idx < layouts.size(); ++idx) {
        const int32_t model_layer_id = (int32_t) layouts[idx].layer_id;

        // Generate continuation with Q-capture.
        llama_q_capture_state q_state;
        q_state.reset((int32_t) n_layer, n_embd_head, n_head_q, config.n_generate_q);

        const auto t_gen_start = std::chrono::steady_clock::now();
        if (!llama_kv_compact_self_study_generate(ctx, q_state, config.n_generate_q, seq_id)) {
            LLAMA_LOG_WARN("sequential on-policy: Q-capture failed at layout %zu (layer %d), stopping\n",
                           idx, model_layer_id);
            break;
        }
        const auto t_gen_end = std::chrono::steady_clock::now();
        total_query_gen_ms += std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();

        // Refit this single layer.
        llama_kv_compact_prefill_q_config refit_config;
        refit_config.max_queries_per_kv_head = config.max_queries_per_kv_head;
        refit_config.nnls_iters              = config.nnls_iters;
        refit_config.lambda                  = config.lambda;

        const auto t_solver_start = std::chrono::steady_clock::now();
        if (!llama_kv_compact_refit_single_layer(kv, seq_id, model_layer_id,
                                                  q_state, refit_config)) {
            LLAMA_LOG_WARN("sequential on-policy: refit failed at layout %zu (layer %d), skipping\n",
                           idx, model_layer_id);
            continue;
        }
        const auto t_solver_end = std::chrono::steady_clock::now();
        total_solver_ms += std::chrono::duration<double, std::milli>(t_solver_end - t_solver_start).count();

        layers_refit++;
    }

    // Disable execution after sequential loop.
    kv.compacted_prefix_set_execution(seq_id, false);

    const auto t_total_end = std::chrono::steady_clock::now();

    LLAMA_LOG_INFO("sequential on-policy: refit %u / %zu layers\n",
                   layers_refit, layouts.size());

    if (stats) {
        stats->query_generation_time_ms = total_query_gen_ms;
        stats->solver_time_ms           = total_solver_ms;
        stats->n_prefix_tokens          = pass0_stats.n_prefix_tokens;
        stats->n_selected_tokens        = pass0_stats.n_selected_tokens;
        stats->total_time_ms = std::chrono::duration<double, std::milli>(
                t_total_end - t_total_start).count();
    }

    return true;
}
