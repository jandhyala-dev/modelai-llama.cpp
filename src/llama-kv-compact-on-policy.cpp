// Approximate on-policy compaction pipeline (Section 3.1 / 4.2).
//
// Two-pass approach:
//   Pass 1: Compact with K-as-Q surrogates (standard solver pipeline)
//   Pass 2: Generate continuation tokens from the compacted model,
//           capture Q via cb_eval, re-run solver with real Q
//
// Captures the first-order effect of layer interaction without
// O(n_layers) forward passes.

#include "llama-kv-compact-pipeline.h"
#include "llama-kv-compact-prefill-q.h"
#include "llama-kv-compact-self-study.h"
#include "llama-kv-cache.h"
#include "llama-context.h"
#include "llama-model.h"

#include "llama-impl.h"

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
    if (seq_id < 0 || target_tokens == 0 || live_suffix_pos0 <= p0 || ctx == nullptr) {
        return false;
    }

    // Pass 1: Initial compaction with K-as-Q surrogates.
    llama_kv_compact_pipeline_stats pass1_stats;
    if (!llama_kv_compact_fit_from_live_kv(kv, seq_id, target_tokens,
                                            live_suffix_pos0, &pass1_stats,
                                            p0, max_queries, nnls_iters, lambda)) {
        return false;
    }

    // Pass 2: Generate continuation from compacted model, capture real Q.
    const auto & hparams = ctx->get_model().hparams;
    const uint32_t n_layer     = hparams.n_layer;
    const uint32_t n_embd_head = hparams.n_embd_head_k(0);
    const uint32_t n_head_q    = hparams.n_head(0);

    llama_q_capture_state q_state;
    q_state.reset((int32_t) n_layer, n_embd_head, n_head_q, n_generate_q);

    if (!llama_kv_compact_self_study_generate(ctx, q_state, n_generate_q, seq_id)) {
        // Q-capture generation failed; return Pass 1 result as best effort.
        LLAMA_LOG_WARN("on-policy: Q-capture generation failed, returning Pass 1 result\n");
        if (stats) { *stats = pass1_stats; }
        return true;
    }

    // Pass 2 solver: re-run selection + fitting with captured Q.
    // Delegates to the prefill-Q shared solver which handles:
    //   - Q/K norm normalization
    //   - GQA regrouping
    //   - Fallback to K-as-Q surrogates on per-head failure
    llama_kv_compact_prefill_q_config pass2_config;
    pass2_config.max_queries_per_kv_head = max_queries;
    pass2_config.nnls_iters              = nnls_iters;
    pass2_config.lambda                  = lambda;

    llama_kv_compact_prefill_q_stats pass2_stats;
    if (!llama_kv_compact_prefill_q_with_captured_state(
                kv, seq_id, target_tokens, live_suffix_pos0,
                q_state, pass2_config, &pass2_stats, p0)) {
        // Pass 2 solver failed; return Pass 1 result.
        LLAMA_LOG_WARN("on-policy: Pass 2 solver failed, returning Pass 1 result\n");
        if (stats) { *stats = pass1_stats; }
        return true;
    }

    if (stats) {
        stats->query_generation_time_ms = pass1_stats.query_generation_time_ms
            + pass2_stats.prefill_time_ms;
        stats->solver_time_ms = pass1_stats.solver_time_ms
            + pass2_stats.solver_time_ms;
        stats->n_prefix_tokens = pass2_stats.n_prefix_tokens;
        stats->n_selected_tokens = pass2_stats.n_selected_tokens;
    }

    return true;
}
