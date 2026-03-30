// Public C API for KV cache compaction (arXiv:2602.16284 Attention Matching).
//
// Thin wrapper around the internal pipeline methods exposed via llama_kv_cache.
// See include/llama.h for usage documentation.

#include "llama.h"
#include "llama-impl.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-kv-cache.h"
#include "llama-kv-compact-utils.h"
#include "llama-kv-compact-pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

// Get the base llama_kv_cache from a context (handles plain, iSWA, hybrid).
static llama_kv_cache * get_kv_cache(llama_context * ctx) {
    return llama_kv_compact_get_cache(ctx->get_memory());
}

struct llama_compact_params llama_compact_default_params(void) {
    struct llama_compact_params params;
    params.method           = LLAMA_COMPACT_METHOD_SELECT;
    params.target_tokens    = -1;
    params.ratio            = 2.0f;
    params.live_suffix_tokens = 0;
    params.p0               = 0;
    params.max_queries      = 0;  // 0 = auto
    params.nnls_iters       = -1; // -1 = auto
    params.lambda           = -1.0f; // < 0 = auto
    params.reclaim          = true;
    return params;
}

int32_t llama_kv_cache_compact(
        struct llama_context * ctx,
        llama_seq_id           seq_id,
        struct llama_compact_params params) {
    if (!ctx) {
        LLAMA_LOG_ERROR("%s: ctx is null\n", __func__);
        return -1;
    }

    auto * kv = get_kv_cache(ctx);
    if (!kv) {
        LLAMA_LOG_ERROR("%s: no KV cache available\n", __func__);
        return -1;
    }
    if (!kv->supports_compaction()) {
        const auto reason = kv->compaction_unsupported_reason();
        LLAMA_LOG_ERROR("%s: compaction not supported: %s\n", __func__,
                        reason.empty() ? "unknown" : reason.c_str());
        return -1;
    }

    // Determine method.
    const enum llama_compact_method method = params.method;

    // Determine compactable range.
    const llama_pos pos_max = kv->seq_pos_max(seq_id);
    if (pos_max < 0) {
        LLAMA_LOG_ERROR("%s: sequence %d has no tokens\n", __func__, seq_id);
        return -1;
    }
    const llama_pos live_suffix_pos0 = pos_max + 1 - params.live_suffix_tokens;
    if (live_suffix_pos0 < 0) {
        LLAMA_LOG_ERROR("%s: live_suffix_tokens (%d) exceeds sequence length (%d)\n",
                        __func__, params.live_suffix_tokens, pos_max + 1);
        return -1;
    }
    if (live_suffix_pos0 <= params.p0) {
        LLAMA_LOG_ERROR("%s: no compactable tokens (pos_max=%d, live_suffix=%d, p0=%d)\n",
                        __func__, pos_max, params.live_suffix_tokens, params.p0);
        return -1;
    }
    const uint32_t compactable = (uint32_t)(live_suffix_pos0 - params.p0);

    // Compute base requested target.
    uint32_t requested_target_tokens;
    bool explicit_target = false;
    if (params.target_tokens > 0) {
        explicit_target = true;
        requested_target_tokens = (uint32_t)params.target_tokens;
    } else {
        if (params.ratio < 1.0f) {
            LLAMA_LOG_ERROR("%s: ratio must be >= 1.0 (got %.2f)\n", __func__, params.ratio);
            return -1;
        }
        requested_target_tokens = std::max(2u, (uint32_t)std::ceil((double)compactable / (double)params.ratio));
    }

    // Resolve hybrid-aware effective budget using the shared helper.
    const auto & hparams = ctx->get_model().hparams;
    const auto hybrid_info = llama_kv_compact_detect_hybrid(hparams, kv->get_compacted_prefix());
    const auto budget = llama_kv_compact_resolve_budget(
        hybrid_info,
        compactable,
        requested_target_tokens,
        explicit_target,
        explicit_target ? 0.0 : (double) params.ratio);

    const auto active_prefix = llama_kv_compact_get_active_prefix_counts(kv, seq_id);
    if (active_prefix.compacted_tokens > 0) {
        LLAMA_LOG_INFO("%s: compacted prefix already active for seq %d, keeping %u / %u compacted tokens\n",
                       __func__, seq_id,
                       active_prefix.compacted_tokens,
                       active_prefix.logical_tokens);
        return (int32_t) active_prefix.compacted_tokens;
    }

    if (!explicit_target && budget.skipped_noop) {
        LLAMA_LOG_INFO("%s: hybrid no-op (seq %d: %u -> %u, scale=%.2f)\n",
                       __func__, seq_id,
                       budget.requested_target_tokens,
                       budget.effective_target_tokens,
                       budget.budget_scale);
        return (int32_t)compactable;
    }

    const uint32_t target_tokens = budget.effective_target_tokens;
    if (target_tokens >= compactable) {
        LLAMA_LOG_WARN("%s: target_tokens=%u >= compactable=%u, nothing to compact\n",
                       __func__, target_tokens, compactable);
        return (int32_t)compactable;
    }

    // Resolve auto-tuning sentinels.
    const bool high_compression = (params.ratio >= 10.0f ||
                                   (params.target_tokens > 0 && compactable / (float)params.target_tokens >= 10.0f));
    uint32_t max_queries = params.max_queries > 0 ? params.max_queries : (high_compression ? 512u : 256u);
    int nnls_iters       = params.nnls_iters >= 0 ? params.nnls_iters : (high_compression ? 4 : 2);
    float lambda         = params.lambda >= 0.0f ? params.lambda : (high_compression ? 1e-5f : 1e-6f);

    // Run compaction.
    llama_kv_compact_pipeline_stats stats = {};
    bool ok = false;

    static const char * method_names[] = { "select", "solver", "omp", "nonuniform", "chunked" };
    const char * method_name = (method >= 0 && method <= LLAMA_COMPACT_METHOD_CHUNKED) ? method_names[method] : "unknown";

    switch (method) {
        case LLAMA_COMPACT_METHOD_SELECT:
            ok = kv->compacted_prefix_select_from_live_kv(
                seq_id, target_tokens, live_suffix_pos0, &stats, params.p0);
            break;
        case LLAMA_COMPACT_METHOD_SOLVER:
            ok = kv->compacted_prefix_fit_from_live_kv(
                seq_id, target_tokens, live_suffix_pos0, &stats, params.p0,
                max_queries, nnls_iters, lambda);
            break;
        case LLAMA_COMPACT_METHOD_OMP:
            ok = kv->compacted_prefix_omp_from_live_kv(
                seq_id, target_tokens, live_suffix_pos0, &stats, params.p0,
                max_queries, nnls_iters, lambda);
            break;
        case LLAMA_COMPACT_METHOD_NONUNIFORM:
            ok = kv->compacted_prefix_nonuniform_from_live_kv(
                seq_id, target_tokens, live_suffix_pos0, &stats, params.p0,
                max_queries, nnls_iters, lambda);
            break;
        case LLAMA_COMPACT_METHOD_CHUNKED:
            ok = kv->compacted_prefix_chunked_from_live_kv(
                seq_id, target_tokens, live_suffix_pos0, &stats, params.p0,
                max_queries, nnls_iters, lambda);
            break;
        default:
            LLAMA_LOG_ERROR("%s: unknown method %d (supported: select=0, solver=1, omp=2, nonuniform=3, chunked=4)\n",
                            __func__, (int)method);
            return -1;
    }

    if (!ok) {
        LLAMA_LOG_ERROR("%s: compaction failed (method=%s, target=%u)\n",
                        __func__, method_name, target_tokens);
        return -1;
    }

    // Enable compacted prefix execution.
    if (!kv->compacted_prefix_set_execution(seq_id, true)) {
        LLAMA_LOG_ERROR("%s: failed to enable compacted prefix execution\n", __func__);
        return -1;
    }

    // Optionally reclaim live KV cells.
    if (params.reclaim) {
        kv->compacted_prefix_reclaim_live_kv(seq_id);
    }

    LLAMA_LOG_INFO("%s: compacted seq %d: %u -> %u tokens (method=%s, prefix_ratio=%.1fx)\n",
                   __func__, seq_id, compactable, target_tokens, method_name,
                   (float)compactable / target_tokens);

    return (int32_t)target_tokens;
}

// NOTE: Not thread-safe — must be called before inference begins or with external synchronization.
void llama_kv_cache_set_auto_compact(
        struct llama_context       * ctx,
               float                 ratio,
        struct llama_compact_params   params) {
    if (!ctx) {
        return;
    }
    if (ratio <= 0.0f) {
        ctx->set_auto_compact(false, 0.0f, llama_compact_default_params());
        LLAMA_LOG_INFO("%s: auto-compaction disabled\n", __func__);
        return;
    }
    ctx->set_auto_compact(true, ratio, params);
    LLAMA_LOG_INFO("%s: auto-compaction enabled (ratio=%.1f, method=%d)\n",
                   __func__, ratio, (int)params.method);
}
