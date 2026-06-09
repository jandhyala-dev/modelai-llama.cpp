#pragma once

// Centralized KV cache extraction from any memory backend.
// Eliminates the dynamic_cast cascade duplicated across API, server, bench, and tests.
// When new memory types are added, update only this function.

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-compacted-prefix.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-hparams.h"
#include "llama-impl.h"

#include <algorithm>
#include <cstdint>

static inline llama_kv_cache * llama_kv_compact_get_cache(llama_memory_i * mem) {
    if (!mem) return nullptr;
    if (auto * kv = dynamic_cast<llama_kv_cache *>(mem)) return kv;
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(mem)) return iswa->get_base();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) return hybrid->get_mem_attn();
    if (auto * hiswa = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) return hiswa->get_mem_attn()->get_base();
    return nullptr;
}

struct llama_kv_compact_active_prefix_counts {
    uint32_t compacted_tokens = 0;
    uint32_t logical_tokens   = 0;
};

static inline llama_kv_compact_active_prefix_counts llama_kv_compact_get_active_prefix_counts(
        const llama_kv_cache * kv,
        llama_seq_id seq_id) {
    if (kv == nullptr || !kv->compacted_prefix_execution_enabled(seq_id)) {
        return {};
    }

    const auto * store = kv->get_compacted_prefix();
    const auto * state = store ? store->get_seq(seq_id) : nullptr;
    if (state == nullptr || !state->enabled || !state->is_execution_enabled()) {
        return {};
    }

    llama_kv_compact_active_prefix_counts out;
    out.compacted_tokens = state->compacted_token_count();
    out.logical_tokens   = state->live_suffix_pos0 >= 0
        ? (uint32_t) state->live_suffix_pos0
        : state->logical_token_count;
    return out;
}

// --- Hybrid architecture detection ---

struct llama_kv_compact_hybrid_info {
    uint32_t n_total_layers       = 0;
    uint32_t n_recurrent_layers   = 0;
    uint32_t n_attn_layers        = 0;
    uint32_t n_compactable_layers = 0;
    bool     is_hybrid            = false;
    bool     layout_count_mismatch = false;
    float    compactable_fraction = 1.0f;
};

// Pure helper — testable without constructing a full llama_kv_cache.
static inline llama_kv_compact_hybrid_info llama_kv_compact_make_hybrid_info(
        uint32_t n_total_layers,
        uint32_t n_recurrent_layers,
        uint32_t n_compactable_layers) {
    llama_kv_compact_hybrid_info info;
    const uint32_t exclusive_attn_layers = n_total_layers >= n_recurrent_layers
        ? (n_total_layers - n_recurrent_layers)
        : 0;

    info.n_total_layers       = n_total_layers;
    info.n_recurrent_layers   = n_recurrent_layers;
    // Some hybrids (for example Falcon-H1) combine recurrent state and
    // attention in the same layer. The compactable-layout count is therefore
    // the minimum reliable count of attention-bearing layers.
    info.n_attn_layers        = std::min(
        n_total_layers,
        std::max(exclusive_attn_layers, n_compactable_layers));
    info.n_compactable_layers = n_compactable_layers;
    info.is_hybrid            = (info.n_recurrent_layers > 0);
    info.layout_count_mismatch = (info.n_compactable_layers != info.n_attn_layers);

    if (info.n_total_layers > 0) {
        info.compactable_fraction = float(info.n_compactable_layers) / float(info.n_total_layers);
    } else {
        info.compactable_fraction = 0.0f;
    }
    return info;
}

// Wrapper that reads live architecture metadata and calls the pure helper.
// Takes hparams + compacted prefix directly (hparams is private on llama_kv_cache).
static inline llama_kv_compact_hybrid_info llama_kv_compact_detect_hybrid(
        const llama_hparams & hparams,
        const llama_compacted_prefix_store * cp) {
    if (!cp) {
        return {};
    }

    const auto & layouts = cp->get_layouts();

    uint32_t n_recurrent_layers = 0;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        if (hparams.is_recr(il)) {
            n_recurrent_layers++;
        }
    }

    auto info = llama_kv_compact_make_hybrid_info(
        hparams.n_layer_all,
        n_recurrent_layers,
        (uint32_t) layouts.size());

    if (info.layout_count_mismatch) {
        LLAMA_LOG_WARN(
            "%s: hybrid layer mismatch (attn=%u, compactable=%u, total=%u)\n",
            __func__,
            info.n_attn_layers,
            info.n_compactable_layers,
            info.n_total_layers);
    }

    return info;
}

// --- Shared budget resolution ---

struct llama_kv_compact_budget_resolution {
    uint32_t requested_target_tokens = 0;
    uint32_t effective_target_tokens = 0;
    double   requested_ratio         = 0.0;
    double   effective_ratio         = 0.0;
    bool     explicit_target         = false;
    bool     hybrid_detected         = false;
    bool     skipped_noop            = false;
    float    budget_scale            = 1.0f;
    llama_kv_compact_hybrid_info hybrid = {};
};

// Shared helper used by both the server path and the public C API.
// Resolves the effective compaction budget for hybrid architectures.
// The hybrid_info should be computed via llama_kv_compact_detect_hybrid() first.
static inline llama_kv_compact_budget_resolution llama_kv_compact_resolve_budget(
        const llama_kv_compact_hybrid_info & hybrid_info,
        uint32_t compactable,
        uint32_t requested_target_tokens,
        bool explicit_target,
        double requested_ratio) {
    llama_kv_compact_budget_resolution out;
    out.requested_target_tokens = requested_target_tokens;
    out.effective_target_tokens = requested_target_tokens;
    out.explicit_target         = explicit_target;
    out.requested_ratio         = requested_ratio;
    out.effective_ratio         = requested_ratio;
    out.hybrid                  = hybrid_info;

    if (compactable == 0 || requested_target_tokens == 0) {
        return out;
    }

    if (explicit_target) {
        out.requested_ratio = 0.0;
        out.effective_ratio = 0.0;
        return out;
    }

    out.hybrid_detected = hybrid_info.is_hybrid && hybrid_info.n_compactable_layers > 0;
    if (!out.hybrid_detected || hybrid_info.compactable_fraction <= 0.0f) {
        return out;
    }

    // Scale the target proportionally to the attention fraction, capped at 4.0x
    // to avoid over-relaxing the budget for extremely sparse hybrids (e.g., 6/52 layers).
    const float scale = std::min(1.0f / hybrid_info.compactable_fraction, 4.0f);
    out.budget_scale = scale;

    const uint32_t scaled_target = (uint32_t) std::min(
        (float) requested_target_tokens * scale,
        (float) compactable);

    if (scaled_target >= compactable - 1u) {
        out.effective_target_tokens = compactable;
        out.effective_ratio = 1.0;
        out.skipped_noop = true;
        return out;
    }

    out.effective_target_tokens = scaled_target;
    out.effective_ratio = compactable > 0
        ? double(compactable) / double(out.effective_target_tokens)
        : 0.0;
    return out;
}
