#pragma once

// Context-prefill and repeat-prefill Q capture pipelines.
//
// These pipelines capture real post-RoPE Q tensors during a forward pass
// through the prefix tokens, providing higher-quality queries than
// K-as-surrogate for the compaction solver.
//
// - Context-prefill: Q captured during the original prefill (requires caller
//   to install cb_eval before decode — server integration needed).
// - Repeat-prefill: Q captured by re-processing the prefix tokens (doubles
//   prefill cost but is self-contained — no server changes needed).
//
// Reference: arXiv:2602.16284 Section 3.1

#include "llama-kv-compact-self-study.h"
#include "llama-kv-compact-solver.h"
#include "llama.h"

#include <cstdint>

class llama_kv_cache;
struct llama_context;

struct llama_kv_compact_prefill_q_config {
    uint32_t max_queries_per_kv_head = 1024;
    int      nnls_iters              = 2;
    float    lambda                  = 1e-6f;
};

struct llama_kv_compact_prefill_q_stats {
    double   prefill_time_ms      = 0.0;
    double   solver_time_ms       = 0.0;
    uint32_t n_prefix_tokens      = 0;
    uint32_t n_selected_tokens    = 0;
    uint32_t n_queries_per_head   = 0;
    uint32_t n_layers_with_q      = 0;
    float    mean_partition_sum_relative_error = 0.0f;  // Phase 8: mean residual
};

// Repeat-prefill Q-capture compaction pipeline.
//
// Re-processes the prefix tokens with cb_eval to capture real Q tensors,
// then uses them for attention scoring and solver fitting.
//
// Pipeline:
//   1. Re-decode prefix tokens in batches with Q-capture callback installed
//   2. Regroup captured Q per KV head (GQA mapping) + subsample
//   3. Accumulate attention scores → global top-k selection
//   4. Fit beta + V per head using real Q
//
// Caller contract:
//   - prefix_tokens: the original token IDs for positions [p0, live_suffix_pos0)
//   - seq_id MUST be 0 (llama_batch_get_one limitation)
//   - KV cache will be overwritten for prefix positions (same data, harmless)
//
// Returns true on success.
bool llama_kv_compact_prefill_q_from_live_kv(
        struct llama_context * ctx,
        llama_kv_cache       & kv,
        llama_seq_id           seq_id,
        uint32_t               target_tokens,
        llama_pos              live_suffix_pos0,
        const llama_token    * prefix_tokens,
        uint32_t               n_prefix_tokens,
        const llama_kv_compact_prefill_q_config & config,
        llama_kv_compact_prefill_q_stats * stats = nullptr,
        llama_pos p0 = 0);

// Prepare Q-capture state for installation before a prefill decode.
// The caller should:
//   1. Call this to get a configured Q-capture state
//   2. Install the callback via ctx->set_eval_callback(llama_q_capture_eval_callback, &state)
//   3. Run their normal prefill llama_decode()
//   4. Restore previous cb_eval
//   5. Pass the captured state to llama_kv_compact_prefill_q_with_captured_state()
void llama_kv_compact_prepare_q_capture(
        struct llama_context * ctx,
        llama_q_capture_state & q_state_out);

// Run compaction using a pre-captured Q state (from prepare + external prefill).
bool llama_kv_compact_prefill_q_with_captured_state(
        llama_kv_cache       & kv,
        llama_seq_id           seq_id,
        uint32_t               target_tokens,
        llama_pos              live_suffix_pos0,
        llama_q_capture_state & q_state,
        const llama_kv_compact_prefill_q_config & config,
        llama_kv_compact_prefill_q_stats * stats = nullptr,
        llama_pos p0 = 0);

// Re-fit beta and V for a single layer using captured Q, without re-selecting
// positions.  The selected positions from the initial solve are preserved.
//
// Phase 8: Used by sequential on-policy mode to refit one layer at a time
// after generating continuation tokens with Q-capture.
//
// Preconditions:
//   - The live KV cache must NOT have been reclaimed (refit reads full prefix K
//     from the live cache).
//   - Compacted prefix must be configured with selected positions.
//   - il is a model layer ID (not a layout index) — resolved internally.
//
// Returns false if the layer is unmapped (e.g. SWA layer) or solver fails.
// On failure, previous layer data is kept intact.
bool llama_kv_compact_refit_single_layer(
        llama_kv_cache       & kv,
        llama_seq_id           seq_id,
        int32_t                il,
        llama_q_capture_state & q_state,
        const llama_kv_compact_prefill_q_config & config);
