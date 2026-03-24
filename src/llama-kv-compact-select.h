#pragma once

#include "llama-kv-compact-solver.h"

#include <cstdint>
#include <vector>

// Score aggregation mode for key selection.
// Reference: arXiv:2602.16284 Appendix F.1
enum llama_kv_compact_score_agg {
    LLAMA_KV_COMPACT_SCORE_AGG_SUM  = 0, // Additive (default)
    LLAMA_KV_COMPACT_SCORE_AGG_RMS  = 1, // Root-mean-square (more robust at extreme ratios)
    LLAMA_KV_COMPACT_SCORE_AGG_MAX  = 2, // Per-key maximum across queries (paper Section 3.3)
    LLAMA_KV_COMPACT_SCORE_AGG_MEAN = 3, // Mean (SUM normalized by n_queries, equivalent after normalization)
};

// Accumulate attention-based importance scores for key positions.
//
// SUM mode: scores_inout[ki] += softmax(q,K)[ki]  (existing behavior)
// RMS mode: scores_inout[ki] += softmax(q,K)[ki]^2 (call finalize_rms after all heads)
//
// n_queries_out: if non-null, incremented by queries.rows (needed for RMS finalization)
void llama_kv_compact_accumulate_attention_scores(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & keys,
        std::vector<float> & scores_inout,
        llama_kv_compact_score_agg agg = LLAMA_KV_COMPACT_SCORE_AGG_SUM,
        uint32_t * n_queries_out = nullptr);

// Finalize RMS-aggregated scores: apply sqrt(score / n_queries) in-place.
// Must be called after all accumulation calls when using AGG_RMS.
// No-op if n_queries == 0.
void llama_kv_compact_finalize_rms_scores(
        std::vector<float> & scores,
        uint32_t n_queries);

// Finalize MEAN-aggregated scores: divide by n_queries in-place.
// Must be called after all accumulation calls when using AGG_MEAN.
// No-op if n_queries == 0.
void llama_kv_compact_finalize_mean_scores(
        std::vector<float> & scores,
        uint32_t n_queries);

// Apply 1D average pooling over position-sorted scores for noise reduction.
// Reference: MIT highest_attention_keys.py — avgpool with kernel_size.
// No-op if kernel_size <= 1 or scores has fewer than 2 elements.
void llama_kv_compact_avgpool_scores(
        std::vector<float> & scores,
        uint32_t kernel_size);

std::vector<uint32_t> llama_kv_compact_select_topk(
        const std::vector<float> & scores,
        uint32_t t);

// Progressive OMP schedule entry.
// Reference: omp.py DEFAULT_PROGRESSIVE_SCHEDULE (lines 120-124)
//
// As OMP selects more keys, it becomes more aggressive:
//   Phase 1 (0-299):    k_choice=1, nnls_interval=1 (conservative)
//   Phase 2 (300-1499): k_choice=2, nnls_interval=2 (moderate)
//   Phase 3 (1500+):    k_choice=4, nnls_interval=2 (aggressive)
struct llama_kv_compact_omp_schedule_entry {
    uint32_t threshold;      // use this config until selected count reaches threshold
    uint32_t k_choice;       // keys to select per iteration
    uint32_t nnls_interval;  // NNLS solve frequency (1=every iter, 2=every other)
};

// MIT default progressive schedule (defined in llama-kv-compact-select.cpp).
extern const llama_kv_compact_omp_schedule_entry LLAMA_KV_COMPACT_DEFAULT_OMP_SCHEDULE[3];

// OMP key selection options.
// Reference: omp.py class OMPCompaction.__init__() lines 138-206
struct llama_kv_compact_omp_opts {
    // Schedule (V2 — GAP-04)
    const llama_kv_compact_omp_schedule_entry * schedule     = LLAMA_KV_COMPACT_DEFAULT_OMP_SCHEDULE;
    uint32_t                                    schedule_len = 3;

    float    lower_bound   = 1e-12f;

    // Drop-key refinement (V2 — GAP-05)
    // After initial selection, drop keys with log(beta) < cutoff and re-select.
    // Set to -INFINITY to disable. Max 3 refinement passes.
    float    drop_key_beta_cutoff = -7.0f;

    // Quality parameters (V2 — GAP-16)
    bool     use_abs_corr         = false;   // use |correlation| for key selection
    bool     normalize_exp_scores = false;   // L2-normalize exp_score columns before correlation

    // Cached selection order (V2 — GAP-15)
    // If non-null, reuse a previously computed selection order instead of running OMP.
    // Only the first t indices are used; beta is recomputed via NNLS.
    std::vector<uint32_t> * cached_selection_order = nullptr;

    // Per-head timeout in milliseconds (V4-H — GAP-M).
    // If OMP exceeds this limit, keep partial greedy selection and fill
    // remaining positions with top-k attention-scored keys. Beta refit
    // runs on the full combined selection. Set to 0 to disable.
    float timeout_ms = 5000.0f;
};

// OMP key selection with progressive schedule + drop-key refinement.
//
// Greedy selection of t keys that best approximate the attention partition
// function. At each step selects the key(s) most correlated with the residual
// between the target partition sum and the current approximation.
//
// V2 enhancements over V1:
//   - Progressive schedule: k_choice/nnls_interval adapt as more keys are selected
//   - Drop-key refinement: post-selection pruning of low-weight keys (max 3 passes)
//   - cached_selection_order: reuse prior OMP order for multi-ratio evaluation
//   - use_abs_corr / normalize_exp_scores: MIT quality parameters
//
// Reference: Algorithm 1, arXiv:2602.16284 Section 3.2
// Reference impl: compaction/algorithms/omp.py lines 478-718
//
// Returns sorted position indices.
// beta_out receives the NNLS-derived log-weights.
std::vector<uint32_t> llama_kv_compact_select_omp(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & keys,
        uint32_t t,
        const llama_kv_compact_omp_opts & opts,
        std::vector<float> & beta_out);
