#pragma once

// Nonuniform per-head budget allocation for KV compaction.
// Reference: arXiv:2602.16284 Section 3.4, Algorithm 4
//
// Different attention heads have different sensitivity to compaction.
// Heads with peaky (low-entropy) attention need more retained tokens,
// while heads with diffuse (high-entropy) attention tolerate aggressive
// compression.  This module computes per-head sensitivity via attention
// entropy and allocates budgets proportionally.

#include "llama-kv-compact-solver.h"

#include <cstdint>
#include <vector>

// Hard minimum budget floor per head — prevents solver starvation at extreme compression.
// Cannot be overridden below this value regardless of caller request. See GAP-K.
static constexpr uint32_t LLAMA_KV_COMPACT_BUDGET_FLOOR_PER_HEAD = 2;

struct llama_kv_compact_budget_opts {
    uint32_t total_budget  = 0;      // total tokens to select across all heads
    uint32_t min_per_head  = 4;      // minimum budget per head (prevents degenerate softmax)
    uint32_t max_per_head  = 0;      // 0 = unlimited (capped at n_prefix)
    float    sensitivity_eps = 1e-3f; // epsilon for entropy floor
};

// Compute attention entropy for one head given its query and key matrices.
// H = -sum_k softmax(QK/sqrt(d))_k * log(softmax(QK/sqrt(d))_k), averaged over queries.
// Lower entropy = peaky attention = more sensitive to compaction.
float llama_kv_compact_head_entropy(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & keys);

// Allocate per-head budgets proportional to inverse entropy (sensitivity).
// Returns a vector of budgets summing to approximately total_budget.
// Heads with lower entropy get larger budgets.
std::vector<uint32_t> llama_kv_compact_allocate_budgets(
        const std::vector<float> & head_entropies,
        const llama_kv_compact_budget_opts & opts);

// Build the union (superset) of per-head selections.
// Returns the sorted union of all per-head selected position indices.
// Also builds a per-head mask: for each head h and union position j,
// per_head_mask[h * union_size + j] is true if head h selected position j.
std::vector<uint32_t> llama_kv_compact_build_union(
        const std::vector<std::vector<uint32_t>> & per_head_selections,
        uint32_t n_heads,
        std::vector<bool> & per_head_mask);

// ---------------------------------------------------------------------------
// Influence-curve budget allocation (V4-I — GAP-B)
// ---------------------------------------------------------------------------

// Influence curve point for one head at one compression ratio.
// Reference: MIT head_budget_optimization/solver.py
struct llama_kv_compact_influence_point {
    float ratio;  // fraction of keys retained (0 < ratio <= 1)
    float error;  // 1 - mean_attention_coverage (0 = perfect, 1 = total loss)
};

// Compute influence curve for one head: attention coverage error at multiple
// compression ratios. At each ratio, selects top-k keys by attention score
// and measures the fraction of attention mass NOT captured.
//
// Reference: arXiv:2602.16284 Section 3.4
//
// ratios/n_ratios: evaluation points. If null/0, uses default
//   [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0].
//
// Returns points sorted by ratio ascending.
std::vector<llama_kv_compact_influence_point> llama_kv_compact_head_influence_curve(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & keys,
        const float * ratios = nullptr,
        uint32_t n_ratios = 0);

// Swap-based budget optimizer using influence curves.
// Starts with uniform allocation, iteratively transfers budget from
// less-sensitive to more-sensitive heads to minimize total error.
//
// Reference: MIT head_budget_optimization/solver.py (swap-based solver)
//
// curves: influence curves for each head (from head_influence_curve).
// total_budget: total tokens to allocate across all heads.
// n_prefix_tokens: total prefix tokens (for ratio computation).
// max_iterations: swap iteration limit (convergence typically < 100).
//
// Returns per-head budgets summing to approximately total_budget.
std::vector<uint32_t> llama_kv_compact_swap_budget_solver(
        const std::vector<std::vector<llama_kv_compact_influence_point>> & curves,
        uint32_t total_budget,
        uint32_t n_prefix_tokens,
        uint32_t min_per_head = 4,
        uint32_t max_per_head = 0,
        uint32_t max_iterations = 1000);

// ---------------------------------------------------------------------------
// Budget JSON loading (V2 — GAP-13)
// ---------------------------------------------------------------------------

// Load per-head budget proportions from a JSON file.
// JSON format: {"L0H0": 0.0025, "L0H1": 0.0015, ...}
// Proportions should sum to approximately 1.0.
//
// n_layers, n_heads: model architecture dimensions.
// proportions_out: output vector of size n_layers * n_heads, indexed as [layer * n_heads + head].
//
// Returns true on success.
bool llama_kv_compact_load_budget_json(
        const char * json_path,
        uint32_t n_layers,
        uint32_t n_heads,
        std::vector<float> & proportions_out);

// Allocate per-head budgets from pre-computed proportions.
// proportions: per-head proportions (sum to ~1.0), indexed as [layer * n_heads + head].
// total_budget: total tokens to allocate.
// min_per_head: minimum budget per head.
// max_per_head: maximum budget per head (0 = unlimited).
//
// Returns vector of budgets indexed same as proportions.
std::vector<uint32_t> llama_kv_compact_allocate_from_proportions(
        const std::vector<float> & proportions,
        uint32_t total_budget,
        uint32_t min_per_head = 4,
        uint32_t max_per_head = 0);
