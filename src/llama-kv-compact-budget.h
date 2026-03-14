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
