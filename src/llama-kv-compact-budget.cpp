#include "llama-kv-compact-budget.h"
#include "llama-kv-compact-math.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>

using llama_kv_compact_math::dot_row;

float llama_kv_compact_head_entropy(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & keys) {
    if (queries.cols == 0 || keys.cols != queries.cols || queries.rows == 0 || keys.rows == 0) {
        return 0.0f;
    }

    const float inv_sqrt_d = 1.0f / std::sqrt(float(keys.cols));
    double entropy_sum = 0.0;

    std::vector<float> weights(keys.rows);

    for (uint32_t qi = 0; qi < queries.rows; ++qi) {
        const float * q = queries.row(qi);
        float row_max = -std::numeric_limits<float>::infinity();
        for (uint32_t ki = 0; ki < keys.rows; ++ki) {
            const float score = dot_row(q, keys.row(ki), keys.cols) * inv_sqrt_d;
            weights[ki] = score;
            row_max = std::max(row_max, score);
        }
        float sum = 0.0f;
        for (uint32_t ki = 0; ki < keys.rows; ++ki) {
            weights[ki] = std::exp(weights[ki] - row_max);
            sum += weights[ki];
        }
        const float inv_sum = 1.0f / std::max(sum, 1e-6f);

        // H = -sum_k p_k * log(p_k)
        double h = 0.0;
        for (uint32_t ki = 0; ki < keys.rows; ++ki) {
            const float p = weights[ki] * inv_sum;
            if (p > 1e-10f) {
                h -= double(p) * std::log(double(p));
            }
        }
        entropy_sum += h;
    }

    return float(entropy_sum / queries.rows);
}

std::vector<uint32_t> llama_kv_compact_allocate_budgets(
        const std::vector<float> & head_entropies,
        const llama_kv_compact_budget_opts & opts) {
    const uint32_t n_heads = (uint32_t) head_entropies.size();
    if (n_heads == 0 || opts.total_budget == 0) {
        return {};
    }

    // Compute inverse-entropy sensitivity weights.
    // Lower entropy = more sensitive = higher weight = larger budget.
    std::vector<float> weights(n_heads);
    float weight_sum = 0.0f;
    for (uint32_t h = 0; h < n_heads; ++h) {
        weights[h] = 1.0f / std::max(head_entropies[h], opts.sensitivity_eps);
        weight_sum += weights[h];
    }

    // Proportional allocation with min/max clamping.
    const uint32_t max_budget = (opts.max_per_head > 0) ? opts.max_per_head : opts.total_budget;
    std::vector<uint32_t> budgets(n_heads);
    uint32_t allocated = 0;

    for (uint32_t h = 0; h < n_heads; ++h) {
        float frac = weights[h] / std::max(weight_sum, 1e-6f);
        uint32_t b = (uint32_t) std::round(frac * opts.total_budget);
        b = std::max(b, opts.min_per_head);
        b = std::min(b, max_budget);
        budgets[h] = b;
        allocated += b;
    }

    // Adjust to match total_budget: scale proportionally if over/under.
    if (allocated != opts.total_budget && allocated > 0) {
        float scale = float(opts.total_budget) / float(allocated);
        allocated = 0;
        for (uint32_t h = 0; h < n_heads; ++h) {
            budgets[h] = std::max(opts.min_per_head,
                         std::min(max_budget,
                                  (uint32_t) std::round(budgets[h] * scale)));
            allocated += budgets[h];
        }
        // Fine-tune: add/remove from the head with the largest/smallest budget
        // to hit the exact target.
        while (allocated < opts.total_budget) {
            uint32_t best = UINT32_MAX;
            for (uint32_t h = 0; h < n_heads; ++h) {
                if (budgets[h] < max_budget) {
                    if (best == UINT32_MAX || weights[h] > weights[best]) {
                        best = h;
                    }
                }
            }
            if (best == UINT32_MAX) {
                break; // all heads at max_budget, cannot allocate further
            }
            budgets[best]++;
            allocated++;
        }
        while (allocated > opts.total_budget) {
            uint32_t best = 0;
            for (uint32_t h = 1; h < n_heads; ++h) {
                if (weights[h] < weights[best] && budgets[h] > opts.min_per_head) {
                    best = h;
                }
            }
            if (budgets[best] <= opts.min_per_head) {
                break; // can't reduce further
            }
            budgets[best]--;
            allocated--;
        }
    }

    return budgets;
}

std::vector<uint32_t> llama_kv_compact_build_union(
        const std::vector<std::vector<uint32_t>> & per_head_selections,
        uint32_t n_heads,
        std::vector<bool> & per_head_mask) {
    // Build sorted union of all selected positions.
    std::set<uint32_t> union_set;
    for (const auto & sel : per_head_selections) {
        for (uint32_t idx : sel) {
            union_set.insert(idx);
        }
    }

    std::vector<uint32_t> union_vec(union_set.begin(), union_set.end());
    const uint32_t union_size = (uint32_t) union_vec.size();

    // Build position-to-union-index map for fast lookup.
    std::vector<uint32_t> pos_to_union(union_vec.empty() ? 0 : union_vec.back() + 1, UINT32_MAX);
    for (uint32_t j = 0; j < union_size; ++j) {
        pos_to_union[union_vec[j]] = j;
    }

    // Build per-head mask: per_head_mask[h * union_size + j] = true if head h selected union position j.
    per_head_mask.assign(size_t(n_heads) * union_size, false);
    for (uint32_t h = 0; h < n_heads && h < per_head_selections.size(); ++h) {
        for (uint32_t idx : per_head_selections[h]) {
            if (idx < pos_to_union.size()) {
                uint32_t j = pos_to_union[idx];
                if (j != UINT32_MAX) {
                    per_head_mask[size_t(h) * union_size + j] = true;
                }
            }
        }
    }

    return union_vec;
}
