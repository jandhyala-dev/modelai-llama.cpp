#include "llama-kv-compact-budget.h"
#include "llama-kv-compact-math.h"

#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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
            // Find the head with the lowest weight that is above min_per_head.
            uint32_t best = UINT32_MAX;
            for (uint32_t h = 0; h < n_heads; ++h) {
                if (budgets[h] > opts.min_per_head) {
                    if (best == UINT32_MAX || weights[h] < weights[best]) {
                        best = h;
                    }
                }
            }
            if (best == UINT32_MAX) {
                break; // all heads at min_per_head, cannot reduce further
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

// ---------------------------------------------------------------------------
// Budget JSON loading (V2 — GAP-13)
// ---------------------------------------------------------------------------
//
// Parses simple JSON: {"L0H0": 0.0025, "L0H1": 0.0015, ...}
// No external JSON library — hand-rolled parser for this fixed format.

bool llama_kv_compact_load_budget_json(
        const char * json_path,
        uint32_t n_layers,
        uint32_t n_heads,
        std::vector<float> & proportions_out) {

    FILE * f = fopen(json_path, "r");
    if (!f) {
        LLAMA_LOG_ERROR("budget: cannot open %s\n", json_path);
        return false;
    }

    // Read entire file into string.
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0 || file_size > 10 * 1024 * 1024) {
        fclose(f);
        LLAMA_LOG_ERROR("budget: file size invalid (%ld bytes)\n", file_size);
        return false;
    }

    std::vector<char> buf(file_size + 1);
    size_t read_bytes = fread(buf.data(), 1, file_size, f);
    fclose(f);
    buf[read_bytes] = '\0';

    const uint32_t total = n_layers * n_heads;
    proportions_out.assign(total, 0.0f);

    // Parse key-value pairs: "LxHy": float
    const char * p = buf.data();
    uint32_t parsed = 0;

    while (*p) {
        // Find next quoted key.
        const char * quote1 = strchr(p, '"');
        if (!quote1) break;
        const char * quote2 = strchr(quote1 + 1, '"');
        if (!quote2) break;

        // Extract key: "LxHy"
        size_t key_len = quote2 - quote1 - 1;
        if (key_len < 4 || quote1[1] != 'L') {
            p = quote2 + 1;
            continue;
        }

        // Parse layer and head from "LxHy".
        uint32_t layer = 0;
        uint32_t head = 0;
        const char * kp = quote1 + 2; // after "L"
        while (kp < quote2 && *kp >= '0' && *kp <= '9') {
            layer = layer * 10 + (*kp - '0');
            kp++;
        }
        if (kp >= quote2 || *kp != 'H') {
            p = quote2 + 1;
            continue;
        }
        kp++; // skip 'H'
        while (kp < quote2 && *kp >= '0' && *kp <= '9') {
            head = head * 10 + (*kp - '0');
            kp++;
        }

        // Find colon + value.
        const char * colon = strchr(quote2 + 1, ':');
        if (!colon) break;

        char * endptr = nullptr;
        float val = strtof(colon + 1, &endptr);
        if (endptr == colon + 1) {
            p = colon + 1;
            continue;
        }

        if (layer < n_layers && head < n_heads) {
            proportions_out[layer * n_heads + head] = val;
            parsed++;
        }

        p = endptr;
    }

    if (parsed == 0) {
        LLAMA_LOG_ERROR("budget: no valid entries parsed from %s\n", json_path);
        return false;
    }

    LLAMA_LOG_INFO("budget: loaded %u/%u head proportions from %s\n", parsed, total, json_path);
    return true;
}

std::vector<uint32_t> llama_kv_compact_allocate_from_proportions(
        const std::vector<float> & proportions,
        uint32_t total_budget,
        uint32_t min_per_head,
        uint32_t max_per_head) {

    const uint32_t n = (uint32_t) proportions.size();
    if (n == 0 || total_budget == 0) {
        return {};
    }

    const uint32_t max_b = (max_per_head > 0) ? max_per_head : total_budget;

    // Normalize proportions.
    float prop_sum = 0.0f;
    for (float p : proportions) {
        prop_sum += p;
    }
    if (prop_sum < 1e-12f) {
        prop_sum = 1.0f;
    }

    std::vector<uint32_t> budgets(n);
    uint32_t allocated = 0;

    for (uint32_t i = 0; i < n; ++i) {
        float frac = proportions[i] / prop_sum;
        uint32_t b = (uint32_t) std::round(frac * total_budget);
        b = std::max(b, min_per_head);
        b = std::min(b, max_b);
        budgets[i] = b;
        allocated += b;
    }

    // Fine-tune to hit exact total_budget.
    while (allocated < total_budget) {
        // Add to head with largest proportion that's not at max.
        uint32_t best = UINT32_MAX;
        float best_prop = -1.0f;
        for (uint32_t i = 0; i < n; ++i) {
            if (budgets[i] < max_b && proportions[i] > best_prop) {
                best = i;
                best_prop = proportions[i];
            }
        }
        if (best == UINT32_MAX) break;
        budgets[best]++;
        allocated++;
    }
    while (allocated > total_budget) {
        // Remove from head with smallest proportion that's above min.
        uint32_t best = UINT32_MAX;
        float best_prop = std::numeric_limits<float>::max();
        for (uint32_t i = 0; i < n; ++i) {
            if (budgets[i] > min_per_head && proportions[i] < best_prop) {
                best = i;
                best_prop = proportions[i];
            }
        }
        if (best == UINT32_MAX) break;
        budgets[best]--;
        allocated--;
    }

    return budgets;
}
