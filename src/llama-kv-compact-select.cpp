#include "llama-kv-compact-select.h"
#include "llama-kv-compact-math.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

using llama_kv_compact_math::dot_row;

void llama_kv_compact_accumulate_attention_scores(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & keys,
        std::vector<float> & scores_inout) {
    if (queries.cols == 0 || keys.cols != queries.cols || scores_inout.size() != keys.rows) {
        return;
    }

    const float inv_sqrt_d = 1.0f / std::sqrt(float(keys.cols));
    std::vector<float> weights(keys.rows, 0.0f);

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
        for (uint32_t ki = 0; ki < keys.rows; ++ki) {
            scores_inout[ki] += weights[ki] * inv_sum;
        }
    }
}

std::vector<uint32_t> llama_kv_compact_select_topk(
        const std::vector<float> & scores,
        uint32_t t) {
    std::vector<uint32_t> idx(scores.size());
    std::iota(idx.begin(), idx.end(), 0);

    if (t >= idx.size()) {
        return idx;
    }

    std::partial_sort(idx.begin(), idx.begin() + t, idx.end(), [&](uint32_t a, uint32_t b) {
        if (scores[a] == scores[b]) {
            return a < b;
        }
        return scores[a] > scores[b];
    });
    idx.resize(t);
    std::sort(idx.begin(), idx.end());
    return idx;
}
