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
        std::vector<float> & scores_inout,
        llama_kv_compact_score_agg agg,
        uint32_t * n_queries_out) {
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
        if (agg == LLAMA_KV_COMPACT_SCORE_AGG_RMS) {
            for (uint32_t ki = 0; ki < keys.rows; ++ki) {
                const float w = weights[ki] * inv_sum;
                scores_inout[ki] += w * w;
            }
        } else {
            for (uint32_t ki = 0; ki < keys.rows; ++ki) {
                scores_inout[ki] += weights[ki] * inv_sum;
            }
        }
    }

    if (n_queries_out) {
        *n_queries_out += queries.rows;
    }
}

void llama_kv_compact_finalize_rms_scores(
        std::vector<float> & scores,
        uint32_t n_queries) {
    if (n_queries == 0) {
        return;
    }
    const float inv_n = 1.0f / float(n_queries);
    for (float & s : scores) {
        s = std::sqrt(s * inv_n);
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

namespace {

bool omp_solve_nnls(
        const llama_kv_compact_matrix & M,
        const std::vector<float> & target,
        float lower_bound,
        std::vector<float> & B_out) {
    const uint32_t n = M.rows;
    const uint32_t t = M.cols;

    // Build normal equations: M^T M x = M^T target
    std::vector<float> mtm(size_t(t) * t, 0.0f);
    std::vector<float> mty(t, 0.0f);

    for (uint32_t r = 0; r < n; ++r) {
        const float * row = M.row(r);
        for (uint32_t i = 0; i < t; ++i) {
            const float ri = row[i];
            for (uint32_t j = 0; j <= i; ++j) {
                mtm[size_t(i) * t + j] += ri * row[j];
            }
            mty[i] += ri * target[r];
        }
    }

    // Symmetrize + regularize
    const float lambda = 1e-6f;
    for (uint32_t i = 0; i < t; ++i) {
        for (uint32_t j = 0; j < i; ++j) {
            mtm[size_t(j) * t + i] = mtm[size_t(i) * t + j];
        }
        mtm[size_t(i) * t + i] += lambda;
    }

    // Cholesky decomposition
    std::vector<float> L = mtm;
    for (uint32_t i = 0; i < t; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            float sum = L[size_t(i) * t + j];
            for (uint32_t k = 0; k < j; ++k) {
                sum -= L[size_t(i) * t + k] * L[size_t(j) * t + k];
            }
            if (i == j) {
                if (sum <= 0.0f) return false;
                L[size_t(i) * t + j] = std::sqrt(sum);
            } else {
                L[size_t(i) * t + j] = sum / L[size_t(j) * t + j];
            }
        }
        for (uint32_t j = i + 1; j < t; ++j) {
            L[size_t(i) * t + j] = 0.0f;
        }
    }

    // Forward substitution
    B_out = mty;
    for (uint32_t i = 0; i < t; ++i) {
        float sum = B_out[i];
        for (uint32_t k = 0; k < i; ++k) {
            sum -= L[size_t(i) * t + k] * B_out[k];
        }
        B_out[i] = sum / L[size_t(i) * t + i];
    }
    // Back substitution
    for (int i = int(t) - 1; i >= 0; --i) {
        float sum = B_out[i];
        for (uint32_t k = uint32_t(i + 1); k < t; ++k) {
            sum -= L[size_t(k) * t + uint32_t(i)] * B_out[k];
        }
        B_out[i] = sum / L[size_t(i) * t + uint32_t(i)];
    }

    // Clamp to non-negative
    for (float & w : B_out) {
        w = std::max(w, lower_bound);
    }
    return true;
}

} // namespace

std::vector<uint32_t> llama_kv_compact_select_omp(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & keys,
        uint32_t t,
        const llama_kv_compact_omp_opts & opts,
        std::vector<float> & beta_out) {
    const uint32_t n = queries.rows;
    const uint32_t T = keys.rows;
    const uint32_t d = keys.cols;
    const float inv_sqrt_d = 1.0f / std::sqrt(float(d));

    t = std::min(t, T);

    // Step 1: Compute exp_scores[n x T] and target[n]
    llama_kv_compact_matrix exp_scores(n, T);
    std::vector<float> target(n, 0.0f);

    for (uint32_t qi = 0; qi < n; ++qi) {
        const float * q = queries.row(qi);
        float row_max = -std::numeric_limits<float>::infinity();
        for (uint32_t ki = 0; ki < T; ++ki) {
            float score = dot_row(q, keys.row(ki), d) * inv_sqrt_d;
            exp_scores(qi, ki) = score;
            row_max = std::max(row_max, score);
        }
        float sum = 0.0f;
        for (uint32_t ki = 0; ki < T; ++ki) {
            float e = std::exp(exp_scores(qi, ki) - row_max);
            exp_scores(qi, ki) = e;
            sum += e;
        }
        target[qi] = sum;
    }

    // Step 2: Greedy OMP loop
    std::vector<uint32_t> selected;
    selected.reserve(t);
    std::vector<bool> mask(T, false);
    std::vector<float> current(n, 0.0f);
    std::vector<float> B;
    std::vector<float> corr(T);

    uint32_t iteration = 0;
    while (selected.size() < t) {
        // Compute correlation of each key with residual
        for (uint32_t ki = 0; ki < T; ++ki) {
            if (mask[ki]) {
                corr[ki] = -std::numeric_limits<float>::infinity();
                continue;
            }
            float c = 0.0f;
            for (uint32_t qi = 0; qi < n; ++qi) {
                c += exp_scores(qi, ki) * (target[qi] - current[qi]);
            }
            corr[ki] = c;
        }

        // Select top k_choice keys
        uint32_t k_select = std::min(opts.k_choice,
                                      uint32_t(t - selected.size()));

        std::vector<uint32_t> candidates(T);
        std::iota(candidates.begin(), candidates.end(), 0);
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + std::min(k_select + uint32_t(selected.size()), T),
            candidates.end(),
            [&](uint32_t a, uint32_t b) { return corr[a] > corr[b]; });

        uint32_t added = 0;
        for (uint32_t ci = 0; ci < T && added < k_select; ++ci) {
            uint32_t idx = candidates[ci];
            if (mask[idx]) continue;
            selected.push_back(idx);
            mask[idx] = true;
            added++;
        }

        // Solve NNLS conditionally based on interval
        bool should_solve = (B.empty())
                         || (iteration % opts.nnls_interval == 0)
                         || (selected.size() >= t);

        if (should_solve) {
            uint32_t i = selected.size();
            llama_kv_compact_matrix M(n, i);
            for (uint32_t qi = 0; qi < n; ++qi) {
                for (uint32_t si = 0; si < i; ++si) {
                    M(qi, si) = exp_scores(qi, selected[si]);
                }
            }
            if (!omp_solve_nnls(M, target, opts.lower_bound, B)) {
                B.resize(selected.size(), opts.lower_bound);
            }

            // OMP key pruning (Appendix C.2): remove keys with log(beta) < threshold.
            // After NNLS, keys with near-zero weight contribute nothing to the
            // approximation. Pruning them frees capacity for better candidates.
            // Guard: always retain at least 1 key to prevent degenerate empty selection.
            if (opts.beta_prune_log_threshold > -std::numeric_limits<float>::infinity()
                    && selected.size() > 1) {
                size_t write = 0;
                for (size_t si = 0; si < selected.size(); ++si) {
                    const float log_b = std::log(std::max(B[si], 1e-30f));
                    if (log_b >= opts.beta_prune_log_threshold) {
                        if (write != si) {
                            selected[write] = selected[si];
                            B[write] = B[si];
                        }
                        write++;
                    } else {
                        mask[selected[si]] = false; // allow re-selection
                    }
                }
                // Retain at least 1 key even if all fail the threshold.
                write = std::max(write, size_t(1));
                if (write < selected.size()) {
                    selected.resize(write);
                    B.resize(write);
                }
            }
        } else {
            B.resize(selected.size(), opts.lower_bound);
        }

        // Update approximation: current = M @ B
        std::fill(current.begin(), current.end(), 0.0f);
        for (uint32_t qi = 0; qi < n; ++qi) {
            for (uint32_t si = 0; si < selected.size(); ++si) {
                current[qi] += exp_scores(qi, selected[si]) * B[si];
            }
        }

        iteration++;
    }

    // Final NNLS if last iteration was skipped
    if (opts.nnls_interval > 1 && !selected.empty()) {
        uint32_t i = selected.size();
        llama_kv_compact_matrix M(n, i);
        for (uint32_t qi = 0; qi < n; ++qi) {
            for (uint32_t si = 0; si < i; ++si) {
                M(qi, si) = exp_scores(qi, selected[si]);
            }
        }
        omp_solve_nnls(M, target, opts.lower_bound, B);
    }

    // Convert to beta (log-weights) and sort by position
    std::vector<uint32_t> order(selected.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) {
                  return selected[a] < selected[b];
              });

    std::vector<uint32_t> result(selected.size());
    beta_out.resize(selected.size());
    for (size_t i = 0; i < order.size(); ++i) {
        result[i] = selected[order[i]];
        beta_out[i] = std::log(std::max(B[order[i]], opts.lower_bound));
    }

    return result;
}
