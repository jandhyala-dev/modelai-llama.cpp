#include "llama-kv-compact-select.h"
#include "llama-kv-compact-math.h"
#include "llama-impl.h"  // F-M-28: LLAMA_LOG_WARN

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

static inline auto dot_row(const float * a, const float * b, uint32_t n) {
    return llama_kv_compact_dot_row(a, b, n);
}

// MIT default progressive schedule (declared extern in header).
const llama_kv_compact_omp_schedule_entry LLAMA_KV_COMPACT_DEFAULT_OMP_SCHEDULE[] = {
    {  300, 1, 1 },
    { 1500, 2, 2 },
    { UINT32_MAX, 4, 2 },
};

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
        if (agg == LLAMA_KV_COMPACT_SCORE_AGG_MAX) {
            for (uint32_t ki = 0; ki < keys.rows; ++ki) {
                const float w = weights[ki] * inv_sum;
                scores_inout[ki] = std::max(scores_inout[ki], w);
            }
        } else if (agg == LLAMA_KV_COMPACT_SCORE_AGG_RMS) {
            for (uint32_t ki = 0; ki < keys.rows; ++ki) {
                const float w = weights[ki] * inv_sum;
                scores_inout[ki] += w * w;
            }
        } else {
            // SUM and MEAN use identical accumulation; MEAN divides in finalize step.
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

void llama_kv_compact_finalize_mean_scores(
        std::vector<float> & scores,
        uint32_t n_queries) {
    if (n_queries == 0) {
        return;
    }
    const float inv_n = 1.0f / float(n_queries);
    for (float & s : scores) {
        s *= inv_n;
    }
}

void llama_kv_compact_avgpool_scores(
        std::vector<float> & scores,
        uint32_t kernel_size) {
    if (kernel_size <= 1 || scores.size() <= 1) {
        return;
    }

    const uint32_t n = (uint32_t) scores.size();
    const uint32_t half = kernel_size / 2;
    std::vector<float> smoothed(n);

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t start = (i >= half) ? i - half : 0;
        const uint32_t end = std::min(i + half + 1, n);
        float sum = 0.0f;
        for (uint32_t j = start; j < end; ++j) {
            sum += scores[j];
        }
        smoothed[i] = sum / float(end - start);
    }

    scores = std::move(smoothed);
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

// Default Tikhonov regularization for the OMP NNLS solver.
// Matches llama_kv_compact_solver_opts::lambda default.
static constexpr float OMP_NNLS_DEFAULT_LAMBDA = 1e-6f;

// Solve NNLS via Cholesky normal equations + clamp.
// Used inside OMP loop for fast beta refit.
//
// NOTE: The V2 solver in llama-kv-compact-solver.cpp (solve_nnls_v2) provides
// a more robust implementation with LAPACK sgels, adaptive ridge scaling, and
// PGD refinement.  That solver is not exposed in the public header (it lives
// inside an anonymous namespace), so we keep this lightweight Cholesky variant
// for the OMP inner loop where latency matters.  The `lambda` parameter here
// corresponds to opts.lambda in llama_kv_compact_solver_opts.
bool omp_solve_nnls(
        const llama_kv_compact_matrix & M,
        const std::vector<float> & target,
        float lower_bound,
        float lambda,
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
    // F-C-15: Upper triangle is uninitialized (MtM only fills lower triangle).
    // Copy lower→upper instead of averaging with uninitialized values.
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

    // Check solution validity before clamping.
    for (uint32_t i = 0; i < t; ++i) {
        if (!std::isfinite(B_out[i])) {
            return false;
        }
    }

    // Clamp to non-negative
    for (float & w : B_out) {
        w = std::max(w, lower_bound);
    }
    return true;
}

// Look up k_choice and nnls_interval from progressive schedule.
// Reference: omp.py _get_schedule_params() lines 211-235
void get_schedule_params(
        const llama_kv_compact_omp_schedule_entry * schedule,
        uint32_t schedule_len,
        uint32_t num_selected,
        uint32_t & k_choice_out,
        uint32_t & nnls_interval_out) {
    for (uint32_t si = 0; si < schedule_len; ++si) {
        if (num_selected < schedule[si].threshold) {
            k_choice_out = schedule[si].k_choice;
            nnls_interval_out = schedule[si].nnls_interval;
            return;
        }
    }
    // Fallback to last entry
    k_choice_out = schedule[schedule_len - 1].k_choice;
    nnls_interval_out = schedule[schedule_len - 1].nnls_interval;
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

    // Optional: L2-normalize exp_score columns (GAP-16).
    // Precompute column norms for correlation computation.
    std::vector<float> col_norms;
    if (opts.normalize_exp_scores) {
        col_norms.resize(T, 0.0f);
        for (uint32_t ki = 0; ki < T; ++ki) {
            float norm_sq = 0.0f;
            for (uint32_t qi = 0; qi < n; ++qi) {
                norm_sq += exp_scores(qi, ki) * exp_scores(qi, ki);
            }
            col_norms[ki] = std::sqrt(norm_sq) + 1e-12f;
        }
    }

    // Cached selection order shortcut (GAP-15).
    if (opts.cached_selection_order && !opts.cached_selection_order->empty()) {
        const auto & cached = *opts.cached_selection_order;
        const uint32_t use_t = std::min(t, (uint32_t) cached.size());

        std::vector<uint32_t> selected(cached.begin(), cached.begin() + use_t);

        // Recompute beta via NNLS for the selected subset.
        llama_kv_compact_matrix M(n, use_t);
        for (uint32_t qi = 0; qi < n; ++qi) {
            for (uint32_t si = 0; si < use_t; ++si) {
                M(qi, si) = exp_scores(qi, selected[si]);
            }
        }
        std::vector<float> B;
        if (!omp_solve_nnls(M, target, opts.lower_bound, OMP_NNLS_DEFAULT_LAMBDA, B)) {
            B.assign(use_t, opts.lower_bound);
        }

        // Sort by position and convert to log-weights.
        std::vector<uint32_t> order(use_t);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](uint32_t a, uint32_t b) { return selected[a] < selected[b]; });

        std::vector<uint32_t> result(use_t);
        beta_out.resize(use_t);
        for (size_t i = 0; i < order.size(); ++i) {
            result[i] = selected[order[i]];
            beta_out[i] = std::log(std::max(B[order[i]], opts.lower_bound));
        }
        return result;
    }

    // Step 2: OMP loop with progressive schedule + drop-key refinement.
    std::vector<uint32_t> selected;
    selected.reserve(t);
    std::vector<bool> mask_selected(T, false);
    std::vector<bool> mask_excluded(T, false);  // permanently excluded (drop-key)
    std::vector<float> current(n, 0.0f);
    std::vector<float> B;
    std::vector<float> corr(T);
    std::vector<uint32_t> candidates(T);  // m-15: hoisted out of while loop

    uint32_t iteration = 0;
    uint32_t refinement_count = 0;
    const auto omp_start_time = std::chrono::steady_clock::now();

    while (true) {
        const uint32_t i = (uint32_t) selected.size();

        if (i < t) {
            // Per-head timeout (V4-H — GAP-M).
            if (opts.timeout_ms > 0.0f) {
                const auto elapsed = std::chrono::steady_clock::now() - omp_start_time;
                if (std::chrono::duration<float, std::milli>(elapsed).count() > opts.timeout_ms) {
                    break;
                }
            }

            // --- Normal selection phase ---
            // Compute correlation of each key with residual.
            for (uint32_t ki = 0; ki < T; ++ki) {
                if (mask_selected[ki] || mask_excluded[ki]) {
                    corr[ki] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float c = 0.0f;
                for (uint32_t qi = 0; qi < n; ++qi) {
                    float es = exp_scores(qi, ki);
                    if (opts.normalize_exp_scores) {
                        es /= col_norms[ki];
                    }
                    c += es * (target[qi] - current[qi]);
                }
                corr[ki] = opts.use_abs_corr ? std::fabs(c) : c;
            }

            // Get k_choice and nnls_interval from progressive schedule.
            uint32_t k_choice = 1;
            uint32_t nnls_interval = 1;
            get_schedule_params(opts.schedule, opts.schedule_len, i,
                                k_choice, nnls_interval);

            uint32_t k_select = std::min(k_choice, t - i);

            // Select top k_select keys by correlation.
            // F-M-28: Guard against excessive candidate count — fallback to top-k
            if (T > 100000) {
                LLAMA_LOG_WARN("%s: OMP candidate count %u exceeds 100k, falling back to top-k\n", __func__, T);
                break;
            }
            // Use partial sort to find the top candidates efficiently.
            std::iota(candidates.begin(), candidates.end(), 0);
            // partial_sort only needs to find the top k_select candidates.
            // Masked keys have corr=-inf and sort to the end, so k_select
            // unmasked candidates will be in the first k_select positions.
            std::partial_sort(
                candidates.begin(),
                candidates.begin() + std::min(k_select, T),
                candidates.end(),
                [&](uint32_t a, uint32_t b) { return corr[a] > corr[b]; });

            uint32_t added = 0;
            for (uint32_t ci = 0; ci < T && added < k_select; ++ci) {
                uint32_t idx = candidates[ci];
                if (mask_selected[idx] || mask_excluded[idx]) continue;
                selected.push_back(idx);
                mask_selected[idx] = true;
                added++;
            }

            if (added == 0) break;  // no more candidates

            // Solve NNLS conditionally based on interval.
            bool should_solve = B.empty()
                             || (iteration % nnls_interval == 0)
                             || (selected.size() >= t);

            if (should_solve) {
                const uint32_t sel_count = (uint32_t) selected.size();
                llama_kv_compact_matrix M(n, sel_count);
                for (uint32_t qi = 0; qi < n; ++qi) {
                    for (uint32_t si = 0; si < sel_count; ++si) {
                        M(qi, si) = exp_scores(qi, selected[si]);
                    }
                }
                if (!omp_solve_nnls(M, target, opts.lower_bound, OMP_NNLS_DEFAULT_LAMBDA, B)) {
                    B.assign(sel_count, opts.lower_bound);
                }
            } else {
                B.resize(selected.size(), opts.lower_bound);
            }

            // Update approximation: current = M @ B
            std::fill(current.begin(), current.end(), 0.0f);
            for (uint32_t qi = 0; qi < n; ++qi) {
                for (size_t si = 0; si < selected.size(); ++si) {
                    current[qi] += exp_scores(qi, selected[si]) * B[si];
                }
            }

            iteration++;

        } else if (i == t && opts.drop_key_beta_cutoff > -std::numeric_limits<float>::infinity()) {
            // --- Drop-key refinement phase (V2 — GAP-05) ---
            // Reference: omp.py lines 629-702
            refinement_count++;
            if (refinement_count > 3) break;  // max 3 refinement passes

            // Always solve NNLS in refinement (need accurate beta for drop decision).
            const uint32_t sel_count = (uint32_t) selected.size();
            llama_kv_compact_matrix M(n, sel_count);
            for (uint32_t qi = 0; qi < n; ++qi) {
                for (uint32_t si = 0; si < sel_count; ++si) {
                    M(qi, si) = exp_scores(qi, selected[si]);
                }
            }
            if (!omp_solve_nnls(M, target, opts.lower_bound, OMP_NNLS_DEFAULT_LAMBDA, B)) {
                break;  // NNLS failed — accept current selection
            }

            // Find keys to drop: log(beta) < cutoff.
            std::vector<bool> drop_mask(sel_count, false);
            uint32_t n_drop = 0;
            for (uint32_t si = 0; si < sel_count; ++si) {
                float log_b = std::log(std::max(B[si], 1e-30f));
                if (log_b < opts.drop_key_beta_cutoff) {
                    drop_mask[si] = true;
                    n_drop++;
                }
            }

            if (n_drop == 0) break;  // stable set — converged

            // Drop keys: mark as permanently excluded, compact selected/B arrays.
            size_t write = 0;
            for (size_t si = 0; si < selected.size(); ++si) {
                if (drop_mask[si]) {
                    mask_selected[selected[si]] = false;
                    mask_excluded[selected[si]] = true;
                } else {
                    if (write != si) {
                        selected[write] = selected[si];
                        B[write] = B[si];
                    }
                    write++;
                }
            }
            selected.resize(write);
            B.resize(write);

            // Update approximation for reduced set.
            std::fill(current.begin(), current.end(), 0.0f);
            for (uint32_t qi = 0; qi < n; ++qi) {
                for (size_t si = 0; si < selected.size(); ++si) {
                    current[qi] += exp_scores(qi, selected[si]) * B[si];
                }
            }

            iteration++;
            // Continue — loop will re-enter i < t to fill dropped slots.

        } else {
            break;  // i == t and no refinement — done
        }
    }

    // Top-k fallback for partial selection (V4-H — GAP-M).
    // If OMP was interrupted (timeout or exhausted candidates), fill remaining
    // positions with top-k attention-scored keys. Greedy OMP property ensures
    // partial selections are locally optimal; top-k fills the rest.
    if (selected.size() < t) {
        std::vector<float> fallback_scores(T, -std::numeric_limits<float>::infinity());
        for (uint32_t ki = 0; ki < T; ++ki) {
            if (mask_selected[ki] || mask_excluded[ki]) {
                continue;
            }
            float score = 0.0f;
            for (uint32_t qi = 0; qi < n; ++qi) {
                score += exp_scores(qi, ki) / std::max(target[qi], 1e-6f);
            }
            fallback_scores[ki] = score;
        }
        const uint32_t need = t - (uint32_t)selected.size();
        auto fallback_idx = llama_kv_compact_select_topk(fallback_scores, need);
        for (uint32_t idx : fallback_idx) {
            if (fallback_scores[idx] > -std::numeric_limits<float>::infinity()) {
                selected.push_back(idx);
                mask_selected[idx] = true;
            }
        }
    }

    // Final NNLS on full merged selection (OMP partial + top-k fallback).
    if (!selected.empty()) {
        const uint32_t sel_count = (uint32_t) selected.size();
        llama_kv_compact_matrix M(n, sel_count);
        for (uint32_t qi = 0; qi < n; ++qi) {
            for (uint32_t si = 0; si < sel_count; ++si) {
                M(qi, si) = exp_scores(qi, selected[si]);
            }
        }
        omp_solve_nnls(M, target, opts.lower_bound, OMP_NNLS_DEFAULT_LAMBDA, B);
    }

    // Convert to beta (log-weights) and sort by position.
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
