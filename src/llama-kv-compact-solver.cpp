#include "llama-kv-compact-solver.h"
#include "llama-kv-compact-math.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

using llama_kv_compact_math::dot_row;

bool solve_spd_cholesky(
        std::vector<float> a,
        uint32_t n,
        std::vector<float> & b,
        uint32_t nrhs) {
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            float sum = a[size_t(i) * n + j];
            for (uint32_t k = 0; k < j; ++k) {
                sum -= a[size_t(i) * n + k] * a[size_t(j) * n + k];
            }
            if (i == j) {
                if (sum <= 0.0f || !std::isfinite(sum)) {
                    return false;
                }
                a[size_t(i) * n + j] = std::sqrt(sum);
            } else {
                a[size_t(i) * n + j] = sum / a[size_t(j) * n + j];
            }
        }
        for (uint32_t j = i + 1; j < n; ++j) {
            a[size_t(i) * n + j] = 0.0f;
        }
    }

    for (uint32_t rhs = 0; rhs < nrhs; ++rhs) {
        float * x = b.data() + size_t(rhs) * n;
        for (uint32_t i = 0; i < n; ++i) {
            float sum = x[i];
            for (uint32_t k = 0; k < i; ++k) {
                sum -= a[size_t(i) * n + k] * x[k];
            }
            x[i] = sum / a[size_t(i) * n + i];
        }
        for (int i = int(n) - 1; i >= 0; --i) {
            float sum = x[i];
            for (uint32_t k = uint32_t(i + 1); k < n; ++k) {
                sum -= a[size_t(k) * n + uint32_t(i)] * x[k];
            }
            x[i] = sum / a[size_t(i) * n + uint32_t(i)];
        }
    }

    return true;
}

bool solve_least_squares_normal_eq(
        const llama_kv_compact_matrix & x,
        const llama_kv_compact_matrix & y,
        float lambda,
        llama_kv_compact_matrix & out) {
    if (x.rows != y.rows || x.cols == 0 || y.cols == 0) {
        return false;
    }

    const uint32_t n = x.rows;
    const uint32_t t = x.cols;
    const uint32_t d = y.cols;

    std::vector<float> xtx(size_t(t) * t, 0.0f);
    std::vector<float> xty(size_t(d) * t, 0.0f);

    for (uint32_t r = 0; r < n; ++r) {
        const float * xr = x.row(r);
        const float * yr = y.row(r);
        for (uint32_t i = 0; i < t; ++i) {
            const float xi = xr[i];
            for (uint32_t j = 0; j <= i; ++j) {
                xtx[size_t(i) * t + j] += xi * xr[j];
            }
            for (uint32_t c = 0; c < d; ++c) {
                xty[size_t(c) * t + i] += xi * yr[c];
            }
        }
    }

    for (uint32_t i = 0; i < t; ++i) {
        for (uint32_t j = 0; j < i; ++j) {
            xtx[size_t(j) * t + i] = xtx[size_t(i) * t + j];
        }
        xtx[size_t(i) * t + i] += lambda;
    }

    if (!solve_spd_cholesky(xtx, t, xty, d)) {
        return false;
    }

    out.resize(t, d);
    for (uint32_t c = 0; c < d; ++c) {
        for (uint32_t i = 0; i < t; ++i) {
            out(i, c) = xty[size_t(c) * t + i];
        }
    }
    return true;
}

bool solve_vector_least_squares(
        const llama_kv_compact_matrix & x,
        const std::vector<float> & y,
        float lambda,
        std::vector<float> & out) {
    if (x.rows != y.size() || x.cols == 0) {
        return false;
    }

    const uint32_t n = x.rows;
    const uint32_t t = x.cols;
    std::vector<float> xtx(size_t(t) * t, 0.0f);
    std::vector<float> xty(t, 0.0f);

    for (uint32_t r = 0; r < n; ++r) {
        const float * xr = x.row(r);
        for (uint32_t i = 0; i < t; ++i) {
            const float xi = xr[i];
            for (uint32_t j = 0; j <= i; ++j) {
                xtx[size_t(i) * t + j] += xi * xr[j];
            }
            xty[i] += xi * y[r];
        }
    }

    for (uint32_t i = 0; i < t; ++i) {
        for (uint32_t j = 0; j < i; ++j) {
            xtx[size_t(j) * t + i] = xtx[size_t(i) * t + j];
        }
        xtx[size_t(i) * t + i] += lambda;
    }

    std::vector<float> rhs = xty;
    if (!solve_spd_cholesky(xtx, t, rhs, 1)) {
        return false;
    }
    out = std::move(rhs);
    return true;
}

float spectral_step_size(const llama_kv_compact_matrix & m, float lambda) {
    const uint32_t n = m.rows;
    const uint32_t t = m.cols;
    std::vector<float> v(t, 1.0f / std::sqrt(float(std::max<uint32_t>(t, 1))));
    std::vector<float> tmp_n(n, 0.0f);
    std::vector<float> tmp_t(t, 0.0f);

    for (int iter = 0; iter < 8; ++iter) {
        std::fill(tmp_n.begin(), tmp_n.end(), 0.0f);
        for (uint32_t r = 0; r < n; ++r) {
            tmp_n[r] = dot_row(m.row(r), v.data(), t);
        }
        std::fill(tmp_t.begin(), tmp_t.end(), 0.0f);
        for (uint32_t r = 0; r < n; ++r) {
            const float scale = tmp_n[r];
            const float * row = m.row(r);
            for (uint32_t c = 0; c < t; ++c) {
                tmp_t[c] += row[c] * scale;
            }
        }
        float norm = 0.0f;
        for (float x : tmp_t) {
            norm += x * x;
        }
        norm = std::sqrt(norm);
        if (norm <= 0.0f || !std::isfinite(norm)) {
            return 1.0f;
        }
        for (uint32_t c = 0; c < t; ++c) {
            v[c] = tmp_t[c] / norm;
        }
    }

    std::fill(tmp_n.begin(), tmp_n.end(), 0.0f);
    for (uint32_t r = 0; r < n; ++r) {
        tmp_n[r] = dot_row(m.row(r), v.data(), t);
    }
    std::fill(tmp_t.begin(), tmp_t.end(), 0.0f);
    for (uint32_t r = 0; r < n; ++r) {
        const float scale = tmp_n[r];
        const float * row = m.row(r);
        for (uint32_t c = 0; c < t; ++c) {
            tmp_t[c] += row[c] * scale;
        }
    }

    float num = 0.0f;
    for (uint32_t c = 0; c < t; ++c) {
        num += v[c] * tmp_t[c];
    }
    const float lipschitz = std::max(num + lambda, 1e-6f);
    return 1.0f / lipschitz;
}

void compute_exp_scores(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & keys,
        llama_kv_compact_matrix & exp_scores,
        std::vector<float> & max_scores,
        std::vector<float> * partition_sums) {
    const float inv_sqrt_d = 1.0f / std::sqrt(float(keys.cols));
    exp_scores.resize(queries.rows, keys.rows);
    max_scores.assign(queries.rows, -std::numeric_limits<float>::infinity());
    if (partition_sums) {
        partition_sums->assign(queries.rows, 0.0f);
    }

    for (uint32_t qi = 0; qi < queries.rows; ++qi) {
        const float * q = queries.row(qi);
        float row_max = -std::numeric_limits<float>::infinity();
        for (uint32_t ki = 0; ki < keys.rows; ++ki) {
            const float score = dot_row(q, keys.row(ki), keys.cols) * inv_sqrt_d;
            exp_scores(qi, ki) = score;
            row_max = std::max(row_max, score);
        }
        max_scores[qi] = row_max;

        float sum = 0.0f;
        for (uint32_t ki = 0; ki < keys.rows; ++ki) {
            const float e = std::exp(exp_scores(qi, ki) - row_max);
            exp_scores(qi, ki) = e;
            sum += e;
        }
        if (partition_sums) {
            (*partition_sums)[qi] = sum;
        }
    }
}

} // namespace

bool llama_kv_compact_fit_beta(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & full_keys,
        const llama_kv_compact_matrix & compacted_keys,
        const llama_kv_compact_solver_opts & opts,
        std::vector<float> & beta_out,
        float * partition_sum_relative_error) {
    if (queries.cols == 0 || full_keys.cols != queries.cols || compacted_keys.cols != queries.cols) {
        return false;
    }

    llama_kv_compact_matrix exp_full;
    llama_kv_compact_matrix exp_compact;
    std::vector<float> max_full;
    std::vector<float> max_compact;
    std::vector<float> target;
    std::vector<float> compact_sums;

    compute_exp_scores(queries, full_keys, exp_full, max_full, &target);
    compute_exp_scores(queries, compacted_keys, exp_compact, max_compact, nullptr);

    // Fix max-shift inconsistency: target uses exp(score - max_full), but
    // exp_compact uses exp(score - max_compact).  Rescale compact rows by
    // exp(max_compact - max_full) so both sides of the NNLS system use the
    // same per-query shift (the full-key max).  Paper formulation operates
    // in the consistent unshifted domain; this rescaling achieves equivalence.
    for (uint32_t qi = 0; qi < queries.rows; ++qi) {
        const float scale = std::exp(max_compact[qi] - max_full[qi]);
        for (uint32_t ki = 0; ki < compacted_keys.rows; ++ki) {
            exp_compact(qi, ki) *= scale;
        }
    }

    std::vector<float> weights;
    {
        float lambda = opts.lambda;
        bool solved = false;
        for (int attempt = 0; attempt < 5; ++attempt) {
            if (solve_vector_least_squares(exp_compact, target, lambda, weights)) {
                solved = true;
                break;
            }
            lambda = std::max(lambda * 10.0f, 1e-6f);
        }
        if (!solved) {
            return false;
        }
    }

    const float step = spectral_step_size(exp_compact, opts.lambda);
    std::vector<float> grad(weights.size(), 0.0f);
    for (int iter = 0; iter < opts.nnls_iters; ++iter) {
        std::fill(grad.begin(), grad.end(), 0.0f);
        for (uint32_t r = 0; r < exp_compact.rows; ++r) {
            const float * row = exp_compact.row(r);
            float pred = 0.0f;
            for (uint32_t c = 0; c < exp_compact.cols; ++c) {
                pred += row[c] * weights[c];
            }
            const float err = pred - target[r];
            for (uint32_t c = 0; c < exp_compact.cols; ++c) {
                grad[c] += row[c] * err;
            }
        }
        for (uint32_t c = 0; c < weights.size(); ++c) {
            weights[c] -= step * grad[c];
            weights[c] = std::max(weights[c], opts.nnls_lower_bound);
            if (opts.nnls_upper_bound > opts.nnls_lower_bound && std::isfinite(opts.nnls_upper_bound)) {
                weights[c] = std::min(weights[c], opts.nnls_upper_bound);
            }
        }
    }

    beta_out.resize(weights.size());
    float rel_err_sum = 0.0f;
    for (uint32_t r = 0; r < exp_compact.rows; ++r) {
        const float * row = exp_compact.row(r);
        float pred = 0.0f;
        for (uint32_t c = 0; c < exp_compact.cols; ++c) {
            pred += row[c] * weights[c];
        }
        rel_err_sum += std::fabs(pred - target[r]) / std::max(target[r], 1e-6f);
    }
    if (partition_sum_relative_error) {
        *partition_sum_relative_error = rel_err_sum / std::max<uint32_t>(1, exp_compact.rows);
    }

    for (uint32_t c = 0; c < weights.size(); ++c) {
        beta_out[c] = std::log(std::max(weights[c], opts.nnls_lower_bound));
    }
    return true;
}

bool llama_kv_compact_fit_values(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & full_keys,
        const llama_kv_compact_matrix & full_values,
        const llama_kv_compact_matrix & compacted_keys,
        const std::vector<float> & beta,
        const llama_kv_compact_solver_opts & opts,
        llama_kv_compact_matrix & compacted_values_out) {
    if (queries.cols == 0 || full_keys.cols != queries.cols || compacted_keys.cols != queries.cols ||
        full_values.rows != full_keys.rows || beta.size() != compacted_keys.rows) {
        return false;
    }

    llama_kv_compact_matrix y;
    llama_kv_compact_attention_output(queries, full_keys, full_values, nullptr, y, nullptr);

    llama_kv_compact_matrix x(queries.rows, compacted_keys.rows);
    const float inv_sqrt_d = 1.0f / std::sqrt(float(compacted_keys.cols));
    for (uint32_t qi = 0; qi < queries.rows; ++qi) {
        const float * q = queries.row(qi);
        float row_max = -std::numeric_limits<float>::infinity();
        for (uint32_t ki = 0; ki < compacted_keys.rows; ++ki) {
            const float score = dot_row(q, compacted_keys.row(ki), compacted_keys.cols) * inv_sqrt_d + beta[ki];
            x(qi, ki) = score;
            row_max = std::max(row_max, score);
        }
        float sum = 0.0f;
        for (uint32_t ki = 0; ki < compacted_keys.rows; ++ki) {
            const float e = std::exp(x(qi, ki) - row_max);
            x(qi, ki) = e;
            sum += e;
        }
        for (uint32_t ki = 0; ki < compacted_keys.rows; ++ki) {
            x(qi, ki) /= std::max(sum, 1e-6f);
        }
    }

    float lambda = opts.lambda;
    for (int attempt = 0; attempt < 5; ++attempt) {
        if (solve_least_squares_normal_eq(x, y, lambda, compacted_values_out)) {
            return true;
        }
        lambda = std::max(lambda * 10.0f, 1e-6f);
    }
    return false;
}

void llama_kv_compact_attention_output(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & keys,
        const llama_kv_compact_matrix & values,
        const std::vector<float> * beta,
        llama_kv_compact_matrix & output,
        std::vector<float> * partition_sums) {
    output.resize(queries.rows, values.cols);
    if (partition_sums) {
        partition_sums->assign(queries.rows, 0.0f);
    }

    const float inv_sqrt_d = 1.0f / std::sqrt(float(keys.cols));
    std::vector<float> attn(keys.rows, 0.0f);
    for (uint32_t qi = 0; qi < queries.rows; ++qi) {
        const float * q = queries.row(qi);
        float row_max = -std::numeric_limits<float>::infinity();
        for (uint32_t ki = 0; ki < keys.rows; ++ki) {
            float score = dot_row(q, keys.row(ki), keys.cols) * inv_sqrt_d;
            if (beta) {
                score += (*beta)[ki];
            }
            attn[ki] = score;
            row_max = std::max(row_max, score);
        }
        float sum = 0.0f;
        for (uint32_t ki = 0; ki < keys.rows; ++ki) {
            attn[ki] = std::exp(attn[ki] - row_max);
            sum += attn[ki];
        }
        if (partition_sums) {
            (*partition_sums)[qi] = sum;
        }
        const float inv_sum = 1.0f / std::max(sum, 1e-6f);
        for (uint32_t c = 0; c < values.cols; ++c) {
            float acc = 0.0f;
            for (uint32_t ki = 0; ki < keys.rows; ++ki) {
                acc += attn[ki] * inv_sum * values(ki, c);
            }
            output(qi, c) = acc;
        }
    }
}

float llama_kv_compact_cosine_similarity(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size() || lhs.empty()) {
        return 0.0f;
    }

    double dot = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        dot += double(lhs[i]) * rhs[i];
        lhs_norm += double(lhs[i]) * lhs[i];
        rhs_norm += double(rhs[i]) * rhs[i];
    }
    if (lhs_norm <= 0.0 || rhs_norm <= 0.0) {
        return 0.0f;
    }
    return float(dot / (std::sqrt(lhs_norm) * std::sqrt(rhs_norm)));
}
