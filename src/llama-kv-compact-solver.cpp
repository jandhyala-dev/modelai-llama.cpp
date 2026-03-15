// V2 solver core — faithful port of MIT reference (algorithms/base.py)
//
// Primary solver: LAPACK sgels (QR-based, condition κ) via macOS Accelerate
// Fallback: Cholesky with symmetrization (condition κ²) + ridge regularization
//
// Changes from V1:
//   - NNLS: lstsq+clamp replaces log-domain gradient descent
//   - V fitting: 3-tier cascade (sgels → cholesky+sym → aggressive_cholesky)
//   - Ridge: spectral/frobenius/fixed modes, spectral→frobenius fallback, no cap
//   - Underdetermined case (n < t) handled via XXᵀ formulation
//   - Bounds: [1e-12, None] replaces [0.05, 20.0]
//   - Max-shift rescaling preserved (correct for C++ fp32)

#include "llama-kv-compact-solver.h"
#include "llama-kv-compact-math.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define LLAMA_KV_COMPACT_HAS_LAPACK 1
typedef __LAPACK_int lapack_int;
#else
#define LLAMA_KV_COMPACT_HAS_LAPACK 0
#endif

namespace {

using llama_kv_compact_math::dot_row;

// -----------------------------------------------------------------------
// Cholesky solver (kept from V1, used as fallback)
// -----------------------------------------------------------------------

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

// -----------------------------------------------------------------------
// Symmetrization — MIT base.py:191-192
// Prevents fp32 rounding from making XᵀX asymmetric before Cholesky
// -----------------------------------------------------------------------

void symmetrize_inplace(std::vector<float> & a, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < i; ++j) {
            float avg = 0.5f * (a[size_t(i) * n + j] + a[size_t(j) * n + i]);
            a[size_t(i) * n + j] = avg;
            a[size_t(j) * n + i] = avg;
        }
    }
}

// -----------------------------------------------------------------------
// Normal equations with symmetrization (V2 fallback solver)
// Solves (XᵀX + λI)C = XᵀY via Cholesky with symmetrization
// -----------------------------------------------------------------------

bool solve_least_squares_cholesky_sym(
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

    // Fill upper triangle
    for (uint32_t i = 0; i < t; ++i) {
        for (uint32_t j = 0; j < i; ++j) {
            xtx[size_t(j) * t + i] = xtx[size_t(i) * t + j];
        }
    }

    // Symmetrize (MIT base.py:191-192) then add ridge
    symmetrize_inplace(xtx, t);
    for (uint32_t i = 0; i < t; ++i) {
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

// Underdetermined case (n < t): solve (XXᵀ + λI)Z = Y, then C = XᵀZ
// MIT base.py:183-189 — minimum-norm solution
bool solve_least_squares_underdetermined(
        const llama_kv_compact_matrix & x,
        const llama_kv_compact_matrix & y,
        float lambda,
        llama_kv_compact_matrix & out) {
    const uint32_t n = x.rows;
    const uint32_t t = x.cols;
    const uint32_t d = y.cols;

    // Build XXᵀ (n × n)
    std::vector<float> xxt(size_t(n) * n, 0.0f);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            float sum = dot_row(x.row(i), x.row(j), t);
            xxt[size_t(i) * n + j] = sum;
            xxt[size_t(j) * n + i] = sum;
        }
    }

    symmetrize_inplace(xxt, n);
    for (uint32_t i = 0; i < n; ++i) {
        xxt[size_t(i) * n + i] += lambda;
    }

    // Solve (XXᵀ + λI)Z = Y → Z is (n, d)
    std::vector<float> z(size_t(d) * n);
    for (uint32_t c = 0; c < d; ++c) {
        for (uint32_t i = 0; i < n; ++i) {
            z[size_t(c) * n + i] = y(i, c);
        }
    }

    if (!solve_spd_cholesky(xxt, n, z, d)) {
        return false;
    }

    // C = XᵀZ → (t, d)
    out.resize(t, d);
    for (uint32_t i = 0; i < t; ++i) {
        for (uint32_t c = 0; c < d; ++c) {
            float sum = 0.0f;
            for (uint32_t r = 0; r < n; ++r) {
                sum += x(r, i) * z[size_t(c) * n + r];
            }
            out(i, c) = sum;
        }
    }
    return true;
}

// Single-RHS version for NNLS beta fitting
bool solve_vector_least_squares_sym(
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
    }

    symmetrize_inplace(xtx, t);
    for (uint32_t i = 0; i < t; ++i) {
        xtx[size_t(i) * t + i] += lambda;
    }

    std::vector<float> rhs = xty;
    if (!solve_spd_cholesky(xtx, t, rhs, 1)) {
        return false;
    }
    out = std::move(rhs);
    return true;
}

// -----------------------------------------------------------------------
// LAPACK sgels wrapper (macOS Accelerate)
// Solves min ||Ax - b||_2 via QR decomposition
// -----------------------------------------------------------------------

#if LLAMA_KV_COMPACT_HAS_LAPACK

// Solve X*C = Y where X is (n, t), Y is (n, d), C is (t, d)
// Uses LAPACK sgels for QR-based least squares (condition κ, not κ²)
// Returns false if sgels fails or produces NaN
bool solve_least_squares_lapack(
        const llama_kv_compact_matrix & x,
        const llama_kv_compact_matrix & y,
        llama_kv_compact_matrix & out) {
    const uint32_t n = x.rows;
    const uint32_t t = x.cols;
    const uint32_t d = y.cols;

    if (n == 0 || t == 0 || d == 0 || n != y.rows) {
        return false;
    }

    // sgels expects column-major. Convert row-major X(n,t) to column-major A(n,t)
    const lapack_int m_l = (lapack_int)n;
    const lapack_int n_l = (lapack_int)t;
    const lapack_int nrhs = (lapack_int)d;
    const lapack_int lda = m_l;
    const lapack_int ldb = std::max(m_l, n_l);

    std::vector<float> a(size_t(m_l) * n_l);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < t; ++j) {
            a[size_t(j) * m_l + i] = x(i, j);
        }
    }

    // B is max(m,n) × nrhs in column-major. Copy Y into first m rows.
    std::vector<float> b(size_t(ldb) * nrhs, 0.0f);
    for (uint32_t c = 0; c < d; ++c) {
        for (uint32_t i = 0; i < n; ++i) {
            b[size_t(c) * ldb + i] = y(i, c);
        }
    }

    // Query optimal workspace
    char trans = 'N';
    lapack_int info = 0;
    float work_query = 0.0f;
    lapack_int lwork = -1;
    lapack_int lda_l = lda;
    lapack_int ldb_l = ldb;

    sgels_(&trans, &m_l, &n_l, &nrhs, a.data(), &lda_l, b.data(), &ldb_l,
           &work_query, &lwork, &info);

    lwork = (lapack_int)work_query;
    if (lwork < 1) lwork = 1;
    std::vector<float> work(static_cast<size_t>(lwork), 0.0f);

    // Solve
    sgels_(&trans, &m_l, &n_l, &nrhs, a.data(), &lda_l, b.data(), &ldb_l,
           work.data(), &lwork, &info);

    if (info != 0) {
        return false;
    }

    // Extract solution from first t rows of B
    out.resize(t, d);
    for (uint32_t c = 0; c < d; ++c) {
        for (uint32_t i = 0; i < t; ++i) {
            float val = b[size_t(c) * ldb + i];
            if (!std::isfinite(val)) {
                return false;
            }
            out(i, c) = val;
        }
    }
    return true;
}

// Single-RHS version for NNLS
bool solve_vector_least_squares_lapack(
        const llama_kv_compact_matrix & x,
        const std::vector<float> & y,
        std::vector<float> & out) {
    const uint32_t n = x.rows;
    const uint32_t t = x.cols;

    if (n == 0 || t == 0 || y.size() != n) {
        return false;
    }

    llama_kv_compact_matrix y_mat(n, 1);
    for (uint32_t i = 0; i < n; ++i) {
        y_mat(i, 0) = y[i];
    }

    llama_kv_compact_matrix out_mat;
    if (!solve_least_squares_lapack(x, y_mat, out_mat)) {
        return false;
    }

    out.resize(t);
    for (uint32_t i = 0; i < t; ++i) {
        out[i] = out_mat(i, 0);
    }
    return true;
}

#endif // LLAMA_KV_COMPACT_HAS_LAPACK

// -----------------------------------------------------------------------
// Spectral norm via power iteration
// Returns largest eigenvalue of MᵀM (= σ_max(M)²)
// -----------------------------------------------------------------------

float compute_spectral_norm(const llama_kv_compact_matrix & m) {
    const uint32_t n = m.rows;
    const uint32_t t = m.cols;
    if (t == 0 || n == 0) {
        return 1.0f;
    }

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

    // Final Rayleigh quotient: vᵀ(MᵀM)v
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
    return std::max(num, 1e-6f);
}

// Frobenius norm squared divided by t = average eigenvalue of XᵀX
float compute_frobenius_scale(const llama_kv_compact_matrix & m) {
    float sum = 0.0f;
    for (size_t i = 0; i < m.data.size(); ++i) {
        sum += m.data[i] * m.data[i];
    }
    return sum / std::max<uint32_t>(m.cols, 1);
}

// -----------------------------------------------------------------------
// Ridge scaling — MIT base.py:146-161
// Spectral→frobenius automatic fallback
// NO CAP (V1 had min(..., 1.0f) which is incorrect)
// -----------------------------------------------------------------------

float compute_effective_lambda(
        const llama_kv_compact_matrix & design,
        float lambda_base,
        llama_kv_compact_ridge_scale mode) {
    if (lambda_base <= 0.0f) {
        return 0.0f;
    }

    switch (mode) {
        case LLAMA_KV_COMPACT_RIDGE_SPECTRAL: {
            float sn = compute_spectral_norm(design);
            if (sn > 1e-6f && std::isfinite(sn)) {
                float scaled = lambda_base * sn;  // sn is already σ²_max
                if (std::isfinite(scaled) && scaled > 0.0f) {
                    return scaled;
                }
            }
            // Fallback to frobenius (MIT base.py:152-155)
            return lambda_base * compute_frobenius_scale(design);
        }
        case LLAMA_KV_COMPACT_RIDGE_FROBENIUS:
            return lambda_base * compute_frobenius_scale(design);
        case LLAMA_KV_COMPACT_RIDGE_FIXED:
            return lambda_base;
        default:
            return lambda_base;
    }
}

// PGD step size: 1 / (σ_max(M)² + λ)
float spectral_step_size(const llama_kv_compact_matrix & m, float lambda) {
    const float spectral_norm = compute_spectral_norm(m);
    const float lipschitz = std::max(spectral_norm + lambda, 1e-6f);
    return 1.0f / lipschitz;
}

// -----------------------------------------------------------------------
// Exp-domain score computation (shared)
// -----------------------------------------------------------------------

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

// -----------------------------------------------------------------------
// V2 NNLS solver — MIT base.py:471-605
//
// Mode 1 (nnls_iters=0): lstsq + clamp (MIT default)
//   Primary: LAPACK sgels (QR, condition κ)
//   Fallback: Cholesky + symmetrization with λ=1e-8
//
// Mode 2 (nnls_iters>0): lstsq + clamp + PGD refinement
// -----------------------------------------------------------------------

bool solve_nnls_v2(
        const llama_kv_compact_matrix & m,
        const std::vector<float> & target,
        const llama_kv_compact_solver_opts & opts,
        std::vector<float> & weights_out) {
    const uint32_t n = m.rows;
    const uint32_t t = m.cols;

    if (n == 0 || t == 0 || target.size() != n) {
        return false;
    }

    const float min_val = (opts.nnls_lower_bound > 0.0f) ? opts.nnls_lower_bound : 1e-12f;
    bool solved = false;

    // Phase 1: Try LAPACK sgels (QR-based, condition κ)
#if LLAMA_KV_COMPACT_HAS_LAPACK
    if (!solved) {
        solved = solve_vector_least_squares_lapack(m, target, weights_out);
        if (solved) {
            // Check for NaN — MIT base.py:500-505
            for (float w : weights_out) {
                if (!std::isfinite(w)) {
                    solved = false;
                    break;
                }
            }
        }
    }
#endif

    // Phase 2: Cholesky fallback with symmetrization — MIT base.py:510-530
    if (!solved) {
        float lam = 1e-6f;  // MIT default fallback lambda
        for (int attempt = 0; attempt < 5; ++attempt) {
            if (solve_vector_least_squares_sym(m, target, lam, weights_out)) {
                solved = true;
                break;
            }
            lam *= 10.0f;
        }
    }

    if (!solved) {
        return false;
    }

    // Phase 3: Clamp to bounds — MIT base.py:548-553
    for (uint32_t c = 0; c < weights_out.size(); ++c) {
        weights_out[c] = std::max(weights_out[c], min_val);
        if (opts.nnls_upper_bound > min_val && std::isfinite(opts.nnls_upper_bound)) {
            weights_out[c] = std::min(weights_out[c], opts.nnls_upper_bound);
        }
    }

    // Phase 4: PGD refinement if nnls_iters > 0 — MIT base.py:561-605
    if (opts.nnls_iters > 0) {
        const float step = spectral_step_size(m, 0.0f);
        std::vector<float> grad(t, 0.0f);

        for (int iter = 0; iter < opts.nnls_iters; ++iter) {
            std::fill(grad.begin(), grad.end(), 0.0f);
            for (uint32_t r = 0; r < n; ++r) {
                const float * row = m.row(r);
                float pred = 0.0f;
                for (uint32_t c = 0; c < t; ++c) {
                    pred += row[c] * weights_out[c];
                }
                const float err = pred - target[r];
                for (uint32_t c = 0; c < t; ++c) {
                    grad[c] += row[c] * err;
                }
            }
            for (uint32_t c = 0; c < t; ++c) {
                weights_out[c] -= step * grad[c];
                weights_out[c] = std::max(weights_out[c], min_val);
                if (opts.nnls_upper_bound > min_val && std::isfinite(opts.nnls_upper_bound)) {
                    weights_out[c] = std::min(weights_out[c], opts.nnls_upper_bound);
                }
            }
        }
    }

    return true;
}

// -----------------------------------------------------------------------
// V2 value fitting — MIT base.py:61-240
//
// 3-tier cascade:
//   Tier 1: LAPACK sgels (QR, condition κ)
//   Tier 2: Cholesky + symmetrization + ridge (condition κ²)
//   Tier 3: Aggressive Cholesky (large λ, biased but stable)
//
// Handles underdetermined case (n < t) via XXᵀ formulation
// -----------------------------------------------------------------------

bool solve_values_v2(
        const llama_kv_compact_matrix & x,
        const llama_kv_compact_matrix & y,
        float effective_lambda,
        llama_kv_compact_matrix & out) {
    const uint32_t n = x.rows;
    const uint32_t t = x.cols;

    // Tier 1: LAPACK sgels (QR-based)
#if LLAMA_KV_COMPACT_HAS_LAPACK
    {
        llama_kv_compact_matrix result;
        if (solve_least_squares_lapack(x, y, result)) {
            bool has_nan = false;
            for (float v : result.data) {
                if (!std::isfinite(v)) {
                    has_nan = true;
                    break;
                }
            }
            if (!has_nan) {
                out = std::move(result);
                return true;
            }
        }
    }
#endif

    // Tier 2: Cholesky with symmetrization and ridge
    float lam = std::max(effective_lambda, 1e-8f);
    if (n < t) {
        // Underdetermined: (XXᵀ + λI)Z = Y, C = XᵀZ
        if (solve_least_squares_underdetermined(x, y, lam, out)) {
            return true;
        }
    } else {
        // Overdetermined: (XᵀX + λI)C = XᵀY
        if (solve_least_squares_cholesky_sym(x, y, lam, out)) {
            return true;
        }
    }

    // Tier 3: Aggressive Cholesky — escalate lambda
    for (int attempt = 0; attempt < 5; ++attempt) {
        lam = std::max(lam * 10.0f, 1e-4f);
        if (n < t) {
            if (solve_least_squares_underdetermined(x, y, lam, out)) {
                return true;
            }
        } else {
            if (solve_least_squares_cholesky_sym(x, y, lam, out)) {
                return true;
            }
        }
    }

    return false;
}

} // namespace

// -----------------------------------------------------------------------
// Public API: Beta fitting (NNLS)
// -----------------------------------------------------------------------

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

    compute_exp_scores(queries, full_keys, exp_full, max_full, &target);
    compute_exp_scores(queries, compacted_keys, exp_compact, max_compact, nullptr);

    // Max-shift rescaling (KEEP from V1 — correct for C++ fp32)
    for (uint32_t qi = 0; qi < queries.rows; ++qi) {
        const float scale = std::exp(max_compact[qi] - max_full[qi]);
        for (uint32_t ki = 0; ki < compacted_keys.rows; ++ki) {
            exp_compact(qi, ki) *= scale;
        }
    }

    // Solve NNLS
    std::vector<float> weights;
    if (!solve_nnls_v2(exp_compact, target, opts, weights)) {
        return false;
    }

    // Compute partition sum relative error
    const float min_val = (opts.nnls_lower_bound > 0.0f) ? opts.nnls_lower_bound : 1e-12f;
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
        beta_out[c] = std::log(std::max(weights[c], min_val));
    }
    return true;
}

// -----------------------------------------------------------------------
// Public API: Value fitting
// -----------------------------------------------------------------------

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

    // Compute target: Y = softmax(QK^T / sqrt(d)) @ V
    llama_kv_compact_matrix y;
    llama_kv_compact_attention_output(queries, full_keys, full_values, nullptr, y, nullptr);

    // Build design matrix: X = softmax(QC1^T / sqrt(d) + beta)
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

    // Compute effective ridge lambda
    float effective_lambda = compute_effective_lambda(x, opts.lambda, opts.ridge_scale);

    // Solve via 3-tier cascade
    return solve_values_v2(x, y, effective_lambda, compacted_values_out);
}

// -----------------------------------------------------------------------
// Public API: Attention output (unchanged from V1)
// -----------------------------------------------------------------------

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

// -----------------------------------------------------------------------
// Public API: Cosine similarity (unchanged from V1)
// -----------------------------------------------------------------------

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
