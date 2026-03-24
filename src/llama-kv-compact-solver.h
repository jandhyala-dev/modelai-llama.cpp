#pragma once

#include "ggml.h"

#include <cstdint>
#include <vector>

struct llama_kv_compact_matrix {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<float> data;

    llama_kv_compact_matrix() = default;
    llama_kv_compact_matrix(uint32_t rows_, uint32_t cols_) : rows(rows_), cols(cols_), data(size_t(rows_) * cols_) {}

    void resize(uint32_t rows_, uint32_t cols_) {
        rows = rows_;
        cols = cols_;
        data.assign(size_t(rows_) * cols_, 0.0f);
    }

    float * row(uint32_t r) { GGML_ASSERT(r < rows); return data.data() + size_t(r) * cols; }
    const float * row(uint32_t r) const { GGML_ASSERT(r < rows); return data.data() + size_t(r) * cols; }

    float & operator()(uint32_t r, uint32_t c) { GGML_ASSERT(r < rows && c < cols); return data[size_t(r) * cols + c]; }
    float   operator()(uint32_t r, uint32_t c) const { GGML_ASSERT(r < rows && c < cols); return data[size_t(r) * cols + c]; }
};

// Ridge scaling modes matching MIT reference (algorithms/base.py:146-161)
enum llama_kv_compact_ridge_scale {
    LLAMA_KV_COMPACT_RIDGE_SPECTRAL  = 0, // λ × σ_max(X)² with frobenius fallback
    LLAMA_KV_COMPACT_RIDGE_FROBENIUS = 1, // λ × (‖X‖²_F / t)
    LLAMA_KV_COMPACT_RIDGE_FIXED     = 2, // λ (raw)
};

struct llama_kv_compact_solver_opts {
    float lambda          = 1e-6f;
    int   nnls_iters      = 0;       // V2: 0 = lstsq+clamp (MIT default), >0 = PGD refinement
    float nnls_lower_bound = 1e-12f; // V2: MIT default (was 0.05 in V1)
    float nnls_upper_bound = 0.0f;   // V2: 0 = no upper bound (MIT default; was 20.0 in V1)

    llama_kv_compact_ridge_scale ridge_scale = LLAMA_KV_COMPACT_RIDGE_SPECTRAL;
};

struct llama_kv_compact_quality_metrics {
    float attention_output_cosine = 0.0f;
    float continuation_logit_cosine = 0.0f;
    float partition_sum_relative_error = 0.0f;
};

bool llama_kv_compact_fit_beta(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & full_keys,
        const llama_kv_compact_matrix & compacted_keys,
        const llama_kv_compact_solver_opts & opts,
        std::vector<float> & beta_out,
        float * partition_sum_relative_error = nullptr);

bool llama_kv_compact_fit_values(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & full_keys,
        const llama_kv_compact_matrix & full_values,
        const llama_kv_compact_matrix & compacted_keys,
        const std::vector<float> & beta,
        const llama_kv_compact_solver_opts & opts,
        llama_kv_compact_matrix & compacted_values_out);

void llama_kv_compact_attention_output(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & keys,
        const llama_kv_compact_matrix & values,
        const std::vector<float> * beta,
        llama_kv_compact_matrix & output,
        std::vector<float> * partition_sums = nullptr);

float llama_kv_compact_cosine_similarity(const std::vector<float> & lhs, const std::vector<float> & rhs);
