#pragma once

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

    float * row(uint32_t r) { return data.data() + size_t(r) * cols; }
    const float * row(uint32_t r) const { return data.data() + size_t(r) * cols; }

    float & operator()(uint32_t r, uint32_t c) { return data[size_t(r) * cols + c]; }
    float   operator()(uint32_t r, uint32_t c) const { return data[size_t(r) * cols + c]; }
};

struct llama_kv_compact_solver_opts {
    float lambda = 1e-6f;
    int   nnls_iters = 2;          // paper uses 0 (OMP) or 2 (HighestAttnKeys)
    float nnls_lower_bound = 0.05f; // paper: e^{-3} ≈ 0.05, prevents near-zero weights
    float nnls_upper_bound = 20.0f; // paper: e^3 ≈ 20.1
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
