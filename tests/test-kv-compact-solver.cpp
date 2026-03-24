#include "src/llama-kv-compact-select.h"
#include "src/llama-kv-compact-solver.h"
#include "kv-compact-test-helpers.h"
#include "kv-compact-thresholds.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using llama_kv_compact_test::check;
using llama_kv_compact_test::fail;

namespace {

static const std::string test_prefix = "test-kv-compact-solver";

} // namespace

int main() {
    int rc = 0;

    {
        const std::vector<float> scores = { 0.1f, 0.9f, 0.3f, 0.8f };
        const auto top = llama_kv_compact_select_topk(scores, 2);
        if (!check(test_prefix, top.size() == 2 && top[0] == 1 && top[1] == 3, "top-k selection should keep the highest-scoring indices in ascending position order", rc)) return rc;
    }

    llama_kv_compact_matrix queries(4, 2);
    llama_kv_compact_matrix full_k(4, 2);
    llama_kv_compact_matrix full_v(4, 2);

    const float qk_vals[4][2] = {
        { 1.0f, 0.0f },
        { 0.0f, 1.0f },
        { 1.0f, 1.0f },
        { 1.0f,-1.0f },
    };
    const float v_vals[4][2] = {
        { 1.0f, 0.0f },
        { 0.0f, 1.0f },
        { 1.0f, 1.0f },
        { 1.0f,-1.0f },
    };

    for (uint32_t r = 0; r < 4; ++r) {
        for (uint32_t c = 0; c < 2; ++c) {
            queries(r, c) = qk_vals[r][c];
            full_k(r, c) = qk_vals[r][c];
            full_v(r, c) = v_vals[r][c];
        }
    }

    llama_kv_compact_matrix compacted_k(2, 2);
    compacted_k(0, 0) = full_k(0, 0); compacted_k(0, 1) = full_k(0, 1);
    compacted_k(1, 0) = full_k(2, 0); compacted_k(1, 1) = full_k(2, 1);

    llama_kv_compact_solver_opts opts = {};
    opts.lambda = 1e-6f;
    opts.nnls_iters = 0;       // V2: lstsq+clamp (MIT default)
    opts.nnls_upper_bound = 0.0f;  // V2: no upper bound

    std::vector<float> beta;
    float partition_rel_err = 0.0f;
    if (!check(test_prefix, llama_kv_compact_fit_beta(queries, full_k, compacted_k, opts, beta, &partition_rel_err), "beta fit should succeed", rc)) return rc;
    if (!check(test_prefix, beta.size() == 2, "beta size should match compacted token count", rc)) return rc;
    if (!check(test_prefix, partition_rel_err >= 0.0f && partition_rel_err < 0.5f, "partition relative error should stay bounded on the synthetic fixture", rc)) return rc;

    llama_kv_compact_matrix compacted_v;
    if (!check(test_prefix, llama_kv_compact_fit_values(queries, full_k, full_v, compacted_k, beta, opts, compacted_v), "value fit should succeed", rc)) return rc;
    if (!check(test_prefix, compacted_v.rows == 2 && compacted_v.cols == 2, "compacted V shape should match selected token count and value dim", rc)) return rc;

    llama_kv_compact_matrix full_out;
    llama_kv_compact_matrix compact_out;
    std::vector<float> partition_sums;
    llama_kv_compact_attention_output(queries, full_k, full_v, nullptr, full_out, &partition_sums);
    llama_kv_compact_attention_output(queries, compacted_k, compacted_v, &beta, compact_out, nullptr);

    if (!check(test_prefix, full_out.data.size() == compact_out.data.size(), "attention outputs should be comparable", rc)) return rc;
    const float cos = llama_kv_compact_cosine_similarity(full_out.data, compact_out.data);
    if (!check(test_prefix, cos >= llama_kv_compact_thresholds::COS_SOLVER_BASELINE, "synthetic attention-output cosine should meet the baseline threshold", rc)) return rc;

    // OMP key selection test
    {
        llama_kv_compact_omp_opts omp_opts;  // defaults: progressive schedule, drop-key enabled
        std::vector<float> omp_beta;

        auto omp_selected = llama_kv_compact_select_omp(
            queries, full_k, 2, omp_opts, omp_beta);

        if (!check(test_prefix, omp_selected.size() == 2, "OMP should select exactly 2 keys", rc)) return rc;
        if (!check(test_prefix, omp_beta.size() == 2, "OMP should produce 2 beta values", rc)) return rc;
        std::printf("  OMP selected positions: %u, %u\n",
                    omp_selected[0], omp_selected[1]);
        std::printf("  OMP beta: %.4f, %.4f\n", omp_beta[0], omp_beta[1]);

        for (float b : omp_beta) {
            if (!check(test_prefix, std::isfinite(b), "OMP beta values should be finite", rc)) return rc;
        }

        // OMP results should be sorted by position
        if (!check(test_prefix, omp_selected[0] < omp_selected[1], "OMP results should be position-sorted", rc)) return rc;
    }

    return 0;
}
