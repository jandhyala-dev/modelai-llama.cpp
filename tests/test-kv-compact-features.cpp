// Unit tests for new KV compaction features:
//   - RMS score aggregation (Appendix F.1)
//   - OMP key pruning (Appendix C.2)
//   - Spectral ridge scaling
//   - Per-head budget allocation (Algorithm 4 / Section 3.4)
//   - Union building with per-head masking

#include "src/llama-kv-compact-select.h"
#include "src/llama-kv-compact-solver.h"
#include "src/llama-kv-compact-budget.h"

#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace {

int n_passed = 0;
int n_failed = 0;

bool check(bool cond, const std::string & msg) {
    if (!cond) {
        std::fprintf(stderr, "  FAIL: %s\n", msg.c_str());
        n_failed++;
        return false;
    }
    std::printf("  PASS: %s\n", msg.c_str());
    n_passed++;
    return true;
}

// Build a simple 4x2 Q/K fixture.
void make_fixture(llama_kv_compact_matrix & queries, llama_kv_compact_matrix & keys) {
    queries.resize(4, 2);
    keys.resize(4, 2);
    const float vals[4][2] = {
        { 1.0f, 0.0f },
        { 0.0f, 1.0f },
        { 1.0f, 1.0f },
        { 1.0f,-1.0f },
    };
    for (uint32_t r = 0; r < 4; ++r) {
        for (uint32_t c = 0; c < 2; ++c) {
            queries(r, c) = vals[r][c];
            keys(r, c)    = vals[r][c];
        }
    }
}

} // namespace

int main() {
    // -----------------------------------------------------------------------
    // 1. RMS score aggregation (Appendix F.1)
    // -----------------------------------------------------------------------
    std::printf("\n=== RMS Score Aggregation ===\n");
    {
        llama_kv_compact_matrix queries, keys;
        make_fixture(queries, keys);

        // SUM mode baseline.
        std::vector<float> sum_scores(4, 0.0f);
        llama_kv_compact_accumulate_attention_scores(
                queries, keys, sum_scores,
                LLAMA_KV_COMPACT_SCORE_AGG_SUM);

        check(sum_scores.size() == 4, "SUM scores have correct size");
        float sum_total = 0.0f;
        for (float s : sum_scores) sum_total += s;
        check(sum_total > 0.0f, "SUM scores are non-zero");

        // RMS mode.
        std::vector<float> rms_scores(4, 0.0f);
        uint32_t n_queries = 0;
        llama_kv_compact_accumulate_attention_scores(
                queries, keys, rms_scores,
                LLAMA_KV_COMPACT_SCORE_AGG_RMS, &n_queries);

        check(n_queries == 4, "RMS n_queries counts all query rows");

        // Before finalization, RMS scores should be sum of squared weights.
        for (float s : rms_scores) {
            check(s >= 0.0f, "RMS pre-finalize scores are non-negative");
        }

        // Finalize: sqrt(mean(squared)).
        llama_kv_compact_finalize_rms_scores(rms_scores, n_queries);

        for (float s : rms_scores) {
            check(s >= 0.0f && std::isfinite(s), "RMS finalized scores are non-negative and finite");
        }

        // RMS scores should generally differ from SUM scores.
        bool any_differ = false;
        for (size_t i = 0; i < 4; ++i) {
            if (std::abs(sum_scores[i] - rms_scores[i]) > 1e-6f) {
                any_differ = true;
            }
        }
        check(any_differ, "RMS and SUM produce different score distributions");

        // Top-k selection should work on both.
        auto topk_sum = llama_kv_compact_select_topk(sum_scores, 2);
        auto topk_rms = llama_kv_compact_select_topk(rms_scores, 2);
        check(topk_sum.size() == 2, "SUM top-k selects correct count");
        check(topk_rms.size() == 2, "RMS top-k selects correct count");
    }

    // -----------------------------------------------------------------------
    // 2. RMS edge cases
    // -----------------------------------------------------------------------
    std::printf("\n=== RMS Edge Cases ===\n");
    {
        // Zero queries.
        std::vector<float> scores(4, 0.0f);
        llama_kv_compact_finalize_rms_scores(scores, 0);
        check(scores[0] == 0.0f, "RMS finalize with 0 queries is no-op");

        // Empty scores vector.
        llama_kv_compact_matrix empty_q(0, 2);
        llama_kv_compact_matrix keys(4, 2);
        for (uint32_t r = 0; r < 4; ++r) {
            keys(r, 0) = float(r);
            keys(r, 1) = 1.0f;
        }
        std::vector<float> empty_scores(4, 0.0f);
        uint32_t nq = 0;
        llama_kv_compact_accumulate_attention_scores(
                empty_q, keys, empty_scores,
                LLAMA_KV_COMPACT_SCORE_AGG_RMS, &nq);
        check(nq == 0, "Empty queries produce zero n_queries");
    }

    // -----------------------------------------------------------------------
    // 3. OMP key pruning (Appendix C.2)
    // -----------------------------------------------------------------------
    std::printf("\n=== OMP Key Pruning ===\n");
    {
        llama_kv_compact_matrix queries, keys;
        make_fixture(queries, keys);

        // Default opts with pruning enabled.
        llama_kv_compact_omp_opts omp_opts;
        omp_opts.k_choice = 1;
        omp_opts.nnls_interval = 1;
        omp_opts.beta_prune_log_threshold = -7.0f;

        std::vector<float> beta;
        auto selected = llama_kv_compact_select_omp(queries, keys, 3, omp_opts, beta);

        check(!selected.empty(), "OMP with pruning selects at least 1 key");
        check(selected.size() <= 3, "OMP selects at most t keys");
        check(beta.size() == selected.size(), "OMP beta size matches selection");

        for (float b : beta) {
            check(std::isfinite(b), "OMP beta values are finite");
        }

        // Verify position ordering.
        for (size_t i = 1; i < selected.size(); ++i) {
            check(selected[i] > selected[i-1], "OMP results are position-sorted");
        }

        // Pruning disabled (threshold = -inf).
        llama_kv_compact_omp_opts no_prune_opts;
        no_prune_opts.k_choice = 1;
        no_prune_opts.nnls_interval = 1;
        no_prune_opts.beta_prune_log_threshold = -std::numeric_limits<float>::infinity();

        std::vector<float> beta_no_prune;
        auto selected_no_prune = llama_kv_compact_select_omp(
                queries, keys, 3, no_prune_opts, beta_no_prune);

        check(selected_no_prune.size() == 3, "OMP without pruning always selects t keys");
    }

    // -----------------------------------------------------------------------
    // 4. Spectral ridge scaling
    // -----------------------------------------------------------------------
    std::printf("\n=== Spectral Ridge Scaling ===\n");
    {
        llama_kv_compact_matrix queries, keys;
        make_fixture(queries, keys);

        llama_kv_compact_matrix compacted_k(2, 2);
        compacted_k(0, 0) = keys(0, 0); compacted_k(0, 1) = keys(0, 1);
        compacted_k(1, 0) = keys(2, 0); compacted_k(1, 1) = keys(2, 1);

        // Without spectral ridge.
        llama_kv_compact_solver_opts opts_no_sr = {};
        opts_no_sr.spectral_ridge = false;
        opts_no_sr.lambda = 1e-6f;
        opts_no_sr.nnls_iters = 2;

        std::vector<float> beta_no_sr;
        float err_no_sr = 0.0f;
        check(llama_kv_compact_fit_beta(queries, keys, compacted_k,
                                         opts_no_sr, beta_no_sr, &err_no_sr),
              "fit_beta without spectral_ridge succeeds");

        // With spectral ridge.
        llama_kv_compact_solver_opts opts_sr = opts_no_sr;
        opts_sr.spectral_ridge = true;

        std::vector<float> beta_sr;
        float err_sr = 0.0f;
        check(llama_kv_compact_fit_beta(queries, keys, compacted_k,
                                         opts_sr, beta_sr, &err_sr),
              "fit_beta with spectral_ridge succeeds");

        check(beta_sr.size() == beta_no_sr.size(),
              "spectral_ridge produces same-size beta");

        // Both should produce finite values.
        for (size_t i = 0; i < beta_sr.size(); ++i) {
            check(std::isfinite(beta_sr[i]), "spectral_ridge beta is finite");
        }

        // V fitting with spectral ridge.
        llama_kv_compact_matrix full_v(4, 2);
        for (uint32_t r = 0; r < 4; ++r) {
            full_v(r, 0) = float(r) * 0.5f;
            full_v(r, 1) = 1.0f - float(r) * 0.25f;
        }

        llama_kv_compact_matrix compacted_v;
        check(llama_kv_compact_fit_values(queries, keys, full_v,
                                           compacted_k, beta_sr, opts_sr,
                                           compacted_v),
              "fit_values with spectral_ridge succeeds");
        check(compacted_v.rows == 2 && compacted_v.cols == 2,
              "spectral_ridge V has correct shape");
    }

    // -----------------------------------------------------------------------
    // 5. Budget allocation (Algorithm 4 / Section 3.4)
    // -----------------------------------------------------------------------
    std::printf("\n=== Per-Head Budget Allocation ===\n");
    {
        // Entropy computation.
        llama_kv_compact_matrix queries, keys;
        make_fixture(queries, keys);

        float entropy = llama_kv_compact_head_entropy(queries, keys);
        check(entropy >= 0.0f, "Entropy is non-negative");
        check(std::isfinite(entropy), "Entropy is finite");
        std::printf("    head entropy = %.4f\n", entropy);

        // Uniform-entropy heads should get equal budgets.
        std::vector<float> uniform_entropies = { 1.0f, 1.0f, 1.0f, 1.0f };
        llama_kv_compact_budget_opts budget_opts;
        budget_opts.total_budget = 100;
        budget_opts.min_per_head = 4;

        auto budgets = llama_kv_compact_allocate_budgets(uniform_entropies, budget_opts);
        check(budgets.size() == 4, "Budget allocation returns correct head count");

        uint32_t total = 0;
        for (uint32_t b : budgets) total += b;
        check(total == 100, "Uniform budgets sum to total_budget");

        for (uint32_t b : budgets) {
            check(b == 25, "Uniform entropy gives equal budgets");
        }

        // Varying entropy: low-entropy heads should get MORE budget.
        std::vector<float> varying_entropies = { 0.5f, 2.0f, 0.5f, 2.0f };
        budget_opts.total_budget = 100;
        auto var_budgets = llama_kv_compact_allocate_budgets(varying_entropies, budget_opts);

        uint32_t var_total = 0;
        for (uint32_t b : var_budgets) var_total += b;
        check(var_total == 100, "Varying budgets sum to total_budget");

        // Low-entropy heads (0, 2) should get more than high-entropy (1, 3).
        check(var_budgets[0] > var_budgets[1],
              "Low-entropy head gets larger budget than high-entropy head");
        check(var_budgets[2] > var_budgets[3],
              "Second low-entropy head also gets larger budget");

        std::printf("    varying budgets: %u, %u, %u, %u\n",
                    var_budgets[0], var_budgets[1], var_budgets[2], var_budgets[3]);

        // Min-per-head enforcement.
        budget_opts.total_budget = 20;
        budget_opts.min_per_head = 5;
        auto min_budgets = llama_kv_compact_allocate_budgets(varying_entropies, budget_opts);
        for (uint32_t b : min_budgets) {
            check(b >= 5, "All budgets respect min_per_head");
        }
    }

    // -----------------------------------------------------------------------
    // 6. Budget allocation edge cases
    // -----------------------------------------------------------------------
    std::printf("\n=== Budget Edge Cases ===\n");
    {
        // Empty input.
        auto empty = llama_kv_compact_allocate_budgets({}, {});
        check(empty.empty(), "Empty entropies returns empty budgets");

        // Single head.
        llama_kv_compact_budget_opts opts;
        opts.total_budget = 50;
        opts.min_per_head = 4;
        auto single = llama_kv_compact_allocate_budgets({1.0f}, opts);
        check(single.size() == 1 && single[0] == 50,
              "Single head gets entire budget");

        // Zero total budget.
        opts.total_budget = 0;
        auto zero = llama_kv_compact_allocate_budgets({1.0f, 2.0f}, opts);
        check(zero.empty(), "Zero total_budget returns empty");

        // Very low entropy (near zero) shouldn't cause NaN/inf.
        std::vector<float> near_zero_ent = { 0.0001f, 0.0001f };
        opts.total_budget = 20;
        opts.min_per_head = 2;
        auto nz = llama_kv_compact_allocate_budgets(near_zero_ent, opts);
        uint32_t nz_total = 0;
        for (uint32_t b : nz) {
            check(b >= 2, "Near-zero entropy respects min_per_head");
            nz_total += b;
        }
        check(nz_total == 20, "Near-zero entropy budgets sum correctly");
    }

    // -----------------------------------------------------------------------
    // 7. Union building with per-head masking
    // -----------------------------------------------------------------------
    std::printf("\n=== Union Building ===\n");
    {
        // Head 0 selects {0, 2, 4}, Head 1 selects {1, 2, 3}.
        std::vector<std::vector<uint32_t>> selections = {
            {0, 2, 4},
            {1, 2, 3},
        };

        std::vector<bool> mask;
        auto union_vec = llama_kv_compact_build_union(selections, 2, mask);

        check(union_vec.size() == 5, "Union has 5 unique positions {0,1,2,3,4}");
        check(union_vec[0] == 0 && union_vec[1] == 1 && union_vec[2] == 2 &&
              union_vec[3] == 3 && union_vec[4] == 4,
              "Union is sorted");

        check(mask.size() == 10, "Mask has n_heads * union_size entries");

        // Head 0 mask: selected {0, 2, 4} -> positions 0, 2, 4 in union.
        check(mask[0*5 + 0] == true,  "Head 0 selected position 0");
        check(mask[0*5 + 1] == false, "Head 0 did not select position 1");
        check(mask[0*5 + 2] == true,  "Head 0 selected position 2");
        check(mask[0*5 + 3] == false, "Head 0 did not select position 3");
        check(mask[0*5 + 4] == true,  "Head 0 selected position 4");

        // Head 1 mask: selected {1, 2, 3} -> positions 1, 2, 3 in union.
        check(mask[1*5 + 0] == false, "Head 1 did not select position 0");
        check(mask[1*5 + 1] == true,  "Head 1 selected position 1");
        check(mask[1*5 + 2] == true,  "Head 1 selected position 2");
        check(mask[1*5 + 3] == true,  "Head 1 selected position 3");
        check(mask[1*5 + 4] == false, "Head 1 did not select position 4");

        // Position 2 is the only one selected by both heads.
        check(mask[0*5 + 2] && mask[1*5 + 2], "Position 2 is in both heads' selections");
    }

    // -----------------------------------------------------------------------
    // 8. Union building edge cases
    // -----------------------------------------------------------------------
    std::printf("\n=== Union Edge Cases ===\n");
    {
        // All heads select the same positions.
        std::vector<std::vector<uint32_t>> same = {{1, 3}, {1, 3}};
        std::vector<bool> mask;
        auto u = llama_kv_compact_build_union(same, 2, mask);
        check(u.size() == 2, "Identical selections produce union of size 2");
        check(mask.size() == 4, "Mask for identical selections");
        for (bool m : mask) {
            check(m == true, "All mask entries true for identical selections");
        }

        // Disjoint selections.
        std::vector<std::vector<uint32_t>> disjoint = {{0, 1}, {2, 3}};
        auto d = llama_kv_compact_build_union(disjoint, 2, mask);
        check(d.size() == 4, "Disjoint selections produce union of size 4");

        // Empty selections.
        std::vector<std::vector<uint32_t>> empty = {{}, {}};
        auto e = llama_kv_compact_build_union(empty, 2, mask);
        check(e.empty(), "Empty selections produce empty union");
    }

    // -----------------------------------------------------------------------
    // 9. Entropy correctness: uniform vs peaked attention
    // -----------------------------------------------------------------------
    std::printf("\n=== Entropy Correctness ===\n");
    {
        // Build a "peaked" head: one key dominates.
        llama_kv_compact_matrix peaked_q(1, 2);
        llama_kv_compact_matrix peaked_k(4, 2);
        peaked_q(0, 0) = 10.0f; peaked_q(0, 1) = 0.0f;
        peaked_k(0, 0) = 10.0f; peaked_k(0, 1) = 0.0f;
        peaked_k(1, 0) = 0.0f;  peaked_k(1, 1) = 0.0f;
        peaked_k(2, 0) = 0.0f;  peaked_k(2, 1) = 0.0f;
        peaked_k(3, 0) = 0.0f;  peaked_k(3, 1) = 0.0f;

        float ent_peaked = llama_kv_compact_head_entropy(peaked_q, peaked_k);

        // Build a "uniform" head: all keys equal.
        llama_kv_compact_matrix uniform_q(1, 2);
        llama_kv_compact_matrix uniform_k(4, 2);
        uniform_q(0, 0) = 1.0f; uniform_q(0, 1) = 0.0f;
        for (uint32_t r = 0; r < 4; ++r) {
            uniform_k(r, 0) = 1.0f; uniform_k(r, 1) = 0.0f;
        }

        float ent_uniform = llama_kv_compact_head_entropy(uniform_q, uniform_k);

        std::printf("    peaked entropy  = %.4f\n", ent_peaked);
        std::printf("    uniform entropy = %.4f\n", ent_uniform);

        check(ent_peaked < ent_uniform,
              "Peaked attention has lower entropy than uniform");

        // Uniform attention entropy should be near ln(4) ≈ 1.386.
        check(std::abs(ent_uniform - std::log(4.0f)) < 0.01f,
              "Uniform attention entropy is approximately ln(n_keys)");
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n", n_passed, n_failed);
    return n_failed > 0 ? 1 : 0;
}
