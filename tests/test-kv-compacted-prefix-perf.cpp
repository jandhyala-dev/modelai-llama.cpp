#include "arg.h"
#include "common.h"
#include "llama.h"
#include "src/llama-context.h"
#include "src/llama-kv-compact-pipeline.h"
#include "src/llama-kv-cache.h"
#include "src/llama-kv-compact-utils.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int fail(const std::string & message) {
    std::fprintf(stderr, "test-kv-compacted-prefix-perf: %s\n", message.c_str());
    return 1;
}

bool decode_repeated_token(llama_context * ctx, llama_token token, int start_pos, int n_decode) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    for (int i = 0; i < n_decode; ++i) {
        common_batch_clear(batch);
        common_batch_add(batch, token, start_pos + i, {0}, true);
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return false;
        }
    }
    llama_batch_free(batch);
    return true;
}

struct run_stats {
    double mean;
    double stddev;
};

run_stats compute_stats(const std::vector<double> & samples) {
    if (samples.empty()) {
        return {0.0, 0.0};
    }
    double sum = 0.0;
    for (double v : samples) {
        sum += v;
    }
    double m = sum / samples.size();
    double var_sum = 0.0;
    for (double v : samples) {
        var_sum += (v - m) * (v - m);
    }
    double sd = samples.size() > 1 ? std::sqrt(var_sum / (samples.size() - 1)) : 0.0;
    return {m, sd};
}

} // namespace

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 1234;
    params.kv_unified = true;
    params.n_parallel = 1;
    params.n_ctx = 512;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (std::getenv("USE_FLASH")) {
        std::printf("USE_FLASH=1: enabling flash attention\n");
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    } else {
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    common_init();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        return fail("failed to initialize model/context");
    }

    GGML_UNUSED(model);

    auto * kv = llama_kv_compact_get_cache(ctx->get_memory());
    if (kv == nullptr) {
        return fail("test requires a llama_kv_cache or llama_kv_cache_iswa memory backend");
    }

    constexpr int seed_tokens = 320;
    constexpr int decode_tokens = 32;
    constexpr int n_iterations = 3;

    std::vector<llama_token> tokens(seed_tokens, 1);
    llama_batch batch = llama_batch_init(seed_tokens, 0, 1);
    for (int i = 0; i < seed_tokens; ++i) {
        common_batch_add(batch, tokens[i], i, {0}, i + 1 == seed_tokens);
    }

    if (llama_decode(ctx, batch)) {
        llama_batch_free(batch);
        return fail("failed to decode seed prompt");
    }

    llama_kv_compact_pipeline_stats stats = {};
    if (!kv->compacted_prefix_select_from_live_kv(0, 64, 256, &stats)) {
        llama_batch_free(batch);
        return fail("failed to select compacted prefix from live KV");
    }
    if (!kv->compacted_prefix_set_execution(0, true)) {
        llama_batch_free(batch);
        return fail("failed to enable compacted-prefix execution");
    }
    if (stats.n_selected_tokens != 64 || stats.n_prefix_tokens != 256) {
        llama_batch_free(batch);
        return fail("unexpected compacted-prefix pipeline stats");
    }

    std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 0));
    const size_t ncopy = llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 0);
    if (ncopy != seq_state.size()) {
        llama_batch_free(batch);
        return fail("failed to save sequence state");
    }

    const uint32_t baseline_n_kv = kv->compacted_prefix_active_n_kv(0);

    // Baseline: multiple decode iterations before reclaim
    std::vector<double> baseline_tps_samples;
    for (int iter = 0; iter < n_iterations; ++iter) {
        if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
            llama_batch_free(batch);
            return fail("failed to restore state for baseline iteration");
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (!decode_repeated_token(ctx, 1, seed_tokens, decode_tokens)) {
            llama_batch_free(batch);
            return fail("baseline decode failed");
        }
        const auto t1 = std::chrono::steady_clock::now();
        baseline_tps_samples.push_back(decode_tokens / std::chrono::duration<double>(t1 - t0).count());
    }

    // Restore and reclaim for the compacted path
    if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
        llama_batch_free(batch);
        return fail("failed to restore sequence state before reclaim benchmark");
    }

    if (!kv->compacted_prefix_reclaim_live_kv(0)) {
        llama_batch_free(batch);
        return fail("failed to reclaim live KV cells");
    }

    const uint32_t reclaimed_n_kv = kv->compacted_prefix_active_n_kv(0);

    // Save reclaimed state for repeated iterations
    std::vector<uint8_t> reclaimed_state(llama_state_seq_get_size(ctx, 0));
    const size_t rcopy = llama_state_seq_get_data(ctx, reclaimed_state.data(), reclaimed_state.size(), 0);
    if (rcopy != reclaimed_state.size()) {
        llama_batch_free(batch);
        return fail("failed to save reclaimed state");
    }

    // Reclaimed: multiple decode iterations after reclaim
    std::vector<double> reclaimed_tps_samples;
    for (int iter = 0; iter < n_iterations; ++iter) {
        if (llama_state_seq_set_data(ctx, reclaimed_state.data(), reclaimed_state.size(), 0) != reclaimed_state.size()) {
            llama_batch_free(batch);
            return fail("failed to restore state for reclaimed iteration");
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (!decode_repeated_token(ctx, 1, seed_tokens, decode_tokens)) {
            llama_batch_free(batch);
            return fail("reclaimed decode failed");
        }
        const auto t1 = std::chrono::steady_clock::now();
        reclaimed_tps_samples.push_back(decode_tokens / std::chrono::duration<double>(t1 - t0).count());
    }

    llama_batch_free(batch);

    const auto baseline = compute_stats(baseline_tps_samples);
    const auto reclaimed = compute_stats(reclaimed_tps_samples);

    std::printf("baseline_active_n_kv=%u\n", baseline_n_kv);
    std::printf("reclaimed_active_n_kv=%u\n", reclaimed_n_kv);
    std::printf("iterations=%d\n", n_iterations);
    std::printf("decode_tokens=%d\n", decode_tokens);
    std::printf("baseline_tok_per_sec=%.1f +/- %.1f\n", baseline.mean, baseline.stddev);
    std::printf("reclaimed_tok_per_sec=%.1f +/- %.1f\n", reclaimed.mean, reclaimed.stddev);

    return 0;
}
