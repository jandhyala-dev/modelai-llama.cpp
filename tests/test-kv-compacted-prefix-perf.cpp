#include "arg.h"
#include "common.h"
#include "llama.h"
#include "src/llama-context.h"
#include "src/llama-kv-cache.h"

#include <chrono>
#include <cstdio>
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

    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    common_init();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        return fail("failed to initialize model/context");
    }

    GGML_UNUSED(model);

    auto * kv = dynamic_cast<llama_kv_cache *>(ctx->get_memory());
    if (kv == nullptr) {
        return fail("test requires a standard llama_kv_cache memory backend");
    }

    constexpr int seed_tokens = 320;
    constexpr int decode_tokens = 32;

    std::vector<llama_token> tokens(seed_tokens, 1);
    llama_batch batch = llama_batch_init(seed_tokens, 0, 1);
    for (int i = 0; i < seed_tokens; ++i) {
        common_batch_add(batch, tokens[i], i, {0}, i + 1 == seed_tokens);
    }

    if (llama_decode(ctx, batch)) {
        llama_batch_free(batch);
        return fail("failed to decode seed prompt");
    }

    if (!kv->compacted_prefix_configure(0, seed_tokens, { 0, 64, 128, 192 }, 256)) {
        llama_batch_free(batch);
        return fail("failed to configure compacted prefix");
    }
    if (!kv->compacted_prefix_set_execution(0, true)) {
        llama_batch_free(batch);
        return fail("failed to enable compacted-prefix execution");
    }

    std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 0));
    const size_t ncopy = llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 0);
    if (ncopy != seq_state.size()) {
        llama_batch_free(batch);
        return fail("failed to save sequence state");
    }

    const uint32_t baseline_n_kv = kv->compacted_prefix_active_n_kv(0);
    const auto t0 = std::chrono::steady_clock::now();
    if (!decode_repeated_token(ctx, 1, seed_tokens, decode_tokens)) {
        llama_batch_free(batch);
        return fail("baseline decode failed");
    }
    const auto t1 = std::chrono::steady_clock::now();

    if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
        llama_batch_free(batch);
        return fail("failed to restore sequence state before reclaim benchmark");
    }

    if (!kv->compacted_prefix_reclaim_live_kv(0)) {
        llama_batch_free(batch);
        return fail("failed to reclaim live KV cells");
    }

    const uint32_t reclaimed_n_kv = kv->compacted_prefix_active_n_kv(0);
    const auto t2 = std::chrono::steady_clock::now();
    if (!decode_repeated_token(ctx, 1, seed_tokens, decode_tokens)) {
        llama_batch_free(batch);
        return fail("reclaimed decode failed");
    }
    const auto t3 = std::chrono::steady_clock::now();

    llama_batch_free(batch);

    const double baseline_sec = std::chrono::duration<double>(t1 - t0).count();
    const double reclaimed_sec = std::chrono::duration<double>(t3 - t2).count();
    const double baseline_tps = decode_tokens / baseline_sec;
    const double reclaimed_tps = decode_tokens / reclaimed_sec;

    std::printf("baseline_active_n_kv=%u\n", baseline_n_kv);
    std::printf("reclaimed_active_n_kv=%u\n", reclaimed_n_kv);
    std::printf("baseline_decode_tokens_per_second=%.4f\n", baseline_tps);
    std::printf("reclaimed_decode_tokens_per_second=%.4f\n", reclaimed_tps);
    std::printf("decode_tokens=%d\n", decode_tokens);

    return 0;
}
