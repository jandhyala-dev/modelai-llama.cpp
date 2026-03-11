#include "arg.h"
#include "common.h"
#include "llama.h"
#include "src/llama-context.h"
#include "src/llama-kv-cache.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int fail(const std::string & message) {
    std::fprintf(stderr, "test-kv-compacted-prefix-pack: %s\n", message.c_str());
    return 1;
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
    std::vector<llama_token> tokens(seed_tokens, 1);
    llama_batch batch = llama_batch_init(seed_tokens, 0, 1);
    for (int i = 0; i < seed_tokens; ++i) {
        common_batch_add(batch, tokens[i], i, {0}, i + 1 == seed_tokens);
    }

    if (llama_decode(ctx, batch)) {
        llama_batch_free(batch);
        return fail("failed to decode seed prompt");
    }

    const uint32_t n_kv_before = kv->compacted_prefix_active_n_kv(0);
    if (n_kv_before <= 256) {
        llama_batch_free(batch);
        return fail("expected active n_kv before reclaim to exceed the 256 floor");
    }

    if (!kv->compacted_prefix_configure(0, seed_tokens, { 0, 64, 128, 192 }, 256)) {
        llama_batch_free(batch);
        return fail("failed to configure compacted prefix");
    }

    if (!kv->compacted_prefix_set_execution(0, true) || !kv->compacted_prefix_execution_enabled(0)) {
        llama_batch_free(batch);
        return fail("failed to enable compacted-prefix execution");
    }

    if (!kv->compacted_prefix_reclaim_live_kv(0)) {
        llama_batch_free(batch);
        return fail("failed to reclaim live KV cells after compacting prefix");
    }

    const uint32_t n_kv_after = kv->compacted_prefix_active_n_kv(0);
    if (n_kv_after >= n_kv_before) {
        llama_batch_free(batch);
        return fail("expected active n_kv to shrink after live-KV reclamation");
    }
    if (n_kv_after != 256) {
        llama_batch_free(batch);
        return fail("expected active n_kv to fall to the padded 256 floor");
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (llama_memory_seq_pos_min(mem, 0) != 256 || llama_memory_seq_pos_max(mem, 0) != seed_tokens - 1) {
        llama_batch_free(batch);
        return fail("live KV position range was not compacted to the retained suffix");
    }

    if (!kv->compacted_prefix_reclaim_live_kv(0)) {
        llama_batch_free(batch);
        return fail("second reclamation pass should be a no-op success");
    }
    if (kv->compacted_prefix_active_n_kv(0) != n_kv_after) {
        llama_batch_free(batch);
        return fail("active n_kv changed on idempotent reclaim");
    }

    common_batch_clear(batch);
    common_batch_add(batch, 1, seed_tokens, {0}, true);
    if (llama_decode(ctx, batch)) {
        llama_batch_free(batch);
        return fail("continuation decode failed after live-KV reclamation");
    }

    if (kv->compacted_prefix_active_n_kv(0) != 256) {
        llama_batch_free(batch);
        return fail("continuation decode should keep the retained suffix within the same padded n_kv bucket");
    }

    llama_batch_free(batch);
    return 0;
}
