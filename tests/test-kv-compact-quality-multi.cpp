#include "arg.h"
#include "common.h"
#include "llama.h"
#include "src/llama-context.h"
#include "src/llama-kv-compact-pipeline.h"
#include "src/llama-kv-cache.h"
#include "src/llama-kv-compact-utils.h"
#include "kv-compact-test-helpers.h"
#include "kv-compact-thresholds.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

using llama_kv_compact_test::cosine_similarity;

namespace {

static const std::string test_prefix = "test-kv-compact-quality-multi";

int fail(const std::string & message) {
    return llama_kv_compact_test::fail(test_prefix, message);
}

std::vector<float> decode_one_and_capture_logits(llama_context * ctx, llama_token token, int pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, token, pos, {0}, true);
    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("decode failed");
    }
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * logits = llama_get_logits_ith(ctx, 0);
    std::vector<float> out(logits, logits + n_vocab);
    llama_batch_free(batch);
    return out;
}

// build a reproducible seed prompt
std::vector<llama_token> build_real_text_prompt(llama_context * ctx, size_t min_tokens) {
    const std::string para =
        "The quarterly letter reviewed liquidity, capital allocation, recurring revenue, customer retention, and operating leverage. "
        "Management discussed cash flow discipline, pricing pressure, inventory turns, software adoption, and regional demand. "
        "Analysts compared the margin profile to prior quarters and noted that guidance depended on enterprise renewals, deferred revenue conversion, and foreign exchange stability. ";
    std::string text;
    std::vector<llama_token> tokens;
    while (tokens.size() < min_tokens) {
        text += para;
        tokens = common_tokenize(ctx, text, true, false);
    }
    tokens.resize(min_tokens);
    return tokens;
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
    llama_context * ctx = llama_init->context();
    llama_model * model = llama_init->model();
    if (!ctx || !model) return fail("init failed");

    auto * kv = llama_kv_compact_get_cache(ctx->get_memory());
    if (!kv) return fail("requires KV cache backend");

    constexpr int seed_tokens = 256;
    constexpr int n_continuation = 8;
    constexpr llama_token continuation_token = 1;

    auto prompt = build_real_text_prompt(ctx, seed_tokens);

    // Phase 1: decode reference (no compaction) — collect N continuation logit vectors
    {
        llama_batch batch = llama_batch_init(seed_tokens, 0, 1);
        for (int i = 0; i < seed_tokens; ++i) {
            common_batch_add(batch, prompt[i], i, {0}, i == seed_tokens - 1);
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return fail("reference decode failed");
        }
        llama_batch_free(batch);
    }

    std::vector<std::vector<float>> ref_logits;
    for (int i = 0; i < n_continuation; ++i) {
        ref_logits.push_back(decode_one_and_capture_logits(ctx, continuation_token, seed_tokens + i));
    }

    // Phase 2: reset, refill, compact at 4x, decode N continuation tokens
    llama_memory_clear(ctx->get_memory(), true);

    {
        llama_batch batch = llama_batch_init(seed_tokens, 0, 1);
        for (int i = 0; i < seed_tokens; ++i) {
            common_batch_add(batch, prompt[i], i, {0}, i == seed_tokens - 1);
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return fail("post-compact decode failed");
        }
        llama_batch_free(batch);
    }

    // Compact at 4x using the public API
    struct llama_compact_params cp = llama_compact_default_params();
    cp.method = LLAMA_COMPACT_METHOD_SELECT;
    cp.ratio = 4.0f;
    int32_t rc = llama_kv_cache_compact(ctx, 0, cp);
    if (rc < 0) return fail("compaction failed");

    std::printf("  Compacted %d -> %d tokens (4x)\n", seed_tokens, rc);

    std::vector<std::vector<float>> comp_logits;
    for (int i = 0; i < n_continuation; ++i) {
        comp_logits.push_back(decode_one_and_capture_logits(ctx, continuation_token, seed_tokens + i));
    }

    // Phase 3: compare each continuation token's logits
    constexpr float threshold = llama_kv_compact_thresholds::COS_MULTI_4X;
    int n_pass = 0;
    for (int i = 0; i < n_continuation; ++i) {
        float cs = cosine_similarity(ref_logits[i], comp_logits[i]);
        std::printf("  token %d: cosine = %.6f (threshold %.3f) %s\n",
                    i, cs, threshold, cs >= threshold ? "PASS" : "FAIL");
        if (cs >= threshold) n_pass++;
    }

    std::printf("Multi-token quality: %d/%d passed\n", n_pass, n_continuation);
    return (n_pass == n_continuation) ? 0 : 1;
}
