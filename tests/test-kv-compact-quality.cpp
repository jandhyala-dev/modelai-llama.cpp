#include "arg.h"
#include "common.h"
#include "llama.h"
#include "src/llama-context.h"
#include "src/llama-kv-compact-pipeline.h"
#include "src/llama-kv-compact-solver.h"
#include "src/llama-kv-cache.h"
#include "src/llama-kv-cache-iswa.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int fail(const std::string & message) {
    std::fprintf(stderr, "test-kv-compact-quality: %s\n", message.c_str());
    return 1;
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
    if (std::getenv("USE_FLASH")) {
        std::printf("USE_FLASH=1: enabling flash attention\n");
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    } else {
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    common_init();
    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_context * ctx = llama_init->context();
    llama_model * model = llama_init->model();
    if (ctx == nullptr || model == nullptr) {
        return fail("failed to initialize model/context");
    }

    auto * kv = dynamic_cast<llama_kv_cache *>(ctx->get_memory());
    llama_kv_cache_iswa * kv_iswa = nullptr;
    if (kv == nullptr) {
        kv_iswa = dynamic_cast<llama_kv_cache_iswa *>(ctx->get_memory());
        if (kv_iswa != nullptr) {
            kv = kv_iswa->get_base();
        }
    }
    if (kv == nullptr) {
        return fail("test requires a llama_kv_cache or llama_kv_cache_iswa memory backend");
    }

    constexpr int seed_tokens = 320;
    constexpr int live_suffix_pos0 = 256;
    constexpr llama_token continuation = 1;

    // Allow overriding compacted token count from env for diagnostics
    int compacted_tokens = 128;
    const char * env_ct = std::getenv("COMPACT_TOKENS");
    if (env_ct) {
        compacted_tokens = std::atoi(env_ct);
        if (compacted_tokens <= 0 || compacted_tokens > live_suffix_pos0) {
            compacted_tokens = 128;
        }
    }

    const std::vector<llama_token> prompt = build_real_text_prompt(ctx, seed_tokens);

    llama_batch batch = llama_batch_init(prompt.size(), 0, 1);
    for (int i = 0; i < seed_tokens; ++i) {
        common_batch_add(batch, prompt[i], i, {0}, i + 1 == seed_tokens);
    }
    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        return fail("failed to decode real-text seed prompt");
    }

    std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 0));
    const size_t ncopy = llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 0);
    if (ncopy != seq_state.size()) {
        llama_batch_free(batch);
        return fail("failed to save baseline state");
    }

    if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
        llama_batch_free(batch);
        return fail("failed to restore baseline state");
    }
    const std::vector<float> baseline_logits = decode_one_and_capture_logits(ctx, continuation, seed_tokens);

    if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
        llama_batch_free(batch);
        return fail("failed to restore compacted state");
    }

    const bool use_solver     = std::getenv("USE_SOLVER")     != nullptr;
    const bool use_omp        = std::getenv("USE_OMP")        != nullptr;
    const bool use_nonuniform = std::getenv("USE_NONUNIFORM") != nullptr;
    const bool use_chunked    = std::getenv("USE_CHUNKED")    != nullptr;
    const bool use_on_policy  = std::getenv("USE_ON_POLICY")  != nullptr;

    llama_kv_compact_pipeline_stats stats = {};

    if (use_on_policy) {
        std::printf("USE_ON_POLICY=1: two-pass on-policy pipeline\n");
        if (!kv->compacted_prefix_on_policy_from_live_kv(ctx, 0, compacted_tokens, live_suffix_pos0, &stats)) {
            llama_batch_free(batch);
            return fail("failed to run on-policy compacted prefix from live KV");
        }
    } else if (use_nonuniform) {
        std::printf("USE_NONUNIFORM=1: nonuniform per-head budget pipeline\n");
        if (!kv->compacted_prefix_nonuniform_from_live_kv(0, compacted_tokens, live_suffix_pos0, &stats)) {
            llama_batch_free(batch);
            return fail("failed to run nonuniform compacted prefix from live KV");
        }
    } else if (use_chunked) {
        std::printf("USE_CHUNKED=1: chunked compaction pipeline\n");
        if (!kv->compacted_prefix_chunked_from_live_kv(0, compacted_tokens, live_suffix_pos0, &stats)) {
            llama_batch_free(batch);
            return fail("failed to run chunked compacted prefix from live KV");
        }
    } else if (use_omp) {
        std::printf("USE_OMP=1: OMP selection + solver pipeline\n");
        if (!kv->compacted_prefix_omp_from_live_kv(0, compacted_tokens, live_suffix_pos0, &stats)) {
            llama_batch_free(batch);
            return fail("failed to run OMP compacted prefix from live KV");
        }
    } else if (use_solver) {
        std::printf("USE_SOLVER=1: full solver pipeline (beta + V fitting)\n");
        if (!kv->compacted_prefix_fit_from_live_kv(0, compacted_tokens, live_suffix_pos0, &stats)) {
            llama_batch_free(batch);
            return fail("failed to fit compacted prefix from live KV");
        }
    } else {
        // Default: selection-only pipeline (raw K/V, zero beta).
        // More robust on GQA architectures than cache-key-query solver.
        if (!kv->compacted_prefix_select_from_live_kv(0, compacted_tokens, live_suffix_pos0, &stats)) {
            llama_batch_free(batch);
            return fail("failed to select compacted prefix from live KV");
        }
    }

    if (!kv->compacted_prefix_set_execution(0, true)) {
        llama_batch_free(batch);
        return fail("failed to enable compacted-prefix execution");
    }
    if (!kv->compacted_prefix_reclaim_live_kv(0)) {
        llama_batch_free(batch);
        return fail("failed to reclaim live KV after fitting");
    }

    const std::vector<float> compacted_logits = decode_one_and_capture_logits(ctx, continuation, seed_tokens);
    const float logits_cos = llama_kv_compact_cosine_similarity(baseline_logits, compacted_logits);

    if (stats.n_prefix_tokens != (uint32_t)live_suffix_pos0) {
        llama_batch_free(batch);
        return fail("unexpected n_prefix_tokens in pipeline stats");
    }
    // Nonuniform may produce fewer selected tokens (union < target); all others should match exactly.
    if (!use_nonuniform && stats.n_selected_tokens != (uint32_t)compacted_tokens) {
        llama_batch_free(batch);
        return fail("unexpected n_selected_tokens in pipeline stats");
    }
    if (use_nonuniform && stats.n_selected_tokens > (uint32_t)compacted_tokens) {
        llama_batch_free(batch);
        return fail("nonuniform n_selected_tokens should not exceed target");
    }
    if ((use_solver || use_nonuniform || use_chunked || use_on_policy) &&
            (stats.query_generation_time_ms <= 0.0 || stats.solver_time_ms <= 0.0)) {
        llama_batch_free(batch);
        return fail("solver pipeline timings should be populated");
    }
    if (kv->compacted_prefix_active_n_kv(0) > 256) {
        llama_batch_free(batch);
        return fail("reclaimed active_n_kv should stay within the retained suffix bucket");
    }
    std::printf("logit_cosine_similarity=%.6f\n", logits_cos);
    std::printf("solver_time_ms=%.1f\n", stats.solver_time_ms);
    std::printf("query_generation_time_ms=%.1f\n", stats.query_generation_time_ms);
    std::printf("n_prefix_tokens=%u\n", stats.n_prefix_tokens);
    std::printf("n_selected_tokens=%u\n", stats.n_selected_tokens);

    if (logits_cos < 0.95f) {
        llama_batch_free(batch);
        return fail("2x continuation-logit cosine should meet the 0.95 threshold");
    }

    // ---- 4x compression test ----
    {
        const int target_4x = live_suffix_pos0 / 4;  // 64 tokens
        std::printf("\n  Testing 4x compression (%d -> %d tokens)...\n",
                    live_suffix_pos0, target_4x);

        if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
            llama_batch_free(batch);
            return fail("failed to restore state for 4x test");
        }

        llama_kv_compact_pipeline_stats stats_4x = {};
        bool ok_4x = false;
        if (use_on_policy) {
            ok_4x = kv->compacted_prefix_on_policy_from_live_kv(ctx, 0, target_4x, live_suffix_pos0, &stats_4x);
        } else if (use_nonuniform) {
            ok_4x = kv->compacted_prefix_nonuniform_from_live_kv(0, target_4x, live_suffix_pos0, &stats_4x);
        } else if (use_chunked) {
            ok_4x = kv->compacted_prefix_chunked_from_live_kv(0, target_4x, live_suffix_pos0, &stats_4x);
        } else if (use_omp) {
            ok_4x = kv->compacted_prefix_omp_from_live_kv(0, target_4x, live_suffix_pos0, &stats_4x);
        } else if (use_solver) {
            ok_4x = kv->compacted_prefix_fit_from_live_kv(0, target_4x, live_suffix_pos0, &stats_4x);
        } else {
            ok_4x = kv->compacted_prefix_select_from_live_kv(0, target_4x, live_suffix_pos0, &stats_4x);
        }
        if (!ok_4x) {
            llama_batch_free(batch);
            return fail("failed to compact at 4x");
        }
        if (!kv->compacted_prefix_set_execution(0, true)) {
            llama_batch_free(batch);
            return fail("failed to enable compacted-prefix execution for 4x");
        }
        if (!kv->compacted_prefix_reclaim_live_kv(0)) {
            llama_batch_free(batch);
            return fail("failed to reclaim live KV after 4x fit");
        }

        const std::vector<float> logits_4x = decode_one_and_capture_logits(ctx, continuation, seed_tokens);
        const float cos_4x = llama_kv_compact_cosine_similarity(baseline_logits, logits_4x);
        std::printf("  4x logit_cosine_similarity=%.6f (threshold >= 0.90)\n", cos_4x);

        if (cos_4x < 0.90f) {
            llama_batch_free(batch);
            return fail("4x continuation-logit cosine should meet the 0.90 threshold");
        }
    }

    // ---- 8x compression test ----
    {
        const int target_8x = live_suffix_pos0 / 8;  // 32 tokens
        std::printf("\n  Testing 8x compression (%d -> %d tokens)...\n",
                    live_suffix_pos0, target_8x);

        if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
            llama_batch_free(batch);
            return fail("failed to restore state for 8x test");
        }

        llama_kv_compact_pipeline_stats stats_8x = {};
        bool ok_8x = false;
        if (use_on_policy) {
            ok_8x = kv->compacted_prefix_on_policy_from_live_kv(ctx, 0, target_8x, live_suffix_pos0, &stats_8x);
        } else if (use_nonuniform) {
            ok_8x = kv->compacted_prefix_nonuniform_from_live_kv(0, target_8x, live_suffix_pos0, &stats_8x);
        } else if (use_chunked) {
            ok_8x = kv->compacted_prefix_chunked_from_live_kv(0, target_8x, live_suffix_pos0, &stats_8x);
        } else if (use_omp) {
            ok_8x = kv->compacted_prefix_omp_from_live_kv(0, target_8x, live_suffix_pos0, &stats_8x);
        } else if (use_solver) {
            ok_8x = kv->compacted_prefix_fit_from_live_kv(0, target_8x, live_suffix_pos0, &stats_8x);
        } else {
            ok_8x = kv->compacted_prefix_select_from_live_kv(0, target_8x, live_suffix_pos0, &stats_8x);
        }
        if (!ok_8x) {
            llama_batch_free(batch);
            return fail("failed to compact at 8x");
        }
        if (!kv->compacted_prefix_set_execution(0, true)) {
            llama_batch_free(batch);
            return fail("failed to enable compacted-prefix execution for 8x");
        }
        if (!kv->compacted_prefix_reclaim_live_kv(0)) {
            llama_batch_free(batch);
            return fail("failed to reclaim live KV after 8x fit");
        }

        const std::vector<float> logits_8x = decode_one_and_capture_logits(ctx, continuation, seed_tokens);
        const float cos_8x = llama_kv_compact_cosine_similarity(baseline_logits, logits_8x);
        std::printf("  8x logit_cosine_similarity=%.6f (threshold >= 0.85)\n", cos_8x);

        if (cos_8x < 0.85f) {
            llama_batch_free(batch);
            return fail("8x continuation-logit cosine should meet the 0.85 threshold");
        }
    }

    llama_batch_free(batch);
    return 0;
}
