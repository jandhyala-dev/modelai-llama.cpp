#include "arg.h"
#include "common.h"
#include "llama.h"
#include "src/llama-context.h"
#include "src/llama-kv-compact-pipeline.h"
#include "src/llama-kv-compact-solver.h"
#include "src/llama-kv-cache.h"

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
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    common_init();
    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_context * ctx = llama_init->context();
    llama_model * model = llama_init->model();
    if (ctx == nullptr || model == nullptr) {
        return fail("failed to initialize model/context");
    }

    auto * kv = dynamic_cast<llama_kv_cache *>(ctx->get_memory());
    if (kv == nullptr) {
        return fail("test requires a standard llama_kv_cache memory backend");
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

    const bool bypass_solver = std::getenv("BYPASS_SOLVER") != nullptr;
    const bool raw_v_mode    = std::getenv("RAW_V") != nullptr;
    const bool zero_beta_mode = std::getenv("ZERO_BETA") != nullptr;

    llama_kv_compact_pipeline_stats stats = {};

    if (bypass_solver) {
        // Diagnostic mode: populate compacted prefix with original K/V and zero beta.
        // This isolates the execution path from the solver.
        std::printf("BYPASS_SOLVER=1: using raw K/V with zero beta\n");

        std::vector<llama_pos> prefix_positions;
        if (!kv->compacted_prefix_seq_positions(0, 0, live_suffix_pos0, prefix_positions)) {
            llama_batch_free(batch);
            return fail("bypass: failed to get prefix positions");
        }

        // For bypass, select all positions (or first compacted_tokens)
        if ((int)prefix_positions.size() > compacted_tokens) {
            prefix_positions.resize(compacted_tokens);
        }

        const llama_pos seq_max = kv->seq_pos_max(0);
        const uint32_t logical_count = seq_max >= 0 ? uint32_t(seq_max + 1) : uint32_t(live_suffix_pos0);
        if (!kv->compacted_prefix_configure(0, logical_count, prefix_positions, live_suffix_pos0)) {
            llama_batch_free(batch);
            return fail("bypass: failed to configure compacted prefix");
        }

        auto * seq = kv->get_compacted_prefix()->get_seq(0);
        if (seq == nullptr || !seq->enabled) {
            llama_batch_free(batch);
            return fail("bypass: sequence not configured");
        }

        const auto & layouts = kv->get_compacted_prefix()->get_layouts();
        const uint32_t n_sel = prefix_positions.size();

        for (size_t li = 0; li < layouts.size(); ++li) {
            const auto & layout = layouts[li];
            auto & dst_layer = seq->layers[li];

            for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
                // Copy K directly from cache
                std::vector<float> k_f32;
                if (!kv->compacted_prefix_copy_k_head_f32(int32_t(layout.layer_id), 0, head, prefix_positions, k_f32)) {
                    llama_batch_free(batch);
                    return fail("bypass: failed to copy K");
                }

                // Write K to store
                auto from_float_k = ggml_get_type_traits(layout.type_k)->from_float_ref;
                const size_t k_token_bytes = ggml_row_size(layout.type_k, layout.n_embd_head_k);
                for (uint32_t t = 0; t < n_sel; ++t) {
                    void * dst_ptr = dst_layer.k_data.data() + (size_t(head) * n_sel + t) * k_token_bytes;
                    from_float_k(k_f32.data() + size_t(t) * layout.n_embd_head_k, dst_ptr, layout.n_embd_head_k);
                }

                // Copy V directly from cache
                std::vector<float> v_f32;
                if (!kv->compacted_prefix_copy_v_head_f32(int32_t(layout.layer_id), 0, head, prefix_positions, v_f32)) {
                    llama_batch_free(batch);
                    return fail("bypass: failed to copy V");
                }

                // Write V to store
                auto from_float_v = ggml_get_type_traits(layout.type_v)->from_float_ref;
                const size_t v_token_bytes = ggml_row_size(layout.type_v, layout.n_embd_head_v);
                for (uint32_t t = 0; t < n_sel; ++t) {
                    void * dst_ptr = dst_layer.v_data.data() + (size_t(head) * n_sel + t) * v_token_bytes;
                    from_float_v(v_f32.data() + size_t(t) * layout.n_embd_head_v, dst_ptr, layout.n_embd_head_v);
                }

                // Set beta to 0
                for (uint32_t t = 0; t < n_sel; ++t) {
                    dst_layer.beta_data[size_t(head) * n_sel + t] = 0.0f;
                }
            }
        }

        stats.n_prefix_tokens = n_sel;
        stats.n_selected_tokens = n_sel;
        stats.query_generation_time_ms = 0.0;
        stats.solver_time_ms = 0.0;
    } else {
        if (!kv->compacted_prefix_fit_from_live_kv(0, compacted_tokens, live_suffix_pos0, &stats)) {
            llama_batch_free(batch);
            return fail("failed to fit compacted prefix from live KV");
        }

        // Post-solver diagnostic overrides
        if (raw_v_mode || zero_beta_mode) {
            auto * seq = kv->get_compacted_prefix()->get_seq(0);
            if (seq == nullptr || !seq->enabled) {
                llama_batch_free(batch);
                return fail("diagnostic: sequence not configured after solver");
            }

            const auto & layouts = kv->get_compacted_prefix()->get_layouts();

            if (zero_beta_mode) {
                std::printf("ZERO_BETA=1: zeroing all beta values (keeping solver K/V)\n");
                for (size_t li = 0; li < layouts.size(); ++li) {
                    auto & dst_layer = seq->layers[li];
                    std::fill(dst_layer.beta_data.begin(), dst_layer.beta_data.end(), 0.0f);
                }
            }

            if (raw_v_mode) {
                std::printf("RAW_V=1: overwriting solver V with original cache V (keeping solver beta)\n");
                for (size_t li = 0; li < layouts.size(); ++li) {
                    const auto & layout = layouts[li];
                    auto & dst_layer = seq->layers[li];
                    const uint32_t n_sel = dst_layer.n_compacted_tokens;

                    for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
                        std::vector<float> v_f32;
                        if (!kv->compacted_prefix_copy_v_head_f32(
                                    int32_t(layout.layer_id), 0, head,
                                    seq->logical_positions, v_f32)) {
                            llama_batch_free(batch);
                            return fail("RAW_V: failed to copy V from cache");
                        }

                        auto from_float_v = ggml_get_type_traits(layout.type_v)->from_float_ref;
                        const size_t v_token_bytes = ggml_row_size(layout.type_v, layout.n_embd_head_v);
                        for (uint32_t t = 0; t < n_sel; ++t) {
                            void * dst_ptr = dst_layer.v_data.data() + (size_t(head) * n_sel + t) * v_token_bytes;
                            from_float_v(v_f32.data() + size_t(t) * layout.n_embd_head_v, dst_ptr, layout.n_embd_head_v);
                        }
                    }
                }
            }
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

    if (!bypass_solver) {
        if (stats.n_prefix_tokens != (uint32_t)live_suffix_pos0 || stats.n_selected_tokens != (uint32_t)compacted_tokens) {
            llama_batch_free(batch);
            return fail("unexpected pipeline stats after compacted fit");
        }
        if (stats.query_generation_time_ms <= 0.0 || stats.solver_time_ms <= 0.0) {
            llama_batch_free(batch);
            return fail("pipeline timings should be populated");
        }
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
        return fail("continuation-logit cosine should meet the baseline threshold");
    }

    llama_batch_free(batch);
    return 0;
}
