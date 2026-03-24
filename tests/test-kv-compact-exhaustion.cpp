// KV cache exhaustion stress test for compaction.
//
// Verifies that compaction can recover from a fully saturated KV cache:
//   1. Fill KV cache to 100% capacity
//   2. Trigger compaction to reclaim prefix tokens
//   3. Verify generation continues after reclaim
//   4. Verify reclaim path frees tokens correctly
//
// Usage:
//   ./test-kv-compact-exhaustion -m /path/to/stories15M-q4_0.gguf

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "src/llama-context.h"
#include "src/llama-kv-compact-pipeline.h"
#include "src/llama-kv-compact-solver.h"
#include "src/llama-kv-cache.h"
#include "src/llama-kv-compact-utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int n_passed  = 0;
int n_failed  = 0;

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

int fail(const std::string & message) {
    std::fprintf(stderr, "test-kv-compact-exhaustion: %s\n", message.c_str());
    return 1;
}

std::vector<llama_token> build_fill_prompt(llama_context * ctx, size_t n_tokens) {
    const std::string para =
        "The quarterly letter reviewed liquidity, capital allocation, recurring revenue, customer retention, and operating leverage. "
        "Management discussed cash flow discipline, pricing pressure, inventory turns, software adoption, and regional demand. "
        "Analysts compared the margin profile to prior quarters and noted that guidance depended on enterprise renewals, deferred revenue conversion, and foreign exchange stability. ";

    std::string text;
    std::vector<llama_token> tokens;
    while (tokens.size() < n_tokens) {
        text += para;
        tokens = common_tokenize(ctx, text, true, false);
    }
    tokens.resize(n_tokens);
    return tokens;
}

} // namespace

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.seed = 1234;
    params.kv_unified = true;
    params.n_parallel = 1;
    // Use a small context so we can fill it completely without excessive runtime.
    params.n_ctx = 512;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();
    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_context * ctx = llama_init->context();
    llama_model * model = llama_init->model();
    if (ctx == nullptr || model == nullptr) {
        return fail("failed to initialize model/context");
    }

    auto * kv = llama_kv_compact_get_cache(ctx->get_memory());
    if (kv == nullptr) {
        return fail("test requires a llama_kv_cache or llama_kv_cache_iswa memory backend");
    }

    if (!kv->supports_compaction()) {
        std::printf("SKIP: model does not support compaction\n");
        return 0;
    }

    const uint32_t n_ctx = llama_n_ctx(ctx);
    // Fill to capacity minus a small margin for the continuation token.
    // We need at least 1 free slot for continuation decode after compaction.
    const uint32_t fill_tokens = n_ctx - 1;

    std::printf("\n=== KV Cache Exhaustion Stress Test ===\n");
    std::printf("n_ctx=%u, fill_tokens=%u\n\n", n_ctx, fill_tokens);

    // =========================================================================
    // TEST 1: Fill KV cache to near-100% capacity
    // =========================================================================
    std::printf("=== TEST 1: Fill KV Cache to Near-100%% Capacity ===\n");
    {
        const std::vector<llama_token> prompt = build_fill_prompt(ctx, fill_tokens);

        // Decode in batches to fill the KV cache.
        const int n_batch = (int) llama_n_batch(ctx);
        for (uint32_t i = 0; i < fill_tokens; ) {
            const uint32_t chunk_end = std::min(i + (uint32_t) n_batch, fill_tokens);
            llama_batch batch = llama_batch_init(chunk_end - i, 0, 1);
            for (uint32_t j = i; j < chunk_end; ++j) {
                common_batch_add(batch, prompt[j], j, {0}, j + 1 == fill_tokens);
            }
            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                return fail("failed to decode fill prompt at position " + std::to_string(i));
            }
            llama_batch_free(batch);
            i = chunk_end;
        }

        check(true, "filled KV cache with " + std::to_string(fill_tokens) + " tokens");
    }

    // Save state before compaction so we can measure things.
    std::vector<uint8_t> full_state(llama_state_seq_get_size(ctx, 0));
    const size_t ncopy = llama_state_seq_get_data(ctx, full_state.data(), full_state.size(), 0);
    check(ncopy == full_state.size(), "saved full-cache state (" + std::to_string(ncopy) + " bytes)");

    // =========================================================================
    // TEST 2: Compact at 2x to free ~50% of prefix tokens
    // =========================================================================
    std::printf("\n=== TEST 2: Compact at 2x to Free Prefix Tokens ===\n");
    {
        // Define compaction parameters: compact 80% of the filled tokens at 2x.
        const int live_suffix_pos0 = (int)(fill_tokens * 0.8);
        const int target = live_suffix_pos0 / 2;

        llama_kv_compact_pipeline_stats stats = {};
        const bool compact_ok = kv->compacted_prefix_select_from_live_kv(
                0, target, live_suffix_pos0, &stats);
        check(compact_ok, "select pipeline succeeds on near-full cache");

        if (compact_ok) {
            check(stats.n_prefix_tokens == (uint32_t) live_suffix_pos0,
                  "prefix token count matches (" + std::to_string(stats.n_prefix_tokens) + ")");
            check(stats.n_selected_tokens == (uint32_t) target,
                  "selected token count matches target (" + std::to_string(stats.n_selected_tokens) + ")");

            kv->compacted_prefix_set_execution(0, true);
            const bool reclaim_ok = kv->compacted_prefix_reclaim_live_kv(0);
            check(reclaim_ok, "reclaim succeeds after compaction");

            if (reclaim_ok) {
                const uint32_t active_after = kv->compacted_prefix_active_n_kv(0);
                check(active_after > 0,
                      "active_n_kv > 0 after reclaim (" + std::to_string(active_after) + ")");
                // After reclaim, active tokens should be reduced compared to the
                // original fill. The exact count depends on implementation details
                // (virtual cells vs physical), but it should be less than fill_tokens.
                check(active_after < fill_tokens,
                      "active_n_kv < fill_tokens (" + std::to_string(active_after) +
                      " < " + std::to_string(fill_tokens) + ")");
            }
        }
    }

    // =========================================================================
    // TEST 3: Verify generation continues after reclaim
    // =========================================================================
    std::printf("\n=== TEST 3: Continue Generation After Reclaim ===\n");
    {
        // After compaction + reclaim, there should be free KV slots.
        // Decode multiple continuation tokens to verify generation works.
        const int n_continuation = 8;
        bool decode_ok = true;
        for (int i = 0; i < n_continuation; ++i) {
            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, 1, fill_tokens + i, {0}, true);
            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                decode_ok = false;
                break;
            }
            llama_batch_free(batch);
        }
        check(decode_ok, "decoded " + std::to_string(n_continuation) + " continuation tokens after reclaim");

        // Verify logits are finite (not NaN/Inf).
        if (decode_ok) {
            const float * logits = llama_get_logits_ith(ctx, 0);
            const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            bool finite = true;
            for (uint32_t v = 0; v < n_vocab; ++v) {
                if (!std::isfinite(logits[v])) { finite = false; break; }
            }
            check(finite, "post-reclaim continuation logits are finite");
        }
    }

    // =========================================================================
    // TEST 4: Reclaim frees tokens correctly (second compaction cycle)
    // =========================================================================
    std::printf("\n=== TEST 4: Second Compaction Cycle (Reclaim Correctness) ===\n");
    {
        // Restore the full-cache state and run a second compaction to verify
        // that reclaim correctly frees tokens and the cycle is repeatable.
        if (llama_state_seq_set_data(ctx, full_state.data(), full_state.size(), 0) != full_state.size()) {
            return fail("failed to restore full-cache state for second cycle");
        }

        // Clear any residual compacted prefix from previous test.
        kv->compacted_prefix_clear(0, true);

        const int live_suffix_pos0 = (int)(fill_tokens * 0.8);
        const int target_4x = live_suffix_pos0 / 4;  // 4x compression

        llama_kv_compact_pipeline_stats stats = {};
        const bool compact_ok = kv->compacted_prefix_select_from_live_kv(
                0, target_4x, live_suffix_pos0, &stats);
        check(compact_ok, "second-cycle 4x compaction succeeds");

        if (compact_ok) {
            kv->compacted_prefix_set_execution(0, true);
            const bool reclaim_ok = kv->compacted_prefix_reclaim_live_kv(0);
            check(reclaim_ok, "second-cycle reclaim succeeds");

            if (reclaim_ok) {
                const uint32_t active_after = kv->compacted_prefix_active_n_kv(0);
                check(active_after > 0, "second-cycle active_n_kv > 0 (" + std::to_string(active_after) + ")");

                // Decode continuation to verify the second reclaim cycle works.
                llama_batch batch = llama_batch_init(1, 0, 1);
                common_batch_add(batch, 1, fill_tokens, {0}, true);
                const bool decode_ok = llama_decode(ctx, batch) == 0;
                llama_batch_free(batch);
                check(decode_ok, "decode succeeds after second-cycle reclaim");

                if (decode_ok) {
                    const float * logits = llama_get_logits_ith(ctx, 0);
                    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
                    bool finite = true;
                    for (uint32_t v = 0; v < n_vocab; ++v) {
                        if (!std::isfinite(logits[v])) { finite = false; break; }
                    }
                    check(finite, "second-cycle continuation logits are finite");
                }
            }
        }
    }

    // =========================================================================
    // TEST 5: Quality check — compare exhausted-cache compaction to baseline
    // =========================================================================
    std::printf("\n=== TEST 5: Quality Check (Exhausted Cache vs Baseline) ===\n");
    {
        // Restore full state and get baseline logits.
        if (llama_state_seq_set_data(ctx, full_state.data(), full_state.size(), 0) != full_state.size()) {
            return fail("failed to restore state for quality check");
        }
        kv->compacted_prefix_clear(0, true);

        llama_batch batch = llama_batch_init(1, 0, 1);
        common_batch_add(batch, 1, fill_tokens, {0}, true);
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return fail("baseline logit decode failed");
        }
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * bl = llama_get_logits_ith(ctx, 0);
        std::vector<float> baseline_logits(bl, bl + n_vocab);
        llama_batch_free(batch);

        // Restore, compact, and compare.
        if (llama_state_seq_set_data(ctx, full_state.data(), full_state.size(), 0) != full_state.size()) {
            return fail("failed to restore state for compacted quality check");
        }
        kv->compacted_prefix_clear(0, true);

        const int live_suffix_pos0 = (int)(fill_tokens * 0.8);
        const int target_2x = live_suffix_pos0 / 2;

        const bool compact_ok = kv->compacted_prefix_select_from_live_kv(
                0, target_2x, live_suffix_pos0);
        check(compact_ok, "quality-check compaction succeeds");

        if (compact_ok) {
            if (!kv->compacted_prefix_set_execution(0, true)) {
                return fail("quality-check: set_execution failed");
            }
            if (!kv->compacted_prefix_reclaim_live_kv(0)) {
                return fail("quality-check: reclaim_live_kv failed");
            }

            batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, 1, fill_tokens, {0}, true);
            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                return fail("compacted logit decode failed");
            }
            const float * cl = llama_get_logits_ith(ctx, 0);
            std::vector<float> compacted_logits(cl, cl + n_vocab);
            llama_batch_free(batch);

            const float cos = llama_kv_compact_cosine_similarity(baseline_logits, compacted_logits);
            std::printf("  exhausted-cache 2x cosine=%.6f\n", cos);
            check(cos >= 0.85f, "exhausted-cache 2x cosine >= 0.85");
        }
    }

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::printf("\n=== SUMMARY ===\n");
    std::printf("  %d passed, %d failed\n", n_passed, n_failed);
    return n_failed > 0 ? 1 : 0;
}
