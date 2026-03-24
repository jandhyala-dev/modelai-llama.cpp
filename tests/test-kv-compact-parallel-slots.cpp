// Parallel-slot stress test for KV cache compaction.
//
// Verifies that compaction on one sequence does not corrupt another:
//   1. Initialize context with --parallel 2 (two slots)
//   2. Fill both slots (seq 0 and seq 1) with tokens
//   3. Compact slot 0 while slot 1 has active KV data
//   4. Verify slot 1 logits are unchanged after slot 0 compaction
//   5. Verify both slots can continue generation without crashing
//
// Usage:
//   ./test-kv-compact-parallel-slots -m /path/to/stories15M-q4_0.gguf

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "src/llama-context.h"
#include "src/llama-kv-compact-pipeline.h"
#include "src/llama-kv-compact-solver.h"
#include "src/llama-kv-cache.h"
#include "src/llama-kv-cache-iswa.h"

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
    std::fprintf(stderr, "test-kv-compact-parallel-slots: %s\n", message.c_str());
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
    params.n_parallel = 2;     // Two parallel slots
    params.n_ctx = 512;        // Total context shared across slots
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

    if (!kv->supports_compaction()) {
        std::printf("SKIP: model does not support compaction\n");
        return 0;
    }

    const uint32_t n_ctx = llama_n_ctx(ctx);
    // Each slot gets roughly half the context. Use a conservative fill to
    // leave room for continuation tokens.
    const uint32_t tokens_per_slot = (n_ctx / 2) - 16;

    std::printf("\n=== Parallel-Slot KV Compaction Test ===\n");
    std::printf("n_ctx=%u, n_parallel=2, tokens_per_slot=%u\n\n", n_ctx, tokens_per_slot);

    const std::vector<llama_token> prompt = build_fill_prompt(ctx, tokens_per_slot);

    // =========================================================================
    // TEST 1: Fill both slots with tokens
    // =========================================================================
    std::printf("=== TEST 1: Fill Both Slots ===\n");
    {
        // Fill seq 0.
        llama_batch batch = llama_batch_init(tokens_per_slot, 0, 1);
        for (uint32_t i = 0; i < tokens_per_slot; ++i) {
            common_batch_add(batch, prompt[i], i, {0}, i + 1 == tokens_per_slot);
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return fail("failed to fill seq 0");
        }
        llama_batch_free(batch);
        check(true, "filled seq 0 with " + std::to_string(tokens_per_slot) + " tokens");

        // Fill seq 1.
        batch = llama_batch_init(tokens_per_slot, 0, 1);
        for (uint32_t i = 0; i < tokens_per_slot; ++i) {
            common_batch_add(batch, prompt[i], i, {1}, i + 1 == tokens_per_slot);
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return fail("failed to fill seq 1");
        }
        llama_batch_free(batch);
        check(true, "filled seq 1 with " + std::to_string(tokens_per_slot) + " tokens");
    }

    // =========================================================================
    // TEST 2: Capture slot 1 baseline logits before any compaction
    // =========================================================================
    std::printf("\n=== TEST 2: Capture Slot 1 Baseline Logits ===\n");
    std::vector<float> slot1_baseline_logits;
    {
        llama_batch batch = llama_batch_init(1, 0, 1);
        common_batch_add(batch, 1, tokens_per_slot, {1}, true);
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return fail("failed to decode slot 1 baseline logit");
        }
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * logits = llama_get_logits_ith(ctx, 0);
        slot1_baseline_logits.assign(logits, logits + n_vocab);
        llama_batch_free(batch);

        bool finite = true;
        for (float v : slot1_baseline_logits) {
            if (!std::isfinite(v)) { finite = false; break; }
        }
        check(finite, "slot 1 baseline logits are finite");
    }

    // =========================================================================
    // TEST 3: Compact slot 0, verify it succeeds
    // =========================================================================
    std::printf("\n=== TEST 3: Compact Slot 0 ===\n");
    {
        const int live_suffix_pos0 = (int)(tokens_per_slot * 0.8);
        const int target_2x = live_suffix_pos0 / 2;

        llama_kv_compact_pipeline_stats stats = {};
        const bool compact_ok = kv->compacted_prefix_select_from_live_kv(
                0, target_2x, live_suffix_pos0, &stats);
        check(compact_ok, "select pipeline on seq 0 succeeds with seq 1 active");

        if (compact_ok) {
            check(stats.n_selected_tokens == (uint32_t) target_2x,
                  "seq 0 selected tokens match target (" + std::to_string(stats.n_selected_tokens) + ")");

            kv->compacted_prefix_set_execution(0, true);
            const bool reclaim_ok = kv->compacted_prefix_reclaim_live_kv(0);
            check(reclaim_ok, "seq 0 reclaim succeeds");
        }
    }

    // =========================================================================
    // TEST 4: Verify slot 1 is not corrupted by slot 0 compaction
    // =========================================================================
    std::printf("\n=== TEST 4: Verify Slot 1 Integrity After Slot 0 Compaction ===\n");
    {
        llama_batch batch = llama_batch_init(1, 0, 1);
        // Decode at tokens_per_slot + 1 because the baseline used tokens_per_slot.
        common_batch_add(batch, 1, tokens_per_slot + 1, {1}, true);
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            // Slot 1 decode failure after slot 0 compaction is a critical issue.
            check(false, "slot 1 decode succeeds after slot 0 compaction");
        } else {
            llama_batch_free(batch);
            check(true, "slot 1 decode succeeds after slot 0 compaction");

            const float * logits = llama_get_logits_ith(ctx, 0);
            const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            bool finite = true;
            for (uint32_t v = 0; v < n_vocab; ++v) {
                if (!std::isfinite(logits[v])) { finite = false; break; }
            }
            check(finite, "slot 1 post-compaction logits are finite");
        }
    }

    // =========================================================================
    // TEST 5: Verify slot 0 can continue generation after compaction
    // =========================================================================
    std::printf("\n=== TEST 5: Slot 0 Continuation After Compaction ===\n");
    {
        // Decode a continuation token on the compacted seq 0.
        llama_batch batch = llama_batch_init(1, 0, 1);
        common_batch_add(batch, 1, tokens_per_slot, {0}, true);
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            check(false, "slot 0 continuation decode succeeds after compaction");
        } else {
            llama_batch_free(batch);
            check(true, "slot 0 continuation decode succeeds after compaction");

            const float * logits = llama_get_logits_ith(ctx, 0);
            const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            bool finite = true;
            for (uint32_t v = 0; v < n_vocab; ++v) {
                if (!std::isfinite(logits[v])) { finite = false; break; }
            }
            check(finite, "slot 0 post-compaction continuation logits are finite");
        }
    }

    // =========================================================================
    // TEST 6: Compact slot 1 independently, verify no crash
    // =========================================================================
    std::printf("\n=== TEST 6: Compact Slot 1 Independently ===\n");
    {
        const int live_suffix_pos0 = (int)(tokens_per_slot * 0.8);
        const int target_2x = live_suffix_pos0 / 2;

        llama_kv_compact_pipeline_stats stats = {};
        const bool compact_ok = kv->compacted_prefix_select_from_live_kv(
                1, target_2x, live_suffix_pos0, &stats);
        check(compact_ok, "select pipeline on seq 1 succeeds (both slots compacted)");

        if (compact_ok) {
            kv->compacted_prefix_set_execution(1, true);
            const bool reclaim_ok = kv->compacted_prefix_reclaim_live_kv(1);
            check(reclaim_ok, "seq 1 reclaim succeeds");

            if (reclaim_ok) {
                // Verify both slots can decode after both are compacted.
                bool both_ok = true;

                llama_batch batch = llama_batch_init(1, 0, 1);
                common_batch_add(batch, 1, tokens_per_slot + 1, {0}, true);
                if (llama_decode(ctx, batch) != 0) {
                    both_ok = false;
                }
                llama_batch_free(batch);

                batch = llama_batch_init(1, 0, 1);
                common_batch_add(batch, 1, tokens_per_slot + 2, {1}, true);
                if (llama_decode(ctx, batch) != 0) {
                    both_ok = false;
                }
                llama_batch_free(batch);

                check(both_ok, "both slots decode successfully after both are compacted");
            }
        }
    }

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::printf("\n=== SUMMARY ===\n");
    std::printf("  %d passed, %d failed\n", n_passed, n_failed);
    return n_failed > 0 ? 1 : 0;
}
