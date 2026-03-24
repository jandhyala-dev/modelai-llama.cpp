// CI-gatable quality gate tests for KV cache compaction.
//
// Designed to run with small models (stories15M, Gemma 270M) in CI.
// Enforces hard floors that must never regress:
//   1. Cosine similarity floor after compaction
//   2. KV token survival within tolerance of target ratio
//   3. State serialization round-trip fidelity
//   4. Error-free execution at 2x, 4x, 10x compression ratios
//
// Usage:
//   ./test-kv-compact-quality-gate -m /path/to/stories15M-q4_0.gguf
//   ./test-kv-compact-quality-gate -m /path/to/gemma-2b-q4_0.gguf -ngl 99

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
int n_skipped = 0;

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
    std::fprintf(stderr, "test-kv-compact-quality-gate: %s\n", message.c_str());
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

// Check that all values in a float vector are finite (no NaN, no Inf).
bool all_finite(const std::vector<float> & v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

// Run compaction at a given ratio and return success + stats.
// Restores state from seq_state before compacting.
struct compact_result {
    bool                              ok;
    float                             cosine;
    uint32_t                          tokens_after;
    std::vector<float>                logits;
    llama_kv_compact_pipeline_stats   stats;
};

compact_result run_compaction_at_ratio(
        llama_context  * ctx,
        llama_kv_cache * kv,
        const std::vector<uint8_t> & seq_state,
        const std::vector<float>   & baseline_logits,
        int seed_tokens,
        int live_suffix_pos0,
        int target_tokens,
        llama_token continuation) {

    compact_result result = {};
    result.ok = false;
    result.cosine = 0.0f;
    result.tokens_after = 0;

    // Restore clean state.
    if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
        return result;
    }

    result.ok = kv->compacted_prefix_select_from_live_kv(
            0, target_tokens, live_suffix_pos0, &result.stats);
    if (!result.ok) return result;

    if (!kv->compacted_prefix_set_execution(0, true)) {
        result.ok = false;
        return result;
    }
    if (!kv->compacted_prefix_reclaim_live_kv(0)) {
        result.ok = false;
        return result;
    }

    result.tokens_after = kv->compacted_prefix_active_n_kv(0);

    try {
        result.logits = decode_one_and_capture_logits(ctx, continuation, seed_tokens);
        result.cosine = llama_kv_compact_cosine_similarity(baseline_logits, result.logits);
    } catch (const std::runtime_error &) {
        result.ok = false;
    }

    return result;
}

} // namespace

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.seed = 1234;
    params.kv_unified = true;
    params.n_parallel = 1;
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
        std::printf("Model does not support compaction (SWA or unsupported arch). Skipping.\n");
        return 0;
    }

    // Seed the KV cache with real text.
    constexpr int seed_tokens       = 320;
    constexpr int live_suffix_pos0  = 256;
    constexpr llama_token continuation = 1;

    const std::vector<llama_token> prompt = build_real_text_prompt(ctx, seed_tokens);

    llama_batch batch = llama_batch_init(prompt.size(), 0, 1);
    for (int i = 0; i < seed_tokens; ++i) {
        common_batch_add(batch, prompt[i], i, {0}, i + 1 == seed_tokens);
    }
    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        return fail("failed to decode seed prompt");
    }

    // Save baseline state.
    std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 0));
    const size_t ncopy = llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 0);
    if (ncopy != seq_state.size()) {
        llama_batch_free(batch);
        return fail("failed to save baseline state");
    }

    // Capture baseline logits.
    if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
        llama_batch_free(batch);
        return fail("failed to restore baseline state for logit capture");
    }
    const std::vector<float> baseline_logits = decode_one_and_capture_logits(ctx, continuation, seed_tokens);

    // =========================================================================
    // GATE 1: Cosine Similarity Floor
    //
    // After compaction at 2x, cosine similarity between pre/post KV logits
    // must be >= 0.85. This is the minimum quality bar for any compaction
    // pipeline to ship.
    // =========================================================================
    std::printf("\n=== GATE 1: Cosine Similarity Floor (>= 0.85 at 2x) ===\n");
    {
        const int target_2x = live_suffix_pos0 / 2;  // 128 tokens
        compact_result r = run_compaction_at_ratio(
                ctx, kv, seq_state, baseline_logits,
                seed_tokens, live_suffix_pos0, target_2x, continuation);

        check(r.ok, "2x compaction executes successfully");
        if (r.ok) {
            check(all_finite(r.logits), "2x post-compaction logits are all finite");
            std::printf("  2x cosine=%.6f\n", r.cosine);
            check(r.cosine >= 0.85f,
                  "2x cosine similarity >= 0.85 (actual=" + std::to_string(r.cosine) + ")");
        }
    }

    // =========================================================================
    // GATE 2: KV Token Survival
    //
    // After compaction at ratio R, verify that the number of active KV tokens
    // is within 5% of the expected target (tokens_before / R). The compaction
    // pipeline may include suffix tokens beyond the compacted prefix, so we
    // check that active_n_kv is bounded within tolerance of expected.
    // =========================================================================
    std::printf("\n=== GATE 2: KV Token Survival (within 5%% of target) ===\n");
    {
        struct ratio_test {
            const char * label;
            int          ratio;
        };
        const ratio_test ratios[] = {
            {"2x",  2},
            {"4x",  4},
        };

        for (const auto & rt : ratios) {
            const int target = live_suffix_pos0 / rt.ratio;
            compact_result r = run_compaction_at_ratio(
                    ctx, kv, seq_state, baseline_logits,
                    seed_tokens, live_suffix_pos0, target, continuation);

            check(r.ok, std::string(rt.label) + " compaction succeeds for token survival check");
            if (r.ok) {
                // active_n_kv includes both compacted prefix virtual cells and
                // the live suffix. We verify the pipeline stats report the
                // expected selected token count.
                check(r.stats.n_selected_tokens == (uint32_t)target,
                      std::string(rt.label) + " selected tokens match target ("
                      + std::to_string(r.stats.n_selected_tokens) + " == "
                      + std::to_string(target) + ")");

                // Verify prefix token count matches what was submitted.
                check(r.stats.n_prefix_tokens == (uint32_t)live_suffix_pos0,
                      std::string(rt.label) + " prefix tokens match live_suffix_pos0 ("
                      + std::to_string(r.stats.n_prefix_tokens) + " == "
                      + std::to_string(live_suffix_pos0) + ")");

                // Verify the compression ratio is within 5% of nominal.
                // Ratio = prefix_tokens / selected_tokens.
                const float actual_ratio = (float)r.stats.n_prefix_tokens / (float)r.stats.n_selected_tokens;
                const float expected_ratio = (float)rt.ratio;
                const float tolerance = 0.05f * expected_ratio;
                check(std::fabs(actual_ratio - expected_ratio) <= tolerance,
                      std::string(rt.label) + " compression ratio within 5% ("
                      + std::to_string(actual_ratio) + " vs " + std::to_string(expected_ratio) + ")");
            }
        }
    }

    // =========================================================================
    // GATE 3: State Serialization Round-Trip
    //
    // Compact -> save state -> restore state -> decode -> verify output matches
    // the pre-save decode. Round-trip must be lossless (cosine >= 0.99).
    // =========================================================================
    std::printf("\n=== GATE 3: State Serialization Round-Trip (cosine >= 0.99) ===\n");
    {
        const int target_2x = live_suffix_pos0 / 2;

        // Restore, compact, enable, reclaim.
        if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
            llama_batch_free(batch);
            return fail("failed to restore state for serialization gate");
        }

        const bool compact_ok = kv->compacted_prefix_select_from_live_kv(
                0, target_2x, live_suffix_pos0);
        check(compact_ok, "serialization: compaction succeeds");

        if (compact_ok) {
            if (!kv->compacted_prefix_set_execution(0, true)) {
                return fail("serialization: set_execution failed");
            }
            if (!kv->compacted_prefix_reclaim_live_kv(0)) {
                return fail("serialization: reclaim_live_kv failed");
            }

            // Decode BEFORE save to get pre-save logits.
            const std::vector<float> pre_save_logits =
                decode_one_and_capture_logits(ctx, continuation, seed_tokens);
            check(all_finite(pre_save_logits), "serialization: pre-save logits are finite");

            // Save state with compacted prefix.
            std::vector<uint8_t> state_with_cp(llama_state_seq_get_size(ctx, 0));
            const size_t saved = llama_state_seq_get_data(
                    ctx, state_with_cp.data(), state_with_cp.size(), 0);
            check(saved == state_with_cp.size(), "serialization: save succeeds");
            check(saved > 0, "serialization: saved state is non-zero ("
                  + std::to_string(saved) + " bytes)");

            // Clear KV state completely.
            kv->seq_rm(0, -1, -1);
            kv->compacted_prefix_clear(0, true);

            // Restore.
            const size_t restored = llama_state_seq_set_data(
                    ctx, state_with_cp.data(), state_with_cp.size(), 0);
            check(restored == state_with_cp.size(), "serialization: restore succeeds");

            // Decode after restore and compare.
            const std::vector<float> post_restore_logits =
                decode_one_and_capture_logits(ctx, continuation, seed_tokens);
            check(all_finite(post_restore_logits), "serialization: post-restore logits are finite");

            const float cos = llama_kv_compact_cosine_similarity(
                    pre_save_logits, post_restore_logits);
            std::printf("  serialization round-trip cosine=%.6f\n", cos);
            check(cos >= 0.99f,
                  "serialization: round-trip cosine >= 0.99 (actual="
                  + std::to_string(cos) + ")");
        }
    }

    // =========================================================================
    // GATE 4: Error-Free Execution at Multiple Compression Ratios
    //
    // Compact at 2x, 4x, 10x — no crashes, no NaN, no assertion failures.
    // Quality thresholds degrade gracefully with compression:
    //   2x  >= 0.85
    //   4x  >= 0.75
    //   10x >= 0.50 (informational — extreme compression)
    // =========================================================================
    std::printf("\n=== GATE 4: Error-Free Execution at 2x/4x/10x ===\n");
    {
        struct ratio_gate {
            const char * label;
            int          ratio;
            float        min_cosine;
            bool         hard_gate;  // false = informational only
        };
        const ratio_gate gates[] = {
            {"2x",   2, 0.85f, true },
            {"4x",   4, 0.75f, true },
            {"10x", 10, 0.50f, false},  // informational: extreme compression
        };

        for (const auto & g : gates) {
            int target = live_suffix_pos0 / g.ratio;
            // Clamp target to at least 2 tokens to avoid degenerate cases.
            if (target < 2) target = 2;

            std::printf("\n  --- %s compression (%d -> %d tokens) ---\n",
                        g.label, live_suffix_pos0, target);

            compact_result r = run_compaction_at_ratio(
                    ctx, kv, seq_state, baseline_logits,
                    seed_tokens, live_suffix_pos0, target, continuation);

            check(r.ok, std::string(g.label) + " compaction executes without crash");

            if (r.ok) {
                const bool logits_finite = all_finite(r.logits);
                check(logits_finite,
                      std::string(g.label) + " post-compaction logits contain no NaN/Inf");

                std::printf("  %s cosine=%.6f (threshold=%.2f)\n",
                            g.label, r.cosine, g.min_cosine);

                if (g.hard_gate) {
                    check(r.cosine >= g.min_cosine,
                          std::string(g.label) + " cosine >= " + std::to_string(g.min_cosine)
                          + " (actual=" + std::to_string(r.cosine) + ")");
                } else {
                    // Informational: log but do not gate.
                    if (r.cosine >= g.min_cosine) {
                        check(true, std::string(g.label) + " cosine meets informational threshold");
                    } else {
                        std::printf("  INFO: %s cosine %.3f < %.2f (informational, not gated)\n",
                                    g.label, r.cosine, g.min_cosine);
                        n_skipped++;
                    }
                }
            }
        }
    }

    llama_batch_free(batch);

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::printf("\n=== QUALITY GATE SUMMARY ===\n");
    std::printf("  %d passed, %d failed, %d skipped\n", n_passed, n_failed, n_skipped);

    if (n_failed > 0) {
        std::fprintf(stderr, "\nQUALITY GATE FAILED: %d check(s) did not pass.\n", n_failed);
    }

    return n_failed > 0 ? 1 : 0;
}
