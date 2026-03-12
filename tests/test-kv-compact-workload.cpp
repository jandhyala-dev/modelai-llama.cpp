// Production workload tests for KV cache compaction (6b-13).
//
// Tests compaction quality and decode performance at production-class context
// lengths (2K+ tokens) with real models.  Outputs structured CSV artifacts
// for benchmark tracking.
//
// Usage:
//   # Smoke test (stories15M, CI-safe):
//   ./test-kv-compact-workload -m models/test/stories15M-q4_0.gguf
//
//   # Production workload (14B-class model):
//   ./test-kv-compact-workload -m models/test/Qwen3-14B-Q4_K_M.gguf -ngl 99 -c 4096
//
//   # With specific pipeline:
//   PIPELINE=omp ./test-kv-compact-workload -m models/test/Qwen3-14B-Q4_K_M.gguf -ngl 99
//
//   # Write CSV artifact:
//   ARTIFACT=results.csv ./test-kv-compact-workload -m models/test/Qwen3-14B-Q4_K_M.gguf -ngl 99

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "src/llama-context.h"
#include "src/llama-kv-compact-pipeline.h"
#include "src/llama-kv-compact-solver.h"
#include "src/llama-kv-cache.h"
#include "src/llama-kv-cache-iswa.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int fail(const std::string & message) {
    std::fprintf(stderr, "test-kv-compact-workload: %s\n", message.c_str());
    return 1;
}

std::vector<llama_token> build_real_text_prompt(llama_context * ctx, size_t min_tokens) {
    // SEC filing-style text to simulate production workload.
    const std::string para =
        "The quarterly letter reviewed liquidity, capital allocation, recurring revenue, customer retention, and operating leverage. "
        "Management discussed cash flow discipline, pricing pressure, inventory turns, software adoption, and regional demand. "
        "Analysts compared the margin profile to prior quarters and noted that guidance depended on enterprise renewals, deferred revenue conversion, and foreign exchange stability. "
        "The board approved a share repurchase program of up to two billion dollars over the next twelve months, subject to market conditions and regulatory approval. "
        "Revenue from cloud services grew twenty-three percent year-over-year, driven by enterprise migration and expanded API usage across financial services clients. "
        "Operating expenses as a percentage of revenue declined by one hundred and forty basis points, reflecting disciplined headcount management and infrastructure efficiency gains. ";

    std::string text;
    std::vector<llama_token> tokens;
    while (tokens.size() < min_tokens) {
        text += para;
        tokens = common_tokenize(ctx, text, true, false);
    }
    tokens.resize(min_tokens);
    return tokens;
}

struct workload_result {
    std::string model_name;
    std::string pipeline;
    std::string backend;
    int         prefix_tokens;
    int         compression_ratio;
    int         compacted_tokens;
    int         continuation_tokens;
    float       logit_cosine;
    double      compaction_time_ms;
    double      solver_time_ms;
    double      query_gen_time_ms;
    double      baseline_decode_tok_s;
    double      compacted_decode_tok_s;
    uint32_t    active_n_kv;
    bool        pass;
    float       threshold;
};

void print_result(const workload_result & r) {
    std::printf("  %dx | cos=%.4f (>= %.2f %s) | compact=%.1fms | "
                "baseline=%.1f tok/s | compacted=%.1f tok/s | active_n_kv=%u\n",
                r.compression_ratio, r.logit_cosine, r.threshold,
                r.pass ? "PASS" : "FAIL",
                r.compaction_time_ms,
                r.baseline_decode_tok_s, r.compacted_decode_tok_s,
                r.active_n_kv);
}

void write_csv_header(FILE * f) {
    std::fprintf(f, "model,pipeline,backend,prefix_tokens,compression_ratio,"
                    "compacted_tokens,continuation_tokens,logit_cosine,threshold,"
                    "pass,compaction_time_ms,solver_time_ms,query_gen_time_ms,"
                    "baseline_decode_tok_s,compacted_decode_tok_s,active_n_kv\n");
}

void write_csv_row(FILE * f, const workload_result & r) {
    std::fprintf(f, "%s,%s,%s,%d,%d,%d,%d,%.6f,%.2f,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%u\n",
                 r.model_name.c_str(), r.pipeline.c_str(), r.backend.c_str(),
                 r.prefix_tokens, r.compression_ratio, r.compacted_tokens,
                 r.continuation_tokens, r.logit_cosine, r.threshold,
                 r.pass ? "PASS" : "FAIL",
                 r.compaction_time_ms, r.solver_time_ms, r.query_gen_time_ms,
                 r.baseline_decode_tok_s, r.compacted_decode_tok_s, r.active_n_kv);
}

// Decode n_tokens starting at pos, return elapsed time in ms.
double decode_burst(llama_context * ctx, int n_tokens, int start_pos) {
    const llama_token tok = 1;  // arbitrary continuation token
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_tokens; ++i) {
        llama_batch batch = llama_batch_init(1, 0, 1);
        common_batch_add(batch, tok, start_pos + i, {0}, true);
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return -1.0;
        }
        llama_batch_free(batch);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

bool run_compaction(llama_kv_cache * kv, const std::string & pipeline,
                    int target, int live_suffix_pos0,
                    llama_kv_compact_pipeline_stats * stats) {
    if (pipeline == "omp") {
        return kv->compacted_prefix_omp_from_live_kv(0, target, live_suffix_pos0, stats);
    }
    if (pipeline == "solver") {
        return kv->compacted_prefix_fit_from_live_kv(0, target, live_suffix_pos0, stats);
    }
    return kv->compacted_prefix_select_from_live_kv(0, target, live_suffix_pos0, stats);
}

} // namespace

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.seed = 1234;
    params.kv_unified = true;
    params.n_parallel = 1;
    params.n_ctx = 4096;

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

    // Configuration from environment.
    const char * env_pipeline = std::getenv("PIPELINE");
    const std::string pipeline = env_pipeline ? env_pipeline : "select";
    if (pipeline != "select" && pipeline != "solver" && pipeline != "omp") {
        return fail("PIPELINE must be 'select', 'solver', or 'omp'");
    }

    const char * env_artifact = std::getenv("ARTIFACT");

    // Determine workload class from model size.
    const uint64_t n_params = llama_model_n_params(model);
    const bool is_smoke = n_params < 500000000;  // < 500M = smoke/W1
    const int n_ctx = (int) llama_n_ctx(ctx);

    // Workload sizing.
    int prefix_tokens;
    int continuation_tokens;
    if (is_smoke) {
        // 4e: Small model smoke — use small context.
        prefix_tokens = std::min(256, n_ctx - 64);
        continuation_tokens = 8;
    } else {
        // 4a/4b: Production workload — use 80% of context for prefix.
        prefix_tokens = std::min((int)(n_ctx * 0.8), n_ctx - 128);
        continuation_tokens = 16;
    }

    const int live_suffix_pos0 = (int)(prefix_tokens * 0.8);  // compact 80% of prefix
    const int seed_tokens = prefix_tokens;

    // Detect backend.
    std::string backend_name = "cpu";
#ifdef GGML_USE_METAL
    backend_name = "metal";
#endif
#ifdef GGML_USE_CUDA
    backend_name = "cuda";
#endif

    // Model identifier from file path.
    std::string model_name = params.model.path;
    {
        auto pos = model_name.find_last_of('/');
        if (pos != std::string::npos) {
            model_name = model_name.substr(pos + 1);
        }
        pos = model_name.find(".gguf");
        if (pos != std::string::npos) {
            model_name = model_name.substr(0, pos);
        }
    }

    std::printf("=== KV Compaction Workload Test ===\n");
    std::printf("model:          %s\n", model_name.c_str());
    std::printf("params:         %.1fB\n", n_params / 1e9);
    std::printf("backend:        %s\n", backend_name.c_str());
    std::printf("pipeline:       %s\n", pipeline.c_str());
    std::printf("n_ctx:          %d\n", n_ctx);
    std::printf("prefix_tokens:  %d\n", prefix_tokens);
    std::printf("live_suffix:    %d (pos >= %d)\n", seed_tokens - live_suffix_pos0, live_suffix_pos0);
    std::printf("continuation:   %d tokens (for tok/s)\n", continuation_tokens);
    const char * wclass = "W2 (standard)";
    if (is_smoke) {
        wclass = "W1 (smoke)";
    } else if (n_params > 10e9) {
        wclass = "W3 (production)";
    }
    std::printf("workload_class: %s\n\n", wclass);

    // Build prompt and prefill.
    const std::vector<llama_token> prompt = build_real_text_prompt(ctx, seed_tokens);

    std::printf("prefilling %d tokens...\n", seed_tokens);
    {
        const int n_batch = (int) llama_n_batch(ctx);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < seed_tokens; ) {
            const int chunk_end = std::min(i + n_batch, seed_tokens);
            llama_batch batch = llama_batch_init(chunk_end - i, 0, 1);
            for (int j = i; j < chunk_end; ++j) {
                common_batch_add(batch, prompt[j], j, {0}, j + 1 == seed_tokens);
            }
            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                return fail("failed to decode seed prompt");
            }
            llama_batch_free(batch);
            i = chunk_end;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("prefill: %.1fms (%.1f tok/s)\n", prefill_ms, seed_tokens / (prefill_ms / 1000.0));
    }

    // Save state for restoration between tests.
    std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 0));
    if (llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
        return fail("failed to save baseline state");
    }

    // Baseline logits (single continuation token).
    if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
        return fail("failed to restore baseline state for logits");
    }
    const llama_token continuation_tok = 1;
    std::vector<float> baseline_logits;
    {
        llama_batch batch = llama_batch_init(1, 0, 1);
        common_batch_add(batch, continuation_tok, seed_tokens, {0}, true);
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return fail("baseline logit decode failed");
        }
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * logits = llama_get_logits_ith(ctx, 0);
        baseline_logits.assign(logits, logits + n_vocab);
        llama_batch_free(batch);
    }

    // Baseline decode tok/s (burst of continuation tokens).
    double baseline_decode_tok_s = 0.0;
    {
        if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
            return fail("failed to restore state for baseline decode");
        }
        const double burst_ms = decode_burst(ctx, continuation_tokens, seed_tokens);
        if (burst_ms < 0) {
            return fail("baseline decode burst failed");
        }
        baseline_decode_tok_s = continuation_tokens / (burst_ms / 1000.0);
        std::printf("baseline decode: %.1f tok/s (%d tokens in %.1fms)\n\n",
                    baseline_decode_tok_s, continuation_tokens, burst_ms);
    }

    // Compression ratios to test.
    // Ratio is relative to the compactable prefix (live_suffix_pos0 tokens).
    const int ratios[] = {2, 4, 8};
    const int n_ratios = sizeof(ratios) / sizeof(ratios[0]);

    // Thresholds calibrated from Qwen3-14B and Qwen3-30B-A3B select pipeline
    // results (logit cosine similarity).  Select consistently exceeds 0.98 at
    // all ratios; these thresholds leave margin for model/architecture variance.
    // Solver and OMP with surrogate queries on GQA models produce lower quality
    // and may fail these thresholds — that is expected and documented.
    const float thresholds[] = {0.95f, 0.90f, 0.85f};

    std::vector<workload_result> results;
    int n_fail = 0;

    for (int ri = 0; ri < n_ratios; ++ri) {
        const int ratio = ratios[ri];
        const int target = live_suffix_pos0 / ratio;
        if (target < 1) {
            continue;
        }

        // Restore full state.
        if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
            return fail("failed to restore state for compression test");
        }

        // Run compaction.
        llama_kv_compact_pipeline_stats stats = {};
        const auto t_compact0 = std::chrono::steady_clock::now();
        if (!run_compaction(kv, pipeline, target, live_suffix_pos0, &stats)) {
            return fail("compaction failed at " + std::to_string(ratio) + "x");
        }
        if (!kv->compacted_prefix_set_execution(0, true)) {
            return fail("set_execution failed at " + std::to_string(ratio) + "x");
        }
        if (!kv->compacted_prefix_reclaim_live_kv(0)) {
            return fail("reclaim failed at " + std::to_string(ratio) + "x");
        }
        const auto t_compact1 = std::chrono::steady_clock::now();
        const double compact_ms = std::chrono::duration<double, std::milli>(t_compact1 - t_compact0).count();

        // Compacted logit cosine.
        float logit_cos = 0.0f;
        {
            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, continuation_tok, seed_tokens, {0}, true);
            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                return fail("compacted logit decode failed at " + std::to_string(ratio) + "x");
            }
            const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            const float * logits = llama_get_logits_ith(ctx, 0);
            std::vector<float> compacted_logits(logits, logits + n_vocab);
            logit_cos = llama_kv_compact_cosine_similarity(baseline_logits, compacted_logits);
            llama_batch_free(batch);
        }

        // Compacted decode tok/s.
        const double burst_ms = decode_burst(ctx, continuation_tokens, seed_tokens + 1);
        double compacted_tok_s = 0.0;
        if (burst_ms > 0) {
            compacted_tok_s = continuation_tokens / (burst_ms / 1000.0);
        }

        const uint32_t active_kv = kv->compacted_prefix_active_n_kv(0);
        const float threshold = thresholds[ri];
        const bool pass = logit_cos >= threshold;

        workload_result r = {};
        r.model_name          = model_name;
        r.pipeline            = pipeline;
        r.backend             = backend_name;
        r.prefix_tokens       = live_suffix_pos0;
        r.compression_ratio   = ratio;
        r.compacted_tokens    = target;
        r.continuation_tokens = continuation_tokens;
        r.logit_cosine        = logit_cos;
        r.compaction_time_ms  = compact_ms;
        r.solver_time_ms      = stats.solver_time_ms;
        r.query_gen_time_ms   = stats.query_generation_time_ms;
        r.baseline_decode_tok_s  = baseline_decode_tok_s;
        r.compacted_decode_tok_s = compacted_tok_s;
        r.active_n_kv         = active_kv;
        r.pass                = pass;
        r.threshold           = threshold;

        results.push_back(r);
        print_result(r);

        if (!pass) {
            ++n_fail;
        }
    }

    // Write CSV artifact if requested.
    if (env_artifact) {
        FILE * f = std::fopen(env_artifact, "w");
        if (f) {
            write_csv_header(f);
            for (const auto & r : results) {
                write_csv_row(f, r);
            }
            std::fclose(f);
            std::printf("\nartifact written: %s\n", env_artifact);
        } else {
            std::fprintf(stderr, "warning: could not open artifact file '%s'\n", env_artifact);
        }
    }

    std::printf("\n=== Summary ===\n");
    std::printf("model=%s pipeline=%s backend=%s prefix=%d\n",
                model_name.c_str(), pipeline.c_str(), backend_name.c_str(), live_suffix_pos0);
    for (const auto & r : results) {
        std::printf("  %dx: cos=%.4f (>= %.2f) %s | compact=%.1fms | decode=%.1f->%.1f tok/s | n_kv=%u\n",
                    r.compression_ratio, r.logit_cosine, r.threshold,
                    r.pass ? "PASS" : "FAIL",
                    r.compaction_time_ms,
                    r.baseline_decode_tok_s, r.compacted_decode_tok_s,
                    r.active_n_kv);
    }

    if (n_fail > 0) {
        std::printf("\n%d/%d compression ratios below provisional threshold\n", n_fail, n_ratios);
    }

    return n_fail > 0 ? 1 : 0;
}
