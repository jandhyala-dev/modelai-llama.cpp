// KV compaction benchmark tool — measures perplexity impact of compaction.
//
// Usage:
//   llama-kv-compact-bench -m model.gguf -f wikitext-2.txt \
//       --ratios 2,5,10,20,50 --methods select,solver --csv results.csv
//
// Ablation flags:
//   --no-beta     Skip beta fitting (zero-beta selection only)
//   --no-cv       Skip C_v fitting (keep original V values)
//   --evict-only  Evict-based: keep first N tokens, drop rest (no scoring)
//
// Environment variables (override per-method defaults):
//   LLAMA_COMPACT_NO_BETA=1   Force zero-beta for all solver methods
//   LLAMA_COMPACT_NO_CV=1     Force original V for all solver methods
//
// Reference: arXiv:2602.16284 Section 4, fabiantax kv-compact-bench.cpp

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "src/llama-context.h"
#include "src/llama-kv-cache.h"
#include "src/llama-kv-cache-iswa.h"
#include "src/llama-memory-hybrid.h"
#include "src/llama-memory-hybrid-iswa.h"
#include "src/llama-kv-compact-pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct bench_config {
    std::vector<float>       ratios;
    std::vector<std::string> methods;
    std::string              csv_path;
    uint32_t                 prefix_tokens = 0;     // 0 = auto (half of context)
    uint32_t                 eval_tokens   = 0;     // 0 = auto (remaining tokens)
    uint32_t                 max_queries   = 256;
    int                      nnls_iters    = 2;
    float                    lambda        = 1e-6f;
    bool                     no_beta       = false;
    bool                     no_cv         = false;
    bool                     evict_only    = false;
    int                      n_runs        = 1;
};

struct bench_result {
    std::string method;
    float       ratio;
    uint32_t    prefix_tokens;
    uint32_t    target_tokens;
    uint32_t    eval_tokens;
    double      ppl;
    double      nll;
    double      time_compact_ms;
    double      time_eval_ms;
    bool        no_beta;
    bool        no_cv;
    int         run;
};

// Get the base llama_kv_cache from a context (handles plain, iSWA, hybrid).
static llama_kv_cache * get_kv_cache(llama_context * ctx) {
    auto * mem = ctx->get_memory();
    if (!mem) {
        return nullptr;
    }
    auto * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (kv) {
        return kv;
    }
    auto * kv_iswa = dynamic_cast<llama_kv_cache_iswa *>(mem);
    if (kv_iswa) {
        return const_cast<llama_kv_cache *>(kv_iswa->get_base());
    }
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem);
    if (hybrid) {
        return const_cast<llama_kv_cache *>(hybrid->get_mem_attn());
    }
    auto * hybrid_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(mem);
    if (hybrid_iswa) {
        return const_cast<llama_kv_cache *>(hybrid_iswa->get_mem_attn()->get_base());
    }
    return nullptr;
}

// Compute log-softmax for a single token prediction.
static double log_softmax_token(int n_vocab, const float * logits, int tok) {
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += std::exp(double(logits[i] - max_logit));
    }
    return double(logits[tok]) - double(max_logit) - std::log(sum_exp);
}

// Prefill tokens into the context. Returns the number of tokens actually prefilled.
static int prefill(llama_context * ctx, const std::vector<llama_token> & tokens, int n_tokens) {
    const int n = std::min(n_tokens, (int) tokens.size());
    if (n == 0) {
        return 0;
    }

    // Process in batches that fit the context.
    const int n_batch = llama_n_batch(ctx);
    int n_past = 0;

    while (n_past < n) {
        const int n_eval = std::min(n_batch, n - n_past);
        llama_batch batch = llama_batch_init(n_eval, 0, 1);

        for (int i = 0; i < n_eval; ++i) {
            common_batch_add(batch, tokens[n_past + i], n_past + i, {0}, false);
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("prefill decode failed at position %d\n", n_past);
            llama_batch_free(batch);
            return n_past;
        }

        llama_batch_free(batch);
        n_past += n_eval;
    }

    return n_past;
}

// Evaluate tokens after compaction to compute PPL.
// Returns (nll_sum, n_eval_tokens).
static std::pair<double, int> eval_ppl(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        int eval_start,
        int eval_end) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const int n_batch = llama_n_batch(ctx);
    double nll_sum = 0.0;
    int n_eval = 0;

    int pos = eval_start;
    while (pos < eval_end) {
        const int n_tokens = std::min(n_batch, eval_end - pos);
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);

        for (int i = 0; i < n_tokens; ++i) {
            // Request logits for all tokens except the last one in the batch,
            // since we need logits[i] to predict tokens[pos+i+1].
            // Actually, we need logits for positions where we can compute loss,
            // i.e., where we know the next token.
            bool get_logits = (pos + i + 1 < eval_end);
            common_batch_add(batch, tokens[pos + i], pos + i, {0}, get_logits);
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("eval decode failed at position %d\n", pos);
            llama_batch_free(batch);
            break;
        }

        // Compute NLL for each token in the batch.
        for (int i = 0; i < n_tokens; ++i) {
            if (pos + i + 1 >= eval_end) {
                break;
            }
            const float * logits = llama_get_logits_ith(ctx, i);
            if (!logits) {
                continue;
            }
            const int next_token = tokens[pos + i + 1];
            const double ls = log_softmax_token(n_vocab, logits, next_token);
            nll_sum -= ls;
            n_eval++;
        }

        llama_batch_free(batch);
        pos += n_tokens;
    }

    return {nll_sum, n_eval};
}

// Run a single benchmark: prefill → compact → eval → PPL.
static bench_result run_bench(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        const bench_config & cfg,
        const std::string & method,
        float ratio,
        int run_idx) {

    bench_result result = {};
    result.method  = method;
    result.ratio   = ratio;
    result.no_beta = cfg.no_beta;
    result.no_cv   = cfg.no_cv;
    result.run     = run_idx;

    const int n_ctx = llama_n_ctx(ctx);
    const int n_tokens = std::min((int) tokens.size(), n_ctx);

    // Determine prefix and eval split.
    uint32_t n_prefix = cfg.prefix_tokens > 0
        ? std::min(cfg.prefix_tokens, (uint32_t)(n_tokens - 1))
        : (uint32_t)(n_tokens / 2);
    uint32_t n_eval = cfg.eval_tokens > 0
        ? std::min(cfg.eval_tokens, (uint32_t)(n_tokens - n_prefix))
        : (uint32_t)(n_tokens - n_prefix);

    if (n_prefix < 4 || n_eval < 2) {
        LOG_ERR("insufficient tokens: prefix=%u eval=%u\n", n_prefix, n_eval);
        result.ppl = -1.0;
        return result;
    }

    uint32_t target = std::max(1u, (uint32_t)(n_prefix / ratio));
    result.prefix_tokens = n_prefix;
    result.target_tokens = target;
    result.eval_tokens   = n_eval;

    // Clear KV cache for fresh run.
    auto * mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);

    // Step 1: Prefill.
    int n_prefilled = prefill(ctx, tokens, n_prefix);
    if ((uint32_t) n_prefilled < n_prefix) {
        LOG_ERR("prefill incomplete: %d / %u\n", n_prefilled, n_prefix);
        result.ppl = -1.0;
        return result;
    }

    // Step 2: Compact (unless baseline).
    auto t_compact_start = std::chrono::steady_clock::now();

    if (method != "baseline") {
        auto * kv = get_kv_cache(ctx);
        if (!kv || !kv->supports_compaction()) {
            LOG_ERR("compaction not supported for this model/context\n");
            result.ppl = -1.0;
            return result;
        }

        const llama_seq_id seq_id = 0;
        const llama_pos live_suffix_pos0 = (llama_pos) n_prefix;
        llama_kv_compact_pipeline_stats stats = {};
        bool compact_ok = false;

        if (method == "select" || cfg.no_beta || cfg.evict_only) {
            compact_ok = kv->compacted_prefix_select_from_live_kv(
                seq_id, target, live_suffix_pos0, &stats);
        } else if (method == "solver") {
            compact_ok = kv->compacted_prefix_fit_from_live_kv(
                seq_id, target, live_suffix_pos0, &stats, 0,
                cfg.max_queries, cfg.nnls_iters, cfg.lambda);
        } else if (method == "omp") {
            compact_ok = kv->compacted_prefix_omp_from_live_kv(
                seq_id, target, live_suffix_pos0, &stats, 0,
                cfg.max_queries, cfg.nnls_iters, cfg.lambda);
        } else if (method == "nonuniform") {
            compact_ok = kv->compacted_prefix_nonuniform_from_live_kv(
                seq_id, target, live_suffix_pos0, &stats, 0,
                cfg.max_queries, cfg.nnls_iters, cfg.lambda);
        } else if (method == "chunked") {
            compact_ok = kv->compacted_prefix_chunked_from_live_kv(
                seq_id, target, live_suffix_pos0, &stats, 0,
                cfg.max_queries, cfg.nnls_iters, cfg.lambda);
        } else {
            LOG_ERR("unknown method: %s\n", method.c_str());
            result.ppl = -1.0;
            return result;
        }

        if (!compact_ok) {
            LOG_ERR("compaction failed: method=%s ratio=%.1f target=%u\n",
                    method.c_str(), ratio, target);
            result.ppl = -1.0;
            return result;
        }

        // Enable compacted prefix execution.
        if (!kv->compacted_prefix_set_execution(seq_id, true)) {
            LOG_ERR("failed to enable compacted prefix execution\n");
            result.ppl = -1.0;
            return result;
        }
    }

    auto t_compact_end = std::chrono::steady_clock::now();
    result.time_compact_ms = std::chrono::duration<double, std::milli>(
        t_compact_end - t_compact_start).count();

    // Step 3: Evaluate remaining tokens for PPL.
    auto t_eval_start = std::chrono::steady_clock::now();

    auto [nll_sum, n_eval_actual] = eval_ppl(ctx, tokens, n_prefix, n_prefix + n_eval);

    auto t_eval_end = std::chrono::steady_clock::now();
    result.time_eval_ms = std::chrono::duration<double, std::milli>(
        t_eval_end - t_eval_start).count();

    if (n_eval_actual > 0) {
        result.nll = nll_sum / n_eval_actual;
        result.ppl = std::exp(result.nll);
    } else {
        result.ppl = -1.0;
        result.nll = -1.0;
    }

    return result;
}

// Parse comma-separated floats.
static std::vector<float> parse_float_list(const std::string & s) {
    std::vector<float> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(std::stof(item));
    }
    return result;
}

// Parse comma-separated strings.
static std::vector<std::string> parse_string_list(const std::string & s) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(item);
    }
    return result;
}

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s -m model.gguf -f input.txt [options]\n", prog);
    fprintf(stderr, "\nRequired:\n");
    fprintf(stderr, "  -m, --model FILE          Model file path\n");
    fprintf(stderr, "  -f, --file FILE           Input text file (e.g., wikitext-2)\n");
    fprintf(stderr, "\nCompaction options:\n");
    fprintf(stderr, "  --ratios LIST             Comma-separated compression ratios (default: 2,5,10,20,50)\n");
    fprintf(stderr, "  --methods LIST            Comma-separated methods (default: baseline,select,solver)\n");
    fprintf(stderr, "  --prefix-tokens N         Prefix token count (default: half of context)\n");
    fprintf(stderr, "  --eval-tokens N           Eval token count (default: remaining)\n");
    fprintf(stderr, "  --max-queries N           Max queries for solver (default: 256)\n");
    fprintf(stderr, "  --nnls-iters N            NNLS iterations (default: 2)\n");
    fprintf(stderr, "  --lambda F                Ridge lambda (default: 1e-6)\n");
    fprintf(stderr, "  --n-runs N                Runs per configuration (default: 1)\n");
    fprintf(stderr, "\nAblation:\n");
    fprintf(stderr, "  --no-beta                 Force zero-beta (selection only)\n");
    fprintf(stderr, "  --no-cv                   Keep original V values (skip C_v fitting)\n");
    fprintf(stderr, "  --evict-only              Evict-based: keep first N, drop rest\n");
    fprintf(stderr, "\nOutput:\n");
    fprintf(stderr, "  --csv FILE                Output CSV file\n");
    fprintf(stderr, "  -c, --ctx-size N          Context size (default: 4096)\n");
    fprintf(stderr, "  -ngl N                    GPU layers (default: 99)\n");
}

int main(int argc, char ** argv) {
    common_params params;
    bench_config cfg;

    // Defaults.
    params.n_ctx     = 4096;
    params.n_batch   = 512;
    params.n_gpu_layers = 99;
    cfg.ratios  = {2.0f, 5.0f, 10.0f, 20.0f, 50.0f};
    cfg.methods = {"baseline", "select", "solver"};

    // Check env vars.
    const char * env_no_beta = std::getenv("LLAMA_COMPACT_NO_BETA");
    const char * env_no_cv   = std::getenv("LLAMA_COMPACT_NO_CV");
    if (env_no_beta && std::string(env_no_beta) == "1") {
        cfg.no_beta = true;
    }
    if (env_no_cv && std::string(env_no_cv) == "1") {
        cfg.no_cv = true;
    }

    // Parse args.
    std::string input_file;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.model.path = argv[++i];
        } else if ((arg == "-f" || arg == "--file") && i + 1 < argc) {
            input_file = argv[++i];
        } else if (arg == "--ratios" && i + 1 < argc) {
            cfg.ratios = parse_float_list(argv[++i]);
        } else if (arg == "--methods" && i + 1 < argc) {
            cfg.methods = parse_string_list(argv[++i]);
        } else if (arg == "--csv" && i + 1 < argc) {
            cfg.csv_path = argv[++i];
        } else if (arg == "--prefix-tokens" && i + 1 < argc) {
            cfg.prefix_tokens = std::atoi(argv[++i]);
        } else if (arg == "--eval-tokens" && i + 1 < argc) {
            cfg.eval_tokens = std::atoi(argv[++i]);
        } else if (arg == "--max-queries" && i + 1 < argc) {
            cfg.max_queries = std::atoi(argv[++i]);
        } else if (arg == "--nnls-iters" && i + 1 < argc) {
            cfg.nnls_iters = std::atoi(argv[++i]);
        } else if (arg == "--lambda" && i + 1 < argc) {
            cfg.lambda = std::stof(argv[++i]);
        } else if (arg == "--n-runs" && i + 1 < argc) {
            cfg.n_runs = std::atoi(argv[++i]);
        } else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) {
            params.n_ctx = std::atoi(argv[++i]);
        } else if (arg == "-ngl" && i + 1 < argc) {
            params.n_gpu_layers = std::atoi(argv[++i]);
        } else if (arg == "--no-beta") {
            cfg.no_beta = true;
        } else if (arg == "--no-cv") {
            cfg.no_cv = true;
        } else if (arg == "--evict-only") {
            cfg.evict_only = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (params.model.path.empty() || input_file.empty()) {
        fprintf(stderr, "error: -m and -f are required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Read input text.
    std::string text;
    {
        std::ifstream ifs(input_file);
        if (!ifs.is_open()) {
            fprintf(stderr, "error: cannot open input file: %s\n", input_file.c_str());
            return 1;
        }
        std::ostringstream ss;
        ss << ifs.rdbuf();
        text = ss.str();
    }

    if (text.empty()) {
        fprintf(stderr, "error: input file is empty\n");
        return 1;
    }

    // Initialize llama.
    llama_backend_init();

    // Load model.
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "error: failed to load model: %s\n", params.model.path.c_str());
        return 1;
    }

    // Create context.
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = params.n_ctx;
    ctx_params.n_batch = params.n_batch;
    // Disable flash attention for compacted prefix compatibility.
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Tokenize input.
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(text.size() + 16);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
                                   tokens.data(), tokens.size(), true, false);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
                                   tokens.data(), tokens.size(), true, false);
    }
    tokens.resize(n_tokens);

    LOG_INF("model: %s\n", params.model.path.c_str());
    LOG_INF("tokens: %d, context: %d\n", n_tokens, llama_n_ctx(ctx));
    LOG_INF("methods: ");
    for (const auto & m : cfg.methods) {
        LOG_INF("%s ", m.c_str());
    }
    LOG_INF("\nratios: ");
    for (float r : cfg.ratios) {
        LOG_INF("%.1f ", r);
    }
    LOG_INF("\n");
    if (cfg.no_beta) LOG_INF("ablation: --no-beta\n");
    if (cfg.no_cv)   LOG_INF("ablation: --no-cv\n");
    if (cfg.evict_only) LOG_INF("ablation: --evict-only\n");

    // Run benchmarks.
    std::vector<bench_result> results;

    for (int run = 0; run < cfg.n_runs; ++run) {
        for (const auto & method : cfg.methods) {
            if (method == "baseline") {
                // Baseline: evaluate without compaction at ratio 1.0.
                bench_result r = run_bench(ctx, tokens, cfg, "baseline", 1.0f, run);
                LOG_INF("run=%d method=baseline ppl=%.4f nll=%.6f eval=%u t_eval=%.1fms\n",
                        run, r.ppl, r.nll, r.eval_tokens, r.time_eval_ms);
                results.push_back(r);
                continue;
            }
            for (float ratio : cfg.ratios) {
                bench_result r = run_bench(ctx, tokens, cfg, method, ratio, run);
                LOG_INF("run=%d method=%s ratio=%.1f ppl=%.4f nll=%.6f "
                        "prefix=%u target=%u eval=%u t_compact=%.1fms t_eval=%.1fms\n",
                        run, method.c_str(), ratio, r.ppl, r.nll,
                        r.prefix_tokens, r.target_tokens, r.eval_tokens,
                        r.time_compact_ms, r.time_eval_ms);
                results.push_back(r);
            }
        }
    }

    // Output CSV.
    FILE * csv_out = stdout;
    FILE * csv_file = nullptr;
    if (!cfg.csv_path.empty()) {
        csv_file = fopen(cfg.csv_path.c_str(), "w");
        if (!csv_file) {
            fprintf(stderr, "error: cannot open CSV file: %s\n", cfg.csv_path.c_str());
        } else {
            csv_out = csv_file;
        }
    }

    fprintf(csv_out, "run,method,ratio,prefix_tokens,target_tokens,eval_tokens,"
                     "ppl,nll,time_compact_ms,time_eval_ms,no_beta,no_cv\n");
    for (const auto & r : results) {
        fprintf(csv_out, "%d,%s,%.1f,%u,%u,%u,%.6f,%.6f,%.1f,%.1f,%d,%d\n",
                r.run, r.method.c_str(), r.ratio,
                r.prefix_tokens, r.target_tokens, r.eval_tokens,
                r.ppl, r.nll, r.time_compact_ms, r.time_eval_ms,
                r.no_beta ? 1 : 0, r.no_cv ? 1 : 0);
    }

    if (csv_file) {
        fclose(csv_file);
        LOG_INF("CSV written to: %s\n", cfg.csv_path.c_str());
    }

    // Cleanup.
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
