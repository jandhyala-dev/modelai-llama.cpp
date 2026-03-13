// Long-context benchmark for KV cache compaction (6b-15 / 6b-15b).
//
// Measures compaction quality and decode throughput across context lengths
// (4K-128K+) to find the throughput crossover point.  Supports QuALITY MC
// evaluation (2,086 questions) and LongHealth MC evaluation (400 questions,
// 60K-token patient records) for paper-aligned accuracy measurement.
//
// Pipelines: baseline, select, solver, omp, self_study (6b-15b)
//
// Usage:
//   # Single pipeline run:
//   PIPELINE=select RATIO=8 ./test-kv-compact-longctx -m model.gguf -c 16384
//
//   # QuALITY MC evaluation (full 2,086 questions):
//   PIPELINE=select RATIO=4 QUALITY_EVAL=1 \
//       ./test-kv-compact-longctx -m model.gguf -c 8192
//
//   # LongHealth MC evaluation (400 questions, 5-option):
//   PIPELINE=select RATIO=4 LONGHEALTH_EVAL=1 \
//       ./test-kv-compact-longctx -m model.gguf -c 65536
//
//   # Real SEC filing prefill:
//   SEC_TEXT_DIR=tests/data/sec-10k-benchmark PIPELINE=select RATIO=4 \
//       ./test-kv-compact-longctx -m model.gguf -c 32768
//
//   # Write CSV artifact:
//   ARTIFACT=results.csv PIPELINE=baseline \
//       ./test-kv-compact-longctx -m model.gguf -c 4096

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "src/llama-context.h"
#include "src/llama-kv-compact-pipeline.h"
#include "src/llama-kv-compact-solver.h"
#include "src/llama-kv-compact-self-study.h"
#include "src/llama-kv-cache.h"
#include "src/llama-kv-cache-iswa.h"

#include "vendor/nlohmann/json.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

int fail(const std::string & message) {
    std::fprintf(stderr, "test-kv-compact-longctx: %s\n", message.c_str());
    return 1;
}

// ---------------------------------------------------------------------------
// Model registry (static, keyed by model_name substring)
// ---------------------------------------------------------------------------

struct model_info {
    const char * name_substr;
    float        params_b;
    const char * architecture;
};

static const model_info model_registry[] = {
    {"Qwen3-30B-A3B",              30.0f, "MoE GQA 8:1"},
    {"Qwen3-14B",                  14.0f, "Dense GQA 5:1"},
    {"Qwen3-8B",                    8.0f, "Dense GQA"},
    {"Qwen2.5-14B",                14.0f, "Dense GQA"},
    {"DeepSeek-R1-Distill-Qwen-14B", 14.0f, "Dense GQA"},
    {"Gemma-3-12B",                12.0f, "iSWA mixed"},
};

static const model_info * lookup_model(const std::string & name) {
    for (const auto & m : model_registry) {
        if (name.find(m.name_substr) != std::string::npos) {
            return &m;
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Workload registry
// ---------------------------------------------------------------------------

static const char * workload_id_from_params(uint64_t n_params) {
    if (n_params < 500000000)  return "W1";
    if (n_params <= 10000000000ULL) return "W2";
    return "W3";
}

static const char * workload_name_from_id(const char * wid) {
    if (std::strcmp(wid, "W1") == 0) return "smoke";
    if (std::strcmp(wid, "W2") == 0) return "standard";
    return "production";
}

// ---------------------------------------------------------------------------
// Threshold table: <workload_id>:<metric>:<ratio> -> value
// Calibrated from 6b-13 production results (Qwen3-14B, Qwen3-30B-A3B).
// ---------------------------------------------------------------------------

static float lookup_threshold(const char * /*workload_id*/, int ratio) {
    // Logit cosine thresholds — looser at higher compression.
    if (ratio <= 2) return 0.95f;
    if (ratio <= 4) return 0.90f;
    return 0.85f;
}

// ---------------------------------------------------------------------------
// Support classification — encodes the measured envelope.
// ---------------------------------------------------------------------------

struct support_classification {
    std::string level;   // "supported" | "experimental" | "blocked"
    std::string reason;
};

static support_classification classify_support(
        const std::string & pipeline,
        int n_ctx,
        int ratio) {
    // Self-study: blocked until quality is proven.
    if (pipeline == "self_study") {
        return {"blocked", "self_study_quality_unproven"};
    }

    // OMP: experimental (insufficient benchmark evidence).
    if (pipeline == "omp") {
        return {"experimental", "insufficient_benchmark_evidence"};
    }

    // Solver: experimental (insufficient benchmark evidence).
    if (pipeline == "solver") {
        return {"experimental", "insufficient_benchmark_evidence"};
    }

    // Baseline: always supported (it's the reference).
    if (pipeline == "baseline") {
        return {"supported", ""};
    }

    // Select pipeline: context + ratio dependent.
    if (pipeline == "select") {
        // 4K at high ratios: experimental.
        if (n_ctx <= 4096 && ratio > 4) {
            return {"experimental", "4k_high_ratio_quality_unproven"};
        }
        // 32K: experimental (throughput regresses).
        if (n_ctx >= 32768) {
            return {"experimental", "32k_throughput_regression"};
        }
        // 8K-16K at any ratio, or 4K at ratio <= 4: supported.
        return {"supported", ""};
    }

    return {"experimental", "unknown_pipeline"};
}

// ---------------------------------------------------------------------------
// QuALITY MC entry
// ---------------------------------------------------------------------------

struct quality_entry {
    std::string article;
    std::string question;
    std::vector<std::string> options;
    int answer;
    bool hard;
};

static std::vector<quality_entry> load_quality_data(const std::string & path) {
    std::vector<quality_entry> entries;
    std::ifstream ifs(path);
    if (!ifs.is_open()) return entries;

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        try {
            auto j = nlohmann::json::parse(line);
            quality_entry e;
            e.article  = j.at("article").get<std::string>();
            e.question = j.at("question").get<std::string>();
            for (const auto & opt : j.at("options")) {
                e.options.push_back(opt.get<std::string>());
            }
            e.answer = j.at("answer").get<int>();
            e.hard   = j.value("hard", false);
            entries.push_back(std::move(e));
        } catch (...) {
            // skip malformed lines
        }
    }
    return entries;
}

// ---------------------------------------------------------------------------
// LongHealth MC entry (5-option: A-E)
// ---------------------------------------------------------------------------

struct longhealth_entry {
    std::string patient_text;  // concatenated patient medical records
    std::string question;
    std::vector<std::string> options;  // 5 options (A-E)
    int answer;  // 0-4 index of correct answer
    std::string patient_id;
};

static std::vector<longhealth_entry> load_longhealth_data(const std::string & path) {
    std::vector<longhealth_entry> entries;
    std::ifstream ifs(path);
    if (!ifs.is_open()) return entries;

    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(content);
    } catch (...) {
        return entries;
    }

    // Format: {"patient_01": {"texts": {"text_0": "...", ...}, "questions": [...]}, ...}
    for (auto it = root.begin(); it != root.end(); ++it) {
        const std::string & patient_id = it.key();
        const auto & patient = it.value();

        // Concatenate all patient texts.
        std::string patient_text;
        if (patient.contains("texts")) {
            const auto & texts = patient["texts"];
            for (auto tit = texts.begin(); tit != texts.end(); ++tit) {
                if (!patient_text.empty()) patient_text += "\n\n";
                patient_text += tit.value().get<std::string>();
            }
        }
        if (patient_text.empty()) continue;

        // Parse questions.
        if (!patient.contains("questions")) continue;
        for (const auto & q : patient["questions"]) {
            longhealth_entry e;
            e.patient_id = patient_id;
            e.patient_text = patient_text;
            e.question = q.value("question", "");
            if (e.question.empty()) continue;

            // 5 answer options.
            const char * answer_keys[] = {"answer_a", "answer_b", "answer_c", "answer_d", "answer_e"};
            for (int i = 0; i < 5; ++i) {
                e.options.push_back(q.value(answer_keys[i], ""));
            }

            // Map "correct" field to answer index.
            std::string correct_text = q.value("correct", "");
            e.answer = -1;
            for (int i = 0; i < 5; ++i) {
                if (e.options[i] == correct_text) {
                    e.answer = i;
                    break;
                }
            }
            if (e.answer < 0) continue;  // skip if correct answer not found

            entries.push_back(std::move(e));
        }
    }
    return entries;
}

// ---------------------------------------------------------------------------
// Decode burst — returns elapsed ms, or -1 on error
// ---------------------------------------------------------------------------

static double decode_burst(llama_context * ctx, int n_tokens, int start_pos) {
    const llama_token tok = 1;
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

// ---------------------------------------------------------------------------
// Batched prefill
// ---------------------------------------------------------------------------

static double prefill_tokens(llama_context * ctx,
                             const std::vector<llama_token> & tokens) {
    const int n = (int) tokens.size();
    const int n_batch = (int) llama_n_batch(ctx);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ) {
        const int end = std::min(i + n_batch, n);
        llama_batch batch = llama_batch_init(end - i, 0, 1);
        for (int j = i; j < end; ++j) {
            common_batch_add(batch, tokens[j], j, {0}, j + 1 == n);
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return -1.0;
        }
        llama_batch_free(batch);
        i = end;
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Compaction dispatch
// ---------------------------------------------------------------------------

static bool run_compaction(llama_kv_cache * kv, llama_context * ctx,
                           const std::string & pipeline,
                           int target, int live_suffix_pos0,
                           llama_kv_compact_pipeline_stats * stats,
                           double * self_study_gen_ms) {
    *self_study_gen_ms = 0.0;

    if (pipeline == "omp") {
        return kv->compacted_prefix_omp_from_live_kv(0, target, live_suffix_pos0, stats);
    }
    if (pipeline == "solver") {
        return kv->compacted_prefix_fit_from_live_kv(0, target, live_suffix_pos0, stats);
    }
    if (pipeline == "self_study") {
        llama_kv_compact_self_study_config cfg;
        cfg.n_generate = 256;
        cfg.max_queries_per_kv_head = 1024;
        llama_kv_compact_self_study_stats ss_stats = {};
        bool ok = kv->compacted_prefix_self_study_from_live_kv(
            ctx, 0, target, live_suffix_pos0, cfg, &ss_stats);
        if (stats) {
            stats->query_generation_time_ms = ss_stats.generation_time_ms + ss_stats.q_capture_time_ms;
            stats->solver_time_ms           = ss_stats.solver_time_ms;
            stats->n_prefix_tokens          = ss_stats.n_prefix_tokens;
            stats->n_selected_tokens        = ss_stats.n_selected_tokens;
        }
        *self_study_gen_ms = ss_stats.generation_time_ms;
        return ok;
    }
    // default: select
    return kv->compacted_prefix_select_from_live_kv(0, target, live_suffix_pos0, stats);
}

// ---------------------------------------------------------------------------
// LongHealth MC scoring — single-forward-pass logit comparison (5-option)
// ---------------------------------------------------------------------------

struct longhealth_score {
    int correct;
    int total;
};

static longhealth_score run_longhealth_eval(
        llama_context * ctx, llama_model * model, llama_kv_cache * kv,
        const std::vector<longhealth_entry> & entries,
        const std::string & pipeline_name,
        int compression_ratio, int n_ctx,
        bool do_compact) {

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const uint32_t n_vocab = llama_vocab_n_tokens(vocab);

    // Look up 5 answer token IDs: try space-prefixed first, then bare.
    llama_token tok_answers[5];
    const char * labels_sp[] = {" A", " B", " C", " D", " E"};
    const char * labels_bare[] = {"A", "B", "C", "D", "E"};
    for (int i = 0; i < 5; ++i) {
        auto toks = common_tokenize(ctx, labels_sp[i], false, false);
        if (toks.size() == 1) {
            tok_answers[i] = toks[0];
        } else {
            toks = common_tokenize(ctx, labels_bare[i], false, false);
            tok_answers[i] = toks.empty() ? -1 : toks[0];
        }
    }

    int correct = 0;
    int total = 0;
    const int n_lh_entries = (int) entries.size();

    for (const auto & entry : entries) {
        // Progress output every 10 questions.
        if (total > 0 && total % 10 == 0) {
            std::printf("  LongHealth %d/%d: correct=%d (%.1f%%)\n",
                        total, n_lh_entries, correct,
                        total > 0 ? 100.0f * correct / total : 0.0f);
            std::fflush(stdout);
        }

        // Build prompt: patient text + question + 5 options + "Answer:"
        std::string prompt_str = entry.patient_text + "\n\nQuestion: " + entry.question + "\n\n";
        for (int i = 0; i < 5; ++i) {
            char label = 'A' + i;
            prompt_str += label;
            prompt_str += ") ";
            prompt_str += entry.options[i];
            prompt_str += "\n";
        }
        prompt_str += "\nAnswer:";

        auto prompt_tokens = common_tokenize(ctx, prompt_str, true, false);

        // Skip if prompt doesn't fit in context.
        if ((int) prompt_tokens.size() >= n_ctx - 1) {
            continue;
        }

        // Clear KV and prefill.
        llama_memory_clear(ctx->get_memory(), true);

        const double pm = prefill_tokens(ctx, prompt_tokens);
        if (pm < 0) continue;

        const int prefix_len = (int) prompt_tokens.size();

        // Optionally compact.
        if (do_compact && pipeline_name != "baseline") {
            const int live_suffix_pos0 = (int)(prefix_len * 0.8);
            const int target = live_suffix_pos0 / compression_ratio;
            if (target < 1) continue;

            llama_kv_compact_pipeline_stats stats = {};
            double ss_gen_ms = 0.0;
            if (!run_compaction(kv, ctx, pipeline_name, target, live_suffix_pos0, &stats, &ss_gen_ms)) {
                continue;
            }
            if (!kv->compacted_prefix_set_execution(0, true)) continue;
            if (!kv->compacted_prefix_reclaim_live_kv(0)) continue;
        }

        // Extract logits at the last position (-1 = last output token).
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) continue;

        // Score: argmax over 5 answer tokens.
        int best_idx = 0;
        float best_logit = -1e30f;
        for (int i = 0; i < 5; ++i) {
            if (tok_answers[i] < 0 || tok_answers[i] >= (llama_token) n_vocab) continue;
            float l = logits[tok_answers[i]];
            if (l > best_logit) {
                best_logit = l;
                best_idx = i;
            }
        }

        if (best_idx == entry.answer) {
            ++correct;
        }
        ++total;
    }

    return {correct, total};
}

// ---------------------------------------------------------------------------
// CSV result row
// ---------------------------------------------------------------------------

struct longctx_result {
    // identity
    int         schema_version;
    std::string run_id;
    std::string workload_id;
    std::string workload_name;
    std::string dataset_id;
    std::string model_name;
    float       model_params_b;
    std::string architecture;
    std::string quantization;
    std::string backend;
    std::string flash_mode;
    std::string pipeline;
    int         n_ctx;
    int         compression_ratio;

    // KV / token counts
    int         prefix_tokens;
    int         compactable_tokens;
    int         live_suffix_tokens;
    int         compacted_tokens;
    int         continuation_tokens;

    // latency / throughput
    double      prefill_ms;
    double      prefill_tok_s;
    double      compaction_time_ms;
    double      query_generation_time_ms;
    double      solver_time_ms;
    double      first_token_ms;
    double      baseline_decode_tok_s;
    double      compacted_decode_tok_s;
    double      in_run_throughput_delta_pct;

    // KV bytes (nullable)
    uint32_t    active_n_kv;
    std::string allocated_kv_bytes;   // "" when unavailable
    std::string reclaimed_kv_bytes;   // "" when unavailable

    // quality
    float       logit_cosine;
    std::string task_metric_name;
    std::string task_metric_value;
    std::string quality_correct;
    std::string quality_total;
    std::string quality_accuracy;
    std::string quality_baseline_accuracy;

    // longhealth
    std::string longhealth_correct;
    std::string longhealth_total;
    std::string longhealth_accuracy;
    std::string longhealth_baseline_accuracy;

    // threshold
    std::string threshold_name;
    float       threshold_value;
    bool        pass;

    // support classification
    std::string support_level;    // "supported", "experimental", "blocked"
    std::string support_reason;   // human-readable reason for classification
    bool        throughput_pass;  // true if compacted >= baseline throughput

    // safety
    bool        fallback_used;
    std::string fallback_reason;
    bool        crash;
    std::string error_text;
    std::string artifact_path;
};

static void write_csv_header(FILE * f) {
    std::fprintf(f,
        "schema_version,run_id,workload_id,workload_name,dataset_id,"
        "model_name,model_params_b,architecture,quantization,backend,flash_mode,"
        "pipeline,n_ctx,prefix_tokens,compactable_tokens,live_suffix_tokens,"
        "compression_ratio,compacted_tokens,continuation_tokens,"
        "prefill_ms,prefill_tok_s,compaction_time_ms,query_generation_time_ms,"
        "solver_time_ms,first_token_ms,baseline_decode_tok_s,compacted_decode_tok_s,"
        "in_run_throughput_delta_pct,active_n_kv,allocated_kv_bytes,reclaimed_kv_bytes,"
        "logit_cosine,task_metric_name,task_metric_value,"
        "quality_correct,quality_total,quality_accuracy,quality_baseline_accuracy,"
        "longhealth_correct,longhealth_total,longhealth_accuracy,longhealth_baseline_accuracy,"
        "threshold_name,threshold_value,pass,fallback_used,fallback_reason,"
        "crash,error_text,artifact_path,"
        "support_level,support_reason,throughput_pass\n");
}

static void write_csv_row(FILE * f, const longctx_result & r) {
    std::fprintf(f,
        "%d,%s,%s,%s,%s,"                     // schema_version..dataset_id
        "%s,%.1f,%s,%s,%s,%s,"                // model_name..flash_mode
        "%s,%d,%d,%d,%d,"                     // pipeline..live_suffix_tokens
        "%d,%d,%d,"                           // compression_ratio..continuation_tokens
        "%.1f,%.1f,%.1f,%.1f,"               // prefill_ms..query_generation_time_ms
        "%.1f,%.1f,%.1f,%.1f,"               // solver_time_ms..compacted_decode_tok_s
        "%.4f,%u,%s,%s,"                      // in_run_throughput_delta_pct..reclaimed_kv_bytes
        "%.6f,%s,%s,"                         // logit_cosine..task_metric_value
        "%s,%s,%s,%s,"                        // quality_correct..quality_baseline_accuracy
        "%s,%s,%s,%s,"                        // longhealth_correct..longhealth_baseline_accuracy
        "%s,%.4f,%s,%s,%s,"                   // threshold_name..fallback_reason
        "%s,%s,%s,"                           // crash..artifact_path
        "%s,%s,%s\n",                         // support_level..throughput_pass
        r.schema_version, r.run_id.c_str(),
        r.workload_id.c_str(), r.workload_name.c_str(), r.dataset_id.c_str(),
        r.model_name.c_str(), r.model_params_b,
        r.architecture.c_str(), r.quantization.c_str(),
        r.backend.c_str(), r.flash_mode.c_str(),
        r.pipeline.c_str(), r.n_ctx, r.prefix_tokens,
        r.compactable_tokens, r.live_suffix_tokens,
        r.compression_ratio, r.compacted_tokens, r.continuation_tokens,
        r.prefill_ms, r.prefill_tok_s, r.compaction_time_ms,
        r.query_generation_time_ms,
        r.solver_time_ms, r.first_token_ms,
        r.baseline_decode_tok_s, r.compacted_decode_tok_s,
        r.in_run_throughput_delta_pct, r.active_n_kv,
        r.allocated_kv_bytes.c_str(), r.reclaimed_kv_bytes.c_str(),
        r.logit_cosine,
        r.task_metric_name.c_str(), r.task_metric_value.c_str(),
        r.quality_correct.c_str(), r.quality_total.c_str(),
        r.quality_accuracy.c_str(), r.quality_baseline_accuracy.c_str(),
        r.longhealth_correct.c_str(), r.longhealth_total.c_str(),
        r.longhealth_accuracy.c_str(), r.longhealth_baseline_accuracy.c_str(),
        r.threshold_name.c_str(), r.threshold_value,
        r.pass ? "true" : "false",
        r.fallback_used ? "true" : "false", r.fallback_reason.c_str(),
        r.crash ? "true" : "false", r.error_text.c_str(),
        r.artifact_path.c_str(),
        r.support_level.c_str(), r.support_reason.c_str(),
        r.throughput_pass ? "true" : "false");
}

// ---------------------------------------------------------------------------
// Load text from file (for real SEC filing prefills)
// ---------------------------------------------------------------------------

static std::string load_text_file(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return "";
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string text(sz, '\0');
    std::fread(&text[0], 1, sz, f);
    std::fclose(f);
    return text;
}

// Pick the best-fit SEC filing from SEC_TEXT_DIR for the given token count.
// Scans *.txt files and picks the one whose size (chars/4) is closest to
// and >= min_tokens.  Falls back to the largest file if none is big enough.
static std::string pick_sec_filing(const std::string & dir, size_t min_tokens) {
    std::string best_path;
    long best_size = 0;
    std::string largest_path;
    long largest_size = 0;

    // Simple directory scan via popen (portable enough for test code).
    std::string cmd = "ls -1 " + dir + "/*.txt 2>/dev/null";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) return "";
    char buf[1024];
    while (std::fgets(buf, sizeof(buf), p)) {
        std::string path(buf);
        while (!path.empty() && (path.back() == '\n' || path.back() == '\r'))
            path.pop_back();
        FILE * f = std::fopen(path.c_str(), "rb");
        if (!f) continue;
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        std::fclose(f);
        if (sz > largest_size) { largest_size = sz; largest_path = path; }
        long approx_tokens = sz / 4;
        if (approx_tokens >= (long)min_tokens) {
            if (best_path.empty() || sz < best_size) {
                best_size = sz;
                best_path = path;
            }
        }
    }
    pclose(p);
    return best_path.empty() ? largest_path : best_path;
}

// ---------------------------------------------------------------------------
// Build text prompt — uses real SEC filing if SEC_TEXT_DIR is set,
// otherwise falls back to synthetic repeated paragraph.
// ---------------------------------------------------------------------------

static std::vector<llama_token> build_real_text_prompt(llama_context * ctx,
                                                       size_t min_tokens) {
    // Try real SEC filing text first.
    const char * sec_dir = std::getenv("SEC_TEXT_DIR");
    if (sec_dir && sec_dir[0]) {
        std::string path = pick_sec_filing(sec_dir, min_tokens);
        if (!path.empty()) {
            std::string text = load_text_file(path);
            if (!text.empty()) {
                std::vector<llama_token> tokens = common_tokenize(ctx, text, true, false);
                if (tokens.size() >= min_tokens) {
                    LLAMA_LOG_INFO("prefill: using SEC filing %s (%zu tokens)\n",
                                   path.c_str(), tokens.size());
                    tokens.resize(min_tokens);
                    return tokens;
                }
                // Filing too short — pad with repetition of its own text.
                LLAMA_LOG_INFO("prefill: SEC filing %s has %zu tokens, padding to %zu\n",
                               path.c_str(), tokens.size(), min_tokens);
                std::vector<llama_token> padded;
                padded.reserve(min_tokens);
                while (padded.size() < min_tokens) {
                    padded.insert(padded.end(), tokens.begin(), tokens.end());
                }
                padded.resize(min_tokens);
                return padded;
            }
        }
    }

    // Fallback: synthetic repeated paragraph.
    const std::string para =
        "The quarterly letter reviewed liquidity, capital allocation, recurring revenue, customer retention, and operating leverage. "
        "Management discussed cash flow discipline, pricing pressure, inventory turns, software adoption, and regional demand. "
        "Analysts compared the margin profile to prior quarters and noted that guidance depended on enterprise renewals, deferred revenue conversion, and foreign exchange stability. "
        "The board approved a share repurchase program of up to two billion dollars over the next twelve months, subject to market conditions and regulatory approval. "
        "Revenue from cloud services grew twenty-three percent year-over-year, driven by enterprise migration and expanded API usage across financial services clients. "
        "Operating expenses as a percentage of revenue declined by one hundred and forty basis points, reflecting disciplined headcount management and infrastructure efficiency gains. ";

    const std::vector<llama_token> para_tokens = common_tokenize(ctx, para, true, false);
    std::vector<llama_token> tokens;
    tokens.reserve(min_tokens);
    while (tokens.size() < min_tokens) {
        tokens.insert(tokens.end(), para_tokens.begin(), para_tokens.end());
    }
    tokens.resize(min_tokens);
    return tokens;
}

// ---------------------------------------------------------------------------
// QuALITY MC scoring — single-forward-pass logit comparison
// ---------------------------------------------------------------------------

struct quality_score {
    int correct;
    int total;
};

static quality_score run_quality_eval(
        llama_context * ctx, llama_model * model, llama_kv_cache * kv,
        const std::vector<quality_entry> & entries,
        const std::string & pipeline_name,
        int compression_ratio, int n_ctx,
        bool do_compact) {

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const uint32_t n_vocab = llama_vocab_n_tokens(vocab);

    // Look up answer token IDs: try space-prefixed first, then bare.
    llama_token tok_answers[4];
    const char * labels_sp[] = {" A", " B", " C", " D"};
    const char * labels_bare[] = {"A", "B", "C", "D"};
    for (int i = 0; i < 4; ++i) {
        auto toks = common_tokenize(ctx, labels_sp[i], false, false);
        if (toks.size() == 1) {
            tok_answers[i] = toks[0];
        } else {
            toks = common_tokenize(ctx, labels_bare[i], false, false);
            tok_answers[i] = toks.empty() ? -1 : toks[0];
        }
    }

    int correct = 0;
    int total = 0;
    const int n_entries = (int) entries.size();

    for (const auto & entry : entries) {
        // Progress output every 10 questions.
        if (total > 0 && total % 10 == 0) {
            std::printf("  QuALITY %d/%d: correct=%d (%.1f%%)\n",
                        total, n_entries, correct,
                        total > 0 ? 100.0f * correct / total : 0.0f);
            std::fflush(stdout);
        }

        // Build prompt: article + question + options + "Answer:"
        std::string prompt_str = entry.article + "\n\nQuestion: " + entry.question + "\n\n";
        for (int i = 0; i < 4; ++i) {
            char label = 'A' + i;
            prompt_str += label;
            prompt_str += ") ";
            prompt_str += entry.options[i];
            prompt_str += "\n";
        }
        prompt_str += "\nAnswer:";

        auto prompt_tokens = common_tokenize(ctx, prompt_str, true, false);

        // Skip if prompt doesn't fit in context.
        if ((int) prompt_tokens.size() >= n_ctx - 1) {
            continue;
        }

        // Clear KV and prefill.
        llama_memory_clear(ctx->get_memory(), true);

        const double pm = prefill_tokens(ctx, prompt_tokens);
        if (pm < 0) continue;

        const int prefix_len = (int) prompt_tokens.size();

        // Optionally compact.
        if (do_compact && pipeline_name != "baseline") {
            const int live_suffix_pos0 = (int)(prefix_len * 0.8);
            const int target = live_suffix_pos0 / compression_ratio;
            if (target < 1) continue;

            llama_kv_compact_pipeline_stats stats = {};
            double ss_gen_ms = 0.0;
            if (!run_compaction(kv, ctx, pipeline_name, target, live_suffix_pos0, &stats, &ss_gen_ms)) {
                continue;
            }
            if (!kv->compacted_prefix_set_execution(0, true)) continue;
            if (!kv->compacted_prefix_reclaim_live_kv(0)) continue;
        }

        // Extract logits at the last position (-1 = last output token).
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) continue;

        // Score: argmax over answer tokens.
        int best_idx = 0;
        float best_logit = -1e30f;
        for (int i = 0; i < 4; ++i) {
            if (tok_answers[i] < 0 || tok_answers[i] >= (llama_token) n_vocab) continue;
            float l = logits[tok_answers[i]];
            if (l > best_logit) {
                best_logit = l;
                best_idx = i;
            }
        }

        if (best_idx == entry.answer) {
            ++correct;
        }
        ++total;
    }

    return {correct, total};
}

} // namespace

// ===========================================================================
// main
// ===========================================================================

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
        return fail("requires llama_kv_cache or llama_kv_cache_iswa memory backend");
    }

    // --- Environment configuration ---
    const char * env_pipeline = std::getenv("PIPELINE");
    const std::string pipeline = env_pipeline ? env_pipeline : "select";
    if (pipeline != "baseline" && pipeline != "select" &&
        pipeline != "solver"   && pipeline != "omp" &&
        pipeline != "self_study") {
        return fail("PIPELINE must be 'baseline', 'select', 'solver', 'omp', or 'self_study'");
    }

    const char * env_ratio = std::getenv("RATIO");
    const int ratio = env_ratio ? std::atoi(env_ratio) : 4;
    if (ratio < 1) return fail("RATIO must be >= 1");

    const char * env_artifact = std::getenv("ARTIFACT");
    const char * env_run_id   = std::getenv("RUN_ID");
    const char * env_quality  = std::getenv("QUALITY_EVAL");
    const bool   do_quality   = env_quality && std::atoi(env_quality) > 0;

    // Quality data path — default to full 2,086-question dataset.
    const char * env_quality_path = std::getenv("QUALITY_DATA");
    const std::string quality_path = env_quality_path
        ? env_quality_path : "tests/data/quality-validation-full.jsonl";

    // Max questions per eval pass (0 = unlimited). Default 100 for inline
    // benchmarks — full 2,086 takes ~45 min per run on an 8B model.
    // Set QUALITY_LIMIT=0 for exhaustive evaluation.
    const char * env_quality_limit = std::getenv("QUALITY_LIMIT");
    const int quality_limit = env_quality_limit ? std::atoi(env_quality_limit) : 100;

    // LongHealth evaluation (5-option MC, 60K-token patient records).
    const char * env_longhealth = std::getenv("LONGHEALTH_EVAL");
    const bool do_longhealth = env_longhealth && std::atoi(env_longhealth) > 0;
    const char * env_longhealth_path = std::getenv("LONGHEALTH_DATA");
    const std::string longhealth_path = env_longhealth_path
        ? env_longhealth_path : "tests/data/longhealth-benchmark-v5.json";

    // Max LongHealth questions per eval (0 = unlimited). Default 50.
    const char * env_lh_limit = std::getenv("LONGHEALTH_LIMIT");
    const int longhealth_limit = env_lh_limit ? std::atoi(env_lh_limit) : 50;

    const int n_ctx = (int) llama_n_ctx(ctx);
    const int continuation_tokens = 16;
    const uint64_t n_params = llama_model_n_params(model);

    // --- Model identification ---
    std::string model_name = params.model.path;
    {
        auto pos = model_name.find_last_of('/');
        if (pos != std::string::npos) model_name = model_name.substr(pos + 1);
        pos = model_name.find(".gguf");
        if (pos != std::string::npos) model_name = model_name.substr(0, pos);
    }

    const model_info * minfo = lookup_model(model_name);
    float  model_params_b = minfo ? minfo->params_b : (float)(n_params / 1e9);
    std::string architecture = minfo ? minfo->architecture : "unknown";

    // Extract quantization from model name (e.g., "Q4_K_M").
    std::string quantization = "unknown";
    {
        auto pos = model_name.rfind('-');
        if (pos != std::string::npos && pos + 1 < model_name.size() &&
            model_name[pos + 1] == 'Q') {
            quantization = model_name.substr(pos + 1);
        }
        // Also try underscore-separated (e.g., Model_Q4_K_M).
        if (quantization == "unknown") {
            pos = model_name.find("_Q");
            if (pos != std::string::npos) {
                quantization = model_name.substr(pos + 1);
            }
        }
    }

    // Backend detection.
    std::string backend_name = "cpu";
#ifdef GGML_USE_METAL
    backend_name = "metal";
#elif defined(GGML_USE_CUDA)
    backend_name = "cuda";
#elif defined(GGML_USE_VULKAN)
    backend_name = "vulkan";
#elif defined(GGML_USE_SYCL)
    backend_name = "sycl";
#elif defined(GGML_USE_HIP)
    backend_name = "hip";
#endif

    // Flash mode: V0 = always off.
    const std::string flash_mode = "off";

    // Workload classification.
    const char * wid = workload_id_from_params(n_params);
    const char * wname = workload_name_from_id(wid);

    // Dataset identification.
    const char * env_sec_dir = std::getenv("SEC_TEXT_DIR");
    const bool use_real_sec = env_sec_dir && env_sec_dir[0];
    std::string dataset_id;
    if (do_quality && do_longhealth) {
        dataset_id = "quality+longhealth";
    } else if (do_quality) {
        dataset_id = "quality-validation";
    } else if (do_longhealth) {
        dataset_id = "longhealth";
    } else if (use_real_sec) {
        dataset_id = "sec-10k-filings";
    } else {
        dataset_id = "synthetic-sec";
    }

    // Run ID.
    std::string run_id = "local";
    if (env_run_id) {
        run_id = env_run_id;
    }

    // Threshold.
    const float threshold_value = lookup_threshold(wid, ratio);
    char threshold_name_buf[64];
    std::snprintf(threshold_name_buf, sizeof(threshold_name_buf),
                  "%s:logit_cosine:%d", wid, ratio);
    const std::string threshold_name = threshold_name_buf;

    // Support classification.
    const auto sc = classify_support(pipeline, n_ctx, ratio);

    // Artifact path.
    std::string artifact_path;
    if (env_artifact) {
        artifact_path = env_artifact;
    }

    // --- Sizing ---
    const int prefix_tokens = std::min((int)(n_ctx * 0.8), n_ctx - 128);
    const int live_suffix_pos0 = (int)(prefix_tokens * 0.8);
    const int compactable_tokens = live_suffix_pos0;
    const int live_suffix_tokens = prefix_tokens - live_suffix_pos0;

    if (prefix_tokens < 64) {
        return fail("n_ctx too small for meaningful benchmark");
    }

    // --- Print config ---
    std::printf("=== KV Compaction Long-Context Benchmark ===\n");
    std::printf("model:      %s (%.1fB, %s)\n", model_name.c_str(), model_params_b, architecture.c_str());
    std::printf("backend:    %s\n", backend_name.c_str());
    std::printf("pipeline:   %s\n", pipeline.c_str());
    std::printf("n_ctx:      %d\n", n_ctx);
    std::printf("ratio:      %dx\n", ratio);
    std::printf("prefix:     %d tokens (compactable: %d, live suffix: %d)\n",
                prefix_tokens, compactable_tokens, live_suffix_tokens);
    std::printf("workload:   %s (%s)\n", wid, wname);
    if (do_quality) {
        std::printf("quality:    QuALITY MC eval enabled (%s, limit=%d)\n",
                    quality_path.c_str(), quality_limit);
    }
    if (do_longhealth) {
        std::printf("longhealth: LongHealth MC eval enabled (%s, limit=%d)\n",
                    longhealth_path.c_str(), longhealth_limit);
    }
    std::printf("\n");

    // --- Initialize result ---
    longctx_result result = {};
    result.schema_version = 2;
    result.run_id = run_id;
    result.workload_id = wid;
    result.workload_name = wname;
    result.dataset_id = dataset_id;
    result.model_name = model_name;
    result.model_params_b = model_params_b;
    result.architecture = architecture;
    result.quantization = quantization;
    result.backend = backend_name;
    result.flash_mode = flash_mode;
    result.pipeline = pipeline;
    result.n_ctx = n_ctx;
    result.prefix_tokens = prefix_tokens;
    result.compactable_tokens = compactable_tokens;
    result.live_suffix_tokens = live_suffix_tokens;
    result.continuation_tokens = continuation_tokens;
    result.allocated_kv_bytes = "";
    result.reclaimed_kv_bytes = "";
    result.fallback_used = false;
    result.crash = false;

    // ===================================================================
    // BASELINE pipeline: no compaction
    // ===================================================================
    if (pipeline == "baseline") {
        result.compression_ratio = 1;
        result.compacted_tokens = 0;

        auto prompt = build_real_text_prompt(ctx, prefix_tokens);
        const double pm = prefill_tokens(ctx, prompt);
        if (pm < 0) {
            result.crash = true;
            result.error_text = "prefill failed";
            result.pass = false;
            goto emit;
        }
        result.prefill_ms = pm;
        result.prefill_tok_s = prefix_tokens / (pm / 1000.0);

        // Baseline decode.
        {
            const auto t0 = std::chrono::steady_clock::now();
            const double burst_ms = decode_burst(ctx, continuation_tokens, prefix_tokens);
            const auto t1 = std::chrono::steady_clock::now();
            if (burst_ms < 0) {
                result.crash = true;
                result.error_text = "baseline decode failed";
                result.pass = false;
                goto emit;
            }
            result.first_token_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / continuation_tokens;
            result.baseline_decode_tok_s = continuation_tokens / (burst_ms / 1000.0);
            result.compacted_decode_tok_s = result.baseline_decode_tok_s;
        }

        result.compaction_time_ms = 0.0;
        result.query_generation_time_ms = 0.0;
        result.solver_time_ms = 0.0;
        result.in_run_throughput_delta_pct = 0.0;
        result.active_n_kv = prefix_tokens + continuation_tokens;
        result.logit_cosine = 1.0f;
        result.threshold_name = "";
        result.threshold_value = 0.0f;
        result.pass = true;
        result.support_level = "supported";
        result.support_reason = "";
        result.throughput_pass = true;

        // QuALITY baseline accuracy.
        if (do_quality) {
            auto entries = load_quality_data(quality_path);
            if (quality_limit > 0 && (int) entries.size() > quality_limit) {
                entries.resize(quality_limit);
            }
            if (!entries.empty()) {
                auto score = run_quality_eval(ctx, model, kv, entries,
                                             "baseline", 1, n_ctx, false);
                result.task_metric_name = "quality_accuracy";
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%d", score.correct);
                result.quality_correct = buf;
                std::snprintf(buf, sizeof(buf), "%d", score.total);
                result.quality_total = buf;
                if (score.total > 0) {
                    float acc = (float) score.correct / score.total;
                    std::snprintf(buf, sizeof(buf), "%.4f", acc);
                    result.quality_accuracy = buf;
                    result.quality_baseline_accuracy = buf;
                    result.task_metric_value = buf;
                    std::printf("QuALITY baseline: %d/%d = %.1f%%\n",
                                score.correct, score.total, acc * 100.0f);
                }
            }
        }

        // LongHealth baseline accuracy (60K-token patient records, 5-option MC).
        if (do_longhealth) {
            auto lh_entries = load_longhealth_data(longhealth_path);
            if (longhealth_limit > 0 && (int) lh_entries.size() > longhealth_limit) {
                lh_entries.resize(longhealth_limit);
            }
            if (!lh_entries.empty()) {
                auto lh_score = run_longhealth_eval(ctx, model, kv, lh_entries,
                                                     "baseline", 1, n_ctx, false);
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%d", lh_score.correct);
                result.longhealth_correct = buf;
                std::snprintf(buf, sizeof(buf), "%d", lh_score.total);
                result.longhealth_total = buf;
                if (lh_score.total > 0) {
                    float acc = (float) lh_score.correct / lh_score.total;
                    std::snprintf(buf, sizeof(buf), "%.4f", acc);
                    result.longhealth_accuracy = buf;
                    result.longhealth_baseline_accuracy = buf;
                    // If no quality eval, use longhealth as task metric.
                    if (!do_quality) {
                        result.task_metric_name = "longhealth_accuracy";
                        result.task_metric_value = buf;
                    }
                    std::printf("LongHealth baseline: %d/%d = %.1f%%\n",
                                lh_score.correct, lh_score.total, acc * 100.0f);
                }
            }
        }

        std::printf("baseline: prefill=%.1fms (%.1f tok/s) decode=%.1f tok/s\n",
                    result.prefill_ms, result.prefill_tok_s,
                    result.baseline_decode_tok_s);
        goto emit;
    }

    // ===================================================================
    // COMPACTED pipelines: select, solver, omp, self_study
    // ===================================================================
    {
        const int target = compactable_tokens / ratio;
        if (target < 1) {
            result.crash = true;
            result.error_text = "compression target < 1";
            result.pass = false;
            goto emit;
        }

        result.compression_ratio = ratio;
        result.compacted_tokens = target;

        auto prompt = build_real_text_prompt(ctx, prefix_tokens);

        // Prefill.
        const double pm = prefill_tokens(ctx, prompt);
        if (pm < 0) {
            result.crash = true;
            result.error_text = "prefill failed";
            result.pass = false;
            goto emit;
        }
        result.prefill_ms = pm;
        result.prefill_tok_s = prefix_tokens / (pm / 1000.0);

        // Save state for baseline measurement.
        std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 0));
        if (llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
            result.crash = true;
            result.error_text = "state save failed";
            result.pass = false;
            goto emit;
        }

        // Baseline logits for cosine similarity.
        {
            if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
                result.crash = true;
                result.error_text = "state restore for baseline logits failed";
                result.pass = false;
                goto emit;
            }
            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, 1, prefix_tokens, {0}, true);
            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                result.crash = true;
                result.error_text = "baseline logit decode failed";
                result.pass = false;
                goto emit;
            }
            llama_batch_free(batch);
        }
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * bl = llama_get_logits_ith(ctx, 0);
        std::vector<float> baseline_logits(bl, bl + n_vocab);

        // Baseline decode tok/s (self-measured within the same invocation).
        {
            if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
                result.crash = true;
                result.error_text = "state restore for baseline decode failed";
                result.pass = false;
                goto emit;
            }
            const double burst_ms = decode_burst(ctx, continuation_tokens, prefix_tokens);
            if (burst_ms < 0) {
                result.crash = true;
                result.error_text = "baseline decode burst failed";
                result.pass = false;
                goto emit;
            }
            result.baseline_decode_tok_s = continuation_tokens / (burst_ms / 1000.0);
        }

        // Restore state and compact.
        if (llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) != seq_state.size()) {
            result.crash = true;
            result.error_text = "state restore for compaction failed";
            result.pass = false;
            goto emit;
        }

        llama_kv_compact_pipeline_stats pstats = {};
        double self_study_gen_ms = 0.0;
        const auto tc0 = std::chrono::steady_clock::now();
        if (!run_compaction(kv, ctx, pipeline, target, live_suffix_pos0, &pstats, &self_study_gen_ms)) {
            result.crash = true;
            result.error_text = "compaction failed";
            result.pass = false;
            goto emit;
        }
        if (!kv->compacted_prefix_set_execution(0, true)) {
            result.crash = true;
            result.error_text = "set_execution failed";
            result.pass = false;
            goto emit;
        }
        if (!kv->compacted_prefix_reclaim_live_kv(0)) {
            result.crash = true;
            result.error_text = "reclaim failed";
            result.pass = false;
            goto emit;
        }
        const auto tc1 = std::chrono::steady_clock::now();
        result.compaction_time_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count();
        result.query_generation_time_ms = pstats.query_generation_time_ms;
        result.solver_time_ms = pstats.solver_time_ms;

        // Compacted logit cosine.
        {
            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, 1, prefix_tokens, {0}, true);
            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                result.crash = true;
                result.error_text = "compacted logit decode failed";
                result.pass = false;
                goto emit;
            }
            const float * cl = llama_get_logits_ith(ctx, 0);
            std::vector<float> compacted_logits(cl, cl + n_vocab);
            result.logit_cosine = llama_kv_compact_cosine_similarity(baseline_logits, compacted_logits);
            llama_batch_free(batch);
        }

        // Compacted decode tok/s + first token timing.
        {
            const auto td0 = std::chrono::steady_clock::now();
            const double burst_ms = decode_burst(ctx, continuation_tokens, prefix_tokens + 1);
            const auto td1 = std::chrono::steady_clock::now();
            if (burst_ms < 0) {
                result.crash = true;
                result.error_text = "compacted decode burst failed";
                result.pass = false;
                goto emit;
            }
            result.compacted_decode_tok_s = continuation_tokens / (burst_ms / 1000.0);
            result.first_token_ms = std::chrono::duration<double, std::milli>(td1 - td0).count() / continuation_tokens;
        }

        result.active_n_kv = kv->compacted_prefix_active_n_kv(0);

        // Throughput delta (emitted by runner, not inferred).
        if (result.baseline_decode_tok_s > 0) {
            result.in_run_throughput_delta_pct =
                ((result.compacted_decode_tok_s - result.baseline_decode_tok_s) /
                 result.baseline_decode_tok_s) * 100.0;
        }

        // Threshold check.
        result.threshold_name = threshold_name;
        result.threshold_value = threshold_value;
        result.pass = result.logit_cosine >= threshold_value;
        result.support_level = sc.level;
        result.support_reason = sc.reason;
        // INFORMATIONAL ONLY — do NOT use to gate result.pass.
        result.throughput_pass = (result.in_run_throughput_delta_pct > -60.0);

        // QuALITY MC evaluation (compacted).
        if (do_quality) {
            auto entries = load_quality_data(quality_path);
            if (quality_limit > 0 && (int) entries.size() > quality_limit) {
                entries.resize(quality_limit);
            }
            if (!entries.empty()) {
                // Compacted accuracy.
                auto score_c = run_quality_eval(ctx, model, kv, entries,
                                                pipeline, ratio, n_ctx, true);
                // Baseline accuracy (for comparison).
                auto score_b = run_quality_eval(ctx, model, kv, entries,
                                                "baseline", 1, n_ctx, false);

                result.task_metric_name = "quality_accuracy";
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%d", score_c.correct);
                result.quality_correct = buf;
                std::snprintf(buf, sizeof(buf), "%d", score_c.total);
                result.quality_total = buf;
                if (score_c.total > 0) {
                    float acc_c = (float) score_c.correct / score_c.total;
                    std::snprintf(buf, sizeof(buf), "%.4f", acc_c);
                    result.quality_accuracy = buf;
                    result.task_metric_value = buf;
                    std::printf("QuALITY compacted: %d/%d = %.1f%%\n",
                                score_c.correct, score_c.total, acc_c * 100.0f);
                }
                if (score_b.total > 0) {
                    float acc_b = (float) score_b.correct / score_b.total;
                    std::snprintf(buf, sizeof(buf), "%.4f", acc_b);
                    result.quality_baseline_accuracy = buf;
                    std::printf("QuALITY baseline:  %d/%d = %.1f%%\n",
                                score_b.correct, score_b.total, acc_b * 100.0f);
                }
            }
        }

        // LongHealth MC evaluation (compacted, 5-option, 60K-token patient records).
        if (do_longhealth) {
            auto lh_entries = load_longhealth_data(longhealth_path);
            if (longhealth_limit > 0 && (int) lh_entries.size() > longhealth_limit) {
                lh_entries.resize(longhealth_limit);
            }
            if (!lh_entries.empty()) {
                // Compacted accuracy.
                auto lh_score_c = run_longhealth_eval(ctx, model, kv, lh_entries,
                                                       pipeline, ratio, n_ctx, true);
                // Baseline accuracy (for comparison).
                auto lh_score_b = run_longhealth_eval(ctx, model, kv, lh_entries,
                                                       "baseline", 1, n_ctx, false);

                char buf[32];
                std::snprintf(buf, sizeof(buf), "%d", lh_score_c.correct);
                result.longhealth_correct = buf;
                std::snprintf(buf, sizeof(buf), "%d", lh_score_c.total);
                result.longhealth_total = buf;
                if (lh_score_c.total > 0) {
                    float acc_c = (float) lh_score_c.correct / lh_score_c.total;
                    std::snprintf(buf, sizeof(buf), "%.4f", acc_c);
                    result.longhealth_accuracy = buf;
                    if (!do_quality) {
                        result.task_metric_name = "longhealth_accuracy";
                        result.task_metric_value = buf;
                    }
                    std::printf("LongHealth compacted: %d/%d = %.1f%%\n",
                                lh_score_c.correct, lh_score_c.total, acc_c * 100.0f);
                }
                if (lh_score_b.total > 0) {
                    float acc_b = (float) lh_score_b.correct / lh_score_b.total;
                    std::snprintf(buf, sizeof(buf), "%.4f", acc_b);
                    result.longhealth_baseline_accuracy = buf;
                    std::printf("LongHealth baseline:  %d/%d = %.1f%%\n",
                                lh_score_b.correct, lh_score_b.total, acc_b * 100.0f);
                }
            }
        }

        std::printf("\n--- %s %dx ---\n", pipeline.c_str(), ratio);
        std::printf("  cosine=%.6f (threshold=%.4f %s)\n",
                    result.logit_cosine, result.threshold_value,
                    result.pass ? "PASS" : "FAIL");
        std::printf("  compact=%.1fms | baseline=%.1f tok/s | compacted=%.1f tok/s | delta=%.2f%%\n",
                    result.compaction_time_ms,
                    result.baseline_decode_tok_s, result.compacted_decode_tok_s,
                    result.in_run_throughput_delta_pct);
        std::printf("  active_n_kv=%u\n", result.active_n_kv);
    }

emit:
    // Write CSV artifact.
    if (env_artifact) {
        // Check if file exists to decide header.
        FILE * fcheck = std::fopen(env_artifact, "r");
        bool exists = fcheck != nullptr;
        if (fcheck) std::fclose(fcheck);

        FILE * f = std::fopen(env_artifact, exists ? "a" : "w");
        if (f) {
            if (!exists) {
                write_csv_header(f);
            }
            write_csv_row(f, result);
            std::fclose(f);
            std::printf("\nartifact %s: %s\n", exists ? "appended" : "written", env_artifact);
        } else {
            std::fprintf(stderr, "warning: could not open artifact file '%s'\n", env_artifact);
        }
    }

    return result.pass ? 0 : 1;
}
