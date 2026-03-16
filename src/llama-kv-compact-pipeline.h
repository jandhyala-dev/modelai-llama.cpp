#pragma once

#include "llama.h"

class llama_kv_cache;

struct llama_kv_compact_pipeline_stats {
    double query_generation_time_ms = 0.0;
    double solver_time_ms = 0.0;
    double k_extraction_time_ms = 0.0;
    double attention_score_time_ms = 0.0;
    double selection_time_ms = 0.0;
    double v_extraction_time_ms = 0.0;
    double kv_write_time_ms = 0.0;
    double total_time_ms = 0.0;
    uint32_t n_prefix_tokens = 0;
    uint32_t n_selected_tokens = 0;
    float mean_partition_sum_relative_error = 0.0f;  // Phase 8: mean residual across all heads
};

// Full solver pipeline: selection + beta fitting + V fitting.
// Uses cache-keys-as-queries for scoring and solver optimization.
bool llama_kv_compact_fit_from_live_kv(
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats = nullptr,
        llama_pos p0 = 0,
        uint32_t max_queries = 256,
        int nnls_iters = 2,
        float lambda = 1e-6f);

// Selection-only pipeline: keeps earliest target_tokens positions from the
// prefix with original K/V values and zero beta.  No solver fitting is
// performed.  This mode avoids quality degradation caused by surrogate
// cache-key queries on GQA architectures and serves as the default v0
// compaction path until real query extraction is implemented.
bool llama_kv_compact_select_from_live_kv(
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats = nullptr,
        llama_pos p0 = 0);

// OMP selection pipeline: uses Orthogonal Matching Pursuit (Algorithm 1,
// arXiv:2602.16284 §3.2) per-head for greedy residual-based key selection,
// then aggregates across heads via vote counting for a global selection set.
// After global selection, per-head beta is refit via NNLS and V is fitted
// via least-squares, identical to the full solver pipeline.
//
// This produces better selection quality than topk on aggregated attention
// scores because OMP greedily minimizes the partition-function residual.
bool llama_kv_compact_omp_from_live_kv(
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats = nullptr,
        llama_pos p0 = 0,
        uint32_t max_queries = 256,
        int nnls_iters = 2,
        float lambda = 1e-6f);

// Solver pipeline with nonuniform per-head budgets (Algorithm 4).
//
// Instead of a single global top-k, each head gets an entropy-proportional
// budget. The union of per-head selections forms the compacted position set.
// Heads that did NOT select a union position receive beta=-inf for it,
// contributing zero attention weight through the softmax.
//
// This is the paper's most impactful ablation — heads with peaky attention
// get more tokens while diffuse heads tolerate aggressive compression.
bool llama_kv_compact_nonuniform_from_live_kv(
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats = nullptr,
        llama_pos p0 = 0,
        uint32_t max_queries = 256,
        int nnls_iters = 2,
        float lambda = 1e-6f,
        uint32_t min_per_head = 4);

// Chunked compaction: split prefix into chunks, compact each independently,
// merge results. Enables compaction of contexts larger than ~8K tokens
// where single-block solver has memory/precision issues.
//
// Reference: arXiv:2602.16284 Section 3.5
bool llama_kv_compact_chunked_from_live_kv(
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats = nullptr,
        llama_pos p0 = 0,
        uint32_t max_queries = 256,
        int nnls_iters = 2,
        float lambda = 1e-6f,
        uint32_t chunk_size = 8192);

// Approximate on-policy pipeline (Section 3.1 / 4.2).
//
// Two-pass approach that captures the first-order effect of layer
// interaction without O(n_layers) forward passes:
//   Pass 1: Compact with K-as-Q surrogates (standard solver pipeline)
//   Pass 2: Generate continuation tokens from the compacted model,
//           capture Q via cb_eval, re-run solver with real Q
//
// This is simpler and cheaper than true sequential on-policy compaction
// but captures most of the quality benefit.
bool llama_kv_compact_on_policy_from_live_kv(
        struct llama_context * ctx,
        llama_kv_cache & kv,
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats = nullptr,
        llama_pos p0 = 0,
        uint32_t max_queries = 256,
        int nnls_iters = 2,
        float lambda = 1e-6f,
        uint32_t n_generate_q = 128);

// Self-study pipelines are declared in llama-kv-compact-self-study.h
// (llama_kv_compact_self_study_from_live_kv and
// llama_kv_compact_chunked_self_study_from_live_kv) — require llama_context
// for Q-capture generation via cb_eval.  The chunked variant (Phase 6)
// splits long prefixes into chunks for per-chunk scoring + selection,
// then runs full-prefix solver with globally merged selections.
