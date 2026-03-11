#pragma once

#include "llama-kv-compact-solver.h"

#include <cstdint>
#include <vector>

void llama_kv_compact_accumulate_attention_scores(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & keys,
        std::vector<float> & scores_inout);

std::vector<uint32_t> llama_kv_compact_select_topk(
        const std::vector<float> & scores,
        uint32_t t);

// OMP key selection options.
// Reference: omp.py class OMPCompaction.__init__() lines 138-206
struct llama_kv_compact_omp_opts {
    uint32_t k_choice      = 1;     // keys per iteration (1=standard, 4=fast)
    uint32_t nnls_interval = 1;     // refit every N iters (1=always, 2=fast)
    float    lower_bound   = 1e-12f;
};

// OMP key selection with periodic NNLS refit.
//
// Greedy selection of t keys that best approximate the attention partition
// function. At each step selects the key most correlated with the residual
// between the target partition sum and the current approximation.
//
// Reference: Algorithm 1, arXiv:2602.16284 Section 3.2
// Reference impl: compaction/algorithms/omp.py lines 478-718
//
// Returns sorted position indices.
// beta_out receives the NNLS-derived log-weights.
std::vector<uint32_t> llama_kv_compact_select_omp(
        const llama_kv_compact_matrix & queries,
        const llama_kv_compact_matrix & keys,
        uint32_t t,
        const llama_kv_compact_omp_opts & opts,
        std::vector<float> & beta_out);
