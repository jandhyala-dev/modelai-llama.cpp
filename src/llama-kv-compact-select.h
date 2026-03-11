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
