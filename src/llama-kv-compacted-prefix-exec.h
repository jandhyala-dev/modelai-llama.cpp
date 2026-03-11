#pragma once

#include "llama-batch.h"
#include "llama-hparams.h"
#include "llama-kv-compacted-prefix.h"

#include "ggml.h"

struct llama_compacted_prefix_exec_candidate {
    llama_seq_id seq_id = -1;
    uint32_t n_tokens = 0;
    bool zero_beta = false; // true when all beta values are zero (selection-only pipeline)
};

bool llama_compacted_prefix_can_execute(
        llama_seq_id seq_id,
        const llama_compacted_prefix_store::sequence_state * state,
        const llama_ubatch & ubatch,
        llama_compacted_prefix_exec_candidate * out = nullptr);

void llama_compacted_prefix_set_input_mask(
        ggml_tensor * dst,
        const llama_compacted_prefix_store::sequence_state & state,
        const llama_ubatch & ubatch,
        const llama_hparams & hparams,
        bool causal_attn);

void llama_compacted_prefix_set_input_k(
        ggml_tensor * dst,
        const llama_compacted_prefix_store::layer_storage & layer);

void llama_compacted_prefix_set_input_v(
        ggml_tensor * dst,
        const llama_compacted_prefix_store::layer_storage & layer);

void llama_compacted_prefix_set_input_beta(
        ggml_tensor * dst,
        const llama_compacted_prefix_store::layer_storage & layer,
        uint32_t n_head);
