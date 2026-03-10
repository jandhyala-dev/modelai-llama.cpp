#include "arg.h"
#include "common.h"
#include "llama.h"
#include "src/llama-context.h"
#include "src/llama-kv-cache.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct compacted_layer_snapshot {
    std::vector<uint8_t> k_data;
    std::vector<float>   beta_data;
    std::vector<uint8_t> v_data;
    uint32_t n_compacted_tokens = 0;
};

struct compacted_seq_snapshot {
    uint32_t logical_token_count = 0;
    llama_pos live_suffix_pos0 = -1;
    std::vector<llama_pos> logical_positions;
    bool execution_enabled = false;
    std::vector<compacted_layer_snapshot> layers;
};

size_t token_bytes(ggml_type type, uint32_t n_embd_head) {
    return ggml_row_size(type, n_embd_head);
}

void fill_token_bytes(
        std::vector<uint8_t> & data,
        uint32_t n_head_kv,
        uint32_t n_tokens,
        size_t bytes_per_token) {
    for (uint32_t head = 0; head < n_head_kv; ++head) {
        for (uint32_t token = 0; token < n_tokens; ++token) {
            const uint8_t value = uint8_t(head * 32 + token);
            const size_t offset = (size_t(head) * n_tokens + token) * bytes_per_token;
            std::fill_n(data.begin() + offset, bytes_per_token, value);
        }
    }
}

void fill_beta(std::vector<float> & beta, uint32_t n_head_kv, uint32_t n_tokens) {
    for (uint32_t head = 0; head < n_head_kv; ++head) {
        for (uint32_t token = 0; token < n_tokens; ++token) {
            beta[size_t(head) * n_tokens + token] = float(head * 100 + token);
        }
    }
}

int fail(const std::string & message) {
    std::fprintf(stderr, "test-state-restore-compacted-prefix: %s\n", message.c_str());
    return 1;
}

compacted_seq_snapshot snapshot_seq(const llama_compacted_prefix_store::sequence_state & seq) {
    compacted_seq_snapshot snap;
    snap.logical_token_count = seq.logical_token_count;
    snap.live_suffix_pos0 = seq.live_suffix_pos0;
    snap.logical_positions = seq.logical_positions;
    snap.execution_enabled = seq.is_execution_enabled();
    snap.layers.resize(seq.layers.size());
    for (size_t i = 0; i < seq.layers.size(); ++i) {
        snap.layers[i].k_data = seq.layers[i].k_data;
        snap.layers[i].beta_data = seq.layers[i].beta_data;
        snap.layers[i].v_data = seq.layers[i].v_data;
        snap.layers[i].n_compacted_tokens = seq.layers[i].n_compacted_tokens;
    }
    return snap;
}

bool equal_snapshot(const compacted_seq_snapshot & lhs, const llama_compacted_prefix_store::sequence_state & rhs) {
    if (lhs.logical_token_count != rhs.logical_token_count ||
        lhs.live_suffix_pos0 != rhs.live_suffix_pos0 ||
        lhs.logical_positions != rhs.logical_positions ||
        lhs.execution_enabled != rhs.is_execution_enabled() ||
        lhs.layers.size() != rhs.layers.size()) {
        return false;
    }

    for (size_t i = 0; i < lhs.layers.size(); ++i) {
        if (lhs.layers[i].n_compacted_tokens != rhs.layers[i].n_compacted_tokens ||
            lhs.layers[i].k_data != rhs.layers[i].k_data ||
            lhs.layers[i].beta_data != rhs.layers[i].beta_data ||
            lhs.layers[i].v_data != rhs.layers[i].v_data) {
            return false;
        }
    }

    return true;
}

} // namespace

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 1234;
    params.kv_unified = true;
    params.n_parallel = 1;
    params.n_ctx = 256;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    common_init();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        return fail("failed to initialize model/context");
    }

    GGML_UNUSED(model);

    auto * kv = dynamic_cast<llama_kv_cache *>(ctx->get_memory());
    if (kv == nullptr) {
        return fail("test requires a standard llama_kv_cache memory backend");
    }

    std::vector<llama_token> tokens(32, 1);
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], i, {0}, i + 1 == tokens.size());
    }

    if (llama_decode(ctx, batch)) {
        llama_batch_free(batch);
        return fail("failed to decode seed prompt");
    }

    if (!kv->compacted_prefix_configure(0, tokens.size(), { 0, 2, 4 }, 5)) {
        llama_batch_free(batch);
        return fail("failed to configure compacted prefix");
    }
    if (!kv->compacted_prefix_set_execution(0, true)) {
        llama_batch_free(batch);
        return fail("failed to enable compacted-prefix execution");
    }

    auto * seq = kv->get_compacted_prefix()->get_seq(0);
    if (seq == nullptr) {
        llama_batch_free(batch);
        return fail("configured compacted-prefix sequence missing");
    }

    for (auto & layer : seq->layers) {
        fill_beta(layer.beta_data, layer.layout.n_head_kv, layer.n_compacted_tokens);
        if (!layer.k_data.empty()) {
            fill_token_bytes(layer.k_data, layer.layout.n_head_kv, layer.n_compacted_tokens, token_bytes(layer.layout.type_k, layer.layout.n_embd_head_k));
        }
        if (!layer.v_data.empty()) {
            fill_token_bytes(layer.v_data, layer.layout.n_head_kv, layer.n_compacted_tokens, token_bytes(layer.layout.type_v, layer.layout.n_embd_head_v));
        }
    }

    const compacted_seq_snapshot before = snapshot_seq(*seq);

    std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 0));
    const size_t ncopy = llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 0);
    if (ncopy != seq_state.size()) {
        llama_batch_free(batch);
        return fail("failed to save sequence state");
    }

    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 0, -1, -1);

    if (kv->compacted_prefix_enabled(0) || kv->compacted_prefix_execution_enabled(0)) {
        llama_batch_free(batch);
        return fail("clearing the sequence should clear compacted-prefix state");
    }

    const size_t nset = llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0);
    if (nset != seq_state.size()) {
        llama_batch_free(batch);
        return fail("failed to restore sequence state");
    }

    const auto * restored = kv->get_compacted_prefix()->get_seq(0);
    if (restored == nullptr || !restored->enabled) {
        llama_batch_free(batch);
        return fail("restored compacted-prefix state missing");
    }
    if (!equal_snapshot(before, *restored)) {
        llama_batch_free(batch);
        return fail("restored compacted-prefix payload does not match saved state");
    }
    if (!kv->compacted_prefix_set_execution(0, false) || kv->compacted_prefix_execution_enabled(0)) {
        llama_batch_free(batch);
        return fail("failed to disable compacted-prefix execution before continuation decode");
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));

    const auto next_token = llama_sampler_sample(smpl, ctx, -1);
    common_batch_clear(batch);
    common_batch_add(batch, next_token, (int) tokens.size(), {0}, true);

    if (llama_decode(ctx, batch)) {
        llama_sampler_free(smpl);
        llama_batch_free(batch);
        return fail("failed to decode after restoring compacted-prefix state");
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);

    return 0;
}
