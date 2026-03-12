#include "src/llama-kv-compacted-prefix-exec.h"

#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int fail(const std::string & message) {
    std::cerr << "test-kv-compacted-prefix-exec: " << message << std::endl;
    return 1;
}

bool check(bool cond, const std::string & message, int & rc) {
    if (!cond) {
        rc = fail(message);
        return false;
    }
    return true;
}

float tensor_f32_at(const ggml_tensor * t, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    const auto * base = reinterpret_cast<const uint8_t *>(t->data);
    return *reinterpret_cast<const float *>(base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
}

template<typename Fn>
bool expect_throw(Fn && fn) {
    try {
        fn();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

struct ggml_test_ctx {
    std::vector<uint8_t> buf;
    ggml_context * ctx = nullptr;

    explicit ggml_test_ctx(size_t size) : buf(size) {
        ggml_init_params params = {
            /* .mem_size   = */ buf.size(),
            /* .mem_buffer = */ buf.data(),
            /* .no_alloc   = */ false,
        };
        ctx = ggml_init(params);
    }

    ~ggml_test_ctx() {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

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

llama_ubatch make_ubatch_single(
        std::vector<llama_pos> & pos,
        std::vector<llama_seq_id> & seq_id_unq,
        llama_seq_id seq_id) {
    seq_id_unq = { seq_id };

    llama_ubatch ubatch = {};
    ubatch.n_tokens = pos.size();
    ubatch.pos = pos.data();
    ubatch.n_seqs_unq = seq_id_unq.size();
    ubatch.seq_id_unq = seq_id_unq.data();
    return ubatch;
}

int test_execution_gate() {
    int rc = 0;

    llama_compacted_prefix_store store({
        {
            /* layer_id      = */ 0,
            /* n_head_kv     = */ 2,
            /* n_embd_head_k = */ 4,
            /* n_embd_head_v = */ 4,
            /* type_k        = */ GGML_TYPE_F16,
            /* type_v        = */ GGML_TYPE_F16,
        },
    });

    if (!check(store.configure_seq(4, 16, { 0, 2 }, 4), "configure_seq should succeed", rc)) return rc;

    auto * seq = store.get_seq(4);
    if (seq == nullptr) {
        return fail("configured seq should exist");
    }

    std::vector<llama_pos> pos = { 4, 5 };
    std::vector<llama_seq_id> seq_id_unq;
    auto ubatch = make_ubatch_single(pos, seq_id_unq, 4);

    llama_compacted_prefix_exec_candidate exec = {};
    if (!check(!llama_compacted_prefix_can_execute(4, seq, ubatch, &exec),
               "execution should stay disabled until explicitly enabled",
               rc)) return rc;

    if (!check(store.set_execution(4, true), "execution should enable through the store API", rc)) return rc;
    if (!check(llama_compacted_prefix_can_execute(4, seq, ubatch, &exec),
               "execution should activate for a matching single-sequence ubatch",
               rc)) return rc;
    if (!check(exec.seq_id == 4 && exec.n_tokens == 2, "execution candidate should return seq id and prefix token count", rc)) return rc;

    pos = { 3 };
    ubatch = make_ubatch_single(pos, seq_id_unq, 4);
    if (!check(!llama_compacted_prefix_can_execute(4, seq, ubatch),
               "ubatch positions before live_suffix_pos0 should be rejected",
               rc)) return rc;

    ubatch.n_seqs_unq = 2;
    if (!check(!llama_compacted_prefix_can_execute(4, seq, ubatch),
               "multi-sequence ubatches should be rejected in P3",
               rc)) return rc;

    return rc;
}

int test_materialization_and_mask() {
    int rc = 0;

    llama_compacted_prefix_store store({
        {
            /* layer_id      = */ 0,
            /* n_head_kv     = */ 2,
            /* n_embd_head_k = */ 4,
            /* n_embd_head_v = */ 8,
            /* type_k        = */ GGML_TYPE_F16,
            /* type_v        = */ GGML_TYPE_F16,
        },
    });

    if (!check(store.configure_seq(7, 12, { 2, 4 }, 5), "configure_seq should succeed", rc)) return rc;

    auto * seq = store.get_seq(7);
    if (seq == nullptr) {
        return fail("sequence 7 should exist");
    }
    if (!check(store.set_execution(7, true), "execution should enable through the store API", rc)) return rc;

    auto & layer = seq->layers[0];
    const size_t k_token_bytes = ggml_row_size(layer.layout.type_k, layer.layout.n_embd_head_k);
    const size_t v_token_bytes = ggml_row_size(layer.layout.type_v, layer.layout.n_embd_head_v);

    fill_token_bytes(layer.k_data, layer.layout.n_head_kv, layer.n_compacted_tokens, k_token_bytes);
    fill_token_bytes(layer.v_data, layer.layout.n_head_kv, layer.n_compacted_tokens, v_token_bytes);
    fill_beta(layer.beta_data, layer.layout.n_head_kv, layer.n_compacted_tokens);

    std::vector<llama_pos> pos = { 5 };
    std::vector<llama_seq_id> seq_id_unq;
    auto ubatch = make_ubatch_single(pos, seq_id_unq, 7);

    llama_hparams hparams = {};
    hparams.use_alibi = false;

    ggml_test_ctx tctx(1u << 16);
    if (tctx.ctx == nullptr) {
        return fail("failed to create ggml test context");
    }

    auto * k = ggml_new_tensor_4d(tctx.ctx, GGML_TYPE_F16, 4, 2, 2, 1);
    auto * v = ggml_new_tensor_4d(tctx.ctx, GGML_TYPE_F16, 8, 2, 2, 1);
    auto * b_base = ggml_new_tensor_4d(tctx.ctx, GGML_TYPE_F32, 4, 1, 4, 1);
    auto * m_base = ggml_new_tensor_4d(tctx.ctx, GGML_TYPE_F32, 4, 1, 1, 1);
    auto * b = ggml_view_4d(tctx.ctx, b_base, 2, 1, 4, 1, b_base->nb[1], b_base->nb[2], b_base->nb[3], 0);
    auto * m = ggml_view_4d(tctx.ctx, m_base, 2, 1, 1, 1, m_base->nb[1], m_base->nb[2], m_base->nb[3], 0);

    llama_compacted_prefix_set_input_k(k, layer);
    llama_compacted_prefix_set_input_v(v, layer);
    llama_compacted_prefix_set_input_beta(b, layer, 4);
    llama_compacted_prefix_set_input_mask(m, *seq, ubatch, hparams, /* causal_attn = */ true);

    const auto * k_bytes = reinterpret_cast<const uint8_t *>(k->data);
    const auto * v_bytes = reinterpret_cast<const uint8_t *>(v->data);

    if (!check(k_bytes[0] == 0, "K token 0 / head 0 should copy first token bytes", rc)) return rc;
    if (!check(*(k_bytes + k->nb[2]) == 1, "K token 1 / head 0 should copy second token bytes", rc)) return rc;
    if (!check(*(k_bytes + k->nb[1]) == 32, "K head 1 / token 0 should preserve head-major payload", rc)) return rc;

    if (!check(v_bytes[0] == 0, "V token 0 / head 0 should copy first token bytes", rc)) return rc;
    if (!check(*(v_bytes + v->nb[2]) == 1, "V token 1 / head 0 should copy second token bytes", rc)) return rc;
    if (!check(*(v_bytes + v->nb[1]) == 32, "V head 1 / token 0 should preserve head-major payload", rc)) return rc;

    if (!check(tensor_f32_at(b, 0, 0, 0, 0) == 0.0f && tensor_f32_at(b, 1, 0, 0, 0) == 1.0f,
               "query head 0 should use KV-head 0 beta values",
               rc)) return rc;
    if (!check(tensor_f32_at(b, 0, 0, 1, 0) == 0.0f && tensor_f32_at(b, 1, 0, 1, 0) == 1.0f,
               "query head 1 should repeat KV-head 0 beta values",
               rc)) return rc;
    if (!check(tensor_f32_at(b, 0, 0, 2, 0) == 100.0f,
               "query head 2 should use KV-head 1 beta values",
               rc)) return rc;
    if (!check(tensor_f32_at(b, 1, 0, 3, 0) == 101.0f,
               "query head 3 should repeat KV-head 1 beta values",
               rc)) return rc;

    if (!check(tensor_f32_at(m, 0, 0, 0, 0) == 0.0f && tensor_f32_at(m, 1, 0, 0, 0) == 0.0f,
               "causal mask should allow prefix positions at or before query pos",
               rc)) return rc;

    pos = { 3 };
    ubatch = make_ubatch_single(pos, seq_id_unq, 7);
    llama_compacted_prefix_set_input_mask(m, *seq, ubatch, hparams, /* causal_attn = */ true);
    if (!check(tensor_f32_at(m, 0, 0, 0, 0) == 0.0f && std::isinf(tensor_f32_at(m, 1, 0, 0, 0)) && tensor_f32_at(m, 1, 0, 0, 0) < 0.0f,
               "causal mask should block prefix positions after the current query pos",
               rc)) return rc;

    hparams.use_alibi = true;
    pos = { 6 };
    ubatch = make_ubatch_single(pos, seq_id_unq, 7);
    llama_compacted_prefix_set_input_mask(m, *seq, ubatch, hparams, /* causal_attn = */ false);
    if (!check(tensor_f32_at(m, 0, 0, 0, 0) == -4.0f && tensor_f32_at(m, 1, 0, 0, 0) == -2.0f,
               "alibi mask should use absolute position deltas on the compacted prefix columns",
               rc)) return rc;

    return rc;
}

int test_type_and_shape_guards() {
    int rc = 0;

    llama_compacted_prefix_store store({
        {
            /* layer_id      = */ 0,
            /* n_head_kv     = */ 1,
            /* n_embd_head_k = */ 4,
            /* n_embd_head_v = */ 4,
            /* type_k        = */ GGML_TYPE_F16,
            /* type_v        = */ GGML_TYPE_F16,
        },
    });

    if (!check(store.configure_seq(9, 4, { 0, 2 }, 2), "configure_seq should succeed", rc)) return rc;

    auto * seq = store.get_seq(9);
    if (seq == nullptr) {
        return fail("sequence 9 should exist");
    }
    if (!check(store.set_execution(9, true), "execution should enable through the store API", rc)) return rc;

    ggml_test_ctx tctx(1u << 15);
    auto * wrong_type = ggml_new_tensor_4d(tctx.ctx, GGML_TYPE_F32, 4, 1, 2, 1);
    if (!check(expect_throw([&]() { llama_compacted_prefix_set_input_k(wrong_type, seq->layers[0]); }),
               "K materialization should reject mismatched tensor types",
               rc)) return rc;

    auto * wrong_shape = ggml_new_tensor_4d(tctx.ctx, GGML_TYPE_F32, 1, 1, 1, 1);
    if (!check(expect_throw([&]() { llama_compacted_prefix_set_input_beta(wrong_shape, seq->layers[0], 1); }),
               "beta materialization should reject mismatched tensor shapes",
               rc)) return rc;

    return rc;
}

int test_non_flash_attention_sanity() {
    int rc = 0;

    llama_compacted_prefix_store store({
        {
            /* layer_id      = */ 0,
            /* n_head_kv     = */ 1,
            /* n_embd_head_k = */ 2,
            /* n_embd_head_v = */ 2,
            /* type_k        = */ GGML_TYPE_F32,
            /* type_v        = */ GGML_TYPE_F32,
        },
    });

    if (!check(store.configure_seq(2, 4, { 0 }, 1), "configure_seq should succeed", rc)) return rc;

    auto * seq = store.get_seq(2);
    if (seq == nullptr) {
        return fail("sequence 2 should exist");
    }
    if (!check(store.set_execution(2, true), "execution should enable through the store API", rc)) return rc;

    auto & layer = seq->layers[0];
    const float prefix_k[2] = { 1.0f, 0.0f };
    const float prefix_v[2] = { 10.0f, 0.0f };
    std::memcpy(layer.k_data.data(), prefix_k, sizeof(prefix_k));
    std::memcpy(layer.v_data.data(), prefix_v, sizeof(prefix_v));
    layer.beta_data[0] = 0.0f;

    std::vector<llama_pos> pos = { 1 };
    std::vector<llama_seq_id> seq_id_unq;
    auto ubatch = make_ubatch_single(pos, seq_id_unq, 2);

    llama_hparams hparams = {};
    hparams.use_alibi = false;

    ggml_test_ctx tctx(1u << 15);
    auto * k_prefix = ggml_new_tensor_4d(tctx.ctx, GGML_TYPE_F32, 2, 1, 1, 1);
    auto * v_prefix = ggml_new_tensor_4d(tctx.ctx, GGML_TYPE_F32, 2, 1, 1, 1);
    auto * b_prefix = ggml_new_tensor_4d(tctx.ctx, GGML_TYPE_F32, 1, 1, 1, 1);
    auto * m_prefix = ggml_new_tensor_4d(tctx.ctx, GGML_TYPE_F32, 1, 1, 1, 1);

    llama_compacted_prefix_set_input_k(k_prefix, layer);
    llama_compacted_prefix_set_input_v(v_prefix, layer);
    llama_compacted_prefix_set_input_beta(b_prefix, layer, 1);
    llama_compacted_prefix_set_input_mask(m_prefix, *seq, ubatch, hparams, /* causal_attn = */ true);

    const auto * prefix_k_data = reinterpret_cast<const float *>(k_prefix->data);
    const auto * prefix_v_data = reinterpret_cast<const float *>(v_prefix->data);
    const auto * prefix_b_data = reinterpret_cast<const float *>(b_prefix->data);
    const auto * prefix_m_data = reinterpret_cast<const float *>(m_prefix->data);

    const float q[2] = { 1.0f, 0.0f };
    const float live_k[2] = { 0.0f, 1.0f };
    const float live_v[2] = { 0.0f, 20.0f };
    const float live_bias = 0.0f;
    const float live_mask = 0.0f;

    const float score_prefix = prefix_k_data[0] * q[0] + prefix_k_data[1] * q[1] + prefix_b_data[0] + prefix_m_data[0];
    const float score_live = live_k[0] * q[0] + live_k[1] * q[1] + live_bias + live_mask;

    const float exp_prefix = std::exp(score_prefix);
    const float exp_live = std::exp(score_live);
    const float norm = exp_prefix + exp_live;
    const float w_prefix = exp_prefix / norm;
    const float w_live = exp_live / norm;

    const float out0 = w_prefix * prefix_v_data[0] + w_live * live_v[0];
    const float out1 = w_prefix * prefix_v_data[1] + w_live * live_v[1];

    if (!check(std::fabs(w_prefix - 0.7310586f) < 1e-4f, "prefix attention weight should match softmax([1, 0])", rc)) return rc;
    if (!check(std::fabs(w_live - 0.2689414f) < 1e-4f, "live attention weight should match softmax([1, 0])", rc)) return rc;
    if (!check(std::fabs(out0 - 7.310586f) < 1e-3f, "prefix/live blend should affect output dim 0", rc)) return rc;
    if (!check(std::fabs(out1 - 5.378828f) < 1e-3f, "prefix/live blend should affect output dim 1", rc)) return rc;

    return rc;
}

int test_execution_state_lifecycle() {
    int rc = 0;

    llama_compacted_prefix_store store({
        {
            /* layer_id      = */ 0,
            /* n_head_kv     = */ 1,
            /* n_embd_head_k = */ 4,
            /* n_embd_head_v = */ 4,
            /* type_k        = */ GGML_TYPE_F16,
            /* type_v        = */ GGML_TYPE_F16,
        },
        {
            /* layer_id      = */ 1,
            /* n_head_kv     = */ 1,
            /* n_embd_head_k = */ 4,
            /* n_embd_head_v = */ 0,
            /* type_k        = */ GGML_TYPE_F16,
            /* type_v        = */ GGML_TYPE_F16,
        },
    });

    if (!check(!store.set_execution(5, true), "execution should not enable before configure_seq", rc)) return rc;
    if (!check(store.configure_seq(5, 8, { 1, 3 }, 4), "configure_seq should succeed", rc)) return rc;
    if (!check(store.set_execution(5, true), "execution should enable after configure_seq", rc)) return rc;
    if (!check(store.execution_enabled(5), "execution_enabled should report the enabled state", rc)) return rc;

    auto * seq = store.get_seq(5);
    if (seq == nullptr) {
        return fail("sequence 5 should exist");
    }
    if (!check(seq->layers.size() == 2 && seq->layers[1].v_data.empty(),
               "layers with n_embd_head_v == 0 should keep an empty V payload",
               rc)) return rc;

    if (!check(store.seq_rm(5, 0, 10), "removing all compacted positions should succeed", rc)) return rc;
    if (!check(!store.is_enabled(5) && !store.execution_enabled(5),
               "removing all compacted positions should clear both enabled and execution state",
               rc)) return rc;

    if (!check(store.configure_seq(5, 8, { 2 }, 3), "reconfigure after clear should succeed", rc)) return rc;
    if (!check(!store.execution_enabled(5), "reconfigure should leave execution disabled until explicitly re-enabled", rc)) return rc;

    return rc;
}

int test_multi_stream_mask_and_beta() {
    int rc = 0;

    llama_compacted_prefix_store store({
        {
            /* layer_id      = */ 0,
            /* n_head_kv     = */ 1,
            /* n_embd_head_k = */ 2,
            /* n_embd_head_v = */ 2,
            /* type_k        = */ GGML_TYPE_F16,
            /* type_v        = */ GGML_TYPE_F16,
        },
    });

    // Prefix tokens at logical positions [0, 2], live suffix starts at pos 3.
    if (!check(store.configure_seq(0, 4, { 0, 2 }, 2), "ms: configure_seq", rc)) return rc;
    auto * seq = store.get_seq(0);
    if (!seq) { return fail("ms: seq should exist"); }
    if (!check(store.set_execution(0, true), "ms: enable exec", rc)) return rc;

    auto & layer = seq->layers[0];
    layer.beta_data[0] = 10.0f;  // head 0, token 0
    layer.beta_data[1] = 20.0f;  // head 0, token 1

    // 2 tokens total, split into 2 streams of 1 token each.
    // Stream 0: pos=1 (between prefix positions → partial mask)
    // Stream 1: pos=3 (after all prefix positions → no masking)
    std::vector<llama_pos> pos = { 1, 3 };
    std::vector<llama_seq_id> seq_id_unq = { 0 };
    llama_ubatch ubatch = {};
    ubatch.n_tokens = 2;
    ubatch.pos = pos.data();
    ubatch.n_seqs_unq = 1;
    ubatch.seq_id_unq = seq_id_unq.data();

    llama_hparams hparams = {};
    hparams.use_alibi = false;

    ggml_test_ctx tctx(1u << 16);

    // Mask: [n_prefix=2, n_tps=1, 1, n_stream=2]
    auto * m = ggml_new_tensor_4d(tctx.ctx, GGML_TYPE_F32, 2, 1, 1, 2);
    // Beta: [n_prefix=2, n_tps=1, n_head=1, n_stream=2]
    auto * b = ggml_new_tensor_4d(tctx.ctx, GGML_TYPE_F32, 2, 1, 1, 2);

    llama_compacted_prefix_set_input_mask(m, *seq, ubatch, hparams, /* causal_attn = */ true);
    llama_compacted_prefix_set_input_beta(b, layer, 1);

    // Stream 0, token at pos=1:
    //   prefix pos 0 <= 1 → unmasked (0.0)
    //   prefix pos 2 > 1  → masked (-inf)
    if (!check(tensor_f32_at(m, 0, 0, 0, 0) == 0.0f,
               "ms: stream 0 prefix pos 0 unmasked (0 <= 1)", rc)) return rc;
    if (!check(std::isinf(tensor_f32_at(m, 1, 0, 0, 0)) && tensor_f32_at(m, 1, 0, 0, 0) < 0.0f,
               "ms: stream 0 prefix pos 2 masked (2 > 1)", rc)) return rc;

    // Stream 1, token at pos=3:
    //   prefix pos 0 <= 3 → unmasked
    //   prefix pos 2 <= 3 → unmasked
    if (!check(tensor_f32_at(m, 0, 0, 0, 1) == 0.0f,
               "ms: stream 1 prefix pos 0 unmasked (0 <= 3)", rc)) return rc;
    if (!check(tensor_f32_at(m, 1, 0, 0, 1) == 0.0f,
               "ms: stream 1 prefix pos 2 unmasked (2 <= 3)", rc)) return rc;

    // Beta should be identical across streams (broadcast, not stream-dependent)
    if (!check(tensor_f32_at(b, 0, 0, 0, 0) == 10.0f, "ms: stream 0 beta[0]", rc)) return rc;
    if (!check(tensor_f32_at(b, 1, 0, 0, 0) == 20.0f, "ms: stream 0 beta[1]", rc)) return rc;
    if (!check(tensor_f32_at(b, 0, 0, 0, 1) == 10.0f, "ms: stream 1 beta[0]", rc)) return rc;
    if (!check(tensor_f32_at(b, 1, 0, 0, 1) == 20.0f, "ms: stream 1 beta[1]", rc)) return rc;

    return rc;
}

} // namespace

int main() {
    if (const int rc = test_execution_gate()) {
        return rc;
    }
    if (const int rc = test_materialization_and_mask()) {
        return rc;
    }
    if (const int rc = test_type_and_shape_guards()) {
        return rc;
    }
    if (const int rc = test_non_flash_attention_sanity()) {
        return rc;
    }
    if (const int rc = test_execution_state_lifecycle()) {
        return rc;
    }
    if (const int rc = test_multi_stream_mask_and_beta()) {
        return rc;
    }

    return 0;
}
