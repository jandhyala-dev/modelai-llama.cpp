#include "src/llama-kv-compacted-prefix.h"

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int fail(const std::string & message) {
    std::cerr << "test-kv-compacted-prefix: " << message << std::endl;
    return 1;
}

bool check(bool cond, const std::string & message, int & rc) {
    if (!cond) {
        rc = fail(message);
        return false;
    }
    return true;
}

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

bool expect_token_value(
        const std::vector<uint8_t> & data,
        uint32_t n_head_kv,
        uint32_t n_tokens,
        size_t bytes_per_token,
        uint32_t head,
        uint32_t token,
        uint8_t expected) {
    if (head >= n_head_kv || token >= n_tokens) {
        return false;
    }
    const size_t offset = (size_t(head) * n_tokens + token) * bytes_per_token;
    for (size_t i = 0; i < bytes_per_token; ++i) {
        if (data[offset + i] != expected) {
            return false;
        }
    }
    return true;
}

void fill_beta(std::vector<float> & beta, uint32_t n_head_kv, uint32_t n_tokens) {
    for (uint32_t head = 0; head < n_head_kv; ++head) {
        for (uint32_t token = 0; token < n_tokens; ++token) {
            beta[size_t(head) * n_tokens + token] = float(head * 100 + token);
        }
    }
}

bool expect_beta(
        const std::vector<float> & beta,
        uint32_t n_head_kv,
        uint32_t n_tokens,
        uint32_t head,
        uint32_t token,
        float expected) {
    if (head >= n_head_kv || token >= n_tokens) {
        return false;
    }
    return beta[size_t(head) * n_tokens + token] == expected;
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

int test_basic_shape_and_ops() {
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
        {
            /* layer_id      = */ 7,
            /* n_head_kv     = */ 1,
            /* n_embd_head_k = */ 16,
            /* n_embd_head_v = */ 0,
            /* type_k        = */ GGML_TYPE_F16,
            /* type_v        = */ GGML_TYPE_F16,
        },
    });

    const std::vector<llama_pos> logical_positions = { 0, 4, 8 };
    if (!check(store.configure_seq(/* seq_id = */ 3, /* logical_token_count = */ 16, logical_positions, /* live_suffix_pos0 = */ 9), "configure_seq should succeed", rc)) return rc;
    if (!check(store.is_enabled(3), "sequence 3 should be enabled", rc)) return rc;
    if (!check(store.seq_pos_min(3) == 0, "sequence 3 min position should be 0", rc)) return rc;
    if (!check(store.seq_pos_max(3) == 8, "sequence 3 max position should be 8", rc)) return rc;
    if (!check(store.seq_allocated_bytes(3) > 0, "sequence 3 should allocate bytes", rc)) return rc;

    const auto * seq3 = store.get_seq(3);
    if (seq3 == nullptr) {
        return fail("sequence 3 should exist");
    }
    if (!check(seq3->logical_token_count == 16, "logical_token_count snapshot should be preserved", rc)) return rc;
    if (!check(seq3->logical_positions.size() == 3, "logical_positions size should be 3", rc)) return rc;
    if (!check(seq3->layers.size() == 2, "layer count should be 2", rc)) return rc;
    if (!check(seq3->layers[0].n_compacted_tokens == 3, "layer 0 compacted token count should be 3", rc)) return rc;
    if (!check(seq3->layers[1].n_compacted_tokens == 3, "layer 1 compacted token count should be 3", rc)) return rc;
    if (!check(seq3->layers[0].beta_data.size() == 6, "layer 0 beta size should be 6", rc)) return rc;
    if (!check(seq3->layers[1].beta_data.size() == 3, "layer 1 beta size should be 3", rc)) return rc;
    if (!check(seq3->layers[1].v_data.empty(), "layer 1 V storage should be empty when n_embd_head_v == 0", rc)) return rc;

    const auto breakdown = store.memory_breakdown();
    if (!check(!breakdown.empty(), "memory breakdown should not be empty", rc)) return rc;
    if (!check(breakdown.begin()->second == store.total_allocated_bytes(), "memory breakdown should match total allocated bytes", rc)) return rc;

    store.seq_add(3, 0, 5, 10);
    if (!check(store.seq_pos_min(3) == 8, "seq_add should shift min position", rc)) return rc;
    if (!check(store.seq_pos_max(3) == 14, "seq_add should shift max position", rc)) return rc;

    store.seq_div(3, 10, 20, 2);
    if (!check(store.seq_pos_min(3) == 5, "seq_div should reduce min position", rc)) return rc;
    if (!check(store.seq_pos_max(3) == 8, "seq_div should leave out-of-range tokens untouched", rc)) return rc;

    store.seq_cp(3, 5, -1, -1);
    if (!check(store.is_enabled(5), "sequence 5 should be enabled after seq_cp", rc)) return rc;
    if (!check(store.seq_pos_min(5) == 5, "sequence 5 min position should be 5", rc)) return rc;
    if (!check(store.seq_pos_max(5) == 8, "sequence 5 max position should be 8", rc)) return rc;
    if (!check(store.seq_allocated_bytes(5) > 0, "sequence 5 should allocate bytes", rc)) return rc;

    store.seq_rm(5, 7, 8);
    if (!check(store.seq_pos_min(5) == 5, "sequence 5 min position should remain 5 after seq_rm", rc)) return rc;
    if (!check(store.seq_pos_max(5) == 8, "sequence 5 max position should remain 8 after seq_rm", rc)) return rc;
    const auto * seq5 = store.get_seq(5);
    if (seq5 == nullptr) {
        return fail("sequence 5 should exist");
    }
    if (!check(seq5->logical_positions.size() == 2,
               "sequence 5 should retain two logical positions (actual size = " + std::to_string(seq5->logical_positions.size()) + ")",
               rc)) return rc;
    if (!check(seq5->logical_positions[0] == 5, "sequence 5 first logical position should be 5", rc)) return rc;
    if (!check(seq5->logical_positions[1] == 8, "sequence 5 second logical position should be 8", rc)) return rc;

    store.seq_keep(5);
    if (!check(!store.is_enabled(3), "seq_keep should clear non-target sequences", rc)) return rc;
    if (!check(store.is_enabled(5), "seq_keep should preserve target sequence", rc)) return rc;

    store.clear_seq(5, false);
    if (!check(!store.is_enabled(5), "clear_seq(false) should disable the sequence", rc)) return rc;
    if (!check(store.seq_allocated_bytes(5) == 0, "clear_seq(false) should zero allocated bytes", rc)) return rc;

    if (!check(store.configure_seq(/* seq_id = */ 1, /* logical_token_count = */ 8, { 2, 4 }, /* live_suffix_pos0 = */ 5), "configure_seq on sequence 1 should succeed", rc)) return rc;
    if (!check(store.is_enabled(1), "sequence 1 should be enabled", rc)) return rc;
    store.clear(true);
    if (!check(!store.is_enabled(1), "clear(true) should disable sequence 1", rc)) return rc;
    if (!check(store.total_allocated_bytes() == 0, "clear(true) should free all bytes", rc)) return rc;

    return rc;
}

int test_seq_cp_data_integrity() {
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

    if (!check(store.configure_seq(3, 16, { 0, 4, 8 }, 4), "configure_seq for integrity test should succeed", rc)) return rc;

    auto * seq3 = store.get_seq(3);
    if (seq3 == nullptr) {
        return fail("sequence 3 should exist for integrity test");
    }

    auto & layer = seq3->layers[0];
    const size_t k_token_bytes = token_bytes(layer.layout.type_k, layer.layout.n_embd_head_k);
    const size_t v_token_bytes = token_bytes(layer.layout.type_v, layer.layout.n_embd_head_v);
    fill_token_bytes(layer.k_data, layer.layout.n_head_kv, 3, k_token_bytes);
    fill_token_bytes(layer.v_data, layer.layout.n_head_kv, 3, v_token_bytes);
    fill_beta(layer.beta_data, layer.layout.n_head_kv, 3);

    store.seq_cp(3, 4, 0, 5);
    const auto * seq4 = store.get_seq(4);
    if (seq4 == nullptr) {
        return fail("sequence 4 should exist after partial seq_cp");
    }
    if (!check(seq4->logical_positions.size() == 2, "partial seq_cp should keep two positions", rc)) return rc;
    if (!check(seq4->logical_positions[0] == 0 && seq4->logical_positions[1] == 4, "partial seq_cp should preserve selected positions", rc)) return rc;
    if (!check(seq4->live_suffix_pos0 == 4, "partial seq_cp should preserve in-range live_suffix_pos0", rc)) return rc;

    const auto & copied = seq4->layers[0];
    if (!check(expect_token_value(copied.k_data, copied.layout.n_head_kv, 2, k_token_bytes, 0, 0, 0), "copied K token 0/head 0 should match source token 0", rc)) return rc;
    if (!check(expect_token_value(copied.k_data, copied.layout.n_head_kv, 2, k_token_bytes, 0, 1, 1), "copied K token 1/head 0 should match source token 1", rc)) return rc;
    if (!check(expect_token_value(copied.k_data, copied.layout.n_head_kv, 2, k_token_bytes, 1, 0, 32), "copied K token 0/head 1 should match source head 1 token 0", rc)) return rc;
    if (!check(expect_token_value(copied.k_data, copied.layout.n_head_kv, 2, k_token_bytes, 1, 1, 33), "copied K token 1/head 1 should match source head 1 token 1", rc)) return rc;
    if (!check(expect_beta(copied.beta_data, copied.layout.n_head_kv, 2, 1, 1, 101.0f), "copied beta should preserve per-head/token values", rc)) return rc;
    if (!check(expect_token_value(copied.v_data, copied.layout.n_head_kv, 2, v_token_bytes, 1, 1, 33), "copied V should preserve per-head/token values", rc)) return rc;

    store.seq_cp(3, 6, 0, 1);
    const auto * seq6 = store.get_seq(6);
    if (seq6 == nullptr) {
        return fail("sequence 6 should exist after narrow seq_cp");
    }
    if (!check(seq6->live_suffix_pos0 == -1, "partial seq_cp should drop out-of-range live_suffix_pos0", rc)) return rc;

    store.seq_cp(3, 3, -1, -1);
    seq3 = store.get_seq(3);
    if (seq3 == nullptr) {
        return fail("sequence 3 should still exist after self seq_cp");
    }
    if (!check(seq3->logical_positions.size() == 3, "self seq_cp should be a no-op", rc)) return rc;
    if (!check(expect_token_value(seq3->layers[0].k_data, layer.layout.n_head_kv, 3, k_token_bytes, 1, 2, 34), "self seq_cp should not corrupt payload data", rc)) return rc;

    return rc;
}

int test_sequence_edge_cases() {
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

    if (!check(store.configure_seq(7, 4, { 0, 1 }, 1), "configure_seq for edge cases should succeed", rc)) return rc;
    if (!check(expect_throw([&]() { store.seq_div(7, 0, 8, 0); }), "seq_div should reject d == 0", rc)) return rc;
    if (!check(expect_throw([&]() { store.seq_div(7, 0, 8, 2); }), "seq_div should reject duplicate positions after division", rc)) return rc;
    if (!check(expect_throw([&]() { store.seq_add(7, 0, 8, -2); }), "seq_add should reject negative positions", rc)) return rc;

    if (!check(store.configure_seq(8, 10, { 2, 4, 6 }, 6), "configure_seq overwrite source should succeed", rc)) return rc;
    if (!check(store.configure_seq(8, 11, { 3 }, 3), "configure_seq overwrite destination should succeed", rc)) return rc;
    const auto * seq8 = store.get_seq(8);
    if (seq8 == nullptr) {
        return fail("sequence 8 should exist after overwrite");
    }
    if (!check(seq8->logical_token_count == 11, "configure overwrite should replace logical_token_count snapshot", rc)) return rc;
    if (!check(seq8->logical_positions.size() == 1 && seq8->logical_positions[0] == 3, "configure overwrite should replace positions", rc)) return rc;

    store.seq_rm(8, -1, -1);
    if (!check(!store.is_enabled(8), "seq_rm all should disable sequence", rc)) return rc;
    if (!check(store.seq_allocated_bytes(8) == 0, "seq_rm all should release compacted-prefix bytes", rc)) return rc;

    llama_compacted_prefix_store empty_store({});
    if (!check(empty_store.configure_seq(2, 4, { 1, 3 }, -1), "zero-layout configure_seq should succeed", rc)) return rc;
    const auto * empty_seq = empty_store.get_seq(2);
    if (empty_seq == nullptr) {
        return fail("zero-layout sequence should exist");
    }
    if (!check(empty_seq->layers.empty(), "zero-layout sequence should have no layer storage", rc)) return rc;
    if (!check(empty_store.total_allocated_bytes() == 0, "zero-layout store should allocate zero bytes", rc)) return rc;

    llama_compacted_prefix_store quantized_store({
        {
            /* layer_id      = */ 0,
            /* n_head_kv     = */ 1,
            /* n_embd_head_k = */ 32,
            /* n_embd_head_v = */ 32,
            /* type_k        = */ GGML_TYPE_Q8_0,
            /* type_v        = */ GGML_TYPE_F16,
        },
    });
    if (!check(expect_throw([&]() { quantized_store.configure_seq(1, 8, { 0, 4 }, -1); }), "quantized K types should be rejected in P2", rc)) return rc;

    return rc;
}

} // namespace

int main() {
    if (const int rc = test_basic_shape_and_ops()) {
        return rc;
    }
    if (const int rc = test_seq_cp_data_integrity()) {
        return rc;
    }
    if (const int rc = test_sequence_edge_cases()) {
        return rc;
    }
    return 0;
}
