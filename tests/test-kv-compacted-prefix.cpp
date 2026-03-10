#include "src/llama-kv-compacted-prefix.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

int main() {
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
    assert(store.configure_seq(/* seq_id = */ 3, /* logical_token_count = */ 16, logical_positions, /* live_suffix_pos0 = */ 9));
    assert(store.is_enabled(3));
    assert(store.seq_pos_min(3) == 0);
    assert(store.seq_pos_max(3) == 8);
    assert(store.seq_allocated_bytes(3) > 0);

    const auto * seq3 = store.get_seq(3);
    if (seq3 == nullptr) {
        return 1;
    }
    assert(seq3->logical_token_count == 16);
    assert(seq3->logical_positions.size() == 3);
    assert(seq3->layers.size() == 2);
    assert(seq3->layers[0].n_compacted_tokens == 3);
    assert(seq3->layers[1].n_compacted_tokens == 3);
    assert(seq3->layers[0].beta_data.size() == 6);
    assert(seq3->layers[1].beta_data.size() == 3);

    const auto breakdown = store.memory_breakdown();
    assert(!breakdown.empty());
    assert(breakdown.begin()->second == store.total_allocated_bytes());

    store.seq_add(3, 0, 5, 10);
    assert(store.seq_pos_min(3) == 8);
    assert(store.seq_pos_max(3) == 14);

    store.seq_div(3, 10, 20, 2);
    assert(store.seq_pos_min(3) == 5);
    assert(store.seq_pos_max(3) == 7);

    store.seq_cp(3, 5, -1, -1);
    assert(store.is_enabled(5));
    assert(store.seq_pos_min(5) == 5);
    assert(store.seq_pos_max(5) == 7);
    assert(store.seq_allocated_bytes(5) > 0);

    store.seq_rm(5, 6, 7);
    assert(store.seq_pos_min(5) == 5);
    assert(store.seq_pos_max(5) == 7);
    const auto * seq5 = store.get_seq(5);
    if (seq5 == nullptr) {
        return 1;
    }
    assert(seq5->logical_positions.size() == 2);
    assert(seq5->logical_positions[0] == 5);
    assert(seq5->logical_positions[1] == 7);

    store.seq_keep(5);
    assert(!store.is_enabled(3));
    assert(store.is_enabled(5));

    store.clear_seq(5, false);
    assert(!store.is_enabled(5));
    assert(store.seq_allocated_bytes(5) == 0);

    store.configure_seq(/* seq_id = */ 1, /* logical_token_count = */ 8, { 2, 4 }, /* live_suffix_pos0 = */ 5);
    assert(store.is_enabled(1));
    store.clear(true);
    assert(!store.is_enabled(1));
    assert(store.total_allocated_bytes() == 0);

    return 0;
}
