#pragma once

#include <cstdint>

static inline int64_t server_checkpoint_snapshot_tokens(int prompt_n_tokens, int n_tokens_cur) {
    return (int64_t) prompt_n_tokens - n_tokens_cur;
}

static inline bool server_should_create_context_checkpoint(
        int pos_min,
        int prompt_n_tokens,
        int n_tokens_cur,
        int64_t last_checkpoint_n_tokens,
        bool has_mtmd) {
    if (has_mtmd || pos_min < 0) {
        return false;
    }

    const int64_t checkpoint_n_tokens = server_checkpoint_snapshot_tokens(prompt_n_tokens, n_tokens_cur);
    if (checkpoint_n_tokens <= 0) {
        return false;
    }

    return last_checkpoint_n_tokens < 0 || checkpoint_n_tokens > last_checkpoint_n_tokens;
}
