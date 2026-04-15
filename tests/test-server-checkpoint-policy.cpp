#include "../tools/server/server-checkpoint-utils.h"
#include "testing.h"

#include <iostream>

static void test_checkpoint_snapshot_tokens(testing & t) {
    t.assert_equal("final short-prompt checkpoint captures prior tokens", int64_t(14), server_checkpoint_snapshot_tokens(18, 4));
    t.assert_equal("whole batch has no prior tokens to snapshot", int64_t(0), server_checkpoint_snapshot_tokens(14, 14));
}

static void test_checkpoint_policy_short_prompt(testing & t) {
    t.assert_true("short hybrid/recurrent prompt can checkpoint before the last few tokens",
        server_should_create_context_checkpoint(/* pos_min = */ 0, /* prompt_n_tokens = */ 18, /* n_tokens_cur = */ 4,
                                                /* last_checkpoint_n_tokens = */ -1, /* has_mtmd = */ false));

    t.assert_true("final short-prompt checkpoint may replace an earlier zero-token checkpoint",
        server_should_create_context_checkpoint(/* pos_min = */ 0, /* prompt_n_tokens = */ 18, /* n_tokens_cur = */ 4,
                                                /* last_checkpoint_n_tokens = */ 0, /* has_mtmd = */ false));

    t.assert_true("checkpoints do not advance when they snapshot the same prompt span",
        !server_should_create_context_checkpoint(/* pos_min = */ 0, /* prompt_n_tokens = */ 18, /* n_tokens_cur = */ 4,
                                                 /* last_checkpoint_n_tokens = */ 14, /* has_mtmd = */ false));
}

static void test_checkpoint_policy_rejects_useless_snapshots(testing & t) {
    t.assert_true("cannot checkpoint before any tokens have been processed",
        !server_should_create_context_checkpoint(/* pos_min = */ 0, /* prompt_n_tokens = */ 14, /* n_tokens_cur = */ 14,
                                                 /* last_checkpoint_n_tokens = */ -1, /* has_mtmd = */ false));

    t.assert_true("cannot checkpoint without cache data",
        !server_should_create_context_checkpoint(/* pos_min = */ -1, /* prompt_n_tokens = */ 18, /* n_tokens_cur = */ 4,
                                                 /* last_checkpoint_n_tokens = */ -1, /* has_mtmd = */ false));

    t.assert_true("mtmd chunks stay non-checkpointed",
        !server_should_create_context_checkpoint(/* pos_min = */ 0, /* prompt_n_tokens = */ 18, /* n_tokens_cur = */ 4,
                                                 /* last_checkpoint_n_tokens = */ -1, /* has_mtmd = */ true));
}

int main(int argc, char ** argv) {
    testing t(std::cout);
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("snapshot_tokens", test_checkpoint_snapshot_tokens);
    t.test("short_prompt", test_checkpoint_policy_short_prompt);
    t.test("rejects_useless_snapshots", test_checkpoint_policy_rejects_useless_snapshots);
    return t.summary();
}
