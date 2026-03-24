// Unit tests for self-study Q-capture components (6b-2, 6b-3).
//
// Tests the host-side data structures and algorithms without requiring a model
// or GPU. The generation loop (6b-4) and full pipeline (6b-5) require a live
// context and are tested in test-kv-compact-quality.cpp with a real model.

#include "src/llama-kv-compact-self-study.h"
#include "kv-compact-test-helpers.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using llama_kv_compact_test::check;
using llama_kv_compact_test::fail;

namespace {

static const std::string test_prefix = "test-kv-compact-self-study";

// Create a CPU-backend-backed F32 3D tensor with data.
// ggml_backend_tensor_get() requires a buffer, so we use the CPU backend.
struct tensor_ctx {
    ggml_context * ctx     = nullptr;
    ggml_tensor  * t       = nullptr;
    ggml_backend_buffer_t buf  = nullptr;
    ggml_backend_t        be   = nullptr;

    ~tensor_ctx() {
        if (buf) { ggml_backend_buffer_free(buf); }
        if (ctx) { ggml_free(ctx); }
        if (be)  { ggml_backend_free(be); }
    }
};

bool make_f32_tensor_2d(tensor_ctx & tc, const char * name,
                        int64_t d0, int64_t d1,
                        const float * data) {
    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 2,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    tc.ctx = ggml_init(params);
    if (!tc.ctx) { return false; }

    tc.t = ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, d0, d1);
    if (!tc.t) { return false; }

    ggml_set_name(tc.t, name);

    tc.be = ggml_backend_cpu_init();
    if (!tc.be) { return false; }

    tc.buf = ggml_backend_alloc_ctx_tensors(tc.ctx, tc.be);
    if (!tc.buf) { return false; }

    ggml_backend_tensor_set(tc.t, data, 0, ggml_nbytes(tc.t));
    return true;
}

bool make_f32_tensor_3d(tensor_ctx & tc, const char * name,
                        int64_t d0, int64_t d1, int64_t d2,
                        const float * data) {
    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 2,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    tc.ctx = ggml_init(params);
    if (!tc.ctx) { return false; }

    tc.t = ggml_new_tensor_3d(tc.ctx, GGML_TYPE_F32, d0, d1, d2);
    if (!tc.t) { return false; }

    ggml_set_name(tc.t, name);

    tc.be = ggml_backend_cpu_init();
    if (!tc.be) { return false; }

    tc.buf = ggml_backend_alloc_ctx_tensors(tc.ctx, tc.be);
    if (!tc.buf) { return false; }

    ggml_backend_tensor_set(tc.t, data, 0, ggml_nbytes(tc.t));
    return true;
}

} // namespace

int main() {
    int rc = 0;

    // -----------------------------------------------------------------------
    // Test 1: llama_q_capture_state::reset
    // -----------------------------------------------------------------------
    {
        llama_q_capture_state state;
        state.reset(3, 4, 2, 10);  // 3 layers, n_embd_head=4, n_head_q=2, reserve 10

        if (!check(test_prefix, state.n_layers == 3, "reset: n_layers should be 3", rc)) return rc;
        if (!check(test_prefix, state.layers.size() == 3, "reset: should have 3 layers", rc)) return rc;
        if (!check(test_prefix, !state.active, "reset: should not be active", rc)) return rc;

        for (int i = 0; i < 3; i++) {
            const auto & lq = state.layers[i];
            if (!check(test_prefix, lq.n_embd_head == 4, "reset: n_embd_head should be 4", rc)) return rc;
            if (!check(test_prefix, lq.n_head_q == 2, "reset: n_head_q should be 2", rc)) return rc;
            if (!check(test_prefix, lq.n_tokens == 0, "reset: n_tokens should be 0", rc)) return rc;
            if (!check(test_prefix, !lq.has_pending, "reset: has_pending should be false", rc)) return rc;
            if (!check(test_prefix, lq.data.empty(), "reset: data should be empty", rc)) return rc;
            if (!check(test_prefix, lq.data.capacity() >= 4u * 2u * 10u, "reset: should pre-reserve", rc)) return rc;
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: append_from_tensor + finalize_step (single token, single tensor)
    // -----------------------------------------------------------------------
    {
        llama_q_capture_state state;
        state.reset(2, 2, 2);  // 2 layers, n_embd_head=2, n_head_q=2

        // Tensor: [n_embd_head=2, n_head_q=2, n_tokens=1]
        // Data: head0=[1,2], head1=[3,4] → flat: [1,2,3,4]
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f};

        tensor_ctx tc;
        if (!check(test_prefix, make_f32_tensor_3d(tc, "Qcur-0", 2, 2, 1, data),
                   "append: failed to create tensor", rc)) return rc;

        state.append_from_tensor(0, tc.t);
        if (!check(test_prefix, state.layers[0].has_pending, "append: should have pending", rc)) return rc;
        if (!check(test_prefix, state.layers[0].n_tokens == 0, "append: n_tokens should still be 0 before finalize", rc)) return rc;
        if (!check(test_prefix, state.layers[0].data.size() == 4, "append: data should have 4 floats", rc)) return rc;

        state.finalize_step();
        if (!check(test_prefix, state.layers[0].n_tokens == 1, "finalize: n_tokens should be 1", rc)) return rc;
        if (!check(test_prefix, !state.layers[0].has_pending, "finalize: pending should be cleared", rc)) return rc;

        // Layer 1 should be untouched
        if (!check(test_prefix, state.layers[1].n_tokens == 0, "finalize: layer 1 should have 0 tokens", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 3: overwrite strategy (multiple tensors per layer per step)
    // -----------------------------------------------------------------------
    {
        llama_q_capture_state state;
        state.reset(1, 2, 2);  // 1 layer, n_embd_head=2, n_head_q=2

        // First tensor for layer 0: [10, 20, 30, 40]
        float data1[] = {10.0f, 20.0f, 30.0f, 40.0f};
        tensor_ctx tc1;
        if (!check(test_prefix, make_f32_tensor_3d(tc1, "Qcur-0", 2, 2, 1, data1),
                   "overwrite: failed to create tensor 1", rc)) return rc;
        state.append_from_tensor(0, tc1.t);

        // Second tensor for same layer 0 (same step): should overwrite
        float data2[] = {100.0f, 200.0f, 300.0f, 400.0f};
        tensor_ctx tc2;
        if (!check(test_prefix, make_f32_tensor_3d(tc2, "Qcur_normed-0", 2, 2, 1, data2),
                   "overwrite: failed to create tensor 2", rc)) return rc;
        state.append_from_tensor(0, tc2.t);

        // Data size should not grow (overwrite, not append)
        if (!check(test_prefix, state.layers[0].data.size() == 4, "overwrite: data size should still be 4", rc)) return rc;

        state.finalize_step();
        if (!check(test_prefix, state.layers[0].n_tokens == 1, "overwrite: should count 1 token, not 2", rc)) return rc;

        // Verify the SECOND tensor's data won (last write wins)
        const auto & d = state.layers[0].data;
        if (!check(test_prefix, d[0] == 100.0f && d[1] == 200.0f && d[2] == 300.0f && d[3] == 400.0f,
                   "overwrite: last tensor should win", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 4: multi-step accumulation
    // -----------------------------------------------------------------------
    {
        llama_q_capture_state state;
        state.reset(1, 2, 1);  // 1 layer, n_embd_head=2, n_head_q=1

        // Step 1
        float d1[] = {1.0f, 2.0f};
        tensor_ctx tc1;
        if (!check(test_prefix, make_f32_tensor_3d(tc1, "Qcur-0", 2, 1, 1, d1),
                   "multi-step: tensor 1", rc)) return rc;
        state.append_from_tensor(0, tc1.t);
        state.finalize_step();

        // Step 2
        float d2[] = {3.0f, 4.0f};
        tensor_ctx tc2;
        if (!check(test_prefix, make_f32_tensor_3d(tc2, "Qcur-0", 2, 1, 1, d2),
                   "multi-step: tensor 2", rc)) return rc;
        state.append_from_tensor(0, tc2.t);
        state.finalize_step();

        // Step 3
        float d3[] = {5.0f, 6.0f};
        tensor_ctx tc3;
        if (!check(test_prefix, make_f32_tensor_3d(tc3, "Qcur-0", 2, 1, 1, d3),
                   "multi-step: tensor 3", rc)) return rc;
        state.append_from_tensor(0, tc3.t);
        state.finalize_step();

        if (!check(test_prefix, state.layers[0].n_tokens == 3, "multi-step: should have 3 tokens", rc)) return rc;
        if (!check(test_prefix, state.layers[0].data.size() == 6, "multi-step: should have 6 floats", rc)) return rc;
        if (!check(test_prefix, state.layers[0].data[0] == 1.0f, "multi-step: tok0 val0", rc)) return rc;
        if (!check(test_prefix, state.layers[0].data[2] == 3.0f, "multi-step: tok1 val0", rc)) return rc;
        if (!check(test_prefix, state.layers[0].data[4] == 5.0f, "multi-step: tok2 val0", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 5: dimension mismatch rejection
    // -----------------------------------------------------------------------
    {
        llama_q_capture_state state;
        state.reset(1, 4, 2);  // expects n_embd_head=4, n_head_q=2

        // Wrong dimensions: n_embd_head=2, n_head_q=2 (ne[0] mismatch)
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        tensor_ctx tc;
        if (!check(test_prefix, make_f32_tensor_3d(tc, "Qcur-0", 2, 2, 1, data),
                   "dim mismatch: create tensor", rc)) return rc;
        state.append_from_tensor(0, tc.t);

        // Should be silently rejected — no data appended
        if (!check(test_prefix, state.layers[0].data.empty(), "dim mismatch: should reject mismatched tensor", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 6: cb_eval callback — ask phase filtering
    // -----------------------------------------------------------------------
    {
        llama_q_capture_state state;
        state.reset(1, 2, 2);
        state.active = true;

        // 3D Qcur tensor — should be accepted in ask phase
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        tensor_ctx tc3d;
        if (!check(test_prefix, make_f32_tensor_3d(tc3d, "Qcur-0", 2, 2, 1, data),
                   "callback ask: create 3d tensor", rc)) return rc;
        bool accepted = llama_q_capture_eval_callback(tc3d.t, true, &state);
        if (!check(test_prefix, accepted, "callback ask: 3D Qcur should be accepted", rc)) return rc;

        // Non-Qcur tensor name — should pass through
        tensor_ctx tc_other;
        if (!check(test_prefix, make_f32_tensor_3d(tc_other, "Kcur-0", 2, 2, 1, data),
                   "callback ask: create non-Qcur tensor", rc)) return rc;
        bool passed_through = llama_q_capture_eval_callback(tc_other.t, true, &state);
        if (!check(test_prefix, passed_through, "callback ask: non-Qcur should pass through", rc)) return rc;

        // Inactive state — should pass through
        state.active = false;
        bool inactive_result = llama_q_capture_eval_callback(tc3d.t, true, &state);
        if (!check(test_prefix, inactive_result, "callback ask: inactive should pass through", rc)) return rc;

        // Null state — should not crash
        bool null_result = llama_q_capture_eval_callback(tc3d.t, true, nullptr);
        if (!check(test_prefix, null_result, "callback ask: null state should pass through", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 7: ask-phase regression — 2D pre-reshape Qcur accepted by ask,
    //         rejected by receive (the new contract from the n_dims fix)
    // -----------------------------------------------------------------------
    {
        // State expects n_embd_head=128, n_head_q=32 (typical Llama dimensions)
        llama_q_capture_state state;
        state.reset(1, 128, 32);
        state.active = true;

        // Pre-reshape Qcur projection: 2D [n_embd=4096, n_tokens=1]
        // This is what cb(Qcur, "Qcur", il) emits BEFORE ggml_reshape_3d.
        // n_embd = n_embd_head * n_head_q = 128 * 32 = 4096
        std::vector<float> proj_data(4096, 1.0f);
        tensor_ctx tc_2d;
        if (!check(test_prefix, make_f32_tensor_2d(tc_2d, "Qcur-0", 4096, 1, proj_data.data()),
                   "ask regression: create 2D pre-reshape tensor", rc)) return rc;

        // Ask phase: must accept (the fix — no more n_dims filter)
        bool ask_result = llama_q_capture_eval_callback(tc_2d.t, true, &state);
        if (!check(test_prefix, ask_result, "ask regression: 2D Qcur should be accepted in ask phase", rc)) return rc;

        // Receive phase: must reject because d0=4096 != n_embd_head=128
        bool recv_result = llama_q_capture_eval_callback(tc_2d.t, false, &state);
        if (!check(test_prefix, recv_result, "ask regression: receive should return true (continue graph)", rc)) return rc;
        if (!check(test_prefix, state.layers[0].data.empty(), "ask regression: no data should be appended", rc)) return rc;
        if (!check(test_prefix, !state.layers[0].has_pending, "ask regression: no pending state", rc)) return rc;
        if (!check(test_prefix, state.layers[0].n_tokens == 0, "ask regression: no tokens committed", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 8: cb_eval callback — receive phase data capture
    // -----------------------------------------------------------------------
    {
        llama_q_capture_state state;
        state.reset(2, 2, 2);
        state.active = true;

        float data[] = {10.0f, 20.0f, 30.0f, 40.0f};
        tensor_ctx tc;
        if (!check(test_prefix, make_f32_tensor_3d(tc, "Qcur-1", 2, 2, 1, data),
                   "callback receive: create tensor", rc)) return rc;

        bool result = llama_q_capture_eval_callback(tc.t, false, &state);
        if (!check(test_prefix, result, "callback receive: should return true (continue)", rc)) return rc;

        // Layer 1 should have pending data, layer 0 should be empty
        if (!check(test_prefix, state.layers[1].has_pending, "callback receive: layer 1 should have pending", rc)) return rc;
        if (!check(test_prefix, state.layers[0].data.empty(), "callback receive: layer 0 should be empty", rc)) return rc;
        if (!check(test_prefix, state.layers[1].data.size() == 4, "callback receive: layer 1 should have 4 floats", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 9: GQA regrouping — non-GQA (n_rep == 1)
    // -----------------------------------------------------------------------
    {
        llama_q_capture_state state;
        state.reset(1, 2, 2);  // n_embd_head=2, n_head_q=2

        // Simulate 2 tokens captured. Token-major layout:
        // tok0: head0=[1,2], head1=[3,4] → [1,2,3,4]
        // tok1: head0=[5,6], head1=[7,8] → [5,6,7,8]
        state.layers[0].data = {1,2,3,4, 5,6,7,8};
        state.layers[0].n_tokens = 2;

        // n_head_kv=2 (same as n_head_q) → n_rep=1
        // KV head 0 → Q head 0 only
        llama_kv_compact_matrix out;
        bool ok = llama_q_capture_regroup_for_kv_head(state, 0, 0, 2, out);
        if (!check(test_prefix, ok, "regroup non-GQA: should succeed", rc)) return rc;
        if (!check(test_prefix, out.rows == 2, "regroup non-GQA: 1 head × 2 tokens = 2 rows", rc)) return rc;
        if (!check(test_prefix, out.cols == 2, "regroup non-GQA: cols = n_embd_head = 2", rc)) return rc;

        // Row 0 = Q head 0, token 0 = [1, 2]
        if (!check(test_prefix, out(0,0) == 1.0f && out(0,1) == 2.0f, "regroup non-GQA: row 0", rc)) return rc;
        // Row 1 = Q head 0, token 1 = [5, 6]
        if (!check(test_prefix, out(1,0) == 5.0f && out(1,1) == 6.0f, "regroup non-GQA: row 1", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 10: GQA regrouping — GQA with n_rep == 2
    // -----------------------------------------------------------------------
    {
        llama_q_capture_state state;
        state.reset(1, 2, 4);  // n_embd_head=2, n_head_q=4

        // 2 tokens, 4 Q heads. Token-major:
        // tok0: h0=[1,2] h1=[3,4] h2=[5,6] h3=[7,8]
        // tok1: h0=[9,10] h1=[11,12] h2=[13,14] h3=[15,16]
        state.layers[0].data = {1,2,3,4,5,6,7,8, 9,10,11,12,13,14,15,16};
        state.layers[0].n_tokens = 2;

        // n_head_kv=2, n_rep=4/2=2
        // KV head 0 → Q heads 0,1
        llama_kv_compact_matrix out;
        bool ok = llama_q_capture_regroup_for_kv_head(state, 0, 0, 2, out);
        if (!check(test_prefix, ok, "regroup GQA: should succeed", rc)) return rc;
        if (!check(test_prefix, out.rows == 4, "regroup GQA: 2 heads × 2 tokens = 4 rows", rc)) return rc;
        if (!check(test_prefix, out.cols == 2, "regroup GQA: cols = n_embd_head = 2", rc)) return rc;

        // Order: Q head 0 all tokens, then Q head 1 all tokens
        // Row 0 = Q head 0, token 0 = [1, 2]
        if (!check(test_prefix, out(0,0) == 1.0f && out(0,1) == 2.0f, "regroup GQA: row 0 (h0,t0)", rc)) return rc;
        // Row 1 = Q head 0, token 1 = [9, 10]
        if (!check(test_prefix, out(1,0) == 9.0f && out(1,1) == 10.0f, "regroup GQA: row 1 (h0,t1)", rc)) return rc;
        // Row 2 = Q head 1, token 0 = [3, 4]
        if (!check(test_prefix, out(2,0) == 3.0f && out(2,1) == 4.0f, "regroup GQA: row 2 (h1,t0)", rc)) return rc;
        // Row 3 = Q head 1, token 1 = [11, 12]
        if (!check(test_prefix, out(3,0) == 11.0f && out(3,1) == 12.0f, "regroup GQA: row 3 (h1,t1)", rc)) return rc;

        // KV head 1 → Q heads 2,3
        llama_kv_compact_matrix out2;
        ok = llama_q_capture_regroup_for_kv_head(state, 0, 1, 2, out2);
        if (!check(test_prefix, ok, "regroup GQA kv1: should succeed", rc)) return rc;
        // Row 0 = Q head 2, token 0 = [5, 6]
        if (!check(test_prefix, out2(0,0) == 5.0f && out2(0,1) == 6.0f, "regroup GQA kv1: row 0 (h2,t0)", rc)) return rc;
        // Row 3 = Q head 3, token 1 = [15, 16]
        if (!check(test_prefix, out2(3,0) == 15.0f && out2(3,1) == 16.0f, "regroup GQA kv1: row 3 (h3,t1)", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 11: subsampling — no-op when rows <= max
    // -----------------------------------------------------------------------
    {
        llama_kv_compact_matrix mat(3, 2);
        mat(0,0) = 1.0f; mat(0,1) = 2.0f;
        mat(1,0) = 3.0f; mat(1,1) = 4.0f;
        mat(2,0) = 5.0f; mat(2,1) = 6.0f;

        llama_q_capture_subsample(mat, 5);  // max > rows → no-op
        if (!check(test_prefix, mat.rows == 3, "subsample no-op: rows should stay 3", rc)) return rc;
        if (!check(test_prefix, mat(2,0) == 5.0f, "subsample no-op: data unchanged", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 12: subsampling — exact match (rows == max)
    // -----------------------------------------------------------------------
    {
        llama_kv_compact_matrix mat(4, 2);
        for (uint32_t r = 0; r < 4; r++) {
            mat(r, 0) = (float)(r * 10);
            mat(r, 1) = (float)(r * 10 + 1);
        }

        llama_q_capture_subsample(mat, 4);  // exact match → no-op
        if (!check(test_prefix, mat.rows == 4, "subsample exact: rows should stay 4", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 13: subsampling — float stepping (Qwen3-14B-like: 1280 → 1024)
    // -----------------------------------------------------------------------
    {
        // Simulate: n_rep=5, n_tokens=256 → 1280 rows, subsample to 1024
        const uint32_t src_rows = 1280;
        const uint32_t max_q = 1024;
        const uint32_t cols = 2;
        llama_kv_compact_matrix mat(src_rows, cols);
        for (uint32_t r = 0; r < src_rows; r++) {
            mat(r, 0) = (float)r;
            mat(r, 1) = (float)(r + 10000);
        }

        llama_q_capture_subsample(mat, max_q);
        if (!check(test_prefix, mat.rows == max_q, "subsample float: rows should be 1024", rc)) return rc;
        if (!check(test_prefix, mat.data.size() == (size_t)max_q * cols, "subsample float: data size", rc)) return rc;

        // Float step = 1280/1024 = 1.25
        // Row 0: src_row = (uint32_t)(0 * 1.25) = 0
        // Row 1: src_row = (uint32_t)(1 * 1.25) = 1
        // Row 4: src_row = (uint32_t)(4 * 1.25) = 5
        // Row 1023: src_row = (uint32_t)(1023 * 1.25) = 1278
        if (!check(test_prefix, mat(0, 0) == 0.0f, "subsample float: row 0 from src 0", rc)) return rc;
        if (!check(test_prefix, mat(1, 0) == 1.0f, "subsample float: row 1 from src 1", rc)) return rc;
        if (!check(test_prefix, mat(4, 0) == 5.0f, "subsample float: row 4 from src 5", rc)) return rc;

        const uint32_t last_src = (uint32_t)(1023 * (1280.0f / 1024.0f));
        if (!check(test_prefix, mat(1023, 0) == (float)last_src, "subsample float: last row", rc)) return rc;

        // Verify NOT just truncation (integer stride=1 would give row 1023 from src 1023)
        if (!check(test_prefix, last_src < 1280, "subsample float: last src < src_rows", rc)) return rc;
        if (!check(test_prefix, last_src > 1023, "subsample float: last src > 1023 (not truncated)", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 14: subsampling — 2x reduction
    // -----------------------------------------------------------------------
    {
        llama_kv_compact_matrix mat(6, 1);
        for (uint32_t r = 0; r < 6; r++) {
            mat(r, 0) = (float)(r * 100);
        }

        llama_q_capture_subsample(mat, 3);
        if (!check(test_prefix, mat.rows == 3, "subsample 2x: rows should be 3", rc)) return rc;

        // step = 6/3 = 2.0
        // Row 0: src 0 → 0
        // Row 1: src 2 → 200
        // Row 2: src 4 → 400
        if (!check(test_prefix, mat(0, 0) == 0.0f, "subsample 2x: row 0", rc)) return rc;
        if (!check(test_prefix, mat(1, 0) == 200.0f, "subsample 2x: row 1", rc)) return rc;
        if (!check(test_prefix, mat(2, 0) == 400.0f, "subsample 2x: row 2", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 15: regroup with empty capture → returns false
    // -----------------------------------------------------------------------
    {
        llama_q_capture_state state;
        state.reset(1, 2, 2);
        // n_tokens == 0 → should return false

        llama_kv_compact_matrix out;
        bool ok = llama_q_capture_regroup_for_kv_head(state, 0, 0, 2, out);
        if (!check(test_prefix, !ok, "regroup empty: should return false", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 16: regroup with out-of-range layer → returns false
    // -----------------------------------------------------------------------
    {
        llama_q_capture_state state;
        state.reset(2, 2, 2);
        state.layers[0].n_tokens = 1;
        state.layers[0].data = {1, 2, 3, 4};

        llama_kv_compact_matrix out;
        bool ok = llama_q_capture_regroup_for_kv_head(state, 5, 0, 2, out);
        if (!check(test_prefix, !ok, "regroup OOR layer: should return false", rc)) return rc;

        ok = llama_q_capture_regroup_for_kv_head(state, -1, 0, 2, out);
        if (!check(test_prefix, !ok, "regroup negative layer: should return false", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 17: overwrite + multi-step interaction
    //          Step 1: two tensors (overwrite), Step 2: one tensor
    // -----------------------------------------------------------------------
    {
        llama_q_capture_state state;
        state.reset(1, 2, 1);  // n_embd_head=2, n_head_q=1

        // Step 1, tensor A
        float a[] = {1.0f, 2.0f};
        tensor_ctx tca;
        if (!check(test_prefix, make_f32_tensor_3d(tca, "Qcur-0", 2, 1, 1, a),
                   "ow+multi: tensor A", rc)) return rc;
        state.append_from_tensor(0, tca.t);

        // Step 1, tensor B (overwrite A)
        float b[] = {10.0f, 20.0f};
        tensor_ctx tcb;
        if (!check(test_prefix, make_f32_tensor_3d(tcb, "Qcur_normed-0", 2, 1, 1, b),
                   "ow+multi: tensor B", rc)) return rc;
        state.append_from_tensor(0, tcb.t);
        state.finalize_step();

        // Step 2, single tensor
        float c[] = {100.0f, 200.0f};
        tensor_ctx tcc;
        if (!check(test_prefix, make_f32_tensor_3d(tcc, "Qcur-0", 2, 1, 1, c),
                   "ow+multi: tensor C", rc)) return rc;
        state.append_from_tensor(0, tcc.t);
        state.finalize_step();

        if (!check(test_prefix, state.layers[0].n_tokens == 2, "ow+multi: 2 tokens total", rc)) return rc;
        if (!check(test_prefix, state.layers[0].data.size() == 4, "ow+multi: 4 floats total", rc)) return rc;
        // Token 0 should be B's data (overwrite), Token 1 should be C's data
        if (!check(test_prefix, state.layers[0].data[0] == 10.0f, "ow+multi: tok0 from B", rc)) return rc;
        if (!check(test_prefix, state.layers[0].data[1] == 20.0f, "ow+multi: tok0 from B (val1)", rc)) return rc;
        if (!check(test_prefix, state.layers[0].data[2] == 100.0f, "ow+multi: tok1 from C", rc)) return rc;
        if (!check(test_prefix, state.layers[0].data[3] == 200.0f, "ow+multi: tok1 from C (val1)", rc)) return rc;
    }

    // -----------------------------------------------------------------------
    // Test 18: quantized transposed V byte-offset arithmetic (6b-8)
    //
    // Verifies that ggml_row_size() produces correct offsets for accessing
    // individual rows in a quantized 1D tensor.  This is the invariant
    // required by compacted_prefix_copy_v_head_f32() for transposed V
    // with quantized types (Q8_0, Q4_0, etc.).
    //
    // Transposed V is a 1D tensor of n_embd_v_gqa * kv_size elements.
    // Logical row d covers all kv_size positions for embedding dim d.
    // When kv_size is block-aligned, each row is independently quantized
    // and starts at byte offset: d * ggml_row_size(type, kv_size).
    // -----------------------------------------------------------------------
    {
        const uint32_t kv_size = 64;   // must be multiple of block_size (32)
        const uint32_t n_rows  = 4;    // simulate 4 embedding dimensions
        const uint32_t n_total = n_rows * kv_size;

        const ggml_type type = GGML_TYPE_Q8_0;
        const int64_t blk = ggml_blck_size(type);
        if (!check(test_prefix, kv_size % blk == 0, "q8_0 offset: kv_size must be block-aligned", rc)) return rc;

        // Generate known data: row r, position p → value (r+1)*1000 + p
        std::vector<float> src(n_total);
        for (uint32_t r = 0; r < n_rows; ++r) {
            for (uint32_t p = 0; p < kv_size; ++p) {
                src[r * kv_size + p] = float((r + 1) * 1000 + p);
            }
        }

        // Quantize the entire buffer (blocks align at kv_size boundaries)
        const size_t total_bytes = ggml_row_size(type, n_total);
        std::vector<uint8_t> quantized(total_bytes);
        auto from_float = ggml_get_type_traits(type)->from_float_ref;
        if (!check(test_prefix, from_float != nullptr, "q8_0 offset: from_float_ref must exist", rc)) return rc;
        from_float(src.data(), quantized.data(), n_total);

        // Verify the CORRECT formula: row_bytes = ggml_row_size(type, kv_size)
        const size_t row_bytes = ggml_row_size(type, kv_size);
        if (!check(test_prefix, n_rows * row_bytes == total_bytes,
                   "q8_0 offset: n_rows * row_bytes must equal total_bytes", rc)) return rc;

        // Verify the OLD formula (kv_size * type_size) was WRONG for quantized types
        const size_t old_row_bytes = size_t(kv_size) * ggml_type_size(type);
        if (!check(test_prefix, old_row_bytes != row_bytes,
                   "q8_0 offset: old formula should differ from correct formula", rc)) return rc;

        // Read back each row using ggml_row_size() offsets and dequantize
        auto to_float_fn = ggml_get_type_traits(type)->to_float;
        if (!check(test_prefix, to_float_fn != nullptr, "q8_0 offset: to_float must exist", rc)) return rc;

        std::vector<float> row_f32(kv_size);
        for (uint32_t r = 0; r < n_rows; ++r) {
            const size_t offset = size_t(r) * row_bytes;
            to_float_fn(quantized.data() + offset, row_f32.data(), kv_size);

            for (uint32_t p = 0; p < kv_size; ++p) {
                const float expected = float((r + 1) * 1000 + p);
                const float rel_err = std::fabs(row_f32[p] - expected) / std::fabs(expected);
                if (!check(test_prefix, rel_err < 0.02f,
                           "q8_0 offset: row " + std::to_string(r) + " pos " + std::to_string(p) +
                           " expected " + std::to_string(expected) + " got " + std::to_string(row_f32[p]),
                           rc)) return rc;
            }
        }
    }

    std::fprintf(stderr, "test-kv-compact-self-study: all 18 tests passed\n");
    return 0;
}
