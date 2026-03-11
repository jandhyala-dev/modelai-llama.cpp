#include "llama-kv-compact-self-study.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstring>

// ---------------------------------------------------------------------------
// llama_q_capture_state
// ---------------------------------------------------------------------------

void llama_q_capture_state::reset(int32_t n_layers_, uint32_t n_embd_head_, uint32_t n_head_q_) {
    active   = false;
    n_layers = n_layers_;
    layers.resize(n_layers_);
    for (auto & lq : layers) {
        lq.n_embd_head = n_embd_head_;
        lq.n_head_q    = n_head_q_;
        lq.n_tokens    = 0;
        lq.data.clear();
    }
}

void llama_q_capture_state::append_from_tensor(int32_t il, const struct ggml_tensor * t) {
    if (il < 0 || il >= n_layers) {
        LLAMA_LOG_WARN("q_capture: layer index %d out of range [0, %d)\n", il, n_layers);
        return;
    }

    auto & lq = layers[il];

    // The tensor is 3D post-RoPE: [n_embd_head, n_head_q, n_tokens]
    // or possibly 4D with ne[3]=1.  We read the first 3 dims.
    const uint32_t d0 = (uint32_t)t->ne[0];  // n_embd_head
    const uint32_t d1 = (uint32_t)t->ne[1];  // n_head_q
    const uint32_t d2 = (uint32_t)t->ne[2];  // n_tokens (usually 1 during autoregressive)

    if (d0 != lq.n_embd_head || d1 != lq.n_head_q) {
        // Mismatched dimensions — this is a pre-reshape 2D projection or
        // a different Qcur variant.  Skip it.
        return;
    }

    // For the overwrite strategy: during autoregressive generation, d2 == 1
    // (one token at a time).  We append each token's Q data.
    //
    // Data layout target: head-major — [head0_tok0..tokN, head1_tok0..tokN, ...]
    // But the tensor is [n_embd_head, n_head_q, 1] — we need to scatter each
    // head's n_embd_head floats to the right position in the head-major layout.
    //
    // For single-token decode (d2 == 1):
    //   For each head h: append n_embd_head floats to position
    //   [h * (n_tokens+1) ... ] — but that requires shifting existing data.
    //
    // Simpler approach for single-token: store token-major during capture,
    // then rearrange in regroup.  But the plan says head-major.
    //
    // Actually for efficiency: store as flat per-token blocks during capture.
    // Each decode step appends [n_embd_head * n_head_q] floats (one block per token).
    // Regrouping (slice 6b-3) will rearrange into per-KV-head matrices.
    //
    // This is token-major: [tok0_head0..headN, tok1_head0..headN, ...]
    // which is the natural tensor layout and avoids scattered inserts.

    const size_t floats_per_token = (size_t)d0 * d1;
    const size_t new_floats = floats_per_token * d2;

    // Ensure the tensor is F32
    if (t->type != GGML_TYPE_F32) {
        LLAMA_LOG_WARN("q_capture: expected F32 tensor, got type %d\n", (int)t->type);
        return;
    }

    const size_t old_size = lq.data.size();
    lq.data.resize(old_size + new_floats);

    // Copy from GPU (or host) into our buffer
    ggml_backend_tensor_get(t, lq.data.data() + old_size, 0, new_floats * sizeof(float));

    lq.n_tokens += d2;
}

// ---------------------------------------------------------------------------
// cb_eval callback
// ---------------------------------------------------------------------------

// Parse layer index from tensor name suffix.
// Expected format: "Qcur-5", "Qcur_normed-12", "Qcur_scaled-0", etc.
// Returns -1 if no valid suffix found.
static int32_t parse_layer_index(const char * name) {
    // Find last '-' in name
    const char * dash = strrchr(name, '-');
    if (!dash || dash == name) {
        return -1;
    }

    // Parse integer after dash
    char * end = nullptr;
    long val = strtol(dash + 1, &end, 10);
    if (end == dash + 1 || *end != '\0') {
        return -1;
    }

    return (int32_t)val;
}

// ---------------------------------------------------------------------------
// GQA regrouping + subsampling
// ---------------------------------------------------------------------------

bool llama_q_capture_regroup_for_kv_head(
        const llama_q_capture_state & q_state,
        int32_t   il,
        uint32_t  h_kv,
        uint32_t  n_head_kv,
        llama_kv_compact_matrix & out) {

    if (il < 0 || il >= q_state.n_layers) {
        return false;
    }

    const auto & lq = q_state.layers[il];
    if (lq.n_tokens == 0) {
        return false;
    }

    const uint32_t n_head_q    = lq.n_head_q;
    const uint32_t n_embd_head = lq.n_embd_head;
    const uint32_t n_tokens    = lq.n_tokens;

    // GQA repetition factor: how many Q heads map to each KV head
    // For non-GQA models: n_rep == 1 (pass-through)
    const uint32_t n_rep = n_head_q / n_head_kv;

    // Q heads for this KV head: [h_kv * n_rep, (h_kv + 1) * n_rep)
    const uint32_t q_head_start = h_kv * n_rep;
    const uint32_t q_head_end   = q_head_start + n_rep;

    if (q_head_end > n_head_q) {
        LLAMA_LOG_WARN("q_capture regroup: q_head_end %u > n_head_q %u\n", q_head_end, n_head_q);
        return false;
    }

    // Output: [n_rep * n_tokens, n_embd_head]
    const uint32_t total_rows = n_rep * n_tokens;
    out.resize(total_rows, n_embd_head);

    // Capture data is token-major:
    //   data[tok * (n_head_q * n_embd_head) + head * n_embd_head ... + n_embd_head]
    //
    // We iterate over each Q head in [q_head_start, q_head_end), and for each
    // captured token, copy its n_embd_head floats into the output matrix.
    const size_t head_stride = (size_t)n_embd_head;
    const size_t tok_stride  = (size_t)n_head_q * n_embd_head;

    uint32_t out_row = 0;
    for (uint32_t qh = q_head_start; qh < q_head_end; qh++) {
        for (uint32_t tok = 0; tok < n_tokens; tok++) {
            const float * src = lq.data.data() + tok * tok_stride + qh * head_stride;
            std::memcpy(out.row(out_row), src, n_embd_head * sizeof(float));
            out_row++;
        }
    }

    return true;
}

void llama_q_capture_subsample(
        llama_kv_compact_matrix & mat,
        uint32_t max_queries) {

    if (mat.rows <= max_queries) {
        return;  // no subsampling needed
    }

    // Uniform stride subsampling: pick every stride-th row
    const uint32_t stride = mat.rows / max_queries;
    const uint32_t cols   = mat.cols;

    uint32_t dst_row = 0;
    for (uint32_t src_row = 0; dst_row < max_queries; src_row += stride, dst_row++) {
        if (dst_row != src_row) {
            std::memcpy(mat.row(dst_row), mat.row(src_row), cols * sizeof(float));
        }
    }

    // Shrink: update row count and trim data
    mat.rows = max_queries;
    mat.data.resize((size_t)max_queries * cols);
}

// ---------------------------------------------------------------------------
// cb_eval callback
// ---------------------------------------------------------------------------

bool llama_q_capture_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = static_cast<llama_q_capture_state *>(user_data);

    if (!state || !state->active) {
        return true;  // not active — pass through, don't abort
    }

    // Filter: name must start with "Qcur"
    if (strncmp(t->name, "Qcur", 4) != 0) {
        return true;  // not interested, continue
    }

    if (ask) {
        // Ask phase: accept 3D+ tensors with Qcur prefix
        return ggml_n_dims(t) >= 3;
    }

    // Receive phase: copy tensor data into capture state
    const int32_t il = parse_layer_index(t->name);
    if (il < 0) {
        return true;  // couldn't parse layer — skip, continue
    }

    state->append_from_tensor(il, t);

    return true;  // continue graph computation
}
