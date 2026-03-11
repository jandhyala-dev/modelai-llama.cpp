#pragma once

#include "llama-kv-compact-solver.h"
#include "llama.h"

#include "ggml.h"

#include <cstdint>
#include <cstring>
#include <vector>

// Configuration for self-study Q-capture generation
struct llama_kv_compact_self_study_config {
    uint32_t n_generate              = 256;    // continuation tokens to generate
    uint32_t max_queries_per_kv_head = 1024;   // subsample limit after GQA regrouping
    int      nnls_iters              = 64;     // solver iterations
    float    lambda                  = 1e-6f;  // solver regularization
    float    temperature             = 0.0f;   // sampling temp (0 = greedy)
};

// Statistics output
struct llama_kv_compact_self_study_stats {
    double   generation_time_ms   = 0.0;
    double   q_capture_time_ms    = 0.0;
    double   solver_time_ms       = 0.0;
    uint32_t n_tokens_generated   = 0;
    uint32_t n_queries_per_head   = 0;
    uint32_t n_prefix_tokens      = 0;
    uint32_t n_selected_tokens    = 0;
};

// Q-capture state (user_data for cb_eval callback)
//
// During autoregressive generation, the cb_eval callback writes post-RoPE Q
// tensor data into this structure.  Data is laid out token-major per layer:
//   layers[il].data = [tok0_head0..headN, tok1_head0..headN, ...]
// where each element is n_embd_head floats.  The regroup functions convert
// this to per-KV-head matrices for the solver.
struct llama_q_capture_state {
    bool    active   = false;
    int32_t n_layers = 0;

    struct layer_q {
        uint32_t n_embd_head  = 0;
        uint32_t n_head_q     = 0;
        uint32_t n_tokens     = 0;   // committed tokens
        bool     has_pending  = false;  // true if current step wrote data
        size_t   pending_off  = 0;      // offset of pending data in data[]
        std::vector<float> data;    // token-major: [tok0_head0..headN, tok1_head0..headN, ...]
    };
    std::vector<layer_q> layers;    // indexed by il

    // Initialize per-layer storage for n_layers layers.
    // n_reserve: expected number of tokens (pre-allocates to avoid hot-path realloc).
    // Must be called before activating the callback.
    void reset(int32_t n_layers, uint32_t n_embd_head, uint32_t n_head_q,
               uint32_t n_reserve = 0);

    // Write tensor data for layer il.  Called from the cb_eval receive phase.
    //
    // Overwrite strategy: multiple 3D Qcur-prefixed tensors may fire per layer
    // per decode step (e.g. Qcur after RoPE, Qcur_normed after norm).  Only
    // the last one per step is kept.  First call for a layer in a step appends;
    // subsequent calls overwrite at the same offset.
    void append_from_tensor(int32_t il, const struct ggml_tensor * t);

    // Finalize the current decode step: commit pending data across all layers.
    // Must be called by the generation loop after each llama_decode().
    void finalize_step();
};

// cb_eval callback function for Q-capture.
//
// Ask phase (ask=true):  returns true for 3D+ tensors whose name starts with "Qcur"
// Receive phase (ask=false): copies GPU-synced tensor data into the capture state
//
// Must return true to continue graph computation, false to abort.
bool llama_q_capture_eval_callback(struct ggml_tensor * t, bool ask, void * user_data);

// ---------------------------------------------------------------------------
// Autoregressive generation loop (slice 6b-4)
// ---------------------------------------------------------------------------

struct llama_context;

// Generate n_generate continuation tokens from the current context state,
// capturing post-RoPE Q tensors into q_state via cb_eval.
//
// Caller contract:
//   - Context must have been prefilled (logits available from last decode)
//   - q_state must be initialized via reset() before this call
//   - KV cache must have room for n_generate additional tokens
//
// The function:
//   1. Checks KV capacity (n_ctx - current_pos >= n_generate)
//   2. Saves existing cb_eval, installs Q-capture callback
//   3. Seeds first token from last prefill logits (argmax)
//   4. Runs autoregressive loop: batch_get_one → decode → finalize_step → argmax
//   5. Restores previous cb_eval
//   6. Removes generated tokens from memory (llama_memory_seq_rm)
//
// EOS tokens are ignored — generation continues for Q diversity.
// On decode failure, breaks with partial capture (still usable).
// Returns true if at least one token was generated.
bool llama_kv_compact_self_study_generate(
        struct llama_context * ctx,
        llama_q_capture_state & q_state,
        uint32_t n_generate,
        llama_seq_id seq_id);

// ---------------------------------------------------------------------------
// GQA regrouping + subsampling (slice 6b-3)
// ---------------------------------------------------------------------------

// Regroup captured Q vectors for a single KV head.
//
// From token-major capture data, extracts Q vectors for the n_rep Q heads
// that map to KV head h_kv, producing a matrix of [n_rep * n_tokens, n_embd_head].
//
// For Qwen3-14B (n_head_q=40, n_head_kv=8, n_rep=5):
//   256 tokens * 5 Q heads = 1280 rows of 128 floats.
//
// Returns false if layer has no captured data.
bool llama_q_capture_regroup_for_kv_head(
        const llama_q_capture_state & q_state,
        int32_t   il,
        uint32_t  h_kv,
        uint32_t  n_head_kv,
        llama_kv_compact_matrix & out);

// Subsample a Q matrix to at most max_queries rows via uniform stride.
// Operates in-place on the input matrix.
// No-op if rows <= max_queries.
void llama_q_capture_subsample(
        llama_kv_compact_matrix & mat,
        uint32_t max_queries);
