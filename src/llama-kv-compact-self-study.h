#pragma once

#include "llama-kv-compact-solver.h"

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
// tensor data into this structure.  Data is laid out head-major per layer:
//   layers[il].data = [head0_tok0..tokN, head1_tok0..tokN, ...]
// where each element is n_embd_head floats.
struct llama_q_capture_state {
    bool    active   = false;
    int32_t n_layers = 0;

    struct layer_q {
        uint32_t n_embd_head = 0;
        uint32_t n_head_q    = 0;
        uint32_t n_tokens    = 0;   // tokens captured so far
        std::vector<float> data;    // head-major: [head0_tok0..tokN, head1_tok0..tokN, ...]
    };
    std::vector<layer_q> layers;    // indexed by il

    // Initialize per-layer storage for n_layers layers.
    // Must be called before activating the callback.
    void reset(int32_t n_layers, uint32_t n_embd_head, uint32_t n_head_q);

    // Append tensor data for layer il.  Called from the cb_eval receive phase.
    // The overwrite strategy means this may be called multiple times per layer
    // per decode step — only the last call's data is kept for each token.
    void append_from_tensor(int32_t il, const struct ggml_tensor * t);
};

// cb_eval callback function for Q-capture.
//
// Ask phase (ask=true):  returns true for 3D+ tensors whose name starts with "Qcur"
// Receive phase (ask=false): copies GPU-synced tensor data into the capture state
//
// Must return true to continue graph computation, false to abort.
bool llama_q_capture_eval_callback(struct ggml_tensor * t, bool ask, void * user_data);
