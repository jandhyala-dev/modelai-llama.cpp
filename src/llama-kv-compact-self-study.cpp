#include "llama-kv-compact-self-study.h"
#include "llama-kv-compact-select.h"
#include "llama-kv-compact-pipeline.h"
#include "llama-kv-compact-solver-metal.h"
#include "llama-kv-cache.h"
#include "llama-kv-compacted-prefix.h"
#include "llama-context.h"
#include "llama-model.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>

// ---------------------------------------------------------------------------
// llama_q_capture_state
// ---------------------------------------------------------------------------

void llama_q_capture_state::reset(int32_t n_layers_, uint32_t n_embd_head_, uint32_t n_head_q_,
                                  uint32_t n_reserve) {
    active   = false;
    n_layers = n_layers_;
    layers.resize(n_layers_);
    const size_t floats_per_token = (size_t)n_embd_head_ * n_head_q_;
    for (auto & lq : layers) {
        lq.n_embd_head  = n_embd_head_;
        lq.n_head_q     = n_head_q_;
        lq.n_tokens     = 0;
        lq.has_pending  = false;
        lq.pending_off  = 0;
        lq.data.clear();
        lq.n_dim_mismatches = 0;
        lq.last_accepted_tensor_name.clear();
        if (n_reserve > 0) {
            lq.data.reserve(floats_per_token * n_reserve);
        }
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
        LLAMA_LOG_WARN("q_capture: skipping tensor '%s' — dims [%d,%d] != expected [%d,%d]\n",
                       t->name, (int)d0, (int)d1, (int)lq.n_embd_head, (int)lq.n_head_q);
        lq.n_dim_mismatches++;
        return;
    }

    // Ensure the tensor is F32
    if (t->type != GGML_TYPE_F32) {
        LLAMA_LOG_WARN("q_capture: expected F32 tensor, got type %d\n", (int)t->type);
        return;
    }

    lq.last_accepted_tensor_name = t->name;

    const size_t floats_per_token = (size_t)d0 * d1;
    const size_t new_floats = floats_per_token * d2;

    // Overwrite strategy: multiple 3D Qcur-prefixed tensors may fire per layer
    // per decode step (e.g. Qcur after RoPE, then Qcur_normed after norm).
    // Graph nodes execute in topological order, so the last one is always the
    // final post-processed version passed to build_attn().
    //
    // First call for this layer in the current step: append new data.
    // Subsequent calls: overwrite at the same offset (last one wins).
    if (lq.has_pending) {
        // Overwrite the pending data at the same offset
        ggml_backend_tensor_get(t, lq.data.data() + lq.pending_off, 0, new_floats * sizeof(float));
    } else {
        // First Qcur for this layer in this step: append
        const size_t old_size = lq.data.size();
        lq.data.resize(old_size + new_floats);
        ggml_backend_tensor_get(t, lq.data.data() + old_size, 0, new_floats * sizeof(float));

        lq.pending_off = old_size;
        lq.has_pending = true;
    }

    // Note: n_tokens is NOT incremented here — finalize_step() does that
    // to avoid double-counting from multiple overwrites.
}

void llama_q_capture_state::finalize_step() {
    for (auto & lq : layers) {
        if (lq.has_pending) {
            // Commit: count the token(s) written in this step.
            // For autoregressive d2 == 1; if a future prefill path uses d2 > 1,
            // the floats already in data[] are correct — just count them.
            const size_t floats_per_token = (size_t)lq.n_embd_head * lq.n_head_q;
            const size_t pending_floats   = lq.data.size() - lq.pending_off;
            const uint32_t d2 = (uint32_t)(pending_floats / floats_per_token);
            lq.n_tokens += d2;
            lq.has_pending = false;
        }
    }
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
// Autoregressive generation loop (slice 6b-4)
// ---------------------------------------------------------------------------

// Sample a token from logits with temperature.
// temperature == 0 → argmax (greedy).
// temperature > 0 → softmax(logits/temp) with random sampling.
static llama_token sample_with_temperature(
        const float * logits,
        int32_t n_vocab,
        float temperature,
        std::mt19937 & rng) {
    if (temperature <= 0.0f) {
        return (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
    }

    // Softmax with temperature
    const float inv_temp = 1.0f / temperature;
    float max_logit = *std::max_element(logits, logits + n_vocab);

    // Guard: if all logits are -inf, softmax produces NaN — fall back to argmax.
    if (!std::isfinite(max_logit)) {
        return (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
    }

    std::vector<float> probs(n_vocab);
    float sum = 0.0f;
    for (int32_t i = 0; i < n_vocab; ++i) {
        probs[i] = std::exp((logits[i] - max_logit) * inv_temp);
        sum += probs[i];
    }

    // Guard: degenerate softmax (sum == 0 or NaN) — fall back to argmax.
    if (sum <= 0.0f || !std::isfinite(sum)) {
        return (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
    }

    const float inv_sum = 1.0f / sum;
    for (int32_t i = 0; i < n_vocab; ++i) {
        probs[i] *= inv_sum;
    }

    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    return (llama_token)dist(rng);
}

bool llama_kv_compact_self_study_generate(
        struct llama_context * ctx,
        llama_q_capture_state & q_state,
        uint32_t n_generate,
        llama_seq_id seq_id,
        float temperature,
        uint32_t seed) {

    // llama_batch_get_one() hardcodes sequence 0 (llama.h).
    // Until we build batches manually, enforce this precondition.
    GGML_ASSERT(seq_id == 0 && "self-study generation requires seq_id == 0 (llama_batch_get_one limitation)");

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab     = llama_vocab_n_tokens(vocab);

    // 1. KV capacity pre-check: current pos + n_generate must fit in n_ctx
    llama_memory_t mem = llama_get_memory(ctx);
    const llama_pos pos_max  = llama_memory_seq_pos_max(mem, seq_id);
    const uint32_t  n_ctx    = llama_n_ctx(ctx);

    if (pos_max < 0) {
        LLAMA_LOG_WARN("self-study: no tokens in seq %d\n", seq_id);
        return false;
    }

    // pos_max is inclusive, so used positions = pos_max + 1
    if ((uint32_t)(pos_max + 1) + n_generate > n_ctx) {
        LLAMA_LOG_WARN("self-study: insufficient KV capacity (%d used + %u needed > %u total)\n",
                       pos_max + 1, n_generate, n_ctx);
        return false;
    }

    // 2. Save existing cb_eval and install Q-capture callback
    const auto & cparams = ctx->get_cparams();
    auto prev_cb = cparams.cb_eval;
    auto prev_ud = cparams.cb_eval_user_data;

    q_state.active = true;
    ctx->set_eval_callback(llama_q_capture_eval_callback, &q_state);

    // 3. Seed token from last prefill logits
    float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) {
        LLAMA_LOG_ERROR("self-study: no logits available from prefill\n");
        q_state.active = false;
        ctx->set_eval_callback(prev_cb, prev_ud);
        return false;
    }

    // Per-round seed provides diversity across rounds (M-01/m-02 fix).
    std::mt19937 rng(seed);
    llama_token token = sample_with_temperature(logits, n_vocab, temperature, rng);

    // 4. Autoregressive generation loop
    uint32_t n_generated = 0;
    for (uint32_t i = 0; i < n_generate; i++) {
        llama_batch batch = llama_batch_get_one(&token, 1);

        if (llama_decode(ctx, batch) != 0) {
            LLAMA_LOG_ERROR("self-study: decode failed at step %u\n", i);
            // Clear any pending state from partial cb_eval callbacks
            // that may have fired before the decode failure.
            for (auto & lq : q_state.layers) {
                if (lq.has_pending) {
                    // Discard uncommitted data: revert to pre-pending size
                    lq.data.resize(lq.pending_off);
                    lq.has_pending = false;
                }
            }
            break;
        }

        // cb_eval fires during decode, capturing Q tensors.
        // Commit this step's pending data before the next decode.
        q_state.finalize_step();
        n_generated++;

        // Sample next token
        logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            LLAMA_LOG_ERROR("self-study: no logits at step %u\n", i);
            break;
        }
        token = sample_with_temperature(logits, n_vocab, temperature, rng);

        // Do NOT stop on EOS — continue for Q diversity (text is discarded)
    }

    // 5. Restore previous cb_eval
    q_state.active = false;
    ctx->set_eval_callback(prev_cb, prev_ud);

    // 6. Remove generated tokens from memory
    //    pos_max + 1 is the first generated position; remove [pos_max+1, pos_max+1+n_generated)
    if (n_generated > 0) {
        const llama_pos gen_start = pos_max + 1;
        const llama_pos gen_end   = gen_start + (llama_pos)n_generated;
        llama_memory_seq_rm(mem, seq_id, gen_start, gen_end);
    }

    LLAMA_LOG_INFO("self-study: captured Q from %u tokens (temp=%.1f, seq %d)\n",
                   n_generated, temperature, seq_id);
    return n_generated > 0;
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
    GGML_ASSERT(n_head_q % n_head_kv == 0 && "GQA requires n_head_q divisible by n_head_kv");
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

    // Float-stepping subsampling: uniformly sample max_queries rows.
    // Integer stride degenerates to truncation when rows is between
    // max_queries and 2*max_queries (e.g. Qwen3-14B: 1280/1024 = stride 1).
    // Float step ensures all GQA head groups are represented proportionally.
    const uint32_t cols = mat.cols;
    const float step = (float)mat.rows / (float)max_queries;

    uint32_t dst_row = 0;
    for (uint32_t i = 0; i < max_queries; i++) {
        const uint32_t src_row = (uint32_t)(i * step);
        if (dst_row != src_row) {
            std::memcpy(mat.row(dst_row), mat.row(src_row), cols * sizeof(float));
        }
        dst_row++;
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
        // Ask phase: accept all Qcur-prefixed tensors.
        // ggml_n_dims(t) >= 3 does NOT work for single-token decode because
        // ne[2]=1 makes ggml report 2D even for post-RoPE [n_embd_head, n_head_q, 1].
        // The receive phase (append_from_tensor) validates dimensions and rejects
        // pre-reshape 2D projections where ne[0]=n_embd, ne[1]=n_tokens.
        return true;
    }

    // Receive phase: copy tensor data into capture state
    const int32_t il = parse_layer_index(t->name);
    if (il < 0) {
        return true;  // couldn't parse layer — skip, continue
    }

    state->append_from_tensor(il, t);

    return true;  // continue graph computation
}

// ---------------------------------------------------------------------------
// Self-study pipeline entry point (slice 6b-5)
// ---------------------------------------------------------------------------

// Compute mean L2 row norm across all rows of a matrix.
static float compute_row_norm_mean(const llama_kv_compact_matrix & m) {
    if (m.rows == 0 || m.cols == 0) return 0.0f;
    double sum = 0.0;
    for (uint32_t r = 0; r < m.rows; ++r) {
        const float * row = m.row(r);
        float norm_sq = 0.0f;
        for (uint32_t c = 0; c < m.cols; ++c) {
            norm_sq += row[c] * row[c];
        }
        sum += std::sqrt(norm_sq);
    }
    return (float)(sum / m.rows);
}

// Gather selected rows from a source matrix into a destination matrix.
static bool gather_matrix_rows(
        const llama_kv_compact_matrix & src,
        const std::vector<uint32_t> & row_indices,
        llama_kv_compact_matrix & dst) {
    dst.resize((uint32_t)row_indices.size(), src.cols);
    for (size_t i = 0; i < row_indices.size(); ++i) {
        const uint32_t src_row = row_indices[i];
        if (src_row >= src.rows) {
            return false;
        }
        std::memcpy(dst.row((uint32_t)i), src.row(src_row), (size_t)src.cols * sizeof(float));
    }
    return true;
}

// Write solver output into the compacted prefix store's quantized payload.
static void write_compacted_payload(
        std::vector<uint8_t> & dst,
        ggml_type type,
        uint32_t n_tokens,
        uint32_t head,
        uint32_t dim,
        const llama_kv_compact_matrix & rows) {
    GGML_ASSERT(rows.rows == n_tokens);
    GGML_ASSERT(rows.cols == dim);

    auto from_float = ggml_get_type_traits(type)->from_float_ref;
    GGML_ASSERT(from_float != nullptr);

    const size_t token_bytes = ggml_row_size(type, dim);
    for (uint32_t token = 0; token < n_tokens; ++token) {
        void * dst_ptr = dst.data() + (size_t(head) * n_tokens + token) * token_bytes;
        from_float(rows.row(token), dst_ptr, dim);
    }
}

bool llama_kv_compact_self_study_from_live_kv(
        struct llama_context * ctx,
        llama_kv_cache       & kv,
        llama_seq_id           seq_id,
        uint32_t               target_tokens,
        llama_pos              live_suffix_pos0,
        const llama_kv_compact_self_study_config & config,
        llama_kv_compact_self_study_stats * stats,
        llama_pos p0) {

    if (seq_id < 0 || target_tokens == 0 || live_suffix_pos0 <= p0) {
        return false;
    }

    // --- Validate cache state ---
    std::vector<llama_pos> prefix_positions;
    if (!kv.compacted_prefix_seq_positions(seq_id, p0, live_suffix_pos0, prefix_positions)) {
        return false;
    }
    if (prefix_positions.empty()) {
        return false;
    }

    const auto & layouts = kv.get_compacted_prefix()->get_layouts();
    if (layouts.empty()) {
        return false;
    }

    const uint32_t n_prefix_tokens = (uint32_t)prefix_positions.size();
    const uint32_t n_selected = std::min<uint32_t>(target_tokens, n_prefix_tokens);

    // --- Phase 1: Multi-round Q-capture generation (V2 — GAP-03) ---
    const auto & model_hparams = ctx->get_model().hparams;
    const uint32_t n_layer     = model_hparams.n_layer;
    const uint32_t n_embd_head = model_hparams.n_embd_head_k(0);
    const uint32_t n_head_q    = model_hparams.n_head(0);

    uint32_t eff_n_rounds = std::max(1u, std::min(config.n_rounds, (uint32_t) LLAMA_KV_COMPACT_MAX_ROUNDS));
    uint32_t eff_n_generate = config.n_generate;

    // Memory budget guard (M-01 fix).
    // Q-capture stores all layers simultaneously during decode:
    //   n_layer * n_tokens * n_head_q * n_embd_head * sizeof(float)
    // For Qwen3-14B with defaults (6000 tokens): 40*6000*40*128*4 = 4.9 GB.
    // Auto-reduce n_generate/n_rounds to stay within budget.
    if (config.max_q_capture_mb > 0) {
        const size_t bytes_per_token = (size_t)n_layer * n_head_q * n_embd_head * sizeof(float);
        const size_t budget_bytes = (size_t)config.max_q_capture_mb * 1024 * 1024;
        const size_t requested_bytes = bytes_per_token * eff_n_generate * eff_n_rounds;

        if (requested_bytes > budget_bytes && bytes_per_token > 0) {
            const uint32_t max_total_tokens = (uint32_t)(budget_bytes / bytes_per_token);

            // Try to keep n_rounds, reduce n_generate first.
            eff_n_generate = std::max(1u, max_total_tokens / eff_n_rounds);
            if (eff_n_generate < 100 && eff_n_rounds > 1) {
                // Per-round tokens too low — reduce rounds instead.
                eff_n_rounds = std::max(1u, max_total_tokens / config.n_generate);
                eff_n_generate = std::max(1u, max_total_tokens / eff_n_rounds);
            }
            LLAMA_LOG_WARN("self-study: Q-capture memory guard — reduced from %u×%u to %u×%u tokens "
                           "(%u MB budget, %.2f MB/token)\n",
                           config.n_generate, config.n_rounds, eff_n_generate, eff_n_rounds,
                           config.max_q_capture_mb,
                           (float)bytes_per_token / (1024.0f * 1024.0f));
        }
    }

    const uint32_t total_generate = eff_n_generate * eff_n_rounds;

    // Q-capture sufficiency check: if estimated queries < target_tokens, the
    // NNLS solver becomes underdetermined and produces NaN/extreme weights.
    // Fall back to the proven select pipeline which uses K-as-surrogate-Q.
    {
        const uint32_t n_head_kv = layouts[0].n_head_kv;
        const uint32_t n_rep = n_head_q / std::max(1u, n_head_kv);
        const uint32_t estimated_queries = total_generate * n_rep;
        if (estimated_queries < target_tokens) {
            LLAMA_LOG_WARN("self-study: Q-capture insufficient (%u queries < %u targets) — "
                           "falling back to select pipeline for stable quality\n",
                           estimated_queries, target_tokens);
            return llama_kv_compact_select_from_live_kv(kv, seq_id, target_tokens, live_suffix_pos0,
                                                         nullptr, p0);
        }
    }

    llama_q_capture_state q_state;
    q_state.reset((int32_t)n_layer, n_embd_head, n_head_q, total_generate);

    const auto t_gen_start = std::chrono::steady_clock::now();
    bool any_generated = false;
    for (uint32_t round = 0; round < eff_n_rounds; ++round) {
        const float temp = config.temperatures[round];
        if (llama_kv_compact_self_study_generate(ctx, q_state, eff_n_generate, seq_id, temp, 42 + round)) {
            any_generated = true;
        } else if (round == 0) {
            // First round must succeed; subsequent rounds are best-effort.
            return false;
        }
    }
    if (!any_generated) {
        return false;
    }
    const auto t_gen_end = std::chrono::steady_clock::now();

    // Q-capture layer diagnostics.
    if (stats) {
        stats->n_layers_with_q = 0;
        stats->n_dim_mismatches = 0;
        for (int il = 0; il < q_state.n_layers; ++il) {
            if (q_state.layers[il].n_tokens > 0) {
                stats->n_layers_with_q++;
            }
            stats->n_dim_mismatches += q_state.layers[il].n_dim_mismatches;
        }
    }

    // Post-capture sufficiency verification: use ACTUAL captured token count
    // rather than pre-capture estimate (which may be wrong for some models).
    {
        uint32_t actual_tokens = 0;
        for (int il = 0; il < q_state.n_layers; ++il) {
            if (q_state.layers[il].n_tokens > 0) {
                actual_tokens = q_state.layers[il].n_tokens;
                break;
            }
        }
        const uint32_t n_head_kv_l0 = layouts[0].n_head_kv;
        const uint32_t n_rep = n_head_q / std::max(1u, n_head_kv_l0);
        const uint32_t actual_queries = actual_tokens * n_rep;
        if (actual_queries < target_tokens) {
            LLAMA_LOG_WARN("self-study: post-capture insufficiency (%u tokens captured, "
                           "%u queries < %u targets) — falling back to select pipeline\n",
                           actual_tokens, actual_queries, target_tokens);
            return llama_kv_compact_select_from_live_kv(kv, seq_id, target_tokens, live_suffix_pos0,
                                                         nullptr, p0);
        }
    }

    // --- Phase 2: Score + select ---
    // For each layer/head: regroup Q, subsample, extract K, accumulate scores.
    // Use Metal GPU when available for attention score computation.
    auto * metal_ctx = llama_kv_compact_metal_create();

    std::vector<float> aggregate_scores(n_prefix_tokens, 0.0f);

    struct head_cache_entry {
        llama_kv_compact_matrix k;       // [n_prefix x n_embd_head_k]
        llama_kv_compact_matrix queries; // [n_queries x n_embd_head_k]
    };
    std::vector<std::vector<head_cache_entry>> layer_cache(layouts.size());

    // Diagnostic accumulators.
    double q_norm_sum = 0.0, k_norm_sum = 0.0;
    double beta_norm_sum = 0.0, beta_sparsity_sum = 0.0;
    double fit_residual_sum = 0.0;
    uint32_t n_heads_seen = 0;
    uint32_t n_beta_heads_seen = 0;

    const auto t_query_start = std::chrono::steady_clock::now();
    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        layer_cache[li].resize(layout.n_head_kv);

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            auto & entry = layer_cache[li][head];

            // Extract K from live cache
            std::vector<float> k_data;
            if (!kv.compacted_prefix_copy_k_head_f32(
                        (int32_t)layout.layer_id, seq_id, head,
                        prefix_positions, k_data)) {
                return false;
            }
            entry.k.resize(n_prefix_tokens, layout.n_embd_head_k);
            entry.k.data = std::move(k_data);

            // Regroup captured Q for this KV head
            if (!llama_q_capture_regroup_for_kv_head(
                        q_state, (int32_t)layout.layer_id,
                        head, layout.n_head_kv, entry.queries)) {
                return false;
            }

            // Subsample to max_queries_per_kv_head
            llama_q_capture_subsample(entry.queries, config.max_queries_per_kv_head);

            // Accumulate raw Q/K norms for diagnostics (pre-normalization).
            if (stats) {
                q_norm_sum += compute_row_norm_mean(entry.queries);
                k_norm_sum += compute_row_norm_mean(entry.k);
                n_heads_seen++;
            }

            // Normalize Q to match K scale (fix Q/K norm mismatch in self-study).
            // OMP uses K-as-surrogate-Q so norms match by construction.
            // Real Q from autoregressive generation has different norm due to
            // separate W_q/W_k projections.  Without normalization, the
            // attention softmax peaks incorrectly and the NNLS solver produces
            // extreme beta weights (observed: q_norm~16, k_norm~28, beta_norm~356).
            {
                float q_norm = compute_row_norm_mean(entry.queries);
                float k_norm = compute_row_norm_mean(entry.k);
                if (q_norm > 1e-8f && k_norm > 1e-8f) {
                    float scale = k_norm / q_norm;
                    for (size_t i = 0; i < entry.queries.data.size(); ++i) {
                        entry.queries.data[i] *= scale;
                    }
                }
            }

            // Accumulate attention scores (GPU when available, CPU fallback).
            if (metal_ctx) {
                std::vector<float> head_scores(n_prefix_tokens, 0.0f);
                if (llama_kv_compact_metal_attention_scores(
                            metal_ctx,
                            entry.queries.data.data(), entry.queries.rows, entry.queries.cols,
                            entry.k.data.data(), entry.k.rows,
                            head_scores.data())) {
                    for (uint32_t ki = 0; ki < n_prefix_tokens; ++ki) {
                        aggregate_scores[ki] += head_scores[ki];
                    }
                } else {
                    // GPU failed — fall back to CPU for this head.
                    llama_kv_compact_accumulate_attention_scores(
                            entry.queries, entry.k, aggregate_scores);
                }
            } else {
                llama_kv_compact_accumulate_attention_scores(
                        entry.queries, entry.k, aggregate_scores);
            }
        }
    }
    const auto t_query_end = std::chrono::steady_clock::now();

    llama_kv_compact_metal_free(metal_ctx);
    metal_ctx = nullptr;

    // Global selection: top-k across aggregated scores
    const std::vector<uint32_t> selected_local = llama_kv_compact_select_topk(aggregate_scores, n_selected);
    std::vector<llama_pos> selected_positions;
    selected_positions.reserve(selected_local.size());
    for (uint32_t idx : selected_local) {
        selected_positions.push_back(prefix_positions[idx]);
    }

    // Configure compacted prefix store
    const llama_pos seq_max = kv.seq_pos_max(seq_id);
    const uint32_t logical_token_count = seq_max >= 0 ? (uint32_t)(seq_max + 1) : (uint32_t)live_suffix_pos0;
    if (!kv.compacted_prefix_configure(seq_id, logical_token_count, selected_positions, live_suffix_pos0)) {
        return false;
    }

    auto * seq = kv.get_compacted_prefix()->get_seq(seq_id);
    if (seq == nullptr || !seq->enabled || seq->layers.size() != layouts.size()) {
        kv.compacted_prefix_clear(seq_id, true);
        return false;
    }

    // --- Phase 3: Solver (reuses cached K + queries from Phase 2) ---
    //
    // After compacted_prefix_configure(), any failure must roll back the
    // store to avoid leaving a partially written sequence enabled.
    const auto t_solver_start = std::chrono::steady_clock::now();
    const llama_kv_compact_solver_opts solver_opts = {
        /* lambda           */ config.lambda,
        /* nnls_iters       */ config.nnls_iters,
        /* nnls_lower_bound */ 1e-12f,
        /* nnls_upper_bound */ 100.0f,   // prevent extreme betas from underdetermined NNLS
    };

    bool solver_ok = true;
    uint32_t actual_queries_per_head = 0;

    for (size_t li = 0; li < layouts.size() && solver_ok; ++li) {
        const auto & layout = layouts[li];
        auto & dst_layer = seq->layers[li];

        for (uint32_t head = 0; head < layout.n_head_kv && solver_ok; ++head) {
            const auto & entry = layer_cache[li][head];

            // Track actual query count (after regroup + subsample)
            if (li == 0 && head == 0) {
                actual_queries_per_head = entry.queries.rows;
            }

            // V extraction (not cached in Phase 2 to save memory)
            std::vector<float> full_v_data;
            if (!kv.compacted_prefix_copy_v_head_f32(
                        (int32_t)layout.layer_id, seq_id, head,
                        prefix_positions, full_v_data)) {
                solver_ok = false;
                break;
            }
            llama_kv_compact_matrix full_v(n_prefix_tokens, layout.n_embd_head_v);
            full_v.data = std::move(full_v_data);

            // Gather selected K rows
            llama_kv_compact_matrix compacted_k;
            if (!gather_matrix_rows(entry.k, selected_local, compacted_k)) {
                solver_ok = false;
                break;
            }

            // NNLS beta fitting
            std::vector<float> beta;
            float head_residual = 0.0f;
            if (!llama_kv_compact_fit_beta(entry.queries, entry.k,
                                            compacted_k, solver_opts,
                                            beta, stats ? &head_residual : nullptr)) {
                solver_ok = false;
                break;
            }

            // Beta quality check (MIT zerobeta fallback).
            // Extreme beta norms indicate the NNLS solver produced poor weights,
            // making the softmax design matrix ill-conditioned.  Zero the betas
            // and use direct V selection instead (MIT: zerobeta=True).
            float beta_norm_sq = 0.0f;
            uint32_t beta_zero_count = 0;
            for (uint32_t bi = 0; bi < (uint32_t)beta.size(); ++bi) {
                beta_norm_sq += beta[bi] * beta[bi];
                if (std::fabs(beta[bi]) < 1e-6f) {
                    beta_zero_count++;
                }
            }
            const float beta_norm = std::sqrt(beta_norm_sq);
            const bool zerobeta = (beta_norm > 50.0f);

            if (zerobeta) {
                std::fill(beta.begin(), beta.end(), 0.0f);
                if (li == 0 && head == 0) {
                    LLAMA_LOG_WARN("self-study: extreme beta_norm (%.1f) — using zerobeta + direct V\n",
                                   beta_norm);
                }
            }

            if (stats) {
                beta_norm_sum += beta_norm;
                beta_sparsity_sum += (float)beta_zero_count / std::max<uint32_t>(1, (uint32_t)beta.size());
                fit_residual_sum += head_residual;
                n_beta_heads_seen++;
            }

            // Least-squares V fitting with direct fallback (MIT c2_method='direct')
            if (layout.n_embd_head_v > 0) {
                llama_kv_compact_matrix compacted_v;
                bool v_fit_ok = false;

                if (!zerobeta) {
                    v_fit_ok = llama_kv_compact_fit_values(
                                entry.queries, entry.k, full_v,
                                compacted_k, beta, solver_opts,
                                compacted_v);
                    // Check fitted V for NaN/Inf
                    if (v_fit_ok) {
                        for (float val : compacted_v.data) {
                            if (!std::isfinite(val)) {
                                v_fit_ok = false;
                                break;
                            }
                        }
                    }
                }

                if (!v_fit_ok) {
                    // Direct V selection (MIT c2_method='direct')
                    if (!gather_matrix_rows(full_v, selected_local, compacted_v)) {
                        solver_ok = false;
                        break;
                    }
                }
                write_compacted_payload(dst_layer.v_data, layout.type_v,
                                        n_selected, head,
                                        layout.n_embd_head_v, compacted_v);
            }

            // Write compacted K payload
            write_compacted_payload(dst_layer.k_data, layout.type_k,
                                    n_selected, head,
                                    layout.n_embd_head_k, compacted_k);

            // Write beta
            for (uint32_t token = 0; token < n_selected; ++token) {
                dst_layer.beta_data[size_t(head) * n_selected + token] = beta[token];
            }
        }
    }
    const auto t_solver_end = std::chrono::steady_clock::now();

    if (!solver_ok) {
        kv.compacted_prefix_clear(seq_id, true);
        return false;
    }

    // --- Populate stats ---
    if (stats) {
        stats->generation_time_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
        stats->q_capture_time_ms  = std::chrono::duration<double, std::milli>(t_query_end - t_query_start).count();
        stats->solver_time_ms     = std::chrono::duration<double, std::milli>(t_solver_end - t_solver_start).count();
        stats->n_tokens_generated = q_state.layers.empty() ? 0 : q_state.layers[0].n_tokens;
        stats->n_queries_per_head = actual_queries_per_head;
        stats->n_prefix_tokens    = n_prefix_tokens;
        stats->n_selected_tokens  = n_selected;

        // Diagnostic averages (Sprint 2).
        stats->q_norm_mean       = (n_heads_seen > 0) ? (float)(q_norm_sum / n_heads_seen) : 0.0f;
        stats->k_norm_mean       = (n_heads_seen > 0) ? (float)(k_norm_sum / n_heads_seen) : 0.0f;
        stats->beta_norm_mean    = (n_beta_heads_seen > 0) ? (float)(beta_norm_sum / n_beta_heads_seen) : 0.0f;
        stats->beta_sparsity     = (n_beta_heads_seen > 0) ? (float)(beta_sparsity_sum / n_beta_heads_seen) : 0.0f;
        stats->fit_residual_mean = (n_beta_heads_seen > 0) ? (float)(fit_residual_sum / n_beta_heads_seen) : 0.0f;

        LLAMA_LOG_INFO("self_study diagnostics: layers_with_q=%u dim_mismatches=%u "
                       "q_norm=%.4f k_norm=%.4f beta_norm=%.4f beta_sparsity=%.4f "
                       "fit_residual=%.6f\n",
                       stats->n_layers_with_q, stats->n_dim_mismatches,
                       stats->q_norm_mean, stats->k_norm_mean,
                       stats->beta_norm_mean, stats->beta_sparsity,
                       stats->fit_residual_mean);
    }

    LLAMA_LOG_INFO("self-study: pipeline complete — %u prefix → %u selected (seq %d)\n",
                   n_prefix_tokens, n_selected, seq_id);
    return true;
}

// ---------------------------------------------------------------------------
// Chunked self-study pipeline (Phase 6)
// ---------------------------------------------------------------------------

bool llama_kv_compact_chunked_self_study_from_live_kv(
        struct llama_context * ctx,
        llama_kv_cache       & kv,
        llama_seq_id           seq_id,
        uint32_t               target_tokens,
        llama_pos              live_suffix_pos0,
        const llama_kv_compact_self_study_config & config,
        llama_kv_compact_self_study_stats * stats,
        llama_pos p0,
        uint32_t chunk_size) {

    if (seq_id < 0 || target_tokens == 0 || live_suffix_pos0 <= p0) {
        return false;
    }

    // --- Validate cache state ---
    std::vector<llama_pos> prefix_positions;
    if (!kv.compacted_prefix_seq_positions(seq_id, p0, live_suffix_pos0, prefix_positions)) {
        return false;
    }
    if (prefix_positions.empty()) {
        return false;
    }

    const auto & layouts = kv.get_compacted_prefix()->get_layouts();
    if (layouts.empty()) {
        return false;
    }

    const uint32_t n_prefix_tokens = (uint32_t)prefix_positions.size();

    // Delegate to non-chunked self-study if prefix fits in one chunk.
    if (n_prefix_tokens <= chunk_size) {
        return llama_kv_compact_self_study_from_live_kv(
                ctx, kv, seq_id, target_tokens, live_suffix_pos0,
                config, stats, p0);
    }

    // ===== Phase 1: Multi-round Q-capture generation =====
    const auto & model_hparams = ctx->get_model().hparams;
    const uint32_t n_layer     = model_hparams.n_layer;
    const uint32_t n_embd_head = model_hparams.n_embd_head_k(0);
    const uint32_t n_head_q    = model_hparams.n_head(0);

    uint32_t eff_n_rounds   = std::max(1u, std::min(config.n_rounds, (uint32_t)LLAMA_KV_COMPACT_MAX_ROUNDS));
    uint32_t eff_n_generate = config.n_generate;

    // Memory budget guard (M-01 fix).
    if (config.max_q_capture_mb > 0) {
        const size_t bytes_per_token = (size_t)n_layer * n_head_q * n_embd_head * sizeof(float);
        const size_t budget_bytes    = (size_t)config.max_q_capture_mb * 1024 * 1024;
        const size_t requested_bytes = bytes_per_token * eff_n_generate * eff_n_rounds;

        if (requested_bytes > budget_bytes && bytes_per_token > 0) {
            const uint32_t max_total_tokens = (uint32_t)(budget_bytes / bytes_per_token);
            eff_n_generate = std::max(1u, max_total_tokens / eff_n_rounds);
            if (eff_n_generate < 100 && eff_n_rounds > 1) {
                eff_n_rounds   = std::max(1u, max_total_tokens / config.n_generate);
                eff_n_generate = std::max(1u, max_total_tokens / eff_n_rounds);
            }
            LLAMA_LOG_WARN("chunked self-study: Q-capture memory guard — reduced from %u×%u to %u×%u tokens "
                           "(%u MB budget, %.2f MB/token)\n",
                           config.n_generate, config.n_rounds, eff_n_generate, eff_n_rounds,
                           config.max_q_capture_mb,
                           (float)bytes_per_token / (1024.0f * 1024.0f));
        }
    }

    const uint32_t total_generate = eff_n_generate * eff_n_rounds;

    // Q-capture sufficiency check: if estimated queries < target_tokens, the
    // NNLS solver becomes underdetermined and produces NaN/extreme weights.
    // Fall back to the proven select pipeline which uses K-as-surrogate-Q.
    {
        const uint32_t n_head_kv = layouts[0].n_head_kv;
        const uint32_t n_rep = n_head_q / std::max(1u, n_head_kv);
        const uint32_t estimated_queries = total_generate * n_rep;
        if (estimated_queries < target_tokens) {
            LLAMA_LOG_WARN("chunked self-study: Q-capture insufficient (%u queries < %u targets) — "
                           "falling back to select pipeline for stable quality\n",
                           estimated_queries, target_tokens);
            return llama_kv_compact_select_from_live_kv(kv, seq_id, target_tokens, live_suffix_pos0,
                                                         nullptr, p0);
        }
    }

    llama_q_capture_state q_state;
    q_state.reset((int32_t)n_layer, n_embd_head, n_head_q, total_generate);

    const auto t_gen_start = std::chrono::steady_clock::now();
    bool any_generated = false;
    for (uint32_t round = 0; round < eff_n_rounds; ++round) {
        const float temp = config.temperatures[round];
        if (llama_kv_compact_self_study_generate(ctx, q_state, eff_n_generate, seq_id, temp, 42 + round)) {
            any_generated = true;
        } else if (round == 0) {
            return false;
        }
    }
    if (!any_generated) {
        return false;
    }
    const auto t_gen_end = std::chrono::steady_clock::now();

    // Q-capture layer diagnostics.
    if (stats) {
        stats->n_layers_with_q  = 0;
        stats->n_dim_mismatches = 0;
        for (int il = 0; il < q_state.n_layers; ++il) {
            if (q_state.layers[il].n_tokens > 0) {
                stats->n_layers_with_q++;
            }
            stats->n_dim_mismatches += q_state.layers[il].n_dim_mismatches;
        }
    }

    // Post-capture sufficiency verification: use ACTUAL captured token count
    // rather than pre-capture estimate (which may be wrong for some models).
    {
        uint32_t actual_tokens = 0;
        for (int il = 0; il < q_state.n_layers; ++il) {
            if (q_state.layers[il].n_tokens > 0) {
                actual_tokens = q_state.layers[il].n_tokens;
                break;
            }
        }
        const uint32_t n_head_kv_l0 = layouts[0].n_head_kv;
        const uint32_t n_rep = n_head_q / std::max(1u, n_head_kv_l0);
        const uint32_t actual_queries = actual_tokens * n_rep;
        if (actual_queries < target_tokens) {
            LLAMA_LOG_WARN("chunked self-study: post-capture insufficiency (%u tokens captured, "
                           "%u queries < %u targets) — falling back to select pipeline\n",
                           actual_tokens, actual_queries, target_tokens);
            return llama_kv_compact_select_from_live_kv(kv, seq_id, target_tokens, live_suffix_pos0,
                                                         nullptr, p0);
        }
    }

    // ===== Phase 2: Precompute + normalize Q per layer/head =====
    // GQA regroup, subsample, then normalize Q to match K scale using a
    // small K sample from the prefix (avoids extracting full K just for norms).

    struct head_q_entry {
        llama_kv_compact_matrix queries;  // [n_queries, n_embd_head]
    };
    std::vector<std::vector<head_q_entry>> q_cache(layouts.size());

    const uint32_t norm_sample_n = std::min(256u, n_prefix_tokens);
    std::vector<llama_pos> norm_sample_pos(
        prefix_positions.begin(),
        prefix_positions.begin() + norm_sample_n);

    double q_norm_sum = 0.0, k_norm_sum = 0.0;
    uint32_t n_heads_seen = 0;

    for (size_t li = 0; li < layouts.size(); ++li) {
        const auto & layout = layouts[li];
        q_cache[li].resize(layout.n_head_kv);

        for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
            auto & entry = q_cache[li][head];

            // GQA regroup
            if (!llama_q_capture_regroup_for_kv_head(
                        q_state, (int32_t)layout.layer_id,
                        head, layout.n_head_kv, entry.queries)) {
                return false;
            }

            // Subsample to max_queries_per_kv_head
            llama_q_capture_subsample(entry.queries, config.max_queries_per_kv_head);

            // Extract a small K sample for Q/K norm matching.
            std::vector<float> sample_k_data;
            if (!kv.compacted_prefix_copy_k_head_f32(
                        (int32_t)layout.layer_id, seq_id, head,
                        norm_sample_pos, sample_k_data)) {
                return false;
            }
            llama_kv_compact_matrix sample_k(norm_sample_n, layout.n_embd_head_k);
            sample_k.data = std::move(sample_k_data);

            float q_norm = compute_row_norm_mean(entry.queries);
            float k_norm = compute_row_norm_mean(sample_k);

            if (stats) {
                q_norm_sum += q_norm;
                k_norm_sum += k_norm;
                n_heads_seen++;
            }

            // Normalize Q to match K scale (same fix as non-chunked self-study).
            if (q_norm > 1e-8f && k_norm > 1e-8f) {
                float scale = k_norm / q_norm;
                for (size_t i = 0; i < entry.queries.data.size(); ++i) {
                    entry.queries.data[i] *= scale;
                }
            }
        }
    }

    // ===== Phase 3: Chunked scoring + selection =====
    uint32_t n_chunks = (n_prefix_tokens + chunk_size - 1) / chunk_size;

    // Merge chunks if more chunks than target tokens.
    if (n_chunks > target_tokens) {
        chunk_size = (n_prefix_tokens + target_tokens - 1) / target_tokens;
        n_chunks   = (n_prefix_tokens + chunk_size - 1) / chunk_size;
    }

    // Compute chunk boundaries and proportional budgets.
    std::vector<uint32_t> chunk_starts(n_chunks);
    std::vector<uint32_t> chunk_sizes_v(n_chunks);
    std::vector<uint32_t> chunk_budgets(n_chunks);

    uint32_t budget_allocated = 0;
    for (uint32_t c = 0; c < n_chunks; ++c) {
        chunk_starts[c]  = c * chunk_size;
        chunk_sizes_v[c] = std::min(chunk_size, n_prefix_tokens - chunk_starts[c]);
        chunk_budgets[c] = (uint32_t)std::round(
            (float)target_tokens * (float)chunk_sizes_v[c] / (float)n_prefix_tokens);
        chunk_budgets[c] = std::max(chunk_budgets[c], 1u);
        budget_allocated += chunk_budgets[c];
    }

    // Adjust budgets to hit exact target.
    if (budget_allocated > target_tokens) {
        std::vector<uint32_t> order(n_chunks);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return chunk_budgets[a] < chunk_budgets[b];
        });
        uint32_t excess = budget_allocated - target_tokens;
        for (uint32_t idx : order) {
            if (excess == 0) break;
            uint32_t reduce = std::min(chunk_budgets[idx], excess);
            chunk_budgets[idx] -= reduce;
            excess -= reduce;
        }
    } else if (budget_allocated < target_tokens) {
        chunk_budgets[n_chunks - 1] += target_tokens - budget_allocated;
    }

    // Per-chunk scoring with Metal GPU.
    auto * metal_ctx = llama_kv_compact_metal_create();

    const auto t_query_start = std::chrono::steady_clock::now();

    std::vector<uint32_t> all_selected_local;
    all_selected_local.reserve(target_tokens);

    for (uint32_t c = 0; c < n_chunks; ++c) {
        const uint32_t cs = chunk_starts[c];
        const uint32_t cn = chunk_sizes_v[c];
        const uint32_t cb = std::min(chunk_budgets[c], cn);

        if (cb == 0) continue;

        std::vector<llama_pos> chunk_positions(
            prefix_positions.begin() + cs,
            prefix_positions.begin() + cs + cn);

        std::vector<float> chunk_scores(cn, 0.0f);

        for (size_t li = 0; li < layouts.size(); ++li) {
            const auto & layout = layouts[li];

            for (uint32_t head = 0; head < layout.n_head_kv; ++head) {
                const auto & q_entry = q_cache[li][head];

                // Extract chunk K from live cache.
                std::vector<float> k_data;
                if (!kv.compacted_prefix_copy_k_head_f32(
                            (int32_t)layout.layer_id, seq_id, head,
                            chunk_positions, k_data)) {
                    llama_kv_compact_metal_free(metal_ctx);
                    return false;
                }

                llama_kv_compact_matrix chunk_k(cn, layout.n_embd_head_k);
                chunk_k.data = std::move(k_data);

                // Score normalized Q against chunk K.
                if (metal_ctx) {
                    std::vector<float> head_scores(cn, 0.0f);
                    if (llama_kv_compact_metal_attention_scores(
                                metal_ctx,
                                q_entry.queries.data.data(), q_entry.queries.rows, q_entry.queries.cols,
                                chunk_k.data.data(), chunk_k.rows,
                                head_scores.data())) {
                        for (uint32_t ki = 0; ki < cn; ++ki) {
                            chunk_scores[ki] += head_scores[ki];
                        }
                    } else {
                        llama_kv_compact_accumulate_attention_scores(
                                q_entry.queries, chunk_k, chunk_scores);
                    }
                } else {
                    llama_kv_compact_accumulate_attention_scores(
                            q_entry.queries, chunk_k, chunk_scores);
                }
            }
        }

        // Top-k selection within this chunk.
        const std::vector<uint32_t> chunk_selected = llama_kv_compact_select_topk(chunk_scores, cb);

        // Map chunk-local indices to global prefix indices.
        for (uint32_t idx : chunk_selected) {
            all_selected_local.push_back(cs + idx);
        }
    }

    llama_kv_compact_metal_free(metal_ctx);
    metal_ctx = nullptr;

    const auto t_query_end = std::chrono::steady_clock::now();

    // Sort and cap global selection.
    std::sort(all_selected_local.begin(), all_selected_local.end());
    if (all_selected_local.size() > target_tokens) {
        all_selected_local.resize(target_tokens);
    }
    const uint32_t actual_selected = (uint32_t)all_selected_local.size();

    // Convert to positions.
    std::vector<llama_pos> selected_positions;
    selected_positions.reserve(actual_selected);
    for (uint32_t idx : all_selected_local) {
        selected_positions.push_back(prefix_positions[idx]);
    }

    // ===== Phase 4: Configure store + full-prefix solver =====
    const llama_pos seq_max = kv.seq_pos_max(seq_id);
    const uint32_t logical_token_count = seq_max >= 0 ? (uint32_t)(seq_max + 1) : (uint32_t)live_suffix_pos0;
    if (!kv.compacted_prefix_configure(seq_id, logical_token_count, selected_positions, live_suffix_pos0)) {
        return false;
    }

    auto * seq = kv.get_compacted_prefix()->get_seq(seq_id);
    if (seq == nullptr || !seq->enabled || seq->layers.size() != layouts.size()) {
        kv.compacted_prefix_clear(seq_id, true);
        return false;
    }

    const auto t_solver_start = std::chrono::steady_clock::now();
    const llama_kv_compact_solver_opts solver_opts = {
        /* lambda           */ config.lambda,
        /* nnls_iters       */ config.nnls_iters,
        /* nnls_lower_bound */ 1e-12f,
        /* nnls_upper_bound */ 100.0f,   // prevent extreme betas from underdetermined NNLS
    };

    bool solver_ok = true;
    uint32_t actual_queries_per_head = 0;

    double beta_norm_sum = 0.0, beta_sparsity_sum = 0.0;
    double fit_residual_sum = 0.0;
    uint32_t n_beta_heads_seen = 0;

    for (size_t li = 0; li < layouts.size() && solver_ok; ++li) {
        const auto & layout = layouts[li];
        auto & dst_layer = seq->layers[li];

        for (uint32_t head = 0; head < layout.n_head_kv && solver_ok; ++head) {
            auto & q_entry = q_cache[li][head];

            if (li == 0 && head == 0) {
                actual_queries_per_head = q_entry.queries.rows;
            }

            // Extract full-prefix K for solver.
            std::vector<float> full_k_data;
            if (!kv.compacted_prefix_copy_k_head_f32(
                        (int32_t)layout.layer_id, seq_id, head,
                        prefix_positions, full_k_data)) {
                solver_ok = false;
                break;
            }
            llama_kv_compact_matrix full_k(n_prefix_tokens, layout.n_embd_head_k);
            full_k.data = std::move(full_k_data);

            // V extraction
            std::vector<float> full_v_data;
            if (!kv.compacted_prefix_copy_v_head_f32(
                        (int32_t)layout.layer_id, seq_id, head,
                        prefix_positions, full_v_data)) {
                solver_ok = false;
                break;
            }
            llama_kv_compact_matrix full_v(n_prefix_tokens, layout.n_embd_head_v);
            full_v.data = std::move(full_v_data);

            // Gather selected K rows.
            llama_kv_compact_matrix compacted_k;
            if (!gather_matrix_rows(full_k, all_selected_local, compacted_k)) {
                solver_ok = false;
                break;
            }

            // NNLS beta fitting (Q already normalized from Phase 2).
            std::vector<float> beta;
            float head_residual = 0.0f;
            if (!llama_kv_compact_fit_beta(q_entry.queries, full_k,
                                            compacted_k, solver_opts,
                                            beta, stats ? &head_residual : nullptr)) {
                solver_ok = false;
                break;
            }

            // Beta quality check (MIT zerobeta fallback).
            float beta_norm_sq = 0.0f;
            uint32_t beta_zero_count = 0;
            for (uint32_t bi = 0; bi < (uint32_t)beta.size(); ++bi) {
                beta_norm_sq += beta[bi] * beta[bi];
                if (std::fabs(beta[bi]) < 1e-6f) {
                    beta_zero_count++;
                }
            }
            const float beta_norm = std::sqrt(beta_norm_sq);
            const bool zerobeta = (beta_norm > 50.0f);

            if (zerobeta) {
                std::fill(beta.begin(), beta.end(), 0.0f);
                if (li == 0 && head == 0) {
                    LLAMA_LOG_WARN("chunked self-study: extreme beta_norm (%.1f) — using zerobeta + direct V\n",
                                   beta_norm);
                }
            }

            if (stats) {
                beta_norm_sum     += beta_norm;
                beta_sparsity_sum += (float)beta_zero_count / std::max<uint32_t>(1, (uint32_t)beta.size());
                fit_residual_sum  += head_residual;
                n_beta_heads_seen++;
            }

            // Least-squares V fitting with direct fallback (MIT c2_method='direct')
            if (layout.n_embd_head_v > 0) {
                llama_kv_compact_matrix compacted_v;
                bool v_fit_ok = false;

                if (!zerobeta) {
                    v_fit_ok = llama_kv_compact_fit_values(
                                q_entry.queries, full_k, full_v,
                                compacted_k, beta, solver_opts,
                                compacted_v);
                    if (v_fit_ok) {
                        for (float val : compacted_v.data) {
                            if (!std::isfinite(val)) {
                                v_fit_ok = false;
                                break;
                            }
                        }
                    }
                }

                if (!v_fit_ok) {
                    if (!gather_matrix_rows(full_v, all_selected_local, compacted_v)) {
                        solver_ok = false;
                        break;
                    }
                }
                write_compacted_payload(dst_layer.v_data, layout.type_v,
                                        actual_selected, head,
                                        layout.n_embd_head_v, compacted_v);
            }

            // Write compacted K payload.
            write_compacted_payload(dst_layer.k_data, layout.type_k,
                                    actual_selected, head,
                                    layout.n_embd_head_k, compacted_k);

            // Write beta.
            for (uint32_t token = 0; token < actual_selected; ++token) {
                dst_layer.beta_data[size_t(head) * actual_selected + token] = beta[token];
            }
        }
    }
    const auto t_solver_end = std::chrono::steady_clock::now();

    if (!solver_ok) {
        kv.compacted_prefix_clear(seq_id, true);
        return false;
    }

    // --- Populate stats ---
    if (stats) {
        stats->generation_time_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
        stats->q_capture_time_ms  = std::chrono::duration<double, std::milli>(t_query_end - t_query_start).count();
        stats->solver_time_ms     = std::chrono::duration<double, std::milli>(t_solver_end - t_solver_start).count();
        stats->n_tokens_generated = q_state.layers.empty() ? 0 : q_state.layers[0].n_tokens;
        stats->n_queries_per_head = actual_queries_per_head;
        stats->n_prefix_tokens    = n_prefix_tokens;
        stats->n_selected_tokens  = actual_selected;

        stats->q_norm_mean       = (n_heads_seen > 0) ? (float)(q_norm_sum / n_heads_seen) : 0.0f;
        stats->k_norm_mean       = (n_heads_seen > 0) ? (float)(k_norm_sum / n_heads_seen) : 0.0f;
        stats->beta_norm_mean    = (n_beta_heads_seen > 0) ? (float)(beta_norm_sum / n_beta_heads_seen) : 0.0f;
        stats->beta_sparsity     = (n_beta_heads_seen > 0) ? (float)(beta_sparsity_sum / n_beta_heads_seen) : 0.0f;
        stats->fit_residual_mean = (n_beta_heads_seen > 0) ? (float)(fit_residual_sum / n_beta_heads_seen) : 0.0f;

        LLAMA_LOG_INFO("chunked_self_study diagnostics: n_chunks=%u layers_with_q=%u "
                       "q_norm=%.4f k_norm=%.4f beta_norm=%.4f beta_sparsity=%.4f "
                       "fit_residual=%.6f\n",
                       n_chunks, stats->n_layers_with_q,
                       stats->q_norm_mean, stats->k_norm_mean,
                       stats->beta_norm_mean, stats->beta_sparsity,
                       stats->fit_residual_mean);
    }

    LLAMA_LOG_INFO("chunked self-study: pipeline complete — %u prefix (%u chunks of %u) → %u selected (seq %d)\n",
                   n_prefix_tokens, n_chunks, chunk_size, actual_selected, seq_id);
    return true;
}
