#pragma once

// Metal GPU acceleration for KV compaction solver hot paths.
//
// Moves attention score computation and XᵀX assembly to GPU, reducing
// compaction latency from ~2-5s to <700ms for 14B models at 4K context.
//
// When Metal is unavailable (non-Apple or headless), all functions return
// false and the caller falls back to CPU.

#include <cstdint>

struct llama_kv_compact_metal_ctx;

#if defined(__APPLE__)

// Create Metal compute context. Returns nullptr if Metal unavailable.
llama_kv_compact_metal_ctx * llama_kv_compact_metal_create();

// Free Metal context and all cached buffers.
void llama_kv_compact_metal_free(llama_kv_compact_metal_ctx * ctx);

// Compute attention scores on GPU.
//
// Equivalent to the CPU path:
//   for each query qi:
//     scores = Q[qi] · Kᵀ / √d
//     weights = softmax(scores)
//     scores_out[ki] += weights[ki]
//
// Output is written fresh (not accumulated). Caller adds to aggregate.
//
// Returns true on success, false on failure (fall back to CPU).
bool llama_kv_compact_metal_attention_scores(
        llama_kv_compact_metal_ctx * ctx,
        const float * Q_data, uint32_t n_queries, uint32_t d,
        const float * K_data, uint32_t n_keys,
        float * scores_out);

// Compute XᵀX on GPU (symmetric normal equations matrix).
//
// output[i*t+j] = sum_r X[r*t+i] * X[r*t+j]
// Computes full symmetric matrix.
//
// Returns true on success, false on failure (fall back to CPU).
bool llama_kv_compact_metal_xtx(
        llama_kv_compact_metal_ctx * ctx,
        const float * X_data, uint32_t n, uint32_t t,
        float * xtx_out);

#else

// Non-Apple stubs — always return false to trigger CPU fallback.
inline llama_kv_compact_metal_ctx * llama_kv_compact_metal_create() { return nullptr; }
inline void llama_kv_compact_metal_free(llama_kv_compact_metal_ctx *) {}
inline bool llama_kv_compact_metal_attention_scores(
        llama_kv_compact_metal_ctx *, const float *, uint32_t, uint32_t,
        const float *, uint32_t, float *) { return false; }
inline bool llama_kv_compact_metal_xtx(
        llama_kv_compact_metal_ctx *, const float *, uint32_t, uint32_t,
        float *) { return false; }

#endif
