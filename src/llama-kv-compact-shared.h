#pragma once

// Shared helpers used by the compaction pipeline, self-study, and prefill-Q
// modules.  Each function is static-inline so that every TU gets its own copy
// without ODR concerns -- the bodies are tiny.

#include "llama-kv-compact-solver.h"   // llama_kv_compact_matrix, ggml.h

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---- Ablation env-var helpers ------------------------------------------------

// Skip beta fitting when LLAMA_COMPACT_NO_BETA=1 (zero-beta, selection only).
static inline bool llama_kv_compact_skip_beta_fit() {
    static const bool skip = [] {
        const char * env = std::getenv("LLAMA_COMPACT_NO_BETA");
        return env && env[0] == '1';
    }();
    return skip;
}

// Skip C_v fitting when LLAMA_COMPACT_NO_CV=1 (keeps original V at selected
// positions).
static inline bool llama_kv_compact_skip_cv_fit() {
    static const bool skip = [] {
        const char * env = std::getenv("LLAMA_COMPACT_NO_CV");
        return env && env[0] == '1';
    }();
    return skip;
}

// ---- Matrix row gathering ---------------------------------------------------

// Gather selected rows from a source matrix into a destination matrix.
static inline bool gather_matrix_rows(
        const llama_kv_compact_matrix & src,
        const std::vector<uint32_t>   & row_indices,
        llama_kv_compact_matrix       & dst) {
    dst.resize((uint32_t) row_indices.size(), src.cols);
    for (size_t i = 0; i < row_indices.size(); ++i) {
        const uint32_t src_row = row_indices[i];
        if (src_row >= src.rows) {
            return false;
        }
        std::memcpy(dst.row((uint32_t) i), src.row(src_row),
                     (size_t) src.cols * sizeof(float));
    }
    return true;
}

// ---- Quantised payload writer -----------------------------------------------

// Write solver output (float rows) into the compacted prefix store's quantised
// KV payload.  The n_head_kv parameter is accepted for call-site uniformity but
// is not used in the computation.
static inline void write_compacted_payload(
        std::vector<uint8_t>          & dst,
        ggml_type                       type,
        uint32_t                        n_head_kv,
        uint32_t                        n_tokens,
        uint32_t                        head,
        uint32_t                        dim,
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

    (void) n_head_kv;
}
