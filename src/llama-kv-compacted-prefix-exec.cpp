#include "llama-kv-compacted-prefix-exec.h"

#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

void require_tensor_type(const ggml_tensor * dst, ggml_type type, const char * what) {
    if (dst->type != type) {
        throw std::runtime_error(std::string("compacted-prefix ") + what + " tensor type mismatch");
    }
}

void require_host_or_direct_data(const ggml_tensor * dst, const char * what) {
    if (dst->data == nullptr) {
        throw std::runtime_error(std::string("compacted-prefix ") + what + " tensor has no writable data");
    }
    if (dst->buffer != nullptr && !ggml_backend_buffer_is_host(dst->buffer)) {
        throw std::runtime_error(std::string("compacted-prefix ") + what + " tensor must be host-backed");
    }
}

void require_dims(const ggml_tensor * dst, int64_t d0, int64_t d1, int64_t d2, int64_t d3, const char * what) {
    if (dst->ne[0] != d0 || dst->ne[1] != d1 || dst->ne[2] != d2 || dst->ne[3] != d3) {
        throw std::runtime_error(std::string("compacted-prefix ") + what + " tensor shape mismatch");
    }
}

} // namespace

bool llama_compacted_prefix_can_execute(
        llama_seq_id seq_id,
        const llama_compacted_prefix_store::sequence_state * state,
        const llama_ubatch & ubatch,
        llama_compacted_prefix_exec_candidate * out) {
    if (out) {
        *out = {};
    }

    if (seq_id < 0 || state == nullptr) {
        return false;
    }

    if (!state->enabled || !state->is_execution_enabled() || state->logical_positions.empty()) {
        return false;
    }

    if (ubatch.n_tokens == 0 || ubatch.n_seqs_unq != 1) {
        return false;
    }

    // V4-J: allow is_pos_2d() batches for IMROPE models (Qwen3.5, etc.).
    // K/V data already has IMROPE rotations applied; causal masking uses
    // scalar positions only, which is safe for text-only IMROPE.
    if (ubatch.is_pos_2d() && !state->is_imrope) {
        return false;
    }

    if (ubatch.seq_id_unq == nullptr || ubatch.seq_id_unq[0] != seq_id) {
        return false;
    }

    if (ubatch.pos == nullptr) {
        return false;
    }

    if (state->live_suffix_pos0 >= 0) {
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (ubatch.pos[i] < state->live_suffix_pos0) {
                return false;
            }
        }
    }

    if (out) {
        out->seq_id = seq_id;
        out->n_tokens = state->logical_positions.size();
        out->zero_beta = state->is_zero_beta();
    }

    return true;
}

void llama_compacted_prefix_set_input_mask(
        ggml_tensor * dst,
        const llama_compacted_prefix_store::sequence_state & state,
        const llama_ubatch & ubatch,
        const llama_hparams & hparams,
        bool causal_attn) {
    const int64_t t0 = ggml_time_us();
    const bool is_f16 = (dst->type == GGML_TYPE_F16);
    if (!is_f16) {
        require_tensor_type(dst, GGML_TYPE_F32, "mask");
    }
    require_host_or_direct_data(dst, "mask");

    const int64_t n_prefix = (int64_t) state.logical_positions.size();
    const int64_t n_stream = dst->ne[3];
    if (n_stream <= 0 || (ubatch.n_tokens % n_stream) != 0) {
        throw std::runtime_error("compacted-prefix mask: n_tokens must be divisible by n_stream");
    }
    const int64_t n_tps = ubatch.n_tokens / n_stream;

    require_dims(dst, n_prefix, n_tps, 1, n_stream, "mask");

    auto * base = reinterpret_cast<uint8_t *>(dst->data);

    for (int64_t s = 0; s < n_stream; ++s) {
        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t i = s*n_tps + ii;
            const llama_pos p1 = ubatch.pos[i];

            for (int64_t j = 0; j < n_prefix; ++j) {
                const llama_pos p0 = state.logical_positions[j];
                float value = 0.0f;

                if (causal_attn && p0 > p1) {
                    value = -INFINITY;
                } else if (hparams.use_alibi) {
                    value = -std::abs(float(p0 - p1));
                }

                if (is_f16) {
                    auto * dst_ptr = reinterpret_cast<ggml_fp16_t *>(base + size_t(s) * dst->nb[3] + size_t(ii) * dst->nb[1] + size_t(j) * dst->nb[0]);
                    *dst_ptr = ggml_fp32_to_fp16(value);
                } else {
                    auto * dst_ptr = reinterpret_cast<float *>(base + size_t(s) * dst->nb[3] + size_t(ii) * dst->nb[1] + size_t(j) * dst->nb[0]);
                    *dst_ptr = value;
                }
            }
        }
    }

    const int64_t t1 = ggml_time_us();
    LLAMA_LOG_DEBUG("compact_exec: mask materialization %.1fms (prefix=%zu)\n",
                    (t1 - t0) / 1000.0, state.logical_positions.size());
}

void llama_compacted_prefix_set_input_k(
        ggml_tensor * dst,
        const llama_compacted_prefix_store::layer_storage & layer) {
    const int64_t t0 = ggml_time_us();

    require_tensor_type(dst, layer.layout.type_k, "K");
    // Note: require_host_or_direct_data removed — caller guarantees dst->data
    // points to a host staging buffer (B5: GPU-resident tensor support).

    const int64_t n_prefix = layer.n_compacted_tokens;
    require_dims(dst, layer.layout.n_embd_head_k, layer.layout.n_head_kv, n_prefix, 1, "K");

    const size_t token_bytes = ggml_row_size(layer.layout.type_k, layer.layout.n_embd_head_k);
    auto * base = reinterpret_cast<uint8_t *>(dst->data);

    for (uint32_t head = 0; head < layer.layout.n_head_kv; ++head) {
        for (uint32_t token = 0; token < layer.n_compacted_tokens; ++token) {
            const size_t src_offset = (size_t(head) * layer.n_compacted_tokens + token) * token_bytes;
            const size_t dst_offset = size_t(head) * dst->nb[1] + size_t(token) * dst->nb[2];
            std::memcpy(base + dst_offset, layer.k_data.data() + src_offset, token_bytes);
        }
    }

    const int64_t t1 = ggml_time_us();
    LLAMA_LOG_DEBUG("compact_exec: K materialization %.1fms (prefix=%u)\n",
                    (t1 - t0) / 1000.0, (unsigned)layer.n_compacted_tokens);
}

void llama_compacted_prefix_set_input_v(
        ggml_tensor * dst,
        const llama_compacted_prefix_store::layer_storage & layer) {
    const int64_t t0 = ggml_time_us();

    require_tensor_type(dst, layer.layout.type_v, "V");
    // Note: require_host_or_direct_data removed — caller guarantees dst->data
    // points to a host staging buffer (B5: GPU-resident tensor support).

    const int64_t n_prefix = layer.n_compacted_tokens;
    require_dims(dst, layer.layout.n_embd_head_v, layer.layout.n_head_kv, n_prefix, 1, "V");

    if (layer.layout.n_embd_head_v == 0 || layer.v_data.empty()) {
        return;
    }

    const size_t token_bytes = ggml_row_size(layer.layout.type_v, layer.layout.n_embd_head_v);
    auto * base = reinterpret_cast<uint8_t *>(dst->data);

    for (uint32_t head = 0; head < layer.layout.n_head_kv; ++head) {
        for (uint32_t token = 0; token < layer.n_compacted_tokens; ++token) {
            const size_t src_offset = (size_t(head) * layer.n_compacted_tokens + token) * token_bytes;
            const size_t dst_offset = size_t(head) * dst->nb[1] + size_t(token) * dst->nb[2];
            std::memcpy(base + dst_offset, layer.v_data.data() + src_offset, token_bytes);
        }
    }

    const int64_t t1 = ggml_time_us();
    LLAMA_LOG_DEBUG("compact_exec: V materialization %.1fms (prefix=%u)\n",
                    (t1 - t0) / 1000.0, (unsigned)layer.n_compacted_tokens);
}

void llama_compacted_prefix_set_input_beta(
        ggml_tensor * dst,
        const llama_compacted_prefix_store::layer_storage & layer,
        uint32_t n_head) {
    const int64_t t0 = ggml_time_us();
    require_tensor_type(dst, GGML_TYPE_F32, "beta");
    // Note: require_host_or_direct_data removed — caller guarantees dst->data
    // points to a host staging buffer (B5: GPU-resident tensor support).

    const int64_t n_prefix = layer.n_compacted_tokens;
    const int64_t n_stream = dst->ne[3];
    const int64_t n_tps = dst->ne[1];

    require_dims(dst, n_prefix, n_tps, n_head, n_stream, "beta");

    if (layer.layout.n_head_kv == 0 || n_head < layer.layout.n_head_kv || n_head % layer.layout.n_head_kv != 0) {
        throw std::runtime_error("compacted-prefix beta expansion requires n_head >= n_head_kv and divisible by n_head_kv");
    }

    const size_t expected_beta_size = size_t(layer.layout.n_head_kv) * layer.n_compacted_tokens;
    if (layer.beta_data.size() < expected_beta_size) {
        throw std::runtime_error("compacted-prefix beta_data is smaller than expected for the configured layout");
    }

    const uint32_t n_rep = n_head / layer.layout.n_head_kv;
    auto * base = reinterpret_cast<uint8_t *>(dst->data);

    for (int64_t s = 0; s < n_stream; ++s) {
        for (uint32_t head = 0; head < n_head; ++head) {
            const uint32_t kv_head = head / n_rep;
            for (int64_t t = 0; t < n_tps; ++t) {
                const size_t src_row = size_t(kv_head) * layer.n_compacted_tokens;
                for (uint32_t j = 0; j < layer.n_compacted_tokens; ++j) {
                    auto * dst_ptr = reinterpret_cast<float *>(base + size_t(s) * dst->nb[3] + size_t(head) * dst->nb[2] + size_t(t) * dst->nb[1] + size_t(j) * dst->nb[0]);
                    *dst_ptr = layer.beta_data[src_row + j];
                }
            }
        }
    }

    const int64_t t1 = ggml_time_us();
    LLAMA_LOG_DEBUG("compact_exec: beta materialization %.1fms (prefix=%u)\n",
                    (t1 - t0) / 1000.0, (unsigned)layer.n_compacted_tokens);
}
