#include "llama-kv-compacted-prefix.h"

#include "ggml.h"
#include "llama-io.h"
#include "llama-impl.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>

namespace {
constexpr uint32_t LLAMA_COMPACTED_PREFIX_STATE_VERSION = 2;

bool is_supported_compacted_type(ggml_type type, uint32_t head_dim_k, uint32_t head_dim_v) {
    // Scalar types always work.
    if (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F32) {
        return true;
    }
    // Quantized types work when head_dim is a multiple of the block size
    // so that per-head data aligns on block boundaries.
    const int64_t blk = ggml_blck_size(type);
    if (blk <= 0) {
        return false;
    }
    // Check both K and V head dims (V may be 0 for V-less architectures).
    if (head_dim_k > 0 && (head_dim_k % blk) != 0) {
        return false;
    }
    if (head_dim_v > 0 && (head_dim_v % blk) != 0) {
        return false;
    }
    return true;
}

size_t compacted_tensor_bytes(ggml_type type, size_t n_elem_per_head, uint32_t n_head_kv) {
    return size_t(n_head_kv) * ggml_row_size(type, n_elem_per_head);
}

size_t compacted_token_bytes(ggml_type type, uint32_t n_embd_head) {
    return ggml_row_size(type, n_embd_head);
}

void copy_selected_token_blocks(
        std::vector<uint8_t> & dst,
        const std::vector<uint8_t> & src,
        uint32_t n_head_kv,
        uint32_t src_n_tokens,
        size_t token_bytes,
        const std::vector<uint32_t> & keep_indices) {
    if (keep_indices.empty() || src.empty() || token_bytes == 0) {
        return;
    }

    const size_t dst_n_tokens = keep_indices.size();
    const size_t src_required = size_t(n_head_kv) * src_n_tokens * token_bytes;
    const size_t dst_required = size_t(n_head_kv) * dst_n_tokens * token_bytes;
    if (src.size() < src_required || dst.size() < dst_required) {
        throw std::runtime_error("compacted-prefix token block copy out of bounds");
    }

    for (uint32_t head = 0; head < n_head_kv; ++head) {
        for (size_t i = 0; i < dst_n_tokens; ++i) {
            const uint32_t src_token = keep_indices[i];
            if (src_token >= src_n_tokens) {
                throw std::runtime_error("compacted-prefix token index out of bounds");
            }
            const size_t src_offset = (size_t(head) * src_n_tokens + src_token) * token_bytes;
            const size_t dst_offset = (size_t(head) * dst_n_tokens + i) * token_bytes;
            std::copy_n(src.data() + src_offset, token_bytes, dst.data() + dst_offset);
        }
    }
}

void copy_selected_beta(
        std::vector<float> & dst,
        const std::vector<float> & src,
        uint32_t n_head_kv,
        uint32_t src_n_tokens,
        const std::vector<uint32_t> & keep_indices) {
    if (keep_indices.empty() || src.empty()) {
        return;
    }

    const size_t dst_n_tokens = keep_indices.size();
    const size_t src_required = size_t(n_head_kv) * src_n_tokens;
    const size_t dst_required = size_t(n_head_kv) * dst_n_tokens;
    if (src.size() < src_required || dst.size() < dst_required) {
        throw std::runtime_error("compacted-prefix beta copy out of bounds");
    }

    for (uint32_t head = 0; head < n_head_kv; ++head) {
        for (size_t i = 0; i < dst_n_tokens; ++i) {
            const uint32_t src_token = keep_indices[i];
            if (src_token >= src_n_tokens) {
                throw std::runtime_error("compacted-prefix beta index out of bounds");
            }
            dst[size_t(head) * dst_n_tokens + i] = src[size_t(head) * src_n_tokens + src_token];
        }
    }
}

std::vector<llama_compacted_prefix_store::layer_storage> rebuild_layers(
        const std::vector<llama_compacted_prefix_store::layer_storage> & source_layers,
        uint32_t src_n_tokens,
        const std::vector<llama_compacted_prefix_layer_layout> & layouts,
        const std::vector<uint32_t> & keep_indices) {
    const uint32_t dst_n_tokens = keep_indices.size();

    std::vector<llama_compacted_prefix_store::layer_storage> rebuilt(layouts.size());

    for (size_t i = 0; i < layouts.size(); ++i) {
        rebuilt[i].layout = layouts[i];
        rebuilt[i].configure(dst_n_tokens);

        if (i >= source_layers.size()) {
            continue;
        }

        const auto & old_layer = source_layers[i];
        auto & new_layer = rebuilt[i];

        copy_selected_token_blocks(
            new_layer.k_data,
            old_layer.k_data,
            new_layer.layout.n_head_kv,
            src_n_tokens,
            compacted_token_bytes(new_layer.layout.type_k, new_layer.layout.n_embd_head_k),
            keep_indices);

        copy_selected_beta(
            new_layer.beta_data,
            old_layer.beta_data,
            new_layer.layout.n_head_kv,
            src_n_tokens,
            keep_indices);

        copy_selected_token_blocks(
            new_layer.v_data,
            old_layer.v_data,
            new_layer.layout.n_head_kv,
            src_n_tokens,
            compacted_token_bytes(new_layer.layout.type_v, new_layer.layout.n_embd_head_v),
            keep_indices);
    }

    return rebuilt;
}

template<typename T>
void io_write_pod(llama_io_write_i & io, const T & value) {
    io.write(&value, sizeof(value));
}

template<typename T>
void io_read_pod(llama_io_read_i & io, T & value) {
    io.read(&value, sizeof(value));
}

void io_write_bytes(llama_io_write_i & io, const std::vector<uint8_t> & data) {
    const uint64_t n_bytes = data.size();
    io_write_pod(io, n_bytes);
    if (n_bytes > 0) {
        io.write(data.data(), n_bytes);
    }
}

void io_read_bytes(llama_io_read_i & io, std::vector<uint8_t> & data) {
    uint64_t n_bytes = 0;
    io_read_pod(io, n_bytes);
    // F-C-06: Practical max payload guard (2 GB) — the size_t::max check is tautological on 64-bit.
    constexpr uint64_t max_payload = uint64_t(2) * 1024 * 1024 * 1024;
    if (n_bytes > max_payload) {
        throw std::runtime_error("compacted-prefix byte payload too large (" + std::to_string(n_bytes) + " bytes)");
    }
    data.resize((size_t) n_bytes);
    if (n_bytes > 0) {
        io.read(data.data(), (size_t) n_bytes);
    }
}

void io_write_floats(llama_io_write_i & io, const std::vector<float> & data) {
    const uint64_t n_elem = data.size();
    io_write_pod(io, n_elem);
    if (n_elem > 0) {
        io.write(data.data(), n_elem * sizeof(float));
    }
}

void io_read_floats(llama_io_read_i & io, std::vector<float> & data) {
    uint64_t n_elem = 0;
    io_read_pod(io, n_elem);
    if (n_elem > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("compacted-prefix float payload too large");
    }
    data.resize((size_t) n_elem);
    if (n_elem > 0) {
        io.read(data.data(), (size_t) n_elem * sizeof(float));
    }
}
} // namespace

void llama_compacted_prefix_store::layer_storage::configure(uint32_t n_tokens) {
    // Sentinel check: types must be explicitly initialized before use.
    if (layout.type_k >= GGML_TYPE_COUNT || layout.type_v >= GGML_TYPE_COUNT) {
        throw std::runtime_error("compacted-prefix layout has uninitialized type");
    }

    if (!is_supported_compacted_type(layout.type_k, layout.n_embd_head_k, layout.n_embd_head_v) ||
        !is_supported_compacted_type(layout.type_v, layout.n_embd_head_k, layout.n_embd_head_v)) {
        throw std::runtime_error(k_quantized_cache_error);
    }

    // BF16 trait validation: ensure conversion functions are available.
    if (layout.type_k == GGML_TYPE_BF16 || layout.type_v == GGML_TYPE_BF16) {
        const auto * traits_k = ggml_get_type_traits(layout.type_k);
        const auto * traits_v = ggml_get_type_traits(layout.type_v);
        if (traits_k->from_float_ref == nullptr || traits_v->from_float_ref == nullptr) {
            throw std::runtime_error("compacted-prefix BF16 layout missing from_float_ref conversion");
        }
    }

    n_compacted_tokens = n_tokens;

    const size_t k_elems = size_t(layout.n_embd_head_k) * n_tokens;
    const size_t v_elems = size_t(layout.n_embd_head_v) * n_tokens;

    k_data.resize(compacted_tensor_bytes(layout.type_k, k_elems, layout.n_head_kv));
    beta_data.resize(size_t(layout.n_head_kv) * n_tokens);
    v_data.resize(compacted_tensor_bytes(layout.type_v, v_elems, layout.n_head_kv));
}

void llama_compacted_prefix_store::layer_storage::clear(bool data) {
    n_compacted_tokens = 0;
    k_data.clear();
    beta_data.clear();
    v_data.clear();
    zero_beta_cached = true;
    if (data) {
        k_data.shrink_to_fit();
        beta_data.shrink_to_fit();
        v_data.shrink_to_fit();
    }
}

void llama_compacted_prefix_store::layer_storage::update_zero_beta_cache() {
    zero_beta_cached = true;
    for (float b : beta_data) {
        if (b != 0.0f) {
            zero_beta_cached = false;
            return;
        }
    }
}

size_t llama_compacted_prefix_store::layer_storage::allocated_bytes() const {
    return k_data.size() + beta_data.size() * sizeof(float) + v_data.size();
}

void llama_compacted_prefix_store::sequence_state::clear(bool data) {
    enabled = false;
    execution_enabled = false;
    is_imrope = false;
    logical_token_count = 0;
    live_suffix_pos0 = -1;
    logical_positions.clear();

    for (auto & layer : layers) {
        layer.clear(true);
    }

    if (data) {
        layers.clear();
        layers.shrink_to_fit();
    }
}

uint32_t llama_compacted_prefix_store::sequence_state::compacted_token_count() const {
    return static_cast<uint32_t>(std::min(logical_positions.size(),
                                          size_t(std::numeric_limits<uint32_t>::max())));  // m-09: fix narrowing warning
}

llama_pos llama_compacted_prefix_store::sequence_state::pos_min() const {
    if (!enabled || logical_positions.empty()) {
        return -1;
    }

    return *std::min_element(logical_positions.begin(), logical_positions.end());
}

llama_pos llama_compacted_prefix_store::sequence_state::pos_max() const {
    if (!enabled || logical_positions.empty()) {
        return -1;
    }

    return *std::max_element(logical_positions.begin(), logical_positions.end());
}

size_t llama_compacted_prefix_store::sequence_state::allocated_bytes() const {
    size_t total = 0;
    for (const auto & layer : layers) {
        total += layer.allocated_bytes();
    }
    return total;
}

bool llama_compacted_prefix_store::sequence_state::set_execution_enabled(bool enabled_) {
    if (!enabled_) {
        execution_enabled = false;
        return true;
    }

    if (!enabled || logical_positions.empty() || layers.empty()) {
        return false;
    }

    const uint32_t n_tokens = logical_positions.size();
    int32_t n_zero_beta = 0;
    for (auto & layer : layers) {
        if (layer.n_compacted_tokens != n_tokens) {
            return false;
        }
        // Update per-layer zero-beta cache before execution begins.
        layer.update_zero_beta_cache();
        if (layer.is_zero_beta()) {
            n_zero_beta++;
        }
    }

    LLAMA_LOG_DEBUG("%s: compacted prefix enabled — %d/%zu layers zero-beta (flash-eligible)\n",
                    __func__, n_zero_beta, layers.size());

    execution_enabled = true;
    return true;
}

bool llama_compacted_prefix_store::sequence_state::is_execution_enabled() const {
    return execution_enabled;
}

bool llama_compacted_prefix_store::sequence_state::is_zero_beta() const {
    for (const auto & layer : layers) {
        if (!layer.is_zero_beta()) {
            return false;
        }
    }
    return true;
}

llama_compacted_prefix_store::llama_compacted_prefix_store(std::vector<llama_compacted_prefix_layer_layout> layouts)
    : layouts(std::move(layouts)), seq_states(LLAMA_MAX_SEQ) {
}

bool llama_compacted_prefix_store::configure_seq(
        llama_seq_id seq_id,
        uint32_t logical_token_count,
        const std::vector<llama_pos> & logical_positions,
        llama_pos live_suffix_pos0,
        bool is_imrope) {
    if (seq_id < 0 || size_t(seq_id) >= seq_states.size()) {
        return false;
    }
    for (llama_pos pos : logical_positions) {
        if (pos < 0) {
            throw std::runtime_error("compacted-prefix logical positions must be >= 0");
        }
    }
    if (live_suffix_pos0 < -1) {
        throw std::runtime_error("compacted-prefix live_suffix_pos0 must be >= -1");
    }
    if (logical_positions.size() > logical_token_count) {
        throw std::runtime_error("compacted-prefix logical_positions.size() must not exceed logical_token_count");
    }

    auto & state = seq(seq_id);
    state.clear(false);
    state.enabled = !logical_positions.empty();
    state.is_imrope = is_imrope;
    state.logical_token_count = logical_token_count;
    state.live_suffix_pos0 = live_suffix_pos0;
    state.logical_positions = logical_positions;
    state.layers.resize(layouts.size());

    for (size_t i = 0; i < layouts.size(); ++i) {
        state.layers[i].layout = layouts[i];
        state.layers[i].configure(logical_positions.size());
    }

    validate_positions(state);

    return true;
}

void llama_compacted_prefix_store::clear(bool data) {
    for (size_t i = 0; i < seq_states.size(); ++i) {
        clear_seq((llama_seq_id) i, data);
    }
}

void llama_compacted_prefix_store::clear_seq(llama_seq_id seq_id, bool data) {
    if (seq_id < 0 || size_t(seq_id) >= seq_states.size()) {
        return;
    }

    auto & state = seq(seq_id);
    state.clear(data);

    if (!data) {
        state.layers.resize(layouts.size());
        for (size_t i = 0; i < layouts.size(); ++i) {
            state.layers[i].layout = layouts[i];
        }
    }
}

bool llama_compacted_prefix_store::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    normalize_range(p0, p1);

    if (seq_id >= 0) {
        auto & state = seq(seq_id);
        if (!state.enabled) {
            return true;
        }

        std::vector<llama_pos> next_positions;
        std::vector<uint32_t> keep_indices;
        next_positions.reserve(state.logical_positions.size());
        keep_indices.reserve(state.logical_positions.size());

        for (uint32_t i = 0; i < state.logical_positions.size(); ++i) {
            const llama_pos pos = state.logical_positions[i];
            if (!pos_in(pos, p0, p1)) {
                keep_indices.push_back(i);
                next_positions.push_back(pos);
            }
        }

        if (state.live_suffix_pos0 >= 0 && pos_in(state.live_suffix_pos0, p0, p1)) {
            state.live_suffix_pos0 = -1;
        }

        if (next_positions.empty()) {
            clear_seq(seq_id, false);
        } else {
            state.layers = rebuild_layers(state.layers, state.logical_positions.size(), layouts, keep_indices);
            state.logical_positions = std::move(next_positions);
            validate_positions(state);
        }
        return true;
    }

    for (size_t i = 0; i < seq_states.size(); ++i) {
        seq_rm((llama_seq_id) i, p0, p1);
    }

    return true;
}

void llama_compacted_prefix_store::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    normalize_range(p0, p1);

    if (seq_id_src < 0 || seq_id_dst < 0 || seq_id_src == seq_id_dst) {
        return;
    }

    const auto & src = seq(seq_id_src);
    auto & dst = seq(seq_id_dst);

    const auto & src_layers = src.layers;
    const auto & src_positions = src.logical_positions;
    const auto src_logical_token_count = src.logical_token_count;
    const auto src_live_suffix_pos0 = src.live_suffix_pos0;

    dst.clear(false);

    if (!src.enabled) {
        return;
    }

    std::vector<uint32_t> keep_indices;
    std::vector<llama_pos> next_positions;
    next_positions.reserve(src_positions.size());
    keep_indices.reserve(src_positions.size());

    for (uint32_t i = 0; i < src_positions.size(); ++i) {
        const llama_pos pos = src_positions[i];
        if (pos_in(pos, p0, p1)) {
            keep_indices.push_back(i);
            next_positions.push_back(pos);
        }
    }

    if (next_positions.empty()) {
        return;
    }

    dst.enabled = true;
    dst.is_imrope = src.is_imrope;
    dst.logical_token_count = src_logical_token_count;
    dst.live_suffix_pos0 = pos_in(src_live_suffix_pos0, p0, p1) ? src_live_suffix_pos0 : -1;
    dst.logical_positions = std::move(next_positions);
    dst.layers = rebuild_layers(src_layers, src_positions.size(), layouts, keep_indices);
    validate_positions(dst);
}

void llama_compacted_prefix_store::seq_keep(llama_seq_id seq_id) {
    for (size_t i = 0; i < seq_states.size(); ++i) {
        if ((llama_seq_id) i == seq_id) {
            continue;
        }
        clear_seq((llama_seq_id) i, true);
    }
}

void llama_compacted_prefix_store::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (shift == 0 || seq_id < 0) {
        return;
    }

    normalize_range(p0, p1);

    auto & state = seq(seq_id);
    if (!state.enabled) {
        return;
    }

    for (auto & pos : state.logical_positions) {
        if (pos_in(pos, p0, p1)) {
            pos += shift;
            if (pos < 0) {
                throw std::runtime_error("compacted-prefix position became negative after seq_add");
            }
        }
    }

    if (state.live_suffix_pos0 >= 0 && pos_in(state.live_suffix_pos0, p0, p1)) {
        state.live_suffix_pos0 += shift;
        if (state.live_suffix_pos0 < 0) {
            throw std::runtime_error("compacted-prefix live_suffix_pos0 became negative after seq_add");
        }
    }

    validate_positions(state);
}

void llama_compacted_prefix_store::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (seq_id < 0) {
        return;
    }
    if (d == 0) {
        throw std::runtime_error("compacted-prefix seq_div does not support d == 0");
    }
    if (d == 1) {
        return;
    }

    normalize_range(p0, p1);

    auto & state = seq(seq_id);
    if (!state.enabled) {
        return;
    }

    // Compute divided positions into a temporary to validate uniqueness before mutating state.
    std::vector<llama_pos> new_positions = state.logical_positions;
    for (auto & pos : new_positions) {
        if (pos_in(pos, p0, p1)) {
            pos /= d;
        }
    }

    llama_pos new_suffix_pos0 = state.live_suffix_pos0;
    if (new_suffix_pos0 >= 0 && pos_in(new_suffix_pos0, p0, p1)) {
        new_suffix_pos0 /= d;
    }

    // F-M-30: lightweight position-only validation — avoid copying full state
    // (which includes per-layer data).  validate_positions only reads
    // logical_positions and live_suffix_pos0, so validate in-place on temps.
    {
        if (new_suffix_pos0 < -1) {
            throw std::runtime_error("compacted-prefix live_suffix_pos0 must remain >= -1");
        }
        std::set<llama_pos> seen;
        for (llama_pos pos : new_positions) {
            if (pos < 0) {
                throw std::runtime_error("compacted-prefix positions must remain non-negative");
            }
            if (!seen.insert(pos).second) {
                throw std::runtime_error("compacted-prefix positions must remain unique");
            }
        }
    }

    state.logical_positions = std::move(new_positions);
    state.live_suffix_pos0 = new_suffix_pos0;
}

llama_pos llama_compacted_prefix_store::seq_pos_min(llama_seq_id seq_id) const {
    if (seq_id < 0 || size_t(seq_id) >= seq_states.size()) {
        return -1;
    }
    return seq(seq_id).pos_min();
}

llama_pos llama_compacted_prefix_store::seq_pos_max(llama_seq_id seq_id) const {
    if (seq_id < 0 || size_t(seq_id) >= seq_states.size()) {
        return -1;
    }
    return seq(seq_id).pos_max();
}

bool llama_compacted_prefix_store::is_enabled(llama_seq_id seq_id) const {
    if (seq_id < 0 || size_t(seq_id) >= seq_states.size()) {
        return false;
    }
    return seq(seq_id).enabled;
}

bool llama_compacted_prefix_store::set_execution(llama_seq_id seq_id, bool enabled) {
    if (seq_id < 0 || size_t(seq_id) >= seq_states.size()) {
        return false;
    }

    return seq(seq_id).set_execution_enabled(enabled);
}

bool llama_compacted_prefix_store::execution_enabled(llama_seq_id seq_id) const {
    if (seq_id < 0 || size_t(seq_id) >= seq_states.size()) {
        return false;
    }

    const auto & state = seq(seq_id);
    return state.enabled && state.is_execution_enabled();
}

void llama_compacted_prefix_store::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    io_write_pod(io, LLAMA_COMPACTED_PREFIX_STATE_VERSION);

    std::vector<llama_seq_id> saved_seq_ids;
    if (seq_id >= 0) {
        if (size_t(seq_id) >= seq_states.size()) {
            throw std::runtime_error("compacted-prefix state_write seq_id out of range");
        }
        if (is_enabled(seq_id)) {
            saved_seq_ids.push_back(seq_id);
        }
    } else {
        for (llama_seq_id cur = 0; cur < (llama_seq_id) seq_states.size(); ++cur) {
            if (is_enabled(cur)) {
                saved_seq_ids.push_back(cur);
            }
        }
    }

    const uint32_t n_seq = saved_seq_ids.size();
    io_write_pod(io, n_seq);

    for (llama_seq_id saved_seq_id : saved_seq_ids) {
        const auto & state = seq(saved_seq_id);

        io_write_pod(io, saved_seq_id);
        io_write_pod(io, state.logical_token_count);
        io_write_pod(io, state.live_suffix_pos0);

        const uint8_t execution = state.is_execution_enabled() ? 1 : 0;
        io_write_pod(io, execution);

        const uint8_t imrope_flag = state.is_imrope ? 1 : 0;
        io_write_pod(io, imrope_flag);

        const uint32_t n_positions = state.logical_positions.size();
        io_write_pod(io, n_positions);
        if (n_positions > 0) {
            io.write(state.logical_positions.data(), size_t(n_positions) * sizeof(llama_pos));
        }

        const uint32_t n_layers = state.layers.size();
        io_write_pod(io, n_layers);

        for (const auto & layer : state.layers) {
            io_write_pod(io, layer.layout.layer_id);
            io_write_pod(io, layer.layout.n_head_kv);
            io_write_pod(io, layer.layout.n_embd_head_k);
            io_write_pod(io, layer.layout.n_embd_head_v);

            const int32_t type_k = (int32_t) layer.layout.type_k;
            const int32_t type_v = (int32_t) layer.layout.type_v;
            io_write_pod(io, type_k);
            io_write_pod(io, type_v);

            io_write_pod(io, layer.n_compacted_tokens);
            io_write_bytes(io, layer.k_data);
            io_write_floats(io, layer.beta_data);
            io_write_bytes(io, layer.v_data);
        }
    }
}

bool llama_compacted_prefix_store::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    // F-C-21: Clear existing state for full restore (seq_id == -1) to avoid stale data.
    if (seq_id < 0) {
        clear(false);
    }

    uint32_t version = 0;
    io_read_pod(io, version);
    if (version != LLAMA_COMPACTED_PREFIX_STATE_VERSION) {
        throw std::runtime_error("compacted-prefix state version mismatch");
    }

    uint32_t n_seq = 0;
    io_read_pod(io, n_seq);
    if (seq_id >= 0 && n_seq > 1) {
        throw std::runtime_error("per-sequence compacted-prefix state contains multiple sequences");
    }

    std::set<llama_seq_id> restored_seq_ids;

    for (uint32_t i = 0; i < n_seq; ++i) {
        llama_seq_id stored_seq_id = -1;
        uint32_t logical_token_count = 0;
        llama_pos live_suffix_pos0 = -1;
        uint8_t execution = 0;
        uint32_t n_positions = 0;
        uint32_t n_layers = 0;

        io_read_pod(io, stored_seq_id);
        io_read_pod(io, logical_token_count);
        io_read_pod(io, live_suffix_pos0);
        io_read_pod(io, execution);

        uint8_t imrope_flag = 0;
        io_read_pod(io, imrope_flag);

        io_read_pod(io, n_positions);

        std::vector<llama_pos> logical_positions(n_positions);
        if (n_positions > 0) {
            io.read(logical_positions.data(), size_t(n_positions) * sizeof(llama_pos));
        }

        io_read_pod(io, n_layers);
        if (n_layers != layouts.size()) {
            throw std::runtime_error("compacted-prefix layer count mismatch");
        }

        const llama_seq_id dst_seq_id = seq_id >= 0 ? seq_id : stored_seq_id;
        if (dst_seq_id < 0 || size_t(dst_seq_id) >= seq_states.size()) {
            throw std::runtime_error("compacted-prefix restore seq_id out of range");
        }
        if (!restored_seq_ids.insert(dst_seq_id).second) {
            throw std::runtime_error("duplicate compacted-prefix sequence restore entry");
        }

        if (!configure_seq(dst_seq_id, logical_token_count, logical_positions, live_suffix_pos0, imrope_flag != 0)) {
            throw std::runtime_error("failed to configure compacted-prefix sequence during restore");
        }

        auto & state = seq(dst_seq_id);
        for (uint32_t layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
            uint32_t layer_id = 0;
            uint32_t n_head_kv = 0;
            uint32_t n_embd_head_k = 0;
            uint32_t n_embd_head_v = 0;
            int32_t type_k_i = 0;
            int32_t type_v_i = 0;
            uint32_t n_compacted_tokens = 0;

            io_read_pod(io, layer_id);
            io_read_pod(io, n_head_kv);
            io_read_pod(io, n_embd_head_k);
            io_read_pod(io, n_embd_head_v);
            io_read_pod(io, type_k_i);
            io_read_pod(io, type_v_i);
            io_read_pod(io, n_compacted_tokens);

            const auto & expected = layouts.at(layer_idx);
            if (layer_id != expected.layer_id ||
                n_head_kv != expected.n_head_kv ||
                n_embd_head_k != expected.n_embd_head_k ||
                n_embd_head_v != expected.n_embd_head_v ||
                type_k_i != (int32_t) expected.type_k ||
                type_v_i != (int32_t) expected.type_v) {
                throw std::runtime_error("compacted-prefix layer layout mismatch during restore");
            }

            auto & layer = state.layers.at(layer_idx);
            if (n_compacted_tokens != layer.n_compacted_tokens) {
                throw std::runtime_error("compacted-prefix token count mismatch during restore");
            }

            // F-C-07: Validate expected sizes BEFORE reading payloads to avoid allocating garbage sizes.
            const size_t expected_k_size = compacted_tensor_bytes(layer.layout.type_k, layer.layout.n_embd_head_k * n_compacted_tokens, layer.layout.n_head_kv);
            const size_t expected_beta_size = size_t(layer.layout.n_head_kv) * n_compacted_tokens;
            const size_t expected_v_size = compacted_tensor_bytes(layer.layout.type_v, layer.layout.n_embd_head_v * n_compacted_tokens, layer.layout.n_head_kv);

            io_read_bytes(io, layer.k_data);
            io_read_floats(io, layer.beta_data);
            io_read_bytes(io, layer.v_data);

            if (layer.k_data.size() != expected_k_size ||
                layer.beta_data.size() != expected_beta_size ||
                layer.v_data.size() != expected_v_size) {
                throw std::runtime_error("compacted-prefix payload size mismatch during restore");
            }
        }

        if (execution != 0 && !set_execution(dst_seq_id, true)) {
            throw std::runtime_error("failed to restore compacted-prefix execution state");
        }
    }

    if (seq_id >= 0 && restored_seq_ids.empty()) {
        clear_seq(seq_id, true);
    }

    return true;
}

size_t llama_compacted_prefix_store::seq_allocated_bytes(llama_seq_id seq_id) const {
    if (seq_id < 0 || size_t(seq_id) >= seq_states.size()) {
        return 0;
    }
    return seq(seq_id).allocated_bytes();
}

size_t llama_compacted_prefix_store::total_allocated_bytes() const {
    size_t total = 0;
    for (const auto & state : seq_states) {
        total += state.allocated_bytes();
    }
    return total;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_compacted_prefix_store::memory_breakdown() const {
    const size_t total = total_allocated_bytes();
    if (total == 0) {
        return {};
    }

    return {
        { ggml_backend_cpu_buffer_type(), total }
    };
}

llama_compacted_prefix_store::sequence_state * llama_compacted_prefix_store::get_seq(llama_seq_id seq_id) {
    if (seq_id < 0 || size_t(seq_id) >= seq_states.size()) {
        return nullptr;
    }
    return &seq(seq_id);
}

const llama_compacted_prefix_store::sequence_state * llama_compacted_prefix_store::get_seq(llama_seq_id seq_id) const {
    if (seq_id < 0 || size_t(seq_id) >= seq_states.size()) {
        return nullptr;
    }
    return &seq(seq_id);
}

const std::vector<llama_compacted_prefix_layer_layout> & llama_compacted_prefix_store::get_layouts() const {
    return layouts;
}

void llama_compacted_prefix_store::normalize_range(llama_pos & p0, llama_pos & p1) {
    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }
}

bool llama_compacted_prefix_store::pos_in(llama_pos pos, llama_pos p0, llama_pos p1) {
    return pos >= p0 && pos < p1;
}

void llama_compacted_prefix_store::validate_positions(const sequence_state & state) {
    if (state.live_suffix_pos0 < -1) {
        throw std::runtime_error("compacted-prefix live_suffix_pos0 must remain >= -1");
    }

    std::set<llama_pos> seen;
    for (llama_pos pos : state.logical_positions) {
        if (pos < 0) {
            throw std::runtime_error("compacted-prefix positions must remain non-negative");
        }
        if (!seen.insert(pos).second) {
            throw std::runtime_error("compacted-prefix positions must remain unique");
        }
    }
}

llama_compacted_prefix_store::sequence_state & llama_compacted_prefix_store::seq(llama_seq_id seq_id) {
    return seq_states.at(seq_id);
}

const llama_compacted_prefix_store::sequence_state & llama_compacted_prefix_store::seq(llama_seq_id seq_id) const {
    return seq_states.at(seq_id);
}
