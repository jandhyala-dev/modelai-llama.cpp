#include "llama-kv-compacted-prefix.h"

#include "ggml.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace {
bool is_supported_compacted_type(ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F32;
}

size_t compacted_tensor_bytes(ggml_type type, uint32_t n_elem_per_head, uint32_t n_head_kv) {
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
}

void llama_compacted_prefix_store::layer_storage::configure(uint32_t n_tokens) {
    if (!is_supported_compacted_type(layout.type_k) || !is_supported_compacted_type(layout.type_v)) {
        throw std::runtime_error(k_quantized_cache_error);
    }

    n_compacted_tokens = n_tokens;

    const uint32_t k_elems = layout.n_embd_head_k * n_tokens;
    const uint32_t v_elems = layout.n_embd_head_v * n_tokens;

    k_data.resize(compacted_tensor_bytes(layout.type_k, k_elems, layout.n_head_kv));
    beta_data.resize(size_t(layout.n_head_kv) * n_tokens);
    v_data.resize(compacted_tensor_bytes(layout.type_v, v_elems, layout.n_head_kv));
}

void llama_compacted_prefix_store::layer_storage::clear(bool data) {
    n_compacted_tokens = 0;
    k_data.clear();
    beta_data.clear();
    v_data.clear();
    if (data) {
        k_data.shrink_to_fit();
        beta_data.shrink_to_fit();
        v_data.shrink_to_fit();
    }
}

size_t llama_compacted_prefix_store::layer_storage::allocated_bytes() const {
    return k_data.size() + beta_data.size() * sizeof(float) + v_data.size();
}

void llama_compacted_prefix_store::sequence_state::clear(bool data) {
    enabled = false;
    execution_enabled = false;
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
    return logical_positions.size();
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
    for (const auto & layer : layers) {
        if (layer.n_compacted_tokens != n_tokens) {
            return false;
        }
    }

    execution_enabled = true;
    return true;
}

bool llama_compacted_prefix_store::sequence_state::is_execution_enabled() const {
    return execution_enabled;
}

llama_compacted_prefix_store::llama_compacted_prefix_store(std::vector<llama_compacted_prefix_layer_layout> layouts)
    : layouts(std::move(layouts)), seq_states(LLAMA_MAX_SEQ) {
}

bool llama_compacted_prefix_store::configure_seq(
        llama_seq_id seq_id,
        uint32_t logical_token_count,
        const std::vector<llama_pos> & logical_positions,
        llama_pos live_suffix_pos0) {
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

    auto & state = seq(seq_id);
    state.clear(false);
    state.enabled = !logical_positions.empty();
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

    for (auto & pos : state.logical_positions) {
        if (pos_in(pos, p0, p1)) {
            pos /= d;
        }
    }

    if (state.live_suffix_pos0 >= 0 && pos_in(state.live_suffix_pos0, p0, p1)) {
        state.live_suffix_pos0 /= d;
    }

    validate_positions(state);
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
