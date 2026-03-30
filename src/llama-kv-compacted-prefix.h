#pragma once

#include "llama-cparams.h"
#include "llama.h"

#include <cstdint>
#include <map>
#include <vector>

class llama_io_write_i;
class llama_io_read_i;

struct llama_compacted_prefix_layer_layout {
    uint32_t layer_id = 0;
    uint32_t n_head_kv = 0;
    uint32_t n_embd_head_k = 0;
    uint32_t n_embd_head_v = 0;
    ggml_type type_k = GGML_TYPE_COUNT;
    ggml_type type_v = GGML_TYPE_COUNT;
};

class llama_compacted_prefix_store {
public:
    static constexpr const char * k_quantized_cache_error =
        "compacted-prefix store requires cache types where head_dim is a multiple of the quantization block size";

    struct layer_storage {
        llama_compacted_prefix_layer_layout layout;
        uint32_t n_compacted_tokens = 0;

        std::vector<uint8_t> k_data;
        std::vector<float>   beta_data;
        std::vector<uint8_t> v_data;

        void configure(uint32_t n_tokens);
        void clear(bool data);

        size_t allocated_bytes() const;

        // Per-layer zero-beta cache for flash attention eligibility.
        // Updated when beta values are written or cleared.
        void update_zero_beta_cache();
        bool is_zero_beta() const { return zero_beta_cached; }

    private:
        bool zero_beta_cached = true;
    };

    struct sequence_state {
        bool enabled = false;

        // V4-J: IMROPE model flag. When true, is_pos_2d() batches are allowed
        // in execution. K/V data already has IMROPE rotations applied; causal
        // masking uses scalar positions only.
        bool is_imrope = false;

        // Snapshot of the logical prefix length represented at configure time.
        // Sequence operations mutate compacted token positions but do not infer a
        // new original logical-prefix length.
        uint32_t logical_token_count = 0;
        llama_pos live_suffix_pos0 = -1;
        std::vector<llama_pos> logical_positions;
        std::vector<layer_storage> layers;

        void clear(bool data);

        uint32_t compacted_token_count() const;
        llama_pos pos_min() const;
        llama_pos pos_max() const;

        size_t allocated_bytes() const;

        bool set_execution_enabled(bool enabled);
        bool is_execution_enabled() const;

        // Returns true when all beta values across all layers are zero.
        // This enables the flash-attention path since kq_b is unnecessary.
        bool is_zero_beta() const;

    private:
        bool execution_enabled = false;
    };

    explicit llama_compacted_prefix_store(std::vector<llama_compacted_prefix_layer_layout> layouts = {});

    bool configure_seq(
            llama_seq_id seq_id,
            uint32_t logical_token_count,
            const std::vector<llama_pos> & logical_positions,
            llama_pos live_suffix_pos0 = -1,
            bool is_imrope = false);

    void clear(bool data);
    void clear_seq(llama_seq_id seq_id, bool data = true);

    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1);
    void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
    void seq_keep(llama_seq_id seq_id);
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift);
    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d);

    llama_pos seq_pos_min(llama_seq_id seq_id) const;
    llama_pos seq_pos_max(llama_seq_id seq_id) const;

    bool is_enabled(llama_seq_id seq_id) const;
    bool set_execution(llama_seq_id seq_id, bool enabled);
    bool execution_enabled(llama_seq_id seq_id) const;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const;
    bool state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1);

    size_t seq_allocated_bytes(llama_seq_id seq_id) const;
    size_t total_allocated_bytes() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    sequence_state * get_seq(llama_seq_id seq_id);
    const sequence_state * get_seq(llama_seq_id seq_id) const;
    const std::vector<llama_compacted_prefix_layer_layout> & get_layouts() const;

private:
    static void normalize_range(llama_pos & p0, llama_pos & p1);
    static bool pos_in(llama_pos pos, llama_pos p0, llama_pos p1);
    static void validate_positions(const sequence_state & state);

    sequence_state & seq(llama_seq_id seq_id);
    const sequence_state & seq(llama_seq_id seq_id) const;

    std::vector<llama_compacted_prefix_layer_layout> layouts;
    std::vector<sequence_state> seq_states;
};
