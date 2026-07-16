#pragma once

#include "llama-batch.h"
#ifdef LLAMA_KV_COMPACTION
#include "llama-kv-compacted-prefix-exec.h"
#include "llama-kv-compacted-prefix.h"
#endif
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <cstdlib>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

#ifdef LLAMA_KV_COMPACTION
struct llama_kv_compact_pipeline_stats;
struct llama_kv_compact_self_study_config;
struct llama_kv_compact_self_study_stats;

#include "llama-kv-compact-on-policy.h"
#endif

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // TODO: refactor the memory instances to not depend on `llama_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llama_memory_params`
    llama_kv_cache(
            const llama_model & model,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share
#ifdef LLAMA_KV_COMPACTION
        ,                bool   enable_compacted_prefix = true
#endif
        );

    ~llama_kv_cache() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

#ifdef LLAMA_KV_COMPACTION
    //
    // compacted-prefix internal API
    //

    bool compacted_prefix_configure(
            llama_seq_id seq_id,
            uint32_t logical_token_count,
            const std::vector<llama_pos> & logical_positions,
            llama_pos live_suffix_pos0 = -1);

    void compacted_prefix_clear(llama_seq_id seq_id = -1, bool data = true);

    bool compacted_prefix_enabled(llama_seq_id seq_id) const;
    bool compacted_prefix_set_execution(llama_seq_id seq_id, bool enabled);
    bool compacted_prefix_execution_enabled(llama_seq_id seq_id) const;

    size_t compacted_prefix_bytes(llama_seq_id seq_id = -1) const;
    uint32_t compacted_prefix_active_n_kv(llama_seq_id seq_id) const;
    bool compacted_prefix_reclaim_live_kv(llama_seq_id seq_id);
    bool compacted_prefix_fit_from_live_kv(
            llama_seq_id seq_id,
            uint32_t target_tokens,
            llama_pos live_suffix_pos0,
            llama_kv_compact_pipeline_stats * stats = nullptr,
            llama_pos p0 = 0,
            uint32_t max_queries = 256,
            int nnls_iters = 2,
            float lambda = 1e-6f);
    bool compacted_prefix_select_from_live_kv(
            llama_seq_id seq_id,
            uint32_t target_tokens,
            llama_pos live_suffix_pos0,
            llama_kv_compact_pipeline_stats * stats = nullptr,
            llama_pos p0 = 0);
    bool compacted_prefix_omp_from_live_kv(
            llama_seq_id seq_id,
            uint32_t target_tokens,
            llama_pos live_suffix_pos0,
            llama_kv_compact_pipeline_stats * stats = nullptr,
            llama_pos p0 = 0,
            uint32_t max_queries = 256,
            int nnls_iters = 2,
            float lambda = 1e-6f);
    bool compacted_prefix_self_study_from_live_kv(
            struct llama_context * ctx,
            llama_seq_id seq_id,
            uint32_t target_tokens,
            llama_pos live_suffix_pos0,
            const llama_kv_compact_self_study_config & config,
            llama_kv_compact_self_study_stats * stats = nullptr,
            llama_pos p0 = 0);
    bool compacted_prefix_chunked_self_study_from_live_kv(
            struct llama_context * ctx,
            llama_seq_id seq_id,
            uint32_t target_tokens,
            llama_pos live_suffix_pos0,
            const llama_kv_compact_self_study_config & config,
            llama_kv_compact_self_study_stats * stats = nullptr,
            llama_pos p0 = 0,
            uint32_t chunk_size = 8192);
    bool compacted_prefix_nonuniform_from_live_kv(
            llama_seq_id seq_id,
            uint32_t target_tokens,
            llama_pos live_suffix_pos0,
            llama_kv_compact_pipeline_stats * stats = nullptr,
            llama_pos p0 = 0,
            uint32_t max_queries = 256,
            int nnls_iters = 2,
            float lambda = 1e-6f,
            uint32_t min_per_head = 4);
    bool compacted_prefix_chunked_from_live_kv(
            llama_seq_id seq_id,
            uint32_t target_tokens,
            llama_pos live_suffix_pos0,
            llama_kv_compact_pipeline_stats * stats = nullptr,
            llama_pos p0 = 0,
            uint32_t max_queries = 256,
            int nnls_iters = 2,
            float lambda = 1e-6f,
            uint32_t chunk_size = 8192);
    bool compacted_prefix_on_policy_from_live_kv(
            struct llama_context * ctx,
            llama_seq_id seq_id,
            uint32_t target_tokens,
            llama_pos live_suffix_pos0,
            llama_kv_compact_pipeline_stats * stats = nullptr,
            llama_pos p0 = 0,
            uint32_t max_queries = 256,
            int nnls_iters = 2,
            float lambda = 1e-6f,
            uint32_t n_generate_q = 128);
    bool compacted_prefix_iterative_on_policy_from_live_kv(
            struct llama_context * ctx,
            llama_seq_id seq_id,
            uint32_t target_tokens,
            llama_pos live_suffix_pos0,
            llama_kv_compact_pipeline_stats * stats = nullptr,
            llama_pos p0 = 0,
            const llama_kv_compact_on_policy_config & config = {});
    bool compacted_prefix_sequential_on_policy_from_live_kv(
            struct llama_context * ctx,
            llama_seq_id seq_id,
            uint32_t target_tokens,
            llama_pos live_suffix_pos0,
            llama_kv_compact_pipeline_stats * stats = nullptr,
            llama_pos p0 = 0,
            const llama_kv_compact_sequential_config & config = {});

    bool compacted_prefix_layer_layout_for_solver(int32_t il, llama_compacted_prefix_layer_layout & out) const;
    bool compacted_prefix_seq_positions(llama_seq_id seq_id, llama_pos p0, llama_pos p1, std::vector<llama_pos> & out) const;
    bool compacted_prefix_copy_k_head_f32(int32_t il, llama_seq_id seq_id, uint32_t head_kv, const std::vector<llama_pos> & positions, std::vector<float> & out) const;
    bool compacted_prefix_copy_v_head_f32(int32_t il, llama_seq_id seq_id, uint32_t head_kv, const std::vector<llama_pos> & positions, std::vector<float> & out) const;

    const llama_compacted_prefix_store * get_compacted_prefix() const;
          llama_compacted_prefix_store * get_compacted_prefix();

    // Compaction capability and state queries (6b-18, 6b-19)
    bool supports_compaction() const;
    std::string compaction_unsupported_reason() const;
    bool has_compacted_prefix() const;
    const std::string & compacted_prefix_method() const;
    bool compacted_prefix_forces_non_flash() const;

    // Per-layer zero-beta query for flash attention eligibility (Phase 7).
    // Returns true if the specified layer has all-zero betas (flash-eligible).
    bool compacted_prefix_layer_zero_beta(llama_seq_id seq_id, int32_t il) const;
#endif // LLAMA_KV_COMPACTION

    ggml_type type_k() const;
    ggml_type type_v() const;

    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

#ifdef LLAMA_KV_COMPACTION
    bool resolve_compacted_prefix_exec(
            const llama_ubatch & ubatch,
            llama_compacted_prefix_exec_candidate & out) const;

    void set_input_compacted_prefix_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn, llama_seq_id seq_id) const;
    void set_input_compacted_prefix_k   (ggml_tensor * dst, int32_t il, llama_seq_id seq_id) const;
    void set_input_compacted_prefix_v   (ggml_tensor * dst, int32_t il, llama_seq_id seq_id) const;
    void set_input_compacted_prefix_kq_b(ggml_tensor * dst, int32_t il, llama_seq_id seq_id) const;

    uint64_t compacted_prefix_state_version() const { return compacted_prefix_version_counter; }

    // Phase 8: Invalidate tensor cache after single-layer refit modifies layer
    // data in place.  Forces re-materialization of cached K/V/beta tensors on
    // the next decode.
    void compacted_prefix_bump_version() { ++compacted_prefix_version_counter; cp_cache.invalidate(); }
#endif // LLAMA_KV_COMPACTION

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
#ifdef LLAMA_KV_COMPACTION
    std::string compacted_prefix_last_method = "none";

    // Version counter for tensor caching — bumped on configure/clear/state_read.
    uint64_t compacted_prefix_version_counter = 0;

    // Tensor cache: stores materialized K/V/beta bytes per layer to avoid
    // repeated strided copies during decode. Mask is NOT cached (depends on
    // ubatch.pos which changes every decode). Invalidated by version bump.
    //
    // Zero-copy optimization: cache buffers are 64-byte aligned so that
    // dst->data can point directly at the cached memory, eliminating the
    // per-decode memcpy for K/V/beta tensors.
    struct aligned_byte_buffer {
        void * ptr = nullptr;
        size_t len = 0;

        aligned_byte_buffer() = default;
        ~aligned_byte_buffer() { clear(); }

        // Non-copyable, movable.
        aligned_byte_buffer(const aligned_byte_buffer &) = delete;
        aligned_byte_buffer & operator=(const aligned_byte_buffer &) = delete;
        aligned_byte_buffer(aligned_byte_buffer && o) noexcept : ptr(o.ptr), len(o.len) {
            o.ptr = nullptr;
            o.len = 0;
        }
        aligned_byte_buffer & operator=(aligned_byte_buffer && o) noexcept {
            if (this != &o) {
                clear();
                ptr = o.ptr;
                len = o.len;
                o.ptr = nullptr;
                o.len = 0;
            }
            return *this;
        }

        void resize(size_t n) {
            if (n == len && ptr) {
                return;
            }
            clear();
            if (n > 0) {
#ifdef _WIN32
                ptr = _aligned_malloc(n, 64);
                GGML_ASSERT(ptr && "aligned_byte_buffer: allocation failed");
#else
                int ret = posix_memalign(&ptr, 64, n);
                GGML_ASSERT(ret == 0 && ptr && "aligned_byte_buffer: allocation failed");
#endif
                len = n;
            }
        }
        void clear() {
            if (ptr) {
#ifdef _WIN32
                _aligned_free(ptr);
#else
                free(ptr);
#endif
                ptr = nullptr;
            }
            len = 0;
        }
        uint8_t *       data()       { return (uint8_t *)ptr; }
        const uint8_t * data() const { return (const uint8_t *)ptr; }
        size_t          size() const { return len; }
        bool           empty() const { return len == 0; }
    };

    struct cp_tensor_cache_t {
        uint64_t    version = 0;
        llama_seq_id seq_id = -1;

        // Per-layer cached bytes (indexed by KV layer id, not model layer id).
        // Aligned to 64 bytes for zero-copy pointer swap into ggml tensors.
        std::vector<aligned_byte_buffer> k_bytes;
        std::vector<aligned_byte_buffer> v_bytes;
        std::vector<aligned_byte_buffer> beta_bytes;
        uint32_t beta_n_tps = 0;  // shape guard for beta

        bool valid(llama_seq_id sid, uint64_t ver) const {
            return seq_id == sid && version == ver && version > 0;
        }
        void invalidate() { version = 0; }
    };
    mutable cp_tensor_cache_t cp_cache;

    bool compacted_prefix_runtime_supported() const;
    bool compacted_prefix_stream_owned_by_seq(uint32_t strm, llama_seq_id seq_id, std::vector<uint32_t> & live_cell_idxs) const;
    void compacted_prefix_pack_stream_tensors(uint32_t strm, const std::vector<uint32_t> & live_cell_idxs);
#endif // LLAMA_KV_COMPACTION

    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    bool attn_rot_k = false;
    bool attn_rot_v = false;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llama_kv_cache * other;

    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;

    llama_kv_cells_vec & v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

#ifdef LLAMA_KV_COMPACTION
    llama_compacted_prefix_store compacted_prefix;
#endif

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

#ifdef LLAMA_KV_COMPACTION
    bool compacted_prefix_active() const;
    llama_seq_id compacted_prefix_seq_id() const;
    uint32_t compacted_prefix_n_tokens() const;
    bool compacted_prefix_zero_beta() const;
    bool compacted_prefix_layer_zero_beta(int32_t il) const;

    void set_input_compacted_prefix_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_compacted_prefix_k   (ggml_tensor * dst, int32_t il) const;
    void set_input_compacted_prefix_v   (ggml_tensor * dst, int32_t il) const;
    void set_input_compacted_prefix_kq_b(ggml_tensor * dst, int32_t il) const;
#endif // LLAMA_KV_COMPACTION

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;

#ifdef LLAMA_KV_COMPACTION
    llama_compacted_prefix_exec_candidate compacted_exec;
#endif
};
