// Stub implementations when KV compaction is disabled (-DLLAMA_KV_COMPACTION=OFF).
// Compiled only when LLAMA_KV_COMPACTION is OFF (see src/CMakeLists.txt).
#ifndef LLAMA_KV_COMPACTION

#include "llama.h"
#include "llama-impl.h"

struct llama_compact_params llama_compact_default_params(void) {
    struct llama_compact_params p = {};
    return p;
}

int32_t llama_kv_cache_compact(
        struct llama_context * /*ctx*/,
        llama_seq_id           /*seq_id*/,
        struct llama_compact_params /*params*/) {
    LLAMA_LOG_WARN("%s: KV compaction not compiled (LLAMA_KV_COMPACTION=OFF)\n", __func__);
    return -1;
}

void llama_kv_cache_set_auto_compact(
        struct llama_context * /*ctx*/,
        float                  /*ratio*/,
        struct llama_compact_params /*params*/) {
    LLAMA_LOG_WARN("%s: KV compaction not compiled (LLAMA_KV_COMPACTION=OFF)\n", __func__);
}

#endif // !LLAMA_KV_COMPACTION
