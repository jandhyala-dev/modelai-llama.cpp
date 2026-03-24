#pragma once

// Centralized KV cache extraction from any memory backend.
// Eliminates the dynamic_cast cascade duplicated across API, server, bench, and tests.
// When new memory types are added, update only this function.

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"

static inline llama_kv_cache * llama_kv_compact_get_cache(llama_memory_i * mem) {
    if (!mem) return nullptr;
    if (auto * kv = dynamic_cast<llama_kv_cache *>(mem)) return kv;
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(mem)) return iswa->get_base();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) return hybrid->get_mem_attn();
    if (auto * hiswa = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) return hiswa->get_mem_attn()->get_base();
    return nullptr;
}
