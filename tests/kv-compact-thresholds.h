#pragma once

// Cosine similarity thresholds for KV compaction quality validation.
// Based on arXiv:2602.16284 Attention Matching empirical results across 15 models.

namespace llama_kv_compact_thresholds {
    // Per-compression-ratio thresholds (single-token lookahead)
    constexpr float COS_2X = 0.95f;   // 2x compression
    constexpr float COS_4X = 0.90f;   // 4x compression
    constexpr float COS_8X = 0.85f;   // 8x compression

    // Multi-token continuation threshold (4x, all tokens)
    constexpr float COS_MULTI_4X = 0.85f;

    // Synthetic solver baseline
    constexpr float COS_SOLVER_BASELINE = 0.95f;
}
