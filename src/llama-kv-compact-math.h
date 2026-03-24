// Shared vectorized math for KV compaction solver.
//
// NEON-optimized dot product for Apple Silicon.
// Falls back to scalar on other architectures.

#pragma once

#include <cstdint>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

static inline float llama_kv_compact_dot_row(const float * a, const float * b, uint32_t n) {
#ifdef __ARM_NEON__
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    uint32_t i = 0;

    for (; i + 8 <= n; i += 8) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        sum0 = vfmaq_f32(sum0, va0, vb0);

        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        sum1 = vfmaq_f32(sum1, va1, vb1);
    }

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum0 = vfmaq_f32(sum0, va, vb);
    }

    sum0 = vaddq_f32(sum0, sum1);
    float v = vaddvq_f32(sum0);

    for (; i < n; ++i) {
        v += a[i] * b[i];
    }
    return v;
#else
    float v = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        v += a[i] * b[i];
    }
    return v;
#endif
}
