#include "src/llama-kv-compact-solver-metal.h"
#include "src/llama-kv-compact-select.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

static float cosine_sim(const float * a, const float * b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0f;
    return (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}

static float max_abs_diff(const float * a, const float * b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

// CPU reference: compute softmax(Q·Kᵀ/√d) summed across queries.
static void cpu_attention_scores(
        const float * Q, uint32_t n, uint32_t d,
        const float * K, uint32_t T,
        float * out) {
    const float inv_sqrt_d = 1.0f / std::sqrt((float)d);
    std::vector<float> weights(T);
    std::memset(out, 0, T * sizeof(float));

    for (uint32_t qi = 0; qi < n; ++qi) {
        float row_max = -INFINITY;
        for (uint32_t ki = 0; ki < T; ++ki) {
            float score = 0.0f;
            for (uint32_t i = 0; i < d; ++i) {
                score += Q[qi * d + i] * K[ki * d + i];
            }
            score *= inv_sqrt_d;
            weights[ki] = score;
            row_max = std::max(row_max, score);
        }
        float sum = 0.0f;
        for (uint32_t ki = 0; ki < T; ++ki) {
            weights[ki] = std::exp(weights[ki] - row_max);
            sum += weights[ki];
        }
        float inv_sum = 1.0f / std::max(sum, 1e-6f);
        for (uint32_t ki = 0; ki < T; ++ki) {
            out[ki] += weights[ki] * inv_sum;
        }
    }
}

// CPU reference: XᵀX
static void cpu_xtx(const float * X, uint32_t n, uint32_t t, float * out) {
    for (uint32_t i = 0; i < t; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (uint32_t r = 0; r < n; ++r) {
                sum += X[r * t + i] * X[r * t + j];
            }
            out[i * t + j] = sum;
            out[j * t + i] = sum;
        }
    }
}

int main() {
    auto * ctx = llama_kv_compact_metal_create();
    if (!ctx) {
        std::printf("test-kv-compact-solver-metal: Metal unavailable — SKIP\n");
        return 0;
    }

    int rc = 0;

    // --- Test 1: Small attention scores (4 queries, 4 keys, d=2) ---
    {
        const uint32_t n = 4, T = 4, d = 2;
        float Q[] = { 1,0, 0,1, 1,1, 1,-1 };
        float K[] = { 1,0, 0,1, 1,1, 1,-1 };

        std::vector<float> gpu_out(T, 0.0f);
        std::vector<float> cpu_out(T, 0.0f);

        cpu_attention_scores(Q, n, d, K, T, cpu_out.data());
        if (!llama_kv_compact_metal_attention_scores(ctx, Q, n, d, K, T, gpu_out.data())) {
            std::fprintf(stderr, "FAIL: small attention scores GPU dispatch\n");
            rc = 1;
        }

        float cos = cosine_sim(cpu_out.data(), gpu_out.data(), T);
        float mad = max_abs_diff(cpu_out.data(), gpu_out.data(), T);
        std::printf("  small attn: cosine=%.6f max_diff=%.6f\n", cos, mad);
        if (cos < 0.999f) {
            std::fprintf(stderr, "FAIL: small attn cosine %.4f < 0.999\n", cos);
            rc = 1;
        }
    }

    // --- Test 2: Medium attention scores (256 queries, 128 keys, d=64) ---
    {
        const uint32_t n = 256, T = 128, d = 64;
        std::vector<float> Q(n * d), K(T * d);

        // Deterministic pseudo-random fill
        for (size_t i = 0; i < Q.size(); ++i) Q[i] = sinf((float)i * 0.1f) * 0.5f;
        for (size_t i = 0; i < K.size(); ++i) K[i] = cosf((float)i * 0.07f) * 0.5f;

        std::vector<float> gpu_out(T, 0.0f);
        std::vector<float> cpu_out(T, 0.0f);

        cpu_attention_scores(Q.data(), n, d, K.data(), T, cpu_out.data());
        if (!llama_kv_compact_metal_attention_scores(ctx, Q.data(), n, d, K.data(), T, gpu_out.data())) {
            std::fprintf(stderr, "FAIL: medium attention scores GPU dispatch\n");
            rc = 1;
        }

        float cos = cosine_sim(cpu_out.data(), gpu_out.data(), T);
        float mad = max_abs_diff(cpu_out.data(), gpu_out.data(), T);
        std::printf("  medium attn: cosine=%.6f max_diff=%.6f\n", cos, mad);
        if (cos < 0.9999f) {
            std::fprintf(stderr, "FAIL: medium attn cosine %.4f < 0.9999\n", cos);
            rc = 1;
        }
    }

    // --- Test 3: Production-scale attention scores (2000 queries, 2048 keys, d=128) ---
    {
        const uint32_t n = 2000, T = 2048, d = 128;
        std::vector<float> Q(n * d), K(T * d);

        for (size_t i = 0; i < Q.size(); ++i) Q[i] = sinf((float)i * 0.013f) * 0.3f;
        for (size_t i = 0; i < K.size(); ++i) K[i] = cosf((float)i * 0.017f) * 0.3f;

        std::vector<float> gpu_out(T, 0.0f);
        std::vector<float> cpu_out(T, 0.0f);

        cpu_attention_scores(Q.data(), n, d, K.data(), T, cpu_out.data());
        if (!llama_kv_compact_metal_attention_scores(ctx, Q.data(), n, d, K.data(), T, gpu_out.data())) {
            std::fprintf(stderr, "FAIL: large attention scores GPU dispatch\n");
            rc = 1;
        }

        float cos = cosine_sim(cpu_out.data(), gpu_out.data(), T);
        float mad = max_abs_diff(cpu_out.data(), gpu_out.data(), T);
        std::printf("  large attn: cosine=%.6f max_diff=%.6f\n", cos, mad);
        if (cos < 0.9999f) {
            std::fprintf(stderr, "FAIL: large attn cosine %.6f < 0.9999\n", cos);
            rc = 1;
        }
    }

    // --- Test 4: Small XᵀX (8×4 matrix) ---
    {
        const uint32_t n = 8, t = 4;
        float X[] = {
            1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1,
            1,1,0,0,  0,1,1,0,  0,0,1,1,  1,0,0,1,
        };

        std::vector<float> gpu_out(t * t, 0.0f);
        std::vector<float> cpu_out(t * t, 0.0f);

        cpu_xtx(X, n, t, cpu_out.data());
        if (!llama_kv_compact_metal_xtx(ctx, X, n, t, gpu_out.data())) {
            std::fprintf(stderr, "FAIL: small xtx GPU dispatch\n");
            rc = 1;
        }

        float mad = max_abs_diff(cpu_out.data(), gpu_out.data(), t * t);
        std::printf("  small xtx: max_diff=%.6f\n", mad);
        if (mad > 0.001f) {
            std::fprintf(stderr, "FAIL: small xtx max_diff %.6f > 0.001\n", mad);
            rc = 1;
        }
    }

    // --- Test 5: Medium XᵀX (512×64 matrix) ---
    {
        const uint32_t n = 512, t = 64;
        std::vector<float> X(n * t);
        for (size_t i = 0; i < X.size(); ++i) X[i] = sinf((float)i * 0.031f) * 0.4f;

        std::vector<float> gpu_out(t * t, 0.0f);
        std::vector<float> cpu_out(t * t, 0.0f);

        cpu_xtx(X.data(), n, t, cpu_out.data());
        if (!llama_kv_compact_metal_xtx(ctx, X.data(), n, t, gpu_out.data())) {
            std::fprintf(stderr, "FAIL: medium xtx GPU dispatch\n");
            rc = 1;
        }

        float cos = cosine_sim(cpu_out.data(), gpu_out.data(), t * t);
        float mad = max_abs_diff(cpu_out.data(), gpu_out.data(), t * t);
        std::printf("  medium xtx: cosine=%.6f max_diff=%.6f\n", cos, mad);
        if (cos < 0.9999f) {
            std::fprintf(stderr, "FAIL: medium xtx cosine %.6f < 0.9999\n", cos);
            rc = 1;
        }
    }

    llama_kv_compact_metal_free(ctx);

    std::printf("test-kv-compact-solver-metal: %s\n", rc == 0 ? "PASSED" : "FAILED");
    return rc;
}
