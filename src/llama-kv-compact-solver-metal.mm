// Metal GPU acceleration for KV compaction solver.
//
// Four compute kernels:
//   1. kernel_attn_score    — Q·Kᵀ/√d score matrix
//   2. kernel_softmax       — per-row softmax in-place
//   3. kernel_col_sum       — column reduction → per-key aggregated scores
//   4. kernel_xtx           — XᵀX symmetric matrix assembly
//
// Buffer strategy: shared storage mode (Apple Silicon unified memory).
// Input buffers allocated per-call; intermediate/output buffers cached in context.

#include "llama-kv-compact-solver-metal.h"
#include "llama-impl.h"

#if defined(__APPLE__)

#import <Metal/Metal.h>

// ---------------------------------------------------------------------------
// Embedded Metal shader source
// ---------------------------------------------------------------------------

static NSString * const metal_shader_source = @R"(
#include <metal_stdlib>
using namespace metal;

// Kernel 1: Attention score matrix.
// score[qi * T + ki] = dot(Q[qi], K[ki]) * inv_sqrt_d
kernel void kernel_attn_score(
        device const float * Q          [[buffer(0)]],
        device const float * K          [[buffer(1)]],
        device float       * scores     [[buffer(2)]],
        constant uint      & n          [[buffer(3)]],
        constant uint      & T          [[buffer(4)]],
        constant uint      & d          [[buffer(5)]],
        constant float     & inv_sqrt_d [[buffer(6)]],
        uint2 gid [[thread_position_in_grid]])
{
    const uint qi = gid.y;
    const uint ki = gid.x;
    if (qi >= n || ki >= T) return;

    device const float * q_row = Q + qi * d;
    device const float * k_row = K + ki * d;

    float sum = 0.0f;
    uint i = 0;
    for (; i + 3 < d; i += 4) {
        sum += q_row[i]   * k_row[i];
        sum += q_row[i+1] * k_row[i+1];
        sum += q_row[i+2] * k_row[i+2];
        sum += q_row[i+3] * k_row[i+3];
    }
    for (; i < d; ++i) {
        sum += q_row[i] * k_row[i];
    }
    scores[qi * T + ki] = sum * inv_sqrt_d;
}

// Kernel 2: Per-row softmax in-place.
// For each row qi: find max, exp(x - max), normalize by sum.
kernel void kernel_softmax(
        device float       * scores [[buffer(0)]],
        constant uint      & n      [[buffer(1)]],
        constant uint      & T      [[buffer(2)]],
        threadgroup float  * shared  [[threadgroup(0)]],
        uint qi    [[threadgroup_position_in_grid]],
        uint tid   [[thread_index_in_threadgroup]],
        uint tg_sz [[threads_per_threadgroup]])
{
    if (qi >= n) return;
    device float * row = scores + qi * T;

    // Phase 1: row max via parallel reduction.
    float local_max = -INFINITY;
    for (uint ki = tid; ki < T; ki += tg_sz) {
        local_max = max(local_max, row[ki]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_sz / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: exp(x - max) and row sum.
    float local_sum = 0.0f;
    for (uint ki = tid; ki < T; ki += tg_sz) {
        float e = exp(row[ki] - row_max);
        row[ki] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_sz / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / max(shared[0], 1e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: normalize.
    for (uint ki = tid; ki < T; ki += tg_sz) {
        row[ki] *= inv_sum;
    }
}

// Kernel 3: Column sum reduction.
// output[ki] = sum_qi scores[qi * T + ki]
kernel void kernel_col_sum(
        device const float * scores [[buffer(0)]],
        device float       * output [[buffer(1)]],
        constant uint      & n      [[buffer(2)]],
        constant uint      & T      [[buffer(3)]],
        threadgroup float  * shared  [[threadgroup(0)]],
        uint ki    [[threadgroup_position_in_grid]],
        uint tid   [[thread_index_in_threadgroup]],
        uint tg_sz [[threads_per_threadgroup]])
{
    if (ki >= T) return;

    float local_sum = 0.0f;
    for (uint qi = tid; qi < n; qi += tg_sz) {
        local_sum += scores[qi * T + ki];
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_sz / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        output[ki] = shared[0];
    }
}

// Kernel 4: XᵀX symmetric matrix.
// out[i * t + j] = sum_r X[r * t + i] * X[r * t + j]
kernel void kernel_xtx(
        device const float * X   [[buffer(0)]],
        device float       * out [[buffer(1)]],
        constant uint      & n   [[buffer(2)]],
        constant uint      & t   [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.y;
    const uint j = gid.x;
    if (i >= t || j > i) return;  // lower triangle + diagonal

    float sum = 0.0f;
    for (uint r = 0; r < n; ++r) {
        sum += X[r * t + i] * X[r * t + j];
    }
    out[i * t + j] = sum;
    out[j * t + i] = sum;
}
)";

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct llama_kv_compact_metal_ctx {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLComputePipelineState> ps_attn_score;
    id<MTLComputePipelineState> ps_softmax;
    id<MTLComputePipelineState> ps_col_sum;
    id<MTLComputePipelineState> ps_xtx;

    // Cached intermediate buffers (grown, never shrunk).
    id<MTLBuffer> buf_scores;
    size_t        scores_cap;
    id<MTLBuffer> buf_output;
    size_t        output_cap;
    id<MTLBuffer> buf_xtx_out;
    size_t        xtx_out_cap;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool ensure_buffer(id<MTLDevice> dev,
                          id<MTLBuffer> __strong & buf,
                          size_t & cap, size_t needed) {
    if (cap >= needed && buf) return true;
    buf = [dev newBufferWithLength:needed
                          options:MTLResourceStorageModeShared];
    if (!buf) return false;
    cap = needed;
    return true;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

llama_kv_compact_metal_ctx * llama_kv_compact_metal_create() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        LLAMA_LOG_WARN("metal solver: no Metal device available\n");
        return nullptr;
    }

    NSError * error = nil;
    MTLCompileOptions * compile_opts = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    compile_opts.fastMathEnabled = YES;
#pragma clang diagnostic pop

    id<MTLLibrary> lib = [device newLibraryWithSource:metal_shader_source
                                              options:compile_opts
                                                error:&error];
    if (!lib) {
        LLAMA_LOG_ERROR("metal solver: shader compilation failed: %s\n",
                        error.localizedDescription.UTF8String);
        return nullptr;
    }

    auto make_ps = [&](const char * name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
        if (!fn) {
            LLAMA_LOG_ERROR("metal solver: function '%s' not found\n", name);
            return nil;
        }
        id<MTLComputePipelineState> ps =
            [device newComputePipelineStateWithFunction:fn error:&error];
        if (!ps) {
            LLAMA_LOG_ERROR("metal solver: pipeline '%s' failed: %s\n",
                            name, error.localizedDescription.UTF8String);
        }
        return ps;
    };

    auto * ctx = new llama_kv_compact_metal_ctx();
    ctx->device = device;
    ctx->queue  = [device newCommandQueue];

    ctx->ps_attn_score = make_ps("kernel_attn_score");
    ctx->ps_softmax    = make_ps("kernel_softmax");
    ctx->ps_col_sum    = make_ps("kernel_col_sum");
    ctx->ps_xtx        = make_ps("kernel_xtx");

    if (!ctx->ps_attn_score || !ctx->ps_softmax ||
        !ctx->ps_col_sum   || !ctx->ps_xtx) {
        delete ctx;
        return nullptr;
    }

    ctx->buf_scores  = nil;
    ctx->scores_cap  = 0;
    ctx->buf_output  = nil;
    ctx->output_cap  = 0;
    ctx->buf_xtx_out = nil;
    ctx->xtx_out_cap = 0;

    LLAMA_LOG_INFO("metal solver: initialized on %s\n",
                   device.name.UTF8String);
    return ctx;
}

void llama_kv_compact_metal_free(llama_kv_compact_metal_ctx * ctx) {
    delete ctx;
}

bool llama_kv_compact_metal_attention_scores(
        llama_kv_compact_metal_ctx * ctx,
        const float * Q_data, uint32_t n, uint32_t d,
        const float * K_data, uint32_t T,
        float * scores_out) {

    if (!ctx || n == 0 || T == 0 || d == 0) return false;

    @autoreleasepool {
        // Input buffers (shared storage — zero-copy on Apple Silicon).
        // Try newBufferWithBytesNoCopy first (requires page-aligned data),
        // fall back to newBufferWithBytes if NoCopy returns nil.
        const size_t Q_bytes = (size_t)n * d * sizeof(float);
        const size_t K_bytes = (size_t)T * d * sizeof(float);

        id<MTLBuffer> buf_Q = [ctx->device
            newBufferWithBytesNoCopy:(void *)Q_data
                              length:Q_bytes
                             options:MTLResourceStorageModeShared
                         deallocator:nil];
        if (!buf_Q) {
            buf_Q = [ctx->device
                newBufferWithBytes:Q_data
                            length:Q_bytes
                           options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> buf_K = [ctx->device
            newBufferWithBytesNoCopy:(void *)K_data
                              length:K_bytes
                             options:MTLResourceStorageModeShared
                         deallocator:nil];
        if (!buf_K) {
            buf_K = [ctx->device
                newBufferWithBytes:K_data
                            length:K_bytes
                           options:MTLResourceStorageModeShared];
        }

        if (!buf_Q || !buf_K) {
            LLAMA_LOG_WARN("metal solver: failed to allocate Q/K input buffers\n");
            return false;
        }

        // Intermediate + output buffers (cached, grown as needed).
        if (!ensure_buffer(ctx->device, ctx->buf_scores, ctx->scores_cap,
                           (size_t)n * T * sizeof(float))) return false;
        if (!ensure_buffer(ctx->device, ctx->buf_output, ctx->output_cap,
                           (size_t)T * sizeof(float))) return false;

        float inv_sqrt_d = 1.0f / sqrtf((float)d);

        id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
        if (!cmdbuf) return false;

        // --- Kernel 1: Attention scores ---
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->ps_attn_score];
            [enc setBuffer:buf_Q          offset:0 atIndex:0];
            [enc setBuffer:buf_K          offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_scores offset:0 atIndex:2];
            [enc setBytes:&n          length:sizeof(n)          atIndex:3];
            [enc setBytes:&T          length:sizeof(T)          atIndex:4];
            [enc setBytes:&d          length:sizeof(d)          atIndex:5];
            [enc setBytes:&inv_sqrt_d length:sizeof(inv_sqrt_d) atIndex:6];
            [enc dispatchThreads:MTLSizeMake(T, n, 1)
               threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
        }

        // --- Kernel 2: Per-row softmax ---
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->ps_softmax];
            [enc setBuffer:ctx->buf_scores offset:0 atIndex:0];
            [enc setBytes:&n length:sizeof(n) atIndex:1];
            [enc setBytes:&T length:sizeof(T) atIndex:2];
            // F-C-18: Threadgroup size must be power-of-2 for the parallel reduction tree in the softmax kernel.
            constexpr uint32_t tg_size = 256;
            static_assert((tg_size & (tg_size - 1)) == 0, "tg_size must be power of 2");
            [enc setThreadgroupMemoryLength:tg_size * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
            [enc endEncoding];
        }

        // --- Kernel 3: Column sum ---
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->ps_col_sum];
            [enc setBuffer:ctx->buf_scores offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_output offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(n) atIndex:2];
            [enc setBytes:&T length:sizeof(T) atIndex:3];
            // F-C-18: Threadgroup size must be power-of-2 for the parallel reduction tree.
            constexpr uint32_t tg_size = 256;
            static_assert((tg_size & (tg_size - 1)) == 0, "tg_size must be power of 2");
            [enc setThreadgroupMemoryLength:tg_size * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
            [enc endEncoding];
        }

        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        if (cmdbuf.status == MTLCommandBufferStatusError) {
            LLAMA_LOG_ERROR("metal solver: attention scores failed: %s\n",
                            cmdbuf.error.localizedDescription.UTF8String);
            return false;
        }

        memcpy(scores_out, ctx->buf_output.contents, (size_t)T * sizeof(float));
    }

    return true;
}

bool llama_kv_compact_metal_xtx(
        llama_kv_compact_metal_ctx * ctx,
        const float * X_data, uint32_t n, uint32_t t,
        float * xtx_out) {

    if (!ctx || n == 0 || t == 0) return false;

    @autoreleasepool {
        id<MTLBuffer> buf_X = [ctx->device
            newBufferWithBytes:X_data
                        length:(size_t)n * t * sizeof(float)
                       options:MTLResourceStorageModeShared];
        if (!buf_X) return false;

        const size_t out_bytes = (size_t)t * t * sizeof(float);
        if (!ensure_buffer(ctx->device, ctx->buf_xtx_out, ctx->xtx_out_cap,
                           out_bytes)) return false;

        id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
        if (!cmdbuf) return false;

        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->ps_xtx];
            [enc setBuffer:buf_X           offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_xtx_out offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(n) atIndex:2];
            [enc setBytes:&t length:sizeof(t) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(t, t, 1)
               threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
        }

        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        if (cmdbuf.status == MTLCommandBufferStatusError) {
            LLAMA_LOG_ERROR("metal solver: xtx failed: %s\n",
                            cmdbuf.error.localizedDescription.UTF8String);
            return false;
        }

        memcpy(xtx_out, ctx->buf_xtx_out.contents, out_bytes);
    }

    return true;
}

#endif // __APPLE__
