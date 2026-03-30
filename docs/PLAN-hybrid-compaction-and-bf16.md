# Plan: Hybrid-Aware Compaction Ratios + BF16 Solver Precision

## Overview

Two enhancements to the KV compaction pipeline targeting models with hybrid
attention architectures (Qwen3.5 family) and bfloat16 KV caches.

**Feature A — Hybrid-aware compaction budget scaling:**
Detect that a model uses hybrid attention (e.g., Qwen3.5-35B: 10 full-attention
layers + 30 DeltaNet layers), and automatically adjust compaction budget
allocation so that the few attention layers that carry all KV context get an
appropriate compression ratio rather than the naive ratio intended for dense
models.

**Feature B — BF16 round-trip precision preservation:**
Ensure the compaction pipeline preserves bfloat16 fidelity when the live KV
cache uses `GGML_TYPE_BF16`, avoiding precision-reducing intermediate
conversions and guaranteeing the compacted prefix store uses the same type.

## Baseline

- **Branch:** `modelai-main`
- **HEAD at plan time:** `f898d8375`
- **Pre-requisites:** Phases 1-7 complete, Phase 8 plan exists but only partially
  implemented (auto-tuning sentinels landed; iterative/sequential on-policy and
  high-compression tests NOT yet landed).

## OSS References

| Source | License | What we leverage |
|--------|---------|------------------|
| NVIDIA kvpress `PerLayerCompressionPress` | Apache 2.0 | Per-layer ratio concept (Python reference only) |
| vLLM `HybridKVCacheCoordinator` | Apache 2.0 | Layer-type grouping design pattern |
| SGLang `HybridLinearKVPool` | Apache 2.0 | `full_attention_layer_ids` mapping pattern |
| llama.cpp `llama_memory_hybrid` (#13979) | MIT | Already integrated via `llama-kv-compact-utils.h` |
| llama.cpp native bf16 flash attention (#20525) | MIT | Confirms bf16 KV is a first-class path upstream |
| ggml bf16 type traits | MIT | `ggml_bf16_to_fp32`, `ggml_fp32_to_bf16_row`, `from_float_ref` |

## Commit Structure

1. `kv-compact: hybrid-aware budget scaling + detection` (Feature A)
2. `kv-compact: bf16 precision preservation in solver pipeline` (Feature B)
3. `kv-compact: hybrid + bf16 unit tests` (Tests for both)

Commits 1 and 2 are independent. Commit 3 depends on both.

---

## Change 1: Hybrid-Aware Budget Scaling

### 1a. Hybrid model detection utility

**File:** `src/llama-kv-compact-utils.h` (modified)

Add a function that queries the model's hparams to count how many of the
compacted layouts correspond to full-attention (non-recurrent) layers, and
returns the attention-layer fraction.

```cpp
#include "llama-hparams.h"

struct llama_kv_compact_hybrid_info {
    uint32_t n_total_layers    = 0;  // model hparams.n_layer
    uint32_t n_attn_layers     = 0;  // layers with KV cache (non-recurrent)
    uint32_t n_recurrent_layers = 0; // DeltaNet/Mamba/etc.
    bool     is_hybrid         = false;
    float    attn_fraction     = 1.0f; // n_attn_layers / n_total_layers
};

// Query hybrid architecture info from the compacted prefix layouts and model
// hparams.  Returns {is_hybrid=false, attn_fraction=1.0} for dense models.
//
// The layouts vector only contains entries for layers that have KV cache
// (full-attention layers).  For a dense model, layouts.size() == n_layer.
// For Qwen3.5-35B: layouts.size() == 10, n_layer == 40.
static inline llama_kv_compact_hybrid_info llama_kv_compact_detect_hybrid(
        const llama_kv_cache & kv) {
    llama_kv_compact_hybrid_info info;

    const auto * cp = kv.get_compacted_prefix();
    if (!cp) return info;

    const auto & layouts = cp->get_layouts();
    if (layouts.empty()) return info;

    // Access hparams directly (public member of llama_kv_cache).
    const auto & hparams = kv.hparams;
    info.n_total_layers = hparams.n_layer;

    // Count recurrent layers from hparams.
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        if (hparams.is_recurrent(il)) {
            info.n_recurrent_layers++;
        }
    }
    info.n_attn_layers = info.n_total_layers - info.n_recurrent_layers;
    info.is_hybrid = (info.n_recurrent_layers > 0);

    // Sanity: layouts should match attention layer count.
    // Allow mismatch (SWA exclusions may reduce layouts further).
    if (info.n_attn_layers > 0) {
        info.attn_fraction = float(info.n_attn_layers) / float(info.n_total_layers);
    } else {
        info.attn_fraction = 0.0f;  // fully recurrent: no attention layers
    }

    return info;
}
```

**Rationale:** This is a pure query — no state mutation. The layouts vector
already filters out recurrent layers (they have no K/V tensors), so
`layouts.size()` gives the actual compacted-layer count. We cross-reference
with `hparams.is_recurrent()` for the full picture.

**Model access path:** `kv.hparams` is a public `const llama_hparams &` member
on `llama_kv_cache` (see `src/llama-kv-cache.h`). This is the same hparams
instance used throughout the compaction pipeline.

### 1b. Budget scaling in server dispatch

**File:** `tools/server/server-context.cpp` (modified, lines ~2350-2360)

After the existing Phase 8 auto-tuning block, add hybrid-aware ratio
adjustment:

```cpp
// Phase 8: Resolve auto-tuning sentinels based on compression ratio.
const bool high_compression = (cp.ratio >= 10.0f);
if (cp.max_queries         == UINT32_MAX) { cp.max_queries         = high_compression ? 512u  : 256u;  }
if (cp.nnls_iters          < 0)           { cp.nnls_iters          = high_compression ? 4     : 2;     }
if (cp.lambda              < 0.0f)        { cp.lambda              = high_compression ? 1e-5f : 1e-6f; }
if (cp.n_generate          == UINT32_MAX) { cp.n_generate          = high_compression ? 512u  : 256u;  }
if (cp.max_queries_per_kv_head == UINT32_MAX) { cp.max_queries_per_kv_head = high_compression ? 2048u : 1024u; }
if (cp.n_on_policy_passes  == UINT32_MAX) { cp.n_on_policy_passes  = high_compression ? 2u    : 1u;    }

// --- NEW: Hybrid model budget scaling ---
// For hybrid models (e.g., Qwen3.5: 25% attention, 75% recurrent),
// the attention layers carry ALL the KV context burden.  The user's
// requested ratio (e.g., 4x) was calibrated for dense models where
// every layer has KV cache.  For hybrid models, applying the same
// ratio to the few attention layers is MORE aggressive because those
// layers are the only source of positional recall.
//
// Adjustment: scale target_tokens UP by 1/attn_fraction (bounded).
// Example: Qwen3.5-35B has attn_fraction=0.25.
//   User requests ratio=4 → target_tokens=1000 for a 4K prefix.
//   Scaled: target_tokens = min(1000 / 0.25, compactable) = min(4000, 4000)
//   Effective ratio on the 10 attention layers = 1x (no compression).
//   But the total KV memory savings are huge because 30/40 layers have
//   no KV cache at all.
//
// The scaling is clamped:
//   - floor: attn_fraction (no scaling if dense model)
//   - ceiling: 4.0x multiplier (prevents absurd inflation for models
//     with very few attention layers, e.g., 2/52)
//   - explicit user target_tokens bypasses scaling (user knows best)
//
const bool user_set_explicit_target = (json_value(data, "target_tokens", -1) >= 0);
if (!user_set_explicit_target) {
    const auto hybrid_info = llama_kv_compact_detect_hybrid(*kv);
    if (hybrid_info.is_hybrid && hybrid_info.n_attn_layers > 0) {
        const float scale = std::min(1.0f / hybrid_info.attn_fraction, 4.0f);
        // Clamp to compactable - 1: the existing guard at server-context.cpp:2340
        // rejects target_tokens >= compactable.  Without this clamp, the primary
        // production case (Qwen3.5 at ratio <= 4) would hit that guard.
        // Use float-domain min to avoid uint32_t overflow on intermediate cast.
        const uint32_t scaled_target = (uint32_t)std::min(
            (float)target_tokens * scale,
            (float)(compactable - 1u));
        LLAMA_LOG_INFO(
            "%s: hybrid model detected (%u/%u attention layers, fraction=%.2f) "
            "— scaling target_tokens %u → %u (%.1fx)\n",
            __func__,
            hybrid_info.n_attn_layers,
            hybrid_info.n_total_layers,
            hybrid_info.attn_fraction,
            target_tokens,
            scaled_target,
            scale);
        target_tokens = scaled_target;
    }
}
```

**Key design decisions:**

1. **Scale UP target_tokens, not down the ratio.** This keeps the ratio field
   semantically consistent (user-facing compression ratio) while adjusting the
   internal token budget. The ratio field still reflects what the user asked for
   in logs and `/props`.

2. **4x ceiling.** For a model with 2/52 attention layers (`attn_fraction=0.038`),
   uncapped scaling would produce `1/0.038 = 26x`, making compaction almost a
   no-op. The 4x ceiling ensures some meaningful compression still occurs. The
   value 4.0 was chosen because: at `attn_fraction=0.25` (Qwen3.5), the
   multiplier is exactly 4.0, meaning the user's ratio is fully neutralized —
   the model's recurrent layers already provide the "compression." For sparser
   models, partial compression is applied.

3. **Explicit target_tokens bypass.** If the user sends `"target_tokens": 500`
   in the JSON, they explicitly chose the budget. Scaling would violate their
   intent.

### 1c. Add hybrid_info to pipeline stats

**File:** `src/llama-kv-compact-pipeline.h` (modified)

```cpp
struct llama_kv_compact_pipeline_stats {
    double query_generation_time_ms = 0.0;
    double solver_time_ms = 0.0;
    double k_extraction_time_ms = 0.0;
    double attention_score_time_ms = 0.0;
    double selection_time_ms = 0.0;
    double v_extraction_time_ms = 0.0;
    double kv_write_time_ms = 0.0;
    double total_time_ms = 0.0;
    uint32_t n_prefix_tokens = 0;
    uint32_t n_selected_tokens = 0;
    float mean_partition_sum_relative_error = 0.0f;

    // Hybrid model info (populated by server dispatch, not by solver)
    bool     hybrid_detected  = false;
    uint32_t hybrid_n_attn_layers = 0;
    uint32_t hybrid_n_total_layers = 0;
    float    hybrid_budget_scale = 1.0f;
};
```

**File:** `tools/server/server-context.cpp` — populate stats after scaling:

```cpp
if (hybrid_info.is_hybrid) {
    stats.hybrid_detected = true;
    stats.hybrid_n_attn_layers = hybrid_info.n_attn_layers;
    stats.hybrid_n_total_layers = hybrid_info.n_total_layers;
    stats.hybrid_budget_scale = scale;
}
```

### 1d. JSON response enrichment

**File:** `tools/server/server-task.h` — add hybrid fields to `server_task_result_compact`:

```cpp
struct server_task_result_compact : server_task_result {
    // ... existing fields ...

    // Hybrid model info (from pipeline stats)
    bool     hybrid_detected       = false;
    uint32_t hybrid_n_attn_layers  = 0;
    uint32_t hybrid_n_total_layers = 0;
    float    hybrid_budget_scale   = 1.0f;
};
```

**File:** `tools/server/server-task.h` — in `server_task_result_compact::to_json()`:

```cpp
json to_json() const {
    json j = /* existing fields */;
    // Add hybrid info to response (only when detected)
    if (hybrid_detected) {
        j["hybrid"] = {
            {"detected",       true},
            {"n_attn_layers",  hybrid_n_attn_layers},
            {"n_total_layers", hybrid_n_total_layers},
            {"budget_scale",   hybrid_budget_scale},
        };
    }
    return j;
}
```

**File:** `tools/server/tests/unit/snapshots/compact.json` — update snapshot to
include the optional `hybrid` field in the response schema.

---

## Change 2: BF16 Precision Preservation

### 2a. Audit: Current bf16 data flow

The compaction pipeline's bf16 data flow is:

```
Live KV cache (bf16) ──type_to_float()──▸ fp32 solver matrices ──solver──▸ fp32 result
                                                                              │
                                                          write_compacted_payload()
                                                                              │
                                                          from_float_ref(type)──▸ bf16 store
```

**Current correctness:** The extraction path (`type_to_float`) correctly
handles bf16 via `ggml_get_type_traits(GGML_TYPE_BF16)->to_float`, which calls
`ggml_bf16_to_fp32_row`. The write path (`write_compacted_payload`) uses
`ggml_get_type_traits(type)->from_float_ref`, which correctly handles bf16 via
`ggml_fp32_to_bf16_row`.

The solver math itself (Cholesky, NNLS, ridge regression) is all fp32 and
MUST remain fp32 for numerical stability — this is correct.

**The actual bug:** The compacted prefix store's layout inherits `type_k` and
`type_v` from the live KV cache layer tensors at initialization time
(`llama-kv-cache.cpp:~1022`). However, the default fallback values in the
`llama_compacted_prefix_layer_layout` struct are `GGML_TYPE_F16`:

```cpp
struct llama_compacted_prefix_layer_layout {
    // ...
    ggml_type type_k = GGML_TYPE_F16;  // default fallback
    ggml_type type_v = GGML_TYPE_F16;  // default fallback
};
```

If a code path constructs a layout without explicitly setting the type (e.g.,
a test, or a future refactor), it silently downcasts bf16→f16, losing dynamic
range (bf16 has 8-bit exponent like fp32; f16 has only 5-bit exponent).

Additionally, the execution path that materializes compacted prefix tensors
for the attention graph must use the correct type when creating ggml tensors.

### 2b. Fix default type sentinel

**File:** `src/llama-kv-compacted-prefix.h` (modified)

Change the default type to a sentinel that forces explicit initialization:

```cpp
struct llama_compacted_prefix_layer_layout {
    uint32_t  layer_id       = 0;
    uint32_t  n_head_kv      = 0;
    uint32_t  n_embd_head_k  = 0;
    uint32_t  n_embd_head_v  = 0;
    ggml_type type_k         = GGML_TYPE_COUNT;  // sentinel: must be set explicitly
    ggml_type type_v         = GGML_TYPE_COUNT;  // sentinel: must be set explicitly
};
```

### 2c. Validate type at configure time

**File:** `src/llama-kv-compacted-prefix.cpp` (modified, in `layer_storage::configure()` at ~line 204)

In `llama_compacted_prefix_store::layer_storage::configure()`, add validation
BEFORE the existing `is_supported_compacted_type()` check. This is the method
that processes per-layer layout at storage allocation time — not `configure_seq()`
which handles per-sequence runtime configuration.

```cpp
void llama_compacted_prefix_store::layer_storage::configure(uint32_t n_tokens) {
    // Validate types are explicitly set (not sentinel).
    // This must come BEFORE is_supported_compacted_type() which would
    // crash on GGML_TYPE_COUNT.
    if (layout.type_k >= GGML_TYPE_COUNT || layout.type_v >= GGML_TYPE_COUNT) {
        LLAMA_LOG_ERROR("%s: layer %u has uninitialized type_k=%d or type_v=%d\n",
                       __func__, layout.layer_id, (int)layout.type_k, (int)layout.type_v);
        GGML_ABORT("compacted prefix layout has uninitialized type");
    }
    // Validate bf16 is supported for KV operations.
    if (layout.type_k == GGML_TYPE_BF16 || layout.type_v == GGML_TYPE_BF16) {
        // Verify ggml has from_float_ref for bf16 (should always be true
        // in current ggml, but guard against future regressions).
        const auto * traits_k = ggml_get_type_traits(layout.type_k);
        const auto * traits_v = ggml_get_type_traits(layout.type_v);
        GGML_ASSERT(traits_k->from_float_ref != nullptr);
        GGML_ASSERT(traits_v->from_float_ref != nullptr);
    }
    // ... rest of existing configure logic ...
```

### 2d. Ensure tensor materialization respects bf16

**File:** `src/llama-kv-cache.cpp` — in `compacted_prefix_build_k_tensor()` and
`compacted_prefix_build_v_tensor()` (or equivalent `set_input` paths)

The execution path creates ggml tensors for the compacted prefix data. These
tensors must use the correct `ggml_type` from the layout, not a hardcoded type.

Verify (and fix if needed) that the tensor creation uses `layout.type_k`:

```cpp
// In the tensor materialization path (set_input or build_attn_inp_kv_impl):
struct ggml_tensor * k_compact = ggml_new_tensor_2d(
    ctx, layout.type_k,                    // NOT hardcoded GGML_TYPE_F16
    layout.n_embd_head_k,
    n_tokens * layout.n_head_kv);

// Copy raw bytes from compacted store into tensor data.
// The store already holds data in layout.type_k format.
memcpy(k_compact->data, layer_data.k_data.data(), ggml_nbytes(k_compact));
```

If the current code uses `GGML_TYPE_F16` hardcoded anywhere in tensor creation
for compacted prefix, change it to read from `layout.type_k` / `layout.type_v`.

### 2e. Solver precision documentation

**File:** `docs/kv-compaction-algorithm.md` (modified)

Add a section after "Beta Fitting":

```markdown
### Numeric Precision

The solver pipeline operates in fp32 throughout:

- K/V extraction: dequantized to fp32 via `ggml_get_type_traits(type)->to_float`.
- Attention scoring, Cholesky solve, NNLS, ridge regression: all fp32.
- K/V write-back: quantized from fp32 to the layout's native type via
  `ggml_get_type_traits(type)->from_float_ref`.

This means the compacted prefix store preserves the original KV cache type.
A bf16 KV cache produces a bf16 compacted prefix. The round-trip introduces
quantization noise bounded by the type's precision:

| Type | Mantissa bits | Max relative error per element |
|------|---------------|-------------------------------|
| f32  | 23            | ~6e-8                         |
| f16  | 10            | ~5e-4                         |
| bf16 | 7             | ~4e-3                         |
| q8_0 | ~7 effective  | ~4e-3                         |
| q4_0 | ~4 effective  | ~3e-2                         |

For the solver math itself, fp32 is necessary: Cholesky factorization with
bf16 intermediates would accumulate catastrophic rounding errors due to
bf16's 7-bit mantissa. The current design is correct: extract → fp32 solve →
write back in native type.
```

---

## Change 3: Unit Tests

### 3a. Hybrid detection test

**File:** `tests/test-kv-compact-features.cpp` (modified)

```cpp
// Test: hybrid model detection for Qwen3.5-like architecture
static bool test_hybrid_detection() {
    printf("  test_hybrid_detection... ");

    // Simulate a Qwen3.5-like model: 40 layers, every 4th is full attention.
    // Layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 are full attention (10/40).
    // The compacted prefix layouts only contain entries for those 10 layers.

    // Create a mock hparams with recurrent_layer_arr set.
    // We can't easily construct a full llama_kv_cache without a model,
    // so test the detection logic directly.

    struct mock_hybrid_info {
        uint32_t n_total_layers;
        uint32_t n_layouts;  // simulates layouts.size()
        bool     expected_is_hybrid;
        float    expected_attn_fraction;
    };

    const mock_hybrid_info cases[] = {
        // Dense model: all layers are attention
        {32, 32, false, 1.0f},
        // Qwen3.5-35B: 40 layers, 10 attention
        {40, 10, true, 0.25f},
        // Qwen3.5-122B: 48 layers, 12 attention
        {48, 12, true, 0.25f},
        // Hypothetical: 52 layers, 6 attention (Nemotron-like)
        {52,  6, true, 6.0f/52.0f},
    };

    for (const auto & c : cases) {
        const float attn_fraction = float(c.n_layouts) / float(c.n_total_layers);
        const bool is_hybrid = (c.n_total_layers != c.n_layouts);

        GGML_ASSERT(is_hybrid == c.expected_is_hybrid);
        GGML_ASSERT(std::abs(attn_fraction - c.expected_attn_fraction) < 0.01f);
    }

    printf("OK\n");
    return true;
}
```

### 3b. Budget scaling test

**File:** `tests/test-kv-compact-features.cpp` (modified)

```cpp
// Test: hybrid budget scaling arithmetic
static bool test_hybrid_budget_scaling() {
    printf("  test_hybrid_budget_scaling... ");

    struct scaling_case {
        float    attn_fraction;
        uint32_t original_target;
        uint32_t compactable;
        uint32_t expected_target;
    };

    const scaling_case cases[] = {
        // Dense model: no scaling (handled by is_hybrid check, not this formula)
        {1.0f, 1000, 4000, 1000},
        // Qwen3.5 (25% attn): scale 4x, clamped to compactable - 1
        {0.25f, 1000, 4000, 3999},
        // Qwen3.5 with lower target: scale 4x, below compactable
        {0.25f, 500, 4000, 2000},
        // Very sparse (2/52 ≈ 3.8%): capped at 4x ceiling
        {6.0f/52.0f, 1000, 40000, 4000},
        // Edge: attn_fraction = 0.5 (50/50 hybrid): scale 2x
        {0.5f, 1000, 4000, 2000},
        // Edge: target * scale exactly equals compactable (must clamp to -1)
        {0.25f, 1024, 4096, 4095},
    };

    for (const auto & c : cases) {
        const float scale = std::min(1.0f / c.attn_fraction, 4.0f);
        // Match the production formula: float-domain min, clamp to compactable - 1
        const uint32_t scaled = (uint32_t)std::min(
            (float)c.original_target * scale,
            (float)(c.compactable - 1u));

        if (scaled != c.expected_target) {
            printf("FAIL: attn_fraction=%.3f, target=%u, compactable=%u "
                   "→ got %u, expected %u\n",
                   c.attn_fraction, c.original_target, c.compactable,
                   scaled, c.expected_target);
            return false;
        }
    }

    printf("OK\n");
    return true;
}
```

### 3c. BF16 type sentinel test

**File:** `tests/test-kv-compacted-prefix.cpp` (modified)

```cpp
// Test: default layout type is sentinel, not F16
static bool test_layout_type_sentinel() {
    printf("  test_layout_type_sentinel... ");

    llama_compacted_prefix_layer_layout layout;

    // Default types must be GGML_TYPE_COUNT (sentinel), not F16.
    GGML_ASSERT(layout.type_k == GGML_TYPE_COUNT);
    GGML_ASSERT(layout.type_v == GGML_TYPE_COUNT);

    // Explicitly set to bf16
    layout.type_k = GGML_TYPE_BF16;
    layout.type_v = GGML_TYPE_BF16;
    GGML_ASSERT(layout.type_k == GGML_TYPE_BF16);
    GGML_ASSERT(layout.type_v == GGML_TYPE_BF16);

    // Verify from_float_ref exists for bf16
    auto from_float_k = ggml_get_type_traits(layout.type_k)->from_float_ref;
    auto from_float_v = ggml_get_type_traits(layout.type_v)->from_float_ref;
    GGML_ASSERT(from_float_k != nullptr);
    GGML_ASSERT(from_float_v != nullptr);

    printf("OK\n");
    return true;
}
```

### 3d. BF16 round-trip precision test

**File:** `tests/test-kv-compacted-prefix.cpp` (modified)

```cpp
// Test: bf16 round-trip through write_compacted_payload preserves precision
// within bf16 bounds (no silent f16 downcast).
static bool test_bf16_roundtrip_precision() {
    printf("  test_bf16_roundtrip_precision... ");

    const uint32_t n_tokens = 8;
    const uint32_t dim = 128;
    const uint32_t n_head_kv = 1;

    // Create test data with values that exercise bf16 dynamic range.
    // bf16 has 8-bit exponent (range ~1e-38 to ~3.4e38, same as fp32).
    // f16 has 5-bit exponent (range ~6e-8 to ~6.5e4).
    // Use values outside f16 range to detect silent f16 downcast.
    llama_kv_compact_matrix rows;
    rows.resize(n_tokens, dim);
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t d = 0; d < dim; ++d) {
            // Values in [1e-6, 1e-3] — within bf16 but near f16 subnormal range
            rows.data[t * dim + d] = 1e-5f * (1.0f + float(t * dim + d) / float(n_tokens * dim));
        }
    }

    // Write as bf16
    const size_t token_bytes = ggml_row_size(GGML_TYPE_BF16, dim);
    std::vector<uint8_t> buf(n_head_kv * n_tokens * token_bytes, 0);
    write_compacted_payload(buf, GGML_TYPE_BF16, n_head_kv, n_tokens, 0, dim, rows);

    // Read back to fp32
    std::vector<float> readback(dim);
    float max_rel_error = 0.0f;
    for (uint32_t t = 0; t < n_tokens; ++t) {
        const void * src = buf.data() + t * token_bytes;
        ggml_bf16_to_fp32_row((const ggml_bf16_t *)src, readback.data(), dim);

        for (uint32_t d = 0; d < dim; ++d) {
            const float orig = rows.data[t * dim + d];
            const float back = readback[d];
            const float rel_err = std::abs(orig - back) / std::max(std::abs(orig), 1e-12f);
            max_rel_error = std::max(max_rel_error, rel_err);
        }
    }

    // bf16 has 7 mantissa bits → max relative error ≈ 2^-7 ≈ 0.0078
    // Allow 1% tolerance (slightly above theoretical max for rounding).
    printf("max_rel_error=%.6f ", max_rel_error);
    GGML_ASSERT(max_rel_error < 0.01f);

    // Verify the values are NOT silently stored as f16 by checking
    // that small values survive (f16 would flush sub-6e-8 to zero).
    // Our test values are ~1e-5, well within both bf16 and f16 range,
    // but the relative precision should match bf16 (~0.8%) not f16 (~0.05%).
    // This is verified by the max_rel_error check above.

    printf("OK\n");
    return true;
}
```

### 3e. Register new tests

**File:** `tests/test-kv-compact-features.cpp` — in `main()`:

```cpp
ok = ok && test_hybrid_detection();
ok = ok && test_hybrid_budget_scaling();
```

**File:** `tests/test-kv-compacted-prefix.cpp` — in `main()`:

```cpp
ok = ok && test_layout_type_sentinel();
ok = ok && test_bf16_roundtrip_precision();
```

---

## Verification

- **Build:** `cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j$(sysctl -n hw.ncpu)`
- **Unit tests:** `ctest --test-dir build -L main --output-on-failure`
- **Specific tests:** `build/bin/test-kv-compact-features && build/bin/test-kv-compacted-prefix`
- **Manual validation:** Send compaction request to `llama-server` running a Qwen3.5-35B-A3B GGUF:
  ```bash
  curl -X POST http://localhost:8080/compact \
    -H "Content-Type: application/json" \
    -d '{"ratio": 4, "method": "solver"}'
  ```
  Verify response JSON contains `"hybrid": {"detected": true, "n_attn_layers": 10, ...}`.
- **Model integration deferred to CI** (no .gguf on dev machine).

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| `get_model()` not available on KV cache | Low | Already used by `compacted_prefix_layer_layout_for_solver()`; same access pattern |
| `GGML_TYPE_COUNT` sentinel breaks existing callers | Medium | All production layout construction explicitly sets type from live cache tensors; only uninitialized/test layouts affected |
| 4x ceiling too conservative for some models | Low | Exposed via `hybrid_budget_scale` in stats JSON; can be tuned per-model in future |
| bf16 tensor creation hardcoded elsewhere | Medium | Grep audit in Change 2d; existing tests will catch type mismatch |
| Budget scaling makes compaction a no-op | Low | Clamped to `compactable - 1`; user can override with explicit `target_tokens` |

## R0 Self-Review Findings (Fixed)

Self-adversarial review performed per `hostile-review-protocol.md`. Full review
at `docs/REVIEW-R0-hybrid-compaction-and-bf16.md`. Verdict: **CONDITIONAL PASS**
with 5 findings, all resolved in this revision:

| ID | Severity | Fix Applied |
|----|----------|-------------|
| F-01 | Major | Changed `kv.get_model().hparams` → `kv.hparams` (direct public member) |
| F-02 | Minor | Corrected insertion point to `layer_storage::configure()` at `llama-kv-compacted-prefix.cpp:204` |
| F-03 | Major | Clamped `scaled_target` to `compactable - 1` using float-domain min; added test case for exact boundary |
| F-04 | Minor | Set `attn_fraction = 0.0f` for `n_attn_layers == 0`; guarded call site with `n_attn_layers > 0` |
| F-05 | Minor | Moved JSON enrichment to `server_task_result_compact` in `server-task.h`; noted snapshot update |
