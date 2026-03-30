# Plan: Hybrid Budget Resolution + BF16 Type Safety

## Overview

This plan hardens two compaction-adjacent areas that are currently underspecified:

1. **Hybrid budget resolution for ratio-based compaction**
   - Detect hybrid architectures such as Qwen3.5.
   - Resolve an **effective** compaction target from the user-requested ratio in one shared helper used by both the server path and the public C API.
   - Make the server response explicit about **requested vs effective** budgets so the feature does not silently drift the `/compact` contract.
   - Short-circuit near-no-op hybrid requests instead of running the full solver/selection pipeline to remove a single token.

2. **BF16 type safety for compacted-prefix storage**
   - Make compacted-prefix layout types fail-closed if not initialized.
   - Preserve the existing exception-based validation contract rather than aborting the process.
   - Prove BF16 payloads are not silently downcast to F16.

This revision incorporates all blocking findings from R0, R1, R2, and R3.

## Baseline

- Branch: `modelai-main`
- Baseline commit under review: `6d3c1ca93`
- Scope: docs + implementation plan only. No code is changed in this document.

## Non-Goals

- No solver algorithm changes.
- No per-model tuning table beyond the initial 4.0x cap.
- No changes to unsupported architecture policy.
- No new request fields in `/compact` JSON.
- No benchmark or website work.

## Design Constraints

1. The server path and the public C API must not diverge on hybrid budget behavior.
2. `ratio`-driven requests may be hybrid-adjusted, but that adjustment must be observable and documented.
3. Explicit `target_tokens` remains authoritative and bypasses hybrid scaling.
4. Near-no-op hybrid outcomes must not pay full compaction cost.
5. BF16 validation failures remain recoverable testable errors, not process aborts.
6. Hybrid behavior must be derived from live architecture metadata (`hparams` + compacted layouts), not hardcoded model-name rules.

## Commit Structure

1. `kv-compact: add shared hybrid budget resolution helper`
2. `kv-compact: use shared hybrid budget resolution in server and C api`
3. `kv-compact: harden compacted-prefix bf16 type validation`
4. `kv-compact: add hybrid and bf16 regression tests`
5. `kv-compact: update compact response schema and docs`

---

## Change 1: Shared Hybrid Detection + Budget Resolution

### 1a. Split hybrid detection into a pure helper and a thin KV wrapper

**File:** `src/llama-kv-compact-utils.h`

Add two layers:

1. A pure helper that is trivial to unit test.
2. A thin wrapper that reads the live KV cache and calls the pure helper.

```cpp
struct llama_kv_compact_hybrid_info {
    uint32_t n_total_layers       = 0;
    uint32_t n_recurrent_layers   = 0;
    uint32_t n_attn_layers        = 0;  // from hparams
    uint32_t n_compactable_layers = 0;  // from compacted-prefix layouts
    bool     is_hybrid            = false;
    bool     layout_count_mismatch = false;
    float    compactable_fraction = 1.0f; // n_compactable_layers / n_total_layers
};

static inline llama_kv_compact_hybrid_info llama_kv_compact_make_hybrid_info(
        uint32_t n_total_layers,
        uint32_t n_recurrent_layers,
        uint32_t n_compactable_layers) {
    llama_kv_compact_hybrid_info info;
    info.n_total_layers       = n_total_layers;
    info.n_recurrent_layers   = n_recurrent_layers;
    info.n_attn_layers        = n_total_layers >= n_recurrent_layers
                              ? (n_total_layers - n_recurrent_layers)
                              : 0;
    info.n_compactable_layers = n_compactable_layers;
    info.is_hybrid            = (info.n_recurrent_layers > 0);
    info.layout_count_mismatch = (info.n_compactable_layers != info.n_attn_layers);

    if (info.n_total_layers > 0) {
        info.compactable_fraction = float(info.n_compactable_layers) / float(info.n_total_layers);
    } else {
        info.compactable_fraction = 0.0f;
    }
    return info;
}

static inline llama_kv_compact_hybrid_info llama_kv_compact_detect_hybrid(
        const llama_kv_cache & kv) {
    const auto * cp = kv.get_compacted_prefix();
    if (!cp) {
        return {};
    }

    const auto & layouts = cp->get_layouts();
    const auto & hparams = kv.hparams;

    uint32_t n_recurrent_layers = 0;
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        if (hparams.is_recurrent(il)) {
            n_recurrent_layers++;
        }
    }

    auto info = llama_kv_compact_make_hybrid_info(
        hparams.n_layer,
        n_recurrent_layers,
        (uint32_t) layouts.size());

    if (info.layout_count_mismatch) {
        LLAMA_LOG_WARN(
            "%s: hybrid layer mismatch (attn=%u, compactable=%u, total=%u)\n",
            __func__,
            info.n_attn_layers,
            info.n_compactable_layers,
            info.n_total_layers);
    }

    return info;
}
```

**Why this shape**

- The pure helper is testable without constructing a full `llama_kv_cache`.
- The wrapper still exercises the real access path through `kv.hparams` and `get_compacted_prefix()`.
- `compactable_fraction` is based on `layouts.size()`, not just `n_attn_layers`, so hybrid+excluded-layer cases do not silently use the wrong numerator.
- A mismatch is logged rather than silently ignored.

### 1b. Add one shared budget-resolution helper used by both server and C API

**File:** `src/llama-kv-compact-utils.h`

Add a small result carrier for hybrid-aware budget resolution.

```cpp
struct llama_kv_compact_budget_resolution {
    uint32_t requested_target_tokens = 0;
    uint32_t effective_target_tokens = 0;
    double   requested_ratio         = 0.0;
    double   effective_ratio         = 0.0;
    bool     explicit_target         = false;
    bool     hybrid_detected         = false;
    bool     skipped_noop            = false;
    float    budget_scale            = 1.0f;
    llama_kv_compact_hybrid_info hybrid = {};
};

static inline llama_kv_compact_budget_resolution llama_kv_compact_resolve_budget(
        const llama_kv_cache & kv,
        uint32_t compactable,
        uint32_t requested_target_tokens,
        bool explicit_target,
        double requested_ratio) {
    llama_kv_compact_budget_resolution out;
    out.requested_target_tokens = requested_target_tokens;
    out.effective_target_tokens = requested_target_tokens;
    out.explicit_target         = explicit_target;
    out.requested_ratio         = requested_ratio;
    out.effective_ratio         = requested_ratio;

    if (compactable == 0 || requested_target_tokens == 0) {
        return out;
    }

    if (explicit_target) {
        // No ratio was requested. Do not synthesize one from compactable/target.
        out.requested_ratio = 0.0;
        out.effective_ratio = 0.0;
        return out;
    }

    out.hybrid = llama_kv_compact_detect_hybrid(kv);
    out.hybrid_detected = out.hybrid.is_hybrid && out.hybrid.n_compactable_layers > 0;
    if (!out.hybrid_detected || out.hybrid.compactable_fraction <= 0.0f) {
        return out;
    }

    const float scale = std::min(1.0f / out.hybrid.compactable_fraction, 4.0f);
    out.budget_scale = scale;

    const uint32_t scaled_target = (uint32_t) std::min(
        (float) requested_target_tokens * scale,
        (float) compactable);

    if (scaled_target >= compactable - 1u) {
        out.effective_target_tokens = compactable;
        out.effective_ratio = 1.0;
        out.skipped_noop = true;
        return out;
    }

    out.effective_target_tokens = scaled_target;
    out.effective_ratio = compactable > 0
        ? double(compactable) / double(out.effective_target_tokens)
        : 0.0;
    return out;
}
```

**Key contract decisions**

1. `ratio` is a **requested** budget, not a guaranteed final target on hybrid models.
2. The **effective** target is what actually runs.
3. The server response will expose both requested and effective values when hybrid adjustment happens.
4. Explicit `target_tokens` bypasses scaling, and the response must not synthesize a fake requested ratio on that path.
5. Ratio-driven requests use a shared minimum target floor of 2 tokens before hybrid adjustment.
6. Near-no-op hybrid outcomes short-circuit without running the pipeline.

This is the change that closes the prior hidden-contract bug: the adjustment is now deliberate and observable.

### 1c. Apply the shared helper in both entry points

**Files:**
- `tools/server/server-context.cpp`
- `src/llama-kv-compact-api.cpp`

#### Server path

Keep the existing ratio validation and base target computation. Then resolve the effective budget through the shared helper.

```cpp
uint32_t requested_target_tokens;
bool explicit_target = false;
if (cp.target_tokens > 0) {
    explicit_target = true;
    requested_target_tokens = (uint32_t) cp.target_tokens;
} else {
    if (cp.ratio < 1.0f) {
        // existing error path
    }
    requested_target_tokens = std::max(2u, (uint32_t) (compactable / cp.ratio));
}

const auto budget = llama_kv_compact_resolve_budget(
    *kv,
    compactable,
    requested_target_tokens,
    explicit_target,
    explicit_target ? 0.0 : cp.ratio);

if (!explicit_target && budget.skipped_noop) {
    auto res = std::make_unique<server_task_result_compact>();
    res->id                  = task.id;
    res->id_slot             = id_slot;
    res->method              = method;
    res->compacted_tokens    = compactable;
    res->original_tokens     = compactable;
    res->compression_ratio   = 1.0;
    res->compaction_time_ms  = 0.0;
    res->active_n_kv_before  = n_kv_before;
    res->active_n_kv_after   = n_kv_before;
    res->reclaimed           = false;
    // hybrid metadata copied below
    queue_results.send(std::move(res));
    break;
}

const uint32_t target_tokens = budget.effective_target_tokens;
if (target_tokens >= compactable) {
    // existing explicit-target / invalid-budget path remains an error
}
```

#### C API path

Apply the same helper before the existing `target_tokens >= compactable` guard.

```cpp
uint32_t requested_target_tokens;
bool explicit_target = false;
if (params.target_tokens > 0) {
    explicit_target = true;
    requested_target_tokens = (uint32_t) params.target_tokens;
} else {
    if (params.ratio < 1.0f) {
        // existing error path
    }
    requested_target_tokens = std::max(2u, (uint32_t) (compactable / params.ratio));
}

const auto budget = llama_kv_compact_resolve_budget(
    *kv,
    compactable,
    requested_target_tokens,
    explicit_target,
    explicit_target ? 0.0 : params.ratio);

if (!explicit_target && budget.skipped_noop) {
    LLAMA_LOG_INFO("%s: hybrid no-op (%u -> %u, scale=%.2f)\n",
                   __func__,
                   budget.requested_target_tokens,
                   budget.effective_target_tokens,
                   budget.budget_scale);
    return (int32_t) compactable;
}

const uint32_t target_tokens = budget.effective_target_tokens;
if (target_tokens >= compactable) {
    // existing no-op return path for invalid explicit targets remains
}
```

Implementation note:
- In both the no-op and real-compaction server paths, copy the resolved `budget`
  fields onto `server_task_result_compact` so the JSON response exposes the same
  requested/effective metadata regardless of whether the pipeline actually ran.

**Why this closes the blocker**

- There is now one hybrid budget rule, not separate server-only behavior.
- The C API and server path no longer silently diverge.

### 1d. Response contract: add requested vs effective hybrid fields directly to `server_task_result_compact`

**File:** `tools/server/server-task.h`

Do **not** push this through `llama_kv_compact_pipeline_stats`. That carrier is method-specific and not shared across all compaction methods.

Extend `server_task_result_compact` directly.

```cpp
struct server_task_result_compact : server_task_result {
    std::string method;
    uint32_t    compacted_tokens     = 0;
    uint32_t    original_tokens      = 0;
    double      compression_ratio    = 0.0;
    double      compaction_time_ms   = 0.0;
    uint32_t    active_n_kv_before   = 0;
    uint32_t    active_n_kv_after    = 0;
    bool        reclaimed            = false;

    bool        hybrid_detected          = false;
    bool        hybrid_skipped_noop      = false;
    uint32_t    hybrid_n_attn_layers     = 0;
    uint32_t    hybrid_n_compactable_layers = 0;
    uint32_t    hybrid_n_total_layers    = 0;
    float       hybrid_budget_scale      = 1.0f;
    uint32_t    requested_target_tokens  = 0;
    uint32_t    effective_target_tokens  = 0;
    double      requested_ratio          = 0.0;
    double      effective_ratio          = 0.0;

    json to_json() override {
        json j = {
            { "success",             true },
            { "id_slot",             id_slot },
            { "method",              method },
            { "compacted_tokens",    compacted_tokens },
            { "original_tokens",     original_tokens },
            { "compression_ratio",   compression_ratio },
            { "compaction_time_ms",  compaction_time_ms },
            { "active_n_kv_before",  active_n_kv_before },
            { "active_n_kv_after",   active_n_kv_after },
            { "reclaimed",           reclaimed },
        };

        if (hybrid_detected) {
            j["hybrid"] = {
                {"detected",               true},
                {"skipped_noop",           hybrid_skipped_noop},
                {"n_attn_layers",          hybrid_n_attn_layers},
                {"n_compactable_layers",   hybrid_n_compactable_layers},
                {"n_total_layers",         hybrid_n_total_layers},
                {"budget_scale",           hybrid_budget_scale},
                {"requested_target_tokens", requested_target_tokens},
                {"effective_target_tokens", effective_target_tokens},
                {"requested_ratio",         requested_ratio},
                {"effective_ratio",         effective_ratio},
            };
        }
        return j;
    }
};
```

**Why this closes the blocker**

- The metadata is attached directly where the response is built.
- No invalid `stats` plumbing.
- No signature drift: `json to_json() override`, not `const`.

### 1e. Update public API comments to match the new behavior

**File:** `include/llama.h`

Update the `llama_kv_cache_compact()` comments so they no longer claim raw `n / ratio` is always the final target.

```cpp
// Compact the KV cache for a sequence using Attention Matching.
// For explicit target_tokens, compaction aims for that exact target.
// For ratio-driven requests, hybrid architectures may resolve to a larger
// effective target to preserve attention-layer context. The return value is the
// authoritative final compacted-prefix token count.
```

This avoids shipping a public API comment that becomes false the moment the feature lands.

### 1f. Snapshot / schema update

**File:** `tools/server/tests/unit/snapshots/compact.json`

Update the snapshot so it matches the live response shape before layering on `hybrid`.

Required top-level fields should include:
- `success`
- `id_slot`
- `method`
- `compacted_tokens`
- `original_tokens`
- `compression_ratio`
- `compaction_time_ms`
- `active_n_kv_before`
- `active_n_kv_after`
- `reclaimed`

Add optional `hybrid` with:
- `detected`
- `skipped_noop`
- `n_attn_layers`
- `n_compactable_layers`
- `n_total_layers`
- `budget_scale`
- `requested_target_tokens`
- `effective_target_tokens`
- `requested_ratio`
- `effective_ratio`

---

## Change 2: BF16 Type Safety

### 2a. Use a sentinel type in the compacted-prefix layout

**File:** `src/llama-kv-compacted-prefix.h`

```cpp
struct llama_compacted_prefix_layer_layout {
    uint32_t  layer_id       = 0;
    uint32_t  n_head_kv      = 0;
    uint32_t  n_embd_head_k  = 0;
    uint32_t  n_embd_head_v  = 0;
    ggml_type type_k         = GGML_TYPE_COUNT;
    ggml_type type_v         = GGML_TYPE_COUNT;
};
```

### 2b. Validate the sentinel with exceptions, not aborts

**File:** `src/llama-kv-compacted-prefix.cpp`

The live contract already throws `std::runtime_error` from `layer_storage::configure()` on invalid types. Preserve that failure mode.

```cpp
void llama_compacted_prefix_store::layer_storage::configure(uint32_t n_tokens) {
    if (layout.type_k >= GGML_TYPE_COUNT || layout.type_v >= GGML_TYPE_COUNT) {
        throw std::runtime_error("compacted-prefix layout has uninitialized type");
    }

    if (!is_supported_compacted_type(layout.type_k, layout.n_embd_head_k, layout.n_embd_head_v) ||
        !is_supported_compacted_type(layout.type_v, layout.n_embd_head_k, layout.n_embd_head_v)) {
        throw std::runtime_error(k_quantized_cache_error);
    }

    if (layout.type_k == GGML_TYPE_BF16 || layout.type_v == GGML_TYPE_BF16) {
        const auto * traits_k = ggml_get_type_traits(layout.type_k);
        const auto * traits_v = ggml_get_type_traits(layout.type_v);
        if (traits_k->from_float_ref == nullptr || traits_v->from_float_ref == nullptr) {
            throw std::runtime_error("compacted-prefix BF16 layout missing from_float_ref conversion");
        }
    }

    // existing allocation logic
}
```

**Why this shape**

- It catches uninitialized type usage early.
- It keeps the existing testable failure contract.
- It does not turn misconfiguration into process death.

### 2c. Materialization audit: no planned code change unless the grep audit proves one is needed

**Files to audit during implementation**
- `src/llama-graph.cpp`
- `src/llama-kv-compacted-prefix-exec.cpp`
- `src/llama-kv-cache.cpp`

Current review evidence shows the graph/materialization path already respects dynamic `type_k` / `type_v`. This plan therefore does **not** require a speculative code change in that path unless the implementation grep finds a real hardcoded `GGML_TYPE_F16` compacted-prefix site.

Implementation rule:
- If the audit finds a hardcoded compacted-prefix tensor type, fix it and add a regression test.
- If the audit is clean, record that fact in the implementation notes and do not make a no-op patch.

### 2d. Precision documentation

**File:** `docs/kv-compaction-algorithm.md`

Add a short numeric-precision note:
- extraction to FP32
- FP32 solver path remains unchanged
- write-back uses native layout type via `from_float_ref`
- BF16 compacted-prefix storage is preserved end to end

---

## Change 3: Regression Tests

### 3a. Test the pure hybrid-info helper directly

**File:** `tests/test-kv-compact-features.cpp`

Replace the mock arithmetic-only test with a pure-helper test over `llama_kv_compact_make_hybrid_info(...)`.

Required cases:
1. Dense: `32 total, 0 recurrent, 32 compactable` -> dense, fraction `1.0`
2. Qwen3.5-35B-like: `40 total, 30 recurrent, 10 compactable` -> hybrid, fraction `0.25`
3. Qwen3.5-122B-like: `48 total, 36 recurrent, 12 compactable` -> hybrid, fraction `0.25`
4. Sparse hybrid example: `52 total, 46 recurrent, 6 compactable` -> hybrid, fraction `6/52`
5. Mismatch case: `52 total, 46 recurrent, 4 compactable` -> `layout_count_mismatch=true`

This closes the prior “test only the arithmetic comment, not the actual helper” gap.

### 3b. Test shared budget resolution directly

**File:** `tests/test-kv-compact-features.cpp`

Add cases that exercise the actual shared budget-resolution helper contract.

Required cases:
1. Dense model ratio path: effective target == requested target
2. Explicit target path: no scaling
3. Qwen3.5-like hybrid with `ratio=8`: scaling applied, effective target larger than requested target, no noop
4. Qwen3.5-like hybrid with `ratio=4`: `skipped_noop=true`, effective target == `compactable`
5. Sparse hybrid cap case: scale capped at `4.0`

Assertions must cover:
- `requested_target_tokens`
- `effective_target_tokens`
- `requested_ratio`
- `effective_ratio`
- `budget_scale`
- `skipped_noop`

### 3c. Add server response contract tests for hybrid metadata

**Files:**
- `tools/server/tests/unit/test_compact.py`
- `tools/server/tests/unit/snapshots/compact.json`

Add tests for:
1. default response still validates against the updated snapshot
2. hybrid response includes requested/effective fields when `hybrid.detected == true`
3. no-op hybrid response returns success with `hybrid.skipped_noop == true`
4. explicit target path does not set `hybrid.skipped_noop`

Use `method = "select"` in default local tests so the V1 beta allowlist is not a false blocker.

### 3d. Add a C API parity test

**File:** `tests/test-kv-compact-quality-multi.cpp` or new dedicated test file

Add a focused C API regression test that verifies hybrid budget resolution is shared, not server-only.

Minimum requirement:
- one model-backed CI test on a Qwen3.5-like GGUF that compares the effective compacted token count from the server path and the C API path for the same ratio-driven request.

Local-dev note:
- if a real hybrid GGUF is unavailable locally, this test is CI-only but must still be part of the implementation plan.

### 3e. BF16 sentinel and exception behavior

**File:** `tests/test-kv-compacted-prefix.cpp`

Required tests:
1. default layout uses `GGML_TYPE_COUNT`
2. uninitialized type throws `std::runtime_error`
3. explicit BF16 type succeeds
4. existing quantized negative tests still throw (not abort)

### 3f. BF16 round-trip test with values outside F16 range

**File:** `tests/test-kv-compacted-prefix.cpp`

Replace the `1e-5`-scale test values with values that actually distinguish BF16 from F16.

Recommended pattern:

```cpp
for (uint32_t t = 0; t < n_tokens; ++t) {
    for (uint32_t d = 0; d < dim; ++d) {
        rows.data[t * dim + d] = 70000.0f + float(t * dim + d);
    }
}
```

Assertions:
- readback remains finite
- readback values stay above `65504.0f`
- relative error remains within BF16 bounds

This closes the prior false-positive test gap where silent F16 downcast would still pass.

### 3g. Keep the sparse-hybrid comments consistent

Use `6/52` consistently in both the plan and tests. Remove the stale `2/52` references.

---

## Verification

### Build + tests

- `cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release`
- `cmake --build build --config Release -j$(sysctl -n hw.ncpu)`
- `ctest --test-dir build -L main --output-on-failure`
- `build/bin/test-kv-compact-features`
- `build/bin/test-kv-compacted-prefix`

### Manual server validation

Use the allowlisted method by default.

```bash
curl -X POST http://localhost:8080/compact \
  -H "Content-Type: application/json" \
  -d '{"ratio": 4, "method": "select"}'
```

Expected on a Qwen3.5-like hybrid:
- success response
- `hybrid.detected == true`
- `hybrid.requested_target_tokens < hybrid.effective_target_tokens`
- if the result is effectively no-op, `hybrid.skipped_noop == true`

### Optional solver validation

If solver is intentionally enabled for local validation, document the required override explicitly:

```bash
LLAMA_COMPACT_ALLOWED_METHODS=select,solver ./build/bin/llama-server ...
```

### CI model-backed validation

Required before implementation is considered done:
1. real Qwen3.5-like GGUF covers the shared helper via both server and C API paths
2. hybrid no-op short-circuit path is exercised
3. BF16 round-trip regression passes with values outside F16 range

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Hybrid scaling still feels too conservative on some models | Medium | expose requested/effective fields in response; tune only after data |
| Shared helper diverges from callers again | Medium | one helper, two callers, parity test in CI |
| Near-no-op success surprises clients | Low | document `hybrid.skipped_noop`; keep top-level `compacted_tokens` authoritative |
| Sentinel default breaks stale tests | Low | add explicit exception-based negative tests |
| Hidden hardcoded F16 compacted-prefix site exists elsewhere | Medium | grep audit during implementation; only patch if real |

---

## Review Delta Incorporated

This revision explicitly absorbs:
- R0: invalid `kv.get_model()` access, wrong insertion point, boundary clamp, wrong JSON target, zero-attention edge case
- R1: server-only scaling, bad `data` variable, near-no-op waste path, BF16 test weakness, detection test weakness, stale snapshot
- R2: ratio-semantics drift, invalid stats plumbing, `to_json()` signature mismatch, abort-vs-throw contract break, stale verification flow
- R3: mismatch logging, stale risk table, sparse-example inconsistency

## Implementation Readiness Bar

This plan is ready for implementation only if reviewers agree that:
1. hybrid scaling is now a shared budget-resolution contract, not a hidden server-only rewrite
2. requested vs effective budget semantics are explicit
3. no-op hybrid cases are handled without running the full pipeline
4. BF16 validation remains exception-based
5. test coverage matches the actual helper and entry-point boundaries
