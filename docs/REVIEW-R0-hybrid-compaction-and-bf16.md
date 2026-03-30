```
─────────────────────────────────────────────────────────
PRIVATE & CONFIDENTIAL

This document is proprietary and intended solely for the
named recipient(s) or authorized audience designated by
the author. Unauthorized reproduction, distribution, or
disclosure — in whole or in part — is strictly prohibited.

If you have received this document in error, or require
additional parties to access its contents, you must obtain
explicit written consent from the author prior to sharing.

Do not forward, copy, or redistribute without authorization.
─────────────────────────────────────────────────────────
```

# R0 Hostile Adversarial Review: PLAN-hybrid-compaction-and-bf16

**Reviewer:** R0 (self-review)
**Date:** 2026-03-29
**Plan:** `docs/PLAN-hybrid-compaction-and-bf16.md`
**Branch:** `modelai-main`
**HEAD at plan time:** `f898d8375`

---

## Verdict: CONDITIONAL PASS

Five findings must be resolved before implementation. Two are **Major** (will produce wrong results or compilation failure on production code paths), three are **Minor**.

---

## 0. Scope Gate

**Severity: OK**

- **In scope:** (A) Hybrid model detection utility, budget scaling in server dispatch, pipeline stats enrichment, JSON response enrichment. (B) BF16 sentinel defaults, configure-time validation, tensor materialization audit, solver precision documentation. (C) Unit tests for both.
- **Out of scope:** Solver algorithm changes, Phase 8 on-policy/sequential-on-policy implementation, per-layer ratio tuning, new JSON API fields in the request schema, iSWA sub-cache compaction.
- **Scope leak:** None observed. The plan explicitly limits itself to detection + scaling + bf16 type safety and does not alter the solver math or add new compaction methods.

---

## 1. Spec Match

**Severity: Major (F-01)**

### F-01: `kv.get_model()` does not exist -- plan references a non-existent API

**Plan lines 77, 87:**
```cpp
const auto & hparams = kv.get_model().hparams;
```

**What the plan claims:** `kv.get_model()` returns the `llama_model &` stored by the KV cache at construction time (plan line 114-116).

**What the code actually has:** `llama_kv_cache` stores `model` as a public member `const llama_model & model` (file `src/llama-kv-cache.h`, line 461) and `const llama_hparams & hparams` as a separate public member (line 462). There is **no `get_model()` method** on `llama_kv_cache`.

**Impact:** The detection function as written will not compile. Additionally, since the detection function takes `const llama_kv_cache &`, the correct access is `kv.hparams` (direct member access), not a method call.

**Minimum fix:** Replace `kv.get_model().hparams` with `kv.hparams` in the detection utility. The function signature and access are otherwise sound since `hparams.is_recurrent()` is a real method.

---

## 2. Contract Boundary Check

**Severity: Minor (F-02)**

### F-02: Plan function `configure()` signature does not match actual API

**Plan line 321-325** specifies:
```cpp
bool llama_compacted_prefix_store::configure(
        llama_seq_id seq_id,
        const std::vector<llama_compacted_prefix_layer_layout> & layouts,
        uint32_t n_tokens,
        const std::vector<llama_pos> & logical_positions)
```

**Actual API** (`src/llama-kv-compacted-prefix.h`, line 86-91):
```cpp
bool llama_compacted_prefix_store::configure_seq(
        llama_seq_id seq_id,
        uint32_t logical_token_count,
        const std::vector<llama_pos> & logical_positions,
        llama_pos live_suffix_pos0,
        bool is_imrope);
```

The method is named `configure_seq`, not `configure`, and has a different parameter list. The layouts are set at construction time and are not per-call. Validation should go in `layer_storage::configure()` (line 204 of `llama-kv-compacted-prefix.cpp`) which processes per-layer layout at storage allocation time, not in `configure_seq`.

**Impact:** Minor -- the plan's intent (validate types) is correct but the insertion point is wrong. A literal implementation would fail at compile time or be placed in a non-existent method.

**Minimum fix:** Specify that type validation goes into `llama_compacted_prefix_store::layer_storage::configure(uint32_t n_tokens)` at `llama-kv-compacted-prefix.cpp:204`, which is where `layout.type_k` and `layout.type_v` are first used.

---

## 3. Trace: Data Flow (Production Trace)

**Severity: OK**

### Trace: Qwen3.5-35B, 4K prefix, ratio=4, no explicit target_tokens

**Given:**
- `n_layer = 40`, `n_recurrent_layers = 30`, `n_attn_layers = 10`
- `attn_fraction = 10/40 = 0.25`
- `prompt_tokens = 4096`, `live_suffix = 0`, `compactable = 4096`
- `ratio = 4.0`
- `target_tokens = max(1, 4096/4.0) = 1024`
- `user_set_explicit_target = false` (JSON has no `target_tokens`)

**Hybrid scaling:**
- `scale = min(1.0/0.25, 4.0) = min(4.0, 4.0) = 4.0`
- `scaled_target = min((uint32_t)(1024 * 4.0), 4096) = min(4096, 4096) = 4096`
- `target_tokens = 4096`

**Post-scaling check at plan insertion point:** `target_tokens >= compactable` (4096 >= 4096) is TRUE. This triggers the error path: "Target tokens must be less than compactable tokens."

### F-03: Scaled target_tokens == compactable triggers the existing error guard

**Severity: Major**

**Plan line 162-164** computes `scaled_target = min((uint32_t)(target_tokens * scale), compactable)` and then assigns it to `target_tokens`. But the existing guard at server-context.cpp line 2340 rejects `target_tokens >= compactable`.

For the primary production scenario (Qwen3.5, ratio=4), the scaled target exactly equals compactable, which means the plan's primary production case will be REJECTED by the existing validation guard, returning an error to the caller instead of performing compaction.

**Impact:** The plan's flagship use case (Qwen3.5-35B at ratio=4) will fail at runtime with an HTTP error. This is not a theoretical edge case -- it is the exact scenario motivating the plan.

**Minimum fix:** Either (a) clamp to `compactable - 1` instead of `compactable` in the scaling code, or (b) insert the hybrid scaling block BEFORE the `target_tokens >= compactable` validation guard so the guard can be adjusted to account for hybrid scaling, or (c) change the guard from `>=` to `>` when hybrid scaling was applied.

Option (a) is simplest:
```cpp
const uint32_t scaled_target = std::min(
    (uint32_t)(target_tokens * scale),
    compactable - 1u);  // must stay strictly below compactable
```

---

## 4. Trace: Error Paths

**Severity: Minor (F-04)**

### F-04: `attn_fraction = 0.0` division-by-zero not guarded at the source

The plan guards `hybrid_info.attn_fraction > 0.0f` at the call site (plan line 160), but the detection utility itself can produce `attn_fraction = 0.0f` if `n_attn_layers == 0` (plan line 101-103: the fraction update is only done when `n_attn_layers > 0`, leaving the default `1.0f`). So `attn_fraction == 0.0f` can never actually reach the call site due to the default initialization.

However, the struct initializes `attn_fraction = 1.0f` AND `is_hybrid = false` for the zero-attn-layers case. So a model where ALL layers are recurrent (theoretically possible) would have `is_hybrid = true` (line 97: `n_recurrent_layers > 0`) but `attn_fraction = 1.0f` (the default, since the `if` on line 101 would be false for `n_attn_layers == 0`).

Wait -- retracing: if `n_attn_layers == 0`, then `n_total_layers - n_recurrent_layers == 0`, so `n_attn_layers == 0`. The condition on line 101 `if (info.n_attn_layers > 0)` is false, so `attn_fraction` stays at `1.0f`. Then at the call site, `is_hybrid` is true (recurrent layers exist), `attn_fraction > 0.0f` is true (it's 1.0), `scale = min(1.0/1.0, 4.0) = 1.0`, and no scaling happens. This is correct behavior.

**Actual issue:** The struct's `attn_fraction` default of `1.0f` is misleading for the `n_attn_layers == 0` case -- it claims 100% attention when there are 0 attention layers. Although the output is functionally harmless (scale=1.0), it makes the struct's semantics inconsistent. A fully-recurrent model would report `{is_hybrid=true, n_attn_layers=0, attn_fraction=1.0}`, which is confusing for logging and JSON response.

**Minimum fix:** Set `attn_fraction = 0.0f` when `n_attn_layers == 0` and add a guard: `if (!hybrid_info.is_hybrid || hybrid_info.n_attn_layers == 0)` at the call site to skip scaling.

---

## 5. Trace: State Machine

**Severity: OK**

The pipeline state machine is not changed. The detection function is a pure query. The budget scaling mutates `target_tokens` (a local variable) before the dispatch block. The stats struct has new fields but they are purely additive (default-initialized to zero/false). No new states are added to the compaction dispatch.

---

## 6. Trace: Concurrency

**Severity: OK (N/A)**

The detection function accesses `const llama_kv_cache &` (read-only), iterating over `hparams.is_recurrent()` which reads from a fixed array (`recurrent_layer_arr`) set at model load time. The budget scaling operates on local variables within the task dispatch handler, which processes one task at a time. No thread safety issues.

---

## 7. Trace: Resource

**Severity: OK**

No new allocations. The `llama_kv_compact_hybrid_info` struct is stack-allocated and returned by value. The stats struct adds 16 bytes of additional fields (bool + 2x uint32_t + float), which is negligible.

---

## 8. Trace: Backwards Compatibility

**Severity: Minor (F-05)**

### F-05: JSON response schema change not reflected in snapshot contract

The plan adds a `"hybrid"` object to the compaction response JSON (plan line 242-249). The existing contract snapshot at `tools/server/tests/unit/snapshots/compact.json` does not include this field. The snapshot currently lists required fields: `success`, `method`, `compacted_tokens`, `original_tokens`, `compression_ratio`, `compaction_time_ms`.

The `server_task_result_compact::to_json()` method (server-task.h line 574-587) does NOT currently include a `hybrid` field. The plan says to add it to the "compaction response JSON builder" but the actual builder is in `server_task_result_compact::to_json()`, not in `server-context.cpp`.

**Impact:** (1) The plan targets the wrong file for JSON enrichment. (2) The `compact.json` snapshot should be updated to include the optional `hybrid` field, or the contract test will fail when the new field is added. (3) The `server_task_result_compact` struct needs to carry the hybrid info so `to_json()` can emit it.

**Minimum fix:** (1) Add hybrid fields to `server_task_result_compact` in `server-task.h`. (2) Update `to_json()` to conditionally emit `"hybrid": {...}`. (3) Update `compact.json` snapshot. (4) The plan should reference `server-task.h` not `server-context.cpp` for the JSON builder.

---

## 9. Multi-Variant Model Trace

**Severity: OK**

| Model Type | `is_recurrent()` pattern | Detection result | Scaling |
|---|---|---|---|
| **Dense (Llama3.1-8B)**: all 32 layers attention | `recurrent_layer_arr` all false | `is_hybrid=false, attn_fraction=1.0` | No scaling (correct) |
| **Hybrid (Qwen3.5-35B)**: 10/40 attn | 30 layers true | `is_hybrid=true, attn_fraction=0.25` | scale=4.0 (correct) |
| **iSWA (Gemma2)**: SWA layers filtered to separate cache | SWA layers filtered by `filter_base` | The `llama_kv_cache` instance only has non-SWA layers. `is_recurrent()` is false for attention layers. | `is_hybrid=false` (correct -- no recurrent layers in the base cache) |
| **MoE (Qwen3.5-MoE)**: same hybrid pattern, different FFN | Recurrent layers same | Same as Qwen3.5-35B | Correct |
| **Pure recurrent (hypothetical)**: all layers recurrent | All true, no KV cache layers | KV cache would be empty, compaction not invoked | N/A |

The detection correctly relies on `hparams.is_recurrent()` which is the source of truth for recurrent layers across all architectures. The KV cache's `layers` vector already excludes recurrent layers (filtered by `hparams.has_kv()` at construction, and recurrent layers have `has_kv() == true` BUT they are handled by `llama_memory_hybrid` which creates separate attention-only and recurrent caches). When accessed via `llama_kv_compact_get_cache()`, the returned `llama_kv_cache *` is always the attention sub-cache.

---

## 10. Precondition Audit

| Assumption | Enforced? | Status |
|---|---|---|
| `kv.get_compacted_prefix()` is non-null | Checked (line 81) | Enforced |
| `layouts` is non-empty | Checked (line 84) | Enforced |
| `kv.get_model()` exists | NOT enforced -- method does not exist | **F-01** |
| `hparams.is_recurrent(il)` valid for all `il < n_layer` | Enforced by GGML_ABORT in hparams | Enforced |
| `n_total_layers > 0` | Implicit (division by n_total_layers on line 102) | **Undocumented** -- could divide by zero if `n_layer == 0`, though this is impossible in practice (no model has 0 layers) |
| `GGML_TYPE_COUNT` is a valid sentinel | Implicit (enum ordering) | Documented in plan, enforced by validation |
| `from_float_ref` is non-null for bf16 | Asserted (line 341) | Enforced |
| `target_tokens * scale` does not overflow uint32_t | NOT enforced | See integer arithmetic trace |

---

## 11. Test Reality Check

**What tests prove:**
- The hybrid detection arithmetic is correct for 4 model configurations (dense, Qwen3.5-35B, Qwen3.5-122B, hypothetical sparse).
- The budget scaling arithmetic is correct for 5 cases including the 4x ceiling clamp.
- The sentinel default is GGML_TYPE_COUNT, not GGML_TYPE_F16.
- bf16 round-trip via `write_compacted_payload` preserves precision within bf16 bounds.

**What tests do NOT prove:**
- The tests for hybrid detection (3a) do NOT test the actual detection function `llama_kv_compact_detect_hybrid`. They test the arithmetic formula directly using mock values. This means a bug in how the function accesses `kv.hparams` or `cp->get_layouts()` would not be caught.
- The budget scaling test (3b) does not test the `user_set_explicit_target` bypass path.
- The bf16 test (3d) tests values in `[1e-5, 2e-5]` which are within both bf16 and f16 normal range. It does NOT test values that would distinguish bf16 from f16 (e.g., values > 65504 which overflow f16 but are normal in bf16, or values < 6e-8 which are subnormal in f16 but normal in bf16).
- No integration test verifies end-to-end behavior with a real hybrid model (acknowledged in plan, deferred to CI).
- No test verifies the JSON response enrichment.
- No test verifies that the `target_tokens >= compactable` guard is handled correctly after scaling (F-03).

**Would tests catch F-03?** No. The budget scaling test (3b) checks only the arithmetic formula in isolation. It does not simulate the server dispatch flow where `target_tokens >= compactable` rejects the request.

---

## 12. Disprove-It Pass

### Attempt 1: Integer overflow in `target_tokens * scale`
- `target_tokens` is uint32_t, `scale` is float.
- `(uint32_t)(target_tokens * scale)`: if `target_tokens = 4,000,000,000` and `scale = 4.0`, then `target_tokens * scale = 16,000,000,000.0f`. Casting to uint32_t: undefined behavior in C++ when the float exceeds UINT32_MAX (4,294,967,295).
- **Does this break?** In practice, `target_tokens` is bounded by `compactable` which is bounded by `prompt_tokens` (uint32_t, max ~2^32-1). With `scale` capped at 4.0, `target_tokens * scale` could reach `~1.7e10` for a 4GB-token prompt (unrealistic but technically possible). The `std::min` with `compactable` clamps the final value, but the intermediate `(uint32_t)` cast happens BEFORE the `std::min`, potentially causing UB.
- **Realistic risk:** Low -- no model has 4 billion tokens in the prefix. But the cast should use `(uint32_t)std::min((float)target_tokens * scale, (float)compactable)` to avoid the intermediate UB.
- **Verdict:** Theoretical UB, not a production issue. Minor.

### Attempt 2: Dense model with `attn_fraction == 1.0`
- `is_hybrid = false` (n_recurrent_layers == 0), so the scaling block is skipped entirely.
- **Verdict:** Correct. No scaling for dense models.

### Attempt 3: Qwen3.5 with explicit `target_tokens` in JSON
- `user_set_explicit_target = true`, scaling block skipped.
- **Verdict:** Correct.

### Attempt 4: Qwen3.5 with ratio=2 (moderate compression)
- `target_tokens = max(1, 4096/2.0) = 2048`
- `scale = 4.0`, `scaled_target = min(8192, 4096) = 4096`
- Post-guard: `4096 >= 4096` -- ERROR. Same as F-03.
- **Verdict:** Confirmed F-03 across all ratios for Qwen3.5.

### Attempt 5: Qwen3.5 with ratio=8 (aggressive compression)
- `target_tokens = max(1, 4096/8.0) = 512`
- `scale = 4.0`, `scaled_target = min(2048, 4096) = 2048`
- Post-guard: `2048 >= 4096` -- false. Proceeds.
- **Verdict:** Only works when ratio > (1/attn_fraction). For Qwen3.5 (attn_fraction=0.25), ratio must be > 4.0 for the plan to not hit the guard.

### Attempt 6: `GGML_TYPE_COUNT` sentinel breaking `is_supported_compacted_type()`
- `layer_storage::configure()` calls `is_supported_compacted_type(layout.type_k, ...)` before the plan's proposed validation.
- If `layout.type_k == GGML_TYPE_COUNT`, `ggml_row_size(GGML_TYPE_COUNT, ...)` will likely crash or return garbage.
- **But:** All production layout construction explicitly sets type from live cache tensors (kv-cache.cpp:215-216). Only test/default-constructed layouts hit the sentinel.
- **Verdict:** Safe for production, but the validation in `configure()` should come BEFORE `is_supported_compacted_type()`, not after.

---

## Dependency / Supply-Chain Check

- **New dependencies:** None. All referenced APIs (`hparams.is_recurrent()`, `ggml_get_type_traits()`, `ggml_bf16_to_fp32_row`) are already in the codebase.
- **License compatibility:** N/A.
- **Known CVEs:** N/A.

---

## Cross-Repo Contract Check

- **Contract touched:** `/compact` REST API response JSON. The plan adds an optional `"hybrid"` field to the response.
- **Consumer:** ModelAI server (consumes `/compact` response).
- **Backward compatibility:** The new field is additive and optional (only present when `hybrid_detected == true`). Consumers using strict schema validation against `compact.json` may fail if they reject unknown fields, but the current schema uses `"type": "object"` without `"additionalProperties": false`, so unknown fields are permitted.
- **Verdict:** Backward compatible, but the snapshot should be updated to document the new field.

---

## Performance Check

- **Hot path affected:** No. The detection function runs once per `/compact` request (not per token or per layer during inference). The `is_recurrent()` loop over `n_layer` is O(40) for Qwen3.5 -- negligible.
- **Measurement:** N/A (not on hot path).

---

## Unsupported / Precondition Audit

| # | Assumption | Status |
|---|---|---|
| 1 | `llama_kv_cache` has public `hparams` member | Enforced (verified in codebase) |
| 2 | `recurrent_layer_arr` is populated at model load | Enforced (set by model loader for hybrid architectures) |
| 3 | KV cache `layers` vector excludes recurrent layers | Enforced (`has_kv()` filter + hybrid memory separation) |
| 4 | `ggml_get_type_traits(GGML_TYPE_BF16)->from_float_ref` is non-null | Enforced (asserted in plan; verified in ggml source) |
| 5 | `GGML_TYPE_COUNT` is the last enum value and invalid as a real type | Documented (standard ggml convention) |
| 6 | `compactable > 0` when hybrid scaling runs | Undocumented -- if `compactable == 0`, the division `compactable / ratio` at line 2338 would produce 0, and the guard at line 2340 would catch it. But with scaling, `target_tokens * scale` with `target_tokens == 0` would produce 0, and `min(0, compactable)` = 0, which is fine. |

---

## Findings Summary

| ID | Severity | Section | Description |
|---|---|---|---|
| F-01 | **Major** | Spec Match | `kv.get_model()` does not exist; should be `kv.hparams` |
| F-02 | Minor | Contract Boundary | Plan references `configure()` but actual method is `configure_seq()` with different signature; validation insertion point is wrong |
| F-03 | **Major** | Data Flow | Scaled `target_tokens == compactable` triggers existing error guard for the primary production scenario (Qwen3.5 at ratio <= 4) |
| F-04 | Minor | Error Paths | `attn_fraction` semantics inconsistent for fully-recurrent models (reports 1.0 when should be 0.0) |
| F-05 | Minor | Backwards Compat | JSON enrichment targets wrong file (`server-context.cpp` vs `server-task.h`); compact.json snapshot not updated |

---

## Fixes Required Before Implementation

1. **F-01 (Major):** Change `kv.get_model().hparams` to `kv.hparams` in `llama_kv_compact_detect_hybrid()`.

2. **F-03 (Major):** Clamp `scaled_target` to `compactable - 1` instead of `compactable`, OR reorder the scaling to occur before the `target_tokens >= compactable` guard, adjusting the guard to permit `target_tokens == compactable - 1` after scaling. This is the most critical finding -- without this fix, the plan's primary production scenario will fail at runtime.

3. **F-02 (Minor):** Correct the validation insertion point to `layer_storage::configure()` in `llama-kv-compacted-prefix.cpp:204`.

4. **F-04 (Minor):** Set `attn_fraction = 0.0f` for `n_attn_layers == 0` and guard the call site with `hybrid_info.n_attn_layers > 0`.

5. **F-05 (Minor):** Move JSON enrichment to `server_task_result_compact` in `server-task.h` and update `compact.json` snapshot.

---

## Deferred Risks (safe to defer)

- **Integration test with real Qwen3.5 model:** Deferred to CI as the plan states. The arithmetic is well-covered by unit tests.
- **4x ceiling tuning:** Conservative but safe. Can be made configurable per-model in a future plan.
- **bf16 test coverage in extreme ranges:** The test covers normal bf16 range but not the full dynamic range distinction from f16. Adequate for a plan review; implementation should add values > 65504 if bf16 testing is a priority.
- **Integer overflow in `(uint32_t)(target_tokens * scale)`:** Theoretical UB for unrealistic input sizes. Safe to address during implementation with a float-domain min.
