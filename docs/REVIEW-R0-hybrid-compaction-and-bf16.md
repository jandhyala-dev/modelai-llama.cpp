# R0 Self-Adversarial Review: Hybrid Budget Resolution + BF16 Type Safety

- Repository: `modelai-llama.cpp`
- Branch: `modelai-main`
- Plan: `docs/PLAN-hybrid-compaction-and-bf16.md`
- Review date: 2026-03-29
- Review scope: final plan revision after R0/R1/R2/R3 findings
- Review standard: repo-local 13-section adversarial protocol in `AGENTS.md`

## Verdict

**PASS — no findings**

This revision absorbs the previously reported blockers and leaves no plausible plan-level correctness bug that should stop implementation.

## 1. Scope Gate

**In scope**
- Hybrid detection
- Shared budget resolution for ratio-driven compaction
- Server/C API parity
- Compact response contract updates
- BF16 sentinel + validation contract
- Test plan changes

**Out of scope**
- Solver math changes
- Unsupported architecture expansion
- Benchmark work
- Website/docs outside the compaction plan slice

**Scope leak**
- None. The plan is bounded to the intended feature and its direct contracts.

## 2. Spec Match

**Status: PASS**

The plan now matches its stated intent:
- hybrid behavior is architecture-derived, not model-name hardcoded
- the server and C API use one shared budget rule
- ratio semantics are explicitly redefined as requested vs effective budget on hybrids
- BF16 hardening preserves the repo's existing exception-based validation style

## 3. Contract Boundary Check

**Status: PASS**

Closed previously broken boundaries:
- server-only scaling -> now shared helper for server + C API
- hidden ratio rewrite -> now explicit requested/effective contract
- pipeline-stats response plumbing -> removed in favor of direct response fields
- `to_json() const` drift -> removed; plan keeps `json to_json() override`
- `GGML_ABORT` sentinel failure mode -> replaced with `std::runtime_error`

## 4. Concrete Traces

### 4a. Production trace

Qwen3.5-like hybrid:
- total layers: `40`
- recurrent layers: `30`
- compactable layers: `10`
- `compactable_fraction = 10 / 40 = 0.25`
- `compactable = 4096`
- `ratio = 4.0`
- base requested target = `4096 / 4 = 1024`
- scale = `min(1 / 0.25, 4.0) = 4.0`
- scaled target = `min(1024 * 4, 4096) = 4096`
- helper marks `skipped_noop = true`, `effective_target_tokens = 4096`, `effective_ratio = 1.0`

Result:
- server returns success with `hybrid.skipped_noop = true`
- C API returns `4096`
- no hidden divergence remains

### 4b. Boundary trace

Explicit target path:
- `target_tokens = 500`
- `explicit_target = true`
- helper returns requested/effective target == 500
- `requested_ratio == 0.0`, `effective_ratio == 0.0`
- no hybrid scaling applied

This preserves the explicit-target contract.

### 4c. Adversarial trace

Qwen3.5-like hybrid, `ratio = 8`:
- requested target = `512`
- scaled target = `2048`
- not noop
- helper returns `requested_ratio = 8.0`, `effective_ratio = 2.0`

This is the intended architecture-aware compaction behavior, and it is now visible in the response contract instead of being silent drift.

### 4d. Integer arithmetic trace

- `requested_target_tokens * scale` is clamped in float-domain before the casted result is used
- `compactable_fraction` uses float division only after `n_total_layers > 0` guard
- no remaining plan snippet performs the old unsafe cast-before-min pattern

### 4e. Security trace

No new security surface is introduced. The plan changes response fields and internal target resolution only.

### 4f. Concurrency trace

The helper reads immutable model/cache metadata and returns a stack-local result. No new shared mutable state is introduced.

## 5. Multi-Variant Model Trace

**Status: PASS**

- Dense causal model: `is_hybrid = false`, no scaling
- Qwen3.5-like hybrid: scaling based on `compactable_fraction`
- Hybrid + layer mismatch: mismatch logged, scaling based on actual compactable layers
- Fully recurrent / zero compactable layers: hybrid adjustment skipped safely

This is a stronger architecture boundary than the prior attn-layer-only plan.

## 6. State-Machine Trace

**Status: PASS**

Server path state machine is coherent:
1. parse request
2. compute base requested target
3. resolve hybrid effective budget
4. either short-circuit noop or run pipeline
5. emit response with requested/effective metadata

BF16 path state machine is coherent:
1. default sentinel type
2. configure-time validation
3. throw on invalid layout
4. proceed to normal allocation/write path on valid BF16 layout

## 7. Unsupported / Precondition Audit

**Status: PASS**

Preconditions are now explicit:
- explicit target bypasses scaling
- hybrid scaling applies only when compactable layers exist
- mismatch is logged
- BF16 validation is fail-closed and recoverable
- no hidden dependence on removed `get_model()` accessors or HTTP-only variables

## 8. Test Reality Check

**Status: PASS**

The plan now closes the earlier test gaps:
- pure helper test instead of arithmetic-only comment test
- shared budget-resolution tests for requested vs effective behavior
- server response schema update
- C API parity test requirement
- BF16 sentinel throw test
- BF16 round-trip values that actually distinguish BF16 from F16
- explicit-target path leaves ratio fields at `0.0`
- server/C API parity uses the same ratio-derived minimum target floor

## 9. Disprove-It Pass

Assumed one remaining bug existed and pushed on the prior failure points:
- server/C API divergence
- hidden ratio semantics drift
- response plumbing through method-specific stats
- abort-vs-throw BF16 validation
- stale verification flow

All were directly addressed by the revised plan structure.

## 10. Dependency Check

No new external dependencies are introduced.

## 11. Performance Regression Check

The plan contains one deliberate performance guard:
- near-no-op hybrid cases short-circuit instead of running the full compaction pipeline

That is strictly better than the prior plan, which could spend full solver/selection cost to remove one token.

## 12. Cross-Repo Contract Check

**Status: PASS**

Relevant shared contracts now line up:
- server path and C API both use the same budget helper
- public API comment is explicitly updated to match the new hybrid-aware behavior
- server response schema is updated together with the JSON builder and snapshot

## 13. Pass Bar

**PASS**

No plausible plan-level production bug remains after the final rewrite.

## Findings Closed by This Revision

Closed from prior review rounds:
- invalid `kv.get_model()` access
- wrong configure insertion point
- scaled-target boundary failure
- zero-attention hybrid edge case
- wrong JSON enrichment file/shape
- server-only scaling
- wrong `data` variable in dispatch snippet
- near-no-op waste path
- stale BF16 round-trip values
- synthetic requested-ratio on explicit-target path
- server/C API floor divergence
- detection test not exercising a real helper
- stale `compact.json` schema
- invalid stats propagation path
- `to_json()` signature mismatch
- abort-vs-throw contract break
- stale manual validation flow
- stale risk-table text
- sparse-example comment mismatch
- BF16 trait check abort-vs-throw mismatch
