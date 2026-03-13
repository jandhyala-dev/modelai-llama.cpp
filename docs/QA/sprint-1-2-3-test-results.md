# Sprint 1-2-3 Test Results

**Date:** 2026-03-12
**Branch:** modelai-main
**Plan baseline:** d1676632
**Implementation range:** 7877fb9d..4d77b546 (13 commits) + bcced551 (2f fix)
**R2 Review:** PASS (no blocking findings)
**Model:** Qwen3-8B-Q4_K_M (8.0B, Dense GQA, Metal backend)

## Test Plan Results

| Phase | Test | Result | Notes |
|-------|------|--------|-------|
| 1 | Clean build | **PASS** | Zero errors, all targets built |
| 2 | Unit tests (ctest -L main) | **PASS** | 45/46 passed. 1 pre-existing failure: test-tokenizers-ggml-vocabs (missing vocab test data) |
| 3 | Model tests (ctest -L model) | **PASS** | 9/9 passed (incl. state-restore, pack, quality) |
| 4 | Server API tests (pytest) | **SKIP** | pytest model download infra issue (HF API 404). Pre-existing. |
| 5 | CSV schema = 53 columns, version = 2 | **PASS** | 53 columns, last 3: support_level, support_reason, throughput_pass. schema_version=2 |
| 6 | classify_support boundaries | **PASS** | select/4/4096=supported, select/8/4096=experimental, baseline/1/4096=supported |
| 7 | Self-study diagnostics | **PASS** | All fields logged. See diagnostic output below |
| 8 | Stats struct = 14 fields | **PASS** | 14 default-initialized fields in stats struct |
| 9 | Counter separation | **PASS** | n_beta_heads_seen separate from n_heads_seen |
| 10 | Solver unmodified | **PASS** | 0 lines changed in llama-kv-compact-solver.cpp |
| 11 | Tensor cache exists | **PASS** | mutable cp_cache, non-mutable version_counter, accessor |
| 12 | Version bumps correct | **PASS** | configure(575), clear(586), state_read(2477). NOT in set_execution |
| 13 | Cache functional test | **PASS** | 8K/8x cosine=0.985, no crash |
| 14 | /compact warnings | **PASS** | 2 SRV_WRN (quality unproven + throughput may regress) |
| 15 | 32K throughput | **PASS** | 32K/50x: cosine=0.997, throughput +38% (B5 cache). 32K/4x: cosine=0.999, throughput -40% |
| 16 | State restore regression | **PASS** | test-state-restore-compacted-prefix passed |
| 17 | Quality regression | **PASS** | test-kv-compact-quality passed (cosine=0.999518 at 8x) |
| 18 | Full campaign | **DEFERRED** | Multi-hour run, not executed in this session |

**Blocking phases passed: 14/16** (2 informational phases: 15, 18)
**Skipped: 1** (Phase 4 — pre-existing test infra issue)

## Self-Study Diagnostic Output (Phase 7 / Sprint 2f)

### Before Q/K Normalization Fix
```
q_capture: skipping tensor 'Qcur-35' — dims [4096,1] != expected [128,32]
self-study: captured Q from 256 tokens (seq 0)
self_study diagnostics: layers_with_q=36 dim_mismatches=7344
  q_norm=17.1562 k_norm=28.2998 beta_norm=535.8967
  beta_sparsity=0.0000 fit_residual=0.166906
self-study: pipeline complete — 2620 prefix -> 1310 selected (seq 0)

cosine=0.710430 (threshold=0.9500 FAIL)
compact=721261.2ms | baseline=3.7 tok/s | compacted=1.7 tok/s | delta=-54.32%
```

### Root Cause Analysis
- **layers_with_q=36**: All 36 layers captured Q successfully
- **dim_mismatches=7344**: 36 layers x ~204 mismatches/layer. Qcur tensors with wrong dims (4096x1 instead of 128x32) are correctly skipped — these are pre-reshape projections
- **q_norm=17.16 vs k_norm=28.30**: Q/K scale mismatch — K is 1.65x larger than Q. This is because self-study Q comes from W_q projection while K comes from W_k projection, which have different learned scales
- **beta_norm=535.90**: Extreme beta weights caused by the Q/K norm mismatch distorting attention softmax
- **beta_sparsity=0.00**: Solver is NOT producing degenerate solutions — weights are all non-zero
- **fit_residual=0.167**: Moderate fit error (17% relative error)

### Fix Applied (bcced551)
Q vectors normalized to match K scale before attention scoring:
`scale = k_norm / q_norm` applied per-head. Diagnostics log pre-normalization values.

### After Q/K Normalization Fix (re-measurement, R2 verified)

Prior re-measurement (cosine=0.725) likely ran against a stale binary. R2 independent
verification after confirmed rebuild (same commit bcced551):

```
RATIO=2:
self_study diagnostics: layers_with_q=36 dim_mismatches=7344
  q_norm=15.9415 k_norm=28.5525 beta_norm=324.4391
  beta_sparsity=0.0000 fit_residual=0.036826
self-study: pipeline complete — 2620 prefix → 1310 selected (seq 0)
cosine=0.995271 (threshold=0.9500 PASS)

RATIO=4:
self_study diagnostics: layers_with_q=36 dim_mismatches=7344
  q_norm=15.9415 k_norm=28.5525 beta_norm=285.1027
  beta_sparsity=0.0000 fit_residual=0.062974
self-study: pipeline complete — 2620 prefix → 655 selected (seq 0)
cosine=0.987612 (threshold=0.9000 PASS)
```

**Conclusion:** Q/K norm mismatch WAS the primary root cause. After fix:
- beta_norm dropped from 536 to 285-324 (40% reduction)
- fit_residual dropped from 0.167 to 0.037-0.063 (63-78% reduction)
- cosine improved from 0.710 to 0.988-0.995 (quality PASSES at both ratios)

## Benchmark Quality Results

| Config | Cosine | Threshold | Pass | Support Level |
|--------|--------|-----------|------|---------------|
| select/2/4096 | 0.973 | 0.950 | PASS | supported |
| select/4/4096 | 0.928 | 0.900 | PASS | supported |
| select/8/4096 | 0.838 | 0.850 | FAIL | experimental |
| select/8/8192 | 0.985 | 0.850 | PASS | supported |
| select/4/32768 | 0.999 | 0.900 | PASS | supported |
| select/50/32768 | 0.997 | 0.850 | PASS | supported (+38% throughput) |
| baseline/1/4096 | — | — | PASS | supported |
| self_study/2/4096 | 0.995 | 0.950 | PASS | experimental |
| self_study/4/4096 | 0.988 | 0.900 | PASS | experimental |

## Classification Updates

- **32K**: upgraded to **supported** (select/50/32K cosine=0.997 PASS, throughput +38% with B5 tensor cache)
- **self_study**: upgraded to **experimental** (cosine=0.988-0.995 PASS; pipeline overhead ~13-22min limits practical use)

## Commits

### Sprint Implementation (13 commits)
| Slice | SHA | Description |
|-------|-----|-------------|
| 1a | 7877fb9d | Support classification in benchmark CSV |
| 1b | 30a691ee | Rename throughput_delta_pct -> in_run_throughput_delta_pct |
| 1c | d57c58a3 | Server capability reporting with supported_envelope |
| 1d | 1bcc4283 | Support envelope documentation |
| 1e | 6ac1a899 | /props test assertions |
| 2a | 72a67f0b | Diagnostic logging for Q capture dim mismatch |
| 2b | 06d1b6d8 | Extended stats struct (7->14 fields) |
| 2c | 077c10ca | Populate diagnostics with separate counters |
| 2d | a25fa5a8 | Log diagnostics after pipeline run |
| 2e-2g | aae167ee | Root cause documentation |
| 3a | d83df58c | Per-function materialization timing |
| 3b | 9c0d6aca | Graph tensor caching for compacted prefix |
| 3c-3d | 4d77b546 | Runtime policy guard in /compact endpoint |

### Post-Sprint Fix
| Slice | SHA | Description |
|-------|-----|-------------|
| 2f | bcced551 | Q/K norm mismatch fix in self-study pipeline |
