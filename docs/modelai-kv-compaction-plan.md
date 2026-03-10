# KV Compaction — Implementation Plan

## Milestone Overview

| PR | Branch | Scope |
|---|---|---|
| PR-0 | `kv-compact-pr0-docs` | Docs baseline |
| PR-1 | `kv-compact-pr1-observability` | Capability flags and observability |
| PR-2 | `kv-compact-pr2-memory-arch` | Compacted-prefix memory architecture |
| PR-3 | `kv-compact-pr3-correctness` | Non-flash correctness path |
| PR-4 | `kv-compact-pr4-session-state` | Session and state integration |
| PR-5 | `kv-compact-pr5-performance` | Real performance path |
| PR-6 | `kv-compact-pr6-coverage` | Coverage expansion |

## PR-0: Docs Baseline

**Objective:** Finalize architecture and implementation docs for the private product fork.

Scope: executive summary, implementation plan, git/CI/release policy.

No runtime code. Merge gate: docs are internally consistent, no overclaims.

## PR-1: Capability Flags and Observability

**Objective:** Add infrastructure ModelAI needs before compaction exists.

Scope:
- Feature flags (compile-time and runtime)
- Capability reporting via `llama-server` endpoints
- Telemetry hooks: allocated KV bytes, active `n_kv`, prefill timing, decode timing/tok-s, fallback-state reporting, `query_generation_time_ms`, `solver_time_ms`

Tests: feature flag tests, capability-query tests, telemetry presence tests, control-path no-regression tests.

Merge gate: flags default off, existing behavior unchanged when disabled.

## PR-2: Compacted-Prefix Memory Architecture

**Objective:** Establish internal representation for compacted KV state.

Scope:
- Separate compacted-prefix representation under `llama_memory_t`
- Per-layer/per-head storage of `(C_k, beta, C_v)`
- Logical position bookkeeping
- Memory accounting hooks

Tests: layout, metadata, position bookkeeping, serialization-shape, memory accounting.

Merge gate: representation is isolated and test-covered.

## PR-3: Non-Flash Correctness Path

**Objective:** End-to-end compaction on narrow supported matrix.

Scope:
- Query extraction from prefill path
- GQA regrouping into KV-head query space
- Key selection fast path (topk by attention score)
- NNLS beta fitting
- OLS value fitting (primary: Householder QR; fallback: regularized Cholesky, lambda=1e-6)
- Chat-template/BOS preservation
- Optional model-specific nonuniform schedules with uniform fallback
- Non-flash execution using additive `kq_b`
- Explicit fallback for unsupported configs

Solver policy: all math in fp32, cast results to model dtype for KV storage.

Tests: solver unit tests, reference parity on fixed fixtures, end-to-end sanity, unsupported-config fallback, no-NaN/no-crash.

Merge gate: supported matrix works end-to-end with tolerance bounds.

## PR-4: Session and State Integration

**Objective:** Compacted state usable in real session lifecycles.

Scope: compact/decompact lifecycle, save/restore handling, invalidation rules, repeated compaction cycles, session continuation.

Tests: save/load roundtrip, failed-restore fallback, repeated cycles, session continuation.

## PR-5: Real Performance Path

**Objective:** Turn logical compaction into measurable wins.

Scope:
- Packed layout so reduced active range lowers real compute
- Real memory reuse/release where possible
- Repeated-turn optimization
- Benchmark harness integration

Benchmarks required: W1-W6 workloads (see below).

Merge gate: measured progress on Goal 1 and/or Goal 2 on at least one ModelAI workload.

## PR-6: Coverage Expansion

**Objective:** Broaden support after narrow path works.

Scope: flash-compatible path with real `kq_b` support, quantized V, SWA evaluation, hybrid-memory evaluation, improved query-generation (self-study, OMP).

Each added path must have isolated tests and benchmark evidence.

## Benchmark Workloads

| ID | Workload | Purpose |
|---|---|---|
| W1 | 80K filing → first answer | Cold long-context behavior |
| W2 | 80K filing → 20 follow-up questions | Repeated-turn speed |
| W3 | Executive summary generation | Standard analyst workflow |
| W4 | Full research report generation | Heavy multi-step workflow |
| W5 | 3 concurrent sessions on 32GB | Local scalability |
| W6 | Save/restore + continue | Session continuity |

## Per-Run Metrics

1. Model name and quantization
2. Backend (Metal / CUDA / CPU)
3. Flash attention on/off
4. Compaction on/off and ratio
5. Prefill latency ms
6. First-token latency ms
7. Decode throughput tok/s
8. Allocated KV bytes
9. Active KV length
10. Quality delta vs full cache
11. `query_generation_time_ms`
12. `solver_time_ms`

## Regression Rules

- Performance regression > 10% on a key metric requires investigation before release
- Quality regression > 1% perplexity delta requires investigation before release
- No benchmark claim without measured outputs
