# KV Compaction — Implementation Plan

## Executive Summary

This plan implements KV compaction as a private product-fork feature for ModelAI, not as an upstream-ready monolith.

The two product goals are:
1. long-session efficiency,
2. post-prefill speed / decode throughput.

The main blocker is runtime and memory architecture, not solver math alone.

The implementation strategy is:
1. establish governance and observability first,
2. build the internal compacted-prefix representation under `llama_memory_t`,
3. prove correctness on a narrow non-flash matrix,
4. integrate session/state lifecycle,
5. pursue real performance gains only after representation and correctness are stable.

Upstream reference point:
- `ggml-org/llama.cpp#20037` is the standing upstream tracking issue for this work.

## Design Constraints

The plan assumes the following current `llama.cpp` realities:

1. `llama_memory_t` is the correct architecture boundary.
2. The current KV allocation model is fixed-size, so logical reduction does not automatically reduce physical memory.
3. Non-flash currently supports additive `kq_b`; flash does not.
4. Quantized V depends on flash attention today.
5. Position continuity and state restore behavior are strict.
6. SWA and hybrid recurrent-memory paths require separate treatment.
7. Active `n_kv` reduction matters only if active cells are packed into a backend-usable layout.

## Supported And Unsupported Matrix

### v0 Supported

- standard causal models with `llama_kv_cache`
- non-flash attention path
- non-quantized V cache
- uncompacted chat-template / BOS prefix
- uniform budgets by default
- precomputed nonuniform schedules where explicitly validated

### v0 Unsupported

- flash-attention compaction path
- quantized V compaction
- SWA / split-memory compaction
- hybrid recurrent + attention compaction
- M-RoPE edge cases not yet validated
- broad public API guarantees

## Milestone Overview

| PR | Branch | Scope |
|---|---|---|
| PR-0 | `kv-compact-pr0-docs` | Docs baseline and governance |
| PR-1 | `kv-compact-pr1-observability` | Capability flags and observability |
| PR-2 | `kv-compact-pr2-memory-arch` | Compacted-prefix memory architecture |
| PR-3 | `kv-compact-pr3-correctness` | Non-flash correctness path |
| PR-4 | `kv-compact-pr4-session-state` | Session and state integration |
| PR-5 | `kv-compact-pr5-performance` | Real performance path |
| PR-6 | `kv-compact-pr6-coverage` | Coverage expansion |

## PR-0: Docs Baseline And Governance

**Objective**

Finalize architecture and implementation docs for the private product fork and make the operating model explicit.

**Scope**

- executive summary
- implementation plan
- git policy
- CI policy
- release checklist

**Non-goals**

- no runtime behavior changes
- no benchmark claims beyond measured CI smoke artifacts

**Merge gate**

- docs are internally consistent
- no exactness or fake VRAM claims
- support matrix is explicit
- governance is documented end-to-end

## PR-1: Capability Flags And Observability

**Objective**

Add the infrastructure ModelAI needs before compaction exists.

**Scope**

- feature flags, default-off
- capability reporting via `llama-server` endpoints, including:
  - effective context window
  - flash-attention availability
  - prompt / prefix cache behavior
  - structured output / JSON-schema availability
  - chat-template control availability
  - embeddings / rerank availability
  - save / restore safety
  - compacted-prefix availability
- endpoint contract:
  - `/props` returns `modelai.capabilities` and `modelai.runtime`
  - `/props` also returns `modelai.contract` for version negotiation and provenance
  - `/models` returns model-level ModelAI metadata for routing and compatibility checks
  - `/metrics` carries the same baseline memory / KV totals needed for dashboards and benchmark capture
  - the concrete consumer-facing schema is defined in ModelAI's `MODELAI_LLAMA_CPP_INTEGRATION_CONTRACT.md`
- telemetry hooks for:
  - allocated KV bytes
  - active `n_kv`
  - prefill timing
  - decode timing / tok-s
  - fallback-state reporting
  - `query_generation_time_ms`
  - `solver_time_ms`
  - `llamacpp:modelai_*` Prometheus gauges for ModelAI-specific runtime telemetry
- build provenance that resolves the upstream base commit deterministically in CI and local managed builds

**Non-goals**

- no compacted-prefix representation yet
- no flash support
- no new public API guarantees outside documented server surfaces
- no non-null compaction timing claims before PR-3
- no assumption that slot/task presence is required for zero-safe KV telemetry

**Tests**

- feature flag tests
- capability-query tests
- telemetry presence tests
- control-path no-regression tests
- sleeping-server `/props` tests
- idle-slot metrics tests

**Merge gate**

- flags default off
- existing behavior unchanged when disabled
- ModelAI can query capabilities and telemetry without needing compaction enabled
- unsupported configs are reported explicitly rather than inferred indirectly
- runtime telemetry is safe for sleeping servers and zero-safe for idle slots
- product-fork internal-header coupling is documented explicitly where PR-1 reaches non-public runtime internals for telemetry

## PR-2: Compacted-Prefix Memory Architecture

**Objective**

Establish the internal representation for compacted KV state.

**Scope**

- separate compacted-prefix representation under `llama_memory_t`
- per-layer / per-KV-head storage of `(C_k, beta, C_v)`
- logical position bookkeeping
- forwarding of core sequence operations into the compacted-prefix state
- memory accounting hooks for host-side sidecar allocation
- scalar compacted cache element types only (`F16`, `BF16`, `F32`)
- stale compacted-prefix invalidation on `state_read`
- non-serialized-state warning on `state_write`
- compacted-prefix ownership limited to the base cache in ISWA configurations
- compacted `V` sidecar stored in canonical `[head][token][embd]` order only; transpose-aware execution remains a later phase

**Non-goals**

- no end-to-end compaction execution yet
- no save / restore serialization yet
- no server/runtime enablement yet
- no flash path
- no quantized K
- no quantized V
- no integration of compacted-prefix positions into live `seq_pos_min` / `seq_pos_max` yet

**Tests**

- layout tests
- metadata tests
- position bookkeeping tests
- sequence-op integration tests
- memory accounting tests
- self-copy and partial-range `seq_cp` tests
- payload integrity tests for `k_data`, `beta_data`, and `v_data`
- guardrail tests for invalid divisors, negative shifts, duplicate positions, zero-layout stores, and unsupported quantized cache types

**Merge gate**

- representation is isolated
- memory accounting is test-covered
- compacted-prefix sequence ops preserve payload integrity on the supported scalar cache types
- state restore cannot leave stale compacted-prefix state behind
- compacted-prefix ownership is explicit for ISWA
- no flat per-slot `beta` design remains in the plan

## PR-3: Non-Flash Correctness Path

**Objective**

Make compaction work end-to-end on the narrow supported matrix.

**Scope**

- query extraction from prefill path
- GQA regrouping into KV-head query space
- key selection fast path (`topk` by attention score)
- NNLS beta fitting
- OLS value fitting:
  - primary: Householder QR
  - fallback: regularized Cholesky with `lambda=1e-6`
- chat-template / BOS preservation
- optional model-specific nonuniform schedules with uniform fallback
- non-flash execution using additive `kq_b`
- explicit fallback for unsupported configs

**Solver policy**

All solver math runs in fp32. Results are cast to the model dtype only for KV storage.

**Tests**

- solver unit tests
- reference parity on fixed fixtures
- end-to-end sanity tests
- unsupported-config fallback tests
- no-NaN / no-crash tests

**Merge gate**

- supported matrix works end-to-end within defined tolerances
- unsupported matrix fails or falls back explicitly

## PR-4: Session And State Integration

**Objective**

Make compacted state usable in real session lifecycles.

**Scope**

- compact / decompact lifecycle
- save / restore handling
- invalidation rules
- repeated compaction cycles
- session continuation behavior

**Tests**

- save/load roundtrip
- failed-restore fallback
- repeated cycles
- session continuation

## PR-5: Real Performance Path

**Objective**

Turn logical compaction into measurable product wins.

**Scope**

- packed layout so reduced active range lowers real compute
- real memory reuse / release where possible
- repeated-turn optimization
- benchmark harness integration

Important note:
- PR-5 is the phase that turns logical KV reduction into real throughput claims,
- PR-3 correctness alone must not be described as a speed win.

**Merge gate**

Measured progress on Goal 1 and/or Goal 2 on at least one ModelAI workload.

## PR-6: Coverage Expansion

**Objective**

Broaden support after the narrow path works.

**Scope**

- flash-compatible path with real `kq_b` support
- quantized V
- SWA evaluation
- hybrid-memory evaluation
- improved query-generation paths such as self-study and OMP

Each added path must have isolated tests and benchmark evidence.

## Benchmark Workloads

| ID | Workload | Purpose |
|---|---|---|
| W1 | 80K filing -> first answer | cold long-context behavior |
| W2 | 80K filing -> 20 follow-up questions | repeated-turn speed |
| W3 | executive summary generation | standard analyst workflow |
| W4 | full research report generation | heavy multi-step workflow |
| W5 | 3 concurrent sessions on 32GB | local scalability |
| W6 | save/restore + continue | session continuity |

## Per-Run Metrics

1. model name and quantization
2. backend (Metal / CUDA / CPU)
3. flash attention on/off
4. compaction on/off and ratio
5. prefill latency ms
6. first-token latency ms
7. decode throughput tok/s
8. allocated KV bytes
9. active KV length
10. quality delta vs full cache
11. `query_generation_time_ms`
12. `solver_time_ms`

## Regression Rules

- performance regression > 10% on a key metric requires investigation before release
- quality regression > 1% perplexity delta requires investigation before release
- no benchmark claim without measured outputs

## Related Governance Documents

- `docs/modelai-git-policy.md`
- `docs/modelai-ci-policy.md`
- `docs/modelai-release-checklist.md`
