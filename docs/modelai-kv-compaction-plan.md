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
| PR-5a | `kv-compact-pr5-performance` | Runtime reclaim and perf slice |
| PR-5b | `kv-compact-pr5b-solver-pipeline` | Solver-derived compaction pipeline |
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

Make compacted-prefix state participate in real non-flash attention execution on the narrow supported matrix.

**Scope**

- explicit internal execution activation per sequence
- compacted-prefix execution eligibility resolution per ubatch
- explicit runtime rejection for:
  - flash-attention execution
  - SWA / split-memory execution
  - hybrid-memory execution
- host-side materialization helpers for:
  - compacted `K`
  - canonical non-transposed compacted `V`
  - per-query-head expanded `beta`
  - compacted prefix mask columns
- non-flash execution using additive `kq_b`
- concatenation of compacted-prefix `K/V/B/mask` with the live KV path inside `build_attn`
- graph reuse disabled while compacted-prefix execution is active
- explicit fallback for unsupported configs or inactive compacted-prefix state
- deterministic correctness tests for:
  - execution gating
  - payload materialization
  - causal masking
  - non-flash attention sanity
- regression coverage to ensure existing state-restore behavior is not broken by the new internal execution path

**Solver policy**

The full fitting pipeline is intentionally deferred. When NNLS / least-squares fitting lands, all solver math will run in fp32 and results will be cast to model dtype only for KV storage.

**Tests**

- execution-gating tests
- materialization tests for compacted `K/V/B/mask`
- alibi-mask tests
- non-flash attention sanity tests
- unsupported-config fallback tests
- execution-state lifecycle tests
- no-NaN / no-crash tests
- state-restore regression tests

**Merge gate**

- compacted-prefix state can participate in non-flash attention execution on the supported matrix
- unsupported matrix falls back explicitly
- graph-path wiring is real, but public runtime enablement remains deferred
- flash-attention use while compacted-prefix execution is active fails explicitly instead of relying on implicit `kq_b` behavior

**Explicit PR-3 deferrals**

The following items are intentionally not part of this branch and must be addressed in follow-on work before public rollout:
- query extraction from the prefill path
- GQA regrouping from projected query heads into KV-head fitting space
- key selection policies (`topk`, OMP, or nonuniform schedules)
- NNLS beta fitting
- least-squares value fitting:
  - primary: `lstsq`-equivalent dense fp32 least-squares
  - fallback: regularized Cholesky with `lambda=1e-6`
- chat-template / BOS preservation policy
- public runtime/server enablement
- save/restore serialization of compacted-prefix execution state
- model-provided `kq_b` tensors that rely on broadcast token dimensions; P3 requires exact non-concat dimensions for `kq_b` concatenation

## PR-4: Session And State Integration

**Objective**

Make compacted state usable in real session lifecycles.

**Scope**

- compact / decompact lifecycle
- save / restore handling
- versioned session / sequence-state format updates for compacted-prefix payloads
- compacted-prefix payload serialization inside `llama_kv_cache::state_write/state_read`
- per-sequence and full-store compacted-prefix roundtrip support
- stale compacted-prefix invalidation before restore
- restore-time validation of layer layout, payload sizes, and execution-state invariants
- atomic rollback if compacted-prefix restore fails after live KV state has already been loaded
- invalidation rules
- repeated compaction cycles
- session continuation behavior

**Non-goals**

- no new public/server enablement yet
- no performance claims from restored compacted-prefix execution
- no query extraction or fitting pipeline work
- no flash / quantized-V execution support
- no requirement that post-restore continuation decode run with compacted-prefix execution still enabled

**Tests**

- compacted-prefix store serialization roundtrip
- public `llama_state_seq_get_data` / `llama_state_seq_set_data` roundtrip for compacted-prefix state
- failed-restore rollback through the public sequence-state API
- repeated cycles
- session continuation after restore
- regression coverage for existing fragmented KV restore behavior

**Merge gate**

- compacted-prefix state survives public save/restore APIs for sequence and full-store paths
- stale compacted-prefix state cannot survive a restore attempt
- restore validates layer layout and payload sizes instead of silently accepting mismatches
- restore failure cannot leave partially restored live-KV state behind
- session continuation remains safe after restore
- file-format changes are versioned so old state files fail cleanly instead of mis-parsing

## PR-5a: Runtime Reclaim And Perf Slice

**Objective**

Turn logical compaction into measurable product wins.

**Scope**

- retire the live prefix once an equivalent compacted prefix exists
- densely repack the retained live suffix to the front of the live KV cache
- lower the runtime-visible active `n_kv` range without changing the fixed buffer allocation
- expose an internal `active_n_kv` query for model-backed verification
- add model-backed regression coverage proving the active range crosses the 256-pad boundary after reclaim
- add a manual benchmark harness that reports before/after `active_n_kv` and decode tok/s for the same compacted execution slice

Important note:
- PR-5 is the phase that turns logical KV reduction into real throughput claims,
- PR-3 correctness alone must not be described as a speed win.
- the first landed P5 slice targets active-range reduction and repeated-turn throughput on the existing fixed KV allocation;
  true physical KV buffer release remains a later extension.

**Merge gate**

Measured progress on Goal 1 and/or Goal 2 on at least one supported workload, with:
- a model-backed regression proving `active_n_kv` shrinks after reclaim, and
- attached benchmark output from the manual compacted-prefix perf harness.

## PR-5b: Solver-Derived Compaction Pipeline

**Objective**

Turn the compacted-prefix store from a manually populated container into a solver-derived representation of the original KV cache.

**Scope**

- real query extraction
- real key selection (`top-k` first, OMP after the baseline is working)
- real NNLS `beta` fitting
- real least-squares `V` fitting
- end-to-end pipeline that writes solver outputs into the existing compacted-prefix store
- quality validation on fixed tolerances
- benchmark proof on a real ModelAI-like workload

**Correctness assumptions**

- the first query-extraction path uses cache keys as surrogate queries
- that baseline is valid only because live cached `K` tensors are already RoPE-applied, so the solver’s query and key sides remain in the same rotated space
- any later pre-RoPE or self-study query source must apply matching RoPE and GQA regrouping before fitting

**Solver dependency and precision policy**

- no LAPACK dependency is allowed for `PR-5b`
- the solver must be implemented as pure dense C++ because the matrix sizes are small enough for an internal fp32 path
- query extraction must upcast runtime K/V data to fp32
- all fitting math runs in fp32
- fitted `C_k` and `C_v` are cast back to store dtype only when written into compacted storage
- `beta` remains fp32 throughout

Primary implementation files:
- `src/llama-kv-compact-solver.h/.cpp`
- `src/llama-kv-compact-select.h/.cpp`
- `src/llama-kv-compact-query.h/.cpp`
- `src/llama-kv-compact-pipeline.h/.cpp`
- `tests/test-kv-compact-quality.cpp`

Also impacted:
- `src/CMakeLists.txt`
- `tests/CMakeLists.txt`
- `src/llama-kv-cache.h/.cpp`
- `docs/modelai-kv-compaction-plan.md`
- `docs/modelai-fork-summary.md`

Required internal read-only accessors in `src/llama-kv-cache.h/.cpp`:
- `compacted_prefix_copy_k_head_f32(...)`
- `compacted_prefix_copy_v_head_f32(...)`
- `compacted_prefix_layer_layout_for_solver(...)`
- `compacted_prefix_seq_positions(...)`

These accessors stay internal to `src/` and must not become part of the public `include/llama.h` API.

Required solver-input behavior:
- live `K` extraction is row-major but must be upcast to fp32
- live `V` extraction must handle `v_trans` correctly and de-transpose to canonical token-major fp32 matrices before fitting

Minimum quality metrics and thresholds:
- attention-output cosine similarity `>= 0.95`
- continuation-logit cosine similarity `>= 0.95`
- partition-sum relative error must be emitted

Minimum benchmark workload for merge:
- model size `>= 1B`
- real-text prefix `>= 2048` tokens
- workload `W2` or `W3`
- quality thresholds satisfied on the same run

Important note:
- `PR-5a` proves that reducing runtime-visible active KV range can improve repeated-turn throughput,
- `PR-5b` is the first branch allowed to claim paper-aligned KV compression because it computes compacted payloads from the original KV cache.
- steps 1-3 of the `PR-5b` implementation order are only unit-testable with synthetic matrices until the pipeline orchestration step exists; merge confidence requires the model-backed path, not synthetic math tests alone

**Quality gate**

Minimum `PR-5b` quality metrics:
- compacted-vs-full attention-output cosine similarity `>= 0.95`
- compacted-vs-full continuation-logit cosine similarity `>= 0.95`
- partition-sum relative error must be reported explicitly

Minimum benchmark workload for merge:
- model size `>= 1B`
- real-text prefix `>= 2048` tokens
- workload `W2` or `W3`
- quality thresholds satisfied on the same run

**Non-goals**

- chat-template / BOS / uncompacted system-prefix policy changes remain out of scope for the first `PR-5b` pass
- V-transpose layout optimizations remain out of scope; the first solver pass may de-transpose live `V` into canonical fp32 rows for fitting

**Merge gate**

All seven paper-aligned deliverables must be complete:
1. query extraction
2. key selection
3. NNLS beta fitting
4. least-squares V fitting
5. solver-populated compacted payloads
6. quality regression coverage
7. benchmark proof on a real ModelAI workload

A branch that lacks any of the above must not be labeled as final `P5` completion.

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
