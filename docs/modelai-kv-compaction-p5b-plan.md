# KV Compaction — P5b Solver Pipeline Plan

## Purpose

`P5b` is the first branch that is allowed to claim paper-aligned KV compaction.

`P5a` proved a narrower runtime slice:
- live-prefix retirement,
- dense live-suffix repack,
- reduced runtime-visible `active_n_kv`,
- a model-backed regression that the active range shrinks,
- a manual perf harness showing the repacked path can improve repeated-turn decode throughput.

`P5a` did **not** implement the mathematical compression pipeline from the MIT Attention Matching paper.

`P5b` closes that gap by computing compacted-prefix payloads from the original KV cache rather than manually configuring the store.

## Required P5b Deliverables

The branch is only complete if all seven conditions below are true end to end.

1. real query extraction exists
2. real key selection exists
3. real NNLS `beta` fitting exists
4. real least-squares `V` fitting exists
5. compacted-prefix payloads are solver-populated from the original KV cache
6. quality is regression-tested on fixed tolerances
7. a real ModelAI-like workload proves Goal 1 and Goal 2

A branch that lacks any of the above must not be labeled as final `P5` completion.

## Existing Infrastructure Reused

`P5b` builds on existing branch work:
- `P2`: compacted-prefix store and payload layout
- `P3`: non-flash compacted-prefix execution path
- `P4`: compacted-prefix save / restore lifecycle
- `P5a`: live-prefix reclamation, active-range reduction, and perf harness

These pieces are content-agnostic. `P5b` supplies solver-computed contents for the existing store.

## Impacted Files Beyond The New Modules

`P5b` is not only four new source files.
It also requires small integration changes in:
- `src/CMakeLists.txt`
- `tests/CMakeLists.txt`
- `src/llama-kv-cache.h`
- `src/llama-kv-cache.cpp`
- `docs/modelai-kv-compaction-plan.md`
- `docs/modelai-fork-summary.md`

## File-Level Implementation Plan

### 1. Solver core

New files:
- `src/llama-kv-compact-solver.h`
- `src/llama-kv-compact-solver.cpp`

Required contents:
- fp32-only solver math
- NNLS beta fitting with:
  - `lstsq` primary path
  - regularized Cholesky fallback with `lambda = 1e-6`
  - positivity clamp and `beta = log(B)` conversion
- least-squares V fitting with:
  - Householder QR primary path
  - regularized Cholesky fallback
- ridge regularization policy documented in code and docs

Output targets:
- `layer_storage::beta_data`
- `layer_storage::v_data`

### 2. Key selection

New files:
- `src/llama-kv-compact-select.h`
- `src/llama-kv-compact-select.cpp`

Required contents:
- top-k selection first
- OMP added in the same module after top-k is working
- per-layer / per-KV-head selected index output
- direct mapping from selected positions into:
  - `compacted_prefix_configure()`
  - `layer_storage::k_data`

Output targets:
- compacted key positions
- compacted key payloads (`C_k`)

### 3. Query extraction

New files:
- `src/llama-kv-compact-query.h`
- `src/llama-kv-compact-query.cpp`

Required first path:
- cache-keys query extraction

Why first:
- simplest viable method
- avoids adding new model-forward instrumentation before the pipeline works

Required behavior:
- extract fp32 per-layer / per-KV-head query matrices from live cache-backed tensors
- support GQA regrouping when queries originate in attention-head space later

Deferred but planned:
- self-study queries
- context-prefill queries

### 4. End-to-end pipeline orchestration

New files:
- `src/llama-kv-compact-pipeline.h`
- `src/llama-kv-compact-pipeline.cpp`

Required contents:
- one driver that performs:
  1. query extraction
  2. key selection
  3. beta fitting
  4. V fitting
  5. store population into the existing compacted-prefix representation
- narrow supported matrix only:
  - standard `llama_kv_cache`
  - non-flash only
  - non-quantized V
  - single-sequence first

Output targets:
- `compacted_prefix_configure()`
- `layer_storage::k_data`
- `layer_storage::beta_data`
- `layer_storage::v_data`

### 5. Quality and benchmark validation

New file:
- `tests/test-kv-compact-quality.cpp`

Required coverage:
- compacted-prefix payloads are solver-derived, not synthetic
- attention-output error stays within an explicit tolerance on fixed fixtures
- quality delta is reported with exact metrics

Required benchmark path:
- extend the current perf harness or add a ModelAI-oriented harness that measures:
  - prefill time
  - follow-up decode tok/s
  - active `n_kv`
  - quality delta
- workload must be closer to ModelAI usage than `stories15M-q4_0`

### 6. OMP expansion

Location:
- `src/llama-kv-compact-select.cpp`

Required contents:
- OMP-class selection after the top-k baseline is working
- explicit comparison against the top-k baseline on the same quality/perf harness

## Recommended Implementation Order

1. `llama-kv-compact-solver.*`
2. `llama-kv-compact-select.*`
3. `llama-kv-compact-query.*`
4. `llama-kv-compact-pipeline.*`
5. `tests/test-kv-compact-quality.cpp`
6. OMP expansion inside `llama-kv-compact-select.*`

This order is intentional.
It gets a solver-complete baseline running sooner because the math is self-contained and testable before full wiring.

## Non-Goals For First P5b Pass

The first `P5b` pass should not expand scope into:
- flash-attention support
- quantized-V support
- public/server enablement
- broad multi-sequence routing
- SWA or hybrid-memory support
- physical KV buffer reallocation/release

Those remain later work.

## Merge Gate

`P5b` is only ready for merge if:
1. all seven P5b deliverables are implemented,
2. compacted-prefix payloads are solver-derived from the original KV cache,
3. the quality test passes on fixed tolerances,
4. the benchmark proof uses a real ModelAI-like workload,
5. the docs do not overclaim beyond measured evidence.

## Relationship To The MIT Paper

This branch should be reviewed directly against:
- the MIT paper PDF,
- the MIT reference implementation,
- the original upstream `llama.cpp` runtime constraints.

The success bar is not “good local infrastructure.”
The success bar is “paper-aligned mathematical compaction integrated into the runtime with measured product value.”
