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

## Correctness Assumptions That Must Be Explicit

### RoPE-baked cache-key queries

The first query-extraction path uses cache keys as surrogate queries.
That works only because both sides live in the same rotated space:
- the live K tensors stored in the KV cache are already RoPE-applied,
- the compacted-prefix execution path also consumes RoPE-aligned attention inputs.

This assumption must stay explicit in code and tests.
If later query paths capture pre-RoPE `Q`, they must apply the same RoPE transform and GQA regrouping before entering the solver pipeline.

## Impacted Files Beyond The New Modules

`P5b` is not only four new source files.
It also requires small integration changes in:
- `src/CMakeLists.txt`
- `tests/CMakeLists.txt`
- `src/llama-kv-cache.h`
- `src/llama-kv-cache.cpp`
- `docs/modelai-kv-compaction-plan.md`
- `docs/modelai-fork-summary.md`

These integration changes are not optional.
The new modules cannot reach `layers`, `v_trans`, or `hparams` directly because those are private to `llama_kv_cache`.

Required internal read-only accessors:
- `compacted_prefix_copy_k_head_f32(...)`
- `compacted_prefix_copy_v_head_f32(...)`
- `compacted_prefix_layer_layout_for_solver(...)`
- `compacted_prefix_seq_positions(...)`

The accessor contract must remain internal to `src/` and must not become a public `include/llama.h` API.

## File-Level Implementation Plan

### 1. Solver core

New files:
- `src/llama-kv-compact-solver.h`
- `src/llama-kv-compact-solver.cpp`

Required contents:
- fp32-only solver math
- NNLS beta fitting with:
  - projected / clamped NNLS built on a pure C++ dense least-squares core
  - regularized Cholesky fallback with `lambda = 1e-6`
  - positivity clamp and `beta = log(B)` conversion
- least-squares V fitting with:
  - pure C++ `lstsq`-equivalent dense least-squares path (normal equations or QR)
  - regularized Cholesky fallback
- ridge regularization policy documented in code and docs

Dependency rule:
- do not add a LAPACK dependency for `P5b`
- the solver implementation must be self-contained C++ because the problem sizes are small enough (`t ~= 32-256`, `n_q ~= 128-1024`)

Precision and casting policy:
- query extraction outputs fp32
- all solver math runs in fp32
- selected `C_k` and fitted `C_v` are cast back to the compacted store dtype only when written into `k_data` / `v_data`
- `beta` remains fp32 in storage and execution

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
- first-pass shared selected-position schedule across the compacted sequence
- per-layer / per-KV-head scoring is allowed to contribute to that shared schedule
- direct mapping from selected positions into:
  - `compacted_prefix_configure()`
  - `layer_storage::k_data`

Output targets:
- one shared compacted key position schedule for the sequence
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
- document explicitly that the cache-keys baseline operates in the RoPE-baked key space

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

Required solver-input behavior:
- `K` and `V` must be copied out of the live cache through the new read-only accessors
- transposed live `V` cache must be de-transposed to canonical token-major fp32 matrices before fitting
- the first pass uses one shared selected-position schedule because the current compacted-prefix store exposes a single logical-position array per sequence; solver-populated `K/V/beta` payloads remain per-layer / per-KV-head on top of that shared schedule

### 5. Quality and benchmark validation

New file:
- `tests/test-kv-compact-quality.cpp`

Required coverage:
- compacted-prefix payloads are solver-derived, not synthetic
- attention-output error stays within an explicit tolerance on fixed fixtures
- quality delta is reported with exact metrics

Required quality metrics and initial tolerances:
- compacted-vs-full attention-output cosine similarity: `>= 0.95`
- compacted-vs-full continuation-logit cosine similarity: `>= 0.95`
- partition-sum relative error: must be reported explicitly

The exact thresholds may be tightened after first calibration, but they must be defined before the branch can merge.

Required benchmark path:
- extend the current perf harness or add a ModelAI-oriented harness that measures:
  - prefill time
  - follow-up decode tok/s
  - active `n_kv`
  - quality delta
- automated model-backed regression may use the repo fixture model for deterministic coverage, but branch closure still requires a separate manual run on the real workload gate below
- the minimum accepted “real ModelAI-like workload” for merge is:
  - model size `>= 1B` parameters
  - real-text prefix `>= 2048` tokens
  - workload `W2` or `W3`
  - quality bar satisfied on the same run

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

Test strategy note:
- steps 1-3 are allowed to land with synthetic/unit coverage while the pipeline is still being wired,
- but the branch merge gate is not satisfied until step 4 and step 5 are complete with model-backed quality evidence.

## Deliberate Scope Limits And Deferred Items

### Chunked compaction (#10)

The MIT paper uses chunked compaction for contexts >4K tokens to control solver memory and
numerical precision. The first P5b pass does **not** implement chunking. The full prefix is
processed as a single block. This is acceptable for the v0 target workloads (2K-8K token prefixes).

Chunked compaction is deferred to a follow-on pass after P5b merges. The pipeline orchestrator
(`llama-kv-compact-pipeline.cpp`) is structured so chunk boundaries can be added around the
outer loop without changing the solver internals.

### Ridge scaling strategy (#11)

The solver uses a fixed ridge regularization parameter (`lambda = 1e-6f` initial) with
automatic escalation: if Cholesky decomposition fails, lambda is multiplied by 10 up to 5
attempts. This applies to both `fit_beta` and `fit_values`.

This is a simple fixed-scaling strategy, not spectral or Frobenius normalization. The initial
value was chosen to match the MIT reference implementation's defaults. Spectral scaling is a
follow-on improvement for higher compression ratios or larger problem sizes.

The escalation logic is implemented in:
- `llama-kv-compact-solver.cpp:fit_beta` — retry with lambda × 10 on Cholesky failure
- `llama-kv-compact-solver.cpp:fit_values` — same retry pattern

### Top-k scoring method (#19)

Key selection uses softmax-normalized attention scores accumulated additively across all
layers and heads. The scoring function in `llama-kv-compact-select.cpp` computes:
- per-query softmax over all prefix keys (with numerical stability via max subtraction)
- per-key scores summed across queries
- aggregated across layers and heads via addition (not max or RMS)

This corresponds to total attention mass received by each key position across the full
model. Top-k selects positions receiving the most aggregate attention.

### q_norm handling (#20)

The first query-extraction path uses RoPE-baked cache keys as surrogate queries
(`llama-kv-compact-query.cpp`). No `q_norm` is applied because:
1. Cache keys do not pass through `q_norm` in the original model forward pass.
2. Both solver sides (queries and keys) are in the same post-RoPE space.
3. Applying `q_norm` would introduce an asymmetry not present in the data.

If a future query path captures real pre-attention `Q` tensors, it must apply `q_norm`
if the model architecture uses it. This is documented here as a constraint for that path.

## Non-Goals For First P5b Pass

The first `P5b` pass should not expand scope into:
- flash-attention support
- quantized-V support
- public/server enablement
- broad multi-sequence routing
- SWA or hybrid-memory support
- physical KV buffer reallocation/release
- changing the established policy that chat-template / BOS / uncompacted system-prefix tokens remain outside the compacted block

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
