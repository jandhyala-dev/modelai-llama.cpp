# ModelAI llama.cpp Fork — Summary

## Purpose

This repository is a private product fork of `ggml-org/llama.cpp` maintained by COT Labs / ModelAI.

It exists to:
1. implement KV cache compaction via Attention Matching for ModelAI’s long-session workloads,
2. make `llama-server` the primary local runtime path for ModelAI,
3. keep release control, CI policy, and benchmarking under direct product ownership.

This is a product fork first. Optional upstreaming is a later track, not the current delivery constraint.

## Fork Goals

### Goal 1: Long-Session Efficiency

ModelAI needs to keep long documents and long analyst sessions usable without forcing the model to attend over the full original prefix forever.

This fork targets that by:
- storing old immutable context as a compacted prefix,
- keeping recent mutable context as a live suffix,
- preserving chat-template / BOS / system-prefix tokens outside the compacted block,
- exposing allocated KV bytes and active KV length as separate runtime signals.

### Goal 2: Post-Prefill Speed / Decode Throughput

ModelAI gets value only if repeated follow-up turns become cheaper after large prefills.

This fork targets that by:
- compacting only old immutable prefixes,
- lowering the effective active KV range on repeated turns,
- packing active cells into a backend-usable layout so logical KV reduction becomes real compute reduction.

Reducing active `n_kv` alone is not enough. It improves throughput only if the active cells are packed low enough for the graph/backend to exploit a shorter effective range.

Important v0 caveat:
- the first compaction-capable path is expected to be non-flash first,
- it may be architecturally correct before it is faster than current defaults on every workload,
- broad performance claims belong only after packed-layout and backend-specific measurements exist.

## What The MIT Paper Contributes

**Paper:** `Fast KV Compaction via Attention Matching`
**Reference code:** `https://github.com/adamzweiger/compaction`
**Upstream tracking issue:** `ggml-org/llama.cpp#20037`

Given an original KV prefix of length `T`, the paper builds a compacted representation of length `t << T`:

1. generate reference queries from the prefill path,
2. select compacted keys (`topk` or more advanced selection such as OMP),
3. solve for `beta` with NNLS to preserve attention mass,
4. solve for compacted values with least-squares to match attention output,
5. evaluate future queries against the compacted prefix instead of the original full prefix.

Modified attention becomes:

`softmax(q @ C_k^T + beta) @ C_v`

Key algorithm facts:
- approximate, not exact,
- `beta` is essential,
- `beta` is per layer / per KV head / per compacted token,
- nonuniform budgets matter at higher compression ratios,
- chat-template / BOS / uncompacted prefix handling is required in a production runtime.

## What The Paper Does Not Give Us For Free

The paper does not solve the main `llama.cpp` integration problems for us:

1. it does not guarantee exact equivalence for arbitrary future queries,
2. it does not produce automatic physical VRAM reduction inside the current fixed-size KV layout,
3. it does not make flash attention, quantized V, SWA, or hybrid memory paths work automatically,
4. it does not solve runtime state/save-restore integration,
5. it does not guarantee throughput gains unless the active memory layout is repacked for the backend.

## Current llama.cpp Constraints

The current runtime constraints that shape this fork are:

1. `llama_memory_t` is the real memory abstraction boundary.
2. KV buffers are fixed-size allocations, so logical reduction does not automatically mean physical memory savings.
3. The non-flash path supports additive `kq_b`; the flash-attention path does not.
4. Quantized V cache currently depends on flash attention.
5. Position continuity and state-restore invariants are strict.
6. Multiple memory types exist:
   - standard `llama_kv_cache`
   - SWA / split-memory cache
   - hybrid recurrent + attention memory
7. `beta` cannot be represented as a flat slot scalar; its dimensionality is per layer / per KV head / per compacted token.
8. PR-1 telemetry currently reaches `memory_breakdown()` through `src/llama-context.h`; that is acceptable for this product fork but it is an explicit upstream-sync risk until a public API exists.

## Expected Results

### Near-Term

- private product-fork docs and governance are in place,
- CI and release discipline exist before engine dependency promotion,
- capability and telemetry surfaces exist before compaction itself lands,
- `llama-server` exposes a stable ModelAI-facing contract on `/props`, `/models`, and `/metrics` before any compacted-prefix implementation is enabled.

Near-term observability requirements:
- allocated context/model/compute bytes are reported separately,
- active KV metrics are reported even for idle slots as zero-safe telemetry,
- compaction timings remain `null` until the compacted-prefix path exists,
- `/props` must remain safe to query while the server is sleeping and must not wake the runtime just to answer capability questions,
- `modelai.contract` metadata and `llamacpp:modelai_*` metrics must be present so ModelAI can negotiate engine version and scrape runtime telemetry deterministically.

### Mid-Term

- PR-2 establishes an internal compacted-prefix store inside `llama_kv_cache` with:
  - per-sequence logical position bookkeeping,
  - per-layer / per-KV-head `(C_k, beta, C_v)` storage shape,
  - forwarding of core sequence ops (`seq_rm`, `seq_cp`, `seq_keep`, `seq_add`, `seq_div`),
  - compacted-prefix bytes folded into context memory accounting as host-side sidecar memory,
  - scalar compacted cache types only in P2 (`F16`, `BF16`, `F32`),
  - stale compacted-prefix state invalidated on `state_read`, and warned as non-serialized on `state_write`,
  - compacted-prefix ownership restricted to the base cache in ISWA layouts,
  - compacted `V` sidecar storage remains logical `[head][token][embd]`; any transpose-sensitive execution compatibility is deferred to PR-3,
- no execution path, serialization lifecycle, or public runtime enablement is part of PR-2,
- compacted-prefix logical positions are not yet merged into the live KV cache `seq_pos_min/seq_pos_max` view before PR-3,
- PR-3 lands the first internal non-flash execution slice:
  - explicit per-sequence compacted-prefix execution gating,
  - compacted-prefix execution eligibility limited to single-sequence, 1D-position, standard `llama_kv_cache` batches,
  - explicit runtime rejection for flash-attention, SWA / split-memory, and hybrid-memory execution,
  - host-side materialization helpers for compacted `K`, canonical non-transposed `V`, per-query-head expanded `beta`, and prefix mask columns,
  - non-flash attention graph wiring that prepends compacted prefix `K/V/B/mask` to the live KV path,
  - graph reuse disabled while the compacted-prefix execution path is active,
  - deterministic P3 tests for execution gating, payload materialization, causal/alibi masking, execution-state lifecycle, and non-flash attention sanity,
- PR-3 deliberately does not yet include query extraction, NNLS/OLS fitting, save/restore serialization, or public runtime enablement,
- model-provided `kq_b` tensors that rely on broadcast token dimensions remain outside the P3 supported matrix and fail explicitly,
- PR-4 adds versioned compacted-prefix save/restore integration for both full-context and per-sequence state paths:
  - compacted-prefix payloads are serialized inside the KV state stream,
  - restore clears stale compacted-prefix state before loading,
  - restore validates layer layout and payload sizes before accepting data,
  - restore rollback now clears partially loaded live/compacted state if compacted-prefix deserialization fails,
  - state file versions are bumped so old files fail cleanly instead of being mis-parsed,
  - public save/restore regression tests now cover compacted-prefix roundtrip, failed-restore rollback, and continuation behavior,
- PR-4 still keeps public runtime enablement and post-restore execution/performance claims out of scope,
- PR-5 begins the real performance path with a narrow but measurable slice:
  - once a sequence has a configured compacted prefix, the live prefix rows before `live_suffix_pos0` can be retired,
  - retained live suffix rows are repacked densely to the front of the live KV cache,
  - `active_n_kv` is reduced without changing the underlying fixed KV allocation,
  - model-backed regression coverage now proves the runtime-visible active range drops after reclaim,
  - a manual compacted-prefix perf harness reports before/after `active_n_kv` and decode tok/s for the same compacted execution slice,
  - on the current Apple Silicon debug smoke run with `stories15M-q4_0`, that harness reduced `active_n_kv` from `512` to `256` and improved continuation throughput from `244.1 tok/s` to `309.8 tok/s`; this is a branch validation result, not a general release claim,
- PR-5b is the first solver-complete compaction milestone and is the only branch allowed to claim paper-aligned compression:
  - query extraction must be implemented from real runtime data,
  - key selection must be implemented (`top-k` baseline, OMP as follow-on),
  - NNLS `beta` fitting must populate `beta_data`,
  - least-squares `V` fitting must populate `v_data`,
  - compacted-prefix payloads must be solver-populated from the original KV cache,
  - quality must be regression-tested on fixed tolerances,
  - a real ModelAI-like workload must prove Goal 1 and Goal 2,
- until PR-5b lands, the compacted-prefix store can be executed and reclaimed but its contents should still be treated as infrastructure-populated rather than mathematically derived by the full paper pipeline,
- narrow v0 compaction path on the supported matrix,
- measured long-session improvements on ModelAI workloads,
- measured repeated-turn follow-up improvements on at least one supported workload.
- explicit fallback to the baseline path on unsupported configs.

### Long-Term

- broader backend/model coverage,
- real packed-layout performance work,
- optional sanitized upstream path after the architecture is proven in product use.

## V0 Support Matrix

The matrix below describes the intended v0 execution-path scope for the fork as a whole. PR-2 only lands the internal memory representation and guardrails needed to reach that scope later.

| Category | Status |
|---|---|
| Standard causal models with `llama_kv_cache` | Supported |
| Non-flash attention path | Supported |
| Internal compacted-prefix execution path for single-sequence, 1D-position batches | Supported |
| Scalar K/V cache element types for compacted-prefix sidecar (`F16`, `BF16`, `F32`) | Supported |
| Non-quantized V cache | Supported |
| Uncompacted chat-template / BOS prefix | Supported |
| Uniform budgets (default) | Supported |
| Precomputed nonuniform schedules | Supported where validated |
| Quantized K compaction | Unsupported |
| Flash-attention compaction path | Unsupported |
| Quantized V compaction | Unsupported |
| SWA / split-memory compaction | Unsupported |
| Hybrid recurrent + attention compaction | Unsupported |
| M-RoPE edge cases | Unsupported |
| Public/server compacted-prefix enablement | Unsupported |
| Public API guarantees | Unsupported |

## Runtime Strategy

- `llama-server` is the primary runtime target for ModelAI.
- Ollama becomes optional compatibility fallback, not the architectural center.
- ModelAI owns model management, process lifecycle, telemetry consumption, and engine pinning.

## Product Track vs Upstream Track

### Track A — Private Product Fork

- repo: `jandhyala-dev/modelai-llama.cpp`
- optimized for product iteration speed and measurable ModelAI value,
- AI-assisted implementation is allowed,
- shipping discipline is governed by `modelai-main`, CI, and release tags.

### Track B — Optional Future Upstreaming

- begins only after Track A has measured results and stable architecture,
- requires a separate upstream-ready cleanup pass,
- must satisfy upstream contribution and review expectations independently.

## Governance And Related Documents

- `docs/modelai-kv-compaction-plan.md` — staged PR plan from docs baseline through coverage expansion
- `docs/modelai-git-policy.md` — upstream sync, branch, merge, and release governance
- `docs/modelai-ci-policy.md` — CI jobs, regression thresholds, and artifact rules
- `docs/modelai-release-checklist.md` — release promotion and rollback checklist

Release discipline:
- `modelai-main` is the stable shipping branch,
- CI is mandatory before merge or release promotion,
- releases are tagged only from `modelai-main`,
- ModelAI pins only to exact tags or SHAs,
- each release records upstream base SHA and benchmark deltas.

Current branch protection state:
- `modelai-main` is protected against force-push and deletion,
- `upstream-master` is protected against force-push and deletion,
- CI is the intended merge gate for promoted changes.
