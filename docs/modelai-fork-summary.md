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

`softmax(q @ C_k^T / sqrt(d) + beta) @ C_v`

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

## Implementation Status

### PR Status

| PR | Scope | Status | Key Commit |
|---|---|---|---|
| PR-0 | Docs baseline and governance | DONE | `25535c9b` |
| PR-1 | Capability flags and observability | DONE | `cf787729` |
| PR-2 | Compacted-prefix memory architecture | DONE | `4f29b389` |
| PR-3 | Non-flash correctness path | DONE | `e1be3dea` |
| PR-4 | Session and state integration | DONE | `30b3c525` |
| PR-5a | Runtime reclaim and perf slice | DONE | `f9f988d6` |
| PR-5b | Solver pipeline (Attention Matching) | DONE | `72d4420e` |
| PR-6 | Coverage expansion (server, self-study, benchmarks) | DONE | `06f178eb`, `c5f2405d` |

### V1 Implementation Plan

The V1 plan covers all remaining work beyond the V0 PRs above: upstream sync, performance bottleneck fixes, 128K context, test/benchmark expansion, and release gates.

See `docs/modelai-v1-implementation-plan.md` (plan commit `2e43e9c8`).

### Expected Results

#### Near-Term (DONE)

- private product-fork docs and governance are in place,
- CI and release discipline exist before engine dependency promotion,
- capability and telemetry surfaces exist before compaction itself lands,
- `llama-server` exposes a stable ModelAI-facing contract on `/props`, `/models`, and `/metrics`,
- narrow v0 compaction path on the supported matrix,
- measured long-session improvements on ModelAI workloads,
- measured repeated-turn follow-up improvements on at least one supported workload,
- explicit fallback to the baseline path on unsupported configs.

#### Long-Term

- broader backend/model coverage,
- real packed-layout performance work,
- optional sanitized upstream path after the architecture is proven in product use.

## V0 Support Matrix

| Category | Status | Conditions |
|---|---|---|
| Standard causal models with `llama_kv_cache` | Supported | — |
| Non-flash attention path | Supported | — |
| Flash-attention compaction path | Supported | Zero-beta only (`kq_b == nullptr`); falls back to standard attention when solver beta is non-zero |
| Internal compacted-prefix execution path for single-sequence, 1D-position batches | Supported | — |
| Scalar K/V cache element types for compacted-prefix sidecar (`F16`, `BF16`, `F32`) | Supported | — |
| Quantized K compaction | Supported | Head dim must be a multiple of the quantization block size (e.g., Q8_0 requires head_dim % 32 == 0); extraction uses `type_to_float` dequantization |
| Quantized V compaction | Supported | Non-transposed V uses per-head dequantization; transposed V uses per-row dequantization with block-aligned kv_size |
| Non-quantized V cache | Supported | — |
| Uncompacted chat-template / BOS prefix | Supported | — |
| Uniform budgets (default) | Supported | — |
| Precomputed nonuniform schedules | Supported where validated | — |
| OMP selection | Supported | Known infeasible at production scale (>23 min for 2x on 14B); use for quality comparison only |
| Self-study queries | Supported | Q-capture + generation + GQA regrouping (PR-6b) |
| Public/server compacted-prefix enablement | Supported | `/props` reports live compaction state via KV cache queries (6b-18, 6b-19) |
| Gemma3-12B | Unsupported | SWA decode broken upstream (0.7 tok/s vs 16.5 Ollama); not a fork regression |
| SWA / split-memory compaction | Unsupported | `compacted_prefix_runtime_supported()` rejects SWA caches (`n_swa > 0`) |
| Hybrid recurrent + attention compaction | Unsupported | Requires `llama_memory_hybrid` (Mamba layers have no KV) |
| M-RoPE edge cases | Unsupported | `compacted_prefix_runtime_supported()` rejects multi-position models (`n_pos_per_embd() > 1`) |
| Public API guarantees | Unsupported | Internal-only; no stable public API contract yet |

### Tested Models

| Model | Quality (2x) | Quality (4x) | Quality (8x) | Status |
|---|---|---|---|---|
| Qwen3-8B | >= 0.95 | >= 0.90 | >= 0.85 | Validated |
| Qwen3-14B | >= 0.95 | >= 0.90 | >= 0.85 | Validated |
| DeepSeek-R1-14B | >= 0.95 | >= 0.90 | >= 0.85 | Validated |
| Qwen3-30B-A3B | >= 0.95 | >= 0.90 | >= 0.85 | Validated |

Quality thresholds: continuation-logit cosine similarity >= 0.95 at 2x, >= 0.90 at 4x, >= 0.85 at 8x.

### Known Limitations

- **B4 GPU solver:** deferred — requires Metal compute shader, out of V1 scope,
- **Flash attention with beta > 0:** blocked — the flash-attention path does not support additive `kq_b`; requires FlashBias or equivalent upstream support,
- **SWA architecture:** compaction is restricted to the base cache only; SWA sub-cache compaction is rejected by `compacted_prefix_runtime_supported()`.

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
- `docs/modelai-v1-implementation-plan.md` — V1 plan: upstream sync, performance, 128K, release gates (commit `2e43e9c8`)
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
