# KV Compaction Integration

## File Map

### Solver and query pipeline

- `src/llama-kv-compact-math.h`: small matrix utilities used by the compaction solver.
- `src/llama-kv-compact-query.h/.cpp`: cache-key surrogate query extraction and live K access helpers.
- `src/llama-kv-compact-select.h/.cpp`: top-k and OMP token selection.
- `src/llama-kv-compact-solver.h/.cpp`: beta NNLS and V least-squares fitting.
- `src/llama-kv-compact-pipeline.h/.cpp`: orchestration for `fit`, `select`, and `omp` pipelines.
- `src/llama-kv-compact-self-study.h/.cpp`: self-study query capture, autoregressive continuation, regrouping, and solver entry.

### Persistent compacted-prefix state

- `src/llama-kv-compacted-prefix.h/.cpp`: per-sequence compacted-prefix store, layer layouts, serialization, sequence lifecycle.
- `src/llama-kv-compacted-prefix-exec.h/.cpp`: execution gating and host-side materialization helpers.

### Runtime integration

- `src/llama-kv-cache.h/.cpp`: cache-level orchestration, live K/V accessors, execution enablement, reclaim, pipeline wrappers.
- `src/llama-kv-cache-iswa.h/.cpp`: iSWA delegation. Base-cache compaction is supported; SWA-layer compaction is not.
- `src/llama-graph.h/.cpp`: graph construction, compacted K/V/mask/beta injection, flash/non-flash selection.
- `src/llama-context.h/.cpp`: context-level eval callback support used by self-study Q capture.
- `include/llama.h`: public callbacks and memory interfaces used by the internal pipeline.

## Architecture Support Matrix

| Memory Layout | KV Cache Extraction | Compaction Status |
|---------------|-------------------|-------------------|
| `llama_kv_cache` (standard) | Direct | Supported |
| `llama_kv_cache_iswa` (iSWA) | `get_base()` non-SWA cache | Supported (base layers only) |
| `llama_memory_hybrid` (SSM+attention) | `get_mem_attn()` | Supported if standard RoPE |
| `llama_memory_hybrid_iswa` (SSM+iSWA) | `get_mem_attn()->get_base()` | Supported if standard RoPE |
| `llama_memory_recurrent` (pure SSM) | No KV cache | N/A |

## Store to KV Cache Integration

The compacted-prefix store is owned by `llama_kv_cache`. Each sequence may have:

- compacted logical positions
- per-layer compacted payloads
- execution enablement state
- serialized state for save/restore

`llama_kv_cache` provides the orchestration layer for:

- building compacted prefixes from live KV
- enabling or disabling compacted execution
- reclaiming live prefix KV after compaction
- exposing read-only K/V accessors used by the solver

On iSWA layouts, only the base cache participates in compaction.

## Graph Construction

`build_attn()` injects compacted-prefix tensors alongside the live KV path.

The flow is:

1. resolve whether compacted execution is enabled for the ubatch
2. materialize compacted `K`, `V`, `beta`, and causal mask columns
3. prepend those tensors to the live KV tensors
4. execute attention with the combined sources

The compacted `V` path is stored canonically and permuted when the live cache uses a transposed V layout.

## `set_input_*()` Data Flow

Materialization helpers fill graph input tensors from store-backed payloads:

- compacted `K`
- compacted `V`
- expanded `beta`
- compacted causal mask columns

The helpers are shape-aware and operate on the graph tensors created for the current ubatch. Quantized transposed V extraction uses `ggml_row_size()` offsets so quantized rows are addressed correctly.

## Execution Gating

Compacted-prefix execution requires all of the following:

- supported model architecture
- supported memory layout
- single logical sequence in the ubatch
- 1D positional semantics
- compacted prefix configured and enabled for the sequence

Compacted execution is explicitly blocked for:

- M-RoPE models
- SWA-layer compaction
- hybrid recurrent-attention models
- unsupported graph/memory layouts

## Pipeline Modes

### `select`

- top-k selection only
- zero beta
- compacted V copied from selected live rows
- flash attention compatible

### `fit`

- cache-key surrogate queries
- top-k selection + beta/V fitting
- non-zero beta
- requires non-flash execution path

### `omp`

- OMP selection + NNLS refits
- non-zero beta
- requires non-flash execution path

### `self-study`

- autoregressive continuation with Q capture via `cb_eval`
- regrouping from Q heads to KV heads
- optional subsampling
- same solver stages as full fit mode

## Internal API Surface

The compaction interfaces remain internal to `src/`. Important entry points include:

- pipeline wrappers on `llama_kv_cache`
- compacted-prefix store configuration and clear methods
- execution enablement and reclaim helpers
- self-study generation and capture helpers

None of these are presented as stable public APIs.

## Configuration

The runtime uses internal config structures for:

- solver regularization and iteration settings
- selection mode
- self-study generation/capture settings
- pipeline stats collection

All solver math is fp32. Fitted results are cast to the model/cache dtype for storage.

## Compile-Time Guards

KV compaction can be disabled at build time:

```bash
cmake -B build -DLLAMA_KV_COMPACTION=OFF
```

When `OFF`:
- All compaction source files are excluded from compilation
- Public API functions (`llama_kv_cache_compact`, `llama_kv_cache_set_auto_compact`) become no-op stubs that log a warning and return -1
- Internal compaction members and methods in `llama_kv_cache` are excluded via `#ifdef LLAMA_KV_COMPACTION`
- The `LLAMA_KV_COMPACTION` define is propagated as a PUBLIC compile definition to all targets linking against llama

Default is `ON` — existing build behavior is unchanged.

## Shared Utilities

- `src/llama-kv-compact-utils.h` — Centralized `llama_kv_compact_get_cache()` for extracting `llama_kv_cache *` from any memory backend (plain, iSWA, hybrid, hybrid-iSWA). Single point of truth for the type-dispatch logic.
- `src/llama-kv-compact-shared.h` — Shared `static inline` math helpers (`gather_matrix_rows`, `write_compacted_payload`, env var skip helpers) used across pipeline, self-study, and prefill-q modules.
