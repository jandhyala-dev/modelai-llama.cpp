# Model Defaults — Shared Configuration

## Overview

Per-model KV allocation and compaction caps are defined in:

```
~/dev/whippet/models/model-defaults.json
```

This file is the single source of truth for all systems (cot, ModelAI, benchmarks) that launch models via `modelai-llama.cpp`.

## Config Format

```json
{
  "Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf": {
    "display_name": "Qwen3 Coder 30B MoE (1M)",
    "kv_alloc": 65536,
    "compact_cap": 262144,
    "compact_method": "select",
    "compact_ratio": 4,
    "native_ctx": 262144
  }
}
```

## Field Definitions

| Field | Type | Description |
|---|---|---|
| `kv_alloc` | int | KV cache allocation. Passed as `--ctx-size` to llama-server |
| `compact_cap` | int or null | Max effective tokens achievable via iterative compaction. Position IDs must not exceed this. null = no compaction supported |
| `compact_method` | string or null | Compaction pipeline method ("select" for V1). null = disabled |
| `compact_ratio` | int or null | Compaction ratio per cycle (4 = 4x compression per cycle) |
| `native_ctx` | int | Model's native context limit from GGUF metadata (RoPE ceiling) |

## How KV alloc and compact cap interact

```
KV alloc (64K)     = how much KV cache RAM to allocate
compact_cap (256K) = how far position IDs can go via iterative compaction
native_ctx (1M)    = model's absolute RoPE limit (not necessarily tested)

Fill 64K → compact at 4x → 16K used + 48K free → fill again → repeat
Each cycle advances position IDs by ~48K
Stop when position IDs reach compact_cap
```

## Current Model Table

| Model | kv_alloc | compact_cap | native_ctx | tok/s |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B-Instruct-1M | 64K | 256K | 1M | 52 |
| Qwen3-30B-A3B-Instruct-2507 | 64K | 256K | 256K | 52 |
| Qwen3-30B-A3B-Thinking-2507 | 64K | 256K | 256K | 51 |
| Qwen3.5-35B-A3B | 64K | 131K | 131K | 23 |
| Qwen3-14B-128K | 64K | 128K | 128K | 18 |
| Qwen3-8B-128K | 64K | 128K | 128K | 31 |
| gemma-3-4b-it | 8K | N/A | 8K | — |
| qwen2.5-14b-instruct | 32K | 32K | 32K | 12 |
| qwen2.5-7b-instruct | 32K | 32K | 32K | 26 |
| bge-m3-f16 | 8K | N/A | 8K | — |

## Consumer Responsibilities

### cot CLI
- Read `model-defaults.json` in `InferenceServer.__init__()` (lifecycle.py)
- Use `kv_alloc` as `--ctx-size` unless user passes explicit `--ctx`
- Enforce `compact_cap` as position ID ceiling during iterative compaction
- See: `cot/docs/implementation/model-defaults-integration.md`

### ModelAI
- Read `model-defaults.json` in `llama-supervisor.cjs._ensureReadyImpl()`
- Override hardware-detect context with `kv_alloc` for listed models
- Enforce `compact_cap` in llamacpp-adapter compaction requests
- See: `ModelAI/docs/Engineering/MODEL_DEFAULTS_INTEGRATION.md`

### Benchmark scripts
- Read `model-defaults.json` to get correct `--ctx-size` per model
- Use `compact_cap` to set iteration limits in compaction benchmarks

## Adding a New Model

1. Download the GGUF to `~/dev/whippet/models/`
2. Add an entry to `model-defaults.json` with the filename as key
3. Set `kv_alloc` based on RAM budget (64K for 32GB machines)
4. Set `compact_cap` to `min(native_ctx, 262144)` — 256K proven safe
5. Set `native_ctx` from model's HuggingFace card or GGUF metadata
