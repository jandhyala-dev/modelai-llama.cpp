# BUG-M01: Q-Capture Memory Exceeds 16 GB Target

**Severity:** Major (M-01)
**Found by:** Reviewer 2 adversarial review (V2 Phases 2+4+5)
**Status:** FIXED (commit TBD)

## Description

V2 self-study config defaults (n_generate=2000, n_rounds=3) produce ~5 GB
Q-capture memory for Qwen3-14B, exceeding the 16 GB MacBook target.

### Memory Calculation

Q-capture stores all layers simultaneously during decode:

```
n_layer × total_tokens × n_head_q × n_embd_head × sizeof(float)
= 40 × (2000 × 3) × 40 × 128 × 4
= 40 × 6000 × 40 × 128 × 4
= 4,915,200,000 bytes ≈ 4.9 GB
```

Combined with model weights (8 GB) + KV cache (1.3 GB) + layer_cache (2.2 GB)
= ~16.4 GB total, exceeding 16 GB MacBook memory.

### Root Cause

The V2 plan §4.3 estimated ~40 MB peak via per-layer Q processing, but this
referred to processing Q data per-layer *after* generation. During generation
itself, all layers capture Q simultaneously because all layers fire during each
decode step. Per-layer generation is not feasible.

## Fix

Added runtime memory budget guard:

1. New config field `max_q_capture_mb` (default: 1024 MB = 1 GB)
2. Before Q-capture generation, computes projected allocation:
   `n_layer * n_head_q * n_embd_head * sizeof(float) * n_generate * n_rounds`
3. If projected exceeds budget, auto-reduces `n_generate` (preserving `n_rounds`
   where possible), falling back to reducing `n_rounds` if per-round tokens
   drops below 100
4. Logs warning when auto-reduction triggers

### Memory After Fix (Qwen3-14B with 1 GB budget)

```
bytes_per_token = 40 × 40 × 128 × 4 = 819,200 bytes ≈ 0.78 MB
max_total_tokens = 1,073,741,824 / 819,200 = 1310
with n_rounds=3: n_generate = 1310 / 3 = 436
Q-capture = 40 × (436×3) × 40 × 128 × 4 ≈ 1.0 GB
Total system: 8 + 1.3 + 1.0 + 2.2 = 12.5 GB (within 16 GB target)
```

## Files Changed

- `src/llama-kv-compact-self-study.h` — added `max_q_capture_mb` config field
- `src/llama-kv-compact-self-study.cpp` — memory guard computation + auto-reduce

## Related Minor Fixes (same commit)

- **m-02**: Per-round RNG seed (was fixed seed 42 for all rounds)
- **m-03**: partial_sort bound reduced from `k_select + i` to `k_select`
- **m-04**: isfinite check added to OMP Cholesky solver
- **R1-Minor-2**: Guard for all-negative-infinity logits in temperature sampling
- **R1-Minor-3**: JSON parser key validation (require digits, consume full key)
