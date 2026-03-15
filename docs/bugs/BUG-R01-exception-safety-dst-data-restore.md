# BUG-R01: Exception Safety in dst->data Save/Restore (B5 Staging Pattern)

| Field | Value |
|-------|-------|
| **Severity** | Minor |
| **Status** | **FIXED** |
| **Source** | Adversarial review of V1 code fixes (commit `3d5132b1`) |
| **Models affected** | All models using compacted prefix |
| **Discovered** | 2026-03-14, adversarial review |
| **Fixed** | 2026-03-14, `src/llama-kv-cache.cpp` |

## Symptoms

If any of the compacted prefix exec functions (`llama_compacted_prefix_set_input_k`,
`llama_compacted_prefix_set_input_v`, `llama_compacted_prefix_set_input_beta`) throws
an exception (e.g., tensor type mismatch, dimension mismatch), the `dst->data` pointer
is left pointing to the host staging buffer instead of the original GPU tensor data.

This corrupts the tensor's data pointer for subsequent graph operations.

## Root Cause

The B5 fix (commit `3d5132b1`) introduced a pattern that temporarily swaps `dst->data`
to a host staging buffer, calls the exec function, then restores the original pointer.
The restore happened only on the success path — if the exec function threw, the restore
was skipped:

```cpp
void * original_data = dst->data;
dst->data = staging;
exec_function(dst, ...);   // can throw!
dst->data = original_data; // skipped on throw
```

## Impact

- **Production impact: None.** The exec functions only throw on precondition violations
  (wrong tensor type/dimensions), which are programmer errors, not runtime data issues.
  After such a throw, the entire decode batch fails and the system is typically in an
  unrecoverable state.
- **Defensive coding: Important.** If error recovery is ever added, the corrupted
  `dst->data` pointer could cause use-after-free or silent data corruption.

## Fix

Added try/catch guards to all three set_input functions to restore `dst->data` before
re-throwing:

```cpp
void * original_data = dst->data;
dst->data = staging;
try {
    exec_function(dst, ...);
} catch (...) {
    dst->data = original_data;
    throw;
}
dst->data = original_data;
```

## Code Locations

- `src/llama-kv-cache.cpp` — `set_input_compacted_prefix_k` (K staging)
- `src/llama-kv-cache.cpp` — `set_input_compacted_prefix_v` (V staging)
- `src/llama-kv-cache.cpp` — `set_input_compacted_prefix_kq_b` (beta staging)

## Test Verification

- ctest 46/47 pass (1 pre-existing upstream tokenizer failure)
- All compaction unit tests pass: `test-kv-compacted-prefix-exec`, `test-kv-compact-solver`,
  `test-kv-compact-self-study`, `test-kv-compact-features`
