# BUG-R02: Thread Safety of Static Warning Flags in compacted_prefix_runtime_supported

| Field | Value |
|-------|-------|
| **Severity** | Minor |
| **Status** | **FIXED** |
| **Source** | Adversarial review of V1 code fixes (commit `3d5132b1`) |
| **Models affected** | SWA models (Gemma3) and M-RoPE models (Qwen2-VL, Qwen3-VL) |
| **Discovered** | 2026-03-14, adversarial review |
| **Fixed** | 2026-03-14, `src/llama-kv-cache.cpp` |

## Symptoms

The `compacted_prefix_runtime_supported()` function used non-atomic `static bool` flags
for one-time warning messages. In a multi-threaded scenario, two threads could both
observe `warned_swa == false`, both print the warning, and both set it to `true`.

The warning would appear twice in the log instead of once.

## Root Cause

```cpp
static bool warned_swa = false;    // non-atomic
if (!warned_swa) {                 // TOCTOU race
    LLAMA_LOG_WARN(...);
    warned_swa = true;
}
```

While C++11 guarantees thread-safe initialization of static locals, the subsequent
read-modify-write (`if (!warned_swa) { ... warned_swa = true; }`) is not atomic.
Two threads can both read `false` before either writes `true`.

## Impact

- **Production impact: Cosmetic only.** The warning is informational — it tells users
  that compaction is unsupported for SWA/M-RoPE models. A duplicate warning does not
  affect correctness.
- **Defensive coding: Important.** Non-atomic flags are a code smell that could mask
  real issues if the flag were later used for gating logic rather than logging.

## Fix

Changed both warning flags to `std::atomic<bool>` with `exchange()` for atomic
test-and-set:

```cpp
static std::atomic<bool> warned_swa{false};
if (!warned_swa.exchange(true)) {
    LLAMA_LOG_WARN(...);
}
```

The `exchange(true)` atomically sets the flag to `true` and returns the previous value.
Only the first thread to call `exchange` sees `false` and enters the warning block.

Added `#include <atomic>` to `llama-kv-cache.cpp`.

## Code Locations

- `src/llama-kv-cache.cpp:1127` — SWA warning (`warned_swa`)
- `src/llama-kv-cache.cpp:1145` — M-RoPE warning (`warned`)

## Test Verification

- ctest 46/47 pass (1 pre-existing upstream tokenizer failure)
- All compaction unit tests pass
