# Hostile Review Protocol

Standard adversarial review process for all code entering `jandhyala-dev/modelai-llama.cpp`.

---

## Review Request Template

Every review request MUST include these fields verbatim. The implementer fills in the values before submitting.

```
REVIEW REQUEST — Slice <ID>: <title>

Repository: jandhyala-dev/modelai-llama.cpp
Branch:     <branch name>
Commit:     <full or short SHA>
Plan baseline: <SHA of the plan commit being implemented against>
Plan reference: <Part/section, with line ranges if known>

Previous review: <PASS / FAIL (N findings) / first review>

Files changed:
  - <file path> (<NEW or +N/-M lines>)
  - ...

Build commands run by implementer:
  cmake -B build -DGGML_METAL=ON -DLLAMA_FATAL_WARNINGS=ON
  cmake --build build --config Release -j$(sysctl -n hw.ncpu)
  Result: <pass/fail, any warnings>

Tests run by implementer:
  <ctest commands, manual tests, or "build-only — no behavioral tests for this slice">

SCOPE: Review ONLY <this slice>. Do NOT review <adjacent slices>.
```

---

## Reviewer Checklist

The reviewer MUST check every item below. Each item gets PASS or FAIL with a specific file:line reference.

### A. Correctness

1. **Plan fidelity** — Does the code match the plan's specification? Signatures, data structures, algorithms, file placement.
2. **Edge cases** — Trace through at least 2 concrete models (e.g., standard Llama, Qwen3-14B GQA, a model with kq_norm). Write the trace in the review.
3. **Arithmetic** — Verify all index arithmetic, stride calculations, and loop bounds. Check for integer division truncation, off-by-one, and remainder handling.
4. **API contracts** — Verify function signatures against actual current headers (`include/llama.h`, `ggml/include/*.h`), not plan pseudocode. If the plan says `llama_foo()`, grep the header and confirm it exists with the expected signature.

### B. Safety

5. **Memory** — No buffer overflows, no use-after-free, no uninitialized reads. Vector bounds checked. GPU-to-host copies use correct byte sizes.
6. **Hot-path allocation** — No heap allocation in callbacks or tight loops unless pre-reserved. Flag any `resize()`, `push_back()`, or `new` in performance-critical paths.
7. **Thread safety** — Flag any shared mutable state. For v0 single-threaded context usage, document the assumption explicitly.
8. **Warning cleanliness** — Code must build with `-DLLAMA_FATAL_WARNINGS=ON` (the CI flag). Unused parameters, signed/unsigned mismatches, and implicit conversions are errors.

### C. Scope

9. **No scope creep** — Only the declared slice scope is implemented. No bonus features, no refactors of adjacent code, no "while I'm here" changes.
10. **No future-slice work** — Nothing from later slices appears in this commit. If a stub or forward declaration is needed for compilation, it must be minimal and documented.

### D. Contracts

11. **Caller contracts** — If this code requires callers to do something (e.g., call `finalize_step()` after each decode), the requirement must be documented in the header and noted in the review.
12. **Consistency** — Comments, header docs, and implementation must all agree on data layouts, ownership, lifetimes, and invariants. A mismatch between header doc and code is a FAIL.

---

## Verdict Format

The reviewer outputs exactly one of:

- **PASS** — All checks satisfied. List each check with status and evidence.
- **FAIL** — One or more checks failed. List findings with severity, file:line, and a concrete trace showing the failure. Then list passing checks.

### Finding Severities

| Severity | Meaning | Blocks? |
|----------|---------|---------|
| **Major** | Correctness bug, data corruption, wrong results | Yes |
| **Minor** | Suboptimal but correct, cosmetic, documentation gap | No (note it) |

### Verdict Table Format

```
| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | Plan fidelity | PASS | file.cpp:42 — signature matches plan line 217 |
| 2 | Edge cases | PASS | Traced Llama (1 Qcur per layer) and Qwen3-14B (3 Qcur per layer) |
| ... | ... | ... | ... |
```

---

## Re-Review After Fix

When a slice fails and is fixed:

1. Implementer commits the fix with a message referencing the finding.
2. Implementer submits a re-review request with:
   - The new commit SHA
   - `Previous review: FAIL (N findings — <summary>)`
   - `SCOPE: Re-review ONLY <the fixed function/area>. All other checks PASS from previous review.`
3. Reviewer re-checks ONLY the failed items. Passing items carry forward.

---

## Rules

- The implementer does NOT proceed to the next slice until the current slice gets PASS.
- If a slice is implemented before the previous slice is approved, that is the implementer's risk — the reviewer may require changes that cascade.
- The reviewer must not suggest improvements beyond the declared scope. File them separately.
- "It works on my machine" is not evidence. The review must trace the code path for at least 2 model architectures.
