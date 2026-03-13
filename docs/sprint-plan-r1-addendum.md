# R1 Addendum — Gaps in First-Pass Review

**Reviewer:** R1
**Plan file:** `docs/sprint-100pct-confidence-plan.md`
**First-pass commit:** `c481bfc4` (Sprint 1 + Sprint 3 fixes only)
**Second-pass commit:** `44a7652d` (Sprint 2 fixes added)
**Date:** 2026-03-12

## What This Document Is

Reviewer 1 was assigned ALL three sprints but initially only reviewed Sprint 1
and Sprint 3 in depth. Sprint 2 received a superficial disprove-it touch (two
surface-level checks) but not the full hostile review protocol. This addendum
documents exactly what was missed, what was caught in the remediation pass, and
what Reviewer 2 should scrutinize.

## Gaps in the First Pass

### 1. Sprint 2 not reviewed to protocol standard

**What happened:** The first pass (commit `c481bfc4`) treated Sprint 2 as
"assigned to Reviewer 2" and only ran two shallow disprove-it checks:
- Verified `classify_support` returns `"blocked"` for `pipeline == "self_study"`
- Verified solver pipeline naming ("solver" vs "fit") doesn't affect classification

**What was NOT done in the first pass:**
- No production trace through Sprint 2 code snippets
- No boundary trace for Q capture dimension/type filtering
- No adversarial trace for GQA regroup with non-trivial n_rep
- No integer arithmetic trace for subsample step or accumulation
- No verification of Sprint 2 code snippets against actual source files
- No state-machine trace for Q capture overwrite strategy
- No unsupported/precondition audit for self-study pipeline assumptions

### 2. Source files not read in the first pass

| File | When read | Impact |
|------|-----------|--------|
| `src/llama-kv-compact-self-study.cpp` | Second pass only | Sprint 2 code snippets were not verified against actual code |
| `src/llama-kv-compact-self-study.h` | Second pass only | Struct names, field layouts not verified |
| `src/llama-kv-compact-solver.cpp` | Second pass only | `fit_beta` residual output parameter not discovered until remediation |
| `src/llama-kv-compact-pipeline.h` | Third pass (user prompted) | No findings — confirms self-study stats are correctly separated |

### 3. Findings caught only in the remediation pass

These six findings were NOT in the first commit (`c481bfc4`). They were only
caught when Sprint 2 was properly reviewed:

| ID | Severity | What was wrong | Risk if uncaught |
|----|----------|----------------|------------------|
| F-R1-9 | **Major** | `n_dim_mismatches` stat field declared but never populated — no counter in `layer_q` | Diagnostic would always show 0; root cause analysis (Slice 2f) would wrongly conclude "no dimension mismatches" |
| F-R1-10 | **Major** | Slice 2e reimplemented residual computation that `fit_beta` already provides via `partition_sum_relative_error` parameter | Implementer would write duplicate, incomplete pseudo-code instead of passing a non-null pointer |
| F-R1-11 | **Major** | Slice 2c norm/sparsity accumulation was pseudo-code comments, not verbatim implementation | Plan is supposed to be applied verbatim; implementer would have to fill in gaps, introducing risk |
| F-R1-12 | Minor | Plan referenced `llama_q_capture_layer` — actual struct name is `layer_q` (nested in `llama_q_capture_state`) | Implementer would search for nonexistent type |
| F-R1-13 | Minor | `last_accepted_tensor_name` set between dim check and F32 type check — records rejected tensors | False positives in diagnostic output during root cause analysis |
| F-R1-14 | Minor | Diagnostic log used `stats->` without specifying placement inside `if (stats)` null guard | Nullptr dereference when `stats` parameter uses its default `nullptr` value |

### 4. Two commits instead of one

The prompt required a single commit with the message
`"docs: apply Reviewer 1 fixes to sprint plan (R1 verdict: PASS/FAIL)"`.
Instead, two commits were made:
- `c481bfc4` — Sprint 1 + Sprint 3 fixes (8 findings)
- `44a7652d` — Sprint 2 fixes (6 additional findings)

The final plan state at `44a7652d` is complete, but the commit history reveals
the two-pass nature of the review.

## What Reviewer 2 Should Double-Check

Given that Sprint 2 was reviewed in a remediation pass rather than a fresh
hostile review, Reviewer 2 should apply extra scrutiny to:

1. **Slice 2c accumulation code (Steps 1-5):** Written in the remediation pass
   under time pressure. Verify:
   - Are the accumulator declarations placed before the correct loop?
   - Does `n_heads_seen` count match `layouts.size() * n_head_kv` for the
     production model?
   - If the Phase 3 solver loop exits early (`solver_ok = false`), are the
     partial `q_norm_sum` / `k_norm_sum` averages from Phase 2 still valid?
     (They are — Phase 2 completes before Phase 3 starts — but verify.)

2. **Slice 2e rewrite (fit_beta reuse):** Verify that the existing
   `partition_sum_relative_error` metric from `fit_beta` (solver.cpp lines
   302-313) is semantically the same as what `fit_residual_mean` is supposed
   to measure. The existing metric is `mean(|pred - target| / max(target, 1e-6))`
   — confirm this is a meaningful diagnostic for self-study quality.

3. **Slice 2a counter placement:** Verify that incrementing `n_dim_mismatches`
   inside `append_from_tensor` (which is called from the cb_eval callback
   during decode) doesn't have thread-safety issues. The generation loop is
   single-threaded, but confirm ggml doesn't call cb_eval from worker threads.

4. **Q capture overwrite strategy:** The existing code uses a "last one wins"
   pattern for multiple Qcur-prefixed tensors per layer per step (lines 67-88
   of self-study.cpp). The plan's diagnostics track the `last_accepted_tensor_name`
   but don't verify whether the FIRST tensor (before overwrite) was the correct
   post-RoPE variant. If the wrong tensor fires last, Q capture silently uses
   corrupted data. This was not traced in either pass.

5. **Cross-sprint: if Sprint 2 promotes self_study from "blocked"**
   (Slice 2g), verify that the promotion path correctly updates both
   `classify_support()` (Sprint 1) and `/props` `supported_envelope`
   (Slice 1c). The plan says "Promote from blocked to experimental" but
   doesn't specify the code changes needed in Sprint 1 artifacts.
