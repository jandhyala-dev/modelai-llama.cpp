# Hostile Review Protocol

All implementation slice reviews in this repository MUST follow this protocol.
This applies to both Claude Code and Codex reviewers.

---

## REVIEW REQUEST TEMPLATE

Every review request MUST include these fields. The implementer fills in the values before submitting.

```
REVIEW REQUEST — Slice <ID>: <title>

Repository: <owner/repo>
Branch:     <branch name>
Commit:     <full or short SHA>
Plan baseline: <SHA of the plan commit being implemented against>
Plan reference: <Part/section, with line ranges if known>

Previous review: <PASS / FAIL (N findings) / first review>

Files changed:
  - <file path> (<NEW or +N/-M lines>)
  - ...

Build commands run by implementer:
  <commands and results>

Tests run by implementer:
  <ctest commands, manual tests, or "build-only — no behavioral tests for this slice">

SCOPE: Review ONLY <this slice>. Do NOT review <adjacent slices>.

Verify:
1. <checkpoint>
2. <checkpoint>
...
```

---

## REVIEWER STANCE

You are performing a hostile, failure-seeking review of a single implementation slice.

Your job is not to confirm that the code looks reasonable.
Your job is to try to break it.

You must review the slice as if it will become a real production dependency and may later be upstreamed. There is no room for "probably fine." If there is any plausible correctness bug in the intended production use of the slice, you must FAIL it.

Do not trust:
- prior review conclusions
- comments
- tests
- docs
- clean-looking code

Treat all of those as claims that must be verified.

---

## MANDATORY REVIEW PROTOCOL

### 0. SCOPE GATE
- State exactly what is IN SCOPE for this slice.
- State exactly what is OUT OF SCOPE for this slice.
- If the implementation leaks into future-slice behavior, silently changes prior-slice behavior, or introduces undocumented side effects outside the stated slice, that is a finding.

### 1. SPEC MATCH
- Check the implementation against the exact plan/spec for this slice.
- Flag any contract drift between:
  - plan text
  - header comments
  - docstrings
  - function signatures
  - struct layouts
  - test assumptions
  - actual code behavior
- If the header says X and the code does Y, that is a finding even if Y might be arguably better.

### 2. CONTRACT BOUNDARY CHECK
- For every public or cross-slice struct/function in this slice, verify:
  a. the caller's assumptions match the callee's behavior
  b. preconditions are documented and enforced
  c. postconditions hold for all non-error paths
  d. ownership/layout/units are explicit
  e. the output contract for the next slice is unambiguous
- If this slice produces output consumed by a future slice, verify that the output format is fully specified:
  - shape
  - layout
  - dtype
  - units
  - ownership
  - lifetime
  - valid ranges
  - failure behavior

### 3. CONCRETE TRACES (MANDATORY — DO NOT SKIP)

#### 3a. PRODUCTION TRACE
- Use the dominant real-world production case for this slice.
- Use real numbers, not symbolic placeholders.
- Walk every:
  - index
  - stride
  - offset
  - shape
  - loop bound
  - selected index
  - intermediate value
- If grouping, sampling, subsampling, or reordering occurs, show the actual chosen source indices and destination indices.
- If the trace involves a callback sequence, you MUST trace the full callback sequence for the production model. Do not summarize it as "works the same way" based on a simpler model.

#### 3b. BOUNDARY TRACE
- Use the smallest or closest-to-threshold valid input.
- Explicitly check:
  - empty input
  - single-element input
  - exact-threshold values
  - max capacity edge
  - ratio == 1
  - count == 0 or 1 if valid
  - first and last legal index
- If the slice has limits or caps, test at, below, and just above the threshold.

#### 3c. ADVERSARIAL TRACE
- Pick a realistic hostile case designed to break the code.
- Prefer:
  - unusual GQA ratios
  - multiple matching tensors per step
  - near-capacity states
  - divisibility edge cases
  - partial restore / partial overwrite paths
  - unsupported-but-plausible dtypes
  - model variants with extra callbacks / intermediate tensors

#### 3d. INTEGER ARITHMETIC TRACE
- For every division, modulo, stride, offset, row count, token count, and index calculation:
  - substitute production values
  - substitute boundary values
  - state whether truncation/rounding changes behavior
- Integer division truncation is a top-priority failure mode.
- If arithmetic silently degenerates into truncation, clipping, or bias, that is a finding.

#### 3e. SECURITY TRACE (MANDATORY FOR WEB/API CODE)
- For every user-facing endpoint, middleware, or data ingestion path, trace:
  - **Injection:** SQL injection (parameterized queries?), command injection (shell calls with user input?), XSS (output encoding?), template injection, formula injection (Excel/CSV contexts)
  - **Authentication/Authorization bypass:** Can an unauthenticated user reach protected routes? Can a lower-privilege actor escalate? Are auth checks applied before business logic?
  - **SSRF/Path traversal:** Does user input flow into URLs, file paths, or hostnames without validation? Can `../` or protocol switching (`file://`, `gopher://`) reach internal resources?
  - **Secrets exposure:** Are API keys, tokens, or credentials logged, included in error responses, or committed to source?
  - **Denial of service:** Unbounded allocations, missing pagination caps, regex catastrophic backtracking (ReDoS), zip bombs, XML entity expansion
  - **Deserialization:** Is untrusted input passed to `JSON.parse` callbacks, `eval`, `new Function`, `vm.runInNewContext`, or YAML/pickle loaders?
  - **CORS/CSP:** Are cross-origin policies overly permissive? Does `Access-Control-Allow-Origin: *` appear on authenticated endpoints?
- For each attack surface found, construct a concrete exploit payload and trace it through the code to determine if it reaches a sink.
- If the code is not web/API (e.g., pure C++ library), this trace may be marked N/A with justification.

#### 3f. CONCURRENCY AND ASYNC TRACE (MANDATORY WHEN APPLICABLE)
- If the slice uses async/await, Promises, threads, mutexes, or shared mutable state, trace:
  - **Race conditions:** Can two concurrent requests read-modify-write the same state? Are database operations atomic or do they need transactions?
  - **Deadlocks:** Can lock acquisition order vary between code paths?
  - **Orphaned work:** When `Promise.race` or timeouts abort a logical operation, do the losing promises continue consuming resources (CPU, memory, network connections, database connections)?
  - **Error propagation:** Do rejected promises in concurrent workers (`Promise.all`, `Promise.allSettled`, worker pools) propagate correctly or are they silently swallowed?
  - **Timer cleanup:** Every `setTimeout`/`setInterval` must have a corresponding `clearTimeout`/`clearInterval` on ALL exit paths (success, error, timeout, early return).
  - **Connection pool exhaustion:** Can concurrent requests drain database/HTTP connection pools? Are connections returned on error paths?
- If no concurrency exists in the slice, mark N/A with justification.

### 4. MULTI-VARIANT MODEL TRACE (WHEN APPLICABLE)
- If the slice touches callback/capture/model-specific graph behavior:
  a. list every matching tensor/callback event in the trace
  b. show the tensor name, shape, branch taken, and state mutation
  c. verify the final state after the full callback sequence
- Minimum requirement:
  - trace the models required by the slice's support matrix
  - plus at least one model variant with multiple matching tensors per layer or per step

### 5. STATE-MACHINE TRACE (WHEN APPLICABLE)
- If the slice mutates persistent state, trace:
  a. before-state
  b. action
  c. after-state
  d. failure-state
  e. rollback-state
- Apply this especially to:
  - cache state
  - save/restore
  - enable/disable flags
  - overwrite/finalize paths
  - partial-failure paths
- If rollback or cleanup is claimed, verify it with a real trace.

### 6. UNSUPPORTED / PRECONDITION AUDIT
- List every assumption the code makes.
- For each assumption, state whether it is:
  - enforced in code
  - documented only
  - undocumented
- Especially audit:
  - dtype assumptions
  - divisibility assumptions
  - architecture assumptions
  - flash/non-flash assumptions
  - single-sequence assumptions
  - thread-safety assumptions
  - device/backend assumptions
  - shape/layout assumptions
- If the code relies on an assumption that is not enforced or documented, that is a finding.

### 7. TEST REALITY CHECK
- Do not assume "test exists" means "test proves correctness."
- Verify:
  a. tests are active in the actual build mode used
  b. no assert/NDEBUG trap
  c. tests cover the dominant production case
  d. tests cover at least one boundary case
  e. tests cover the most likely bug class for this slice
  f. tests would have caught the concrete failure modes you identified
- If tests are weak, incomplete, or misleading, that is a finding.

### 8. DISPROVE-IT PASS (MANDATORY BEFORE ANY PASS VERDICT)
- Assume the code has exactly one bug. Try to find it.
- Systematically test:
  a. smallest valid input
  b. largest expected input
  c. common production input
  d. input where integer division rounds down
  e. input where a loop bound is off by one
  f. input where two code paths interact
  g. input where state persists across calls
  h. input where the ratio of two values crosses an integer boundary (for example, 1280/1024 = 1.25 truncating to 1)
- You must explicitly document what you tried to break and whether it broke.

### 9. SKEPTICISM HIERARCHY
- Be most skeptical of:
  - indexing
  - layouts
  - grouping
  - integer division
  - state machines
  - quantization boundaries
  - persistence / rollback
- Be moderately skeptical of:
  - memory ownership
  - error paths
  - logging accuracy
  - callback sequencing
- Be least skeptical of:
  - naming
  - style
  - grammar
- Clean code is not evidence of correctness.
- Passing the production trace is not enough. Boundary and adversarial traces exist to break what the happy path cannot.

### 10. DEPENDENCY AND SUPPLY-CHAIN CHECK
- For every new dependency added in this slice:
  a. Is it actively maintained (last commit < 12 months)?
  b. Is the license compatible (MIT, Apache-2.0, ISC, BSD are safe; GPL/AGPL require review)?
  c. Does it pull in transitive dependencies with known CVEs? (`npm audit`, `pip audit`, or equivalent)
  d. Is the package name correct (typosquatting check)?
  e. Is the dependency pinned to a specific version or range?
- For dependency upgrades: are there breaking changes in the changelog between old and new versions?
- If no new dependencies, mark N/A.

### 11. PERFORMANCE REGRESSION CHECK
- If the slice modifies a hot path (request handler, loop body, data pipeline, render path):
  a. Is there a before/after measurement or benchmark?
  b. Does the change add O(n) or worse complexity where O(1) existed?
  c. Are there unnecessary allocations inside loops (object creation, string concatenation, array spread)?
  d. Does the change add synchronous I/O, blocking calls, or `await` in a tight loop?
  e. For database queries: are new queries indexed? Do they avoid full table scans?
- If the slice is not on a hot path (config, docs, one-time setup), mark N/A with justification.

### 12. CROSS-REPO CONTRACT CHECK
- If this slice produces or consumes an interface shared across repositories (REST API, CLI flags, file formats, database schemas, environment variables, IPC protocols):
  a. List every cross-repo contract touched
  b. Verify the producer's output matches the consumer's expectations (field names, types, units, error shapes)
  c. Verify backward compatibility — will the other repo break if it hasn't been updated yet?
  d. If a breaking change is intentional, verify that all consuming repos are updated in the same release
- Cross-repo contracts in the ModelAI ecosystem include:
  - modelai-llama.cpp REST API (health, props, models, completions, metrics) consumed by ModelAI server
  - Supabase schema (tables, RLS policies, storage buckets) shared across ModelAI server and marketing site
  - Environment variables and config contracts between repos
  - Git webhook payloads between GitHub and the deployment pipeline
- If no cross-repo contracts are touched, mark N/A.

### 13. PASS BAR
- PASS only if:
  a. no plausible production bug remains
  b. critical contracts are explicit and consistent
  c. assumptions are either enforced or clearly documented
  d. tests are meaningful for this slice
  e. your disprove-it pass did not find a credible failure
  f. no unmitigated security vulnerabilities exist (from security trace)
  g. no timer/resource leaks exist (from concurrency trace)
  h. no breaking cross-repo contract changes are unaddressed
- Otherwise FAIL.

---

## REQUIRED OUTPUT FORMAT

```
## Verdict: PASS / FAIL

## Scope Check
- In scope:
- Out of scope:
- Any scope leak:

## Findings
For each finding:
- Severity: Critical / Major / Minor
- File:line reference
- What the code does (with concrete values)
- What it should do
- Why this matters in production
- Minimum fix

## Traces Executed

### Production trace
- model:
- concrete inputs:
- step-by-step values:
- result:

### Boundary trace
- case:
- concrete inputs:
- step-by-step values:
- result:

### Adversarial trace
- case:
- concrete inputs:
- step-by-step values:
- result:

### Integer arithmetic audit
- every division/modulo/stride/offset with substituted values
- state where truncation/rounding changes behavior

### Security trace (if applicable)
- attack surfaces identified
- concrete exploit payloads attempted
- for each: did it reach a sink? was it blocked? by what mechanism?

### Concurrency/async trace (if applicable)
- race conditions tested
- timer/resource cleanup verified on all exit paths
- orphaned work identified

### Multi-variant model trace (if applicable)
- per-model callback/tensor sequence
- final state after all matching callbacks

### State-machine trace (if applicable)
- before state
- action
- after state
- failure state
- rollback state

## Dependency / Supply-Chain Check
- new dependencies added: (list or N/A)
- license compatibility: (verified or N/A)
- known CVEs: (clean or list)

## Cross-Repo Contract Check
- contracts touched: (list or N/A)
- backward compatibility: (verified or breaking — with justification)

## Performance Check
- hot path affected: (yes/no)
- measurement: (before/after or N/A)

## Unsupported / Precondition Audit
- list every assumption
- state whether it is:
  - enforced
  - documented
  - undocumented

## Test Reality Check
- what tests actually prove
- what they do not prove
- whether they would catch the failure modes considered

## Disprove-it Attempt
- what you tried to break
- why you chose those cases
- whether it broke

## Deferred Risks
- only list items that are genuinely safe to defer
- if an issue can affect correctness in the current slice, it is not deferrable

## Pass Justification (if PASS)
- why the traces, assumptions, and tests are sufficient for this slice
```

---

## FINDING SEVERITIES

| Severity | Meaning | Blocks? |
|----------|---------|---------|
| **Critical** | Data corruption, security issue, crash in production path | Yes |
| **Major** | Correctness bug, wrong results, contract violation | Yes |
| **Minor** | Suboptimal but correct, cosmetic, documentation gap | No (note it) |

---

## RE-REVIEW AFTER FIX

When a slice fails and is fixed:

1. Implementer commits the fix with a message referencing the finding.
2. Implementer submits a re-review request with:
   - The new commit SHA
   - `Previous review: FAIL (N findings — <summary>)`
   - `SCOPE: Re-review ONLY <the fixed function/area>. All other checks PASS from previous review.`
3. Reviewer re-checks ONLY the failed items. Passing items carry forward.
4. The full hostile protocol (traces, disprove-it) applies to the re-reviewed scope.

---

## RULES

- The implementer does NOT proceed to the next slice until the current slice gets PASS.
- If a slice is implemented before the previous slice is approved, that is the implementer's risk — the reviewer may require changes that cascade.
- The reviewer must not suggest improvements beyond the declared scope. File them separately.
- "It works on my machine" is not evidence. The review must trace the code path with concrete values.

---

## MANDATORY TESTING-REVIEW LOOP

All testing MUST follow a break-fix-review cycle. No exceptions.

1. **Tests must attempt to break the code.** Happy-path-only testing is not acceptable. Tests must include adversarial inputs, boundary conditions, resource exhaustion, and cross-engine comparisons.
2. **Every bug found MUST be fixed, committed, and pushed** before the review cycle concludes.
3. **After fixes, a full adversarial review MUST run again** using this protocol. No partial re-reviews — the fix may have introduced new issues.
4. **The cycle repeats** (test → break → fix → commit → push → review) until the adversarial review returns PASS with zero Critical or Major findings.
5. **This rule applies to ALL repositories** in the ModelAI ecosystem (modelai-llama.cpp, modelai, COT Labs). Update review standards in each repo accordingly.

This loop is mandatory for:
- All benchmark result commits
- All implementation slice commits
- All bug fix commits
- All CI/CD pipeline changes

---

## NON-NEGOTIABLE RULES

- Do not give credit for "looks correct."
- Do not soften the review.
- Do not skip numeric traces on math-heavy logic.
- Do not skip the disprove-it pass.
- If there is any plausible correctness bug in the intended production use of the slice, FAIL it.
- Tests must break things. Every bug must be fixed. Adversarial review must pass. No exceptions.
