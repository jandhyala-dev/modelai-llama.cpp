# Future Enhancement Plan: Multimodal Reliability and Vision-Aware KV Compaction

## Status

This is a future exploration track. It is intentionally separate from `docs/PLAN-hybrid-compaction-and-bf16.md`.

Current repo reality:
- `llama-server` already supports multimodal inference through `libmtmd` and `--mmproj`.
- The current KV compaction path supports standard causal, iSWA, hybrid SSM+attention, hybrid-iSWA, and IMROPE text-only.
- Spatial M-RoPE multimodal compaction is intentionally blocked today.

Relevant current references:
- `docs/multimodal.md`
- `docs/DESIGN-DECISIONS.md`
- `docs/kv-compaction-integration.md`
- `src/llama-kv-cache.cpp`
- `src/llama-kv-cells.h`
- `src/llama-kv-compacted-prefix.h`

## Why this needs its own track

The current compaction pipeline assumes text-style 1D ordering for prefix positions. That assumption is valid for:
- standard text models
- standard hybrid recurrent+attention models with standard RoPE
- IMROPE text-only workloads

That assumption is not valid for spatial multimodal M-RoPE models where attention depends on structured position metadata such as:
- x/y coordinates
- image patch order
- page or region boundaries
- mixed modality boundaries across text, image, and audio

Trying to force full multimodal compaction into the current text-centric compaction lane would create correctness risk and make failure analysis much harder.

## Product goal

Make this fork strong on long multimodal sessions in three stages:
1. reliable multimodal inference in `llama-server`
2. pinned visual-prefix reuse plus text-tail-only compaction for mixed sessions
3. later full M-RoPE-aware multimodal compaction when the correctness model is explicit and testable

## Non-goals for the first vision track

Do not treat these as phase-1 goals:
- generic "all multimodal models supported" marketing
- full spatial-token compaction on day 1
- merging visual patch tokens with text tokens under current 1D prefix logic
- changing the current text-only IMROPE contract
- changing unsupported-model claims before tests prove it

## Candidate model lanes

Primary multimodal / vision reliability lane:
- `Qwen2.5-VL`
- `Llama-4-Scout`
- one smaller Gemma or SmolVLM lane for local regression coverage

Primary text-tail compaction validation lane:
- one multimodal model with repeated image/document follow-up questions
- one OCR-like document QA workload
- one chart/table reasoning workload

Primary dense control lane:
- `Qwen3-8B`
- `Llama-3.2-3B`

## Phase 1: Multimodal inference reliability in llama-server

### Goal

Make multimodal serving reliable and observable before changing compaction semantics.

### Scope

Focus on:
- model loading with `--mmproj`
- request parsing for image/audio payloads
- error surfaces when multimodal projector is missing or incompatible
- stable `llama-server` request/response behavior for multimodal chat
- visibility into whether the loaded model is multimodal-capable

### Files likely involved

- `tools/server/server-common.cpp`
- `tools/server/server-context.cpp`
- `tools/server/README.md`
- `docs/multimodal.md`
- `src/llama-model.cpp`
- `src/llama.cpp`

### Required implementation outcomes

1. Server startup clearly reports:
- text-only model
- multimodal-capable model
- multimodal projector loaded or missing

2. Request-time errors are explicit for:
- multimodal request on non-multimodal model
- missing `mmproj`
- malformed multimodal payload

3. `/props` or equivalent runtime surface should expose enough state to answer:
- is multimodal supported?
- was projector loaded?
- is compaction currently available for this loaded architecture?

4. Multimodal requests must not silently fall into broken text-only behavior.

### Tests required

- server unit tests for multimodal request validation
- one model-backed smoke test for `llama-server` with `--mmproj`
- one negative test proving missing projector fails loudly
- one contract test for server capability/status reporting

## Phase 2: Pinned visual prefix and text-tail-only compaction

### Goal

Support long multimodal sessions without claiming full visual-token compaction.

### Key idea

Treat the multimodal prefix as two logical regions:
1. a pinned visual or multimodal prefix that remains intact
2. a later text-only conversational tail that may be compacted

This is the highest-leverage near-term path because many real multimodal sessions look like:
- image or document uploaded once
- long text follow-up over the same visual context

### Scope

Add a compaction mode that:
- preserves visual prefix tokens
- preserves modality boundaries
- applies compaction only to the post-prefix text tail
- avoids trying to compress spatial patch tokens under the current 1D assumptions

### Design requirements

1. Prefix pinning must be explicit and queryable.
2. Compaction selection must never cross the pinned multimodal boundary.
3. Server/API surfaces must make clear that this is `text-tail-only` compaction, not full multimodal compaction.
4. Save/restore must preserve the pinned-prefix boundary.

### Files likely involved

- `src/llama-kv-cache.cpp`
- `src/llama-kv-compacted-prefix.h`
- `src/llama-kv-compacted-prefix.cpp`
- `src/llama-kv-compacted-prefix-exec.cpp`
- `src/llama-kv-cells.h`
- `tools/server/server-context.cpp`
- `include/llama.h`

### Likely API / state additions

Add explicit sequence-local metadata for:
- pinned multimodal prefix length
- text-tail compaction start position
- whether multimodal prefix reuse is active
- whether full multimodal compaction is disabled for the current sequence

Possible names are illustrative only:
- `pinned_prefix_tokens`
- `text_tail_pos0`
- `multimodal_prefix_reused`
- `compaction_mode: text_only | text_tail_only | full_multimodal`

### Tests required

- unit test for pinned-boundary preservation
- unit test proving selection never reaches into visual prefix tokens
- save/restore regression test for pinned prefix metadata
- server contract test showing `text_tail_only` mode in response/status surfaces
- model-backed repeated-image follow-up benchmark proving lower KV pressure without wrong-answer regressions

## Phase 3: Full M-RoPE-aware multimodal compaction

### Goal

Generalize compaction so spatial multimodal models can be compacted safely.

### Hard problem to solve

Current compaction assumes 1D token order. Full multimodal compaction requires a richer token model with modality-aware and spatial-aware semantics.

### Architectural requirements

1. Generalize token metadata beyond scalar text position.
2. Preserve enough spatial structure to avoid nonsensical attention after compaction.
3. Distinguish modality classes such as:
- text
- image
- audio
- multimodal special tokens
- projector-derived tokens

4. Distinguish position semantics such as:
- 1D text order
- 2D spatial position
- page/region grouping
- sequence-level segment boundaries

### Code seams to explore

- `src/llama-kv-cells.h`
- `src/llama-kv-cache.cpp`
- `src/llama-kv-compacted-prefix.h`
- `src/llama-kv-compacted-prefix.cpp`
- `src/llama-kv-compacted-prefix-exec.cpp`
- selection / query / pipeline files under `src/llama-kv-compact-*`

### Candidate metadata expansion

The existing `llama_kv_cell_ext` already carries `x` and `y`. Full multimodal compaction likely needs more than that.

Potential future fields:
- modality tag
- segment id
- page id or region id
- pinned / compactable flag
- position semantics enum
- optional temporal index for video/audio-style cases

### Required constraints

1. Do not ship full multimodal compaction without model-backed correctness tests.
2. Do not silently downgrade M-RoPE into text-only assumptions.
3. Do not claim support for spatial multimodal compaction until request/response surfaces clearly expose the active mode.

### Tests required

- token-metadata unit tests
- compaction selection tests over mixed text-image-text sequences
- execution tests for M-RoPE-aware masking/materialization
- save/restore tests for new multimodal metadata
- server contract tests for capability disclosure
- model-backed correctness suites on at least two multimodal model families

## Benchmark and evaluation matrix

This future track needs model-backed workloads, not only helper arithmetic tests.

### Multimodal reliability workloads

1. image caption / grounding sanity
2. OCR-like document extraction
3. chart / table interpretation
4. repeated multimodal follow-up over the same static image or document
5. mixed text-image-text dialogue

### Metrics

Track at minimum:
- correctness / answer fidelity
- memory footprint
- prompt eval time
- decode throughput
- compaction latency
- reuse hit rate for pinned multimodal prefix
- wrong-answer regression count versus no-compaction baseline

### Required comparisons

1. no compaction
2. text-tail-only compaction
3. pinned-prefix reuse without compaction
4. later full multimodal compaction when implemented

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Visual prefix boundaries are inferred incorrectly | High | make prefix pinning explicit and sequence-local |
| M-RoPE support is claimed before spatial semantics are preserved | High | keep full multimodal compaction gated behind separate tests and capability flags |
| Multimodal server behavior fails silently when projector/model mismatch occurs | High | loud server startup and request-time validation |
| Prefix reuse leaks stale visual context across sequences | High | sequence ownership checks and save/restore tests |
| Performance work regresses text-only paths | Medium | keep dense text controls in CI and benchmark side-by-side |

## Recommended implementation order

1. Multimodal inference reliability in `llama-server`
2. Capability and status reporting for multimodal + compaction support
3. Pinned visual prefix metadata
4. Text-tail-only compaction
5. Save/restore for pinned multimodal state
6. Model-backed repeated-image/document tests
7. Only then begin full M-RoPE-aware multimodal compaction design

## Completion criteria for each phase

### Phase 1 complete when
- multimodal `llama-server` startup and request failures are explicit
- model/projector mismatch is observable and test-covered
- capability/status reporting is clear enough for clients and benchmarks

### Phase 2 complete when
- pinned multimodal prefixes survive repeated follow-up sessions and save/restore
- text-tail-only compaction reduces KV pressure without touching visual tokens
- server/API surfaces clearly state the active compaction mode

### Phase 3 complete when
- at least two multimodal model families pass model-backed correctness tests
- spatial token semantics are explicit in code and serialization
- M-RoPE-aware compaction no longer relies on text-only assumptions

## Out of scope for the current hybrid/BF16 plan

The current `PLAN-hybrid-compaction-and-bf16.md` should remain focused on:
- hybrid recurrent-attention budget handling
- Qwen3.5 text-only IMROPE behavior
- BF16 compacted-prefix validation and testing

Do not mix this future vision track into that implementation unless a concrete dependency is proven.
