# ModelAI Llama.cpp Fork — Git Operating Policy

## Purpose

This repository is a private product fork of `ggml-org/llama.cpp` used as a runtime dependency for ModelAI.

Its operating goals are:
1. keep upstream `llama.cpp` improvements flowing into the product fork,
2. develop ModelAI-specific runtime improvements safely,
3. ship stable engine tags that ModelAI can pin to,
4. preserve an optional future upstream path without slowing product work.

## Repository Model

Primary repo:
- `https://github.com/jandhyala-dev/modelai-llama.cpp`

Reference repos:
- upstream engine source: `https://github.com/ggml-org/llama.cpp`
- optional public fork for future upstream work: `https://github.com/jandhyala-dev/llama.cpp`
- MIT Attention Matching reference repo: read-only algorithm reference only

The MIT repo must never become a code dependency of the engine.

## Branch Model

### `upstream-master`

Purpose:
- pristine mirror of upstream `ggml-org/llama.cpp/master`

Rules:
- never commit product changes here
- update by fast-forward only
- use only as upstream intake branch

### `modelai-main`

Purpose:
- stable internal shipping branch
- ModelAI pins only to commits or tags from this branch

Rules:
- receives upstream merges from `upstream-master`
- receives completed feature branch merges
- never rebase
- every merge must pass CI
- release tags are created from this branch only

### Feature branches

Examples:
- `kv-compact-pr0-docs`
- `kv-compact-pr1-observability`
- `kv-compact-pr2-memory-arch`
- `kv-compact-pr3-correctness`
- `kv-compact-pr4-session-state`
- `kv-compact-pr5-performance`
- `kv-compact-pr6-coverage`

Rules:
- always branch from `modelai-main`
- one milestone per branch
- rebase onto `modelai-main` when needed
- merge back only after local validation and CI success

## Merge Policy

| Branch | Receives | Method |
|---|---|---|
| `upstream-master` | upstream changes only | fast-forward only |
| `modelai-main` | `upstream-master` | merge commit |
| `modelai-main` | completed feature branches | merge commit |
| feature branches | `modelai-main` updates | rebase |

### Why merge for `modelai-main`

A merge-based history preserves the exact point where upstream entered the product branch and separates upstream intake from product work. This matters for runtime regressions and release triage.

### Why rebase for feature branches

Feature branches are temporary and local to a milestone. Rebasing them onto `modelai-main` keeps patch stacks small and conflict resolution localized.

## Upstream Sync Policy

### Cadence

Sync upstream:
1. weekly during active development
2. immediately for changes affecting:
   - KV cache or memory model
   - flash attention
   - state save/restore
   - server/runtime internals
   - `llama_memory_t`
3. before starting a new milestone

### Procedure

1. `git fetch upstream`
2. `git checkout upstream-master`
3. `git merge --ff-only upstream/master`
4. `git checkout modelai-main`
5. `git merge upstream-master`
6. run CI
7. if CI fails, fix on a short-lived integration branch before promotion

### Principle

The objective is not “always latest in production.”
The objective is “latest validated upstream.”

## CI Policy

CI is mandatory for:
1. every merge into `modelai-main`
2. every upstream merge into `modelai-main`
3. every release candidate tag

Minimum CI requirements are defined in `docs/modelai-ci-policy.md`.

## Branch Protection

GitHub-side protection is enabled on the product repo for the two critical branches:

- `modelai-main`
  - no force pushes
  - no deletions
- `upstream-master`
  - no force pushes
  - no deletions

Operational rule:
- CI remains the effective promotion gate even when GitHub protection exists
- branch protection prevents obvious damage; it does not replace release discipline

## Release Policy

Stable engine releases must be tagged from `modelai-main` only.

Tag format:
- Releases: `modelai-engine-v0.1`, `modelai-engine-v0.2`
- Release candidates: `modelai-engine-v0.1-rc1`, `modelai-engine-v0.1-rc2`

Every tag must be annotated with:
1. exact upstream base commit
2. included milestone range
3. supported platform/backend matrix
4. known limitations
5. benchmark summary reference

Example annotation:
- `Based on ggml-org/llama.cpp@<sha>`

## Change Isolation Rules

1. keep observability separate from memory representation changes
2. keep correctness work separate from performance work
3. keep session/state work separate from backend coverage work
4. avoid unrelated refactors
5. document supported/unsupported matrix at each milestone

These rules reduce merge pain and make upstream intake safer.

## ModelAI Dependency Policy

ModelAI treats this engine as a versioned dependency.

Rules:
1. ModelAI pins to an exact engine tag or SHA from `modelai-main`
2. engine upgrades are intentional, never automatic
3. every upgrade records:
   - previous engine version
   - new engine version
   - upstream base delta
   - benchmark delta
   - known regressions or caveats

## MIT Reference Policy

The MIT Attention Matching repo is:
- a reference input
- not vendored into the engine
- not a runtime dependency

Use it to:
1. validate algorithm details
2. compare correctness behavior
3. port ideas deliberately

Do not:
1. auto-sync it into the engine repo
2. depend on its internal code structure
3. import it into the product runtime

## Rollback Policy

If an upstream merge or milestone destabilizes `modelai-main`:
1. revert or back out the merge
2. keep feature work isolated on its branch
3. keep the previous release tag available
4. do not promote ModelAI to a new engine version until a new stable tag exists

## Ownership Policy

This fork may use AI assistance extensively because it is a private product repo.

Even so:
1. every promoted change must be understood by a human owner
2. every release tag must have a human-reviewed benchmark summary
3. every major architectural change must have a rollback path
4. upstream Track B work must be re-reviewed under upstream contribution constraints before submission
