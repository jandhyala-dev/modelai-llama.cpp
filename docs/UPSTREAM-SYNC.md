# Upstream Sync Process

modelai-llama.cpp tracks [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and merges upstream changes on a regular cadence. This document describes the process and what has been validated.

---

## Branch Model

| Branch | Purpose |
|--------|---------|
| `modelai-main` | Working branch. All development, PRs, and releases. |
| `upstream-master` | Clean upstream tracking. Read-only mirror of `ggml-org/llama.cpp:master`. |
| `upstream-sync` | Single reusable sync staging branch. All merge history in one place. |

Three long-lived branches, no sprawl. Feature work uses short-lived branches off `modelai-main`.

## Sync Cadence

- **Weekly:** Automated CI sync every Saturday 2PM PDT (`modelai-upstream-sync.yml`). Updates `upstream-master` tracking branch, merges `upstream/master` into `upstream-sync`, builds, tests, and auto-merges into `modelai-main`. Opens a GitHub Issue on failure.
- **Emergency:** Security patches (e.g., RCE fixes) are synced and merged same-day via `workflow_dispatch`.

## What Gets Validated on Each Merge

1. **Build:** `cmake -B build -DGGML_METAL=ON && cmake --build build --config Release`
2. **Tests:** `ctest --test-dir build -L main --output-on-failure` — all CI-gated tests must pass
3. **Conflict resolution:** Merge conflicts in compaction files are resolved manually and re-tested
4. **CI workflow audit:** New upstream workflows are disabled to prevent billing drain on the fork (only `modelai-ci`, `modelai-server-smoke`, `modelai-perf-smoke` are active)
5. **Upstream KV cache watch:** Check for open PRs or merged changes that touch KV cache internals (see Weekly Upstream Watch below)

## Weekly Upstream Watch

Runs every Saturday at 2 PM MT via GitHub Actions (`modelai-upstream-watch.yml`). Checks for upstream activity that could affect our compaction code:

**1. Open PRs touching KV cache:**
```bash
gh search prs --repo ggml-org/llama.cpp --state open -q "kv cache" --json number,title
```

**2. Recent upstream commits to KV cache files:**
```bash
gh api "repos/ggml-org/llama.cpp/commits?path=src/llama-kv-cache.cpp&since=$(date -v-7d +%Y-%m-%dT%H:%M:%SZ)" --jq '.[].commit.message | split("\n")[0]'
gh api "repos/ggml-org/llama.cpp/commits?path=src/llama-kv-cache.h&since=$(date -v-7d +%Y-%m-%dT%H:%M:%SZ)" --jq '.[].commit.message | split("\n")[0]'
```

If either check finds results, the workflow creates a GitHub Issue tagged `upstream-watch` with the findings. No email — just an issue to review during the next sync.

## Sync History

| Date | Commits Synced | Notable Changes | Merge Commit |
|------|---------------|-----------------|-------------|
| 2026-03-23 | 223 | Metal mul_mv_ext, CUDA bf16 flash attention, grammar fixes | `356b23be3` |
| 2026-03-23 | 2 (emergency) | **RPC RCE security patch** (#20908), PR template | `6618b8889` |
| 2026-03-23 | 8 | Per-platform CI split, server improvements | `d4a3d9ae9` |
| 2026-03-23 | 14 CI workflows disabled | Upstream split build.yml into per-platform files | `4e618aac9` |

## Upstream Issues Verified Compatible

Before shipping compaction, we audited every upstream KV cache change that could interact with our code:

| Issue | Title | Verification |
|-------|-------|-------------|
| [#10873](https://github.com/ggml-org/llama.cpp/issues/10873) | KV cache defrag corruption | Safe — defrag was removed upstream; compaction uses `seq_rm()` not defrag |
| [#12695](https://github.com/ggml-org/llama.cpp/issues/12695) | KV guard refactor | Compatible — fork uses current post-refactor API (`used_max_p1`, `is_empty`, `seq_has`, `pos_get`) |
| [#13194](https://github.com/ggml-org/llama.cpp/issues/13194) | SWA KV cache support | Compatible — SWA sub-cache correctly rejected at runtime (`n_swa > 0` check) |
| [#17450](https://github.com/ggml-org/llama.cpp/issues/17450) | Unified KV buffer | Compatible — all tests use `kv_unified=true` |
| [#12253](https://github.com/ggml-org/llama.cpp/issues/12253) | Shift/defrag correctness | Safe — `has_shift()` guard prevents interaction with compaction |
| [#11213](https://github.com/ggml-org/llama.cpp/issues/11213) | KV cells unified | N/A — decomposed into #12695 and #13194 |

## Upstream Features Targeted for Sync

| Feature | Upstream PR | Impact |
|---------|-----------|--------|
| Fused multiply-add for Q4/Q5/Q6_K | [#20032](https://github.com/ggml-org/llama.cpp/pull/20032) | 16–28% faster quantized matmul |
| Metal mul_mv_ext | [#20250](https://github.com/ggml-org/llama.cpp/pull/20250) | Metal performance improvement |
| High-throughput mode | [#14363](https://github.com/ggml-org/llama.cpp/pull/14363) | Multi-user serving optimization |

## Open Upstream Bugs Affecting This Fork

| Issue | Severity | Description |
|-------|----------|-------------|
| [#19679](https://github.com/ggml-org/llama.cpp/issues/19679), [#19304](https://github.com/ggml-org/llama.cpp/issues/19304) | Critical | Grammar stack crash on Apple Metal (flash attention + jinja, or 86K context + 50 tool calls) |
| [#11970](https://github.com/ggml-org/llama.cpp/issues/11970) | Medium | KV cache truncation on chat completions — silent context loss |

## CI Workflow Management

Forking llama.cpp inherits all upstream GitHub Actions workflows. We disable inherited workflows to prevent:
- CI billing drain (minutes consumed on every push)
- Self-hosted runner queue failures (runners don't exist in the fork)

**Active workflows (7):** `modelai-ci`, `modelai-server-smoke`, `modelai-perf-smoke`, `modelai-ci-windows`, `modelai-upstream-sync`, `modelai-dashboard`, `modelai-auto-label`
**Disabled:** All inherited upstream workflows (renamed to `.disabled`)

When upstream adds new workflow files (e.g., splitting `build.yml` into per-platform files), they are disabled in the next sync commit.
