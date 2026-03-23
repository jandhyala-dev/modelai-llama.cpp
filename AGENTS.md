# AGENTS.md — modelai-llama.cpp

This file provides context for AI agents (Claude Code, Codex, Gemini, etc.) working on this repository.

## Repository

Production fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) adding KV cache compaction via Attention Matching ([arXiv:2602.16284](https://arxiv.org/abs/2602.16284)).

**Stable branch:** `modelai-main`

## Build

```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
ctest --test-dir build -L main --output-on-failure
```

## Code Conventions

- Follow llama.cpp style: C-style C++, minimal STL, GGML for tensor ops
- No external dependencies (cmake-only builds)
- Use `LLAMA_LOG_INFO/WARN/ERROR` for logging
- Prefix compaction functions with `llama_kv_compact_`
- All solver math in fp32; cast to model dtype for KV storage
- Commit style: short imperative messages, prefix `kv-compact:` for compaction work

## Review Protocol

All code entering this repo is reviewed using an adversarial, failure-seeking review protocol. The reviewer's job is to try to break the code, not confirm it looks reasonable. If there is any plausible correctness bug, the review must FAIL.

### Mandatory Sections (all 13 required)

1. **Scope Gate** — What is in/out of scope. Flag scope leaks.
2. **Spec Match** — Implementation matches stated intent. Flag drift between comments, signatures, and behavior.
3. **Contract Boundary Check** — Caller/callee assumptions match. Preconditions enforced. Postconditions hold.
4. **Concrete Traces** — Walk real numbers through the code:
   - Production trace (dominant real-world case)
   - Boundary trace (smallest/threshold inputs)
   - Adversarial trace (hostile case designed to break the code)
   - Integer arithmetic trace (every division, modulo, stride)
   - Security trace (when applicable)
   - Concurrency trace (when applicable)
5. **Multi-Variant Model Trace** — Trace through different architectures (GQA, iSWA, hybrid).
6. **State-Machine Trace** — Before/after/failure/rollback states for persistent mutations.
7. **Unsupported / Precondition Audit** — Every assumption listed: enforced, documented, or undocumented.
8. **Test Reality Check** — Tests cover production + boundary cases and would catch identified failures.
9. **Disprove-It Pass** — Assume one bug exists. Systematically try to find it. Mandatory before PASS.
10. **Dependency Check** — License, CVEs, version pinning.
11. **Performance Regression Check** — Before/after for hot path changes.
12. **Cross-Repo Contract Check** — Producer/consumer match for shared interfaces.
13. **Pass Bar** — PASS only if no plausible production bug remains after all traces.

### Severities

| Severity | Meaning | Blocks? |
|----------|---------|---------|
| Critical | Data corruption, security, crash | Yes |
| Major | Correctness bug, wrong results | Yes |
| Minor | Suboptimal but correct | No |

## Key Documentation

- `docs/kv-compaction-algorithm.md` — Algorithm overview
- `docs/kv-compaction-integration.md` — File map and architecture support matrix
- `docs/BUGS-AND-FIXES.md` — All bugs found, fixed, and tracked
- `docs/UPSTREAM-SYNC.md` — Upstream sync process and verification
- `docs/benchmark-fork-vs-upstream.md` — Performance comparison

## Support Matrix

**Supported:** Standard causal models, iSWA (base layers), hybrid SSM+attention (attention layers, standard RoPE), IMROPE text-only (Qwen3.5), quantized K (Q8_0, Q4_K)

**Unsupported:** Flash attention with non-zero beta, M-RoPE (Qwen2-VL, GLM4), pure recurrent (Mamba/RWKV), SWA sub-cache
