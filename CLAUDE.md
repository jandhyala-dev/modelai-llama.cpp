# CLAUDE.md — modelai-llama.cpp (Private Product Fork)

## Review Standards
All code reviews follow `docs/review-standards/hostile-review-protocol.md`.

## What This Repo Is

Private product fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) for:
1. **KV cache compaction** via Attention Matching (arXiv:2602.16284)
2. **ModelAI runtime integration** — `llama-server` as primary local inference engine

**Owner:** Ajay Jandhyala — ajay@model-ai.app (COT Labs / ModelAI)
**Repo:** `jandhyala-dev/modelai-llama.cpp`
**Stable branch:** `modelai-main`

## Key Documentation

- `docs/modelai-fork-summary.md` — Fork executive summary and goals
- `docs/modelai-kv-compaction-plan.md` — Staged implementation plan (PR-0 through PR-6)
- `docs/modelai-git-policy.md` — Branching, upstream sync, merge, and release governance
- `docs/modelai-ci-policy.md` — CI gates, jobs, regression thresholds, and artifact rules
- `docs/modelai-release-checklist.md` — Release promotion and rollback checklist

## Build Commands

```bash
# macOS Apple Silicon
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)

# Run tests
ctest --test-dir build -L main --output-on-failure

# Run a model
./build/bin/llama-cli -m /path/to/model.gguf -p "Hello" -n 128
```

## Branch Model

- `upstream-master` — pristine upstream intake
- `modelai-main` — stable shipping branch (merge-only)
- `kv-compact-*` — feature branches per milestone
- `feature/**` — other feature branches

## Code Conventions

- Follow llama.cpp style: C-style C++, minimal STL, GGML for tensor ops
- No external dependencies (cmake-only builds)
- Use `LLAMA_LOG_INFO/WARN/ERROR` for logging
- Prefix new compaction functions with `llama_kv_compact_`
- Commit style: short imperative messages, prefix `kv-compact:` for compaction work
- All solver math in fp32; cast to model dtype for KV storage

## V0 Support Matrix

**Supported:** standard causal models, non-flash attention, non-quantized V, uncompacted chat-template/BOS prefix, hybrid SSM+attention (attention layers only, if standard RoPE), iSWA (base layers only), IMROPE text-only models (Qwen3.5, Qwen3.5-MOE — V4-J)
**Unsupported:** flash attention, quantized V, M-RoPE models (Qwen2-VL, GLM4), pure recurrent (Mamba/RWKV), public API guarantees
