# modelai-llama.cpp

> Production fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) adding **KV cache compaction** via Attention Matching.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## What This Fork Adds

Instead of truncating or evicting old context, KV cache compaction compresses the KV cache into a smaller learned representation that preserves attention behavior. The model produces near-identical outputs after compaction.

Based on ["Fast KV Compaction via Attention Matching"](https://arxiv.org/abs/2602.16284) (Zweiger et al., MIT Han Lab).

## Key Numbers

| Metric | Value |
|--------|-------|
| **Quality** | 0.946-0.999 logit cosine similarity across 15 models |
| **Decode speedup** | Up to +63% at 8K context (8x compression) |
| **Max effective context** | 256K tokens from 64K physical KV cache |
| **Baseline overhead** | Zero -- fork matches upstream decode speed within 2% |
| **Models validated** | 17 tested, 15 pass quality gate |
| **Architectures** | Standard, iSWA, hybrid SSM+attention, hybrid-iSWA, IMROPE |

## Build

```bash
cmake -B build -DGGML_METAL=ON     # macOS Apple Silicon
cmake -B build -DGGML_CUDA=ON      # Linux with NVIDIA GPU
cmake -B build                      # CPU only

cmake --build build --config Release -j$(nproc)
```

## Quick Start

```bash
# Run with auto-compaction at 2x ratio
./build/bin/llama-server \
    -m model.gguf \
    --endpoint-compact \
    -c 8192
```

## Documentation

- [HIGHLIGHTS.md](docs/HIGHLIGHTS.md) -- Full benchmark tables and feature summary
- [BUGS-AND-FIXES.md](docs/BUGS-AND-FIXES.md) -- All bugs found and fixed
- [CHANGELOG.md](docs/CHANGELOG.md) -- Implementation history
- [DESIGN-DECISIONS.md](docs/DESIGN-DECISIONS.md) -- Architecture rationale
- [UPSTREAM-SYNC.md](docs/UPSTREAM-SYNC.md) -- Upstream sync process
- [CONTRIBUTING.md](CONTRIBUTING.md) -- How to contribute
- [AGENTS.md](AGENTS.md) -- AI agent review protocol

## Upstream Relationship

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). We sync weekly with upstream and maintain full backwards compatibility. All compaction code lives in `src/llama-kv-compact-*` files with minimal hooks into upstream code.

## License

MIT -- same as upstream llama.cpp. See [LICENSE](LICENSE) for details.
