# modelai-llama.cpp

> Production fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) adding **KV cache compaction** via Attention Matching.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/jandhyala-dev/modelai-llama.cpp/actions/workflows/modelai-ci.yml/badge.svg?branch=modelai-main)](https://github.com/jandhyala-dev/modelai-llama.cpp/actions/workflows/modelai-ci.yml)

## What This Fork Adds

Instead of truncating or evicting old context, KV cache compaction compresses the KV cache into a smaller learned representation that preserves attention behavior. The model produces near-identical outputs after compaction.

Based on ["Fast KV Compaction via Attention Matching"](https://arxiv.org/abs/2602.16284) (Zweiger et al., MIT Han Lab).

## Key Features

- **Attention Matching compaction** — lossless-quality KV cache compression (0.946-0.999 cosine similarity)
- **9 compaction methods** — select, solver, omp, self_study, chunked, on_policy, nonuniform, sequential_on_policy, context_prefill
- **Architecture support** — standard, iSWA, hybrid SSM+attention, IMROPE
- **Metal GPU acceleration** — solver runs on Apple Silicon GPU
- **Zero baseline overhead** — matches upstream decode speed within 2%
- **REST API** — `/compact` endpoint with configurable method and ratio

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

## Server API

The `/compact` endpoint is gated behind `--endpoint-compact` (disabled by default).

```bash
curl -X POST http://localhost:8080/compact \
  -H "Content-Type: application/json" \
  -d '{"id_slot": 0, "method": "select", "ratio": 4.0}'
```

**Methods:** `select` (fast, default), `solver` (higher quality), `omp`, `self_study`, `chunked`, `on_policy`, `nonuniform`, `sequential_on_policy`, `context_prefill`

**Response:**
```json
{
  "success": true,
  "method": "select",
  "compacted_tokens": 512,
  "original_tokens": 2048,
  "compression_ratio": 4.0,
  "compaction_time_ms": 45.2
}
```

## Support Matrix

| Feature | Status |
|---------|--------|
| Standard attention (Llama, Qwen, Gemma, Mistral) | Supported |
| iSWA (interleaved sliding window) | Supported |
| Hybrid SSM+Attention | Supported |
| IMROPE (multi-resolution RoPE) | Supported |
| Metal (Apple Silicon) GPU solver | Supported |
| State save/restore | Supported |
| Flash attention (zero-beta layers) | Supported |
| Quantized V cache (Q4_0, Q8_0) | Planned |
| CUDA GPU solver | Planned |
| Pure recurrent (Mamba) | Out of scope |

## Benchmarks

See [bench-results/](bench-results/) for raw benchmark data across 17 models and [bench-results/history.jsonl](bench-results/history.jsonl) for dashboard-ready time series.

## Testing

8 engine test tiers, 7 CI workflows:

```bash
# Run main test suite (48 tests)
ctest --test-dir build -L main --output-on-failure

# Run server pytests
cd tools/server/tests && python3 -m pytest unit/ -v -x -m "not slow"

# Run perf regression check
python3 scripts/perf-regression-check.py --model-path model.gguf
```

See [docs/CHANGELOG.md](docs/CHANGELOG.md#oss-launch-infrastructure) for the full test tier breakdown.

## ModelAI Integration

This fork exposes optional `modelai_*` Prometheus metrics and a contract endpoint used by the [ModelAI](https://github.com/jandhyala-dev/modelai) application. These hooks are not required for standalone use -- the server works identically to upstream llama.cpp when `--endpoint-compact` is not set.

## V0 Limitations

- Flash attention: layers with non-zero beta fall back to standard attention
- Quantized V types (Q4_0, Q8_0): not yet supported for compacted values
- M-RoPE (multi-modal positional encoding): untested
- Pure recurrent architectures: not applicable (no KV cache)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting issues, pull requests, and the review process. All compaction-related changes go through the adversarial review protocol documented in [AGENTS.md](AGENTS.md).

## Community & Support

This project is maintained on a **best-effort basis with no SLA**. Bug reports and feature requests are welcome via GitHub Issues. We aim to triage issues weekly but response times may vary.

**Scope:** This fork focuses on KV cache compaction. Features unrelated to compaction should be contributed upstream to [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

## Documentation
- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

- [HIGHLIGHTS.md](docs/HIGHLIGHTS.md) -- Full benchmark tables and feature summary
- [BUGS-AND-FIXES.md](docs/BUGS-AND-FIXES.md) -- All bugs found and fixed
- [CHANGELOG.md](docs/CHANGELOG.md) -- Implementation history
- [DESIGN-DECISIONS.md](docs/DESIGN-DECISIONS.md) -- Architecture rationale
- [UPSTREAM-SYNC.md](docs/UPSTREAM-SYNC.md) -- Upstream sync process
- [AGENTS.md](AGENTS.md) -- AI agent review protocol

## Upstream Relationship

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). Automated weekly sync every Saturday (CI-gated: merge + build + test). Three branches: `modelai-main` (working), `upstream-master` (tracking), `upstream-sync` (staging). All compaction code lives in `src/llama-kv-compact-*` files with minimal hooks into upstream code.

Related upstream discussion: [ggml-org/llama.cpp#20037](https://github.com/ggml-org/llama.cpp/issues/20037)

## License

MIT -- same as upstream llama.cpp. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.
