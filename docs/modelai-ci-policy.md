# CI Policy — modelai-llama.cpp

## Purpose

CI exists to prevent unstable engine states from entering `modelai-main` and becoming a ModelAI dependency.

## CI Triggers

Run CI on:
1. every push to feature branches
2. every proposed merge into `modelai-main`
3. every upstream merge into `modelai-main`
4. every release candidate tag

## Build Matrix

### Phase 1-2 minimum

- macOS Apple Silicon / Metal
- Linux CPU

### Phase 3+

- macOS Apple Silicon / Metal
- Linux CPU
- Linux CUDA when the CUDA path becomes relevant to supported releases

## Core Jobs

### 1. Build

Required on all relevant milestones.

Commands:
```bash
cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build --config Release -j
```

### 2. Core Test Suite

Required whenever engine code changes.

Commands:
```bash
ctest --test-dir build --output-on-failure
```

### 3. Backend / Operator Tests

Required whenever backend math, GGML operators, or attention kernels change.

Commands:
```bash
./build/bin/test-backend-ops
```

### 4. Runtime / Perf Smoke Tests

Required on milestones that affect runtime behavior, memory layout, or performance.

Commands:
```bash
./build/bin/llama-bench ...
./build/bin/llama-perplexity ...
```

### 5. ModelAI Compatibility Smoke Tests

Required once ModelAI begins depending on the engine.

Must cover:
1. HTTP request/response via `llama-server`
2. streaming response path
3. `GET /props`, `GET /models`, and `GET /metrics` capability/telemetry sanity path
4. long-session sanity path
5. save/restore sanity path when that feature exists
6. contract-version and provenance fields (`modelai.contract`) are present and non-empty

CI jobs that configure the engine must fetch `upstream-master` before the configure step so `upstream_base_commit` is populated deterministically in managed builds.

## Milestone Gates

| Milestone | Required CI |
|---|---|
| PR-0 docs | docs review only |
| PR-1 observability | build + `ctest` + server smoke |
| PR-2 memory architecture | build + `ctest` + state/layout tests |
| PR-3 correctness | build + `ctest` + compacted-prefix execution/materialization fixtures + state regression |
| PR-4 session/state | build + `ctest` + save/restore tests |
| PR-5a runtime/perf slice | build + `ctest` + model-backed active-range regression + compacted-prefix perf harness output |
| PR-5b solver pipeline | build + `ctest` + quality tests + ModelAI-like benchmark proof + attached benchmark artifacts |
| PR-6 coverage | build + `ctest` + backend-specific regression suite |

## Benchmark Regression Detection

### Required workloads

- W1: 80K filing -> first answer
- W2: 80K filing -> 20 follow-up questions
- W3: executive summary generation
- W4: full research report generation
- W5: 3 concurrent sessions on 32GB
- W6: save/restore + continue

### Metrics to record per run

1. model name and quantization
2. backend
3. flash attention on/off
4. compaction on/off and ratio
5. prefill latency ms
6. first-token latency ms
7. decode throughput tok/s
8. allocated KV bytes
9. active KV length
10. quality delta vs full cache
11. query generation time ms
12. solver time ms

### Regression thresholds

- performance regression > 10% on a key metric requires investigation before release
- quality regression > 1% perplexity delta requires investigation before release
- upstream merge into `modelai-main` requires at minimum a W2 smoke run once the benchmark harness exists

## Failure Handling

If CI fails:
1. block promotion into `modelai-main`
2. fix on the feature branch or a short-lived integration branch
3. rerun CI
4. merge only after green state is restored

If an upstream merge fails CI:
1. keep `upstream-master` current
2. fix or isolate the conflict in an integration branch
3. merge into `modelai-main` only after CI passes

## GitHub Actions Structure

Recommended workflow split:
1. `modelai-ci.yml`
   - build matrix
   - `ctest`
2. `modelai-server-smoke.yml`
   - `llama-server` startup
   - HTTP request/response test
   - streaming test
3. `modelai-perf-smoke.yml`
   - `llama-bench`
   - `llama-perplexity`
   - artifact upload

## Artifact Policy

CI must archive:
1. benchmark outputs
2. test logs on failure
3. build metadata
4. release-candidate benchmark summaries

No benchmark claim belongs in planning or release documents unless measured outputs are attached or referenced.

For the current PR-5 slice, the minimum attached evidence is:
1. model-backed `test-kv-compacted-prefix-pack`
2. manual `test-kv-compacted-prefix-perf` output showing before/after `active_n_kv`
3. matching before/after decode tok/s from the same compacted execution slice


For `PR-5b`, CI evidence must include:
1. solver-path quality regression output
2. attached benchmark artifacts from a ModelAI-like workload
3. explicit reporting of quality delta alongside tok/s and active `n_kv`
