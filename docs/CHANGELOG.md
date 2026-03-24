# Changelog

Implementation history of KV cache compaction in modelai-llama.cpp, from first commit to production readiness. Each milestone built on the previous, with adversarial review gating every merge.

---

## V0: Foundation (PR-0 through PR-6b)

### PR-0: Fork Setup
- Forked ggml-org/llama.cpp, established `modelai-main` branch
- Added CI workflows (`modelai-ci`, `modelai-server-smoke`, `modelai-perf-smoke`)
- Disabled 21 inherited upstream workflows to prevent CI billing drain

### PR-1: Server Contract and Observability
- `/compact` REST endpoint for triggering compaction
- `/props` response extended with `compacted_prefix` state
- Prometheus gauges: `llamacpp:modelai_active_n_kv_total`, `llamacpp:modelai_active_n_kv_max`, `llamacpp:modelai_sequence_state_bytes_total`

### PR-2: Compacted-Prefix Memory Architecture
- Per-sequence compacted-prefix store inside `llama_kv_cache`
- Per-layer compacted payloads (K, V, beta)
- State serialization (save/restore)
- Sequence lifecycle management (copy, clear, remove)

### PR-3: Non-Flash Compacted-Prefix Execution
- Graph construction: prepend compacted K/V/beta tensors to live KV
- Additive `kq_b` (beta) injection for attention bias
- Causal mask column generation for compacted positions
- Non-flash attention path (required for non-zero beta)

### PR-4: Session-State Integration
- Full state serialization (version 1) for compacted prefix
- Save/restore across server restart
- Restore rollback on partial failure

### PR-5a: Live-KV Reclaim
- After compaction, reclaim live KV positions consumed by the compacted prefix
- Position-based `seq_rm()` for safe removal
- Shift guard: reject reclaim when KV shift is pending

### PR-5b: Solver Pipeline
Pure C++ dense fp32 solver for Attention Matching (arXiv:2602.16284):
- **NNLS beta fitting** — Projected Gradient Descent with lstsq+clamp fallback
- **Least-squares V fitting** — 3-tier cascade: LAPACK `sgels`, Cholesky, aggressive Cholesky
- **Top-k selection** (`select`) — Attention-score-weighted position selection, zero beta, flash-compatible
- **OMP selection** — Orthogonal Matching Pursuit with progressive schedule and drop-key refinement
- **Score aggregation** — 4 modes (SUM, RMS, MAX, MEAN) + optional 1D avgpool smoothing
- **Spectral ridge scaling** — Power iteration for regularization parameter
- **NEON-optimized** dot product for Apple Silicon
- Zero external dependencies

### PR-6: Coverage Expansion
- **Flash attention with zero-beta** — Zero-beta layers use flash attention; non-zero use standard path
- **OMP key selection** — Greedy residual correlation with cached selection order
- **Quantized V extraction** — Non-transposed and transposed V with block-aligned offsets (Q8_0, Q4_K)
- **iSWA base-layer support** — Compaction applies to base (non-SWA) cache; SWA sub-cache correctly rejected

### PR-6b: Self-Study Query Generation (21 slices, adversarial-reviewed)
- **cb_eval Q-capture** — Intercept post-RoPE Q tensors during autoregressive generation
- **GQA regrouping** — Map Q heads to KV heads with correct stride
- **Q subsampling** — Configurable max queries per KV head
- **Autoregressive generation loop** — Multi-round with configurable temperatures
- **Self-study solver integration** — Full fit pipeline with captured Q
- **Q-capture memory guard** — `max_q_capture_mb` (default 1 GB) with auto-reduce
- **M-RoPE safety guard** — Block compaction on spatial RoPE models (Qwen2-VL, GLM4)
- **Quantized transposed V extraction** — Per-row dequantization with block-aligned kv_size
- **Production workload test** — stories15M quality gate + benchmark harness
- **Long-context benchmark** — 4K-32K, 6 models, CSV output

---

## V1: 15-Model Validation

### Implementation
- **Upstream bug audit** — Verified compatibility with 6 upstream KV cache changes (#10873, #12695, #13194, #17450, #12253, #11213)
- **Per-stage timing instrumentation** — k_extraction, attention_score, selection, v_extraction, kv_write breakdown
- **Server hardening** — Method allowlist, experimental method gating

### Bug Fixes (3 Critical/Major)
- **BUG-I01** (Critical): Nonuniform pipeline catastrophic quality loss at 8K — fallback to `select` when >50% heads masked
- **BUG-I02** (Major): GPU throughput regression at 32K — `ggml_backend_tensor_set()` for backend-agnostic upload
- **BUG-U01** (Major): Gemma3-12B SWA 20x slower than Ollama — runtime warning, model listed unsupported
- **BUG-R01** (Minor): Exception safety in tensor staging — try/catch guards
- **BUG-R02** (Minor): Thread safety of static warning flags — `std::atomic<bool>` with `exchange()`

### Benchmark Results
- **17 models tested** — 15 pass quality gate (cosine >= 0.950 at 2x)
- **Zero regression** — Average pp512 delta +0.1%, tg128 delta -1.0% (within noise)
- **Best quality:** Qwen3-30B-A3B = 0.999, stories15M = 1.000
- **Incompatible:** Gemma3-12B (SWA performance), GPT-OSS-20B (unknown arch), Phi4-14B (GGML hash set)

---

## V2: Solver Rewrite + MIT Gap Closure

### Gap Analysis
Audited fork against arXiv:2602.16284 reference implementation. Identified 16 gaps (GAP-01 through GAP-16):
- NNLS solver fundamentally different from paper
- Self-study generated 50x fewer queries
- No progressive OMP schedule
- Entropy proxy instead of influence curves

### Implementation (6 phases)
1. **Solver core rewrite** — LAPACK `sgels` QR-based least-squares, 3-tier cascade, MIT-aligned bounds
2. **Query diversity** — Progressive OMP schedule, per-head budget JSON import
3. **Metal GPU solver** — GPU attention scores + XtX assembly (`llama-kv-compact-solver-metal.mm`)
4. **Q-capture memory fix** — BUG-M01: `max_q_capture_mb` guard (4.9 GB → 1.0 GB for Qwen3-14B)
5. **Chunked self-study** — Non-overlapping chunk-and-merge for contexts >8K, zerobeta fallback
6. **Per-layer flash eligibility** — Zero-beta layers use flash; non-zero use standard attention

### Bug Fixes
- **Q/K norm mismatch** (Critical): Per-head Q normalization (cosine 0.710 → 0.988-0.995)
- **3 solver bugs per arXiv review** (Major): Max-shift consistency, beta lower bound, solver stabilization
- **Biased K norm sampling** (Major): Uniform strided sampling instead of first-N
- **F16 mask cast guard** (Major): Non-flash ALiBi models
- **Chunked budget overshoot** (Major): Excess distribution + n_selected cap

---

## V5: Safety, Public API, and Production Hardening

### Sprint 1: Safety + Measurement Foundation
- Crash protection: budget floor (2 tok/head), NaN guard, total budget floor
- PPL benchmark tool (`tools/kv-compact-bench/`, 593 lines)

### Sprint 2: Public C API + Auto-Compaction
- `llama_compact_params`, `llama_compact_default_params()`
- `llama_kv_cache_compact()`, `llama_kv_cache_set_auto_compact()`
- Threshold-triggered auto-compaction with one-shot guard

### Sprint 3: OMP Resilience
- Per-head 5-second timeout with top-k fallback
- OMP >600s on 30B models → graceful degradation

### V4 Features (influence curves + IMROPE)
- **Influence-based per-head budgets** — Entropy-based allocation with influence-curve swap solver
- **IMROPE text-only support** — Qwen3.5, Qwen3.5-MOE safe for compaction. M-RoPE (spatial) correctly blocked
- **is_imrope serialization** — State version 2, survives save/restore and seq_cp

### Bug Fixes
- **Ablation flags dead code** (Major): `--no-beta`/`--no-cv` now functional
- **is_imrope serialization** (Major): Bump to state version 2
- **seq_cp not copying is_imrope** (Major): Flag propagation in sequence copy

---

## Phase 8: Iterative On-Policy + High Compression

### Implementation
- **Iterative on-policy refinement** — Quality-gated convergence: K-as-Q baseline → generate continuation → re-solve with captured Q → repeat until residual plateaus
- **Per-layer sequential on-policy** — Compact layer 0 → generate on-policy Q → compact layer 1 → ... exploits sequential dependency
- **Single-layer refit** — Re-fit beta/V for one layer without re-selecting positions
- **High-compression auto-tuning** — When ratio >= 10x, auto-apply stronger solver parameters
- **Residual reporting** — Mean partition-sum relative error for quality gate decisions
- **Tensor cache invalidation** — `bump_version()` after single-layer refit

### Benchmark Results
- **50x compression** on Qwen3-30B-A3B: cosine 0.9967, 10/10 fact recall
- **256K effective context** from 64K physical window: 49-57 compaction cycles, 100% recall at every checkpoint

---

## Phase D: End-to-End Test Harness

### Implementation
- **Phase D orchestrator** — Automated: start server → fill context → compact → verify → repeat
- **Model registry** — Per-model configs for benchmark automation
- **Aggregation** — Multi-run statistical analysis
- **E2E validation** — Qwen3-8B and Coder-30B verified through full compaction cycles

### Bug Fixes
- **Post-compaction amnesia** (Major): Server forgot conversation after compaction+reclaim
- **Post-compaction crash** (Critical): Stale prompt cache caused crash on next request

---

## OSS Launch Infrastructure

### Engine Test Tiers (8 tiers)
| Tier | Name | Implementation |
|------|------|---------------|
| 1 | Server pytests | `modelai-server-smoke.yml` — fast unit tests in CI |
| 2 | API contract snapshots | `test_api_contract.py` — schema validation against `snapshots/*.json` |
| 3 | Perf regression | `perf-regression-check.py` — speed/RSS/compact latency vs baseline |
| 4 | Quality gate | `test-kv-compact-quality-gate.cpp` — cosine floor, KV survival, state round-trip |
| 5 | Windows CI | `modelai-ci-windows.yml` — MSVC x64 CPU-only build + test |
| 6 | Stress tests | `test-kv-compact-exhaustion.cpp`, `test-kv-compact-parallel-slots.cpp` |
| 7 | ModelAI contract | `test_modelai_contract.py` — /props, /metrics, /compact contract validation |
| 8 | Live dashboard | `update-dashboard-data.py` + `modelai-dashboard.yml` → `history.jsonl` |

### CI Workflows (7 active)
| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `modelai-ci` | push, PR | Build + 48 main-label tests |
| `modelai-server-smoke` | push, PR | Server smoke + pytests |
| `modelai-perf-smoke` | push, PR | Performance regression detection |
| `modelai-ci-windows` | push, PR | Windows MSVC build + test |
| `modelai-upstream-sync` | Saturday 2PM PDT | Weekly upstream merge + build + test |
| `modelai-dashboard` | after CI/perf success | Aggregate bench data to history.jsonl |
| `modelai-auto-label` | issues, PRs | Auto-label by path/keyword |

---

## Upstream Sync

### Process
- Weekly CI sync of `ggml-org/llama.cpp:master` into `upstream-sync` branch (Saturday 2PM PDT)
- 3 long-lived branches: `modelai-main`, `upstream-master`, `upstream-sync`
- Emergency same-day sync for security patches

### Sync Events
| Date | Commits | Notable |
|------|---------|---------|
| 2026-03-23 | 223 | Metal mul_mv_ext, CUDA bf16 flash attention |
| 2026-03-23 | 2 (emergency) | **RPC RCE security patch** (#20908) |

### Upstream Compatibility
6 upstream KV cache changes verified compatible before V1 ship:
- #10873 (defrag), #12695 (KV guard), #13194 (SWA), #17450 (unified buffer), #12253 (shift/defrag), #11213 (cells unified)

---

## By the Numbers

| Metric | Value |
|--------|-------|
| Fork-specific commits | 150+ |
| Fix commits | 29 |
| Critical/Major bugs fixed | 18 |
| Models validated | 17 (15 pass quality gate) |
| Architectures supported | 5 (standard, iSWA, hybrid, hybrid-iSWA, IMROPE) |
| Compaction pipelines | 7 (select, solver, OMP, self-study, chunked, on-policy, sequential) |
| Best quality (cosine) | 0.999 (Qwen3-30B-A3B at 4K/2x) |
| Best speedup | +63% decode at 8K/8x (Qwen3-8B) |
| Max effective context | 256K from 64K physical (100% recall) |
| C++ test points | 51 CI-gated, 64 total |
| Engine test tiers | 8 (pytests → live dashboard) |
| CI workflows | 7 active |
| Adversarial review rounds | Every merge gated |
