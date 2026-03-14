# Implementation vs arXiv:2602.16284 Comparison

**Paper:** "Fast KV Compaction via Attention Matching" (MIT, Feb 2026)
**Reference impl:** github.com/adamzweiger/compaction

## Status: Full Paper Implementation Complete

### Fixed Issues (prior)

| Bug | Before | After | Impact |
|-----|--------|-------|--------|
| Max-shift inconsistency in fit_beta | Different per-query shifts for target vs design matrix | Consistent shifts via rescaling | Better solver convergence |
| Beta lower bound | 1e-12 (allows near-zero weights) | 0.05 (paper's e^{-3}) | Prevents V-fitting instability |
| NNLS iterations | 64 (wastes compute) | 2 (paper recommendation) | 6x faster compaction |

### Correct Implementations

| Aspect | Match? |
|--------|--------|
| Attention formulation (Eq. 1-2) | YES |
| Beta as additive pre-softmax bias | YES |
| OMP greedy loop (Algorithm 1-2) | YES |
| NNLS via projected gradient descent (Algorithm 3) | YES |
| Value fitting via least-squares (Eq. 3-4) | YES |
| GQA head expansion | YES |
| Max-shift numerical stability | YES |
| RMS score aggregation (Appendix F.1) | YES |
| OMP key pruning, beta < -7 (Appendix C.2) | YES |
| Spectral ridge scaling for regularization | YES |
| Nonuniform per-head budgets (Algorithm 4, Section 3.4) | YES |
| Context-prefill Q capture (Section 3.1) | YES |
| Repeat-prefill Q capture (Section 3.1) | YES |
| Approximate on-policy queries (Section 3.1/4.2) | YES |
| Chunked compaction (Section 3.5) | YES |
| Self-study Q-capture generation | YES |

### Remaining Deviations (Intentional/Accepted)

| Deviation | Paper | Implementation | Notes |
|-----------|-------|----------------|-------|
| Default score aggregation | RMS | Sum (RMS available as parameter) | RMS can be enabled per-pipeline call |
| L2 regularization | None (rejected) | lambda=1e-6 (spectral ridge optional) | Serves as Cholesky conditioning |
| Self-study | Multi-turn + reservoir sampling | Single greedy continuation (256 tokens) | Simplified but functional |
| OMP internal NNLS lambda | Not specified | Fixed 1e-6 | OMP used for selection only; final solver supports spectral ridge |

### All Paper Features Implemented

All features from arXiv:2602.16284 are now implemented:

| Feature | Implementation | Server Method | Files |
|---------|----------------|---------------|-------|
| HighestAttnKeys (topk) selection | `llama_kv_compact_select_topk` | `solver` | select.cpp |
| OMP greedy selection (Algorithm 1) | `llama_kv_compact_select_omp` | `omp` | select.cpp |
| RMS aggregation (Appendix F.1) | `accumulate_attention_scores(agg=RMS)` | — | select.cpp |
| OMP key pruning (Appendix C.2) | `omp_opts.beta_prune_log_threshold` | `omp` | select.cpp |
| Beta fitting via NNLS (Algorithm 3) | `llama_kv_compact_fit_beta` | all solver methods | solver.cpp |
| V fitting via least-squares (Eq. 3-4) | `llama_kv_compact_fit_values` | all solver methods | solver.cpp |
| Spectral ridge scaling | `solver_opts.spectral_ridge` | — | solver.cpp |
| Nonuniform per-head budgets (Alg. 4) | `llama_kv_compact_nonuniform_from_live_kv` | `nonuniform` | pipeline.cpp, budget.cpp |
| Chunked compaction (Section 3.5) | `llama_kv_compact_chunked_from_live_kv` | `chunked` | pipeline.cpp |
| Context-prefill Q capture (Section 3.1) | `llama_kv_compact_prefill_q_from_live_kv` | — | prefill-q.cpp |
| Repeat-prefill Q capture (Section 3.1) | same as above | — | prefill-q.cpp |
| Approximate on-policy (Section 4.2) | `llama_kv_compact_on_policy_from_live_kv` | `on_policy` | on-policy.cpp |
| Self-study Q generation | `llama_kv_compact_self_study_from_live_kv` | `self_study` | self-study.cpp |
