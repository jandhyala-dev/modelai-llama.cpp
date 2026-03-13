# Implementation vs arXiv:2602.16284 Comparison

**Paper:** "Fast KV Compaction via Attention Matching" (MIT, Feb 2026)
**Reference impl:** github.com/adamzweiger/compaction

## Status: Bugs Fixed (3572cde3)

### Fixed Issues

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

### Remaining Deviations (Intentional/Accepted)

| Deviation | Paper | Implementation | Notes |
|-----------|-------|----------------|-------|
| Score aggregation | RMS | Sum | Minor quality impact |
| L2 regularization | None (rejected) | lambda=1e-6 | Serves as Cholesky conditioning |
| Query source | Context-prefill Q | K-as-surrogate (select/omp), real Q (self-study) | K-as-Q works well with Q normalization |
| Self-study | Multi-turn + reservoir sampling | Single greedy continuation (256 tokens) | Simplified but functional |

### Missing Features (Not Yet Implemented)

| Feature | Paper Reference | Priority | Impact |
|---------|----------------|----------|--------|
| Nonuniform head budgets | Section 3.4, Algorithm 4 | HIGH | Paper's most impactful ablation component |
| Context-prefill queries | Section 3.1 | HIGH | Free Q capture during prefill (no generation needed) |
| On-policy queries | Section 3.1 | MEDIUM | Sequential layer compaction with re-extracted Q |
| Chunked compaction | Section 3.5 | MEDIUM | Needed for contexts > n_ctx |
| RMS score aggregation | Appendix F.1 | LOW | More robust at extreme ratios |
| OMP key pruning (beta < -7) | Appendix C.2 | LOW | Stability improvement for OMP |
| Repeat-prefill queries | Section 3.1 | LOW | Alternative Q generation strategy |
