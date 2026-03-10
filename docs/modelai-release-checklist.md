# Release Management Checklist — modelai-llama.cpp

## Pre-Release Checks

- [ ] All CI passes on `modelai-main`
- [ ] Upstream base commit identified and recorded
- [ ] Included milestone scope identified
- [ ] All milestone tests pass
- [ ] Benchmark results collected for W1-W6 as applicable
- [ ] Benchmark results compared to previous release
- [ ] Known limitations documented
- [ ] Supported platform/model/backend matrix documented
- [ ] Rollback path verified
- [ ] ModelAI compatibility smoke tests pass

## Release Steps

- [ ] Create annotated tag on `modelai-main`
- [ ] Record upstream provenance in tag message: `Based on ggml-org/llama.cpp@<sha>`
- [ ] Build platform-specific binaries
- [ ] Publish or archive release artifacts
- [ ] Update ModelAI dependency pin to the new engine tag/SHA
- [ ] Record benchmark delta from the previous release
- [ ] Update release notes
- [ ] Record known caveats and unsupported matrix

## Post-Release Checks

- [ ] ModelAI smoke tests pass against the released engine tag
- [ ] No regression in W1-W6 key workloads beyond approved thresholds
- [ ] Previous release tag remains available for rollback
- [ ] CI benchmark artifacts archived and linked from release notes
- [ ] Release metadata stored with:
  - [ ] fork commit SHA
  - [ ] upstream base SHA
  - [ ] supported platforms
  - [ ] supported model/backend matrix
  - [ ] benchmark artifact references

## Required Release Notes Fields

- [ ] release tag
- [ ] fork commit SHA
- [ ] upstream base SHA
- [ ] included milestone range
- [ ] supported platform/backend matrix
- [ ] unsupported matrix
- [ ] benchmark summary
- [ ] known limitations
- [ ] rollback target
