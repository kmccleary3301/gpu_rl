# Benchmark Fidelity And Anti-Stale Note

This note freezes the current benchmark-fidelity semantics for the Phase 5 tooling/environment substrate.

## What is now explicit

- benchmark artifacts carry:
  - timing method
  - compile-vs-runtime split
  - benchmark protocol version
  - command digests
  - hardware fingerprint when available
  - benchmark provenance summary
- compare packets surface perf localization from those benchmark artifacts instead of only a raw p50 delta
- candidate-mutation actions now mark prior perf state as invalidated in environment metadata:
  - `stale_perf_invalidated: true`
  - `stale_perf_invalidation_reason: patch_candidate | branch_candidate | revert_candidate | promote_candidate`

## Trusted semantics

- run bundles are immutable and compare always references explicit run refs
- command digests make benchmark-input drift visible
- compile-time and steady-state runtime are separated in the benchmark payload
- local/remote parity has been validated on a candidate-bearing attention-score episode via `scripts/run_phase5_remote_parity_probe.py`

## Anti-hack posture

- stale benchmark reuse is treated as invalid once candidate code changes
- compare/branch/promote actions only become learning-useful when they are attached to real run-producing evidence
- reward shaping no longer grants compare or near-miss bonuses for pure protocol usage without real optimize evidence

## Known remaining caveats

- live timing numbers remain noisy across runs on GB10 even when semantics match
- the current fidelity layer does not yet include deeper profiler-derived hotspot or launch-shape inference for every task family
- official held-out KB-v3 tasks currently enter the hard freeze as reference-only teacher traces, not mutation-heavy optimize traces
