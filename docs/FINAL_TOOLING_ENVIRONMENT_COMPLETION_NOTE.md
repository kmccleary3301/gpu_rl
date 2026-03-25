# Final Tooling/Environment Completion Note

This note freezes what is now considered complete in the current tooling/environment scope.

## What is final in scope

- candidate-centric optimize environment with:
  - patch
  - branch
  - revert
  - promote
  - best-known tracking
  - endgame recommendations
- default hard-optimize packet spine with:
  - task card
  - candidate brief
  - candidate-tree brief
  - compare brief
  - localization brief
  - budget brief
- benchmark artifacts with:
  - timing method
  - compile/runtime split
  - protocol version
  - command digests
  - hardware fingerprint
  - provenance summary
- canonical hard-freeze corpus:
  - `artifacts/training/phase5_hard_trace_freeze_v2`
- aligned learning views from the same freeze:
  - artifact-feedback distill
  - teacher-correction
  - freeze-based narrow RWR
  - pairwise ranking
- KernelBench-v3 first stable slice with explicit official-vs-curated provenance
- limited local/remote semantic parity validated through the neutral remote-session abstraction on a candidate-bearing task

## What is intentionally out of scope

- broad remote-provider execution and scale-out
- model-capability claims beyond what the frozen benchmark slices show
- proving learned-agent superiority on the hardest positives
- solving the north star itself

## What remaining failures should now be blamed on

Unless a new regression is introduced, remaining failures on the active hard-task slices should now be treated primarily as one or more of:

- model capability limits
- search/control limits
- teacher-data limitations
- reward/data-design limitations

rather than missing environment semantics.
