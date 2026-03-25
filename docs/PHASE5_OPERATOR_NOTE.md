# Phase 5 Operator Note

This note captures the minimal operator flow for the current hard optimize / policy-learning substrate.

## Hard-Slice Generation

1. Run or refresh the target GPT-5.4 tranche.
2. Freeze the tranche into a curated manifest.
3. Inspect task/outcome coverage before training use.

Current relevant builders:

- `scripts/build_phase5_hard_trace_freeze.py`
- `scripts/build_phase5_block_b_ablation_report.py`

## Packet Validation

Before trusting a new family or benchmark slice:

1. Inspect a representative run with `section="quality"` and `section="profile"`.
2. Confirm:
   - `candidate_tree_brief`
   - `compare_brief`
   - `failure_localization`
   - `perf_localization`
3. Confirm benchmark protocol artifacts exist when benchmarking is in play.

## Policy-Learning Exports

Current Phase 5 views:

- Artifact-feedback distill:
  - `scripts/build_phase5_artifact_feedback_distill_dataset.py`
- Teacher correction:
  - `scripts/build_phase5_teacher_corrected_dataset.py`
- Branch-aware narrow RL / RWR:
  - `scripts/build_phase5_branch_aware_rwr_dataset.py`

These should all be producible from the same frozen hard slice. If a new family needs special-case export logic, treat that as a substrate gap and fix the family integration instead of forking the pipeline.

## Benchmark-Fidelity Checks

For any benchmark-bearing family, verify:

- `perf/benchmark.json`
- `perf/benchmark_protocol.json`
- `perf/raw_timings.json`
- hardware fingerprint presence when available
- command digest presence
- compile/runtime split metadata

## Current Caveat

The local GB10 / Triton environment is still noisier than the pure schema/unit-test surface. Prefer focused validation on touched families instead of claiming repo-wide green unless the heavier runtime tests are actually run.
