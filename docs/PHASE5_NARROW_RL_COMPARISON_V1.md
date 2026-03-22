# Phase 5 Narrow RL Comparison v1

## Scope

This note freezes the first bounded Phase 5 RL-style comparison on the frozen tranche-1 hard slice.

The source policy for both updates was:

- `qwen7b_phase5_artifact_feedback_distill_v1`

Source tranche:

- `artifacts/baselines/qwen7b_phase5_tranche1_eval_distill_v1`

The comparison intentionally stayed narrow:

- no broad curriculum change
- no larger model change
- no new environment changes during the run

## Variants

### Baseline narrow RWR

Builder:

- `scripts/build_optimize_rwr_dataset.py`

Config:

- `configs/training/sft_qwen7b_phase5_optimize_rwr_baseline_v1.json`

Intent:

- generic reward-weighted regression on the tranche-1 batch

### Branch-aware narrow RWR

Builder:

- `scripts/build_phase5_branch_aware_rwr_dataset.py`

Config:

- `configs/training/sft_qwen7b_phase5_optimize_rwr_branchaware_v1.json`

Intent:

- give extra credit to:
  - useful branch-heavy near-misses
  - branch/compare depth
  - successful promote closeout

## Training Outputs

- baseline: `artifacts/training/qwen7b_phase5_optimize_rwr_baseline_v1/training_run_report.json`
- branch-aware: `artifacts/training/qwen7b_phase5_optimize_rwr_branchaware_v1/training_run_report.json`

Both runs completed cleanly.

## Tranche-1 Eval

Frozen eval config:

- `configs/training/qwen7b_phase5_tranche1_eval_v1.json`

Results:

- artifact distill: `1/4`
- narrow RWR baseline: `1/4`
- narrow RWR branch-aware: `1/4`

Reports:

- distill: `artifacts/baselines/qwen7b_phase5_tranche1_eval_distill_v1/batch_report.json`
- baseline RWR: `artifacts/baselines/qwen7b_phase5_tranche1_eval_rwr_baseline_v1/batch_report.json`
- branch-aware RWR: `artifacts/baselines/qwen7b_phase5_tranche1_eval_rwr_branchaware_v1/batch_report.json`

Per-task read:

- `attention_score` positive: all three checkpoints still failed by budget exhaustion
- `kernelbench/level1/23_softmax_wide` positive: all three checkpoints still failed by budget exhaustion
- `kernelbench/level1/47_sum_reduction` positive: all three learned checkpoints succeeded
- `kernelbench/level1/23_softmax` negative:
  - artifact distill: `negative_trace_complete`
  - both RWR variants: `multi_candidate_negative_complete`

## Interpretation

- The first artifact-feedback distill step already captured the main available gain on this narrow tranche.
- The bounded RL-style updates trained cleanly but did not improve the headline tranche-1 result beyond the distill checkpoint.
- The branch-aware shaping variant did not produce a decisive improvement over generic narrow RWR.
- The remaining blocker is still the same:
  - the two hardest positive tasks need stronger search/control and stronger hard-positive teacher data more than another tiny local RL-style update.

## Decision

- Count Block E as complete.
- Keep `qwen7b_phase5_artifact_feedback_distill_v1` as the strongest current learned checkpoint on the frozen tranche-1 slice.
- Do not claim a branch-aware RL-style win from these bounded updates.
- Treat future RL work as contingent on either:
  - better hard-slice teacher data
  - stronger search/control support
  - or a different model-capacity lane
