# Phase 5 Artifact-Feedback Distill v1

## Scope

This note freezes the first Phase 5 artifact-conditioned distillation experiment.

It uses the hard-slice freeze from:

- `artifacts/training/phase5_hard_trace_freeze_v1`

and trains a local `Qwen/Qwen2.5-Coder-7B-Instruct` QLoRA adapter on turn-level GPT-5.4 observation/action pairs taken from:

- hard positives on `attention_score`
- hard positives on `kernelbench/level1/23_softmax_wide`
- one internal hard near-miss
- branch/revert negatives on public softmax
- one auxiliary three-attempt branching positive

## Dataset

Builder:

- `scripts/build_phase5_artifact_feedback_distill_dataset.py`

Outputs:

- `datasets/phase5_artifact_feedback_distill_train_v1`
- `datasets/phase5_artifact_feedback_distill_dev_v1`

The dataset preserves:

- original observation packets
- hard-slice quality labels
- source block provenance
- teacher-selected action JSON

## Training Run

Config:

- `configs/training/sft_qwen7b_phase5_artifact_feedback_distill_v1.json`

Output:

- `artifacts/training/qwen7b_phase5_artifact_feedback_distill_v1/training_run_report.json`

Training completed cleanly on the local Spark lane.

## Tranche-1 Eval

Eval config:

- `configs/training/qwen7b_phase5_tranche1_eval_v1.json`

Control:

- `artifacts/baselines/qwen7b_phase5_tranche1_eval_base_v1/batch_report.json`

Distill checkpoint:

- `artifacts/baselines/qwen7b_phase5_tranche1_eval_distill_v1/batch_report.json`

Headline result:

- `base`: `0/4`
- `artifact_feedback_distill_v1`: `1/4`

Per-task read:

- `attention_score` positive: still failed by budget exhaustion
- `kernelbench/level1/23_softmax_wide` positive: still failed by budget exhaustion
- `kernelbench/level1/47_sum_reduction` positive: improved from failure to success
- `kernelbench/level1/23_softmax` negative: remained non-success, but shortened to a cleaner `negative_trace_complete`

## Interpretation

- This is the first Phase 5 learned-policy result that beats the current base control on the frozen tranche-1 slice.
- The win is narrow and should not be overstated.
- The gain appears on the positive control and in cleaner negative closeout behavior, not yet on the two hardest targets.
- The result supports the planner-guided thesis that artifact-conditioned policy learning is worth treating as a first-class lane.
- The remaining bottleneck on the hardest targets still looks like a mix of search/control and hard-positive data scarcity, not just missing local training plumbing.
