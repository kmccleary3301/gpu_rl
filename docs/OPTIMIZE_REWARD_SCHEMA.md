# Optimize Reward Schema

This repo now treats optimize-task learning reward and dataset governance as separate signals.

## Principle

- Learning reward answers: "How much useful optimization signal is in this episode for policy learning?"
- Governance answers: "Is this run or episode admissible for benchmark reporting, SFT collection, or RL trace retention?"

These should not be conflated. Replay completeness, provenance completeness, and artifact hygiene are governance signals. They should not directly inflate learning reward.

## Learning Reward v1

Schema id: `optimize_reward_v1`

Core components:

- `task_success`: `0.60`
- `correctness`: `0.25`
- `determinism`: `0.10`
- `perf_improvement`: `0.05`

Shaping components:

- `tool_cost`: bounded negative shaping, clipped to `[-0.20, 0.0]`
- `compare_use_bonus`: optional positive shaping, currently `0.02`, only when a compare action actually surfaces candidate or optimize evidence on a patch-bearing trace
- `candidate_regression_penalty`: bounded negative shaping applied when compare shows the current candidate regressed against the best-known candidate
- `best_known_supersede_bonus`: small positive shaping when compare shows the current candidate recovered correctness or improved performance relative to the current best-known anchor
- `revert_recovery_bonus`: small positive shaping when the policy reverts after a recorded regression instead of continuing to drift
- `promote_closeout_bonus`: small positive shaping when a promoted candidate is part of a successful closeout
- `near_miss_progress_bonus`: small positive shaping for structured optimize near-misses that recover correctness and use patch plus compare correctly but still fail final closeout

Rules:

- Correctness and task success dominate the signal.
- `compare_use_bonus` is intentionally too small to dominate correctness and is not awarded for empty compare spam.
- `candidate_regression_penalty` is bounded and intended to discourage repeated low-value regressions without overpowering correctness.
- branch-aware shaping is intentionally small and only reflects useful loop behavior, not benchmark success
- Governance signals such as evidence/provenance completeness are excluded from the learning-reward total.

Excluded governance signals:

- `evidence_score`
- `required_artifact_completeness`
- `replay_completeness`
- `build_completeness`
- `profile_completeness`
- `provenance_completeness`
- `benchmark_reporting`
- `sft_collection`
- `rl_reward_trace_readiness`

## Artifacts

Run-level eval bundles now emit:

- `eval/learning_reward_trace.json`

Trajectory episodes now carry:

- `governance_score`
- `learning_reward_trace`
- `optimize_trace_snapshots`

`optimize_trace_snapshots` packages:

- candidate-state snapshots
- compare payload snapshots
- failure-localization snapshots

Dataset curation separately recognizes:

- `usable_positive`
- `usable_negative`
- `near_miss`

`near_miss` is a dataset-quality category, not a direct learning-reward class. It is used for training-readiness and preference-style ranking decisions rather than benchmark success claims.

In the final local v1 ledger, a bounded `near_miss_progress_bonus` exists only as shaping for RL-style optimization traces. It requires real correctness recovery and is intentionally too small to let near-misses outrank true successful closes.

## Why This Matters

This split lets us change data-admission policy without silently changing the reward a tuned or RL-trained model sees. It also makes optimize traces more stable as training data.
