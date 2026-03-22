# Final Local Training Status

This note freezes the current local learned-agent position after dataset v4, `pilot_v3`, and the first bounded RWR reruns.

## Current Best Control

- base checkpoint on the final v3 held-out optimize set: `2/5`

Reference:

- [batch_report.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/baselines/qwen7b_optimize_eval_base_v3/batch_report.json)

## Current Best Trained Checkpoint Artifact

- `pilot_v3`

References:

- [training_run_report.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/training/qwen7b_optimize_trace_qlora_pilot_v3/training_run_report.json)
- [checkpoint_suite_leaderboard.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/training/qwen7b_optimize_trace_qlora_pilot_v3/checkpoint_suite_leaderboard.json)
- [post_train_check.md](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/training/qwen7b_optimize_trace_qlora_pilot_v3/post_train_check.md)

## RL-Style Follow-Up

Two bounded reward-weighted regression reruns were executed on the final local surface:

- `rwr_v1`
- `rwr_v2`

References:

- [training_run_report.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/training/qwen7b_optimize_rwr_v1/training_run_report.json)
- [training_run_report.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/training/qwen7b_optimize_rwr_v2/training_run_report.json)
- [optimize_rwr_reward_report_v2.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/datasets/optimize_rwr_reward_report_v2.json)
- [checkpoint_suite_leaderboard.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/training/final_local_checkpoint_suite_v1/checkpoint_suite_leaderboard.json)
- [checkpoint_suite_report.md](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/training/final_local_checkpoint_suite_v1/checkpoint_suite_report.md)

## Honest Current Read

- the local learned-agent lane is real
- the local SFT pipeline is stable and repeatable
- the tuned checkpoints still do not beat the base model reliably on the strongest held-out positive optimize surface
- the first bounded RL-style reruns trained cleanly but also did not beat the `base_v3` control on the final five-task held-out suite

## Current Decision

- keep iterating locally, but stop treating more SFT alone as an automatically winning move
- use the final held-out v3 surface as the control benchmark
- treat the reward/data surface as usable for early RL-style experiments, but not yet sufficient for superiority claims
- keep `base_v3` as the strongest current control and keep `pilot_v3` as the strongest learned SFT artifact
- treat the next serious proof step as either:
  - better positive data on the harder branching and wide-softmax surfaces
  - or a stronger RL/data-design turn than the current bounded RWR reruns
