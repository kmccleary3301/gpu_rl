# Optimize Lane Contract

This document freezes the v3 bounded optimize-lane contract used by the GPT-5.4 harness and the scripted reference environment.

## Intended Loops

The base optimize lane is a candidate-centric loop:

`bench -> patch_candidate -> bench -> compare -> eval`

The model may inspect or query knowledge between these steps, but the lane is designed to converge on this shape.

The multi-candidate positive lane extends that loop:

`bench -> patch_candidate -> bench -> compare -> branch_candidate -> promote_candidate -> eval`

The multi-candidate negative lane extends it differently:

`bench -> patch_candidate -> bench -> compare -> branch_candidate -> revert_candidate -> eval`

The two-attempt positive lane is the current highest-value bounded optimize surface:

`bench -> patch_candidate (candidate A) -> bench -> compare -> branch_candidate -> patch_candidate (candidate B) -> bench -> compare -> promote_candidate -> eval`

The first deeper candidate-tree lane now extends that pattern:

`bench -> patch_candidate (candidate A) -> bench -> compare -> branch_candidate -> patch_candidate (candidate B) -> bench -> compare -> branch_candidate -> patch_candidate (candidate C) -> bench -> compare -> promote_candidate -> eval`

## Required Surfaces

Each bounded optimize task should provide:

- a baseline command
- a patchable candidate command
- a promoted optimize candidate command
- a hidden eval requirement that distinguishes the promoted candidate from the patchable candidate
- a patch target file and scripted patch text for positive and negative variants

## Controller Rules

For bounded optimize tasks:

- `bench` should happen before `eval`
- once a candidate exists, the candidate should be benchmarked before `eval`
- once both baseline and candidate benches exist, `compare` should happen before the closing `eval`
- once patch and compare budgets are spent, the only valid closeout action should be `eval`
- in `branch_promote_positive_v1`, `promote_candidate` is not valid before `branch_candidate`
- in `branch_promote_positive_v1`, `eval` is not valid before both `branch_candidate` and `promote_candidate`
- in `branch_revert_negative_v1`, `revert_candidate` is not valid before `branch_candidate`
- in `branch_revert_negative_v1`, `eval` is not valid before both `branch_candidate` and `revert_candidate`
- in `two_attempt_positive_v1`, the first candidate must be compared before `branch_candidate` becomes valid
- in `two_attempt_positive_v1`, the second candidate patch is not valid until the branch exists
- in `two_attempt_positive_v1`, the second compare must happen before `promote_candidate`
- in `two_attempt_positive_v1`, `eval` is only valid after the promoted second-attempt candidate exists
- in `three_attempt_positive_v1`, the second branch is not valid until the second compare has happened
- in `three_attempt_positive_v1`, the third candidate patch is not valid until the second branch exists
- in `three_attempt_positive_v1`, the third compare must happen before `promote_candidate`
- in `three_attempt_positive_v1`, `eval` is only valid after the promoted third-attempt candidate exists

## Build Rules

- `build` is only exposed when the task has a real pre/post build spec
- optimize tasks without a meaningful build surface should not advertise `build`

## Candidate Requirements

The promoted optimize candidate should preserve visible and hidden correctness while emitting an `optimization_summary` that identifies:

- the strategy change
- the promoted candidate ref
- the relevant baseline or source reference

Candidate state and compare output should also surface:

- current candidate role and role-group
- parent candidate ref
- sibling candidate refs when the current candidate has siblings
- current candidate attempt index
- best-known candidate ref and best-known reason
- a compact candidate tree brief in the observation packet
- a compact candidate delta brief in compare output
- best-known candidate tracking in the environment state

The default trace-generation surface for bounded optimize tasks is now:

- compare digest enabled
- localized failure payloads enabled
- bounded branch / revert / promote actions enabled
- two-attempt branching enabled on tasks that define attempt plans

## Canonical v3 Exemplars

- Internal positive multi-candidate:
  [task__reduction_row_sum__eval__v1__positive.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/baselines/gpt54_reduction_row_sum_multi_candidate_positive_probe_v1/batch_v2_retry1/task__reduction_row_sum__eval__v1__positive.json)
- Public positive multi-candidate:
  [task__kernelbench__level1__23_softmax__eval__v1__positive.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/baselines/gpt54_kernelbench_softmax_multi_candidate_positive_probe_v1/batch_v2_retry1/task__kernelbench__level1__23_softmax__eval__v1__positive.json)
- Public negative multi-candidate:
  [task__kernelbench__level1__23_softmax__eval__v1__negative.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/baselines/gpt54_kernelbench_softmax_multi_candidate_negative_probe_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax__eval__v1__negative.json)
- Internal two-attempt positive:
  [task__reduction_row_sum__eval__v1__positive.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/baselines/gpt54_reduction_row_sum_two_attempt_positive_probe_v1/batch_v1_retry1/task__reduction_row_sum__eval__v1__positive.json)
- Public harder two-attempt near-miss:
  [task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/baselines/gpt54_kernelbench_softmax_wide_two_attempt_positive_probe_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json)
- Internal deeper candidate-tree success:
  [task__reduction_row_sum_branching__eval__v1__positive.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/baselines/gpt54_reduction_row_sum_three_attempt_positive_probe_v1/batch_v2_branching_task_retry1/task__reduction_row_sum_branching__eval__v1__positive.json)

## Current Covered Tasks

- `task/attention_score/eval/v1`
- `task/reduction_row_sum/eval/v1`
- `task/kv_cache_gather/eval/v1`
- `task/kernelbench/level1/47_sum_reduction/eval/v1`
- `task/kernelbench/level1/23_softmax/eval/v1`
- `task/kernelbench/level1/23_softmax_wide/eval/v1`
- `task/reduction_row_sum_branching/eval/v1`
