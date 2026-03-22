# Local Platform Status

This note freezes the final local-platform position for the current Modal-excluded scope.

## Current Local Policy

- `32B` local training on this GB10 stack remains deferred.
- `7B` local QLoRA is the active learned-agent path.
- GPT-5.4 remains the frontier teacher-policy and tooling-pressure baseline.

## Why 32B Is Deferred

The local `32B` path was blocked before first train-step execution on this GB10 software stack.

Ground truth references:

- [blocked_target_report.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/training/qwen32b_debug_repair_qlora_spark_smoke_v1/blocked_target_report.json)
- [final_spark_decision_memo.md](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/training/qwen7b_debug_repair_qlora_spark_smoke_v1/final_spark_decision_memo.md)

Working local policy:

- do not spend more local GB10 cycles on repeated `32B` retries without a materially different runtime variable
- keep `7B` as the active local training lane
- use GPT-5.4 to generate higher-value traces and pressure-test the environment

## What Is Stable Locally

- Spark runtime bootstrap
- Triton/task execution on the current bounded optimize surfaces
- `7B` smoke and pilot SFT
- repeatable local optimize-trace training
- local checkpoint eval on bounded optimize tasks

## What Is Still An Open Research Problem

- getting learned checkpoints to beat the base model reliably on positive held-out optimize tasks
- building a productive local RL loop on top of the new reward/data surface
- extending the bounded optimize loop to much harder GPU-engineering tasks
