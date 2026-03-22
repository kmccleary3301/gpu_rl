# Phase 3 Optimize Loop Note

Phase 3 is now materially beyond a single scripted patch lane and into a bounded candidate-engineering loop with a reusable two-attempt interface.

Current loop surface:

- `patch_candidate`
- `branch_candidate`
- `revert_candidate`
- `promote_candidate`
- `bench`
- `compare`
- `eval`

Current contract properties:

- candidate lineage is explicit in run bundles
- comparison includes candidate delta and optimize delta summaries
- observations surface candidate lineage and candidate delta briefs
- public and internal optimize lanes share a registry-backed patch-plan surface

Verified behavior so far:

- internal optimize lanes can execute a real branch/revert negative loop
- internal optimize lanes can execute a real branch/promote positive loop
- GPT-5.4 can complete bounded optimize loops on internal and public tasks
- GPT-5.4 can complete multi-candidate positive loops on both an internal and a public task
- GPT-5.4 can complete a true two-attempt positive loop on an internal task where candidate B only becomes available after comparing candidate A
- GPT-5.4 can produce a structured harder public near-miss on the same two-attempt interface instead of an opaque failure
- GPT-5.4 can complete a true three-attempt positive loop on a branching-aware internal row-sum task where candidate C only becomes available after a second compare-gated branch
- compare is now a first-class optimize step rather than optional post-hoc analysis
- reward and governance are now split in the trajectory/eval contracts
- optimize-trace dataset v3 now includes positives, negatives, near-misses, and multi-candidate traces across internal and public tasks
- GPT-5.4 ablation v2 indicates the default trace-generation surface should keep compare digest, failure localization, and two-attempt branching together
- a real non-smoke local 7B QLoRA pilot has been trained on the optimize-trace v3 slice, and a repeatability run with the same config also succeeded cleanly
- base-vs-tuned checkpoint eval now exists for a broader five-task bounded optimize set
- pilot v2 is the official local learned-agent baseline for the next tranche, while the untuned base checkpoint remains the control because headline held-out success is still tied
- the learned-agent lane is real, but positive held-out solve-rate improvement remains narrow and task-dependent

Current local baseline references:

- interface ablation: [ablation_report.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/baselines/gpt54_optimize_ablation_v2/ablation_report.json)
- dataset v3 manifest: [optimize_trace_manifest.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/training/optimize_trace_dataset_v3/optimize_trace_manifest.json)
- deeper branching exemplar: [task__reduction_row_sum_branching__eval__v1__positive.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/baselines/gpt54_reduction_row_sum_three_attempt_positive_probe_v1/batch_v2_branching_task_retry1/task__reduction_row_sum_branching__eval__v1__positive.json)
- pilot v2 training report: [training_run_report.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/training/qwen7b_optimize_trace_qlora_pilot_v2/training_run_report.json)
- repeatability training report: [training_run_report.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/training/qwen7b_optimize_trace_qlora_pilot_v2_repeat/training_run_report.json)
- checkpoint suite leaderboard: [checkpoint_suite_leaderboard.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/training/qwen7b_optimize_trace_qlora_pilot_v2/checkpoint_suite_leaderboard.json)

Immediate next pressure points:

- increase positive multi-candidate coverage on harder public tasks before spending more compute on longer local SFT
- improve held-out positive closeout examples for the learned-agent lane
- add stronger reward-model and RL-ready shaping on top of the current ledger surface
- push the same loop into the first hard-task ladder where the north-star gap is still large
