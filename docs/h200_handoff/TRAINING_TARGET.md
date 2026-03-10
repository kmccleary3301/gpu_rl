# First Training Target

## Model Choice

- use a strong sub-40B code model on the `2xH200` node
- default target: `Qwen/Qwen3.5-Coder-32B`
- adaptation strategy: `LoRA` or equivalent PEFT-first path

## Training Order

1. build the pre-H200 train/dev corpora locally or on the remote node with `scripts/build_pre_h200_training_assets.py`
2. validate datasets and smoke configs on the remote node
3. run warm-start SFT on patch-bearing `debug` / `diagnose` / `reformulate` traces
4. evaluate on held-out scripted episodes
5. only then introduce narrow RL rollouts for bounded tool-use

## First Policy

- bounded tool-use
- primary verbs: `debug`, `diagnose`
- secondary verb: `reformulate`
- defer open-ended `optimize` as a first-wave RL target

## Success Metrics

- hidden debug-task solve rate improves over the scripted baseline
- packaged repair traces remain lineage-safe and governance-valid
- patch-bearing episodes are preferred over thin benchmark-only traces
- tool-use stays within bounded step budgets
- held-out scripted baseline remains reproducible from checked-in configs and scripts

## Checked-In References

- split config: `configs/training/first_target_splits_v1.json`
- held-out rollout baseline: `configs/training/rollout_debug_repair_heldout_v1.json`
- handoff inventory: `docs/h200_handoff/handoff_manifest.json`

## Stop Conditions

- if held-out hidden debug solve rate stalls or regresses after SFT
- if rollout traces drift toward benchmark-only or unusable governance classes
- if patch-bearing repairs are not preserved through packaging and evaluation
- if the remote node cannot validate the checked-in smoke configs and fixtures cleanly
