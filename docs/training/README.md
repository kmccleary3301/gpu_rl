# Training Preparation

This directory defines the minimum checked-in training-preparation surface for moving the project from environment work into a dedicated training environment.

## Scope

- validate the first SFT configuration against local manifests and checked-in golden fixtures
- validate the first scripted rollout configuration for narrow debug/diagnose/reformulate episodes
- preserve the first training target, assumptions, and smoke procedures as source-controlled artifacts

## Files

- `TRAINING_TARGET.md`: frozen first training target and stop conditions
- `REMOTE_BOOTSTRAP.md`: ordered bootstrap checklist for the target training environment
- `SMOKE_SEQUENCE.md`: exact smoke commands to run before any real training
- `CHECKLIST.md`: mechanical operator checklist for training preparation
- `configs/training/training_inventory_v1.json`: checked-in inventory of configs, scripts, docs, and expected outputs
- `training_manifest.json`: generated validation manifest built from the checked-in inventory, including resolved config refs and inventory-ref validation
- `scripts/run_training_preparation_verification.py`: one-shot verification sweep for the checked-in training-preparation surface

## Non-Goals

- no requirement that one specific machine or cluster topology be used
- no expectation that dense training is part of the default local workflow
- no orchestration layer beyond the checked-in smoke scaffolding
