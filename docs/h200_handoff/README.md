# H200 Handoff

This directory defines the minimum checked-in handoff surface for moving the project from environment work into a dedicated training environment.

## Scope

- validate the first SFT configuration against local manifests and checked-in golden fixtures
- validate the first scripted rollout configuration for narrow debug/diagnose/reformulate episodes
- preserve the first training target, assumptions, and smoke procedures as source-controlled artifacts

## Files

- `TRAINING_TARGET.md`: frozen first training target and stop conditions
- `REMOTE_BOOTSTRAP.md`: ordered bootstrap checklist for the target training environment
- `SMOKE_SEQUENCE.md`: exact smoke commands to run before any real training
- `handoff_manifest.json`: machine-readable inventory of configs, scripts, and required datasets

## Non-Goals

- no requirement that one specific machine or cluster topology be used
- no expectation that dense training is part of the default local workflow
- no orchestration layer beyond the checked-in smoke scaffolding
