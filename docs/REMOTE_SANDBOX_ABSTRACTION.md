# Remote Sandbox Abstraction

This document freezes the minimum neutral remote-session contract for the current phase.

## Objective

The next executor layer should support remote workspaces without baking in Modal-specific semantics.

The contract is intentionally narrow:

- `run`
- `put_file`
- `get_file`
- `sync_tree`
- `terminate`

## Session Identity

Each remote session should carry:

- `session_id`
- `executor_kind`
- `workspace_root`
- `cwd`
- environment map
- timeout policy
- sync policy
- artifact transfer policy

The checked-in contract model for this lives in `gpu_cockpit/contracts/remote_session.py`.

## Timeout Policy

The neutral timeout model is:

- command timeout
- idle timeout
- session TTL

This keeps local and remote scheduling concerns explicit without assuming any one provider's lifecycle model.

## Default Sync Allowlist

The default repo paths considered safe and sufficient for remote task execution are:

- `gpu_cockpit`
- `workloads`
- `configs`
- `knowledge`
- `scripts`
- `tests`
- `docs`
- `pyproject.toml`
- `README.md`

The default exclusions are:

- `.git/**`
- `.venv/**`
- `.local_pkgs/**`
- `artifacts/**`
- `runs/**`
- `__pycache__/**`
- `*.pyc`

This keeps transient state out of the default sync path while preserving the code, tasks, docs, and test fixtures needed to execute bounded episodes remotely.

## Artifact Transfer Semantics

Artifact movement should remain explicit rather than implicit filesystem magic.

- Run bundles are pulled from `runs/`
- Patch and candidate lineage artifacts are pulled from `runs/*/patches` and `runs/*/candidate`
- Replay packs are pulled from `runs/*/replay`
- Training and report outputs are pulled from `artifacts/`

The remote provider may store these differently internally, but the local-facing contract should preserve these relative roots.

## Source Compatibility Boundary

Existing local execution paths should remain source-compatible at the call site:

- `executor="local_host"` continues to mean direct local subprocess execution
- `executor="local_docker"` continues to mean local containerized execution

The remote-session abstraction is additive. It should not require current local runner, environment, or benchmark code to be rewritten around provider-specific objects.

## Implementation Guidance

- Keep provider selection out of the contract layer
- Keep sync policy declarative
- Reuse `CommandResult` for `run(...)` responses so local and remote command summaries stay comparable
- Allow remote implementations to translate `cwd`, environment variables, and transfer roots to provider-native APIs behind the interface

## Deferred Work

This document does not claim:

- a Modal implementation
- a remote bundle cache design
- a remote artifact deduplication protocol
- a remote multi-session scheduler

Those belong in the next implementation tranche after the neutral contract is exercised locally in tests.
