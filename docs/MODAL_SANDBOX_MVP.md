# Modal Sandbox MVP

This document freezes the first provider-backed remote execution path behind the neutral remote-session contract.

Current planning status:

- implementation is kept checked in
- real provider execution is deferred to the end-of-project externalization phase unless remote scale becomes the active bottleneck earlier

## Scope

The Modal MVP is intentionally narrow:

- create a remote workspace session
- sync the local repo subset into the remote workspace
- run a command remotely
- pull `runs/` and `artifacts/` back to the local workspace
- terminate the remote session

The implementation lives in:

- `gpu_cockpit/executors/modal_remote_session.py`

## Operator Config

The Modal path is activated with `--executor modal` through the normal CLI flow after the optional dependency is installed.

Relevant environment variables:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- `GPU_COCKPIT_MODAL_APP_NAME`
- `GPU_COCKPIT_MODAL_ENVIRONMENT`
- `GPU_COCKPIT_MODAL_PYTHON_VERSION`
- `GPU_COCKPIT_MODAL_GPU`
- `GPU_COCKPIT_MODAL_CPU`
- `GPU_COCKPIT_MODAL_MEMORY_MB`
- `GPU_COCKPIT_MODAL_PIP_PACKAGES`

Credential resolution order in the current repo path:

- current shell environment
- `gpu_rl/.env` through `scripts/run_with_spark_env.sh`
- Modal's standard config files if present

Default behavior:

- base image: `modal.Image.debian_slim(python_version=3.12)`
- apt packages: `bash`, `coreutils`, `findutils`, `tar`
- default Python package set: `pydantic>=2.8,<3`
- default repo sync roots follow the neutral remote-session allowlist

## Execution Model

`ModalExecutor` creates a `ModalWorkspaceSession` behind the existing `CommandExecutor` surface.

For each command:

1. sync the allowed local repo roots into `/workspace`
2. execute the command in the remote workspace
3. pull back top-level artifact roots derived from the neutral artifact policy

This keeps the current `run`, `eval`, `bench`, and scripted environment code paths source-compatible at the call site.

## Intended First-Wave Uses

- smoke path:
  - `task/profile_diagnose/eval/v1`
- report path:
  - `configs/training/rollout_profile_diagnose_modal_smoke_v1.json`
  - later expansion to `configs/training/rollout_debug_repair_heldout_modal_v1.json` when the remote image can satisfy GPU-heavy tasks
- patch-bearing path:
  - `task/reduction_debug/eval/v1` or `task/attention_reformulate/eval/v1` once the remote image includes the needed GPU stack

## Current Limits

- system tracing, Triton build capture, and sanitizer flows remain local-only because the runner already gates them to `executor=local_host`
- GPU-heavy remote tasks still depend on the remote image carrying the needed runtime stack
- this repo does not currently ship Modal credentials, so real provider runs remain blocked until auth is configured
- because of that and because the local optimize/data/training tranche is now productive, Modal provider reruns are intentionally deferred rather than kept on the critical path
- direct evidence of the current auth block is archived under:
  - `artifacts/modal_mvp/modal_smoke_attempt_01.md`
  - `artifacts/modal_mvp/modal_eval_attempt_01.md`
  - `artifacts/modal_mvp/modal_patch_attempt_01.md`
  - `artifacts/modal_mvp/modal_auth_inspection.md`
