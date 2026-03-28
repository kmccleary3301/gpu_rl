# Phase 5A Tranche Update

## Scope

This update freezes the main repo-side changes from the current Phase 5A tranche:

- reliable partial-artifact writing for local checkpoint eval
- the first bounded Phase 5A branch-aware RWR-style learner run
- the new `kv_cache_gather_hard` task lane
- the first GPT-5.4 probe on that harder KV task

## Main empirical read

- Hard-pair learner comparison on `attention_score` and `softmax_wide` stayed at `0/2` for:
  - `base`
  - `distill_v2`
  - `teacher_corrected_v1`
  - `rwr_branchaware_v1`
- The bounded RWR run trained cleanly, but did not beat the existing learner views.
- The new `kv_cache_gather_hard` GPT-5.4 probe produced a useful harder-task near miss rather than a substrate failure.

## Repo-side implementation changes

- `scripts/run_local_optimize_checkpoint_eval.py`
  - writes per-task `*.partial.json` progress artifacts during long local eval runs
- `scripts/build_phase5a_branch_aware_rwr_dataset.py`
  - builds the Phase 5A branch-aware RWR training slice
- `gpu_cockpit/engine/optimize_patch_registry.py`
  - now includes `task/kv_cache_gather_hard/eval/v1` with a three-attempt candidate plan
- `workloads/tasks/kv_cache_gather_hard_eval_v1.json`
  - adds the new harder internal KV task surface
- `workloads/reference/triton_kv_cache_gather_hard_*`
  - add patchable, first optimize-ready, and superseding optimize-ready hard KV candidates
- `workloads/tests/kv_cache_gather_hard_hidden.py`
  - adds the harder hidden contract for the KV task

## Verification

- focused unit coverage was added for:
  - local partial eval artifact writing
  - hard KV task registry / patch-registry contract
- direct visible/hidden validation passed for both accepted hard-KV candidate summaries

## Artifact note

The tranche also produced local artifacts, reports, and dataset freezes under:

- `artifacts/`
- `datasets/`
- `docs_tmp/`

Those remain local because this repo intentionally ignores those paths.
