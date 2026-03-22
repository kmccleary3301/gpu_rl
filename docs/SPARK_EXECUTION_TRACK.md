# Spark Execution Track

This document freezes the immediate DGX Spark execution path for the current phase.

## Frozen First Target

- model_id: `Qwen/Qwen2.5-Coder-32B-Instruct`
- tokenizer_id: `Qwen/Qwen2.5-Coder-32B-Instruct`
- default Spark adapter path: `QLoRA`
- comparison adapter path: `LoRA`
- first task focus:
  - `debug`
  - `diagnose`
  - `reformulate`
- first dataset slice:
  - `datasets/first_target_transition_train_v1`
  - `datasets/first_target_transition_dev_v1`
  - `datasets/first_target_sft_train_v1`
  - `datasets/first_target_sft_dev_v1`

## Spark Box Reality

- host arch: `aarch64`
- gpu: `NVIDIA GB10`
- driver: `580.95.05`
- cuda runtime: `13.0`
- python: `3.12.3`
- torch wheel used for local execution: `2.10.0+cu130`
- triton wheel used for local execution: `3.6.0`

## Required Local Runtime Shim

The Spark runs on this box currently require:

- repo-local `.venv`
- repo-local extracted Python headers under `.local_pkgs/python312dev/extracted/usr/include`
- `PATH` preferring `.venv/bin`
- `C_INCLUDE_PATH` including:
  - `.local_pkgs/python312dev/extracted/usr/include/python3.12`
  - `.local_pkgs/python312dev/extracted/usr/include`

Use [`scripts/run_with_spark_env.sh`](../scripts/run_with_spark_env.sh) for reproducible local execution.

## Known Spark Degradations

- `nvidia-smi` reports non-numeric memory and power fields on this GB10 path; doctor now degrades cleanly.
- Triton helper compilation requires repo-local Python headers because system headers are incomplete.
- `nvdisasm` cannot currently decode `SM121a`; Triton build capture degrades to PTX/no-SASS instead of failing.
- PyTorch `2.10.0+cu130` executes on this box but warns that official support caps at capability `12.0` while GB10 reports `12.1`.

## Frozen Dataset Baseline

These counts came from `artifacts/training/first_target_data_report.json` generated on March 14, 2026:

- train trajectory episodes: `4`
- dev trajectory episodes: `2`
- train SFT examples: `4`
- dev SFT examples: `1`
- held-out scripted baseline task count: `3`
- held-out scripted baseline success count: `3`

## Sufficiency Decision

The current slice is sufficient for:

- Spark smoke SFT
- Spark training-stack validation
- post-smoke held-out proxy evaluation

The current slice is not yet sufficient for a meaningful pilot-quality learning claim without one of:

- additional local collection
- expanded usable-negative coverage
- GPT-5.4 trace collection

## Smoke Outcome

The Spark smoke path has now been exercised against the frozen 32B target under real GB10 conditions.

- attempt 1:
  - artifact: `artifacts/training/qwen32b_debug_repair_qlora_spark_smoke_v1/systemd_smoke_attempt_01_report.json`
  - result: `oom-kill`
  - classification: model-load attempt failed before any fresh training report was written
- attempt 2:
  - artifact: `artifacts/training/qwen32b_debug_repair_qlora_spark_smoke_v1/systemd_smoke_attempt_02_report.json`
  - result: `oom-kill`
  - memory peak: `53.8G`
  - classification: model-load OOM before first bounded train step
- attempt 3:
  - artifact: `artifacts/training/qwen32b_debug_repair_qlora_spark_smoke_v1/systemd_smoke_attempt_03_report.json`
  - result: `exit-code`
  - classification: local instrumentation bug while emitting model-load stage metadata
- attempt 4:
  - live snapshot: `artifacts/training/qwen32b_debug_repair_qlora_spark_smoke_v1/systemd_smoke_attempt_04_live_snapshot.json`
  - final note: `artifacts/training/qwen32b_debug_repair_qlora_spark_smoke_v1/systemd_smoke_attempt_04_report.json`
  - classification: stalled after `model_load_begin` under the offload retry, with threads parked on `futex_wait_queue`
  - manual action: stopped intentionally after repeated no-progress observation to avoid wasting the box

Observed threshold status:

- tokenizer load: reached
- model load begin: reached
- model load done: not reached
- trainer init: not reached
- bounded train step: not reached

## Post-Smoke Proxy Evaluation

The exact proxy-eval command is:

```bash
bash scripts/run_with_spark_env.sh ./.venv/bin/python scripts/run_post_smoke_proxy_eval.py
```

The output directory is:

- `artifacts/training/qwen32b_debug_repair_qlora_spark_smoke_v1/post_smoke_proxy_eval`

Current proxy-eval result:

- report: `artifacts/training/qwen32b_debug_repair_qlora_spark_smoke_v1/post_smoke_proxy_eval/rollout_report.json`
- task count: `3`
- success count: `3`
- patch-bearing count: `2`
- average final reward: `0.8`

This is still a proxy path. It does not claim that the blocked 32B smoke adapter learned or executed successfully.

## Pilot Decision

- PTX-only build evidence is acceptable for pilot gating on this box.
- `nvdisasm`/`SM121a` degradation affects build evidence capture, not the observed training failure mode.
- the immediate pilot decision is still `blocked` for the frozen 32B Spark target because the smoke run never reached trainer initialization on GB10.

## Systems Interpretation

- the PyTorch GB10 capability warning remained advisory in the narrow sense that execution began and reached tokenizer/model-load setup; from direct evidence, the hard failures were model-load OOM and later a no-progress stall, not an immediate capability rejection
- the repeated stall after `model_load_begin` means a deeper runtime change is required before another 32B Spark smoke relaunch is justified

## Fallback Smoke Outcome

The same-family 7B fallback path has now been exercised on the same GB10 box using the same first-target dataset slice.

- config: `configs/training/sft_qwen7b_debug_repair_qlora_spark_smoke.json`
- smoke archive: `artifacts/training/qwen7b_debug_repair_qlora_spark_smoke_v1/systemd_smoke_attempt_01_report.json`
- smoke training report: `artifacts/training/qwen7b_debug_repair_qlora_spark_smoke_v1/training_run_report.json`
- comparison note: `artifacts/training/qwen7b_debug_repair_qlora_spark_smoke_v1/spark_fallback_delta_report.md`

Observed threshold status:

- tokenizer load: reached
- model load begin: reached
- model load done: reached
- prepare k-bit: reached
- trainer init: reached
- bounded train step: reached
- train done: reached

Observed fallback metrics:

- target: `Qwen/Qwen2.5-Coder-7B-Instruct`
- systemd result: `success`
- host memory peak: `28.6G`
- GPU peak allocated: `10022.4 MB`
- GPU peak reserved: `12334.0 MB`
- smoke wall time: about `208.1s`
- adapter output: present
- tokenizer output: present

## Fallback Post-Smoke Proxy Evaluation

The exact fallback proxy-eval command is:

```bash
bash scripts/run_with_spark_env.sh ./.venv/bin/python scripts/run_post_smoke_proxy_eval.py --training-config configs/training/sft_qwen7b_debug_repair_qlora_spark_smoke.json --out-dir artifacts/training/qwen7b_debug_repair_qlora_spark_smoke_v1/post_smoke_proxy_eval
```

The output directory is:

- `artifacts/training/qwen7b_debug_repair_qlora_spark_smoke_v1/post_smoke_proxy_eval`

Current fallback proxy-eval result:

- report: `artifacts/training/qwen7b_debug_repair_qlora_spark_smoke_v1/post_smoke_proxy_eval/rollout_report.json`
- task count: `3`
- success count: `3`
- patch-bearing count: `2`
- average final reward: `1.0667`

## Fallback Pilot Status

- pilot config: `configs/training/sft_qwen7b_debug_repair_qlora_spark_pilot.json`
- archive: `artifacts/training/qwen7b_debug_repair_qlora_spark_pilot_v1/systemd_pilot_attempt_01_report.json`
- training report: `artifacts/training/qwen7b_debug_repair_qlora_spark_pilot_v1/training_run_report.json`
- launch mode: `systemd --user`
- result: `success`
- host memory peak: `14.9G`
- GPU peak allocated: `10177.42 MB`
- GPU peak reserved: `12708.0 MB`
- effective max steps: `5`
- train runtime: `40.7378s`

## Current Spark Decision

- frozen 32B target: blocked and deferred on the current GB10 stack
- same-family 7B fallback: smoke validated and short pilot completed successfully
- final memo: `artifacts/training/qwen7b_debug_repair_qlora_spark_smoke_v1/final_spark_decision_memo.md`
