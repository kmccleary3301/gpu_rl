# Training Preparation Checklist

Use this checklist before starting any nontrivial training run in a dedicated training environment.

## 1. Clone and install

- [ ] clone the repository
- [ ] install the pinned Python environment
- [ ] confirm `gpc --help` works

## 2. Build datasets and reports

- [ ] optionally run `python3 scripts/run_training_preparation_verification.py` for the full verification sweep
- [ ] run `python3 scripts/build_first_target_training_assets.py`
- [ ] run `python3 scripts/build_dataset_governance_report.py`
- [ ] run `python3 scripts/build_training_manifest.py`
- [ ] run `python3 scripts/build_heldout_baseline_report.py`

## 3. Validate smoke paths

- [ ] run `python3 scripts/smoke_sft_train.py`
- [ ] run `python3 scripts/smoke_rollout_eval.py`

## 4. Validate configs

- [ ] validate `configs/training/sft_qwen32b_debug_repair_lora.json`
- [ ] validate `configs/training/rollout_debug_repair_heldout_v1.json`

## 5. Confirm outputs

- [ ] training asset report exists
- [ ] dataset governance report exists
- [ ] smoke SFT report exists
- [ ] held-out scripted baseline report exists
- [ ] rollout smoke report exists

## 6. Confirm readiness

- [ ] task inventories and governance semantics are understood
- [ ] benchmark-only traces are excluded unless explicitly allowed
- [ ] first-wave target and stop conditions are understood

## 7. Only then start real training

- [ ] warm-start SFT
- [ ] held-out evaluation
- [ ] narrow RL only after SFT validation passes
