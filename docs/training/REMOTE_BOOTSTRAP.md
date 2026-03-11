# Remote Bootstrap

## Purpose

Bring the target training environment into a known-good state before any real SFT or RL run starts.

## Checklist

1. Clone the repo and install the pinned Python environment.
2. Prefer `python3 scripts/run_training_preparation_verification.py` for a full verification sweep.
3. If you need the individual steps instead, run:
   - `python3 scripts/build_first_target_training_assets.py`
   - `python3 scripts/build_dataset_governance_report.py`
   - `python3 scripts/build_training_manifest.py`
   - `python3 scripts/smoke_sft_train.py`
   - `python3 scripts/build_heldout_baseline_report.py`
   - `python3 scripts/smoke_rollout_eval.py`

## Expected Outputs

- `artifacts/training/first_target_data_report.json`
- `artifacts/training/dataset_governance_report.json`
- `docs/training/training_manifest.json`
- `artifacts/training/smoke_sft_report.json`
- `artifacts/training/heldout_scripted_baseline_v1/rollout_report.json`
- `artifacts/training/rollout_smoke/rollout_report.json`

## Failure Policy

- do not begin real SFT if any required dataset is missing
- do not begin real SFT if the held-out scripted baseline report cannot be produced
- do not begin RL if the SFT smoke validation or rollout smoke validation fails
