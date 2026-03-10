# Remote Bootstrap

## Purpose

Bring the `2xH200` node into a known-good state before any real SFT or RL run starts.

## Checklist

1. Clone the repo and install the pinned Python environment.
2. Run `python3 scripts/build_pre_h200_training_assets.py`.
3. Run `python3 scripts/build_h200_handoff.py`.
4. Run `python3 scripts/smoke_sft_train.py`.
5. Run `python3 scripts/build_heldout_baseline_report.py`.
6. Run `python3 scripts/smoke_rollout_eval.py`.

## Expected Outputs

- `artifacts/training/pre_h200_data_report.json`
- `docs/h200_handoff/handoff_manifest.json`
- `artifacts/training/smoke_sft_report.json`
- `artifacts/training/heldout_scripted_baseline_v1/rollout_report.json`
- `artifacts/training/rollout_smoke/rollout_report.json`

## Failure Policy

- do not begin real SFT if any required dataset is missing
- do not begin real SFT if the held-out scripted baseline report cannot be produced
- do not begin RL if the SFT smoke validation or rollout smoke validation fails
