# Smoke Sequence

Run these in order on the target training environment. The goal is to validate the training-preparation surface mechanically before any real training job starts.

```bash
python3 scripts/run_training_preparation_verification.py
```

If you prefer the individual steps instead of the one-shot sweep:

```bash
python3 scripts/build_first_target_training_assets.py
python3 scripts/build_dataset_governance_report.py
python3 scripts/build_training_manifest.py
python3 scripts/smoke_sft_train.py
python3 scripts/build_heldout_baseline_report.py
python3 scripts/smoke_rollout_eval.py
```

## Success Conditions

- the generated train/dev datasets validate cleanly
- the dataset governance report shows the expected mix of positive and usable-negative traces
- the training manifest reports `status: ok`
- the SFT smoke report validates all required dataset refs
- the held-out scripted baseline report completes with a non-zero success count
- the rollout smoke report completes without schema or lineage failures

## Non-Goals

- no real SFT optimization run
- no real RL rollout collection
- no distributed orchestration bring-up
