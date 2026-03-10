# Smoke Sequence

Run these in order on the future H200 node. The goal is to validate the handoff mechanically before any real training job starts.

```bash
python3 scripts/build_pre_h200_training_assets.py
python3 scripts/build_h200_handoff.py
python3 scripts/smoke_sft_train.py
python3 scripts/build_heldout_baseline_report.py
python3 scripts/smoke_rollout_eval.py
```

## Success Conditions

- the generated train/dev datasets validate cleanly
- the handoff manifest reports `status: ok`
- the SFT smoke report validates all required dataset refs
- the held-out scripted baseline report completes with a non-zero success count
- the rollout smoke report completes without schema or lineage failures

## Non-Goals

- no real SFT optimization run
- no real RL rollout collection
- no distributed orchestration bring-up
