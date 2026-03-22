# Local Optimize Training Operator Note

This note freezes the repeatable local Spark path for optimize-trace SFT and checkpoint eval.

## Current Stable Training Path

Primary pilot config:

- [sft_qwen7b_optimize_trace_qlora_pilot_v2.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/configs/training/sft_qwen7b_optimize_trace_qlora_pilot_v2.json)

Repeatability config:

- [sft_qwen7b_optimize_trace_qlora_pilot_v2_repeat.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/configs/training/sft_qwen7b_optimize_trace_qlora_pilot_v2_repeat.json)

Run pattern:

```bash
gpu_rl/scripts/run_with_spark_env.sh gpu_rl/.venv/bin/python - <<'PY'
from pathlib import Path
import sys
root = Path("gpu_rl").resolve()
sys.path.insert(0, str(root))
from gpu_cockpit.engine.training import run_sft_training_job
report = run_sft_training_job(
    root,
    root / "configs" / "training" / "sft_qwen7b_optimize_trace_qlora_pilot_v2.json",
)
print(report)
PY
```

Checkpoint eval pattern:

```bash
gpu_rl/scripts/run_with_spark_env.sh gpu_rl/.venv/bin/python \
  gpu_rl/scripts/run_local_optimize_checkpoint_eval.py \
  gpu_rl/configs/training/qwen7b_optimize_checkpoint_eval_v2.json \
  --adapter-dir gpu_rl/artifacts/training/qwen7b_optimize_trace_qlora_pilot_v2 \
  --out-dir gpu_rl/artifacts/baselines/qwen7b_optimize_eval_pilot2_v2 \
  --label pilot_v2
```

## Repeatability Check

Observed training reports:

- [training_run_report.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/training/qwen7b_optimize_trace_qlora_pilot_v2/training_run_report.json)
- [training_run_report.json](/home/kmccleary/projects/gpu_code_agents/gpu_rl/artifacts/training/qwen7b_optimize_trace_qlora_pilot_v2_repeat/training_run_report.json)

Observed stability:

- both runs finished `ok`
- both runs used `16` effective steps
- train loss was effectively identical: `2.2758` vs `2.2751`
- peak allocated memory was effectively identical: `10151.28 MB` vs `10154.98 MB`
- peak reserved memory stayed close: `12456.0 MB` vs `12186.0 MB`
- artifact directory size stayed effectively identical: `357741547` bytes vs `357730740` bytes

Interpretation:

- the local optimize-trace pilot lane is repeatable enough to use for further checkpoint iteration on this Spark
- current variance is small relative to the size of the training/eval gap we still need to close
- future iterations should focus on better positive trace coverage before increasing step budgets further
