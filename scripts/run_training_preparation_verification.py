from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


COMMANDS = [
    ["python3", "scripts/export_schemas.py"],
    ["python3", "scripts/generate_transition_goldens.py"],
    ["python3", "scripts/build_first_target_training_assets.py"],
    ["python3", "scripts/build_dataset_governance_report.py"],
    ["python3", "scripts/build_heldout_baseline_report.py"],
    ["python3", "scripts/smoke_sft_train.py"],
    ["python3", "scripts/smoke_rollout_eval.py"],
    ["python3", "-m", "unittest", "discover", "-s", "tests", "-v"],
]


def main() -> int:
    for command in COMMANDS:
        print(f"$ {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
