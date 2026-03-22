from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> int:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded Modal MVP command path.")
    parser.add_argument(
        "--mode",
        choices=("smoke", "eval", "patch"),
        required=True,
        help="Which bounded Modal MVP path to execute.",
    )
    args = parser.parse_args()

    base = [sys.executable, "-m", "gpu_cockpit.cli.main"]
    if args.mode == "smoke":
        command = base + [
            "eval",
            "--task",
            "task/profile_diagnose/eval/v1",
            "--executor",
            "modal",
            "--",
            "python3",
            "workloads/reference/profile_diagnose_candidate.py",
            "--benchmark-repeats",
            "50",
        ]
    elif args.mode == "eval":
        command = base + [
            "rollout",
            "scripted",
            "configs/training/rollout_profile_diagnose_modal_smoke_v1.json",
            "--out-dir",
            "artifacts/modal_mvp/profile_diagnose_rollout_v1",
        ]
    else:
        command = base + [
            "env",
            "scripted",
            "--task",
            "task/reduction_debug/eval/v1",
            "--out",
            "artifacts/modal_mvp/reduction_debug_patch_episode.json",
            "--policy-id",
            "modal_mvp_patch_v1",
            "--step-budget",
            "6",
            "--workflow",
            "auto",
            "--executor",
            "modal",
            "--",
            "python3",
            "workloads/reference/triton_row_sum_debug_candidate.py",
        ]
    return _run(command)


if __name__ == "__main__":
    raise SystemExit(main())
