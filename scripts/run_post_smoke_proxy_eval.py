from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts import RLRolloutConfig
from gpu_cockpit.engine.knowledge import build_knowledge_index
from gpu_cockpit.engine.rollout import run_scripted_rollout_suite
from gpu_cockpit.engine.training import load_training_config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the held-out post-smoke proxy evaluation path.")
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training/sft_qwen32b_debug_repair_qlora_spark_smoke.json"),
        help="Training config that owns the smoke output tree",
    )
    parser.add_argument(
        "--rollout-config",
        type=Path,
        default=Path("configs/training/rollout_debug_repair_heldout_v1.json"),
        help="Held-out rollout config to run",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional evaluation output directory override",
    )
    args = parser.parse_args()

    training_config_path = (ROOT / args.training_config).resolve() if not args.training_config.is_absolute() else args.training_config
    rollout_config_path = (ROOT / args.rollout_config).resolve() if not args.rollout_config.is_absolute() else args.rollout_config

    training_config = load_training_config(training_config_path)
    smoke_out_dir = (ROOT / training_config.output_dir).resolve()
    training_report_path = smoke_out_dir / "training_run_report.json"
    if not training_report_path.exists():
        raise FileNotFoundError(f"Missing smoke training report: {training_report_path}")
    training_report = _read_json(training_report_path)

    rollout_config = RLRolloutConfig.model_validate(_read_json(rollout_config_path))
    rollout_notes = list(rollout_config.notes)
    rollout_notes.extend(
        [
            f"proxy_eval_source_training_report:{_relative_to_root(training_report_path)}",
            f"proxy_eval_source_training_status:{training_report.get('status', 'unknown')}",
        ]
    )
    adapter_dir = training_report.get("adapter_dir")
    if adapter_dir:
        rollout_notes.append(f"proxy_eval_adapter_dir:{adapter_dir}")

    proxy_config = rollout_config.model_copy(
        update={
            "config_id": f"{rollout_config.config_id}/post_smoke_proxy",
            "policy_id": "post_smoke_proxy_v1",
            "notes": rollout_notes,
        }
    )
    out_dir = ((ROOT / args.out_dir).resolve() if args.out_dir is not None and not args.out_dir.is_absolute() else args.out_dir)
    if out_dir is None:
        out_dir = smoke_out_dir / "post_smoke_proxy_eval"

    build_knowledge_index(ROOT)
    report = run_scripted_rollout_suite(ROOT, proxy_config, out_dir)
    report_path = out_dir / "rollout_report.json"
    print(_relative_to_root(report_path))
    print(json.dumps(report.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
