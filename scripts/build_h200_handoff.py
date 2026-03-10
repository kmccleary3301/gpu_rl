from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.training import load_training_config, validate_sft_training_config


def main() -> int:
    handoff_dir = ROOT / "docs" / "h200_handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    split_config_path = ROOT / "configs" / "training" / "first_target_splits_v1.json"
    sft_config_path = ROOT / "configs" / "training" / "sft_qwen32b_debug_repair_lora.json"
    rollout_config_path = ROOT / "configs" / "training" / "rollout_debug_repair_v1.json"
    heldout_rollout_config_path = ROOT / "configs" / "training" / "rollout_debug_repair_heldout_v1.json"
    sft_config = load_training_config(sft_config_path)
    validation = validate_sft_training_config(ROOT, sft_config)
    payload = {
        "handoff_id": "handoff/h200/debug_repair/v1",
        "node_profile": "2xH200",
        "model_target": sft_config.model_id,
        "training_strategy": "sft_then_narrow_rl",
        "config_refs": {
            "splits": str(split_config_path.relative_to(ROOT)),
            "sft": str(sft_config_path.relative_to(ROOT)),
            "rollout": str(rollout_config_path.relative_to(ROOT)),
            "heldout_rollout": str(heldout_rollout_config_path.relative_to(ROOT)),
        },
        "script_refs": {
            "build_pre_h200_assets": "scripts/build_pre_h200_training_assets.py",
            "smoke_sft": "scripts/smoke_sft_train.py",
            "smoke_rollout": "scripts/smoke_rollout_eval.py",
            "heldout_baseline": "scripts/build_heldout_baseline_report.py",
            "build_handoff": "scripts/build_h200_handoff.py",
        },
        "generated_artifact_refs": {
            "pre_h200_data_report": "artifacts/training/pre_h200_data_report.json",
            "smoke_sft_report": "artifacts/training/smoke_sft_report.json",
            "heldout_baseline_report": "artifacts/training/heldout_scripted_baseline_v1/rollout_report.json",
            "rollout_smoke_report": "artifacts/training/rollout_smoke/rollout_report.json",
        },
        "doc_refs": {
            "training_target": "docs/h200_handoff/TRAINING_TARGET.md",
            "remote_bootstrap": "docs/h200_handoff/REMOTE_BOOTSTRAP.md",
            "smoke_sequence": "docs/h200_handoff/SMOKE_SEQUENCE.md",
        },
        "validation": validation,
        "required_checked_in_fixtures": [
            "tests/golden_datasets/transition_collection_v1",
            "tests/golden_datasets/transition_negative_collection_v1",
            "tests/golden_datasets/transition_sft_v1",
            "tests/golden_datasets/transition_negative_sft_v1",
            "tests/golden_runs/reduction_debug_patch_transition_v1",
            "tests/golden_episodes/reduction_debug_patch_episode_v1.json",
            "tests/golden_episodes/attention_reformulate_patch_episode_v1.json",
            "tests/golden_episodes/reduction_debug_negative_episode_v1.json",
            "tests/golden_episodes/attention_reformulate_negative_episode_v1.json",
        ],
    }
    out_path = handoff_dir / "handoff_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
