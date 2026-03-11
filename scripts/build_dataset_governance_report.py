from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_manifest(path_str: str) -> dict[str, object]:
    path = ROOT / path_str
    return _load_json(path)


def _manifest_metadata(manifest: dict[str, object]) -> dict[str, object]:
    raw = manifest.get("metadata", {})
    return raw if isinstance(raw, dict) else {}


def main() -> int:
    split_config = _load_json(ROOT / "configs" / "training" / "first_target_splits_v1.json")
    outputs = dict(split_config["generated_outputs"])

    train_trajectory = _read_manifest(f"{outputs['train_trajectory']}/trajectory_dataset_manifest.json")
    dev_trajectory = _read_manifest(f"{outputs['dev_trajectory']}/trajectory_dataset_manifest.json")
    train_sft = _read_manifest(f"{outputs['train_sft']}/sft_dataset_manifest.json")
    dev_sft = _read_manifest(f"{outputs['dev_sft']}/sft_dataset_manifest.json")
    train_sft_meta = _manifest_metadata(train_sft)
    dev_sft_meta = _manifest_metadata(dev_sft)

    report = {
        "report_id": "report/dataset_governance/v1",
        "source_config": "configs/training/first_target_splits_v1.json",
        "trajectory_datasets": {
            "train": {
                "path": f"{outputs['train_trajectory']}/trajectory_dataset_manifest.json",
                "episode_count": train_trajectory.get("episode_count", 0),
                "governance_counts": train_trajectory.get("episode_governance_counts", {}),
                "readiness_counts": train_trajectory.get("readiness_counts", {}),
                "transition_kind_counts": train_trajectory.get("transition_kind_counts", {}),
                "patch_kind_counts": train_trajectory.get("patch_kind_counts", {}),
                "patch_bearing_episode_count": train_trajectory.get("patch_bearing_episode_count", 0),
                "patch_bearing_negative_episode_count": train_trajectory.get("patch_bearing_negative_episode_count", 0),
                "usable_positive_episode_count": train_trajectory.get("usable_positive_episode_count", 0),
                "usable_negative_episode_count": train_trajectory.get("usable_negative_episode_count", 0),
            },
            "dev": {
                "path": f"{outputs['dev_trajectory']}/trajectory_dataset_manifest.json",
                "episode_count": dev_trajectory.get("episode_count", 0),
                "governance_counts": dev_trajectory.get("episode_governance_counts", {}),
                "readiness_counts": dev_trajectory.get("readiness_counts", {}),
                "transition_kind_counts": dev_trajectory.get("transition_kind_counts", {}),
                "patch_kind_counts": dev_trajectory.get("patch_kind_counts", {}),
                "patch_bearing_episode_count": dev_trajectory.get("patch_bearing_episode_count", 0),
                "patch_bearing_negative_episode_count": dev_trajectory.get("patch_bearing_negative_episode_count", 0),
                "usable_positive_episode_count": dev_trajectory.get("usable_positive_episode_count", 0),
                "usable_negative_episode_count": dev_trajectory.get("usable_negative_episode_count", 0),
            },
        },
        "sft_datasets": {
            "train": {
                "path": f"{outputs['train_sft']}/sft_dataset_manifest.json",
                "example_count": train_sft.get("example_count", 0),
                "task_verb_counts": train_sft_meta.get("verb_counts", {}),
                "operator_family_counts": train_sft_meta.get("operator_family_counts", {}),
                "training_example_kind_counts": train_sft_meta.get("training_example_kind_counts", {}),
                "episode_governance_counts": train_sft_meta.get("episode_governance_counts", {}),
                "patch_kind_counts": train_sft_meta.get("patch_kind_counts", {}),
                "transition_kind_counts": train_sft_meta.get("transition_kind_counts", {}),
                "patch_bearing_example_count": train_sft_meta.get("patch_bearing_example_count", 0),
            },
            "dev": {
                "path": f"{outputs['dev_sft']}/sft_dataset_manifest.json",
                "example_count": dev_sft.get("example_count", 0),
                "task_verb_counts": dev_sft_meta.get("verb_counts", {}),
                "operator_family_counts": dev_sft_meta.get("operator_family_counts", {}),
                "training_example_kind_counts": dev_sft_meta.get("training_example_kind_counts", {}),
                "episode_governance_counts": dev_sft_meta.get("episode_governance_counts", {}),
                "patch_kind_counts": dev_sft_meta.get("patch_kind_counts", {}),
                "transition_kind_counts": dev_sft_meta.get("transition_kind_counts", {}),
                "patch_bearing_example_count": dev_sft_meta.get("patch_bearing_example_count", 0),
            },
        },
        "notes": [
            "Generated from the checked-in first training split config.",
            "Intended as a final audit of trajectory governance and SFT packaging composition before transfer.",
        ],
    }

    out_path = ROOT / "artifacts" / "training" / "dataset_governance_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(out_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
