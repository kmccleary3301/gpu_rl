from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.environment import run_scripted_reference_episode
from gpu_cockpit.engine.knowledge import build_knowledge_index
from gpu_cockpit.engine.sft import package_trajectory_dataset_as_sft, validate_sft_dataset
from gpu_cockpit.engine.trajectory import export_episode_dataset, validate_trajectory_dataset


def _load_config() -> dict[str, object]:
    path = ROOT / "configs" / "training" / "first_target_splits_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _build_episodes(specs: list[dict[str, object]]) -> list[object]:
    episodes = []
    for spec in specs:
        episodes.append(
            run_scripted_reference_episode(
                ROOT,
                str(spec["task"]),
                [str(item) for item in spec["command"]],
                policy_id="scripted_reference_first_target_v1",
                workflow=str(spec.get("workflow", "auto")),
                step_budget=int(spec.get("step_budget", 12)),
                section=str(spec.get("section", "quality")),
                include_build=bool(spec.get("include_build", False)),
                triton_build_spec=str(spec["triton_build_spec"]) if spec.get("triton_build_spec") else None,
            )
        )
    return episodes


def main() -> int:
    config = _load_config()
    outputs = dict(config["generated_outputs"])
    build_knowledge_index(ROOT)

    train_episodes = _build_episodes(list(config["train_episodes"]))
    dev_episodes = _build_episodes(list(config["dev_episodes"]))

    train_trajectory_dir = ROOT / str(outputs["train_trajectory"])
    dev_trajectory_dir = ROOT / str(outputs["dev_trajectory"])
    train_sft_dir = ROOT / str(outputs["train_sft"])
    dev_sft_dir = ROOT / str(outputs["dev_sft"])

    train_manifest = export_episode_dataset(
        train_episodes,
        train_trajectory_dir,
        policy_id="scripted_reference_first_target_v1",
        split="train",
        metadata={"collection_kind": "first_target_train"},
    )
    dev_manifest = export_episode_dataset(
        dev_episodes,
        dev_trajectory_dir,
        policy_id="scripted_reference_first_target_v1",
        split="dev",
        metadata={"collection_kind": "first_target_dev"},
    )

    train_sft_manifest = package_trajectory_dataset_as_sft(
        ROOT,
        train_trajectory_dir,
        train_sft_dir,
        split="train",
        include_failures=True,
        patch_bearing_only=False,
        governance_allowlist=["usable_positive_sft", "usable_negative_debug", "usable_negative_transition"],
        transition_kind_allowlist=["repaired", "reformulated", "patch_applied"],
        verb_allowlist=["debug", "reformulate"],
    )
    dev_sft_manifest = package_trajectory_dataset_as_sft(
        ROOT,
        dev_trajectory_dir,
        dev_sft_dir,
        split="dev",
        include_failures=True,
        governance_allowlist=["usable_positive_sft", "usable_negative_debug", "usable_negative_transition"],
        verb_allowlist=["diagnose", "optimize", "debug", "reformulate"],
    )

    report = {
        "config_id": config["config_id"],
        "train_trajectory_manifest": str(train_manifest.relative_to(ROOT)),
        "dev_trajectory_manifest": str(dev_manifest.relative_to(ROOT)),
        "train_sft_manifest": str(train_sft_manifest.relative_to(ROOT)),
        "dev_sft_manifest": str(dev_sft_manifest.relative_to(ROOT)),
        "train_trajectory_validation": validate_trajectory_dataset(train_trajectory_dir),
        "dev_trajectory_validation": validate_trajectory_dataset(dev_trajectory_dir),
        "train_sft_validation": validate_sft_dataset(train_sft_dir),
        "dev_sft_validation": validate_sft_dataset(dev_sft_dir),
        "train_tasks": sorted({episode.task_id for episode in train_episodes}),
        "dev_tasks": sorted({episode.task_id for episode in dev_episodes}),
        "train_episode_governance": train_manifest and json.loads(train_manifest.read_text(encoding="utf-8")).get("episode_governance_counts", {}),
        "dev_episode_governance": dev_manifest and json.loads(dev_manifest.read_text(encoding="utf-8")).get("episode_governance_counts", {}),
    }
    report_path = ROOT / str(outputs["data_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
