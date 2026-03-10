from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpu_cockpit.contracts import DatasetRef, SFTDatasetManifest, SFTExample, SFTTrainingConfig, TrajectoryDatasetManifest, TrajectoryEpisode


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sft_dataset(dataset_dir: Path) -> tuple[SFTDatasetManifest, list[SFTExample]]:
    manifest_path = dataset_dir / "sft_dataset_manifest.json"
    manifest = SFTDatasetManifest.model_validate(_load_json(manifest_path))
    examples = [
        SFTExample.model_validate(_load_json(dataset_dir / str(relative_ref)))
        for relative_ref in manifest.example_refs
    ]
    return manifest, examples


def load_trajectory_dataset(dataset_dir: Path) -> tuple[TrajectoryDatasetManifest, list[TrajectoryEpisode]]:
    manifest_path = dataset_dir / "trajectory_dataset_manifest.json"
    manifest = TrajectoryDatasetManifest.model_validate(_load_json(manifest_path))
    episodes = [
        TrajectoryEpisode.model_validate(_load_json(dataset_dir / str(relative_ref)))
        for relative_ref in manifest.episode_refs
    ]
    return manifest, episodes


def load_training_config(path: Path) -> SFTTrainingConfig:
    return SFTTrainingConfig.model_validate(_load_json(path))


def _resolve_dataset_ref(root: Path, ref: DatasetRef) -> Path:
    path = Path(ref.path)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def validate_sft_training_config(root: Path, config: SFTTrainingConfig) -> dict[str, object]:
    missing_required_datasets: list[str] = []
    dataset_summaries: list[dict[str, object]] = []
    for ref in [*config.dataset_refs, *config.eval_dataset_refs]:
        resolved = _resolve_dataset_ref(root, ref)
        exists = resolved.exists()
        if ref.required and not exists:
            missing_required_datasets.append(ref.path)
            continue
        if not exists:
            dataset_summaries.append({"path": ref.path, "exists": False, "dataset_kind": ref.dataset_kind})
            continue
        if ref.dataset_kind == "sft":
            manifest, examples = load_sft_dataset(resolved)
            dataset_summaries.append(
                {
                    "path": ref.path,
                    "exists": True,
                    "dataset_kind": ref.dataset_kind,
                    "example_count": manifest.example_count,
                    "task_count": len(manifest.task_ids),
                    "split": manifest.split,
                }
            )
        elif ref.dataset_kind == "trajectory":
            manifest, episodes = load_trajectory_dataset(resolved)
            dataset_summaries.append(
                {
                    "path": ref.path,
                    "exists": True,
                    "dataset_kind": ref.dataset_kind,
                    "episode_count": manifest.episode_count,
                    "task_count": len(manifest.task_ids),
                    "split": manifest.split,
                }
            )
        else:
            dataset_summaries.append({"path": ref.path, "exists": True, "dataset_kind": ref.dataset_kind})
    return {
        "status": "ok" if not missing_required_datasets else "failed",
        "config_id": config.config_id,
        "model_id": config.model_id,
        "adapter_mode": config.adapter_mode,
        "missing_required_datasets": missing_required_datasets,
        "datasets": dataset_summaries,
    }


def write_sft_smoke_report(root: Path, config_path: Path, out_path: Path) -> Path:
    config = load_training_config(config_path)
    payload = validate_sft_training_config(root, config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path
