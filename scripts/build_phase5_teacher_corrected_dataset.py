from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_gpt54_first_wave_baseline as harness

from gpu_cockpit.contracts import SFTDatasetManifest, SFTExample


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _response_json(turn: dict[str, Any]) -> str:
    selected_action = str(turn.get("selected_action", ""))
    parsed = turn.get("parsed_response", {})
    if not isinstance(parsed, dict):
        parsed = {}
    response: dict[str, Any] = {"action_name": selected_action}
    query = parsed.get("query")
    if selected_action == "knowledge_query" and isinstance(query, str) and query.strip():
        response["query"] = query.strip()
    return json.dumps(response, indent=2, sort_keys=True)


def _example_slug(source_episode_ref: str) -> str:
    ref_path = Path(source_episode_ref)
    parts = list(ref_path.parts)
    if parts and parts[-1].endswith(".json"):
        parts[-1] = parts[-1][:-5]
    return "__".join(parts[-4:]).replace("-", "_")


def _teacher_prompt(observation_packet: dict[str, Any], row: dict[str, Any], turn: dict[str, Any]) -> str:
    correction_metadata = {
        "quality_bucket": row["quality_bucket"],
        "slice_role": row["slice_role"],
        "source_block": row["source_block"],
        "teacher_model": turn.get("model"),
        "teacher_provider": turn.get("provider"),
        "correction_mode": "teacher_correction_v1",
    }
    return (
        f"System prompt:\n{harness.SYSTEM_PROMPT}\n\n"
        f"Teacher-correction metadata:\n{json.dumps(correction_metadata, indent=2, sort_keys=True)}\n\n"
        f"Observation packet:\n{json.dumps(observation_packet, indent=2, sort_keys=True)}"
    )


def _examples_for_row(*, row: dict[str, Any], split: str) -> list[SFTExample]:
    payload = _read_json(ROOT / str(row["episode_ref"]))
    turns = payload.get("model_turns", [])
    if not isinstance(turns, list):
        return []
    examples: list[SFTExample] = []
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        observation_packet = turn.get("observation_packet", {})
        if not isinstance(observation_packet, dict):
            observation_packet = {}
        examples.append(
            SFTExample(
                example_id=f"phase5_teacher_corrected__{_example_slug(str(row['episode_ref']))}__{turn_index}",
                created_at=datetime.now(tz=UTC),
                split=split,
                task_id=str(row["task_ref"]),
                prompt_family="optimize_action_json",
                prompt=_teacher_prompt(observation_packet, row, turn),
                response=_response_json(turn),
                source_episode_ref=str(row["episode_ref"]),
                metadata={
                    "training_example_kind": "teacher_corrected_trace",
                    "quality_bucket": row["quality_bucket"],
                    "slice_role": row["slice_role"],
                    "source_block": row["source_block"],
                    "selected_action": turn.get("selected_action"),
                    "teacher_model": turn.get("model"),
                    "teacher_provider": turn.get("provider"),
                    "turn_index": turn_index,
                },
            )
        )
    return examples


def _count_examples_by(examples: list[SFTExample], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        value = str(example.metadata.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _write_dataset(
    *,
    rows: list[dict[str, Any]],
    split: str,
    out_dir: Path,
    source_manifest_path: Path,
) -> SFTDatasetManifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    examples: list[SFTExample] = []
    for row in rows:
        examples.extend(_examples_for_row(row=row, split=split))
    example_refs: list[str] = []
    for example in examples:
        example_path = examples_dir / f"{example.example_id}.json"
        example_path.write_text(json.dumps(example.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
        example_refs.append(str(example_path.relative_to(out_dir)))
    manifest = SFTDatasetManifest(
        dataset_id=f"phase5_teacher_corrected_{split}_v1",
        created_at=datetime.now(tz=UTC),
        split=split,
        example_count=len(example_refs),
        example_refs=example_refs,
        task_ids=sorted({example.task_id for example in examples}),
        metadata={
            "source_freeze_manifest": str(source_manifest_path.relative_to(ROOT)),
            "quality_counts": _count_examples_by(examples, "quality_bucket"),
            "slice_role_counts": _count_examples_by(examples, "slice_role"),
            "source_block_counts": _count_examples_by(examples, "source_block"),
            "action_counts": _count_examples_by(examples, "selected_action"),
            "teacher_model_counts": _count_examples_by(examples, "teacher_model"),
        },
    )
    (out_dir / "sft_dataset_manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=ROOT / "artifacts" / "training" / "phase5_hard_trace_freeze_v2",
    )
    parser.add_argument(
        "--train-out-dir",
        type=Path,
        default=ROOT / "datasets" / "phase5_teacher_corrected_train_v2",
    )
    parser.add_argument(
        "--dev-out-dir",
        type=Path,
        default=ROOT / "datasets" / "phase5_teacher_corrected_dev_v2",
    )
    args = parser.parse_args()

    args.freeze_dir = args.freeze_dir.resolve()
    args.train_out_dir = args.train_out_dir.resolve()
    args.dev_out_dir = args.dev_out_dir.resolve()

    manifest_path = args.freeze_dir / "optimize_trace_manifest.json"
    report_path = args.freeze_dir / "teacher_corrected_dataset_report.json"
    manifest = _read_json(manifest_path)
    rows = manifest.get("episodes", [])
    if not isinstance(rows, list):
        raise SystemExit(f"Invalid manifest rows in {manifest_path}")
    train_rows = [row for row in rows if isinstance(row, dict) and row.get("split") == "train"]
    dev_rows = [row for row in rows if isinstance(row, dict) and row.get("split") == "dev"]
    train_manifest = _write_dataset(
        rows=train_rows,
        split="train",
        out_dir=args.train_out_dir,
        source_manifest_path=manifest_path,
    )
    dev_manifest = _write_dataset(
        rows=dev_rows,
        split="dev",
        out_dir=args.dev_out_dir,
        source_manifest_path=manifest_path,
    )
    report = {
        "report_id": f"{args.freeze_dir.name}_teacher_corrected_dataset",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "train_manifest": str((args.train_out_dir / "sft_dataset_manifest.json").relative_to(ROOT)),
        "dev_manifest": str((args.dev_out_dir / "sft_dataset_manifest.json").relative_to(ROOT)),
        "train_example_count": train_manifest.example_count,
        "dev_example_count": dev_manifest.example_count,
        "train_task_ids": train_manifest.task_ids,
        "dev_task_ids": dev_manifest.task_ids,
        "notes": [
            "Teacher-corrected dataset view built from the same frozen Phase 5 hard slice as distill and narrow RL.",
            "Examples preserve observation packets and teacher-selected actions but carry explicit correction metadata.",
            "This view is intended for on-policy teacher-correction or iterative self-distillation experiments.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
