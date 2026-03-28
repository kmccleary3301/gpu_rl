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

from gpu_cockpit.contracts.sft import SFTDatasetManifest, SFTExample


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_score(payload: dict[str, Any]) -> tuple[float, list[str]]:
    success = bool(payload.get("success"))
    terminal_reason = str(payload.get("terminal_reason", ""))
    counters = payload.get("counters", {})
    if not isinstance(counters, dict):
        counters = {}
    score = 0.0
    reasons: list[str] = []
    if success:
        score += 3.5
        reasons.append("success")
    elif terminal_reason in {"negative_trace_complete", "multi_candidate_negative_complete"}:
        score += 1.75
        reasons.append("usable_negative")
    elif terminal_reason in {"two_attempt_positive_complete", "three_attempt_positive_complete", "post_patch_eval_failed"}:
        score += 1.25
        reasons.append("near_miss")
    elif terminal_reason == "budget_exhausted" and int(counters.get("patches", 0)) > 0 and int(counters.get("compares", 0)) > 0:
        score += 1.0
        reasons.append("branch_budget_near_miss")
    else:
        score += 0.15
        reasons.append("weak_trace")
    if int(counters.get("compares", 0)) > 0:
        score += 0.25
        reasons.append("compare_used")
    if int(counters.get("branches", 0)) > 0:
        score += 0.35
        reasons.append("branch_used")
    if int(counters.get("reverts", 0)) > 0:
        score += 0.2
        reasons.append("revert_used")
    if int(counters.get("promotes", 0)) > 0 and success:
        score += 0.35
        reasons.append("promote_closeout")
    if int(counters.get("branches", 0)) > 0 and int(counters.get("compares", 0)) >= 2:
        score += 0.2
        reasons.append("branch_compare_depth")
    if terminal_reason == "budget_exhausted" and int(counters.get("eval_actions", 0)) == 0:
        score -= 0.2
        reasons.append("no_eval_closeout_penalty")
    return round(score, 4), reasons


def _duplication_count(score: float) -> int:
    if score >= 3.8:
        return 4
    if score >= 2.0:
        return 3
    if score >= 1.0:
        return 2
    return 1


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


def _build_example(
    *,
    payload: dict[str, Any],
    row: dict[str, Any],
    turn: dict[str, Any],
    split: str,
    dup_index: int,
    turn_index: int,
    score: float,
    score_reasons: list[str],
) -> SFTExample:
    observation_packet = turn.get("observation_packet", {})
    if not isinstance(observation_packet, dict):
        observation_packet = {}
    prompt = (
        f"System prompt:\n{harness.SYSTEM_PROMPT}\n\n"
        f"Phase 5A reward-weighted metadata:\n"
        f"{json.dumps({'quality_bucket': row.get('quality_bucket'), 'slice_role': row.get('slice_role'), 'source_block': row.get('source_block'), 'episode_reward_score': score, 'score_reasons': score_reasons}, indent=2, sort_keys=True)}\n\n"
        f"Observation packet:\n{json.dumps(observation_packet, indent=2, sort_keys=True)}"
    )
    return SFTExample(
        example_id=f"phase5a_rwr_{Path(str(row['episode_ref'])).stem}_{dup_index}_{turn_index}",
        created_at=datetime.now(tz=UTC),
        split=split,
        task_id=str(row["task_ref"]),
        prompt_family="optimize_action_json",
        prompt=prompt,
        response=_response_json(turn),
        source_episode_ref=str(row["episode_ref"]),
        metadata={
            "training_example_kind": "positive_rl_trace" if bool(payload.get("success")) else "reward_weighted_rl_trace",
            "episode_reward_score": score,
            "score_reasons": score_reasons,
            "quality_bucket": row.get("quality_bucket"),
            "slice_role": row.get("slice_role"),
            "source_block": row.get("source_block"),
            "selected_action": turn.get("selected_action"),
            "model": turn.get("model"),
            "provider": turn.get("provider"),
        },
    )


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
    min_score: float,
) -> SFTDatasetManifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    examples: list[SFTExample] = []
    score_index: dict[str, dict[str, Any]] = {}

    for row in rows:
        episode_ref = str(row["episode_ref"])
        payload = _read_json(ROOT / episode_ref)
        score, reasons = _episode_score(payload)
        score_index[episode_ref] = {
            "score": score,
            "reasons": reasons,
            "success": bool(payload.get("success")),
            "terminal_reason": payload.get("terminal_reason"),
        }
        if score < min_score:
            continue
        turns = payload.get("model_turns", [])
        if not isinstance(turns, list):
            continue
        dup_count = 1 if split == "dev" else _duplication_count(score)
        for dup_index in range(dup_count):
            for turn_index, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    continue
                examples.append(
                    _build_example(
                        payload=payload,
                        row=row,
                        turn=turn,
                        split=split,
                        dup_index=dup_index,
                        turn_index=turn_index,
                        score=score,
                        score_reasons=reasons,
                    )
                )

    example_refs: list[str] = []
    for example in examples:
        example_path = examples_dir / f"{example.example_id}.json"
        example_path.write_text(json.dumps(example.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
        example_refs.append(str(example_path.relative_to(out_dir)))

    manifest = SFTDatasetManifest(
        dataset_id=f"phase5a_branchaware_rwr_{split}_v1",
        created_at=datetime.now(tz=UTC),
        split=split,
        example_count=len(example_refs),
        example_refs=example_refs,
        task_ids=sorted({example.task_id for example in examples}),
        metadata={
            "source_freeze_manifest": str(source_manifest_path.relative_to(ROOT)),
            "scoring_policy": "phase5a_branch_aware_rwr_v1",
            "min_score": min_score,
            "quality_counts": _count_examples_by(examples, "quality_bucket"),
            "slice_role_counts": _count_examples_by(examples, "slice_role"),
            "source_block_counts": _count_examples_by(examples, "source_block"),
            "action_counts": _count_examples_by(examples, "selected_action"),
            "episode_scores": score_index,
        },
    )
    (out_dir / "sft_dataset_manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=ROOT / "artifacts" / "training" / "phase5a_hard_trace_freeze_v1",
    )
    parser.add_argument(
        "--train-out-dir",
        type=Path,
        default=ROOT / "datasets" / "phase5a_optimize_rwr_branchaware_train_v1",
    )
    parser.add_argument(
        "--dev-out-dir",
        type=Path,
        default=ROOT / "datasets" / "phase5a_optimize_rwr_branchaware_dev_v1",
    )
    parser.add_argument("--min-score", type=float, default=0.75)
    args = parser.parse_args()

    manifest_path = args.freeze_dir / "optimize_trace_manifest.json"
    source_manifest = _read_json(manifest_path)
    rows = source_manifest.get("episodes", [])
    if not isinstance(rows, list):
        raise SystemExit(f"Invalid manifest rows in {manifest_path}")

    train_rows = [row for row in rows if isinstance(row, dict) and row.get("split") == "train"]
    dev_rows = [row for row in rows if isinstance(row, dict) and row.get("split") == "dev" and row.get("held_out") is not True]

    train_manifest = _write_dataset(
        rows=train_rows,
        split="train",
        out_dir=args.train_out_dir,
        source_manifest_path=manifest_path,
        min_score=args.min_score,
    )
    dev_manifest = _write_dataset(
        rows=dev_rows,
        split="dev",
        out_dir=args.dev_out_dir,
        source_manifest_path=manifest_path,
        min_score=args.min_score,
    )
    report = {
        "report_id": f"{args.freeze_dir.name}_branchaware_rwr_dataset",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "train_manifest": str((args.train_out_dir / 'sft_dataset_manifest.json').relative_to(ROOT)),
        "dev_manifest": str((args.dev_out_dir / 'sft_dataset_manifest.json').relative_to(ROOT)),
        "train_example_count": train_manifest.example_count,
        "dev_example_count": dev_manifest.example_count,
        "train_task_ids": train_manifest.task_ids,
        "dev_task_ids": dev_manifest.task_ids,
        "min_score": args.min_score,
        "notes": [
            "Phase 5A branch-aware narrow-RL view built directly from the aligned hard freeze.",
            "Uses the same episode universe as the distill and teacher-corrected views where allowed by split/provenance.",
            "Official held-out KB-v3 traces are excluded from the train and dev RL views.",
        ],
    }
    report_path = args.freeze_dir / "branchaware_rwr_dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
