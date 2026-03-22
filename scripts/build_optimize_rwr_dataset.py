from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_gpt54_first_wave_baseline as harness

from gpu_cockpit.contracts.sft import SFTDatasetManifest, SFTExample


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _root_relative(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT.resolve()))


def _episode_score(payload: dict[str, object]) -> tuple[float, list[str]]:
    success = bool(payload.get("success"))
    terminal_reason = str(payload.get("terminal_reason", ""))
    counters = payload.get("counters", {})
    if not isinstance(counters, dict):
        counters = {}
    score = 0.0
    reasons: list[str] = []
    if success:
        score += 3.0
        reasons.append("success")
    elif terminal_reason in {"negative_trace_complete", "multi_candidate_negative_complete"}:
        score += 1.5
        reasons.append("usable_negative")
    elif terminal_reason in {"two_attempt_positive_complete", "three_attempt_positive_complete", "post_patch_eval_failed"}:
        score += 1.0
        reasons.append("near_miss")
    else:
        score += 0.25
        reasons.append("weak_trace")
    if int(counters.get("compares", 0)) > 0:
        score += 0.2
        reasons.append("compare_used")
    if int(counters.get("branches", 0)) > 0:
        score += 0.2
        reasons.append("branch_used")
    if int(counters.get("reverts", 0)) > 0:
        score += 0.1
        reasons.append("revert_used")
    if int(counters.get("promotes", 0)) > 0 and success:
        score += 0.2
        reasons.append("promote_closeout")
    return round(score, 4), reasons


def _duplication_count(score: float) -> int:
    if score >= 3.3:
        return 4
    if score >= 2.0:
        return 3
    if score >= 1.0:
        return 2
    return 1


def _response_json(turn: dict[str, object]) -> str:
    selected_action = str(turn.get("selected_action", ""))
    parsed = turn.get("parsed_response", {})
    if not isinstance(parsed, dict):
        parsed = {}
    response: dict[str, object] = {"action_name": selected_action}
    query = parsed.get("query")
    if selected_action == "knowledge_query" and isinstance(query, str) and query.strip():
        response["query"] = query.strip()
    return json.dumps(response, indent=2, sort_keys=True)


def _build_example(
    *,
    payload: dict[str, object],
    turn: dict[str, object],
    split: str,
    source_episode_ref: str,
    dup_index: int,
    turn_index: int,
    score: float,
    score_reasons: list[str],
) -> SFTExample:
    task_id = str(payload.get("task_ref", ""))
    observation_packet = turn.get("observation_packet", {})
    prompt = (
        f"System prompt:\n{harness.SYSTEM_PROMPT}\n\n"
        f"Observation packet:\n{json.dumps(observation_packet, indent=2, sort_keys=True)}"
    )
    return SFTExample(
        example_id=f"rwr_{Path(source_episode_ref).stem}_{dup_index}_{turn_index}",
        created_at=datetime.now(tz=UTC),
        split=split,
        task_id=task_id,
        prompt_family="optimize_action_json",
        prompt=prompt,
        response=_response_json(turn),
        source_episode_ref=source_episode_ref,
        metadata={
            "training_example_kind": "positive_rl_trace" if bool(payload.get("success")) else "reward_weighted_rl_trace",
            "episode_reward_score": score,
            "score_reasons": score_reasons,
            "selected_action": turn.get("selected_action"),
            "model": turn.get("model"),
            "provider": turn.get("provider"),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--out-train-dir", type=Path, required=True)
    parser.add_argument("--out-dev-dir", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    episode_paths = sorted(args.batch_dir.glob("task__*.json"))
    if not episode_paths:
        raise SystemExit(f"No episode artifacts found in {args.batch_dir}")

    episode_payloads = [(path, _read_json(path)) for path in episode_paths]
    scored = []
    for path, payload in episode_payloads:
        score, reasons = _episode_score(payload)
        if score >= args.min_score:
            scored.append((path, payload, score, reasons))
    if not scored:
        raise SystemExit(f"No episodes met min-score >= {args.min_score}")

    # Hold out the strongest successful episode as dev when possible.
    ranked = sorted(scored, key=lambda item: (item[2], bool(item[1].get("success"))), reverse=True)
    dev_path = ranked[0][0]

    for out_dir, split in ((args.out_train_dir, "train"), (args.out_dev_dir, "dev")):
        examples_dir = out_dir / "examples"
        examples_dir.mkdir(parents=True, exist_ok=True)
        examples: list[SFTExample] = []
        for path, payload, score, reasons in scored:
            use_in_split = (path == dev_path) if split == "dev" else (path != dev_path)
            if not use_in_split:
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
                            turn=turn,
                            split=split,
                            source_episode_ref=_root_relative(path),
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
            dataset_id=f"optimize_rwr_dataset_{split}_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now(tz=UTC),
            split=split,
            example_count=len(example_refs),
            example_refs=example_refs,
            task_ids=sorted({example.task_id for example in examples}),
            metadata={
                "source_batch_dir": str(args.batch_dir),
                "scoring_policy": "reward_weighted_regression_v1",
                "min_score": args.min_score,
                "episode_scores": {
                    str(path.name): {
                        "score": score,
                        "reasons": reasons,
                        "success": bool(payload.get("success")),
                        "terminal_reason": payload.get("terminal_reason"),
                    }
                    for path, payload, score, reasons in scored
                },
            },
        )
        (out_dir / "sft_dataset_manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")

    dataset_suffix = args.out_train_dir.name.removeprefix("optimize_rwr_train_") or "custom"
    reward_report = {
        "report_id": f"optimize_rwr_reward_report_{dataset_suffix}",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source_batch_dir": str(args.batch_dir),
        "min_score": args.min_score,
        "episodes": [
            {
                "episode_ref": _root_relative(path),
                "score": score,
                "score_reasons": reasons,
                "success": bool(payload.get("success")),
                "terminal_reason": payload.get("terminal_reason"),
            }
            for path, payload, score, reasons in scored
        ],
    }
    report_path = args.out_train_dir.parent / f"optimize_rwr_reward_report_{dataset_suffix}.json"
    report_path.write_text(json.dumps(reward_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train_dir": str(args.out_train_dir), "dev_dir": str(args.out_dev_dir), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
