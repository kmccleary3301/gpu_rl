from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_gpt54_first_wave_baseline as harness

from gpu_cockpit.engine.training import _build_model_load_kwargs


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _step_count(step_records: list[dict[str, Any]]) -> int:
    return len([step for step in step_records if step.get("step_label") != "controller_error"])


def _partial_payload(
    *,
    phase: str,
    task_ctx: dict[str, Any],
    state: Any,
    model_label: str,
    terminal_reason: str,
    counters: dict[str, int],
    step_records: list[dict[str, Any]],
    model_turns: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "phase": phase,
        "updated_at": datetime.now(tz=UTC).isoformat(),
        "task_ref": task_ctx["task_ref"],
        "variant": task_ctx["variant"],
        "verb": task_ctx["verb"],
        "model": model_label,
        "terminal_reason": terminal_reason,
        "counters": counters,
        "state": harness._state_snapshot(state),
        "step_count": _step_count(step_records),
        "steps": step_records,
        "model_turns": model_turns,
    }
    if extra:
        payload.update(extra)
    return payload


def _load_local_policy(
    *,
    model_id: str,
    tokenizer_id: str | None,
    adapter_mode: str,
    adapter_dir: Path | None,
):
    effective_tokenizer_id = tokenizer_id or model_id
    tokenizer = AutoTokenizer.from_pretrained(effective_tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    use_cuda = torch.cuda.is_available()
    model_kwargs = _build_model_load_kwargs(adapter_mode, use_cuda=use_cuda)
    if use_cuda:
        model_kwargs["offload_folder"] = str((ROOT / "artifacts" / "training" / "hf_offload_local_eval").resolve())
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer


def _generate_action(
    *,
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1] :]
    content = tokenizer.decode(generated, skip_special_tokens=True)
    parsed = harness._extract_json_object(content)
    return {"content": content, "parsed": parsed}


def _run_episode(
    config: dict[str, Any],
    task_spec: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    model_label: str,
    partial_output_path: Path | None = None,
) -> dict[str, Any]:
    budgets = {key: int(value) for key, value in config["budgets"].items()}
    task_ctx = harness._task_context(ROOT, str(task_spec["task_ref"]), str(task_spec["variant"]))
    for extra_key in ("multi_candidate_mode",):
        if extra_key in task_spec:
            task_ctx[extra_key] = task_spec[extra_key]
    workspace_snapshot = harness._capture_restore_targets(task_ctx)
    action_specs = [spec.model_dump(mode="json") for spec in harness.list_action_space()]
    allowed_actions = harness._allowed_actions(action_specs, task_ctx)
    state = harness.initialize_environment_state(ROOT, task_ctx["task_ref"], policy_id=str(config["policy_id"]), step_budget=budgets["step_budget"])
    step_records: list[dict[str, Any]] = []
    model_turns: list[dict[str, Any]] = []
    counters = {
        "model_calls": 0,
        "provider_failures": 0,
        "failed_tool_calls": 0,
        "controller_rejections": 0,
        "knowledge_queries": 0,
        "patches": 0,
        "branches": 0,
        "reverts": 0,
        "promotes": 0,
        "compares": 0,
        "replays": 0,
        "eval_actions": 0,
        "bench_actions": 0,
    }
    terminal_reason = "budget_exhausted"
    generation = dict(config.get("generation", {}))

    def write_partial(phase: str, extra: dict[str, Any] | None = None) -> None:
        if partial_output_path is None:
            return
        _write_json(
            partial_output_path,
            _partial_payload(
                phase=phase,
                task_ctx=task_ctx,
                state=state,
                model_label=model_label,
                terminal_reason=terminal_reason,
                counters=counters,
                step_records=step_records,
                model_turns=model_turns,
                extra=extra,
            ),
        )

    try:
        write_partial("initialized")
        while state.step_budget_remaining > 0:
            state_view = harness._state_snapshot(state)
            observation_packet = harness._observation_packet(
                task_ctx=task_ctx,
                allowed_actions=allowed_actions,
                state_snapshot=state_view,
                budgets=budgets,
                counters=counters,
                step_records=step_records,
            )
            user_prompt = json.dumps(observation_packet, indent=2, sort_keys=True)
            write_partial(
                "awaiting_model_response",
                {"turn_index": len(model_turns), "observation_packet": observation_packet},
            )
            response = _generate_action(
                model=model,
                tokenizer=tokenizer,
                system_prompt=harness.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=float(generation.get("temperature", 0.0)),
                max_new_tokens=int(generation.get("max_new_tokens", 256)),
            )
            counters["model_calls"] += 1
            parsed = response.get("parsed", {})
            requested_action = str(parsed.get("action_name", "")).strip()
            query = str(parsed.get("query", "")).strip() or None
            if (
                requested_action not in allowed_actions
                or not harness._action_within_limits(requested_action, counters, budgets, task_ctx)
                or not harness._action_allowed_in_state(requested_action, state, task_ctx, counters, budgets)
            ):
                counters["controller_rejections"] += 1
                requested_action = harness._fallback_action(task_ctx, state, counters, budgets)
            model_turns.append(
                {
                    "turn_index": len(model_turns),
                    "model": model_label,
                    "observation_packet": observation_packet,
                    "raw_response": response["content"],
                    "parsed_response": parsed,
                    "selected_action": requested_action,
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                }
            )
            write_partial(
                "model_response_received",
                {"turn_index": len(model_turns) - 1, "selected_action": requested_action},
            )
            try:
                kwargs = harness._resolve_action_kwargs(requested_action, task_ctx, state, query)
                state, step = harness.step_environment(ROOT, state, action_name=requested_action, **kwargs)
                step_dict = harness._step_record(step)
                step_records.append(step_dict)
                if requested_action == "knowledge_query":
                    counters["knowledge_queries"] += 1
                elif requested_action == "patch_candidate":
                    counters["patches"] += 1
                elif requested_action == "compare":
                    counters["compares"] += 1
                elif requested_action == "branch_candidate":
                    counters["branches"] += 1
                elif requested_action == "revert_candidate":
                    counters["reverts"] += 1
                elif requested_action == "promote_candidate":
                    counters["promotes"] += 1
                elif requested_action == "replay":
                    counters["replays"] += 1
                elif requested_action == "eval":
                    counters["eval_actions"] += 1
                elif requested_action == "bench":
                    counters["bench_actions"] += 1
                write_partial(
                    "step_applied",
                    {
                        "turn_index": len(model_turns) - 1,
                        "selected_action": requested_action,
                        "last_step": step_dict,
                    },
                )
                should_stop, terminal_reason = harness._should_stop(task_ctx, step_dict, counters, budgets)
                if should_stop:
                    break
            except Exception as exc:
                counters["failed_tool_calls"] += 1
                step_records.append(
                    {
                        "step_index": len(step_records),
                        "step_label": "controller_error",
                        "action_name": requested_action,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                if counters["failed_tool_calls"] > budgets["max_retries"]:
                    terminal_reason = "tool_failure"
                write_partial(
                    "tool_failure",
                    {
                        "turn_index": len(model_turns) - 1,
                        "selected_action": requested_action,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                if counters["failed_tool_calls"] > budgets["max_retries"]:
                    break
    finally:
        harness._restore_files(workspace_snapshot)

    if terminal_reason == "continue" and state.step_budget_remaining <= 0:
        terminal_reason = "budget_exhausted"

    success = False
    if step_records:
        last = step_records[-1]
        success = str(last.get("action_name")) == "eval" and str(last.get("observation", {}).get("status")) == "ok"
    return {
        "task_ref": task_ctx["task_ref"],
        "variant": task_ctx["variant"],
        "verb": task_ctx["verb"],
        "success": success,
        "terminal_reason": terminal_reason,
        "model": model_label,
        "step_count": _step_count(step_records),
        "counters": counters,
        "state": harness._state_snapshot(state),
        "steps": step_records,
        "model_turns": model_turns,
    }
    write_partial("completed", {"success": success, "report": report})
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--label", type=str, default="base_qwen7b")
    args = parser.parse_args()

    config = _read_json(args.config_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = _load_local_policy(
        model_id=str(config["model_id"]),
        tokenizer_id=str(config.get("tokenizer_id")) if config.get("tokenizer_id") is not None else None,
        adapter_mode=str(config.get("adapter_mode", "qlora")),
        adapter_dir=args.adapter_dir,
    )
    episode_reports: list[dict[str, Any]] = []
    for task_spec in config.get("tasks", []):
        label = f"{task_spec['task_ref'].replace('/', '__')}__{task_spec['variant']}.json"
        partial_label = label.replace(".json", ".partial.json")
        report = _run_episode(
            config,
            task_spec,
            model=model,
            tokenizer=tokenizer,
            model_label=args.label,
            partial_output_path=args.out_dir / partial_label,
        )
        episode_reports.append(report)
        _write_json(args.out_dir / label, report)
    summary = harness._summarize_batch(config, episode_reports)
    batch_report = {
        "report_id": f"local_checkpoint_eval_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "config": config,
        "model_label": args.label,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir is not None else None,
        "summary": summary,
        "episodes": [
            {
                "task_ref": report["task_ref"],
                "variant": report["variant"],
                "success": report["success"],
                "terminal_reason": report["terminal_reason"],
                "step_count": report["step_count"],
                "counters": report["counters"],
            }
            for report in episode_reports
        ],
    }
    _write_json(args.out_dir / "batch_report.json", batch_report)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
