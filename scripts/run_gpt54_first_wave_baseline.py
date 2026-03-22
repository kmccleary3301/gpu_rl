from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.environment import initialize_environment_state, list_action_space, step_environment
from gpu_cockpit.engine.optimize_patch_registry import resolve_optimize_patch_harness
from gpu_cockpit.engine.task_registry import TaskRegistry


SYSTEM_PROMPT = """You are operating a bounded GPU-agent environment.
Choose exactly one next action from the allowed action list.
Respond with a single JSON object only.
Never write patch text or shell code.
Use the environment actions rather than chatting.
Keep reasoning to one or two short sentences.
Do not choose inspect, inspect_build, inspect_profile, inspect_quality, replay, or compare before a run-producing action exists.
For optimize tasks, prefer bench before eval when a baseline command is available.

Return this shape:
{
  "reasoning": "short justification",
  "action_name": "one allowed action name",
  "query": "optional short knowledge query only when action_name is knowledge_query"
}
"""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("\"").strip("'")
    return env


def _provider_settings(config: dict[str, Any], provider: str, dotenv_values: dict[str, str]) -> dict[str, str]:
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY") or dotenv_values.get("OPENAI_API_KEY", "")
        return {
            "provider": provider,
            "model": str(config["model_by_provider"][provider]),
            "url": "https://api.openai.com/v1/chat/completions",
            "api_key": key,
        }
    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY") or dotenv_values.get("OPENROUTER_API_KEY", "")
        return {
            "provider": provider,
            "model": str(config["model_by_provider"][provider]),
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "api_key": key,
        }
    raise ValueError(f"Unsupported provider: {provider}")


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if "\n" in candidate:
            candidate = candidate.split("\n", 1)[1]
        if candidate.endswith("```"):
            candidate = candidate[:-3]
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    if start == -1:
        raise ValueError("Model response did not contain a JSON object.")
    depth = 0
    in_string = False
    escape = False
    end = None
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break
    if end is None:
        raise ValueError("Model response did not contain a complete JSON object.")
    payload = json.loads(candidate[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON was not an object.")
    return payload


def _capture_restore_targets(task_ctx: dict[str, Any]) -> dict[Path, str]:
    patch = task_ctx.get("patch")
    if not isinstance(patch, dict):
        return {}
    target_file = patch.get("patch_target_file")
    if not isinstance(target_file, str):
        return {}
    path = ROOT / target_file
    if not path.exists():
        return {}
    snapshot = {path: path.read_text(encoding="utf-8")}
    initial_target_text = patch.get("initial_target_text")
    if isinstance(initial_target_text, str) and snapshot[path] != initial_target_text:
        path.write_text(initial_target_text, encoding="utf-8")
    return snapshot


def _restore_files(snapshot: dict[Path, str]) -> None:
    for path, original_text in snapshot.items():
        current_text = path.read_text(encoding="utf-8") if path.exists() else None
        if current_text != original_text:
            path.write_text(original_text, encoding="utf-8")


def _chat_completion(
    config: dict[str, Any],
    provider_info: dict[str, str],
    *,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    generation = dict(config.get("generation", {}))
    timeout_s = int(generation.get("timeout_s", 90))
    payload = {
        "model": provider_info["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(generation.get("temperature", 0.2)),
        "max_completion_tokens": int(generation.get("max_completion_tokens", 900)),
    }
    headers = {
        "Authorization": f"Bearer {provider_info['api_key']}",
        "Content-Type": "application/json",
    }
    if provider_info["provider"] == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/openai/codex"
        headers["X-Title"] = "gpu-rl-gpt54-baseline"
    started = time.perf_counter()
    response = _post_json(provider_info["url"], headers, payload, timeout_s)
    elapsed_s = round(time.perf_counter() - started, 3)
    choices = response.get("choices", [])
    if not choices:
        raise ValueError(f"No choices returned by provider {provider_info['provider']}.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str):
        raise ValueError("Model content was not a string.")
    parsed = _extract_json_object(content)
    usage = response.get("usage", {})
    return {
        "provider": provider_info["provider"],
        "model": provider_info["model"],
        "content": content,
        "parsed": parsed,
        "usage": usage if isinstance(usage, dict) else {},
        "elapsed_s": elapsed_s,
    }


def _first_projection_excerpt(value: Any, *, depth: int = 0) -> Any:
    if depth >= 2:
        if isinstance(value, dict):
            return {"keys": list(value)[:6]}
        if isinstance(value, list):
            return value[:2]
        return value
    if isinstance(value, dict):
        items = list(value.items())[:8]
        return {str(key): _first_projection_excerpt(item, depth=depth + 1) for key, item in items}
    if isinstance(value, list):
        return [_first_projection_excerpt(item, depth=depth + 1) for item in value[:3]]
    return value


def _candidate_tree_brief_from_state(state: Any) -> dict[str, Any]:
    lineage_events = [event for event in list(getattr(state, "candidate_lineage_events", []) or []) if isinstance(event, dict)]
    current_candidate_id = getattr(state, "current_candidate_id", None)
    current_parent_candidate_id = getattr(state, "current_candidate_parent_id", None)
    current_event = next((event for event in reversed(lineage_events) if event.get("candidate_id") == current_candidate_id), None)
    sibling_candidate_refs: list[str] = []
    if current_parent_candidate_id is not None and current_candidate_id is not None:
        for event in lineage_events:
            if event.get("parent_candidate_id") != current_parent_candidate_id:
                continue
            sibling_id = event.get("candidate_id")
            if sibling_id and sibling_id != current_candidate_id and sibling_id not in sibling_candidate_refs:
                sibling_candidate_refs.append(str(sibling_id))
    return {
        "history_length": len(list(getattr(state, "candidate_history", []) or [])),
        "current_candidate_id": current_candidate_id,
        "current_parent_candidate_id": current_parent_candidate_id,
        "current_status": getattr(state, "current_candidate_status", None),
        "current_candidate_attempt_index": getattr(state, "current_candidate_attempt_index", None),
        "current_candidate_ref": getattr(state, "current_candidate_run_ref", None),
        "candidate_role": current_event.get("candidate_role") if isinstance(current_event, dict) else None,
        "parent_candidate_ref": current_parent_candidate_id,
        "sibling_candidate_refs": sibling_candidate_refs,
        "why_this_candidate_exists": current_event.get("summary") if isinstance(current_event, dict) else None,
        "recent_events": _first_projection_excerpt(lineage_events[-3:]),
        "best_known_candidate_id": getattr(state, "best_known_candidate_id", None),
        "best_known_candidate_parent_id": getattr(state, "best_known_candidate_parent_id", None),
        "best_known_candidate_run_ref": getattr(state, "best_known_candidate_run_ref", None),
        "best_known_candidate_reason": getattr(state, "best_known_candidate_reason", None),
    }


def _task_context(root: Path, task_ref: str, variant: str) -> dict[str, Any]:
    task = TaskRegistry(root).get(task_ref)
    baseline_payload = None
    if task.baseline_ref:
        baseline_payload = _read_json(root / task.baseline_ref)
    default_command = ["python3", task.reference_impl_ref, "--benchmark-repeats", "2"] if task.reference_impl_ref else []
    baseline_command = list(baseline_payload.get("command", [])) if baseline_payload else None
    if baseline_command and task.reference_impl_ref and len(baseline_command) >= 2 and baseline_command[1] == task.reference_impl_ref:
        default_command = list(baseline_command)
        if "--benchmark-repeats" in default_command:
            idx = default_command.index("--benchmark-repeats")
            if idx + 1 < len(default_command):
                default_command[idx + 1] = "2"
    ctx: dict[str, Any] = {
        "task_ref": task.task_id,
        "task_prompt": task.prompt,
        "verb": task.verb,
        "operator_family": task.operator_family,
        "allowed_backends": list(task.allowed_backends),
        "default_command": default_command,
        "baseline_command": baseline_command,
        "variant": variant,
        "multi_candidate_mode": None,
    }
    if task.task_id == "task/reduction_debug/eval/v1":
        if variant == "negative":
            ctx["patch"] = {
                "patch_target_file": "workloads/reference/triton_row_sum_patchable_candidate.py",
                "initial_target_text": (root / "workloads/reference/triton_row_sum_broken_candidate.py").read_text(encoding="utf-8"),
                "patch_text": (root / "workloads/reference/triton_row_sum_broken_candidate.py").read_text(encoding="utf-8"),
                "patch_intent": "Attempt a reduction repair but leave the broken last-column mask behavior in place.",
                "patch_expected_effect": "Keep the candidate reproducibly broken so the episode stays a usable negative repair trace.",
                "patch_kind": "no_op",
                "patch_transition_kind": "patch_applied",
                "eval_command": ["python3", "workloads/reference/triton_row_sum_patchable_candidate.py", "--benchmark-repeats", "2"],
                "pre_patch_build_spec": "workloads/reference/triton_row_sum_broken_kernel.py:get_build_spec",
                "post_patch_build_spec": "workloads/reference/triton_row_sum_broken_kernel.py:get_build_spec",
            }
        else:
            ctx["patch"] = {
                "patch_target_file": "workloads/reference/triton_row_sum_patchable_candidate.py",
                "initial_target_text": (root / "workloads/reference/triton_row_sum_broken_candidate.py").read_text(encoding="utf-8"),
                "patch_text": (root / "workloads/reference/triton_row_sum_debug_candidate.py").read_text(encoding="utf-8"),
                "patch_intent": "Repair the reduction candidate so it uses the corrected Triton row-sum implementation.",
                "patch_expected_effect": "Turn the broken reduction candidate into a correct implementation without changing the task contract.",
                "patch_kind": "bug_fix",
                "patch_transition_kind": "repaired",
                "eval_command": ["python3", "workloads/reference/triton_row_sum_patchable_candidate.py", "--benchmark-repeats", "2"],
                "pre_patch_build_spec": "workloads/reference/triton_row_sum_broken_kernel.py:get_build_spec",
                "post_patch_build_spec": "workloads/reference/triton_row_sum_repaired_kernel.py:get_build_spec",
            }
        ctx["default_command"] = list(ctx["patch"]["eval_command"])
    elif task.task_id == "task/attention_reformulate/eval/v1":
        if variant == "negative":
            ctx["patch"] = {
                "patch_target_file": "workloads/reference/triton_attention_score_weak_baseline.py",
                "patch_text": (root / "workloads/reference/triton_attention_score_weak_baseline.py").read_text(encoding="utf-8"),
                "patch_intent": "Attempt a reformulation without changing the weak causal attention-score implementation.",
                "patch_expected_effect": "Preserve the baseline implementation so the perf gate still fails and the episode stays a usable negative transform trace.",
                "patch_kind": "no_op",
                "patch_transition_kind": "patch_applied",
                "eval_command": ["python3", "workloads/reference/triton_attention_score_weak_baseline.py", "--benchmark-repeats", "2"],
                "pre_patch_build_spec": None,
                "post_patch_build_spec": None,
            }
        else:
            ctx["patch"] = {
                "patch_target_file": "workloads/reference/triton_attention_score_weak_baseline.py",
                "patch_text": (root / "workloads/reference/triton_attention_score_reformulate_candidate.py").read_text(encoding="utf-8"),
                "patch_intent": "Replace the naive causal attention-score baseline with the tiled Triton implementation.",
                "patch_expected_effect": "Preserve score semantics while improving the implementation strategy and performance profile.",
                "patch_kind": "perf_transform",
                "patch_transition_kind": "reformulated",
                "eval_command": ["python3", "workloads/reference/triton_attention_score_weak_baseline.py", "--benchmark-repeats", "2"],
                "pre_patch_build_spec": None,
                "post_patch_build_spec": "workloads/reference/triton_attention_score_kernel.py:get_build_spec",
            }
        ctx["default_command"] = list(ctx["patch"]["eval_command"])
    else:
        ctx["patch"] = resolve_optimize_patch_harness(root, task.task_id, variant)
        if isinstance(ctx["patch"], dict):
            ctx["default_command"] = list(ctx["patch"]["eval_command"])
    return ctx


def _patch_supports_build(task_ctx: dict[str, Any]) -> bool:
    patch = task_ctx.get("patch")
    if not isinstance(patch, dict):
        return False
    return bool(patch.get("pre_patch_build_spec") or patch.get("post_patch_build_spec"))


def _allowed_actions(action_specs: list[dict[str, Any]], task_ctx: dict[str, Any]) -> list[str]:
    names = [str(spec["action_name"]) for spec in action_specs]
    if task_ctx["task_ref"] == "task/profile_diagnose/eval/v1":
        return [name for name in names if name in {"knowledge_query", "eval", "inspect", "inspect_quality", "inspect_profile", "replay", "compare"}]
    if task_ctx["task_ref"] == "task/attention_reformulate/eval/v1":
        return [
            name
            for name in names
            if name
            in {
                "knowledge_query",
                "build",
                "bench",
                "eval",
                "inspect",
                "inspect_build",
                "inspect_quality",
                "compare",
                "replay",
                "patch_candidate",
                "branch_candidate",
                "revert_candidate",
                "promote_candidate",
            }
        ]
    if task_ctx["task_ref"] == "task/reduction_debug/eval/v1":
        return [
            name
            for name in names
            if name
            in {
                "knowledge_query",
                "build",
                "eval",
                "inspect",
                "inspect_build",
                "inspect_quality",
                "compare",
                "replay",
                "patch_candidate",
                "branch_candidate",
                "revert_candidate",
                "promote_candidate",
            }
        ]
    if task_ctx["verb"] == "optimize" and task_ctx.get("patch"):
        allowed = {
            "knowledge_query",
            "bench",
            "eval",
            "inspect",
            "inspect_build",
            "inspect_profile",
            "inspect_quality",
            "compare",
            "replay",
            "patch_candidate",
            "branch_candidate",
            "revert_candidate",
            "promote_candidate",
        }
        if _patch_supports_build(task_ctx):
            allowed.add("build")
        return [name for name in names if name in allowed]
    if task_ctx["verb"] == "optimize":
        return [name for name in names if name in {"knowledge_query", "run", "bench", "eval", "inspect", "inspect_profile", "inspect_quality", "compare", "replay"}]
    return names


def _task_summary(task_ctx: dict[str, Any]) -> dict[str, Any]:
    patch = task_ctx.get("patch")
    return {
        "task_ref": task_ctx["task_ref"],
        "variant": task_ctx["variant"],
        "verb": task_ctx["verb"],
        "operator_family": task_ctx["operator_family"],
        "prompt": task_ctx["task_prompt"],
        "allowed_backends": task_ctx["allowed_backends"],
        "default_command": task_ctx["default_command"],
        "baseline_command": task_ctx.get("baseline_command"),
        "patch_available": patch is not None,
        "patch_kind": patch.get("patch_kind") if isinstance(patch, dict) else None,
        "candidate_edit_mode": "bounded_patch" if isinstance(patch, dict) else None,
        "multi_candidate_mode": task_ctx.get("multi_candidate_mode"),
    }


def _controller_hints(task_ctx: dict[str, Any], state_snapshot: dict[str, Any], counters: dict[str, int], budgets: dict[str, int]) -> dict[str, Any]:
    has_run = state_snapshot["last_run_ref"] is not None
    opening_actions = ["eval"]
    priority_actions: list[str] = []
    if task_ctx["verb"] == "debug":
        opening_actions = ["build", "eval", "patch_candidate"]
    elif task_ctx["verb"] == "reformulate":
        opening_actions = ["build", "bench", "patch_candidate", "eval"]
    elif task_ctx["verb"] == "diagnose":
        opening_actions = ["eval", "knowledge_query"]
    elif task_ctx["verb"] == "optimize":
        opening_actions = ["bench", "patch_candidate", "eval", "knowledge_query"] if task_ctx.get("patch") else ["bench", "eval", "knowledge_query"]
        if task_ctx.get("multi_candidate_mode") == "branch_revert_negative_v1":
            if counters["patches"] == 0:
                priority_actions = ["patch_candidate", "bench"]
            elif counters["bench_actions"] <= 1:
                priority_actions = ["bench", "compare"]
            elif counters["compares"] < budgets["max_compares"]:
                priority_actions = ["compare", "branch_candidate"]
            elif counters.get("branches", 0) == 0:
                priority_actions = ["branch_candidate", "revert_candidate"]
            elif counters.get("reverts", 0) == 0:
                priority_actions = ["revert_candidate", "eval"]
            else:
                priority_actions = ["eval"]
        elif task_ctx.get("multi_candidate_mode") == "branch_promote_positive_v1":
            if counters["patches"] == 0:
                priority_actions = ["patch_candidate", "bench"]
            elif counters["bench_actions"] <= 1:
                priority_actions = ["bench", "compare"]
            elif counters["compares"] < budgets["max_compares"]:
                priority_actions = ["compare", "branch_candidate"]
            elif counters.get("branches", 0) == 0:
                priority_actions = ["branch_candidate", "promote_candidate"]
            elif counters.get("promotes", 0) == 0:
                priority_actions = ["promote_candidate", "eval"]
            else:
                priority_actions = ["eval"]
        elif task_ctx.get("multi_candidate_mode") == "two_attempt_positive_v1":
            if counters["patches"] == 0:
                priority_actions = ["patch_candidate", "bench"]
            elif counters["bench_actions"] <= 1:
                priority_actions = ["bench", "compare"]
            elif counters["compares"] == 0:
                priority_actions = ["compare", "branch_candidate"]
            elif counters.get("branches", 0) == 0:
                priority_actions = ["branch_candidate", "patch_candidate"]
            elif counters["patches"] < budgets["max_patches"]:
                priority_actions = ["patch_candidate", "bench"]
            elif counters["bench_actions"] <= 2:
                priority_actions = ["bench", "compare"]
            elif counters["compares"] < budgets["max_compares"]:
                priority_actions = ["compare", "promote_candidate"]
            elif counters.get("promotes", 0) == 0:
                priority_actions = ["promote_candidate", "eval"]
            else:
                priority_actions = ["eval"]
            if bool((state_snapshot.get("metadata") or {}).get("last_compare_regression_against_best_known")):
                priority_actions = ["promote_candidate", "eval"] if counters.get("promotes", 0) == 0 else ["eval"]
        elif task_ctx.get("multi_candidate_mode") == "three_attempt_positive_v1":
            if state_snapshot.get("current_candidate_status") == "draft":
                priority_actions = ["patch_candidate", "inspect_quality"]
            elif counters["patches"] == 0:
                priority_actions = ["patch_candidate", "bench"]
            elif counters["bench_actions"] <= 1:
                priority_actions = ["bench", "compare"]
            elif counters["compares"] == 0:
                priority_actions = ["compare", "branch_candidate"]
            elif counters.get("branches", 0) == 0:
                priority_actions = ["branch_candidate", "patch_candidate"]
            elif counters["patches"] == 1:
                priority_actions = ["patch_candidate", "bench"]
            elif counters["bench_actions"] <= 2:
                priority_actions = ["bench", "compare"]
            elif counters["compares"] == 1:
                priority_actions = ["compare", "branch_candidate"]
            elif counters.get("branches", 0) == 1:
                priority_actions = ["branch_candidate", "patch_candidate"]
            elif counters["patches"] == 2:
                priority_actions = ["patch_candidate", "bench"]
            elif counters["bench_actions"] <= 3:
                priority_actions = ["bench", "compare"]
            elif counters["compares"] < budgets["max_compares"]:
                priority_actions = ["compare", "promote_candidate"]
            elif counters.get("promotes", 0) == 0:
                priority_actions = ["promote_candidate", "eval"]
            else:
                priority_actions = ["eval"]
            if bool((state_snapshot.get("metadata") or {}).get("last_compare_regression_against_best_known")):
                priority_actions = ["branch_candidate", "patch_candidate"] if counters.get("branches", 0) < budgets.get("max_branches", 2) else (["promote_candidate", "eval"] if counters.get("promotes", 0) == 0 else ["eval"])
        if (
            task_ctx.get("multi_candidate_mode") is None
            and task_ctx.get("patch")
            and state_snapshot.get("current_candidate_id") is not None
            and counters["compares"] >= budgets["max_compares"]
            and not bool((state_snapshot.get("metadata") or {}).get("last_eval_solved"))
        ):
            priority_actions = ["eval"]
        elif (
            task_ctx.get("multi_candidate_mode") is None
            and
            task_ctx.get("patch")
            and state_snapshot.get("current_candidate_id") is not None
            and counters["bench_actions"] <= 1
        ):
            priority_actions = ["bench", "eval"]
        elif (
            task_ctx.get("multi_candidate_mode") is None
            and
            task_ctx.get("patch")
            and state_snapshot.get("current_candidate_id") is not None
            and counters["bench_actions"] >= 2
            and counters["compares"] < budgets["max_compares"]
        ):
            priority_actions = ["compare", "eval"]
    return {
        "run_producing_actions": ["run", "build", "bench", "eval", "patch_candidate", "branch_candidate", "revert_candidate", "promote_candidate"],
        "requires_prior_run": ["inspect", "inspect_build", "inspect_profile", "inspect_quality", "replay", "compare"],
        "opening_actions": opening_actions if not has_run else [],
        "priority_actions": priority_actions,
    }


def _observation_packet(
    *,
    task_ctx: dict[str, Any],
    allowed_actions: list[str],
    state_snapshot: dict[str, Any],
    budgets: dict[str, int],
    counters: dict[str, int],
    step_records: list[dict[str, Any]],
) -> dict[str, Any]:
    last_step = step_records[-1] if step_records else None
    return {
        "task": _task_summary(task_ctx),
        "allowed_actions": allowed_actions,
        "controller_hints": _controller_hints(task_ctx, state_snapshot, counters, budgets),
        "limits": {
            "step_budget_total": budgets["step_budget"],
            "step_budget_remaining": state_snapshot["step_budget_remaining"],
            "max_retries": budgets["max_retries"],
            "max_patches": budgets["max_patches"],
            "max_compares": budgets["max_compares"],
            "max_replays": budgets["max_replays"],
            "max_knowledge_queries": budgets["max_knowledge_queries"],
        },
        "counters": counters,
        "state": state_snapshot,
        "last_step": last_step,
        "recent_history": step_records[-3:],
    }


def _state_snapshot(state: Any) -> dict[str, Any]:
    return {
        "episode_id": state.episode_id,
        "step_budget_total": state.step_budget_total,
        "step_budget_remaining": state.step_budget_remaining,
        "steps_taken": state.steps_taken,
        "last_run_ref": state.last_run_ref,
        "last_run_id": state.last_run_id,
        "current_candidate_id": state.current_candidate_id,
        "current_candidate_parent_id": state.current_candidate_parent_id,
        "current_candidate_run_ref": state.current_candidate_run_ref,
        "current_candidate_status": state.current_candidate_status,
        "current_candidate_attempt_index": getattr(state, "current_candidate_attempt_index", None),
        "best_known_candidate_id": getattr(state, "best_known_candidate_id", None),
        "best_known_candidate_parent_id": getattr(state, "best_known_candidate_parent_id", None),
        "best_known_candidate_run_ref": getattr(state, "best_known_candidate_run_ref", None),
        "best_known_candidate_reason": getattr(state, "best_known_candidate_reason", None),
        "candidate_history": list(getattr(state, "candidate_history", []) or []),
        "candidate_run_history": list(getattr(state, "candidate_run_history", []) or []),
        "candidate_lineage": _candidate_tree_brief_from_state(state),
        "comparison_anchor_run_ref": state.comparison_anchor_run_ref,
        "comparison_anchor_label": state.comparison_anchor_label,
        "metadata": _first_projection_excerpt(state.metadata),
    }


def _step_record(step: Any) -> dict[str, Any]:
    return {
        "step_index": step.step_index,
        "step_label": step.step_label,
        "action_name": step.action.action_type,
        "reward_total": step.reward_total,
        "reward_components": step.reward_components,
        "recommended_next_actions": step.recommended_next_actions,
        "transition_kind": step.transition_kind,
        "observation": {
            "type": step.observation.observation_type,
            "status": step.observation.status,
            "run_id": step.observation.run_id,
            "task_id": step.observation.task_id,
            "summary_ref": step.observation.summary_ref,
            "salient_artifact_refs": step.observation.salient_artifact_refs,
            "projection_excerpt": _first_projection_excerpt(step.observation.projection),
        },
    }


def _resolve_build_spec(task_ctx: dict[str, Any], state: Any) -> str | None:
    patch = task_ctx.get("patch")
    if not isinstance(patch, dict):
        return None
    if state.current_candidate_id:
        return patch.get("post_patch_build_spec")
    return patch.get("pre_patch_build_spec") or patch.get("post_patch_build_spec")


def _resolve_patch_attempt(task_ctx: dict[str, Any], state: Any) -> dict[str, Any]:
    patch = task_ctx.get("patch")
    if not isinstance(patch, dict):
        raise RuntimeError("patch_candidate requested for a task without patch support.")
    attempt_plans = patch.get("attempt_patch_plans")
    if not isinstance(attempt_plans, list) or not attempt_plans:
        return {
            "patch_text": str(patch["patch_text"]),
            "patch_intent": str(patch["patch_intent"]),
            "patch_expected_effect": str(patch["patch_expected_effect"]),
            "patch_kind": str(patch["patch_kind"]),
            "patch_transition_kind": str(patch["patch_transition_kind"]),
            "candidate_attempt_index": 1 if getattr(state, "current_candidate_attempt_index", None) is None else int(getattr(state, "current_candidate_attempt_index")) + 1,
            "candidate_attempt_reason": "default_single_attempt_patch",
        }
    current_attempt_index = getattr(state, "current_candidate_attempt_index", None)
    next_attempt_index = 1 if current_attempt_index is None else int(current_attempt_index) + 1
    if next_attempt_index < 1 or next_attempt_index > len(attempt_plans):
        next_attempt_index = len(attempt_plans)
    plan = attempt_plans[next_attempt_index - 1]
    if not isinstance(plan, dict):
        raise RuntimeError("attempt_patch_plans contained a non-mapping entry.")
    return {
        "patch_text": str(plan["patch_text"]),
        "patch_intent": str(plan["patch_intent"]),
        "patch_expected_effect": str(plan["patch_expected_effect"]),
        "patch_kind": str(plan["patch_kind"]),
        "patch_transition_kind": str(plan["patch_transition_kind"]),
        "candidate_attempt_index": int(plan.get("attempt_index", next_attempt_index)),
        "candidate_attempt_reason": str(plan.get("attempt_reason", f"attempt_{next_attempt_index}")),
    }


def _resolve_action_kwargs(action_name: str, task_ctx: dict[str, Any], state: Any, query: str | None) -> dict[str, Any]:
    if action_name == "knowledge_query":
        return {"task_ref": task_ctx["task_ref"], "query": query or f"{task_ctx['operator_family']} {task_ctx['verb']}"}
    if action_name == "build":
        triton_build_spec = _resolve_build_spec(task_ctx, state)
        if not triton_build_spec:
            raise RuntimeError("build requested but no Triton build spec is available for this task state.")
        return {"task_ref": task_ctx["task_ref"], "command": [], "section": "build", "triton_build_spec": triton_build_spec}
    if action_name == "bench":
        patch = task_ctx.get("patch")
        if isinstance(patch, dict) and state.current_candidate_id:
            command = patch.get("bench_command") or patch.get("eval_command") or task_ctx["default_command"]
        else:
            command = task_ctx.get("baseline_command") or task_ctx["default_command"]
        return {"task_ref": task_ctx["task_ref"], "command": list(command), "section": "summary"}
    if action_name == "eval":
        patch = task_ctx.get("patch")
        command = patch.get("eval_command") if isinstance(patch, dict) else task_ctx["default_command"]
        return {"task_ref": task_ctx["task_ref"], "command": list(command), "section": "eval"}
    if action_name == "run":
        return {"task_ref": task_ctx["task_ref"], "command": list(task_ctx["default_command"]), "section": "summary"}
    if action_name == "patch_candidate":
        patch = task_ctx.get("patch")
        plan = _resolve_patch_attempt(task_ctx, state)
        return {
            "task_ref": task_ctx["task_ref"],
            "patch_target_file": str(patch["patch_target_file"]),
            "patch_text": plan["patch_text"],
            "patch_intent": plan["patch_intent"],
            "patch_expected_effect": plan["patch_expected_effect"],
            "patch_kind": plan["patch_kind"],
            "patch_transition_kind": plan["patch_transition_kind"],
            "candidate_attempt_index": plan["candidate_attempt_index"],
            "candidate_attempt_reason": plan["candidate_attempt_reason"],
        }
    if action_name == "branch_candidate":
        if state.current_candidate_id is None:
            raise RuntimeError("branch_candidate requested before a candidate exists.")
        return {
            "task_ref": task_ctx["task_ref"],
            "patch_intent": "Branch the current candidate to explore an alternate optimization path.",
            "patch_expected_effect": "Create a new candidate lineage point without mutating the current code yet.",
            "branch_label": f"{task_ctx['operator_family']}_branch",
        }
    if action_name == "revert_candidate":
        if state.current_candidate_id is None:
            raise RuntimeError("revert_candidate requested before a candidate exists.")
        patch = task_ctx.get("patch")
        return {
            "task_ref": task_ctx["task_ref"],
            "patch_intent": "Revert the current candidate back to its previous workspace text for comparison.",
            "patch_expected_effect": "Restore the previous candidate text and preserve lineage for a compare-native loop.",
            "patch_target_file": str(patch["patch_target_file"]) if isinstance(patch, dict) else None,
            "patch_text": str(patch["initial_target_text"]) if isinstance(patch, dict) else None,
            "revert_target_candidate_id": state.current_candidate_id,
        }
    if action_name == "promote_candidate":
        if state.current_candidate_id is None:
            raise RuntimeError("promote_candidate requested before a candidate exists.")
        return {
            "task_ref": task_ctx["task_ref"],
            "patch_intent": "Promote the current candidate as the preferred branch outcome.",
            "patch_expected_effect": "Mark the current candidate as promoted without changing the code again.",
            "promote_label": "preferred_candidate",
        }
    if action_name in {"inspect", "inspect_build", "inspect_profile", "inspect_quality"}:
        if not state.last_run_ref:
            raise RuntimeError(f"{action_name} requested before any run-producing action.")
        return {"run_ref": state.last_run_ref, "section": "quality" if action_name == "inspect_quality" else "summary"}
    if action_name == "replay":
        if not state.last_run_ref:
            raise RuntimeError("replay requested before any run-producing action.")
        return {"run_ref": state.last_run_ref}
    if action_name == "compare":
        if not state.last_run_ref:
            raise RuntimeError("compare requested before any run-producing action.")
        lhs_run_ref = state.comparison_anchor_run_ref or (state.run_history[0] if state.run_history else None)
        if (
            task_ctx.get("multi_candidate_mode") == "two_attempt_positive_v1"
            and getattr(state, "best_known_candidate_run_ref", None)
            and getattr(state, "current_candidate_attempt_index", None)
            and int(getattr(state, "current_candidate_attempt_index")) >= 2
        ):
            lhs_run_ref = getattr(state, "best_known_candidate_run_ref")
        if not lhs_run_ref:
            raise RuntimeError("compare requested without a comparison anchor.")
        return {"lhs_run_ref": lhs_run_ref, "rhs_run_ref": state.last_run_ref}
    raise RuntimeError(f"Unsupported action resolution: {action_name}")


def _fallback_action(task_ctx: dict[str, Any], state: Any, counters: dict[str, int], budgets: dict[str, int]) -> str:
    if state.last_run_ref is None:
        if task_ctx["verb"] == "optimize" and task_ctx.get("baseline_command") and counters["bench_actions"] == 0:
            return "bench"
        if task_ctx["verb"] == "debug" and counters["eval_actions"] == 0:
            return "eval"
        if task_ctx.get("baseline_command") and counters["bench_actions"] == 0:
            return "bench"
        if task_ctx.get("patch") and counters["patches"] == 0 and task_ctx["verb"] in {"debug", "reformulate"}:
            return "patch_candidate"
        return "eval"
    if task_ctx["verb"] == "optimize" and task_ctx.get("patch") and counters["patches"] == 0 and state.current_candidate_id is None:
        return "patch_candidate"
    if task_ctx["verb"] == "optimize" and task_ctx.get("patch") and state.current_candidate_id is not None:
        if task_ctx.get("multi_candidate_mode") == "branch_revert_negative_v1":
            if counters["compares"] < budgets["max_compares"] and counters["bench_actions"] >= 2:
                return "compare"
            if counters.get("branches", 0) == 0:
                return "branch_candidate"
            if counters.get("reverts", 0) == 0:
                return "revert_candidate"
            if counters["eval_actions"] == 0:
                return "eval"
        if task_ctx.get("multi_candidate_mode") == "branch_promote_positive_v1":
            if counters["compares"] < budgets["max_compares"] and counters["bench_actions"] >= 2:
                return "compare"
            if counters.get("branches", 0) == 0:
                return "branch_candidate"
            if counters.get("promotes", 0) == 0:
                return "promote_candidate"
            if counters["eval_actions"] == 0:
                return "eval"
    if task_ctx.get("multi_candidate_mode") == "two_attempt_positive_v1":
        if counters["patches"] == 0:
            return "patch_candidate"
        if counters["bench_actions"] < 2:
            return "bench"
        if counters["compares"] == 0:
            return "compare"
        if counters.get("branches", 0) == 0:
            return "branch_candidate"
        if counters["patches"] < budgets["max_patches"]:
            return "patch_candidate"
        if counters["bench_actions"] < 3:
            return "bench"
        if counters["compares"] < budgets["max_compares"]:
            return "compare"
        if counters.get("promotes", 0) == 0:
            return "promote_candidate"
        if counters["eval_actions"] == 0:
            return "eval"
    if task_ctx.get("multi_candidate_mode") == "three_attempt_positive_v1":
        if counters["patches"] == 0:
            return "patch_candidate"
        if counters["bench_actions"] < 2:
            return "bench"
        if counters["compares"] == 0:
            return "compare"
        if counters.get("branches", 0) == 0:
            return "branch_candidate"
        if counters["patches"] == 1:
            return "patch_candidate"
        if counters["bench_actions"] < 3:
            return "bench"
        if counters["compares"] == 1:
            return "compare"
        if counters.get("branches", 0) == 1:
            return "branch_candidate"
        if counters["patches"] == 2:
            return "patch_candidate"
        if counters["bench_actions"] < 4:
            return "bench"
        if counters["compares"] < budgets["max_compares"]:
            return "compare"
        if counters.get("promotes", 0) == 0:
            return "promote_candidate"
        if counters["eval_actions"] == 0:
            return "eval"
    if task_ctx["verb"] == "optimize" and task_ctx.get("patch") and state.current_candidate_id is not None:
        if counters["bench_actions"] <= 1:
            return "bench"
        if counters["compares"] < budgets["max_compares"]:
            return "compare"
        if counters["eval_actions"] == 0:
            return "eval"
    if counters["knowledge_queries"] < budgets["max_knowledge_queries"] and task_ctx["verb"] != "optimize":
        return "knowledge_query"
    if counters["replays"] < budgets["max_replays"]:
        return "replay"
    return "inspect_quality"


def _action_within_limits(action_name: str, counters: dict[str, int], budgets: dict[str, int], task_ctx: dict[str, Any]) -> bool:
    if action_name == "knowledge_query":
        return counters["knowledge_queries"] < budgets["max_knowledge_queries"]
    if action_name == "patch_candidate":
        return task_ctx.get("patch") is not None and counters["patches"] < budgets["max_patches"]
    if action_name == "branch_candidate":
        return task_ctx.get("patch") is not None and counters.get("branches", 0) < budgets.get("max_branches", 1)
    if action_name == "revert_candidate":
        return task_ctx.get("patch") is not None and counters.get("reverts", 0) < budgets.get("max_reverts", 1)
    if action_name == "promote_candidate":
        return task_ctx.get("patch") is not None and counters.get("promotes", 0) < budgets.get("max_promotes", 1)
    if action_name == "compare":
        return counters["compares"] < budgets["max_compares"]
    if action_name == "replay":
        return counters["replays"] < budgets["max_replays"]
    return True


def _action_allowed_in_state(action_name: str, state: Any, task_ctx: dict[str, Any], counters: dict[str, int], budgets: dict[str, int]) -> bool:
    if (
        task_ctx.get("multi_candidate_mode") in {"two_attempt_positive_v1", "three_attempt_positive_v1"}
        and action_name in {"inspect", "inspect_build", "inspect_profile", "inspect_quality", "replay"}
        and counters.get("compares", 0) > 0
        and counters.get("promotes", 0) == 0
    ):
        return False
    if (
        task_ctx.get("multi_candidate_mode") == "three_attempt_positive_v1"
        and action_name == "knowledge_query"
        and counters.get("patches", 0) >= 2
        and counters.get("compares", 0) >= 2
        and counters.get("promotes", 0) == 0
    ):
        return False
    if (
        task_ctx.get("multi_candidate_mode") in {"two_attempt_positive_v1", "three_attempt_positive_v1"}
        and getattr(state, "current_candidate_status", None) == "draft"
        and action_name in {"bench", "build", "compare", "eval", "promote_candidate", "branch_candidate"}
    ):
        return False
    if (
        task_ctx.get("multi_candidate_mode") is None
        and
        task_ctx["verb"] == "optimize"
        and task_ctx.get("patch") is not None
        and state.current_candidate_id is not None
        and counters["patches"] >= budgets["max_patches"]
        and counters["compares"] >= budgets["max_compares"]
        and not bool(state.metadata.get("last_eval_solved"))
    ):
        return action_name == "eval"
    if action_name in {"inspect", "inspect_build", "inspect_profile", "inspect_quality", "replay"}:
        return state.last_run_ref is not None
    if action_name == "compare":
        return state.last_run_ref is not None and (state.comparison_anchor_run_ref is not None or bool(state.run_history))
    if (
        task_ctx.get("multi_candidate_mode") == "branch_revert_negative_v1"
        and action_name == "revert_candidate"
        and state.current_candidate_id is not None
    ):
        if counters.get("branches", 0) == 0:
            return False
    if (
        task_ctx.get("multi_candidate_mode") == "branch_promote_positive_v1"
        and action_name == "promote_candidate"
        and state.current_candidate_id is not None
    ):
        if counters.get("branches", 0) == 0:
            return False
    if (
        task_ctx.get("multi_candidate_mode") == "branch_promote_positive_v1"
        and action_name == "branch_candidate"
        and state.current_candidate_id is not None
    ):
        if counters.get("promotes", 0) > 0:
            return False
    if (
        task_ctx.get("multi_candidate_mode") == "branch_revert_negative_v1"
        and action_name == "branch_candidate"
        and state.current_candidate_id is not None
    ):
        if counters.get("reverts", 0) > 0:
            return False
    if (
        task_ctx.get("multi_candidate_mode") == "two_attempt_positive_v1"
        and action_name == "branch_candidate"
        and state.current_candidate_id is not None
    ):
        if counters.get("compares", 0) == 0 or counters.get("branches", 0) > 0:
            return False
    if (
        task_ctx.get("multi_candidate_mode") == "two_attempt_positive_v1"
        and action_name == "patch_candidate"
        and state.current_candidate_id is not None
    ):
        if counters.get("patches", 0) >= budgets.get("max_patches", 1):
            return False
        if counters.get("branches", 0) == 0:
            return False
    if (
        task_ctx.get("multi_candidate_mode") == "two_attempt_positive_v1"
        and action_name == "promote_candidate"
        and state.current_candidate_id is not None
    ):
        if counters.get("patches", 0) < budgets.get("max_patches", 2) or counters.get("compares", 0) < budgets.get("max_compares", 2):
            return False
    if (
        task_ctx.get("multi_candidate_mode") == "branch_revert_negative_v1"
        and action_name == "eval"
        and state.current_candidate_id is not None
    ):
        if counters.get("branches", 0) == 0 or counters.get("reverts", 0) == 0:
            return False
    if (
        task_ctx.get("multi_candidate_mode") == "branch_promote_positive_v1"
        and action_name == "eval"
        and state.current_candidate_id is not None
    ):
        if counters.get("branches", 0) == 0 or counters.get("promotes", 0) == 0:
            return False
    if (
        task_ctx.get("multi_candidate_mode") == "two_attempt_positive_v1"
        and action_name == "eval"
        and state.current_candidate_id is not None
    ):
        if counters.get("promotes", 0) == 0:
            return False
    if (
        task_ctx.get("multi_candidate_mode") == "three_attempt_positive_v1"
        and action_name == "branch_candidate"
        and state.current_candidate_id is not None
    ):
        required_compares = counters.get("branches", 0) + 1
        if counters.get("compares", 0) < required_compares or counters.get("branches", 0) >= budgets.get("max_branches", 2):
            return False
    if (
        task_ctx.get("multi_candidate_mode") == "three_attempt_positive_v1"
        and action_name == "patch_candidate"
        and state.current_candidate_id is not None
    ):
        if counters.get("patches", 0) >= budgets.get("max_patches", 3):
            return False
        if counters.get("branches", 0) < counters.get("patches", 0):
            return False
    if (
        task_ctx.get("multi_candidate_mode") == "three_attempt_positive_v1"
        and action_name == "promote_candidate"
        and state.current_candidate_id is not None
    ):
        if counters.get("patches", 0) < budgets.get("max_patches", 3) or counters.get("compares", 0) < budgets.get("max_compares", 3):
            return False
    if action_name in {"branch_candidate", "revert_candidate", "promote_candidate"}:
        return state.current_candidate_id is not None
    if (
        action_name == "eval"
        and task_ctx["verb"] == "optimize"
        and task_ctx.get("patch") is not None
        and state.current_candidate_id is not None
        and counters["bench_actions"] <= 1
    ):
        return False
    if (
        action_name == "eval"
        and task_ctx["verb"] == "optimize"
        and task_ctx.get("patch") is not None
        and state.current_candidate_id is not None
        and counters["bench_actions"] >= 2
        and counters["compares"] < budgets["max_compares"]
    ):
        return False
    if (
        task_ctx.get("multi_candidate_mode") == "two_attempt_positive_v1"
        and action_name == "compare"
        and state.current_candidate_id is not None
    ):
        required_patch_count = counters.get("compares", 0) + 1
        required_bench_count = required_patch_count + 1
        if counters.get("patches", 0) < required_patch_count:
            return False
        if counters["bench_actions"] < required_bench_count:
            return False
    if (
        task_ctx.get("multi_candidate_mode") == "three_attempt_positive_v1"
        and action_name == "compare"
        and state.current_candidate_id is not None
    ):
        required_patch_count = counters.get("compares", 0) + 1
        required_bench_count = required_patch_count + 1
        if counters.get("patches", 0) < required_patch_count:
            return False
        if counters["bench_actions"] < required_bench_count:
            return False
    if (
        task_ctx.get("multi_candidate_mode") == "three_attempt_positive_v1"
        and action_name == "eval"
        and state.current_candidate_id is not None
    ):
        if counters.get("promotes", 0) == 0:
            return False
    return True


def _should_stop(task_ctx: dict[str, Any], last_step: dict[str, Any] | None, counters: dict[str, int], budgets: dict[str, int]) -> tuple[bool, str]:
    if last_step is None:
        return False, "no_steps_yet"
    action_name = str(last_step.get("action_name"))
    obs = last_step.get("observation", {})
    status = str(obs.get("status"))
    if action_name == "eval" and status == "ok":
        return True, "success_after_eval"
    if (
        task_ctx["verb"] == "optimize"
        and task_ctx["variant"] == "negative"
        and counters["patches"] >= 1
        and counters["compares"] >= 1
        and counters["eval_actions"] >= 1
        and action_name in {"replay", "inspect_quality"}
    ):
        return True, "negative_trace_complete"
    if (
        task_ctx.get("multi_candidate_mode") == "branch_revert_negative_v1"
        and counters.get("branches", 0) >= 1
        and counters.get("reverts", 0) >= 1
        and action_name == "eval"
    ):
        return True, "multi_candidate_negative_complete"
    if (
        task_ctx.get("multi_candidate_mode") == "two_attempt_positive_v1"
        and counters.get("patches", 0) >= budgets.get("max_patches", 2)
        and counters.get("compares", 0) >= budgets.get("max_compares", 2)
        and counters.get("promotes", 0) >= 1
        and action_name == "eval"
    ):
        return True, "two_attempt_positive_complete"
    if (
        task_ctx.get("multi_candidate_mode") == "three_attempt_positive_v1"
        and counters.get("patches", 0) >= budgets.get("max_patches", 3)
        and counters.get("compares", 0) >= budgets.get("max_compares", 3)
        and counters.get("promotes", 0) >= 1
        and action_name == "eval"
    ):
        return True, "three_attempt_positive_complete"
    if action_name == "eval" and status != "ok" and counters["patches"] >= budgets["max_patches"] and counters["eval_actions"] >= 2:
        return True, "post_patch_eval_failed"
    if task_ctx["verb"] == "diagnose" and action_name in {"eval", "inspect_quality", "replay"} and counters["eval_actions"] >= 1:
        return True, "diagnose_episode_complete"
    return False, "continue"


def _run_episode(
    root: Path,
    config: dict[str, Any],
    provider_info: dict[str, str],
    task_spec: dict[str, Any],
) -> dict[str, Any]:
    budgets = {key: int(value) for key, value in config["budgets"].items()}
    task_ctx = _task_context(root, str(task_spec["task_ref"]), str(task_spec["variant"]))
    for extra_key in ("multi_candidate_mode",):
        if extra_key in task_spec:
            task_ctx[extra_key] = task_spec[extra_key]
    workspace_snapshot = _capture_restore_targets(task_ctx)
    action_specs = [spec.model_dump(mode="json") for spec in list_action_space()]
    allowed_actions = _allowed_actions(action_specs, task_ctx)
    state = initialize_environment_state(root, task_ctx["task_ref"], policy_id=str(config["policy_id"]), step_budget=budgets["step_budget"])
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

    try:
        while state.step_budget_remaining > 0:
            state_view = _state_snapshot(state)
            observation_packet = _observation_packet(
                task_ctx=task_ctx,
                allowed_actions=allowed_actions,
                state_snapshot=state_view,
                budgets=budgets,
                counters=counters,
                step_records=step_records,
            )
            user_prompt = json.dumps(observation_packet, indent=2, sort_keys=True)
            model_response: dict[str, Any] | None = None
            last_provider_error: dict[str, Any] | None = None
            for provider in config["provider_priority"]:
                active_provider = provider if provider_info["provider"] == provider else provider
                dotenv_values = _load_dotenv(root / ".env")
                candidate_provider_info = _provider_settings(config, active_provider, dotenv_values)
                if not candidate_provider_info["api_key"]:
                    last_provider_error = {"provider": active_provider, "error": "missing_api_key"}
                    continue
                try:
                    model_response = _chat_completion(config, candidate_provider_info, system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
                    provider_info = candidate_provider_info
                    break
                except (error.HTTPError, error.URLError, TimeoutError, ValueError) as exc:
                    counters["provider_failures"] += 1
                    body = None
                    if isinstance(exc, error.HTTPError):
                        try:
                            body = exc.read().decode("utf-8")
                        except Exception:
                            body = None
                    last_provider_error = {
                        "provider": active_provider,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "body": body,
                    }
            if model_response is None:
                terminal_reason = "provider_failure"
                model_turns.append(
                    {
                        "turn_index": len(model_turns),
                        "observation_packet": observation_packet,
                        "provider_error": last_provider_error,
                    }
                )
                break

            counters["model_calls"] += 1
            parsed = model_response.get("parsed", {})
            requested_action = str(parsed.get("action_name", "")).strip()
            query = str(parsed.get("query", "")).strip() or None
            if (
                requested_action not in allowed_actions
                or not _action_within_limits(requested_action, counters, budgets, task_ctx)
                or not _action_allowed_in_state(requested_action, state, task_ctx, counters, budgets)
            ):
                counters["controller_rejections"] += 1
                requested_action = _fallback_action(task_ctx, state, counters, budgets)
            turn_record = {
                "turn_index": len(model_turns),
                "provider": model_response["provider"],
                "model": model_response["model"],
                "elapsed_s": model_response["elapsed_s"],
                "usage": model_response["usage"],
                "observation_packet": observation_packet,
                "raw_response": model_response["content"],
                "parsed_response": parsed,
                "selected_action": requested_action,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }
            model_turns.append(turn_record)

            try:
                kwargs = _resolve_action_kwargs(requested_action, task_ctx, state, query)
                state, step = step_environment(root, state, action_name=requested_action, **kwargs)
                step_dict = _step_record(step)
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
                should_stop, terminal_reason = _should_stop(task_ctx, step_dict, counters, budgets)
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
                    break
    finally:
        _restore_files(workspace_snapshot)

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
        "provider": provider_info["provider"],
        "model": provider_info["model"],
        "step_count": len([step for step in step_records if step.get("step_label") != "controller_error"]),
        "counters": counters,
        "state": _state_snapshot(state),
        "steps": step_records,
        "model_turns": model_turns,
    }


def _summarize_batch(config: dict[str, Any], episode_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not episode_reports:
        return {
            "config_id": str(config["config_id"]),
            "task_count": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "avg_step_count": 0.0,
            "avg_patch_count": 0.0,
            "avg_compare_count": 0.0,
            "avg_replay_count": 0.0,
            "avg_failed_tool_calls": 0.0,
        }
    return {
        "config_id": str(config["config_id"]),
        "task_count": len(episode_reports),
        "success_count": sum(1 for report in episode_reports if report["success"]),
        "success_rate": round(sum(1 for report in episode_reports if report["success"]) / len(episode_reports), 4),
        "avg_step_count": round(sum(int(report["step_count"]) for report in episode_reports) / len(episode_reports), 4),
        "avg_patch_count": round(sum(int(report["counters"]["patches"]) for report in episode_reports) / len(episode_reports), 4),
        "avg_compare_count": round(sum(int(report["counters"]["compares"]) for report in episode_reports) / len(episode_reports), 4),
        "avg_replay_count": round(sum(int(report["counters"]["replays"]) for report in episode_reports) / len(episode_reports), 4),
        "avg_failed_tool_calls": round(sum(int(report["counters"]["failed_tool_calls"]) for report in episode_reports) / len(episode_reports), 4),
        "success_by_variant": {
            "positive": sum(1 for report in episode_reports if report["variant"] == "positive" and report["success"]),
            "negative": sum(1 for report in episode_reports if report["variant"] == "negative" and report["success"]),
        },
        "success_by_verb": {
            verb: {
                "tasks": sum(1 for report in episode_reports if report["verb"] == verb),
                "successes": sum(1 for report in episode_reports if report["verb"] == verb and report["success"]),
            }
            for verb in sorted({str(report["verb"]) for report in episode_reports})
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path, help="Path to the GPT-5.4 baseline config JSON")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for batch artifacts")
    parser.add_argument("--variant", choices=["positive", "negative", "all"], default="all", help="Optional variant filter")
    parser.add_argument("--task-ref", action="append", default=[], help="Optional task_ref filter; repeat to include multiple tasks")
    parser.add_argument("--task-limit", type=int, default=None, help="Optional cap on the number of configured task variants")
    parser.add_argument("--provider", choices=["openai", "openrouter"], default=None, help="Force one provider instead of using configured priority order")
    args = parser.parse_args()

    config = _read_json(args.config_path)
    if args.provider is not None:
        config["provider_priority"] = [args.provider]
    dotenv_values = _load_dotenv(ROOT / ".env")
    provider_info = _provider_settings(config, str(config["provider_priority"][0]), dotenv_values)

    tasks = list(config.get("tasks", []))
    if args.variant != "all":
        tasks = [task for task in tasks if str(task.get("variant")) == args.variant]
    if args.task_ref:
        allowed_task_refs = set(args.task_ref)
        tasks = [task for task in tasks if str(task.get("task_ref")) in allowed_task_refs]
    if args.task_limit is not None:
        tasks = tasks[: args.task_limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    episode_reports: list[dict[str, Any]] = []
    for task_spec in tasks:
        report = _run_episode(ROOT, config, provider_info, task_spec)
        episode_reports.append(report)
        label = f"{task_spec['task_ref'].replace('/', '__')}__{task_spec['variant']}.json"
        (args.out_dir / label).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    batch_report = {
        "report_id": f"gpt54_first_wave_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "config": config,
        "summary": _summarize_batch(config, episode_reports),
        "episodes": [
            {
                "task_ref": report["task_ref"],
                "variant": report["variant"],
                "verb": report["verb"],
                "success": report["success"],
                "terminal_reason": report["terminal_reason"],
                "provider": report["provider"],
                "model": report["model"],
                "step_count": report["step_count"],
                "counters": report["counters"],
            }
            for report in episode_reports
        ],
    }
    (args.out_dir / "batch_report.json").write_text(json.dumps(batch_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(batch_report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
