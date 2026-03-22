from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_gpt54_first_wave_baseline as harness

from gpu_cockpit.engine.environment import initialize_environment_state, step_environment


TARGET_FILE = "workloads/reference/triton_row_sum_semioopen_kernel.py"
TASK_REF = "task/reduction_row_sum_semioopen/eval/v1"
EVAL_COMMAND = ["python3", "workloads/reference/triton_row_sum_semioopen_candidate.py", "--benchmark-repeats", "2"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_prompt(task_prompt: str, target_text: str) -> str:
    payload = {
        "task_ref": TASK_REF,
        "goal": task_prompt,
        "target_file": TARGET_FILE,
        "instructions": [
            "Return JSON only.",
            "Include a top-level key `replacement_text` containing the full replacement Python module for the target file.",
            "Do not include markdown fences.",
            "Keep the kernel implementation Triton-based and CUDA-only.",
        ],
        "current_file_text": target_text,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    task = harness.TaskRegistry(ROOT).get(TASK_REF)
    target_path = ROOT / TARGET_FILE
    original_text = target_path.read_text(encoding="utf-8")
    dotenv_values = harness._load_dotenv(ROOT / ".env")
    provider_info = harness._provider_settings(
        {
            "provider_priority": [args.provider],
            "model_by_provider": {
                "openai": "gpt-5.4",
                "openrouter": "openai/gpt-5.4-pro",
            },
            "generation": {
                "temperature": 0.2,
                "max_completion_tokens": 1400,
                "timeout_s": 120,
            },
        },
        args.provider,
        dotenv_values,
    )
    prompt = _render_prompt(task.prompt, original_text)
    response = harness._chat_completion(
        {
            "generation": {
                "temperature": 0.2,
                "max_completion_tokens": 1400,
                "timeout_s": 120,
            }
        },
        provider_info,
        system_prompt="You write precise kernel code edits. Return valid JSON only.",
        user_prompt=prompt,
    )
    parsed = harness._extract_json_object(response["content"])
    replacement_text = str(parsed.get("replacement_text", "")).strip()
    report: dict[str, Any] = {
        "report_id": f"gpt54_semioopen_kernel_probe_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "provider": response["provider"],
        "model": response["model"],
        "usage": response["usage"],
        "task_ref": TASK_REF,
        "target_file": TARGET_FILE,
        "raw_response": response["content"],
        "parsed_response": parsed,
        "status": "failed",
    }
    if not replacement_text:
        (args.out_dir / "probe_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 1
    try:
        target_path.write_text(replacement_text + "\n", encoding="utf-8")
        state = initialize_environment_state(ROOT, TASK_REF, policy_id="gpt54_semioopen_kernel_probe_v1", step_budget=2)
        state, step = step_environment(ROOT, state, action_name="eval", task_ref=TASK_REF, command=list(EVAL_COMMAND), section="eval")
        report["status"] = "ok" if step.observation.status == "ok" else "failed"
        report["eval_run_ref"] = state.last_run_ref
        report["eval_run_id"] = state.last_run_id
        report["observation_status"] = step.observation.status
        report["projection_excerpt"] = harness._first_projection_excerpt(step.observation.projection)
    finally:
        target_path.write_text(original_text, encoding="utf-8")
    (args.out_dir / "probe_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "eval_run_ref": report.get("eval_run_ref")}, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
