from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def _emit_failure(code: str, message: str, **details: object) -> None:
    payload = {"code": code, **details}
    print(f"GPC_FAILURE_JSON:{json.dumps(payload, sort_keys=True)}", file=sys.stderr)
    raise SystemExit(message)


def _load_payload() -> dict[str, object]:
    stdout_path = Path(os.environ["GPC_STDOUT_PATH"])
    lines = [line for line in stdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit("No stdout payload found")
    return json.loads(lines[-1])


def main() -> None:
    payload = _load_payload()
    expected = [8.0, 3.0, 6.0]
    if payload.get("hidden_row_sum") != expected:
        _emit_failure(
            "hidden_row_sum_mismatch",
            f"Unexpected hidden row sums: {payload.get('hidden_row_sum')} != {expected}",
            section="hidden_tests",
            expected=expected,
            observed=payload.get("hidden_row_sum"),
        )
    summary = payload.get("optimization_summary")
    if not isinstance(summary, dict):
        _emit_failure(
            "missing_optimization_summary",
            "Missing optimization_summary",
            section="hidden_tests",
            expected_any_of=["optimization_summary"],
        )
    if summary.get("strategy_change") != "author_semioopen_triton_row_sum_kernel":
        _emit_failure(
            "unexpected_optimization_strategy_change",
            f"Unexpected optimization strategy change: {summary.get('strategy_change')}",
            section="hidden_tests",
            observed=summary.get("strategy_change"),
        )
    if summary.get("kernel_ref") != "workloads/reference/triton_row_sum_semioopen_kernel.py":
        _emit_failure(
            "unexpected_kernel_ref",
            f"Unexpected kernel ref: {summary.get('kernel_ref')}",
            section="hidden_tests",
            observed=summary.get("kernel_ref"),
        )


if __name__ == "__main__":
    main()
