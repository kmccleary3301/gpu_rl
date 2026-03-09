from __future__ import annotations

from pathlib import Path
from typing import Any

from gpu_cockpit.engine.inspector import load_run_summary


def list_runs(
    root: Path,
    *,
    task_id: str | None = None,
    backend: str | None = None,
    vendor: str | None = None,
    status: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return []

    rows: list[dict[str, Any]] = []
    for run_dir in sorted((path for path in runs_dir.iterdir() if path.is_dir()), key=lambda path: path.name, reverse=True):
        try:
            summary = load_run_summary(root, str(run_dir))
        except FileNotFoundError:
            continue
        if task_id and summary.task_id != task_id:
            continue
        if backend and summary.backend != backend:
            continue
        if vendor and summary.vendor != vendor:
            continue
        if status and summary.status != status:
            continue
        rows.append(
            {
                "run_id": summary.run_id,
                "task_id": summary.task_id,
                "status": summary.status,
                "backend": summary.backend,
                "vendor": summary.vendor,
                "trace_enabled": summary.trace_enabled,
                "exit_code": summary.exit_code,
                "duration_ms": summary.duration_ms,
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows
