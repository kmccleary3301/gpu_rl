from __future__ import annotations

import json
from pathlib import Path

from gpu_cockpit.contracts import TaskSpec


class TaskRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.tasks_dir = root / "workloads" / "tasks"

    def iter_task_paths(self) -> list[Path]:
        if not self.tasks_dir.exists():
            return []
        return sorted(self.tasks_dir.rglob("*.json"))

    def load_all(self) -> list[TaskSpec]:
        return [self.load_from_path(path) for path in self.iter_task_paths()]

    def load_from_path(self, path: Path) -> TaskSpec:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return TaskSpec.model_validate(payload)

    def get(self, task_ref: str) -> TaskSpec:
        candidate = Path(task_ref)
        if candidate.exists():
            return self.load_from_path(candidate)

        for path in self.iter_task_paths():
            task = self.load_from_path(path)
            if task.task_id == task_ref:
                return task
        raise FileNotFoundError(f"Task not found: {task_ref}")
