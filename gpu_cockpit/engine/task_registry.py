from __future__ import annotations

import json
from pathlib import Path
import subprocess

from pydantic import ValidationError

from gpu_cockpit.contracts import TaskSpec


class TaskRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.tasks_dir = root / "workloads" / "tasks"

    def iter_task_paths(self) -> list[Path]:
        if not self.tasks_dir.exists():
            return []
        paths = sorted(self.tasks_dir.rglob("*.json"))
        try:
            toplevel = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.root,
                text=True,
                capture_output=True,
                check=False,
            )
            if toplevel.returncode != 0 or Path(toplevel.stdout.strip()).resolve() != self.root.resolve():
                return paths
            ignored = subprocess.run(
                ["git", "check-ignore", "--stdin"],
                cwd=self.root,
                input="\n".join(str(path.relative_to(self.root)) for path in paths),
                text=True,
                capture_output=True,
                check=False,
            )
        except (OSError, ValueError):
            return paths
        ignored_paths = set(ignored.stdout.splitlines()) if ignored.returncode in {0, 1} else set()
        return [path for path in paths if str(path.relative_to(self.root)) not in ignored_paths]

    def load_all(self) -> list[TaskSpec]:
        tasks: list[TaskSpec] = []
        for path in self.iter_task_paths():
            try:
                tasks.append(self.load_from_path(path))
            except ValidationError:
                continue
        return tasks

    def load_from_path(self, path: Path) -> TaskSpec:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return TaskSpec.model_validate(payload)

    def get(self, task_ref: str) -> TaskSpec:
        candidate = Path(task_ref)
        if candidate.exists():
            return self.load_from_path(candidate)

        for path in self.iter_task_paths():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("task_id") != task_ref:
                continue
            return TaskSpec.model_validate(payload)
        raise FileNotFoundError(f"Task not found: {task_ref}")
