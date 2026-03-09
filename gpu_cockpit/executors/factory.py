from __future__ import annotations

import os
from pathlib import Path

from gpu_cockpit.executors.base import CommandExecutor
from gpu_cockpit.executors.local_docker import LocalDockerExecutor
from gpu_cockpit.executors.local_host import LocalHostToolExecutor


def make_executor(name: str, workspace_root: Path) -> CommandExecutor:
    if name == "local_host":
        return LocalHostToolExecutor()
    if name == "local_docker":
        return LocalDockerExecutor(
            workspace_root=workspace_root,
            image=os.environ.get("GPU_COCKPIT_DOCKER_IMAGE", "python:3.12-slim"),
        )
    raise ValueError(f"Unknown executor: {name}")
