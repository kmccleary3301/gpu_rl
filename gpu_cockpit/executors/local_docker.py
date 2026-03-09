from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from gpu_cockpit.executors.base import CommandExecutor, CommandResult


class LocalDockerExecutor(CommandExecutor):
    def __init__(self, workspace_root: Path, image: str = "python:3.12-slim") -> None:
        self.workspace_root = workspace_root.resolve()
        self.image = image

    def _container_cwd(self, cwd: Path | None) -> str:
        host_cwd = (cwd or self.workspace_root).resolve()
        try:
            relative = host_cwd.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(f"cwd {host_cwd} is outside mounted workspace {self.workspace_root}") from exc
        return str(Path("/workspace") / relative)

    def run(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{self.workspace_root}:/workspace",
            "-w",
            self._container_cwd(cwd),
        ]
        run_env = {}
        if env:
            run_env.update(env)
        for key, value in run_env.items():
            docker_cmd.extend(["-e", f"{key}={value}"])
        docker_cmd.append(self.image)
        docker_cmd.extend(command)

        started = time.monotonic()
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=os.environ.copy(),
        )
        return CommandResult(
            command=list(command),
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
