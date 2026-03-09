from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from gpu_cockpit.executors.base import CommandExecutor, CommandResult


class LocalHostToolExecutor(CommandExecutor):
    def run(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        started = time.monotonic()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=run_env,
            cwd=str(cwd) if cwd is not None else None,
            timeout=timeout,
        )
        return CommandResult(
            command=list(command),
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
