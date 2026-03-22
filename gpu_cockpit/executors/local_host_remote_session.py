from __future__ import annotations

import shutil
from pathlib import Path

from gpu_cockpit.executors.base import CommandResult
from gpu_cockpit.executors.local_host import LocalHostToolExecutor
from gpu_cockpit.executors.remote_session import RemoteWorkspaceSession


class LocalHostRemoteSession(RemoteWorkspaceSession):
    def __init__(self, *, session_id: str, workspace_root: Path, cwd: Path | None = None) -> None:
        self._session_id = session_id
        self.workspace_root = workspace_root.resolve()
        self.cwd = cwd.resolve() if cwd is not None else self.workspace_root
        self._executor = LocalHostToolExecutor()

    @property
    def session_id(self) -> str:
        return self._session_id

    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        run_cwd = cwd.resolve() if cwd is not None else self.cwd
        return self._executor.run(command, cwd=run_cwd, env=env, timeout=timeout)

    def put_file(self, local_path: Path, remote_path: Path) -> None:
        source = local_path.resolve()
        destination = self.workspace_root / remote_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def get_file(self, remote_path: Path, local_path: Path) -> None:
        source = self.workspace_root / remote_path
        destination = local_path.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def sync_tree(
        self,
        local_root: Path,
        remote_root: Path,
        *,
        allowlist_roots: list[str] | None = None,
        exclude_globs: list[str] | None = None,
    ) -> list[str]:
        copied: list[str] = []
        source_root = local_root.resolve()
        destination_root = self.workspace_root / remote_root
        destination_root.mkdir(parents=True, exist_ok=True)
        excludes = tuple(exclude_globs or [])
        for relative in allowlist_roots or []:
            source = source_root / relative
            if not source.exists():
                continue
            if any(source.match(pattern) for pattern in excludes):
                continue
            destination = destination_root / relative
            if source.is_dir():
                shutil.copytree(
                    source,
                    destination,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(*[pattern for pattern in excludes if "/" not in pattern]),
                )
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            copied.append(str(relative))
        return copied

    def terminate(self) -> None:
        return None
