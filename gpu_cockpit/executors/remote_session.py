from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from gpu_cockpit.executors.base import CommandResult


class RemoteWorkspaceSession(ABC):
    @property
    @abstractmethod
    def session_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        raise NotImplementedError

    @abstractmethod
    def put_file(self, local_path: Path, remote_path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_file(self, remote_path: Path, local_path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def sync_tree(
        self,
        local_root: Path,
        remote_root: Path,
        *,
        allowlist_roots: list[str] | None = None,
        exclude_globs: list[str] | None = None,
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def terminate(self) -> None:
        raise NotImplementedError
