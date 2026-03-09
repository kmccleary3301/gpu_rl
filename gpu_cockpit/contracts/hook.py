from __future__ import annotations

from gpu_cockpit.contracts.base import ContractModel


class HookExecution(ContractModel):
    name: str
    ref: str
    exit_code: int
    passed: bool
    stdout_path: str | None = None
    stderr_path: str | None = None
