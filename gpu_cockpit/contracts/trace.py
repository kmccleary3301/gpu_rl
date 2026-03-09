from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class SystemTraceSummary(ContractModel):
    backend: str
    command: list[str] = Field(default_factory=list)
    trace_enabled: bool
    exit_code: int
    duration_ms: int = Field(ge=0)
    report_path: str | None = None
    sqlite_path: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
