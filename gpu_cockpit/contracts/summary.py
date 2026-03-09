from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class RunSummary(ContractModel):
    run_id: str
    task_id: str
    status: str
    trace_enabled: bool
    backend: str
    vendor: str
    exit_code: int | None = None
    duration_ms: int | None = None
    key_artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
