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
    parent_run_id: str | None = None
    candidate_id: str | None = None
    parent_candidate_id: str | None = None
    patch_present: bool = False
    patch_kind: str | None = None
    transition_kind: str | None = None
    candidate_role: str | None = None
    exit_code: int | None = None
    duration_ms: int | None = None
    key_artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
