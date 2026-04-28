from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class BaselineSpec(ContractModel):
    baseline_id: str
    baseline_kind: str | None = None
    command: list[str] = Field(default_factory=list)
    benchmark_args: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
