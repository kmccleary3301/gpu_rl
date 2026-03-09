from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class BenchmarkCaseSpec(ContractModel):
    case_id: str
    adapter: str
    task_ref: str
    description: str
    source_benchmark: str | None = None
    source_case_ref: str | None = None
    source_case_version: str | None = None
    operator_family: str | None = None
    difficulty: str | None = None
    allowed_backends: list[str] = Field(default_factory=list)
    feature_requirements: list[str] = Field(default_factory=list)
    default_command: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
