from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class BudgetSpec(ContractModel):
    wall_seconds: int = Field(gt=0)
    compile_attempts: int = Field(ge=0)
    bench_runs: int = Field(ge=0)
    profile_runs: int = Field(ge=0)
    artifact_mb: int = Field(gt=0)


class SeedPack(ContractModel):
    global_seed: int
    input_seed: int


class ArtifactRef(ContractModel):
    artifact_id: str
    kind: str
    path: str


class ToolVersionSet(ContractModel):
    versions: dict[str, str] = Field(default_factory=dict)


class MetricValue(ContractModel):
    name: str
    value: int | float | str | bool
    unit: str | None = None


class TimestampedRef(ContractModel):
    ref: str
    created_at: datetime


StatusLiteral = Literal["ok", "error", "partial"]
