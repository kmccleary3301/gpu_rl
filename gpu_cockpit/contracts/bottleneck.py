from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class BottleneckEvidence(ContractModel):
    type: str
    name: str | None = None
    value: int | float | str | bool | None = None
    artifact_ref: str | None = None
    line: int | None = Field(default=None, ge=1)


class BottleneckCard(ContractModel):
    card_id: str
    run_id: str
    subject: str
    primary_bottleneck: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[BottleneckEvidence] = Field(default_factory=list)
    why: str
    next_actions: list[str] = Field(default_factory=list)
