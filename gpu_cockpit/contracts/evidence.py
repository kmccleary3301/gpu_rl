from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class ReadinessDecision(ContractModel):
    eligible: bool
    reasons: list[str] = Field(default_factory=list)


class EvidenceQualityReport(ContractModel):
    run_id: str
    task_id: str | None = None
    required_artifact_count: int = 0
    missing_required_artifact_count: int = 0
    required_artifact_completeness: float
    replay_completeness: float
    build_completeness: float
    profile_completeness: float
    eval_completeness: float
    provenance_completeness: float = 0.0
    overall_score: float
    benchmark_reporting: ReadinessDecision
    sft_collection: ReadinessDecision
    rl_reward_trace: ReadinessDecision
    missing_provenance_fields: list[str] = Field(default_factory=list)
    training_example_kind: str = "unusable"
    notes: list[str] = Field(default_factory=list)
