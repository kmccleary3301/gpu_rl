from __future__ import annotations

from datetime import datetime

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class TrajectoryAction(ContractModel):
    action_type: str
    step_kind: str | None = None
    command: list[str] = Field(default_factory=list)
    target_run_id: str | None = None
    target_task_id: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class TrajectoryObservation(ContractModel):
    observation_type: str
    run_id: str | None = None
    task_id: str | None = None
    status: str | None = None
    backend: str | None = None
    vendor: str | None = None
    summary_ref: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    projection: dict[str, object] = Field(default_factory=dict)


class TrajectoryStep(ContractModel):
    step_index: int = Field(ge=0)
    action: TrajectoryAction
    observation: TrajectoryObservation
    reward_components: dict[str, float] = Field(default_factory=dict)
    reward_total: float
    terminal: bool = False
    terminal_state: str | None = None


class TrajectoryEpisode(ContractModel):
    episode_id: str
    created_at: datetime
    policy_id: str
    task_id: str
    task_verb: str | None = None
    operator_family: str | None = None
    source_run_id: str
    source_run_ref: str
    episode_kind: str = "single_run_projection"
    steps: list[TrajectoryStep] = Field(default_factory=list)
    final_reward: float
    terminal_state: str
    environment_hash: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class TrajectoryDatasetManifest(ContractModel):
    dataset_id: str
    created_at: datetime
    policy_id: str
    split: str
    format_version: str = "1.1.0"
    episode_count: int = Field(ge=0)
    successful_episode_count: int = Field(ge=0)
    failed_episode_count: int = Field(ge=0)
    episode_refs: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    verb_counts: dict[str, int] = Field(default_factory=dict)
    operator_family_counts: dict[str, int] = Field(default_factory=dict)
    terminal_state_counts: dict[str, int] = Field(default_factory=dict)
    readiness_counts: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)
