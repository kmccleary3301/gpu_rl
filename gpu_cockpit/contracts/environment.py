from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class AgentActionSpec(ContractModel):
    action_name: str
    description: str
    requires_task: bool = False
    requires_command: bool = False
    produces_run_bundle: bool = False
    cost_units: float = 0.0
    observation_focus: str | None = None
    recommended_verbs: list[str] = Field(default_factory=list)


class AgentEnvironmentState(ContractModel):
    episode_id: str
    policy_id: str
    task_id: str
    step_budget_total: int = Field(ge=1)
    step_budget_remaining: int = Field(ge=0)
    steps_taken: int = Field(ge=0, default=0)
    last_run_id: str | None = None
    last_run_ref: str | None = None
    current_candidate_id: str | None = None
    current_candidate_parent_id: str | None = None
    current_candidate_run_ref: str | None = None
    current_candidate_status: str | None = None
    current_candidate_attempt_index: int | None = None
    best_known_candidate_id: str | None = None
    best_known_candidate_parent_id: str | None = None
    best_known_candidate_run_ref: str | None = None
    best_known_candidate_reason: str | None = None
    candidate_history: list[str] = Field(default_factory=list)
    candidate_run_history: list[str] = Field(default_factory=list)
    candidate_lineage_events: list[dict[str, object]] = Field(default_factory=list)
    comparison_anchor_run_ref: str | None = None
    comparison_anchor_label: str | None = None
    run_history: list[str] = Field(default_factory=list)
    done: bool = False
    terminal_state: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
