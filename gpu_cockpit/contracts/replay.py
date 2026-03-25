from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class ReplayPack(ContractModel):
    run_id: str
    replay_version: str
    image_digest: str | None = None
    git_snapshot: str | None = None
    seed_pack: dict[str, int] = Field(default_factory=dict)
    hardware_fingerprint_ref: str | None = None
    input_snapshot_ref: str | None = None
    task_ref: str
    commands_ref: str | None = None
    candidate_id: str | None = None
    parent_candidate_id: str | None = None
    source_candidate_id: str | None = None
    source_run_ref: str | None = None
    patch_ref: str | None = None
    diff_ref: str | None = None
    transition_ref: str | None = None
    operation_ref: str | None = None
    candidate_role: str | None = None
    candidate_role_group: str | None = None
    candidate_status: str | None = None
    candidate_tree_depth: int | None = None
    candidate_origin_kind: str | None = None
    candidate_operation_kind: str | None = None
    transition_kind: str | None = None
    best_known_candidate_id: str | None = None
    best_known_candidate_reason: str | None = None
    supersede_reason: str | None = None
    branch_state: str | None = None
    endgame_recommendation: str | None = None
    legal_next_actions: list[str] = Field(default_factory=list)
    dominated_candidate_ids: list[str] = Field(default_factory=list)
    active_candidate_ids: list[str] = Field(default_factory=list)
    archived_candidate_ids: list[str] = Field(default_factory=list)
    sibling_candidate_refs: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
