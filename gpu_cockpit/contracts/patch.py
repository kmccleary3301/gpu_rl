from __future__ import annotations

from typing import Literal

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


PatchKind = Literal["bug_fix", "perf_transform", "refactor", "instrumentation", "no_op", "rejected"]
PatchContentMode = Literal["replace", "create", "delete", "unified_diff"]
TransitionKind = Literal["inspect_only", "patch_applied", "benchmarked", "evaluated", "repaired", "reformulated", "reverted"]


class PatchRequest(ContractModel):
    target_file: str
    patch_kind: PatchKind
    intent: str
    expected_effect: str | None = None
    content_mode: PatchContentMode = "replace"
    patch_text: str
    metadata: dict[str, object] = Field(default_factory=dict)


class AppliedPatch(ContractModel):
    patch_id: str
    target_file: str
    patch_kind: PatchKind
    intent: str
    expected_effect: str | None = None
    content_mode: PatchContentMode = "replace"
    accepted: bool = True
    patch_hash: str
    changed_line_count: int = Field(ge=0, default=0)
    before_ref: str | None = None
    after_ref: str | None = None
    diff_ref: str | None = None
    error_message: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class CandidateState(ContractModel):
    candidate_id: str
    task_id: str
    workspace_root: str
    source_run_ref: str | None = None
    source_run_id: str | None = None
    parent_candidate_id: str | None = None
    parent_run_ref: str | None = None
    patch_id: str | None = None
    patch_hash: str | None = None
    patch_kind: PatchKind | None = None
    changed_files: list[str] = Field(default_factory=list)
    candidate_role: str | None = None
    lineage_depth: int = Field(ge=0, default=0)
    metadata: dict[str, object] = Field(default_factory=dict)


class CandidateTransition(ContractModel):
    transition_id: str
    transition_kind: TransitionKind
    task_id: str
    input_candidate_id: str | None = None
    output_candidate_id: str | None = None
    source_run_ref: str | None = None
    target_run_ref: str | None = None
    patch_id: str | None = None
    patch_hash: str | None = None
    patch_kind: PatchKind | None = None
    changed_files: list[str] = Field(default_factory=list)
    status: str = "ok"
    summary: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
