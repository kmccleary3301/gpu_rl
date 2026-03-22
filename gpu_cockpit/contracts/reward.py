from __future__ import annotations

from typing import Any

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class RewardLedgerEntry(ContractModel):
    step_index: int = Field(ge=0)
    action_type: str
    reward_components: dict[str, float] = Field(default_factory=dict)
    shaping_components: dict[str, float] = Field(default_factory=dict)
    total_delta: float = 0.0
    notes: list[str] = Field(default_factory=list)


class RewardLedger(ContractModel):
    schema_id: str = "optimize_reward_ledger_v1"
    task_id: str | None = None
    task_verb: str | None = None
    task_outcome: str | None = None
    trace_usability: str | None = None
    entries: list[RewardLedgerEntry] = Field(default_factory=list)
    total_reward_components: dict[str, float] = Field(default_factory=dict)
    total_shaping_components: dict[str, float] = Field(default_factory=dict)
    total_reward: float = 0.0


class LearningRewardTrace(ContractModel):
    schema_id: str = "optimize_reward_v1"
    task_id: str | None = None
    task_verb: str | None = None
    terminal_state: str | None = None
    task_outcome: str | None = None
    trace_usability: str | None = None
    task_success: bool = False
    correctness_passed: bool = False
    determinism_passed: bool = False
    anti_hack_passed: bool = False
    perf_gate: str | None = None
    compare_used: bool = False
    patch_bearing: bool = False
    branch_count: int = 0
    revert_count: int = 0
    promote_count: int = 0
    reward_components: dict[str, float] = Field(default_factory=dict)
    shaping_components: dict[str, float] = Field(default_factory=dict)
    excluded_governance_signals: list[str] = Field(default_factory=list)
    total_reward: float = 0.0
    notes: list[str] = Field(default_factory=list)
    reward_ledger: RewardLedger | None = None


class OptimizeTraceSnapshot(ContractModel):
    step_index: int = Field(ge=0)
    action_type: str
    snapshot_kind: str
    run_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class OptimizeTraceSnapshots(ContractModel):
    candidate_snapshots: list[OptimizeTraceSnapshot] = Field(default_factory=list)
    compare_snapshots: list[OptimizeTraceSnapshot] = Field(default_factory=list)
    failure_localization_snapshots: list[OptimizeTraceSnapshot] = Field(default_factory=list)
