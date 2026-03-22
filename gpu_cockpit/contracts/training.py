from __future__ import annotations

from datetime import datetime

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class DatasetRef(ContractModel):
    dataset_kind: str
    path: str
    split: str | None = None
    required: bool = True
    notes: list[str] = Field(default_factory=list)


class SFTTrainingConfig(ContractModel):
    config_id: str
    model_id: str
    tokenizer_id: str | None = None
    adapter_mode: str = "lora"
    dataset_refs: list[DatasetRef] = Field(default_factory=list)
    eval_dataset_refs: list[DatasetRef] = Field(default_factory=list)
    context_length: int = 8192
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 1
    max_steps: int | None = None
    output_dir: str
    notes: list[str] = Field(default_factory=list)


class RLRolloutConfig(ContractModel):
    config_id: str
    policy_id: str = "scripted_reference_v1"
    task_refs: list[str] = Field(default_factory=list)
    step_budget: int = 10
    determinism_runs: int = 2
    workflow: str = "auto"
    executor: str = "local_host"
    reward_weights: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class RolloutTaskResult(ContractModel):
    task_id: str
    terminal_state: str
    final_reward: float
    training_example_kind: str = "unusable"
    episode_governance_kind: str = "unusable"
    patch_bearing: bool = False
    step_count: int = 0
    episode_ref: str | None = None


class RolloutEvaluationReport(ContractModel):
    report_id: str
    created_at: datetime
    config_id: str
    policy_id: str
    task_count: int = 0
    success_count: int = 0
    patch_bearing_count: int = 0
    avg_final_reward: float = 0.0
    results: list[RolloutTaskResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
