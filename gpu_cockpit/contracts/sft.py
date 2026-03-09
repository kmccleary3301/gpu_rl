from __future__ import annotations

from datetime import datetime

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class SFTExample(ContractModel):
    example_id: str
    created_at: datetime
    split: str
    task_id: str
    prompt_family: str
    prompt: str
    response: str
    source_episode_ref: str
    metadata: dict[str, object] = Field(default_factory=dict)


class SFTDatasetManifest(ContractModel):
    dataset_id: str
    created_at: datetime
    split: str
    example_count: int = Field(ge=0)
    example_refs: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
