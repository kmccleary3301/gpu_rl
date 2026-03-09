from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel
from gpu_cockpit.contracts.common import BudgetSpec, SeedPack
from gpu_cockpit.versions import PACKAGE_VERSION


class RunSpec(ContractModel):
    run_id: str
    parent_run_id: str | None = None
    created_at: datetime
    engine_version: str = PACKAGE_VERSION
    task_ref: str
    mode: Literal["human", "agent", "replay", "eval"]
    target_backend: Literal["triton", "cuda", "cute", "cutile", "hip"]
    target_vendor: Literal["nvidia", "amd"]
    executor: str
    image_digest: str | None = None
    tool_versions: dict[str, str] = Field(default_factory=dict)
    policy_pack: str
    budgets: BudgetSpec
    seed_pack: SeedPack
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
