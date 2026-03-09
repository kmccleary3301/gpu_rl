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
    required_artifacts: list[str] = Field(default_factory=list)
