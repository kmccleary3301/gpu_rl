from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class Event(ContractModel):
    event_id: str
    run_id: str
    seq: int = Field(ge=0)
    ts: datetime
    scope: str
    kind: Literal["started", "completed", "failed", "info"]
    payload: dict[str, Any] = Field(default_factory=dict)
