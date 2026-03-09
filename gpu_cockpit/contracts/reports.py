from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel
from gpu_cockpit.contracts.bottleneck import BottleneckCard, BottleneckEvidence
from gpu_cockpit.contracts.build import BuildRecord
from gpu_cockpit.contracts.correctness import CorrectnessReport
from gpu_cockpit.contracts.determinism import DeterminismAttempt, DeterminismReport
from gpu_cockpit.contracts.perf import PerfReport
from gpu_cockpit.contracts.profile import ProfileReport
from gpu_cockpit.contracts.replay import ReplayPack


class EvalEnvelope(ContractModel):
    compile_gate: str
    correctness_gate: str
    anti_hack_gate: str
    determinism_gate: str
    perf_gate: str
    reward_components: dict[str, float] = Field(default_factory=dict)
    final_score: float
