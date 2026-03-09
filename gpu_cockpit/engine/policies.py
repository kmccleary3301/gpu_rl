from __future__ import annotations

from gpu_cockpit.contracts.common import BudgetSpec


POLICY_PACKS: dict[str, BudgetSpec] = {
    "conservative": BudgetSpec(
        wall_seconds=120,
        compile_attempts=1,
        bench_runs=1,
        profile_runs=0,
        artifact_mb=128,
    ),
    "balanced": BudgetSpec(
        wall_seconds=300,
        compile_attempts=3,
        bench_runs=3,
        profile_runs=1,
        artifact_mb=512,
    ),
    "exploratory": BudgetSpec(
        wall_seconds=900,
        compile_attempts=6,
        bench_runs=8,
        profile_runs=3,
        artifact_mb=2048,
    ),
}


def resolve_policy_pack(name: str) -> BudgetSpec:
    try:
        return POLICY_PACKS[name].model_copy(deep=True)
    except KeyError as exc:
        raise KeyError(f"Unknown policy pack: {name}") from exc
