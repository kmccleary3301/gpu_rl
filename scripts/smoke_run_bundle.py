from __future__ import annotations

from datetime import UTC, datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts import RunSpec
from gpu_cockpit.contracts.common import BudgetSpec, SeedPack
from gpu_cockpit.engine import RunBundleWriter


def main() -> None:
    run_spec = RunSpec(
        run_id="run_smoke_001",
        created_at=datetime.now(tz=UTC),
        task_ref="task/smoke/diagnose/v1",
        mode="human",
        target_backend="triton",
        target_vendor="nvidia",
        executor="local_docker",
        policy_pack="balanced",
        budgets=BudgetSpec(
            wall_seconds=300,
            compile_attempts=2,
            bench_runs=2,
            profile_runs=1,
            artifact_mb=128,
        ),
        seed_pack=SeedPack(global_seed=1234, input_seed=5678),
        tags=["smoke"],
        notes="Initial smoke bundle",
    )
    writer = RunBundleWriter(root=ROOT)
    run_dir = writer.initialize(run_spec)
    info_event = writer.append_event(
        scope="tool.schema_export",
        kind="completed",
        payload={"status": "ok", "message": "Smoke run initialized"},
    )
    artifact = writer.write_artifact(
        relative_path="summary/hello.txt",
        kind="text_note",
        content="gpu-cockpit smoke artifact\n",
        mime="text/plain",
        semantic_tags=["smoke", "summary"],
        producer_event_id=info_event.event_id,
    )
    writer.append_event(
        scope="artifact",
        kind="completed",
        payload={"artifact_id": artifact.artifact_id, "path": artifact.path},
    )
    writer.append_event(scope="run", kind="completed", payload={"status": "ok"})
    print(run_dir)


if __name__ == "__main__":
    main()
