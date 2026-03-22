from __future__ import annotations

import json
import os

from gpu_cockpit.contracts import SystemTraceSummary
from gpu_cockpit.executors import CommandExecutor, LocalHostToolExecutor
from gpu_cockpit.engine.command_utils import local_python_build_env, normalize_python_command
from gpu_cockpit.engine.run_bundle import RunBundleWriter


def run_command(
    writer: RunBundleWriter,
    command: list[str],
    scope: str = "tool.run_command",
    executor: CommandExecutor | None = None,
) -> SystemTraceSummary:
    executor = executor or LocalHostToolExecutor()
    normalized_command = normalize_python_command(command)
    completed = writer.append_event(scope=scope, kind="started", payload={"command": normalized_command})
    run_env = os.environ.copy()
    run_env.update(local_python_build_env(writer.root))
    result = executor.run(normalized_command, cwd=writer.root, env=run_env)
    stdout_artifact = writer.write_artifact(
        relative_path="command/stdout.txt",
        kind="command_stdout",
        content=result.stdout,
        mime="text/plain",
        semantic_tags=["command", "stdout"],
        producer_event_id=completed.event_id,
    )
    stderr_artifact = writer.write_artifact(
        relative_path="command/stderr.txt",
        kind="command_stderr",
        content=result.stderr,
        mime="text/plain",
        semantic_tags=["command", "stderr"],
        producer_event_id=completed.event_id,
    )
    summary = SystemTraceSummary(
        backend="subprocess",
        command=normalized_command,
        trace_enabled=False,
        exit_code=result.exit_code,
        duration_ms=result.duration_ms,
        stdout_path=stdout_artifact.path,
        stderr_path=stderr_artifact.path,
    )
    writer.write_artifact(
        relative_path="command/summary.json",
        kind="command_summary",
        content=json.dumps(summary.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["command", "summary"],
        producer_event_id=completed.event_id,
    )
    writer.append_event(
        scope=scope,
        kind="completed" if result.exit_code == 0 else "failed",
        payload={"exit_code": result.exit_code, "duration_ms": result.duration_ms},
    )
    return summary
