from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from gpu_cockpit.backends.amd.rocprof import profile_kernel_amd, trace_system_amd
from gpu_cockpit.backends.nvidia.disassembly import emit_disassembly_nvidia
from gpu_cockpit.backends.nvidia.ncu import profile_kernel_nvidia
from gpu_cockpit.backends.nvidia.nsys import trace_system_nvidia
from gpu_cockpit.backends.nvidia.sanitizer import sanitize_nvidia
from gpu_cockpit.backends.triton.build import compile_triton_build_spec
from gpu_cockpit.contracts import BuildRecord, DoctorReport, ProfileReport, RunSpec, RunSummary, SanitizerReport, SystemTraceSummary, TaskSpec
from gpu_cockpit.contracts.common import SeedPack
from gpu_cockpit.executors import make_executor
from gpu_cockpit.engine.bottlenecks import build_bottleneck_card
from gpu_cockpit.engine.command_runner import run_command
from gpu_cockpit.engine.doctor import collect_doctor_report
from gpu_cockpit.engine.policies import resolve_policy_pack
from gpu_cockpit.engine.replay import write_replay_pack
from gpu_cockpit.engine.run_bundle import RunBundleWriter
from gpu_cockpit.engine.task_registry import TaskRegistry


def build_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}_{uuid4().hex[:6]}"


def build_run_spec(
    task: TaskSpec,
    backend: str,
    vendor: str,
    executor: str,
    policy_pack: str,
    tool_versions: dict[str, str] | None = None,
) -> RunSpec:
    return RunSpec(
        run_id=build_run_id(),
        created_at=datetime.now(tz=UTC),
        task_ref=task.task_id,
        mode="human",
        target_backend=backend,  # type: ignore[arg-type]
        target_vendor=vendor,  # type: ignore[arg-type]
        executor=executor,
        policy_pack=policy_pack,
        tool_versions=tool_versions or {},
        budgets=resolve_policy_pack(policy_pack),
        seed_pack=SeedPack(global_seed=1234, input_seed=5678),
        tags=["manual-run"],
    )


def write_task_artifacts(writer: RunBundleWriter, task: TaskSpec, doctor_report: DoctorReport) -> None:
    public_task_spec = task.model_copy(deep=True)
    public_task_spec.hidden_tests_ref = None
    writer.write_artifact(
        relative_path="prompt/task_spec.json",
        kind="task_spec",
        content=json.dumps(public_task_spec.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["task", "spec"],
    )
    writer.write_artifact(
        relative_path="meta/task_spec_full.json",
        kind="task_spec_full",
        content=json.dumps(task.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["task", "spec", "internal"],
    )
    writer.write_artifact(
        relative_path="meta/doctor_report.json",
        kind="doctor_report",
        content=json.dumps(doctor_report.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["doctor", "environment"],
    )
    if doctor_report.hardware_fingerprints:
        writer.write_artifact(
            relative_path="meta/hardware_fingerprint.json",
            kind="hardware_fingerprint",
            content=json.dumps(doctor_report.hardware_fingerprints[0].model_dump(mode="json"), indent=2) + "\n",
            mime="application/json",
            semantic_tags=["doctor", "hardware"],
        )


def write_run_summary(writer: RunBundleWriter, summary: RunSummary) -> None:
    warning_lines = [f"- {warning}" for warning in summary.warnings] or ["- none"]
    writer.write_artifact(
        relative_path="summary.json",
        kind="run_summary",
        content=json.dumps(summary.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["summary", "run"],
    )
    markdown = "\n".join(
        [
            f"# Run {summary.run_id}",
            "",
            f"- task: `{summary.task_id}`",
            f"- status: `{summary.status}`",
            f"- trace_enabled: `{summary.trace_enabled}`",
            f"- backend: `{summary.backend}`",
            f"- vendor: `{summary.vendor}`",
            f"- parent_run_id: `{summary.parent_run_id}`",
            f"- candidate_id: `{summary.candidate_id}`",
            f"- parent_candidate_id: `{summary.parent_candidate_id}`",
            f"- patch_present: `{summary.patch_present}`",
            f"- patch_kind: `{summary.patch_kind}`",
            f"- transition_kind: `{summary.transition_kind}`",
            f"- candidate_role: `{summary.candidate_role}`",
            f"- candidate_status: `{summary.candidate_status}`",
            f"- candidate_origin_kind: `{summary.candidate_origin_kind}`",
            f"- candidate_operation_kind: `{summary.candidate_operation_kind}`",
            f"- candidate_diff_ref: `{summary.candidate_diff_ref}`",
            f"- exit_code: `{summary.exit_code}`",
            f"- duration_ms: `{summary.duration_ms}`",
            "",
            "## Key Artifacts",
            *[f"- `{path}`" for path in summary.key_artifacts],
            "",
            "## Warnings",
            *warning_lines,
        ]
    )
    writer.write_artifact(
        relative_path="summary.md",
        kind="run_summary_markdown",
        content=markdown + "\n",
        mime="text/markdown",
        semantic_tags=["summary", "run", "markdown"],
    )


def run_task(
    root: Path,
    task_ref: str,
    command: list[str],
    trace_system: bool = False,
    profile_kernel: bool = False,
    profile_pack: str = "quick",
    sanitize: bool = False,
    sanitize_tool: str = "memcheck",
    emit_disassembly: bool = False,
    triton_build_spec: str | None = None,
    source_file: str | None = None,
    binary_file: str | None = None,
    ptx_file: str | None = None,
    sass_file: str | None = None,
    backend: str | None = None,
    vendor: str | None = None,
    executor: str = "local_host",
    policy_pack: str = "balanced",
    lineage: dict[str, object] | None = None,
) -> Path:
    registry = TaskRegistry(root)
    task = registry.get(task_ref)
    selected_backend = backend or (task.allowed_backends[0] if task.allowed_backends else "triton")
    selected_vendor = vendor or "nvidia"
    doctor_report = collect_doctor_report()
    run_spec = build_run_spec(
        task=task,
        backend=selected_backend,
        vendor=selected_vendor,
        executor=executor,
        policy_pack=policy_pack,
        tool_versions={tool.name: tool.version for tool in doctor_report.available_tools if tool.available and tool.version},
    )
    lineage = lineage or {}
    if lineage.get("parent_run_id") is not None:
        run_spec.parent_run_id = str(lineage["parent_run_id"])
    writer = RunBundleWriter(root=root)
    run_dir = writer.initialize(run_spec)
    write_task_artifacts(writer, task, doctor_report)
    writer.append_event(scope="task", kind="info", payload={"task_id": task.task_id, "operator_family": task.operator_family})
    command_executor = make_executor(executor, root)

    command_summary: SystemTraceSummary | None = None
    build_record: BuildRecord | None = None
    profile_summary: ProfileReport | None = None
    sanitizer_summary: SanitizerReport | None = None
    extra_warnings: list[str] = []
    if trace_system and not command:
        raise RuntimeError("System tracing requires a command.")
    if profile_kernel and not command:
        raise RuntimeError("Kernel profiling requires a command.")
    if sanitize and not command:
        raise RuntimeError("Sanitizer workflows require a command.")

    if command:
        if trace_system and selected_vendor == "nvidia":
            if executor != "local_host":
                raise RuntimeError("System tracing currently requires executor=local_host.")
            command_summary = trace_system_nvidia(writer, command, executor=command_executor)
        elif trace_system and selected_vendor == "amd":
            if executor != "local_host":
                raise RuntimeError("System tracing currently requires executor=local_host.")
            command_summary = trace_system_amd(writer, command, executor=command_executor)
        else:
            command_summary = run_command(writer, command, executor=command_executor)
        if profile_kernel:
            if selected_vendor != "nvidia":
                if selected_vendor != "amd":
                    raise RuntimeError("Kernel profiling currently supports vendor=nvidia or vendor=amd only.")
            if executor != "local_host":
                raise RuntimeError("Kernel profiling currently requires executor=local_host.")
            try:
                if selected_vendor == "amd":
                    profile_summary = profile_kernel_amd(writer, command, profile_pack=profile_pack, executor=command_executor)
                else:
                    profile_summary = profile_kernel_nvidia(writer, command, profile_pack=profile_pack, executor=command_executor)
            except RuntimeError as exc:
                extra_warnings.append(str(exc))
                writer.append_event(
                    scope="tool.profile_kernel_amd" if selected_vendor == "amd" else "tool.profile_kernel_nvidia",
                    kind="failed",
                    payload={"error": str(exc), "profile_pack": profile_pack},
                )
        if sanitize:
            if selected_vendor != "nvidia":
                raise RuntimeError("Sanitizer workflows currently support vendor=nvidia only.")
            if executor != "local_host":
                raise RuntimeError("Sanitizer workflows currently require executor=local_host.")
            try:
                sanitizer_summary = sanitize_nvidia(writer, command, tool=sanitize_tool, executor=command_executor)
            except RuntimeError as exc:
                extra_warnings.append(str(exc))
                writer.append_event(
                    scope="tool.sanitize_nvidia",
                    kind="failed",
                    payload={"error": str(exc), "tool": sanitize_tool},
                )
        if emit_disassembly:
            try:
                if triton_build_spec is not None:
                    if executor != "local_host":
                        raise RuntimeError("Triton build-spec compilation currently requires executor=local_host.")
                    build_record = compile_triton_build_spec(writer, root, triton_build_spec)
                else:
                    if selected_vendor != "nvidia":
                        raise RuntimeError("Disassembly extraction currently supports vendor=nvidia only.")
                    if executor != "local_host":
                        raise RuntimeError("Disassembly extraction currently requires executor=local_host.")
                    build_record = emit_disassembly_nvidia(
                        writer,
                        command,
                        source_file=source_file,
                        binary_file=binary_file,
                        ptx_file=ptx_file,
                        sass_file=sass_file,
                        executor=command_executor,
                    )
            except RuntimeError as exc:
                extra_warnings.append(str(exc))
                writer.append_event(
                    scope="tool.emit_disassembly_nvidia",
                    kind="failed",
                    payload={"error": str(exc)},
                )
        if profile_summary is not None:
            build_bottleneck_card(writer, profile_summary, sanitizer_summary)
    elif not emit_disassembly:
        writer.append_event(scope="task", kind="info", payload={"message": "No command was provided; metadata-only run"})

    if emit_disassembly and build_record is None:
        try:
            if triton_build_spec is not None:
                if executor != "local_host":
                    raise RuntimeError("Triton build-spec compilation currently requires executor=local_host.")
                build_record = compile_triton_build_spec(writer, root, triton_build_spec)
            else:
                if selected_vendor != "nvidia":
                    raise RuntimeError("Disassembly extraction currently supports vendor=nvidia only.")
                if executor != "local_host":
                    raise RuntimeError("Disassembly extraction currently requires executor=local_host.")
                build_record = emit_disassembly_nvidia(
                    writer,
                    command,
                    source_file=source_file,
                    binary_file=binary_file,
                    ptx_file=ptx_file,
                    sass_file=sass_file,
                    executor=command_executor,
                )
        except RuntimeError as exc:
            extra_warnings.append(str(exc))
            writer.append_event(
                scope="tool.emit_disassembly_nvidia",
                kind="failed",
                payload={"error": str(exc)},
            )

    key_artifacts = [
        "manifest.json",
        "events.jsonl",
        "prompt/task_spec.json",
        "meta/task_spec_full.json",
        "meta/doctor_report.json",
    ]
    if doctor_report.hardware_fingerprints:
        key_artifacts.append("meta/hardware_fingerprint.json")
    if command_summary:
        key_artifacts.extend(
            [
                command_summary.report_path,
                command_summary.sqlite_path,
                command_summary.stdout_path,
                command_summary.stderr_path,
            ]
        )
    if profile_summary is not None:
        key_artifacts.extend(
            [
                profile_summary.raw_profile_ref,
                profile_summary.csv_profile_ref,
                profile_summary.stdout_path,
                profile_summary.stderr_path,
                "profiles/kernel/summary.json",
                "profiles/kernel/summary.md",
                "bottlenecks/primary.json",
            ]
        )
    if build_record is not None:
        key_artifacts.extend(
            [
                "build/build_record.json",
                "build/tri_view.json",
                "build/tri_view.md",
                "build/source_ptx_sass_map.json",
                "build/source_map_summary.json" if (writer.run_dir / "build" / "source_map_summary.json").exists() else None,
                "build/triton_compile_metadata.json" if (writer.run_dir / "build" / "triton_compile_metadata.json").exists() else None,
                "build/ttir.mlir" if (writer.run_dir / "build" / "ttir.mlir").exists() else None,
                "build/ttgir.mlir" if (writer.run_dir / "build" / "ttgir.mlir").exists() else None,
                "build/llir.ll" if (writer.run_dir / "build" / "llir.ll").exists() else None,
                build_record.stdout_ref,
                build_record.stderr_ref,
                build_record.ptx_ref,
                build_record.sass_ref,
                "build/source.txt" if (writer.run_dir / "build" / "source.txt").exists() else None,
            ]
        )
    if sanitizer_summary is not None:
        key_artifacts.extend(
            [
                sanitizer_summary.raw_log_ref,
                sanitizer_summary.stdout_path,
                sanitizer_summary.stderr_path,
                f"sanitize/{sanitize_tool}_summary.json",
                f"sanitize/{sanitize_tool}_summary.md",
            ]
        )
    key_artifacts = [path for path in key_artifacts if path]

    run_summary = RunSummary(
        run_id=run_spec.run_id,
        task_id=task.task_id,
        status="ok",
        trace_enabled=bool(command_summary and command_summary.trace_enabled),
        backend=selected_backend,
        vendor=selected_vendor,
        parent_run_id=run_spec.parent_run_id,
        candidate_id=str(lineage["candidate_id"]) if lineage.get("candidate_id") is not None else None,
        parent_candidate_id=str(lineage["parent_candidate_id"]) if lineage.get("parent_candidate_id") is not None else None,
        patch_present=bool(lineage.get("patch_present", False)),
        patch_kind=str(lineage["patch_kind"]) if lineage.get("patch_kind") is not None else None,
        transition_kind=str(lineage["transition_kind"]) if lineage.get("transition_kind") is not None else None,
        candidate_role=str(lineage["candidate_role"]) if lineage.get("candidate_role") is not None else None,
        candidate_status=str(lineage["candidate_status"]) if lineage.get("candidate_status") is not None else None,
        candidate_origin_kind=str(lineage["candidate_origin_kind"]) if lineage.get("candidate_origin_kind") is not None else None,
        candidate_operation_kind=str(lineage["candidate_operation_kind"]) if lineage.get("candidate_operation_kind") is not None else None,
        candidate_diff_ref=str(lineage["diff_ref"]) if lineage.get("diff_ref") is not None else None,
        exit_code=command_summary.exit_code if command_summary else None,
        duration_ms=command_summary.duration_ms if command_summary else None,
        key_artifacts=key_artifacts,
        warnings=(command_summary.warnings if command_summary else [])
        + ([f"disassembly status: {build_record.status}"] if build_record is not None and build_record.status != "ok" else [])
        + (profile_summary.warnings if profile_summary else [])
        + (sanitizer_summary.warnings if sanitizer_summary else [])
        + extra_warnings,
    )
    write_run_summary(writer, run_summary)
    write_replay_pack(
        writer=writer,
        task=task,
        doctor_report=doctor_report,
        command=command,
        required_artifacts=run_summary.key_artifacts + ["summary.json", "summary.md"],
        environment={
            "executor": executor,
            "policy_pack": policy_pack,
            "target_backend": selected_backend,
            "target_vendor": selected_vendor,
            "trace_system": trace_system,
            "profile_kernel": profile_kernel,
            "profile_pack": profile_pack if profile_kernel else None,
            "sanitize": sanitize,
            "sanitize_tool": sanitize_tool if sanitize else None,
            "emit_disassembly": emit_disassembly,
            "triton_build_spec": triton_build_spec,
            "source_file": source_file,
            "binary_file": binary_file,
            "ptx_file": ptx_file,
            "sass_file": sass_file,
            "workspace_root": str(root.resolve()),
            "cwd": str(Path.cwd().resolve()),
            "python_executable": doctor_report.python_executable,
            "python_version": doctor_report.python_version,
            "selected_env": {
                key: value
                for key, value in {
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
                    "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
                }.items()
                if value is not None
            },
        },
        lineage={
            "candidate_id": lineage.get("candidate_id"),
            "parent_candidate_id": lineage.get("parent_candidate_id"),
            "source_run_ref": lineage.get("source_run_ref"),
            "patch_ref": lineage.get("patch_ref"),
            "diff_ref": lineage.get("diff_ref"),
            "transition_ref": lineage.get("transition_ref"),
            "operation_ref": lineage.get("operation_ref"),
            "candidate_role": lineage.get("candidate_role"),
            "candidate_status": lineage.get("candidate_status"),
            "candidate_origin_kind": lineage.get("candidate_origin_kind"),
            "candidate_operation_kind": lineage.get("candidate_operation_kind"),
            "transition_kind": lineage.get("transition_kind"),
        },
    )
    writer.append_event(scope="run", kind="completed", payload={"status": "ok"})
    return run_dir
