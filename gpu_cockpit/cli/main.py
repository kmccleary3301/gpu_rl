from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu_cockpit.contracts import RunSummary
from gpu_cockpit.executors import make_executor
from gpu_cockpit.engine.adapter_registry import describe_adapter, get_adapter, list_adapter_cases, list_adapters
from gpu_cockpit.engine.benchmark import run_subprocess_benchmark, run_task_benchmark
from gpu_cockpit.engine.doctor import collect_doctor_report
from gpu_cockpit.engine.evaluator import run_evaluation_hooks
from gpu_cockpit.engine.environment import list_action_space, run_scripted_reference_episode
from gpu_cockpit.engine.indexer import list_runs
from gpu_cockpit.engine.inspector import compare_runs, inspect_run
from gpu_cockpit.engine.knowledge import build_knowledge_index, query_knowledge, retrieve_similar_for_task
from gpu_cockpit.engine.replay import export_proof_bundle, validate_run_bundle, write_replay_pack
from gpu_cockpit.engine.rollout import run_scripted_rollout_suite
from gpu_cockpit.engine.run_bundle import RunBundleWriter
from gpu_cockpit.engine.runner import build_run_id, build_run_spec, write_run_summary, write_task_artifacts
from gpu_cockpit.engine.runner import run_task
from gpu_cockpit.engine.sft import package_trajectory_dataset_as_sft, validate_sft_dataset
from gpu_cockpit.engine.task_registry import TaskRegistry
from gpu_cockpit.engine.training import load_training_config, run_sft_training_job, validate_sft_training_config, write_sft_smoke_report
from gpu_cockpit.engine.trajectory import export_trajectory_dataset, validate_trajectory_dataset, write_episode, write_trajectory_episode
from gpu_cockpit.contracts import RLRolloutConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpc", description="GPU cockpit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect local toolchain and hardware")
    doctor.add_argument("--json-out", type=Path, default=None, help="Optional path to write JSON report")

    task = subparsers.add_parser("task", help="Inspect task registry")
    task_subparsers = task.add_subparsers(dest="task_command", required=True)
    task_subparsers.add_parser("list", help="List registered tasks")

    adapter = subparsers.add_parser("adapter", help="Inspect benchmark adapters")
    adapter_subparsers = adapter.add_subparsers(dest="adapter_command", required=True)
    adapter_subparsers.add_parser("list", help="List registered benchmark adapters")
    adapter_cases = adapter_subparsers.add_parser("cases", help="List benchmark cases for an adapter")
    adapter_cases.add_argument("adapter_name", help="Adapter name")
    adapter_summary = adapter_subparsers.add_parser("summary", help="Summarize coverage for a benchmark adapter")
    adapter_summary.add_argument("adapter_name", help="Adapter name")
    adapter_show = adapter_subparsers.add_parser("show", help="Show a single benchmark case")
    adapter_show.add_argument("adapter_name", help="Adapter name")
    adapter_show.add_argument("case_id", help="Benchmark case id")

    inspect = subparsers.add_parser("inspect", help="Inspect a run bundle")
    inspect.add_argument("run_ref", help="Run id or path")
    inspect.add_argument(
        "--section",
        default="full",
        choices=["full", "summary", "build", "eval", "profile", "replay", "quality"],
        help="Optional focused inspection section",
    )

    runs = subparsers.add_parser("runs", help="Query indexed run summaries")
    runs_subparsers = runs.add_subparsers(dest="runs_command", required=True)
    runs_list = runs_subparsers.add_parser("list", help="List run summaries with optional filters")
    runs_list.add_argument("--task", dest="task_id", default=None, help="Filter by task id")
    runs_list.add_argument("--backend", default=None, help="Filter by backend")
    runs_list.add_argument("--vendor", default=None, help="Filter by vendor")
    runs_list.add_argument("--status", default=None, help="Filter by run status")
    runs_list.add_argument("--limit", type=int, default=20, help="Maximum number of rows to return")

    replay = subparsers.add_parser("replay", help="Validate and inspect replay metadata for a run bundle")
    replay.add_argument("run_ref", help="Run id or path")

    bundle = subparsers.add_parser("bundle", help="Export a proof bundle for a run")
    bundle.add_argument("run_ref", help="Run id or path")
    bundle.add_argument("--out", type=Path, default=None, help="Optional output zip path")
    bundle.add_argument("--full", action="store_true", help="Include all files instead of summary-oriented proof artifacts")

    compare = subparsers.add_parser("compare", help="Compare two run bundles")
    compare.add_argument("lhs_run_ref", help="Left-hand run id or path")
    compare.add_argument("rhs_run_ref", help="Right-hand run id or path")

    build = subparsers.add_parser("build", help="Emit build/disassembly artifacts without requiring a full eval flow")
    build.add_argument("--task", required=True, help="Task id or path to TaskSpec JSON")
    build.add_argument("--triton-build-spec", default=None, help="Optional Triton build spec module path, optionally suffixed with :function")
    build.add_argument("--source-file", default=None, help="Optional source file for tri-view/disassembly")
    build.add_argument("--binary-file", default=None, help="Optional binary file to disassemble")
    build.add_argument("--ptx-file", default=None, help="Optional PTX file to include in tri-view")
    build.add_argument("--sass-file", default=None, help="Optional SASS file to include in tri-view")
    build.add_argument("--backend", default=None, help="Override backend")
    build.add_argument("--vendor", default=None, help="Override vendor")
    build.add_argument("--executor", default="local_host", help="Execution backend")
    build.add_argument("--policy-pack", default="balanced", help="Budget policy pack")
    build.add_argument("cmd", nargs=argparse.REMAINDER, help="Optional command to execute after '--'")

    run = subparsers.add_parser("run", help="Create a run bundle and optionally execute a command")
    run.add_argument("--task", required=True, help="Task id or path to TaskSpec JSON")
    run.add_argument("--trace-system", action="store_true", help="Use NVIDIA Nsight Systems tracing when available")
    run.add_argument("--profile-kernel", action="store_true", help="Run NVIDIA Nsight Compute kernel profiling")
    run.add_argument("--profile-pack", default="quick", help="Kernel profile pack: quick, memory, compute, deep")
    run.add_argument("--sanitize", action="store_true", help="Run NVIDIA Compute Sanitizer")
    run.add_argument("--sanitize-tool", default="memcheck", help="Sanitizer tool: memcheck, racecheck, initcheck, synccheck")
    run.add_argument("--emit-disassembly", action="store_true", help="Emit source/PTX/SASS tri-view artifacts")
    run.add_argument("--triton-build-spec", default=None, help="Optional Triton build spec module path, optionally suffixed with :function")
    run.add_argument("--source-file", default=None, help="Optional source file for tri-view/disassembly")
    run.add_argument("--binary-file", default=None, help="Optional binary file to disassemble")
    run.add_argument("--ptx-file", default=None, help="Optional PTX file to include in tri-view")
    run.add_argument("--sass-file", default=None, help="Optional SASS file to include in tri-view")
    run.add_argument("--backend", default=None, help="Override backend")
    run.add_argument("--vendor", default=None, help="Override vendor")
    run.add_argument("--executor", default="local_host", help="Execution backend")
    run.add_argument("--policy-pack", default="balanced", help="Budget policy pack")
    run.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to execute after '--'")

    eval_cmd = subparsers.add_parser("eval", help="Run command plus task-defined evaluation hooks")
    eval_cmd.add_argument("--task", required=True, help="Task id or path to TaskSpec JSON")
    eval_cmd.add_argument("--trace-system", action="store_true", help="Use NVIDIA Nsight Systems tracing when available")
    eval_cmd.add_argument("--backend", default=None, help="Override backend")
    eval_cmd.add_argument("--vendor", default=None, help="Override vendor")
    eval_cmd.add_argument("--executor", default="local_host", help="Execution backend")
    eval_cmd.add_argument("--policy-pack", default="balanced", help="Budget policy pack")
    eval_cmd.add_argument("--determinism-runs", type=int, default=2, help="Number of runs to compare for determinism")
    eval_cmd.add_argument("--scan-path", action="append", default=[], help="Additional file path to scan for forbidden patterns")
    eval_cmd.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to evaluate after '--'")

    bench = subparsers.add_parser("bench", help="Benchmark a command and emit PerfReport")
    bench.add_argument("--task", default=None, help="Task id or path to TaskSpec JSON")
    bench.add_argument("--adapter", default=None, help="Benchmark adapter name")
    bench.add_argument("--case", default=None, help="Benchmark case id")
    bench.add_argument("--executor", default="local_host", help="Execution backend")
    bench.add_argument("--policy-pack", default="balanced", help="Budget policy pack")
    bench.add_argument("--warmups", type=int, default=1)
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to benchmark after '--'")

    trajectory = subparsers.add_parser("trajectory", help="Capture or export offline trajectory datasets from run bundles")
    trajectory_subparsers = trajectory.add_subparsers(dest="trajectory_command", required=True)
    trajectory_capture = trajectory_subparsers.add_parser("capture", help="Write one trajectory episode from a run bundle")
    trajectory_capture.add_argument("run_ref", help="Run id or path")
    trajectory_capture.add_argument("--out", type=Path, required=True, help="Destination episode JSON path")
    trajectory_capture.add_argument("--policy-id", default="reference_policy_v1", help="Policy identifier")
    trajectory_capture.add_argument("--section", default="full", choices=["full", "summary", "build", "eval", "profile", "replay", "quality"])
    trajectory_export = trajectory_subparsers.add_parser("export", help="Export a dataset manifest plus episodes from run bundles")
    trajectory_export.add_argument("run_refs", nargs="+", help="Run ids or paths")
    trajectory_export.add_argument("--out-dir", type=Path, required=True, help="Destination dataset directory")
    trajectory_export.add_argument("--policy-id", default="reference_policy_v1", help="Policy identifier")
    trajectory_export.add_argument("--split", default="seed", help="Dataset split label")
    trajectory_export.add_argument("--section", default="full", choices=["full", "summary", "build", "eval", "profile", "replay", "quality"])
    trajectory_validate = trajectory_subparsers.add_parser("validate", help="Validate a trajectory dataset directory")
    trajectory_validate.add_argument("dataset_dir", type=Path, help="Dataset directory")

    env = subparsers.add_parser("env", help="Bounded agent-environment helpers for scripted collection")
    env_subparsers = env.add_subparsers(dest="env_command", required=True)
    env_subparsers.add_parser("action-space", help="List the bounded action space for v0")
    env_scripted = env_subparsers.add_parser("scripted", help="Run the scripted reference policy and emit one episode")
    env_scripted.add_argument("--task", required=True, help="Task id or path to TaskSpec JSON")
    env_scripted.add_argument("--out", type=Path, required=True, help="Destination episode JSON path")
    env_scripted.add_argument("--policy-id", default="scripted_reference_v1", help="Policy identifier")
    env_scripted.add_argument("--step-budget", type=int, default=5, help="Maximum scripted environment steps")
    env_scripted.add_argument("--section", default="summary", choices=["full", "summary", "build", "eval", "profile", "replay", "quality"])
    env_scripted.add_argument("--with-build", action="store_true", help="Include a build step when a Triton build spec is provided")
    env_scripted.add_argument("--triton-build-spec", default=None, help="Optional Triton build spec module path, optionally suffixed with :function")
    env_scripted.add_argument("--backend", default=None, help="Override backend")
    env_scripted.add_argument("--vendor", default=None, help="Override vendor")
    env_scripted.add_argument("--executor", default="local_host", help="Execution backend")
    env_scripted.add_argument("--policy-pack", default="balanced", help="Budget policy pack")
    env_scripted.add_argument("--determinism-runs", type=int, default=2, help="Number of runs to compare for determinism")
    env_scripted.add_argument("--workflow", default="auto", choices=["auto", "standard", "diagnose_first", "debug", "reformulate"], help="Scripted workflow shape")
    env_scripted.add_argument("--no-knowledge", action="store_true", help="Skip the initial knowledge query step")
    env_scripted.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to evaluate after '--'")

    sft = subparsers.add_parser("sft", help="Package trajectory datasets into warm-start SFT examples")
    sft_subparsers = sft.add_subparsers(dest="sft_command", required=True)
    sft_package = sft_subparsers.add_parser("package", help="Package a trajectory dataset into SFT examples")
    sft_package.add_argument("dataset_dir", type=Path, help="Trajectory dataset directory")
    sft_package.add_argument("--out-dir", type=Path, required=True, help="Destination SFT dataset directory")
    sft_package.add_argument("--split", default="train", help="Dataset split label")
    sft_package.add_argument("--success-only", action="store_true", help="Exclude failed episodes")
    sft_package.add_argument("--public-only", action="store_true", help="Keep only public benchmark tasks")
    sft_package.add_argument("--include-benchmark-only", action="store_true", help="Allow benchmark-only episodes into the packaged dataset")
    sft_package.add_argument("--patch-bearing-only", action="store_true", help="Keep only patch-bearing episodes")
    sft_package.add_argument("--verb", action="append", default=[], help="Restrict packaging to one or more task verbs")
    sft_package.add_argument("--governance", action="append", default=[], help="Restrict packaging to one or more episode governance kinds")
    sft_package.add_argument("--transition-kind", action="append", default=[], help="Require at least one matching transition kind")
    sft_validate = sft_subparsers.add_parser("validate", help="Validate an SFT dataset directory")
    sft_validate.add_argument("dataset_dir", type=Path, help="SFT dataset directory")

    knowledge = subparsers.add_parser("knowledge", help="Build and query the local knowledge index")
    knowledge_subparsers = knowledge.add_subparsers(dest="knowledge_command", required=True)
    knowledge_build = knowledge_subparsers.add_parser("build-index", help="Build the knowledge index")
    knowledge_build.add_argument("--out-dir", type=Path, default=None, help="Optional output directory")
    knowledge_query = knowledge_subparsers.add_parser("query", help="Query the knowledge index")
    knowledge_query.add_argument("--query", default="", help="Free-text query")
    knowledge_query.add_argument("--operator-family", default=None, help="Optional operator-family filter")
    knowledge_query.add_argument("--verb", default=None, help="Optional task-verb filter")
    knowledge_query.add_argument("--benchmark-name", default=None, help="Optional benchmark-family filter")
    knowledge_query.add_argument("--backend", default=None, help="Optional backend filter")
    knowledge_query.add_argument("--vendor", default=None, help="Optional vendor filter")
    knowledge_query.add_argument("--kind", default=None, help="Optional entry kind filter")
    knowledge_query.add_argument("--limit", type=int, default=5, help="Maximum rows to return")
    knowledge_query.add_argument("--prefer-mixed", action="store_true", help="Interleave docs and run examples when possible")
    knowledge_query.add_argument("--index-dir", type=Path, default=None, help="Optional index directory")
    knowledge_similar = knowledge_subparsers.add_parser("similar-task", help="Retrieve similar knowledge for a task")
    knowledge_similar.add_argument("task_id", help="Task id")
    knowledge_similar.add_argument("--limit", type=int, default=5, help="Maximum rows to return")
    knowledge_similar.add_argument("--index-dir", type=Path, default=None, help="Optional index directory")

    train = subparsers.add_parser("train", help="Training-config validation and smoke scaffolding")
    train_subparsers = train.add_subparsers(dest="train_command", required=True)
    train_validate = train_subparsers.add_parser("validate-config", help="Validate an SFT training config")
    train_validate.add_argument("config_path", type=Path, help="Path to SFT training config JSON")
    train_smoke = train_subparsers.add_parser("smoke-sft", help="Write a smoke SFT training report from a config")
    train_smoke.add_argument("config_path", type=Path, help="Path to SFT training config JSON")
    train_smoke.add_argument("--out", type=Path, required=True, help="Destination report path")
    train_run = train_subparsers.add_parser("run-sft", help="Run a bounded local SFT job from a config")
    train_run.add_argument("config_path", type=Path, help="Path to SFT training config JSON")
    train_run.add_argument("--model-override", default=None, help="Optional model override for local smoke debugging")
    train_run.add_argument("--max-steps-override", type=int, default=None, help="Optional max-steps override")

    rollout = subparsers.add_parser("rollout", help="Scripted rollout suite scaffolding for held-out debug/diagnose tasks")
    rollout_subparsers = rollout.add_subparsers(dest="rollout_command", required=True)
    rollout_run = rollout_subparsers.add_parser("scripted", help="Run a scripted rollout suite from a rollout config")
    rollout_run.add_argument("config_path", type=Path, help="Path to rollout config JSON")
    rollout_run.add_argument("--out-dir", type=Path, required=True, help="Destination rollout report directory")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "doctor":
        report = collect_doctor_report()
        payload = report.model_dump(mode="json")
        rendered = json.dumps(payload, indent=2)
        print(rendered)
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(rendered + "\n", encoding="utf-8")
        return 0

    if args.command == "task" and args.task_command == "list":
        registry = TaskRegistry(Path.cwd())
        rows = [
            {
                "task_id": task.task_id,
                "verb": task.verb,
                "operator_family": task.operator_family,
                "backends": task.allowed_backends,
            }
            for task in registry.load_all()
        ]
        print(json.dumps(rows, indent=2))
        return 0

    if args.command == "adapter" and args.adapter_command == "list":
        print(json.dumps(list_adapters(Path.cwd()), indent=2))
        return 0

    if args.command == "adapter" and args.adapter_command == "cases":
        rows = [case.model_dump(mode="json") for case in list_adapter_cases(Path.cwd(), args.adapter_name)]
        print(json.dumps(rows, indent=2))
        return 0

    if args.command == "adapter" and args.adapter_command == "summary":
        print(json.dumps(describe_adapter(Path.cwd(), args.adapter_name), indent=2))
        return 0

    if args.command == "adapter" and args.adapter_command == "show":
        adapter = get_adapter(args.adapter_name)
        case = adapter.load_case(Path.cwd(), args.case_id)
        task = adapter.load_task(Path.cwd(), args.case_id)
        print(
            json.dumps(
                {
                    "case": case.model_dump(mode="json"),
                    "task": task.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "inspect":
        print(json.dumps(inspect_run(Path.cwd(), args.run_ref, section=args.section), indent=2))
        return 0

    if args.command == "runs" and args.runs_command == "list":
        rows = list_runs(
            Path.cwd(),
            task_id=args.task_id,
            backend=args.backend,
            vendor=args.vendor,
            status=args.status,
            limit=args.limit,
        )
        print(json.dumps(rows, indent=2))
        return 0

    if args.command == "replay":
        print(json.dumps(validate_run_bundle(Path.cwd(), args.run_ref), indent=2))
        return 0

    if args.command == "bundle":
        out_path = export_proof_bundle(Path.cwd(), args.run_ref, out_path=args.out, include_raw=args.full)
        print(out_path)
        return 0

    if args.command == "compare":
        comparison = compare_runs(Path.cwd(), args.lhs_run_ref, args.rhs_run_ref)
        print(json.dumps(comparison.model_dump(mode="json"), indent=2))
        return 0

    if args.command == "build":
        command = list(args.cmd)
        if command[:1] == ["--"]:
            command = command[1:]
        run_dir = run_task(
            root=Path.cwd(),
            task_ref=args.task,
            command=command,
            emit_disassembly=True,
            triton_build_spec=args.triton_build_spec,
            source_file=args.source_file,
            binary_file=args.binary_file,
            ptx_file=args.ptx_file,
            sass_file=args.sass_file,
            backend=args.backend,
            vendor=args.vendor,
            executor=args.executor,
            policy_pack=args.policy_pack,
        )
        print(run_dir)
        return 0

    if args.command == "run":
        command = list(args.cmd)
        if command[:1] == ["--"]:
            command = command[1:]
        run_dir = run_task(
            root=Path.cwd(),
            task_ref=args.task,
            command=command,
            trace_system=args.trace_system,
            profile_kernel=args.profile_kernel,
            profile_pack=args.profile_pack,
            sanitize=args.sanitize,
            sanitize_tool=args.sanitize_tool,
            emit_disassembly=args.emit_disassembly,
            triton_build_spec=args.triton_build_spec,
            source_file=args.source_file,
            binary_file=args.binary_file,
            ptx_file=args.ptx_file,
            sass_file=args.sass_file,
            backend=args.backend,
            vendor=args.vendor,
            executor=args.executor,
            policy_pack=args.policy_pack,
        )
        print(run_dir)
        return 0

    if args.command == "bench":
        command = list(args.cmd)
        if command[:1] == ["--"]:
            command = command[1:]
        registry = TaskRegistry(Path.cwd())
        if args.task:
            task = registry.get(args.task)
        elif args.adapter and args.case:
            adapter = get_adapter(args.adapter)
            case = adapter.load_case(Path.cwd(), args.case)
            task = adapter.load_task(Path.cwd(), args.case)
            if not command:
                command = list(case.default_command)
        else:
            parser.error("bench requires --task or the pair --adapter and --case")
        if not command:
            parser.error("bench requires a command, either after '--' or from the selected case")
        doctor_report = collect_doctor_report()
        command_executor = make_executor(args.executor, Path.cwd())
        run_spec = build_run_spec(
            task=task,
            backend=task.allowed_backends[0] if task.allowed_backends else "triton",
            vendor="nvidia",
            executor=args.executor,
            policy_pack=args.policy_pack,
            tool_versions={tool.name: tool.version for tool in doctor_report.available_tools if tool.available and tool.version},
        )
        run_spec.run_id = build_run_id(prefix="bench")
        writer = RunBundleWriter(Path.cwd())
        run_dir = writer.initialize(run_spec)
        write_task_artifacts(writer, task, doctor_report)
        if args.task or (args.adapter and args.case):
            task.perf_protocol.warmups = args.warmups
            task.perf_protocol.repeats = args.repeats
            perf = run_task_benchmark(writer, root=Path.cwd(), task=task, command=command, executor=command_executor)
        else:
            perf = run_subprocess_benchmark(writer, command=command, warmups=args.warmups, repeats=args.repeats, executor=command_executor)
        write_run_summary(
            writer,
            summary=RunSummary(
                run_id=run_spec.run_id,
                task_id=task.task_id,
                status="ok",
                trace_enabled=False,
                backend=run_spec.target_backend,
                vendor=run_spec.target_vendor,
                exit_code=0,
                duration_ms=int(perf.steady_state_ms_p50),
                key_artifacts=(
                    [
                        "manifest.json",
                        "events.jsonl",
                        "prompt/task_spec.json",
                        "meta/task_spec_full.json",
                        "meta/doctor_report.json",
                    ]
                    + (["meta/hardware_fingerprint.json"] if doctor_report.hardware_fingerprints else [])
                    + [
                        "perf/benchmark.json",
                        "perf/raw_timings.json",
                    ]
                ),
            ),
        )
        write_replay_pack(
            writer=writer,
            task=task,
            doctor_report=doctor_report,
            command=command,
            required_artifacts=(
                [
                    "manifest.json",
                    "events.jsonl",
                    "prompt/task_spec.json",
                    "meta/task_spec_full.json",
                    "meta/doctor_report.json",
                ]
                + (["meta/hardware_fingerprint.json"] if doctor_report.hardware_fingerprints else [])
                + [
                    "perf/benchmark.json",
                    "perf/raw_timings.json",
                    "summary.json",
                    "summary.md",
                ]
            ),
            environment={
                "executor": "local_host",
                "executor_kind": args.executor,
                "policy_pack": args.policy_pack,
                "target_backend": run_spec.target_backend,
                "target_vendor": run_spec.target_vendor,
                "warmups": args.warmups,
                "repeats": args.repeats,
            },
        )
        writer.append_event(scope="run", kind="completed", payload={"status": "ok"})
        print(run_dir)
        return 0

    if args.command == "trajectory" and args.trajectory_command == "capture":
        out_path = write_trajectory_episode(
            Path.cwd(),
            args.run_ref,
            args.out,
            policy_id=args.policy_id,
            section=args.section,
        )
        print(out_path)
        return 0

    if args.command == "trajectory" and args.trajectory_command == "export":
        manifest_path = export_trajectory_dataset(
            Path.cwd(),
            args.run_refs,
            args.out_dir,
            policy_id=args.policy_id,
            split=args.split,
            section=args.section,
        )
        print(manifest_path)
        return 0

    if args.command == "trajectory" and args.trajectory_command == "validate":
        print(json.dumps(validate_trajectory_dataset(args.dataset_dir), indent=2))
        return 0

    if args.command == "env" and args.env_command == "action-space":
        print(json.dumps([row.model_dump(mode="json") for row in list_action_space()], indent=2))
        return 0

    if args.command == "env" and args.env_command == "scripted":
        command = list(args.cmd)
        if command[:1] == ["--"]:
            command = command[1:]
        episode = run_scripted_reference_episode(
            Path.cwd(),
            args.task,
            command,
            policy_id=args.policy_id,
            step_budget=args.step_budget,
            section=args.section,
            include_knowledge=not args.no_knowledge,
            include_build=args.with_build,
            triton_build_spec=args.triton_build_spec,
            backend=args.backend,
            vendor=args.vendor,
            executor=args.executor,
            policy_pack=args.policy_pack,
            determinism_runs=args.determinism_runs,
            workflow=args.workflow,
        )
        out_path = write_episode(episode, args.out)
        print(out_path)
        return 0

    if args.command == "sft" and args.sft_command == "package":
        manifest_path = package_trajectory_dataset_as_sft(
            Path.cwd(),
            args.dataset_dir,
            args.out_dir,
            split=args.split,
            include_failures=not args.success_only,
            only_public_benchmarks=args.public_only,
            include_benchmark_only=args.include_benchmark_only,
            patch_bearing_only=args.patch_bearing_only,
            verb_allowlist=list(args.verb),
            governance_allowlist=list(args.governance),
            transition_kind_allowlist=list(args.transition_kind),
        )
        print(manifest_path)
        return 0

    if args.command == "sft" and args.sft_command == "validate":
        print(json.dumps(validate_sft_dataset(args.dataset_dir), indent=2))
        return 0

    if args.command == "knowledge" and args.knowledge_command == "build-index":
        manifest_path = build_knowledge_index(Path.cwd(), out_dir=args.out_dir)
        print(manifest_path)
        return 0

    if args.command == "knowledge" and args.knowledge_command == "query":
        rows = query_knowledge(
            Path.cwd(),
            query=args.query,
            operator_family=args.operator_family,
            verb=args.verb,
            benchmark_name=args.benchmark_name,
            backend=args.backend,
            vendor=args.vendor,
            kind=args.kind,
            limit=args.limit,
            prefer_mixed=args.prefer_mixed,
            index_dir=args.index_dir,
        )
        print(json.dumps(rows, indent=2))
        return 0

    if args.command == "knowledge" and args.knowledge_command == "similar-task":
        rows = retrieve_similar_for_task(
            Path.cwd(),
            args.task_id,
            limit=args.limit,
            index_dir=args.index_dir,
        )
        print(json.dumps(rows, indent=2))
        return 0

    if args.command == "train" and args.train_command == "validate-config":
        config = load_training_config(args.config_path)
        print(json.dumps(validate_sft_training_config(Path.cwd(), config), indent=2))
        return 0

    if args.command == "train" and args.train_command == "smoke-sft":
        report_path = write_sft_smoke_report(Path.cwd(), args.config_path, args.out)
        print(report_path)
        return 0

    if args.command == "train" and args.train_command == "run-sft":
        report_path = run_sft_training_job(
            Path.cwd(),
            args.config_path,
            model_override=args.model_override,
            max_steps_override=args.max_steps_override,
        )
        print(report_path)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        return 0 if payload.get("status") == "ok" else 1

    if args.command == "rollout" and args.rollout_command == "scripted":
        config_payload = json.loads(args.config_path.read_text(encoding="utf-8"))
        config = RLRolloutConfig.model_validate(config_payload)
        report = run_scripted_rollout_suite(Path.cwd(), config, args.out_dir)
        print(json.dumps(report.model_dump(mode="json"), indent=2))
        return 0

    if args.command == "eval":
        command = list(args.cmd)
        if command[:1] == ["--"]:
            command = command[1:]
        registry = TaskRegistry(Path.cwd())
        task = registry.get(args.task)
        doctor_report = collect_doctor_report()
        command_executor = make_executor(args.executor, Path.cwd())
        run_spec = build_run_spec(
            task=task,
            backend=args.backend or (task.allowed_backends[0] if task.allowed_backends else "triton"),
            vendor=args.vendor or "nvidia",
            executor=args.executor,
            policy_pack=args.policy_pack,
            tool_versions={tool.name: tool.version for tool in doctor_report.available_tools if tool.available and tool.version},
        )
        run_spec.run_id = build_run_id(prefix="eval")
        writer = RunBundleWriter(Path.cwd())
        run_dir = writer.initialize(run_spec)
        write_task_artifacts(writer, task, doctor_report)

        command_summary = None
        perf_report = None
        if args.trace_system and run_spec.target_vendor == "nvidia":
            if args.executor != "local_host":
                parser.error("eval --trace-system currently requires --executor local_host")
            from gpu_cockpit.backends.nvidia.nsys import trace_system_nvidia

            command_summary = trace_system_nvidia(writer, command)
        else:
            from gpu_cockpit.engine.command_runner import run_command

            command_summary = run_command(writer, command, executor=command_executor)

        if task.baseline_ref:
            perf_report = run_task_benchmark(
                writer,
                root=Path.cwd(),
                task=task,
                command=command,
                scope="tool.run_benchmark.eval",
                executor=command_executor,
            )

        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=Path.cwd(),
            task=task,
            command=command,
            command_summary=command_summary,
            perf_report=perf_report,
            scan_paths=[Path(path) for path in args.scan_path],
            determinism_runs=args.determinism_runs,
            executor=command_executor,
        )

        status = "ok" if envelope.final_score > 0 else "failed"
        key_artifacts = [
            "manifest.json",
            "events.jsonl",
            "prompt/task_spec.json",
            "meta/task_spec_full.json",
            "meta/doctor_report.json",
            "correctness/correctness.json",
            "correctness/determinism.json",
            "eval/anti_hack_report.json",
            "eval/eval_envelope.json",
        ]
        if doctor_report.hardware_fingerprints:
            key_artifacts.insert(4, "meta/hardware_fingerprint.json")
        if command_summary is not None:
            key_artifacts.extend(
                [
                    path
                    for path in [
                        command_summary.stdout_path,
                        command_summary.stderr_path,
                        command_summary.report_path,
                        command_summary.sqlite_path,
                    ]
                    if path
                ]
            )
        if perf_report is not None:
            key_artifacts.extend(["perf/benchmark.json", "perf/raw_timings.json"])
        write_run_summary(
            writer,
            summary=RunSummary(
                run_id=run_spec.run_id,
                task_id=task.task_id,
                status=status,
                trace_enabled=bool(command_summary.trace_enabled if command_summary else False),
                backend=run_spec.target_backend,
                vendor=run_spec.target_vendor,
                exit_code=command_summary.exit_code if command_summary else None,
                duration_ms=command_summary.duration_ms if command_summary else None,
                key_artifacts=key_artifacts,
                warnings=(command_summary.warnings if command_summary else []) + anti_hack.warnings,
            ),
        )
        write_replay_pack(
            writer=writer,
            task=task,
            doctor_report=doctor_report,
            command=command,
            required_artifacts=key_artifacts + ["summary.json", "summary.md"],
            environment={
                "executor": args.executor,
                "policy_pack": args.policy_pack,
                "target_backend": run_spec.target_backend,
                "target_vendor": run_spec.target_vendor,
                "trace_system": args.trace_system,
                "determinism_runs": args.determinism_runs,
            },
        )
        writer.append_event(
            scope="run",
            kind="completed" if status == "ok" else "failed",
            payload={
                "status": status,
                "compile_ok": correctness.compile_ok,
                "visible_tests_ok": correctness.visible_tests_ok,
                "hidden_tests_ok": correctness.hidden_tests_ok,
                "anti_hack_passed": anti_hack.passed,
                "determinism_passed": determinism.passed,
                "final_score": envelope.final_score,
            },
        )
        print(run_dir)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
