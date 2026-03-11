from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_cockpit.contracts import KnowledgeEntry, KnowledgeIndexManifest, TrajectoryEpisode
from gpu_cockpit.engine.evidence import assess_run_evidence
from gpu_cockpit.engine.indexer import list_runs
from gpu_cockpit.engine.inspector import inspect_run, resolve_run_dir
from gpu_cockpit.engine.task_registry import TaskRegistry


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_+-]{3,}", text.lower())


def _summarize_markdown(path: Path) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    title = path.stem.replace("_", " ").title()
    summary = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            continue
        if stripped and not stripped.startswith("#"):
            summary = stripped
            break
    return title, summary


def _build_markdown_entries(root: Path) -> list[KnowledgeEntry]:
    knowledge_root = root / "knowledge"
    if not knowledge_root.exists():
        return []
    entries: list[KnowledgeEntry] = []
    for path in sorted(knowledge_root.rglob("*.md")):
        relative_path = path.relative_to(root)
        parent = relative_path.parts[1] if len(relative_path.parts) > 1 else "misc"
        title, summary = _summarize_markdown(path)
        tags = [parent, *[part.replace("_", "-") for part in relative_path.parts[1:-1]]]
        operator_family = None
        if parent == "operator_families":
            operator_family = path.stem
            tags.append(operator_family)
        entries.append(
            KnowledgeEntry(
                entry_id=f"knowledge:{relative_path.as_posix()}",
                kind="knowledge_doc",
                title=title,
                path=relative_path.as_posix(),
                source_type="markdown",
                summary=summary,
                tags=list(dict.fromkeys(tags)),
                keywords=list(dict.fromkeys(_tokenize(f"{title} {summary} {' '.join(tags)}"))),
                operator_family=operator_family,
            )
        )
    return entries


def _build_task_entries(root: Path) -> list[KnowledgeEntry]:
    registry = TaskRegistry(root)
    entries: list[KnowledgeEntry] = []
    for task in registry.load_all():
        tags = [task.verb, task.operator_family, *task.allowed_backends, *task.feature_requirements]
        entries.append(
            KnowledgeEntry(
                entry_id=f"task:{task.task_id}",
                kind="task",
                title=task.task_id,
                path=f"workloads/tasks/{task.task_id.replace('/', '_')}.json",
                source_type="task_spec",
                summary=task.prompt,
                tags=list(dict.fromkeys(tags)),
                keywords=list(dict.fromkeys(_tokenize(f"{task.task_id} {task.prompt} {' '.join(tags)}"))),
                operator_family=task.operator_family,
                backend=task.allowed_backends[0] if task.allowed_backends else None,
                vendor="nvidia" if "cuda" in task.feature_requirements or "triton" in task.allowed_backends else None,
                metadata={
                    "difficulty": task.difficulty,
                    "reference_impl_ref": task.reference_impl_ref,
                    "verb": task.verb,
                    "benchmark_name": _benchmark_name(task.task_id),
                },
            )
        )
    return entries


def _benchmark_name(task_id: str | None) -> str | None:
    if task_id is None:
        return None
    if task_id.startswith("task/kernelbench/"):
        return "KernelBench"
    if task_id.startswith("task/computeeval/"):
        return "ComputeEval"
    return None


def _build_run_entry(root: Path, run_dir: Path, *, entry_prefix: str) -> KnowledgeEntry | None:
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    if not summary_path.exists() and not manifest_path.exists():
        return None
    inspected = inspect_run(root, str(run_dir))
    summary_payload = inspected if "projection" in inspected else {}
    projection = inspected.get("projection", {}) if isinstance(inspected, dict) else {}
    task_spec = projection.get("replay_pack", {}) if isinstance(projection, dict) else {}
    run_id = str(summary_payload.get("run_id", run_dir.name))
    task_id = summary_payload.get("task_id")
    operator_family = None
    task_spec_payload = json.loads((run_dir / "prompt" / "task_spec.json").read_text(encoding="utf-8")) if (run_dir / "prompt" / "task_spec.json").exists() else {}
    if task_spec_payload.get("operator_family") is not None:
        operator_family = str(task_spec_payload["operator_family"])
    task_verb = str(task_spec_payload["verb"]) if task_spec_payload.get("verb") is not None else None
    backend = summary_payload.get("backend")
    vendor = summary_payload.get("vendor")
    status = str(summary_payload.get("status", "unknown"))
    evidence = assess_run_evidence(run_dir)
    eval_envelope = projection.get("eval_envelope", {}) if isinstance(projection, dict) else {}
    bottleneck = projection.get("bottleneck_card", {}) if isinstance(projection, dict) else {}
    benchmark_name = _benchmark_name(str(task_id) if task_id is not None else None)

    summary_parts = [
        f"Run {run_id} for {task_id or 'unknown task'} finished with status {status} on {backend or 'unknown'}/{vendor or 'unknown'}.",
        f"Evidence score {evidence.overall_score:.2f}.",
    ]
    final_score = eval_envelope.get("final_score") if isinstance(eval_envelope, dict) else None
    if isinstance(final_score, (int, float)):
        summary_parts.append(f"Final eval score {float(final_score):.2f}.")
    primary_bottleneck = bottleneck.get("primary_bottleneck") if isinstance(bottleneck, dict) else None
    if primary_bottleneck is not None:
        summary_parts.append(f"Primary bottleneck {primary_bottleneck}.")
    failure_triage = projection.get("failure_triage", {}) if isinstance(projection, dict) else {}
    failure_class = failure_triage.get("failure_class") if isinstance(failure_triage, dict) else None
    if failure_class not in {None, "ready_positive"}:
        summary_parts.append(f"Failure class {failure_class}.")
    sanitizer_summary = projection.get("sanitizer_summary", {}) if isinstance(projection, dict) else {}
    dominant_sanitizer_family = (
        sanitizer_summary.get("dominant_failure_family")
        if isinstance(sanitizer_summary, dict)
        else None
    )
    if dominant_sanitizer_family is not None:
        summary_parts.append(f"Sanitizer family {dominant_sanitizer_family}.")
    if benchmark_name is not None:
        summary_parts.append(f"Benchmark family {benchmark_name}.")
    patch_kind = summary_payload.get("patch_kind")
    transition_kind = summary_payload.get("transition_kind")
    if summary_payload.get("patch_present"):
        summary_parts.append(f"Patch-bearing candidate transition {transition_kind or 'applied'} using {patch_kind or 'unspecified patch'} change.")
    summary = " ".join(summary_parts)

    tags = [status, "run-example"]
    if backend is not None:
        tags.append(str(backend))
    if vendor is not None:
        tags.append(str(vendor))
    if operator_family is not None:
        tags.append(operator_family)
    if task_verb is not None:
        tags.append(task_verb)
    if benchmark_name is not None:
        tags.append(benchmark_name.lower())
    if summary_payload.get("patch_present"):
        tags.extend(["patch-bearing", "repair-example" if task_verb == "debug" else "transform-example"])
    if isinstance(failure_class, str) and failure_class:
        tags.append(failure_class)
    if isinstance(dominant_sanitizer_family, str) and dominant_sanitizer_family:
        tags.append(dominant_sanitizer_family)
    if isinstance(patch_kind, str) and patch_kind:
        tags.append(patch_kind)
    if isinstance(transition_kind, str) and transition_kind:
        tags.append(transition_kind)
    if evidence.benchmark_reporting.eligible:
        tags.append("benchmark-ready")
    if evidence.sft_collection.eligible:
        tags.append("sft-ready")
    if evidence.rl_reward_trace.eligible:
        tags.append("rl-trace-ready")

    return KnowledgeEntry(
        entry_id=f"{entry_prefix}:{run_id}",
        kind="run_example",
        title=run_id,
        path=str(run_dir.relative_to(root)),
        source_type="run_bundle",
        summary=summary,
        tags=list(dict.fromkeys(tags)),
        keywords=list(dict.fromkeys(_tokenize(f"{summary} {task_id or ''} {' '.join(tags)}"))),
        operator_family=operator_family,
        backend=str(backend) if backend is not None else None,
        vendor=str(vendor) if vendor is not None else None,
        metadata={
            "task_id": task_id,
            "status": status,
            "benchmark_name": benchmark_name,
            "task_verb": task_verb,
            "evidence_score": evidence.overall_score,
            "benchmark_ready": evidence.benchmark_reporting.eligible,
            "sft_ready": evidence.sft_collection.eligible,
            "rl_trace_ready": evidence.rl_reward_trace.eligible,
            "training_example_kind": evidence.training_example_kind,
            "patch_present": bool(summary_payload.get("patch_present", False)),
            "patch_kind": patch_kind,
            "transition_kind": transition_kind,
            "candidate_role": summary_payload.get("candidate_role"),
            "failure_class": failure_class,
            "dominant_sanitizer_family": dominant_sanitizer_family,
        },
    )


def _build_run_entries(root: Path, limit: int = 6) -> list[KnowledgeEntry]:
    candidate_dirs: list[tuple[str, Path]] = []
    golden_root = root / "tests" / "golden_runs"
    if golden_root.exists():
        for path in sorted(golden_root.iterdir()):
            if path.is_dir():
                candidate_dirs.append(("golden_run", path))

    rows = list_runs(root, limit=100)
    selected_run_dirs: list[Path] = []
    seen_statuses: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status"))
        current = seen_statuses.get(status, 0)
        if current >= max(1, limit // 2):
            continue
        selected_run_dirs.append(resolve_run_dir(root, str(row["run_id"])))
        seen_statuses[status] = current + 1
        if len(selected_run_dirs) >= limit:
            break
    candidate_dirs.extend(("run", run_dir) for run_dir in selected_run_dirs)

    entries: list[KnowledgeEntry] = []
    for entry_prefix, run_dir in candidate_dirs:
        entry = _build_run_entry(root, run_dir, entry_prefix=entry_prefix)
        if entry is not None:
            entries.append(entry)
    return entries


def _build_episode_entry(root: Path, episode_path: Path, *, entry_prefix: str) -> KnowledgeEntry | None:
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    episode = TrajectoryEpisode.model_validate(payload)
    governance = episode.governance
    patch_present = any(step.action.action_type == "patch_candidate" for step in episode.steps)
    patch_kinds = sorted(
        {
            str(step.action.metadata["patch_kind"])
            for step in episode.steps
            if isinstance(step.action.metadata.get("patch_kind"), str)
        }
    )
    patch_intents = [
        str(step.action.metadata["patch_intent"])
        for step in episode.steps
        if isinstance(step.action.metadata.get("patch_intent"), str)
    ]
    transition_kinds = sorted({step.transition_kind for step in episode.steps if isinstance(step.transition_kind, str)})
    summary_parts = [
        f"Episode {episode.episode_id} for {episode.task_id} ended in {episode.terminal_state}.",
        f"Policy {episode.policy_id}.",
    ]
    if governance is not None:
        summary_parts.append(f"Governance {governance.episode_governance_kind}.")
    if patch_present:
        summary_parts.append("Patch-bearing repair or transformation example.")
    if patch_kinds:
        summary_parts.append(f"Patch kinds: {', '.join(patch_kinds)}.")
    if transition_kinds:
        summary_parts.append(f"Transitions: {', '.join(transition_kinds)}.")
    summary = " ".join(summary_parts)
    tags = ["episode-example", episode.terminal_state]
    if episode.operator_family:
        tags.append(episode.operator_family)
    if episode.task_verb:
        tags.append(episode.task_verb)
    if patch_present:
        tags.append("patch-bearing")
    tags.extend(transition_kinds)
    if governance is not None:
        tags.append(governance.episode_governance_kind)
    return KnowledgeEntry(
        entry_id=f"{entry_prefix}:{episode.episode_id}",
        kind="episode_example",
        title=episode.episode_id,
        path=str(episode_path.relative_to(root)),
        source_type="trajectory_episode",
        summary=summary,
        tags=list(dict.fromkeys(tags)),
        keywords=list(dict.fromkeys(_tokenize(f"{summary} {' '.join(tags)} {episode.task_id}"))),
        operator_family=episode.operator_family,
        metadata={
            "task_id": episode.task_id,
            "task_verb": episode.task_verb,
            "terminal_state": episode.terminal_state,
            "patch_present": patch_present,
            "patch_kinds": patch_kinds,
            "patch_intents": patch_intents,
            "transition_kinds": transition_kinds,
            "episode_governance_kind": governance.episode_governance_kind if governance is not None else None,
            "training_example_kind": governance.training_example_kind if governance is not None else episode.metadata.get("training_example_kind"),
        },
    )


def _build_episode_entries(root: Path) -> list[KnowledgeEntry]:
    episode_paths: list[tuple[str, Path]] = []
    golden_episode_root = root / "tests" / "golden_episodes"
    if golden_episode_root.exists():
        for path in sorted(golden_episode_root.glob("*.json")):
            episode_paths.append(("golden_episode", path))
    golden_dataset_root = root / "tests" / "golden_datasets"
    if golden_dataset_root.exists():
        for dataset_dir in sorted(golden_dataset_root.iterdir()):
            episodes_dir = dataset_dir / "episodes"
            if not episodes_dir.exists():
                continue
            for path in sorted(episodes_dir.glob("*.json")):
                episode_paths.append(("golden_dataset_episode", path))
    dataset_root = root / "datasets"
    if dataset_root.exists():
        for dataset_dir in sorted(dataset_root.iterdir()):
            episodes_dir = dataset_dir / "episodes"
            if not episodes_dir.exists():
                continue
            for path in sorted(episodes_dir.glob("*.json")):
                episode_paths.append(("dataset_episode", path))
    entries: list[KnowledgeEntry] = []
    for entry_prefix, episode_path in episode_paths:
        entry = _build_episode_entry(root, episode_path, entry_prefix=entry_prefix)
        if entry is not None:
            entries.append(entry)
    return entries


def build_knowledge_index(root: Path, *, out_dir: Path | None = None) -> Path:
    destination = out_dir or (root / "knowledge" / "index")
    destination.mkdir(parents=True, exist_ok=True)
    entries = _build_markdown_entries(root) + _build_task_entries(root) + _build_run_entries(root) + _build_episode_entries(root)
    entries_payload = [entry.model_dump(mode="json") for entry in entries]
    entries_path = destination / "entries.json"
    entries_path.write_text(json.dumps(entries_payload, indent=2) + "\n", encoding="utf-8")
    manifest = KnowledgeIndexManifest(
        index_id=f"knowledge_index_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}",
        created_at=datetime.now(tz=UTC),
        entry_count=len(entries),
        source_roots=["knowledge", "workloads/tasks", "runs"],
        entries_ref=str(entries_path.relative_to(destination)),
        metadata={"kinds": sorted({entry.kind for entry in entries})},
    )
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _load_entries(index_dir: Path) -> list[KnowledgeEntry]:
    entries_path = index_dir / "entries.json"
    if not entries_path.exists():
        raise FileNotFoundError(f"Knowledge index missing entries.json in {index_dir}")
    payload = json.loads(entries_path.read_text(encoding="utf-8"))
    return [KnowledgeEntry.model_validate(item) for item in payload]


def query_knowledge(
    root: Path,
    *,
    query: str = "",
    operator_family: str | None = None,
    verb: str | None = None,
    benchmark_name: str | None = None,
    backend: str | None = None,
    vendor: str | None = None,
    kind: str | None = None,
    limit: int = 5,
    prefer_mixed: bool = False,
    index_dir: Path | None = None,
) -> list[dict[str, Any]]:
    resolved_index_dir = index_dir or (root / "knowledge" / "index")
    entries = _load_entries(resolved_index_dir)
    query_tokens = set(_tokenize(query))
    rows: list[tuple[int, KnowledgeEntry]] = []
    for entry in entries:
        if operator_family and entry.operator_family != operator_family:
            continue
        entry_verb = None
        if isinstance(entry.metadata, dict) and entry.metadata.get("verb") is not None:
            entry_verb = str(entry.metadata["verb"])
        if isinstance(entry.metadata, dict) and entry.metadata.get("task_verb") is not None:
            entry_verb = str(entry.metadata["task_verb"])
        if verb and entry_verb != verb:
            continue
        entry_benchmark_name = None
        if isinstance(entry.metadata, dict) and entry.metadata.get("benchmark_name") is not None:
            entry_benchmark_name = str(entry.metadata["benchmark_name"])
        if benchmark_name and entry_benchmark_name != benchmark_name:
            continue
        if backend and entry.backend not in {backend, None}:
            continue
        if vendor and entry.vendor not in {vendor, None}:
            continue
        if kind and entry.kind != kind:
            continue
        score = 0
        if query_tokens:
            entry_tokens = set(entry.keywords) | set(_tokenize(entry.title)) | set(_tokenize(entry.summary))
            score += len(query_tokens & entry_tokens) * 10
        if operator_family and entry.operator_family == operator_family:
            score += 5
        if verb and entry_verb == verb:
            score += 5
        if benchmark_name and entry_benchmark_name == benchmark_name:
            score += 4
        if backend and entry.backend == backend:
            score += 3
        if vendor and entry.vendor == vendor:
            score += 3
        if entry.kind in {"run_example", "episode_example"} and isinstance(entry.metadata, dict):
            evidence_score = entry.metadata.get("evidence_score")
            if isinstance(evidence_score, (int, float)):
                score += min(5, int(float(evidence_score) * 5))
            patch_query_tokens = {"patch", "repair", "fix", "reformulate", "transform"}
            if entry.metadata.get("patch_present") is True and query_tokens & patch_query_tokens:
                score += 6
            if verb and entry_verb == verb and entry.metadata.get("patch_present") is True:
                score += 4
            patch_kinds = entry.metadata.get("patch_kinds")
            if isinstance(patch_kinds, list):
                score += len(query_tokens & {str(kind) for kind in patch_kinds}) * 3
            patch_intents = entry.metadata.get("patch_intents")
            if isinstance(patch_intents, list):
                intent_tokens = set()
                for intent in patch_intents:
                    intent_tokens |= set(_tokenize(str(intent)))
                score += len(query_tokens & intent_tokens) * 2
            if entry.metadata.get("rl_trace_ready") is True:
                score += 4
            elif entry.metadata.get("sft_ready") is True:
                score += 2
            failure_class = entry.metadata.get("failure_class")
            if isinstance(failure_class, str) and failure_class:
                failure_tokens = set(_tokenize(failure_class.replace("_", " ")))
                score += len(query_tokens & failure_tokens) * 4
            dominant_sanitizer_family = entry.metadata.get("dominant_sanitizer_family")
            if isinstance(dominant_sanitizer_family, str) and dominant_sanitizer_family:
                sanitizer_tokens = set(_tokenize(dominant_sanitizer_family.replace("_", " ")))
                score += len(query_tokens & sanitizer_tokens) * 4
            training_example_kind = entry.metadata.get("training_example_kind")
            if isinstance(training_example_kind, str):
                example_kind_tokens = set(_tokenize(training_example_kind.replace("_", " ")))
                score += len(query_tokens & example_kind_tokens) * 2
        if not query_tokens:
            score += 1
        if score > 0:
            rows.append((score, entry))
    rows.sort(key=lambda row: (-row[0], row[1].entry_id))
    selected = rows[:limit]
    if prefer_mixed:
        docs = [row for row in rows if row[1].kind == "knowledge_doc"][: max(1, limit // 2)]
        runs = [row for row in rows if row[1].kind == "run_example"][: max(1, limit // 2)]
        others = [row for row in rows if row[1].kind not in {"knowledge_doc", "run_example"}]
        selected = []
        while docs or runs or others:
            if docs and len(selected) < limit:
                selected.append(docs.pop(0))
            if runs and len(selected) < limit:
                selected.append(runs.pop(0))
            if others and len(selected) < limit:
                selected.append(others.pop(0))
            if len(selected) >= limit:
                break
    return [
        {
            "score": score,
            **entry.model_dump(mode="json"),
        }
        for score, entry in selected[:limit]
    ]


def retrieve_similar_for_task(root: Path, task_id: str, *, limit: int = 5, index_dir: Path | None = None) -> list[dict[str, Any]]:
    task = TaskRegistry(root).get(task_id)
    query = f"{task.operator_family} {task.verb} {' '.join(task.feature_requirements)}"
    return query_knowledge(
        root,
        query=query,
        operator_family=task.operator_family,
        verb=task.verb,
        benchmark_name=_benchmark_name(task.task_id),
        backend=task.allowed_backends[0] if task.allowed_backends else None,
        limit=limit,
        prefer_mixed=True,
        index_dir=index_dir,
    )
