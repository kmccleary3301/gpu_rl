from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from gpu_cockpit.contracts import BuildRecord, TriViewArtifact, TriViewLine
from gpu_cockpit.executors import CommandExecutor, LocalHostToolExecutor
from gpu_cockpit.engine.run_bundle import RunBundleWriter


def _read_text(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _infer_source_path(command: list[str], source_file: str | None) -> Path | None:
    if source_file:
        return Path(source_file)
    for arg in command:
        candidate = Path(arg)
        if candidate.suffix in {".py", ".cu", ".cuh", ".cpp", ".c", ".cc", ".h", ".hpp", ".ptx", ".sass"} and candidate.exists():
            return candidate
    return None


def _infer_anchor_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in text.replace("(", " ").replace(")", " ").replace(",", " ").replace(";", " ").split():
        cleaned = token.strip()
        if len(cleaned) >= 4 and any(ch.isalpha() for ch in cleaned):
            tokens.append(cleaned)
    return tokens[:3]


_PTX_FILE_RE = re.compile(r'^\s*\.file\s+(?P<index>\d+)\s+"(?P<path>[^"]+)"')
_PTX_LOC_RE = re.compile(r'^\s*\.loc\s+(?P<file>\d+)\s+(?P<line>\d+)\s+(?P<col>\d+)')


def _is_ptx_instruction(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith(("//", ".", "{", "}"))


def _is_sass_instruction(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith(("Function:", "....")) and not stripped.endswith(":")


def _target_source_matches(candidate_path: str | None, source_path: str | None) -> bool:
    if candidate_path is None or source_path is None:
        return False
    try:
        candidate = Path(candidate_path).name
        target = Path(source_path).name
        return candidate == target or candidate_path == source_path
    except Exception:
        return False


def _build_loc_mapped_triview_lines(
    source_text: str | None,
    source_path: str | None,
    ptx_text: str,
    sass_text: str | None,
    max_lines: int = 160,
) -> list[TriViewLine]:
    source_lines = source_text.splitlines() if source_text else []
    ptx_lines = ptx_text.splitlines()
    sass_lines = sass_text.splitlines() if sass_text else []
    sass_instruction_lines = [(index + 1, line) for index, line in enumerate(sass_lines) if _is_sass_instruction(line)]
    file_map: dict[int, str] = {}
    loc_entries: list[tuple[int, int, str | None, int | None, str | None]] = []

    for index, line in enumerate(ptx_lines):
        file_match = _PTX_FILE_RE.match(line)
        if file_match:
            file_map[int(file_match.group("index"))] = file_match.group("path")
            continue
        loc_match = _PTX_LOC_RE.match(line)
        if not loc_match:
            continue
        file_idx = int(loc_match.group("file"))
        source_line = int(loc_match.group("line"))
        mapped_path = file_map.get(file_idx)
        ptx_line_number: int | None = None
        ptx_text_value: str | None = None
        for follow_index in range(index + 1, len(ptx_lines)):
            candidate = ptx_lines[follow_index]
            if _PTX_LOC_RE.match(candidate):
                break
            if _is_ptx_instruction(candidate):
                ptx_line_number = follow_index + 1
                ptx_text_value = candidate
                break
        loc_entries.append((source_line, index + 1, mapped_path, ptx_line_number, ptx_text_value))

    filtered = [entry for entry in loc_entries if _target_source_matches(entry[2], source_path)]
    entries = filtered or loc_entries
    if not entries:
        return []

    lines: list[TriViewLine] = []
    for ordinal, (source_line_no, _loc_line_no, mapped_path, ptx_line_no, ptx_line_text) in enumerate(entries[:max_lines]):
        source_text_value = source_lines[source_line_no - 1] if 1 <= source_line_no <= len(source_lines) else None
        sass_line_no: int | None = None
        sass_text_value: str | None = None
        if ordinal < len(sass_instruction_lines):
            sass_line_no, sass_text_value = sass_instruction_lines[ordinal]
        anchors: list[str] = []
        if source_text_value:
            anchors.extend(_infer_anchor_tokens(source_text_value))
        if ptx_line_text:
            anchors.extend(_infer_anchor_tokens(ptx_line_text))
        if sass_text_value:
            anchors.extend(_infer_anchor_tokens(sass_text_value))
        if mapped_path:
            anchors.append(Path(mapped_path).name)
        lines.append(
            TriViewLine(
                source_line=source_line_no,
                ptx_line=ptx_line_no,
                sass_line=sass_line_no,
                source_text=source_text_value,
                ptx_text=ptx_line_text,
                sass_text=sass_text_value,
                anchors=list(dict.fromkeys(anchors))[:6],
            )
        )
    return lines


def _build_triview_lines(
    source_text: str | None,
    source_path: str | None,
    ptx_text: str | None,
    sass_text: str | None,
    max_lines: int = 120,
) -> list[TriViewLine]:
    if ptx_text:
        loc_mapped = _build_loc_mapped_triview_lines(source_text, source_path, ptx_text, sass_text, max_lines=max_lines)
        if loc_mapped:
            return loc_mapped
    source_lines = source_text.splitlines() if source_text else []
    ptx_lines = [line for line in (ptx_text.splitlines() if ptx_text else []) if line.strip()]
    sass_lines = [line for line in (sass_text.splitlines() if sass_text else []) if line.strip()]
    limit = min(max(len(source_lines), len(ptx_lines), len(sass_lines), 1), max_lines)
    lines: list[TriViewLine] = []
    source_cursor = 0
    ptx_cursor = 0
    sass_cursor = 0
    for index in range(limit):
        source_line = source_lines[source_cursor] if source_cursor < len(source_lines) else None
        ptx_line = ptx_lines[ptx_cursor] if ptx_cursor < len(ptx_lines) else None
        sass_line = sass_lines[sass_cursor] if sass_cursor < len(sass_lines) else None
        anchors: list[str] = []
        if source_line:
            anchors.extend(_infer_anchor_tokens(source_line))
        if ptx_line:
            anchors.extend(_infer_anchor_tokens(ptx_line))
        if sass_line:
            anchors.extend(_infer_anchor_tokens(sass_line))
        deduped_anchors = list(dict.fromkeys(anchors))[:5]
        lines.append(
            TriViewLine(
                source_line=source_cursor + 1 if source_line is not None else None,
                ptx_line=ptx_cursor + 1 if ptx_line is not None else None,
                sass_line=sass_cursor + 1 if sass_line is not None else None,
                source_text=source_line,
                ptx_text=ptx_line,
                sass_text=sass_line,
                anchors=deduped_anchors,
            )
        )
        if source_cursor < len(source_lines):
            source_cursor += 1
        if ptx_cursor < len(ptx_lines):
            ptx_cursor += 1
        if sass_cursor < len(sass_lines):
            sass_cursor += 1
    return lines


def _build_source_map_summary(tri_view: TriViewArtifact) -> dict[str, object]:
    source_lines = [entry.source_line for entry in tri_view.lines if entry.source_line is not None]
    ptx_lines = [entry.ptx_line for entry in tri_view.lines if entry.ptx_line is not None]
    sass_lines = [entry.sass_line for entry in tri_view.lines if entry.sass_line is not None]
    source_spans: list[dict[str, int]] = []
    if source_lines:
        start = source_lines[0]
        end = source_lines[0]
        row_count = 1
        for line_no in source_lines[1:]:
            if line_no == end or line_no == end + 1:
                end = line_no
                row_count += 1
            else:
                source_spans.append({"start_line": start, "end_line": end, "row_count": row_count})
                start = line_no
                end = line_no
                row_count = 1
        source_spans.append({"start_line": start, "end_line": end, "row_count": row_count})
    return {
        "backend": tri_view.backend,
        "correlation_method": tri_view.correlation_method,
        "source_path": tri_view.source_path,
        "line_count": len(tri_view.lines),
        "unique_source_lines": len(set(source_lines)),
        "mapped_ptx_lines": len(ptx_lines),
        "mapped_sass_lines": len(sass_lines),
        "source_spans": source_spans[:24],
        "warnings": tri_view.warnings,
    }


def emit_disassembly_bundle(
    *,
    writer: RunBundleWriter,
    source_text: str | None,
    source_path: str | None = None,
    ptx_text: str | None,
    sass_text: str | None,
    ttir_text: str | None = None,
    ttgir_text: str | None = None,
    llir_text: str | None = None,
    compiler: str,
    compiler_version: str,
    extra_metadata: dict[str, object] | None = None,
    stdout_text: str = "",
    stderr_text: str = "",
    warnings: list[str] | None = None,
    producer_event_id: str | None = None,
) -> BuildRecord:
    started = time.monotonic()
    warnings = list(warnings or [])
    if source_text is not None:
        source_ref = writer.write_artifact(
            relative_path="build/source.txt",
            kind="source_text",
            content=source_text,
            mime="text/plain",
            semantic_tags=["build", "source"],
            producer_event_id=producer_event_id,
        ).path
    else:
        source_ref = None
        warnings.append("no source file was provided or inferred")

    ptx_ref = None
    if ptx_text is not None:
        ptx_ref = writer.write_artifact(
            relative_path="build/ptx.txt",
            kind="ptx_text",
            content=ptx_text,
            mime="text/plain",
            semantic_tags=["build", "ptx"],
            producer_event_id=producer_event_id,
        ).path
    else:
        warnings.append("no PTX content was provided or extracted")

    sass_ref = None
    if sass_text is not None:
        sass_ref = writer.write_artifact(
            relative_path="build/sass.txt",
            kind="sass_text",
            content=sass_text,
            mime="text/plain",
            semantic_tags=["build", "sass"],
            producer_event_id=producer_event_id,
        ).path
    else:
        warnings.append("no SASS content was provided or extracted")

    ttir_ref = None
    if ttir_text is not None:
        ttir_ref = writer.write_artifact(
            relative_path="build/ttir.mlir",
            kind="ttir_text",
            content=ttir_text,
            mime="text/plain",
            semantic_tags=["build", "triton", "ttir"],
            producer_event_id=producer_event_id,
        ).path

    ttgir_ref = None
    if ttgir_text is not None:
        ttgir_ref = writer.write_artifact(
            relative_path="build/ttgir.mlir",
            kind="ttgir_text",
            content=ttgir_text,
            mime="text/plain",
            semantic_tags=["build", "triton", "ttgir"],
            producer_event_id=producer_event_id,
        ).path

    llir_ref = None
    if llir_text is not None:
        llir_ref = writer.write_artifact(
            relative_path="build/llir.ll",
            kind="llir_text",
            content=llir_text,
            mime="text/plain",
            semantic_tags=["build", "triton", "llir"],
            producer_event_id=producer_event_id,
        ).path

    stdout_ref = None
    if stdout_text:
        stdout_ref = writer.write_artifact(
            relative_path="build/disassembly_stdout.txt",
            kind="disassembly_stdout",
            content=stdout_text,
            mime="text/plain",
            semantic_tags=["build", "stdout"],
            producer_event_id=producer_event_id,
        ).path
    stderr_ref = None
    if stderr_text:
        stderr_ref = writer.write_artifact(
            relative_path="build/disassembly_stderr.txt",
            kind="disassembly_stderr",
            content=stderr_text,
            mime="text/plain",
            semantic_tags=["build", "stderr"],
            producer_event_id=producer_event_id,
        ).path

    tri_view = TriViewArtifact(
        backend="nvidia_disassembly" if compiler != "triton" else "triton_nvidia",
        correlation_method="ptx_loc_source_map_v1" if ptx_text and _build_loc_mapped_triview_lines(source_text, source_path, ptx_text, sass_text, max_lines=4) else "heuristic_anchor_alignment_v2",
        source_path=source_path,
        source_ref=source_ref,
        ttir_ref=ttir_ref,
        ttgir_ref=ttgir_ref,
        llir_ref=llir_ref,
        ptx_ref=ptx_ref,
        sass_ref=sass_ref,
        lines=_build_triview_lines(source_text, source_path, ptx_text, sass_text),
        warnings=warnings,
    )
    line_map_ref = writer.write_artifact(
        relative_path="build/source_ptx_sass_map.json",
        kind="source_ptx_sass_map",
        content=json.dumps(tri_view.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["build", "triview", "map"],
        producer_event_id=producer_event_id,
    ).path
    tri_view.line_map_ref = line_map_ref
    writer.write_artifact(
        relative_path="build/tri_view.json",
        kind="tri_view",
        content=json.dumps(tri_view.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["build", "triview"],
        producer_event_id=producer_event_id,
    )
    writer.write_artifact(
        relative_path="build/source_map_summary.json",
        kind="source_map_summary",
        content=json.dumps(_build_source_map_summary(tri_view), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["build", "triview", "summary"],
        producer_event_id=producer_event_id,
    )
    markdown = "\n".join(
        [
            "# Source / PTX / SASS Tri-View",
            "",
            f"- backend: `{tri_view.backend}`",
            f"- correlation_method: `{tri_view.correlation_method}`",
            f"- source_path: `{tri_view.source_path}`",
            f"- source_ref: `{source_ref}`",
            f"- ttir_ref: `{ttir_ref}`",
            f"- ttgir_ref: `{ttgir_ref}`",
            f"- llir_ref: `{llir_ref}`",
            f"- ptx_ref: `{ptx_ref}`",
            f"- sass_ref: `{sass_ref}`",
            "",
            "## Preview",
            *[
                f"- line {entry.source_line or entry.ptx_line or entry.sass_line}: source=`{(entry.source_text or '').strip()[:60]}` ptx=`{(entry.ptx_text or '').strip()[:60]}` sass=`{(entry.sass_text or '').strip()[:60]}`"
                for entry in tri_view.lines[:10]
            ],
            "",
            "## Warnings",
            *([f"- {warning}" for warning in warnings] or ["- none"]),
        ]
    )
    writer.write_artifact(
        relative_path="build/tri_view.md",
        kind="tri_view_markdown",
        content=markdown + "\n",
        mime="text/markdown",
        semantic_tags=["build", "triview", "markdown"],
        producer_event_id=producer_event_id,
    )

    digest_source = "\n".join(filter(None, [source_text, ptx_text, sass_text])).encode("utf-8")
    build_record = BuildRecord(
        compiler=compiler,
        compiler_version=compiler_version,
        status="ok" if ptx_ref is not None and sass_ref is not None else "partial",
        duration_ms=int((time.monotonic() - started) * 1000),
        stdout_ref=stdout_ref,
        stderr_ref=stderr_ref,
        ptx_ref=ptx_ref,
        sass_ref=sass_ref,
        binary_hash=hashlib.sha256(digest_source).hexdigest() if digest_source else None,
    )
    writer.write_artifact(
        relative_path="build/build_record.json",
        kind="build_record",
        content=json.dumps(
            {
                **build_record.model_dump(mode="json"),
                "extra_metadata": extra_metadata or {},
            },
            indent=2,
        )
        + "\n",
        mime="application/json",
        semantic_tags=["build", "record"],
        producer_event_id=producer_event_id,
    )
    return build_record


def emit_disassembly_nvidia(
    writer: RunBundleWriter,
    command: list[str],
    *,
    source_file: str | None = None,
    binary_file: str | None = None,
    ptx_file: str | None = None,
    sass_file: str | None = None,
    executor: CommandExecutor | None = None,
) -> BuildRecord:
    executor = executor or LocalHostToolExecutor()
    started_event = writer.append_event(
        scope="tool.emit_disassembly_nvidia",
        kind="started",
        payload={
            "command": command,
            "source_file": source_file,
            "binary_file": binary_file,
            "ptx_file": ptx_file,
            "sass_file": sass_file,
        },
    )
    started = time.monotonic()

    resolved_source = _infer_source_path(command, source_file)
    resolved_ptx = Path(ptx_file) if ptx_file else None
    resolved_sass = Path(sass_file) if sass_file else None
    resolved_binary = Path(binary_file) if binary_file else None

    warnings: list[str] = []
    ptx_text = _read_text(resolved_ptx)
    sass_text = _read_text(resolved_sass)
    source_text = _read_text(resolved_source)

    compiler = "manual"
    compiler_version = "unknown"
    stdout_text = ""
    stderr_text = ""

    if ptx_text is None and resolved_binary is not None:
        if shutil.which("cuobjdump") is not None:
            compiler = "cuobjdump"
            result = executor.run(["cuobjdump", "--dump-ptx", str(resolved_binary)])
            stdout_text += result.stdout
            stderr_text += result.stderr
            if result.exit_code == 0 and result.stdout.strip():
                ptx_text = result.stdout
            else:
                warnings.append("cuobjdump PTX extraction failed")
        else:
            warnings.append("cuobjdump is not installed.")

    if sass_text is None and resolved_binary is not None:
        if shutil.which("nvdisasm") is not None:
            compiler = "nvdisasm"
            result = executor.run(["nvdisasm", str(resolved_binary)])
            stdout_text += result.stdout
            stderr_text += result.stderr
            if result.exit_code == 0 and result.stdout.strip():
                sass_text = result.stdout
            else:
                warnings.append("nvdisasm SASS extraction failed")
        elif shutil.which("cuobjdump") is not None:
            compiler = "cuobjdump"
            result = executor.run(["cuobjdump", "--dump-sass", str(resolved_binary)])
            stdout_text += result.stdout
            stderr_text += result.stderr
            if result.exit_code == 0 and result.stdout.strip():
                sass_text = result.stdout
            else:
                warnings.append("cuobjdump SASS extraction failed")
        else:
            warnings.append("nvdisasm and cuobjdump are not installed.")

    build_record = emit_disassembly_bundle(
        writer=writer,
        source_text=source_text,
        source_path=str(resolved_source.resolve()) if resolved_source is not None else None,
        ptx_text=ptx_text,
        sass_text=sass_text,
        compiler=compiler,
        compiler_version=compiler_version,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        warnings=warnings,
        producer_event_id=started_event.event_id,
    )
    build_record.duration_ms = int((time.monotonic() - started) * 1000)
    writer.write_artifact(
        relative_path="build/build_record.json",
        kind="build_record",
        content=json.dumps(build_record.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["build", "record"],
        producer_event_id=started_event.event_id,
    )
    writer.append_event(
        scope="tool.emit_disassembly_nvidia",
        kind="completed" if build_record.status == "ok" else "failed",
        payload={"status": build_record.status, "duration_ms": build_record.duration_ms},
    )
    return build_record
