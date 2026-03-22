from __future__ import annotations

import importlib.util
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from gpu_cockpit.contracts import BuildRecord
from gpu_cockpit.engine.run_bundle import RunBundleWriter


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Triton module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_build_spec_ref(build_spec_ref: str) -> tuple[Path, str]:
    if ":" in build_spec_ref:
        module_ref, func_name = build_spec_ref.split(":", 1)
    else:
        module_ref, func_name = build_spec_ref, "get_build_spec"
    return Path(module_ref), func_name


def compile_triton_build_spec(writer: RunBundleWriter, root: Path, build_spec_ref: str) -> BuildRecord:
    from gpu_cockpit.backends.nvidia.disassembly import emit_disassembly_bundle
    import triton

    module_path, func_name = _parse_build_spec_ref(build_spec_ref)
    if not module_path.is_absolute():
        module_path = (root / module_path).resolve()
    module = _load_module(module_path)
    if not hasattr(module, func_name):
        raise RuntimeError(f"Build spec function `{func_name}` not found in {module_path}")

    started_event = writer.append_event(
        scope="tool.compile_triton_build_spec",
        kind="started",
        payload={"build_spec_ref": build_spec_ref},
    )
    started = time.monotonic()
    build_spec = getattr(module, func_name)()
    kernel = build_spec["kernel"]
    warmup_args = tuple(build_spec["warmup_args"])
    grid = tuple(build_spec["grid"])
    kwargs = dict(build_spec.get("kwargs", {}))
    source_path = Path(build_spec.get("source_file", module_path))
    if not source_path.is_absolute():
        source_path = (root / source_path).resolve()

    compiled = kernel.warmup(*warmup_args, grid=grid, **kwargs)
    asm = compiled.asm
    sass_text = None
    warnings: list[str] = []
    try:
        sass_value = asm["sass"]
        sass_text = sass_value.decode("utf-8", errors="replace") if isinstance(sass_value, bytes) else str(sass_value)
    except KeyError:
        sass_text = None
    except Exception as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            warnings.append(f"sass extraction unavailable: {' '.join(exc.cmd)} exited with code {exc.returncode}")
        else:
            warnings.append(f"sass extraction unavailable: {exc}")
        sass_text = None
    build_record = emit_disassembly_bundle(
        writer=writer,
        source_text=source_path.read_text(encoding="utf-8"),
        source_path=str(source_path),
        ptx_text=str(asm["ptx"]) if "ptx" in asm else None,
        sass_text=sass_text,
        ttir_text=str(asm["ttir"]) if "ttir" in asm else None,
        ttgir_text=str(asm["ttgir"]) if "ttgir" in asm else None,
        llir_text=str(asm["llir"]) if "llir" in asm else None,
        compiler="triton",
        compiler_version=getattr(module, "__triton_version__", triton.__version__),
        extra_metadata={
            "kernel_name": getattr(compiled, "name", getattr(kernel, "__name__", "unknown_kernel")),
            "build_spec_ref": build_spec_ref,
            "grid": list(grid),
            "kwargs": kwargs,
        },
        producer_event_id=started_event.event_id,
    )
    writer.write_artifact(
        relative_path="build/triton_compile_metadata.json",
        kind="triton_compile_metadata",
        content=json.dumps(
            {
                "kernel_name": getattr(compiled, "name", getattr(kernel, "__name__", "unknown_kernel")),
                "grid": list(grid),
                "kwargs": kwargs,
                "metadata": dict(compiled.metadata._asdict()),
                "warnings": warnings,
            },
            default=str,
            indent=2,
        )
        + "\n",
        mime="application/json",
        semantic_tags=["build", "triton", "metadata"],
        producer_event_id=started_event.event_id,
    )
    writer.append_event(
        scope="tool.compile_triton_build_spec",
        kind="completed",
        payload={
            "duration_ms": int((time.monotonic() - started) * 1000),
            "build_spec_ref": build_spec_ref,
            "warnings": warnings,
        },
    )
    return build_record
