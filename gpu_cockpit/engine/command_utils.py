from __future__ import annotations

from pathlib import Path
import sys


def normalize_python_command(command: list[str]) -> list[str]:
    if not command:
        return []
    executable = str(command[0]).strip()
    if executable in {"python", "python3"}:
        return [sys.executable, *command[1:]]
    return list(command)


def local_python_build_env(cwd: Path | None) -> dict[str, str]:
    if cwd is None:
        return {}
    for search_root in [cwd, *cwd.parents]:
        include_root = search_root / ".local_pkgs" / "python312dev" / "extracted" / "usr" / "include"
        python_include = include_root / "python3.12" / "Python.h"
        if python_include.exists():
            include_paths = [str(include_root / "python3.12"), str(include_root)]
            return {"C_INCLUDE_PATH": ":".join(include_paths)}
    return {}
