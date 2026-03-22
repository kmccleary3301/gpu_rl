#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="$ROOT/.venv/bin"
PYTHON_INCLUDE_ROOT="$ROOT/.local_pkgs/python312dev/extracted/usr/include"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export PATH="$VENV_BIN:$PATH"
export C_INCLUDE_PATH="$PYTHON_INCLUDE_ROOT/python3.12:$PYTHON_INCLUDE_ROOT${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"

exec "$@"
