from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.training import load_training_config


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)


def _systemctl_show(unit: str, properties: list[str]) -> dict[str, str]:
    command = ["systemctl", "--user", "show", unit, *[f"-p{item}" for item in properties]]
    result = _run_command(command)
    rows: dict[str, str] = {"_returncode": str(result.returncode)}
    if result.stdout:
        for line in result.stdout.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            rows[key] = value
    if result.stderr:
        rows["_stderr"] = result.stderr.strip()
    return rows


def _journal_tail(unit: str, lines: int) -> list[str]:
    result = _run_command(["journalctl", "--user", "-u", unit, "--no-pager", "-n", str(lines)])
    if result.returncode != 0:
        return [f"journalctl_failed: {result.stderr.strip()}"]
    return result.stdout.splitlines()


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _parse_systemd_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%a %Y-%m-%d %H:%M:%S %Z", "%a %Y-%m-%d %H:%M:%S %z"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _iter_output_files(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not out_dir.exists():
        return rows
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append(
            {
                "path": _relative_to_root(path),
                "size_bytes": stat.st_size,
                "modified_at_epoch": round(stat.st_mtime, 3),
            }
        )
    return rows


def build_archive_payload(
    *,
    unit: str,
    config_path: Path,
    journal_lines: int,
) -> dict[str, Any]:
    config = load_training_config(config_path)
    out_dir = ROOT / config.output_dir
    training_report_path = out_dir / "training_run_report.json"
    show = _systemctl_show(
        unit,
        [
            "Id",
            "ActiveState",
            "SubState",
            "Result",
            "ExecMainPID",
            "ExecMainCode",
            "ExecMainStatus",
            "MainPID",
            "MemoryCurrent",
            "MemoryPeak",
            "CPUUsageNSec",
            "ActiveEnterTimestamp",
            "ActiveExitTimestamp",
            "InactiveEnterTimestamp",
            "ExecMainStartTimestamp",
            "ExecMainExitTimestamp",
        ],
    )
    training_report = _read_json(training_report_path)
    training_report_mtime_epoch = training_report_path.stat().st_mtime if training_report_path.exists() else None
    unit_start = _parse_systemd_timestamp(show.get("ExecMainStartTimestamp") or show.get("ActiveEnterTimestamp"))
    report_finished_at = None if training_report is None else training_report.get("finished_at")
    report_finished_dt = None
    if isinstance(report_finished_at, str):
        try:
            report_finished_dt = datetime.fromisoformat(report_finished_at)
        except ValueError:
            report_finished_dt = None
    training_report_fresh_for_unit = None
    if unit_start is not None and report_finished_dt is not None:
        training_report_fresh_for_unit = report_finished_dt.replace(tzinfo=None) >= unit_start.replace(tzinfo=None)
    payload: dict[str, Any] = {
        "unit": unit,
        "config_path": _relative_to_root(config_path),
        "config_id": config.config_id,
        "effective_model_id": config.model_id,
        "effective_tokenizer_id": config.tokenizer_id or config.model_id,
        "output_dir": _relative_to_root(out_dir),
        "training_report_path": _relative_to_root(training_report_path),
        "training_report_exists": training_report is not None,
        "training_report_mtime_epoch": training_report_mtime_epoch,
        "training_report_status": None if training_report is None else training_report.get("status"),
        "training_report_started_at": None if training_report is None else training_report.get("started_at"),
        "training_report_finished_at": None if training_report is None else training_report.get("finished_at"),
        "training_report_fresh_for_unit": training_report_fresh_for_unit,
        "systemd_show": show,
        "journal_tail": _journal_tail(unit, journal_lines),
        "output_files": _iter_output_files(out_dir),
    }
    if training_report is not None:
        payload["training_report"] = training_report
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive a Spark smoke run from a systemd user unit.")
    parser.add_argument("--unit", required=True, help="systemd --user unit name")
    parser.add_argument("--config", required=True, type=Path, help="SFT config path")
    parser.add_argument("--out", required=True, type=Path, help="Destination JSON path")
    parser.add_argument("--journal-lines", type=int, default=120, help="Journal lines to capture")
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    out_path = args.out
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()

    payload = build_archive_payload(unit=args.unit, config_path=config_path, journal_lines=args.journal_lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(_relative_to_root(out_path))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
