from __future__ import annotations

import platform
import re
import shutil
import sys
from typing import Iterable

from gpu_cockpit.contracts import DoctorReport, HardwareFingerprint
from gpu_cockpit.contracts.common import ToolVersionSet
from gpu_cockpit.contracts.doctor import ToolStatus
from gpu_cockpit.executors import LocalHostToolExecutor


TOOL_COMMANDS: tuple[str, ...] = (
    "nvidia-smi",
    "nsys",
    "ncu",
    "compute-sanitizer",
    "cuobjdump",
    "nvdisasm",
    "nvcc",
    "rocminfo",
    "rocm-smi",
    "hipcc",
    "rocprof",
    "rocprofv3",
)


def _run_command(args: list[str]) -> str | None:
    executor = LocalHostToolExecutor()
    try:
        result = executor.run(args, timeout=5)
    except OSError:
        return None
    except Exception:
        return None
    output = result.stdout.strip() or result.stderr.strip()
    return output or None


def _tool_version(command: str) -> str | None:
    version_flags: dict[str, list[str]] = {
        "nvidia-smi": [command, "--query-gpu=driver_version", "--format=csv,noheader"],
        "nsys": [command, "--version"],
        "ncu": [command, "--version"],
        "compute-sanitizer": [command, "--version"],
        "cuobjdump": [command, "--version"],
        "nvdisasm": [command, "--version"],
        "nvcc": [command, "--version"],
        "rocminfo": [command],
        "rocm-smi": [command, "--showproductname"],
        "hipcc": [command, "--version"],
        "rocprof": [command, "--version"],
        "rocprofv3": [command, "--version"],
    }
    raw = _run_command(version_flags[command])
    if raw is None:
        return None
    return raw.splitlines()[0]


def _parse_rocm_runtime_version(tool_statuses: list[ToolStatus]) -> str:
    for name in ("hipcc", "rocminfo", "rocprof", "rocprofv3"):
        status = next((item for item in tool_statuses if item.name == name and item.available and item.version), None)
        if status is None or status.version is None:
            continue
        match = re.search(r"([0-9]+\.[0-9]+(?:\.[0-9]+)?)", status.version)
        if match:
            return match.group(1)
    return "unknown"


def _parse_rocm_driver_version(tool_statuses: list[ToolStatus]) -> str:
    status = next((item for item in tool_statuses if item.name == "rocm-smi" and item.available and item.version), None)
    if status is None or status.version is None:
        return "unknown"
    match = re.search(r"([0-9]+\.[0-9]+(?:\.[0-9]+)?)", status.version)
    if match:
        return match.group(1)
    return status.version


def _collect_tool_statuses(commands: Iterable[str]) -> list[ToolStatus]:
    statuses: list[ToolStatus] = []
    for command in commands:
        path = shutil.which(command)
        statuses.append(
            ToolStatus(
                name=command,
                path=path,
                version=_tool_version(command) if path else None,
                available=path is not None,
            )
        )
    return statuses


def _parse_rocminfo_devices(info: str) -> list[dict[str, object]]:
    if not info.strip():
        return []
    devices: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw_line in info.splitlines():
        line = raw_line.rstrip()
        if re.match(r"^\*+\s+Agent\s+\d+\s+\*+$", line):
            if current and str(current.get("device_type", "")).upper() == "GPU":
                devices.append(current)
            current = {}
            continue
        if current is None:
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        current[normalized_key] = value.strip()
    if current and str(current.get("device_type", "")).upper() == "GPU":
        devices.append(current)
    return devices


def _parse_rocm_smi_power_limits(info: str) -> dict[str, int]:
    limits: dict[str, int] = {}
    for line in info.splitlines():
        match = re.search(r"(?:card|GPU\[)(?P<index>\d+)\]? .*?([Pp]ower|Watt).*?(?P<watts>\d+)\s*W", line)
        if match:
            limits[f"card{match.group('index')}"] = int(match.group("watts"))
    return limits


def _collect_nvidia_fingerprints(tool_statuses: list[ToolStatus]) -> list[HardwareFingerprint]:
    if shutil.which("nvidia-smi") is None:
        return []

    query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,power.limit",
            "--format=csv,noheader,nounits",
        ]
    )
    if query is None:
        return []

    tool_versions = {
        status.name: status.version
        for status in tool_statuses
        if status.available
        and status.version is not None
        and status.name in {"nvidia-smi", "nsys", "ncu", "compute-sanitizer", "cuobjdump", "nvdisasm", "nvcc"}
    }

    fingerprints: list[HardwareFingerprint] = []
    for index, line in enumerate(query.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_name, driver_version, memory_total, power_limit = parts[:4]
        fingerprints.append(
            HardwareFingerprint(
                vendor="nvidia",
                gpu_name=gpu_name,
                arch=f"gpu_{index}",
                driver_version=driver_version,
                runtime_version=tool_versions.get("nvcc") or "unknown",
                memory_gb=int(float(memory_total) / 1024),
                power_limit_w=int(float(power_limit)) if power_limit else None,
                clock_state="unknown",
                mig_mode=False,
                mps_mode=False,
                host_kernel=platform.release(),
                container_runtime="docker" if shutil.which("docker") else "unknown",
                tool_versions=ToolVersionSet(versions=tool_versions),
            )
        )
    return fingerprints


def _collect_amd_fingerprints(tool_statuses: list[ToolStatus]) -> list[HardwareFingerprint]:
    if shutil.which("rocminfo") is None:
        return []
    info = _run_command(["rocminfo"])
    if info is None:
        return []
    rocm_smi_info = _run_command(["rocm-smi", "--showproductname", "--showpower"]) if shutil.which("rocm-smi") else None
    tool_versions = {
        status.name: status.version
        for status in tool_statuses
        if status.available
        and status.version is not None
        and status.name in {"rocminfo", "rocm-smi", "hipcc", "rocprof", "rocprofv3"}
    }
    power_limits = _parse_rocm_smi_power_limits(rocm_smi_info or "")
    runtime_version = _parse_rocm_runtime_version(tool_statuses)
    driver_version = _parse_rocm_driver_version(tool_statuses)
    fingerprints: list[HardwareFingerprint] = []
    for index, device in enumerate(_parse_rocminfo_devices(info)):
        arch = str(device.get("name") or "unknown")
        gpu_name = str(device.get("marketing_name") or arch or "unknown_amd_gpu")
        memory_bytes = 0
        for key in ("global_memory_size", "size"):
            value = device.get(key)
            if isinstance(value, str):
                numeric = re.search(r"([0-9]+)", value.replace(",", ""))
                if numeric:
                    memory_bytes = max(memory_bytes, int(numeric.group(1)))
        memory_gb = int(memory_bytes / (1024**3)) if memory_bytes else 0
        card_key = f"card{index}"
        fingerprints.append(
            HardwareFingerprint(
                vendor="amd",
                gpu_name=gpu_name,
                arch=arch,
                driver_version=driver_version,
                runtime_version=runtime_version,
                memory_gb=memory_gb,
                power_limit_w=power_limits.get(card_key),
                clock_state="unknown",
                mig_mode=False,
                mps_mode=False,
                host_kernel=platform.release(),
                container_runtime="docker" if shutil.which("docker") else "unknown",
                tool_versions=ToolVersionSet(versions=tool_versions),
            )
        )
    return fingerprints


def collect_doctor_report() -> DoctorReport:
    tool_statuses = _collect_tool_statuses(TOOL_COMMANDS)
    warnings: list[str] = []
    if not any(status.name == "ncu" and status.available for status in tool_statuses):
        warnings.append("Nsight Compute CLI not found; kernel-level NVIDIA profiling is unavailable.")
    if not any(status.name == "compute-sanitizer" and status.available for status in tool_statuses):
        warnings.append("Compute Sanitizer not found; sanitizer workflows are unavailable.")
    if not any(status.name in {"cuobjdump", "nvdisasm"} and status.available for status in tool_statuses):
        warnings.append("cuobjdump/nvdisasm not found; NVIDIA disassembly extraction is unavailable.")
    if not any(status.name == "rocminfo" and status.available for status in tool_statuses):
        warnings.append("ROCm tooling not found; AMD workflows are currently unavailable.")
    if not any(status.name in {"rocprof", "rocprofv3"} and status.available for status in tool_statuses):
        warnings.append("rocprof not found; AMD trace/profile workflows are unavailable.")

    return DoctorReport(
        host_platform=platform.platform(),
        host_kernel=platform.release(),
        python_executable=sys.executable,
        python_version=platform.python_version(),
        container_runtime="docker" if shutil.which("docker") else None,
        available_tools=tool_statuses,
        hardware_fingerprints=_collect_nvidia_fingerprints(tool_statuses) + _collect_amd_fingerprints(tool_statuses),
        warnings=warnings,
    )
