from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def diagnose_profile_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    profile = dict(payload.get("profile_summary", {}))
    classification = str(profile.get("classification", "unknown"))
    occupancy = float(profile.get("occupancy", 0.0) or 0.0)
    dram_pct = float(profile.get("dram_throughput_pct_peak", 0.0) or 0.0)
    registers = int(profile.get("registers_per_thread", 0) or 0)
    kernel_name = str(profile.get("kernel_name", payload.get("subject", "unknown_kernel")))

    if classification == "memory_bound" or (dram_pct >= 70.0 and occupancy < 65.0):
        return {
            "primary_bottleneck": "memory_bound",
            "reason_code": "dram_throughput",
            "evidence_artifact": "profiles/kernel/summary.json",
            "subject": kernel_name,
            "confidence": 0.8,
            "next_action": "inspect_memory_access_and_l2_reuse",
        }
    if classification == "occupancy_limited" or (occupancy < 40.0 and registers >= 64):
        return {
            "primary_bottleneck": "occupancy_limited",
            "reason_code": "low_occupancy",
            "evidence_artifact": "profiles/kernel/summary.json",
            "subject": kernel_name,
            "confidence": 0.8,
            "next_action": "reduce_register_pressure_or_block_size",
        }
    return {
        "primary_bottleneck": classification,
        "reason_code": "reported_classification",
        "evidence_artifact": "profiles/kernel/summary.json",
        "subject": kernel_name,
        "confidence": 0.6,
        "next_action": "inspect_raw_metrics_before_tuning",
    }
