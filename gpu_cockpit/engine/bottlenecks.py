from __future__ import annotations

import json

from gpu_cockpit.contracts import BottleneckCard, BottleneckEvidence, ProfileReport, SanitizerReport
from gpu_cockpit.engine.run_bundle import RunBundleWriter


def build_bottleneck_card(
    writer: RunBundleWriter,
    profile_report: ProfileReport,
    sanitizer_report: SanitizerReport | None = None,
) -> BottleneckCard:
    evidence: list[BottleneckEvidence] = []
    next_actions: list[str] = []
    subject = profile_report.kernel_name
    primary_bottleneck = profile_report.classification
    confidence = 0.6
    why = f"Primary kernel `{profile_report.kernel_name}` classified as `{profile_report.classification}` from normalized profiler metrics."

    if sanitizer_report is not None and not sanitizer_report.passed:
        primary_bottleneck = "correctness_safety"
        confidence = 0.95
        why = "Compute Sanitizer reported correctness-affecting findings; performance tuning should wait until these are resolved."
        next_actions.extend(
            [
                "Fix sanitizer-reported memory or synchronization errors before trusting benchmark or profiler output.",
                "Re-run profiling after sanitizer is clean to confirm the bottleneck classification still holds.",
            ]
        )
        evidence.append(
            BottleneckEvidence(
                type="sanitizer_error_count",
                value=sanitizer_report.error_count,
                artifact_ref=f"sanitize/{sanitizer_report.tool}_summary.json",
            )
        )
    else:
        if profile_report.occupancy is not None:
            evidence.append(BottleneckEvidence(type="occupancy", value=profile_report.occupancy, artifact_ref="profiles/kernel/summary.json"))
        if profile_report.dram_throughput_pct_peak is not None:
            evidence.append(
                BottleneckEvidence(
                    type="dram_pct_peak",
                    value=profile_report.dram_throughput_pct_peak,
                    artifact_ref="profiles/kernel/summary.json",
                )
            )
        if profile_report.registers_per_thread is not None:
            evidence.append(
                BottleneckEvidence(
                    type="registers_per_thread",
                    value=profile_report.registers_per_thread,
                    artifact_ref="profiles/kernel/summary.json",
                )
            )
        if profile_report.classification == "memory_bound":
            confidence = 0.8
            next_actions.extend(
                [
                    "Inspect memory access coalescing and L2 hit rate before changing launch geometry.",
                    "Use the raw profile CSV to identify whether DRAM throughput or cache hit rate dominates the stall picture.",
                ]
            )
        elif profile_report.classification == "occupancy_limited":
            confidence = 0.8
            next_actions.extend(
                [
                    "Reduce register pressure or shared-memory footprint to raise active warps.",
                    "Compare occupancy limits against launch configuration and per-thread register usage.",
                ]
            )
        elif profile_report.classification == "register_pressure":
            confidence = 0.8
            next_actions.extend(
                [
                    "Inspect local load/store spill metrics and simplify live ranges or tile sizes.",
                    "Check whether unrolling or accumulator count is forcing excessive register allocation.",
                ]
            )
        elif profile_report.classification == "compute_bound":
            confidence = 0.75
            next_actions.extend(
                [
                    "Inspect instruction mix and tensor/FMA utilization before chasing memory counters.",
                    "Compare source and PTX to see whether the generated kernel is using the expected fast path.",
                ]
            )
        else:
            next_actions.append("Use the raw metric dump to narrow the bottleneck before making algorithmic changes.")

    card = BottleneckCard(
        card_id=f"bn_{writer.run_spec.run_id if writer.run_spec else 'pending'}",
        run_id=writer.run_spec.run_id if writer.run_spec else "pending",
        subject=subject,
        primary_bottleneck=primary_bottleneck,
        confidence=confidence,
        evidence=evidence,
        why=why,
        next_actions=next_actions,
    )
    writer.write_artifact(
        relative_path="bottlenecks/primary.json",
        kind="bottleneck_card",
        content=json.dumps(card.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["bottleneck", "summary"],
    )
    return card
