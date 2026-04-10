from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))

from workloads.reference.kernelbench_v3_reference_runner import build_case_payload


CASE_CONFIG_PATH = (
    HERE.parent
    / "public_benchmarks"
    / "kernelbench_v3"
    / "v3_1"
    / "cases"
    / "level1_023_online_softmax_streaming_harder.json"
).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    args = parser.parse_args()
    payload = build_case_payload(
        CASE_CONFIG_PATH,
        benchmark_repeats=args.benchmark_repeats,
        extra_payload={
            # Keep subprocess output size close to the promoted variants so the
            # perf gate is not dominated by JSON serialization overhead.
            "candidate_stub": {
                "strategy_change": "patchable_curated_kernelbench_v3_online_softmax_streaming_harder_candidate_wrapper",
                "candidate_ref": "workloads/reference/kernelbench_v3_online_softmax_streaming_harder_optimize_patchable_candidate.py",
                "baseline_ref": "workloads/reference/kernelbench_v3_reference_runner.py",
                "case_config_ref": "workloads/public_benchmarks/kernelbench_v3/v3_1/cases/level1_023_online_softmax_streaming_harder.json",
                "supersedes_candidate_ref": None,
            }
        },
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
