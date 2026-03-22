from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from kernelbench_reference_runner import build_case_payload


CASE_CONFIG_PATH = (HERE.parent / "public_benchmarks" / "kernelbench" / "v0_1" / "cases" / "level1_023_softmax_wide.json").resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    args = parser.parse_args()
    payload = build_case_payload(
        CASE_CONFIG_PATH,
        benchmark_repeats=args.benchmark_repeats,
        extra_payload={
            "optimization_summary": {
                "strategy_change": "promote_curated_kernelbench_softmax_wide_candidate_wrapper",
                "candidate_ref": "workloads/reference/kernelbench_softmax_wide_optimize_candidate.py",
                "baseline_ref": "workloads/reference/kernelbench_reference_runner.py",
                "case_config_ref": "workloads/public_benchmarks/kernelbench/v0_1/cases/level1_023_softmax_wide.json",
            }
        },
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
