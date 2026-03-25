from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))

from workloads.reference.kernelbench_v3_reference_runner import build_case_payload


CASE_CONFIG_PATH = (HERE.parent / "public_benchmarks" / "kernelbench_v3" / "v3_1" / "cases" / "level1_023_softmax_wide.json").resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    args = parser.parse_args()
    print(
        json.dumps(
            build_case_payload(
                CASE_CONFIG_PATH,
                benchmark_repeats=args.benchmark_repeats,
                extra_payload={
                    "optimization_summary": {
                        "strategy_change": "reference_kernelbench_v3_curated_softmax_wide_candidate",
                        "candidate_ref": "workloads/reference/kernelbench_v3_softmax_wide_candidate.py",
                        "baseline_ref": "workloads/reference/kernelbench_v3_softmax_wide_candidate.py",
                        "case_config_ref": "workloads/public_benchmarks/kernelbench_v3/v3_1/cases/level1_023_softmax_wide.json",
                    }
                },
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
