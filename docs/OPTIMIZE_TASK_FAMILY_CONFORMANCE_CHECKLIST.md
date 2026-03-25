# Optimize Task Family Conformance Checklist

Use this checklist before calling a new task family "integrated" into the candidate-engineering environment.

## Task + Case Surface

- Task spec exists under `workloads/tasks/`.
- Benchmark case spec exists under the relevant `workloads/benchmarks/*_cases/` directory.
- Case metadata clearly labels:
  - source benchmark
  - source release/version
  - official vs curated provenance
  - held-out vs trainable status
  - operator family
  - benchmark level / difficulty
- Baseline command exists and is runnable.
- Reference or candidate wrapper exists and is runnable.

## Candidate-Loop Semantics

- Task works with the default observation spine:
  - `task_card`
  - `candidate_brief`
  - `candidate_tree_brief`
  - `compare_brief`
  - `localization_brief`
  - `budget_brief`
- Candidate state transitions are preserved in replay artifacts.
- Compare packets produce a stable `compare_type`.
- Best-known candidate and endgame recommendation are surfaced when candidate search is relevant.

## Evaluation Semantics

- Visible tests emit structured failures.
- Hidden tests emit structured failures.
- Failure packets include:
  - `code`
  - `fix_family`
  - `likely_next_actions`
  - provenance fields sufficient to localize the failure
- Perf artifacts include:
  - `perf/benchmark.json`
  - `perf/benchmark_protocol.json`
  - `perf/raw_timings.json`
- Benchmark protocol is self-describing:
  - timer / timing method
  - warmups / repeats
  - compile/runtime split policy
  - command digests
  - hardware fingerprint when available

## Provenance Discipline

- Official tasks remain clearly official in prompts, case metadata, and output payloads.
- Curated tasks remain clearly curated in prompts, case metadata, and output payloads.
- Held-out tasks are not silently reused as trainable traces.
- Source repo / commit / problem reference are recorded for external benchmarks.

## Policy-Learning Readiness

- Task can appear in a hard-slice freeze without bespoke per-task JSON surgery.
- Trajectory export includes candidate snapshots and compare snapshots when relevant.
- Task can feed at least one of:
  - artifact-feedback distill
  - teacher-correction
  - narrow RL / RWR
- Governance labels remain separate from learning reward.

## Minimum Validation

- `py_compile` passes on touched files.
- Adapter registry tests pass if a new adapter or case family is introduced.
- At least one evaluator or environment test proves the task runs end to end.
- If the task is meant for the GPT harness, at least one harness-facing test verifies the packet/action surface.
