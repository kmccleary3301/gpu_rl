# KernelBench-v3 Integration Note

This note freezes the first stable KernelBench-v3 slice in the environment.

## Current slice

- official held-out case:
  - `case/kernelbench_v3/level1/23_softmax_official/v3_1`
  - task: `task/kernelbench_v3/level1/23_softmax_official/eval/v1`
- curated derivative:
  - `case/kernelbench_v3/level1/23_softmax_wide/v3_1`
  - task: `task/kernelbench_v3/level1/23_softmax_wide/eval/v1`

## Provenance rules

- official and curated variants share the same adapter family: `kernelbench_v3`
- official-vs-curated provenance is preserved in:
  - case metadata
  - task ids
  - hard-freeze episode metadata
  - coverage and contamination reports
- held-out status is preserved explicitly:
  - official case: `held_out`
  - curated derivative: `trainable`

## Environment stance

- KernelBench-v3 is a held-out benchmark pillar and task-source library, not the whole curriculum
- official traces can appear in dev/held-out evaluation corpora
- curated derivatives can appear in trainable hard slices
- official traces must not silently enter train splits

## Current evidence

- the official held-out task passes end to end in evaluator coverage
- GPT-5.4 teacher episodes exist for both the official held-out case and the curated derivative under:
  - `artifacts/baselines/gpt54_phase5_kbv3_teacher_slice_v1`
- the canonical hard freeze v2 records both:
  - `kbv3_curated_positive`
  - `kbv3_official_positive`
