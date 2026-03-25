# Packet Conformance Note

The default hard-optimize packet is now treated as a stable substrate surface.

## Default packet spine

Every hard optimize task should expose these top-level sections:

- `task_card`
- `candidate_brief`
- `candidate_tree_brief`
- `compare_brief`
- `localization_brief`
- `budget_brief`

## Expectations

- `compare_brief` should carry decision-oriented compare content only
- `localization_brief` should normalize failure and perf localization across internal, KernelBench, and KernelBench-v3 tasks
- packet shape should remain bounded and avoid dumping full raw artifacts by default

## Current profile variants

- `v3_current`
- `compare_packet_v1`
- `compare_plus_localization_v1`
- `compare_plus_localization_plus_branch_v1`

The default environment-standard profile remains:

- `compare_plus_localization_plus_branch_v1`

## Regression bar

Adding a new optimize task family is only considered complete when:

- it passes the task-family conformance checklist
- it projects into the default packet spine without bespoke packet surgery
- it preserves provenance and contamination metadata when included in a frozen corpus
