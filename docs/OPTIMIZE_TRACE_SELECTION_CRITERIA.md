# Optimize Trace Selection Criteria

This document freezes the current v2 selection criteria for optimize-trace curation.

## Usable Positive

An optimize trace is a usable positive when all of the following are true:

- the episode terminal action is a successful `eval`
- the trace includes at least one candidate-changing step such as `patch_candidate`, `branch_candidate`, `revert_candidate`, or `promote_candidate`
- the trace includes at least one `compare`
- the trace does not contain failed tool calls

## Usable Negative

An optimize trace is a usable negative when all of the following are true:

- the episode does not solve the task
- the terminal reason is one of:
  - `negative_trace_complete`
  - `multi_candidate_negative_complete`
- the trace includes at least one candidate-changing step
- the trace includes at least one `compare`
- the trace does not end because of provider failure or repeated tool failure

## Near-Miss

An optimize trace is a near-miss when all of the following are true:

- the episode does not solve the task
- the trace remains tool-clean
- the trace shows meaningful optimization work rather than immediate drift
- current operational proxy:
  - optimize verb
  - at least two `bench` actions
  - at least two `eval` actions
  - zero failed tool calls

Near-miss traces are currently analysis-first rather than SFT-first.

## Excluded

A trace should be excluded from the current training pool when any of the following are true:

- provider failure prevented a real episode
- repeated tool failure dominates the trace
- no meaningful candidate change occurred on a patch-capable optimize task
- the trace is missing `compare` on a patch-capable optimize task
- a failed optimize trace is missing localized failure evidence where that surface is expected
- the tail is dominated by uncontrolled low-value actions after the useful work is already over

## Notes

- Multi-candidate traces are valuable even when negative, because they teach candidate-tree semantics.
- Near-miss traces remain important for evaluation and future reward modeling even when not yet part of the default SFT-positive pool.
