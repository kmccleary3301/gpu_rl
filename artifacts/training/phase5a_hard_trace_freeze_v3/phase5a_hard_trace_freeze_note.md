# Phase 5A Hard Trace Freeze v3

## Purpose

Freeze the post-controlfix Phase 5A slice with explicit local control-surface winner traces folded into the trainable corpus.

This slice is intended to support:

- artifact-distill refreshes that learn from both GPT-5.4 hard positives and the winning local control policy
- continued teacher-correction and ranking exports off a provenance-clean shared freeze
- clearer separation between broader-tranche control gains and narrower learner gains

## Composition

- dataset id: `phase5a_hard_trace_freeze_v3`
- episode count: `19`
- train count: `12`
- dev count: `3`
- analysis count: `4`
- usable positives: `11`
- contamination audit status: `pass`

## What changed from v2

- preserves the existing GPT-5.4 hard tranche, KB-v3, and near-miss rows from freeze v2
- adds the broader control winner successes from `base_controlfix_v2`
- adds the attention-only control probe success so the tranche has a successful local attention trace
- keeps the previous analysis-only learner rows intact for comparison
