# Phase 5A Hard Trace Freeze v4

## Purpose

Refresh the canonical Phase 5A slice after the harder KV-budget rerun succeeded and the narrow learner repair turn was completed.

This slice is intended to support:

- the next artifact-distill refresh with a real harder KV success in the trainable tranche
- continued teacher-correction and ranking exports off the newest shared freeze
- comparison against the narrow sum-reduction repair turn without treating that learner trace as a canonical teacher win

## Composition

- dataset id: `phase5a_hard_trace_freeze_v4`
- episode count: `22`
- train count: `13`
- dev count: `3`
- analysis count: `6`
- usable positives: `12`
- contamination audit status: `pass`

## What changed from v3

- adds the successful `gpt54_kv_cache_gather_hard_three_attempt_positive_probe_v2` episode into the trainable hard tranche
- preserves the broader control-surface winner rows from freeze v3
- adds the narrow `distill_v5` repair outcomes as analysis-only comparison rows
- keeps the previous GPT-5.4 tranche, KB-v3, and near-miss rows intact
