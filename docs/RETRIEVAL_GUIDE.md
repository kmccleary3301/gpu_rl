# Retrieval Guide

## Purpose

The local knowledge layer is meant to return a useful mixture of:

- operator docs
- profiler playbooks
- transformation cards
- prior runs
- prior patch-bearing episodes

## Recommended Query Patterns

### Diagnose queries

- query by failure mode
- query by bottleneck class
- query by operator family

Examples:

- `memory bound reduction`
- `occupancy limited attention`
- `register pressure failed repair`

### Debug queries

- query by bug class
- query by patch intent
- query by operator family plus verb

Examples:

- `mask bug debug`
- `stride bug reduction repair`
- `failed repair patch candidate`

### Reformulate queries

- query by transform intent
- query by operator family
- query by perf regression or improvement

Examples:

- `tiling reformulate attention`
- `layout change kv cache`
- `perf regression after fix`

## Retrieval Expectations

The preferred mixed result set is:

1. one or two high-signal docs
2. one or two patch-bearing examples
3. one or two related run bundles or tasks

That blend is typically more useful than returning only prose or only runs.
