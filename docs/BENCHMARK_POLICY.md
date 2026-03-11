# Benchmark Positioning and Training Eligibility

## Why Public Benchmarks Exist Here

Public benchmark imports serve three useful roles:

- external provenance
- broader operator coverage
- retrieval and comparison anchors outside the bespoke internal task set

## Why They Are Not the Default Training Core

Most public benchmark traces are weaker than internal transition-rich episodes along at least one of these axes:

- patch-bearing state transitions
- hidden failure semantics
- candidate lineage
- explicit repair or reformulation logic
- richer governance signals

For that reason, the packaging defaults deliberately favor internal repair/diagnose/reformulate traces over thin benchmark wrappers.

## Default Policy

### Training-eligible by default

- internal patch-bearing debug traces
- internal patch-bearing reformulate traces
- governed diagnose traces when explicitly allowed

### Benchmark-only by default

- public benchmark runs without meaningful transitions
- public benchmark runs that are useful for reporting or transfer evaluation but not for default SFT packaging

### Retrieval-eligible

- both internal and public examples, provided provenance and artifact quality are intact

## Override Policy

Benchmark-only traces may be included in packaging only when a packaging command or config explicitly allows them.

This keeps the default training corpus closer to the actual policy-learning objective.
