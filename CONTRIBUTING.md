# Contributing

## Scope

This repository is research infrastructure. Contributions are most useful when they preserve:

- artifact quality
- replayability
- governance clarity
- task and benchmark provenance

## Preferred Changes

- focused improvements to task surfaces, inspections, packaging, and handoff docs
- additive benchmark/task coverage with clear provenance
- regression tests and golden fixtures that protect new behavior
- docs that clarify semantics or usage without leaking personal-development assumptions

## Expectations

- keep changes small and reviewable where possible
- add or update tests for behavior changes
- do not silently weaken task correctness, determinism, or anti-hack semantics
- prefer stable, inspectable artifacts over hidden side effects
- preserve tracked/generated boundaries in `.gitignore`

## Development Loop

```bash
pip install -e .
python3 scripts/export_schemas.py
python3 -m unittest discover -s tests -v
```

## Golden Fixtures

Checked-in golden datasets, episodes, and run bundles are part of the verification surface.

If you intentionally update them:

1. regenerate them deterministically
2. review the diff carefully
3. update tests or docs if semantics changed
