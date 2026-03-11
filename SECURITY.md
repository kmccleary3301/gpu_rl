# Security Policy

## Scope

This repository is research infrastructure, not a hosted service. The most useful security reports are issues that could affect:

- local command execution boundaries
- artifact or bundle integrity
- dataset or replay provenance
- accidental execution of untrusted code paths during task evaluation

## Reporting

If you find a security-sensitive issue, please avoid opening a public exploit write-up before maintainers have a chance to assess it.

Use a private maintainer contact path if one is available. If not, open a minimal issue that states:

- the affected surface
- the impact
- the smallest reliable reproduction you can provide without publishing unnecessary exploit detail

## Current Non-Goals

The repository currently does not claim:

- hardened sandbox isolation for arbitrary third-party task execution
- multi-tenant security guarantees
- production-service security properties

The intended operating model is controlled research and engineering environments with explicit operator review.
