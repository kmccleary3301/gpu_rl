# AMD Scope Boundary

This project includes a deliberately narrow AMD/ROCm mirror of the core replay, trace, profile, and inspection surfaces. The intent is contract parity where it is cheap and useful, not broad vendor-feature parity before the first training wave.

## Covered in the training-preparation scope

- `doctor` detection for ROCm-adjacent tooling such as `rocminfo`, `rocm-smi`, `rocprof`, `rocprofv3`, and `hipcc`
- realistic parsing of `rocminfo` and `rocm-smi`-style outputs into the shared hardware and tool-status contracts
- normalized AMD trace and profile reports through the ROCm backend
- markdown and JSON summary emission for AMD trace/profile outputs
- replay, inspect, compare, and evidence-quality projections for AMD-shaped bundles
- checked-in AMD golden fixtures and tests that protect the normalized mirrored path

## Explicitly not covered yet

- sanitizer parity with the NVIDIA workflow
- deep ROCm kernel-analysis coverage comparable to the NVIDIA `ncu` path
- broad HIP benchmark ingestion
- first-wave training that depends on real AMD task execution
- vendor-parity claims beyond the checked-in trace/profile/replay contracts

## How to read AMD support in this repository

- AMD support is a mirrored control-plane and artifact-plane path.
- NVIDIA remains the primary first-wave execution and training-facing path.
- AMD fixtures keep the contracts honest and the repo vendor-aware without forcing broad parity before the first dedicated training runs.

## Related references

- [AMD parity knowledge note](../knowledge/hardware_notes/amd_parity.md)
- [Project scope](./PROJECT_SCOPE.md)
- [Policy interface](./POLICY_INTERFACE.md)
