# AMD Parity Boundary

The AMD branch is currently a normalized mirror for trace, profile, replay, and inspection workflows. It is intentionally narrow.

What is supported now:

- `doctor` can detect ROCm tools and parse realistic `rocminfo`-style GPU metadata.
- `rocprof` trace and profile outputs normalize into the same top-level trace/profile contracts used elsewhere.
- AMD traces and profiles emit JSON plus markdown summaries.
- Replay, inspect, and compare can operate on AMD-shaped bundles and golden fixtures.
- A metadata-only AMD smoke task exists at `task/amd/smoke/diagnose/v1`.

What is not yet supported:

- first-class AMD sanitizer parity
- broad HIP benchmark ingestion
- real ROCm task execution coverage on this host
- feature-complete parity with NVIDIA profiling depth

The design intent is to keep the contracts shared even while the underlying vendor tools remain asymmetric.
