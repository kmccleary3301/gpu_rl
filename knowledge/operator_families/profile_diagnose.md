# Profile Diagnose

This workload family is not kernel synthesis. It is profiler interpretation: given normalized profile fixtures, identify the primary bottleneck and the next tuning move with enough structure that the result can be scored automatically.

It is useful for collecting diagnose trajectories that do not require code generation but still demand real systems reasoning. The current fixture set covers memory-bound and occupancy-limited cases.
