# Memory-Bound Playbook

When DRAM throughput is high while SM throughput and occupancy do not explain the slowdown, treat the kernel as memory-bound first. Look for gather/scatter access, low data reuse, unnecessary loads, and whether staging or layout changes can reduce traffic.
