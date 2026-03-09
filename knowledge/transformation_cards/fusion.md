# Fusion

Fusion removes intermediate launches and memory traffic by combining adjacent stages into one kernel. It improves throughput when launch overhead or round-trips through global memory dominate, but it can also raise register pressure and hurt occupancy.
