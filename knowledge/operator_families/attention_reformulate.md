# Attention Reformulate

This workload family captures transformation episodes where the starting point is correct but strategically weak. The current case reformulates a naive causal attention-score implementation into a tiled Triton kernel so the episode includes semantic preservation, strategy description, and a real perf gate.
