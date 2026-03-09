# Attention Fragments

Attention fragments cover the kernels that sit around full attention implementations: score tiles, mask application, KV-cache movement, and decode-time paged access. The planner notes made this an explicit target family because it connects directly to real inference bottlenecks and cross-vendor Triton portability.
