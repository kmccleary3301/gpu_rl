# Attention Score Tiles

This workload family covers the score stage of attention: computing masked `QK^T / sqrt(d)` tiles before softmax or value application. It is narrower than full attention, but it is the core arithmetic path that exposes memory movement, tiling, and masking tradeoffs in a way that is easy to inspect in profiler traces and tri-view artifacts.

The current cockpit task uses a causal mask and asks for raw score tiles rather than a fused softmax/value path. That keeps correctness checks simple while still exercising:

- row-wise query reuse
- block-wise key loading
- masking behavior
- register pressure versus block size
- attention-specific bottleneck reasoning

In practice, the main failure modes here are:

- falling back to `torch.matmul` or higher-level attention helpers instead of a kernelized path
- incorrect causal masking
- poor block sizing that turns the kernel into a memory-bandwidth problem
