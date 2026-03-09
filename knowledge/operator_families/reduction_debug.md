# Reduction Debug

This workload family targets repair episodes rather than synthesis. The current case models a Triton reduction bug where the mask excludes the last column, producing silently wrong row sums while still looking superficially plausible on many shapes.

It is intended to produce debug trajectories with explicit bug classification and repair metadata, not just corrected outputs.
