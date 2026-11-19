# Technical notes

## CT runner performance

Measurements used an Apple M1 Max with 64 GB of unified memory, macOS 27.2.1,
MLX 0.32.4, NumPy 2.5.4, and perfkit 0.3.0.dev0. The baseline was `db29b38`
with the same box-prompt validation and docstring edits. The measured change
combined integer lookup-table scaling with encoder prefetch for runner-prepared
slices.

The runner read a compressed NIfTI CT, loaded `MedSAM2_CTLesion.safetensors`,
propagated in both directions with the default eight-empty-slice stopping rule,
and wrote a uint16 NIfTI mask. Both HCC-TACE-Seg prompts came from the tumor
annotations in the source voxel grid.

- HCC_063 was a 512 × 512 × 99 int16 CT with box `[98, 213, 149, 263]` on
  zero-based slice 66 and no HU window.
- HCC_097 was a 512 × 512 × 111 int32 CT with foreground point `[85, 307]` on
  zero-based slice 55 and HU window `[-160, 240]`.

Perfkit alternated baseline and candidate over six fresh-process rounds per
case. Each worker ran one cold call, one discarded warm-up call, and three
measured warm calls. Cold timing includes lazy MLX import and Metal startup.
Warm calls repeat the full runner. Process footprint is the peak physical
footprint above the post-setup baseline. MLX peak is the allocator high-water
mark from `mlx.core.get_peak_memory()`. Both peaks cover the full worker run.

| CT and prompt | Cold (ms) | Warm (ms) | Peak footprint delta (MiB) | Peak MLX (MiB) |
| --- | ---: | ---: | ---: | ---: |
| HCC_063 box | 1,646.6 → 1,499.5 | 1,375.7 → 1,254.9 | 3,857.1 → 3,252.6 | 1,127.0 → 888.6 |
| HCC_097 point | 1,529.3 → 1,461.1 | 1,223.5 → 1,164.7 | 4,332.5 → 4,320.0 | 1,126.9 → 888.6 |

The table shows paired baseline and candidate medians. Perfkit's 96.9% paired
intervals put the warm latency reduction at 8.2–9.0% for the box and 4.1–5.3%
for the point. The written masks matched clean `db29b38` byte for byte on both
CTs.

Starting the MLX and predictor imports on a worker thread while the runner
reads the volume was measured in the same campaign and not adopted. Its cold
latency change was inconclusive at perfkit's 2% minimum-effect band for the
box (0.9–2.9% faster, 96.9% interval) and the point (0.8–3.5% faster, 99.2%
interval). Warm latency and both memory peaks were unchanged, and with MLX
unavailable the runner reported the missing runtime before an invalid prompt
frame.
