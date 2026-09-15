# DG-Force adaptation

The patch/edge bottlenecks run after JPEG fusion at strides 4, 8, 16 and 32.
Training uses soft occupancy BCE + Dice (weight 0.5) and balanced boundary BCE
(weight 0.5, band width 3 stage cells). Auxiliary prediction heads are omitted
from evaluation. The baseline keeps the module disabled.

| Config | Behavior |
|---|---|
| disentangle_supervision.yaml | Training-only supervision; no inference overhead |
| disentangle_fuse.yaml | Supervision plus spatially balanced, channel-gated residuals |
| disentangle_fuse_cross32.yaml | Also transfers stride-16 context to stride 32 with attention |

Run from the project root in the challenges environment:

```bash
python -m src.training --config configs/experiments/disentangle_fuse_cross32.yaml
```

All recipes inherit baseline.yaml and the machine's runtime settings from .env.
Use the same forward batch, accumulation, hardware and seeds across comparisons.
The supplied training log used forward/effective batch 16 on one GPU; local
settings can differ. Training is not started by installing these files.

Integration preserves JPEG similarity and creates the disentangle module after
existing modules to preserve their seeded initialization. This intentionally
differs from the supplied patch's construction order, so a new training run is
not a bit-for-bit reproduction of its logged trajectory. Model state-dict keys
for the supplied architecture are retained; exact optimizer-state resume from
an externally trained run requires matching its parameter ordering.

Reference: https://doi.org/10.1007/978-3-032-37432-5_9
