# JPEG self-similarity experiment

## Question

Does within-image similarity of JPEG forensic features improve development AIC
over the current JPEG640 baseline, particularly for small masks, without raising
false positives on unmodified images?

This is an IIS-inspired ablation, not a reproduction of Lee et al. (2024),
https://doi.org/10.1016/j.imavis.2024.105138. It reuses the JPEG encoder instead
of adding Laplacian preprocessing, patchifying RGB, or building GFNet.

## Architecture

After native JPEG features are aligned to the RGB crop/frame, take stride-32
features (128 channels). Pool to at most 20 x 20 and project to 64 channels.
Each spatial position is a token. Aggregate scaled dot products with every other
token in the same image, excluding itself. Apply softmax across positions,
multiply by token count, and subtract one so uniform scores produce zero.

For tokens z_i, compute dot(z_i, sum_j(z_j) - z_i) / sqrt(64). This equals
summing the off-diagonal pairwise matrix without allocating it. Accumulation
and softmax use FP32 under AMP. The order follows the published code's
sum-before-softmax, not equations 6-7 of the paper.

Resize the one-channel cue to the stride-32 feature size, project to 128
channels and add it through a zero-initialized per-channel gate before existing
LocalFusion. The module has 8,512 parameters. It compares the current aligned
crop during crop augmentation; full-frame phases compare the complete frame.
Missing-JPEG examples bypass the branch as before.

The flag defaults to false. New modules are constructed after all baseline
modules, preserving their seeded weights and old checkpoint keys when disabled.
The enabled flag is included in snapshots and used by inference and resume.
The similarity channel gate uses the existing optimizer convention (no weight decay).

## Controlled runs

| Arm | Config | Run directory |
|---|---|---|
| Control | configs/experiments/jpeg_similarity_control.yaml | runs/jpeg_similarity_control |
| Similarity | configs/experiments/jpeg_similarity.yaml | runs/jpeg_similarity |

Both inherit baseline.yaml: seed 42, 6 epochs, 24,000 sampled rows per epoch,
25% negatives, final 2 full-frame epochs, the same loss, augmentations, optimizer,
threshold grid, protocol and evaluation. Start both from the standard RGB/JPEG
pretraining, not a trained baseline checkpoint. Keep hardware and effective AND
forward batch identical across arms. Run sequentially to avoid GPU contention.

For comparison with the historical short baseline, use its recorded setup:
2 GPUs, per-GPU batch 8, accumulation 1, bf16. The current local .env instead
resolves to batch 2 / accumulation 8 on one GPU; it is not the historical setup.
Do not compare those setups as if similarity were their only difference.

From the repository root on the server, using the project's challenges environment:

```bash
python -m src.training --config configs/experiments/jpeg_similarity_control.yaml
python -m src.training --config configs/experiments/jpeg_similarity.yaml
```

Alternatively open notebooks/jpeg_similarity.ipynb and select either arm.
Configure server paths, devices, batch, accumulation and workers in .env first.
The notebook displays the resolved configuration before its training cell.
Provide the existing protocol and DCT_djpeg.pth; RGB pretraining must be cached
or downloadable. The unchanged runner enforces protocol and resume checks.
Resume is enabled only within each unique arm's own directory.

## Evaluation

Use development only for checkpoint/threshold selection, preserving the existing
weighted selection score (small-mask weight 1.6). Report separately:

- ordinary AIC and Dice_pos from summary.development.combined;
- weighted selection AIC from summary.development.selection;
- FPR_neg for combined, provided negatives and originals;
- available per-area Dice diagnostics, especially (1%, 5%];
- model choice (raw/EMA), chosen thresholds, compute and measured latency.

Historical baseline reference: ordinary development AIC 0.9180208,
Dice_pos 0.8605647, FPR_neg 0.0163020; weighted selection AIC 0.9091187.
Use the new matched control for attribution. A single seed is a screening run;
repeat promising results before claiming a reliable improvement. Inspect any
FPR increase even if combined AIC improves. Do not tune on holdout or test.

## Local validation, 2026-09-15

No full training or quality evaluation was run locally, as requested.
Full test suite: 377 passed, 1 skipped. Notebook code cells parse successfully.
Smoke artifacts: runs/jpeg_similarity_smoke/verify.py and result.json.
Four optimizer steps on one positive and one negative real TRAIN example at
640 x 640, batch 2, bf16, randomly initialized model (no pretrained weights):
finite losses and gradients, and a nonzero update of the similarity embedding
through both initially zero gates. Peak allocated CUDA memory: 2.501 GiB on
RTX 3070 8 GB. This does not guarantee memory use on larger native frames.

Full eval-forward counts from FlopCounterMode at RGB 640 and native JPEG
1024 x 1024:

| Backend | Control GFLOPs | Similarity GFLOPs | Delta |
|---|---:|---:|---:|
| CUDA | 90.571447 | 90.578103 | 0.006656 |
| CPU fallback | 71.566007 | 71.572663 | 0.006656 |

The JPEG backend changes operation counting; use the CUDA reference for the
server comparison. This reference meets 100 GFLOPs, but larger native inputs
still require checks. H100 latency <=50 ms has not been measured.

On this Windows machine NumPy must be imported before PyTorch to avoid an MKL
import abort. The smoke script and experiment notebook do so. Triton requires
a writable TRITON_CACHE_DIR; local smoke used runs/.triton_cache. Neither
workaround changes server training settings or model behavior.
