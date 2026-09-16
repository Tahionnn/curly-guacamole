# Final-mask boundary BCE fine-tune

E1: `disentangle_b2_li760_r8_boundary_ft.yaml`.
Matched E0: `disentangle_b2_li760_r8_boundary_control_ft.yaml`.
Each has a corresponding notebook in `notebooks/`; the final cell runs training.

Both start from the EMA of `disentangle_b2_li760_r8_long/ckpt/best.pt`, with fresh
optimizer/scheduler, 3 x 24,000 examples, encoder LR 1e-5 and JPEG/head LR 3e-5.
All epochs use full frames. Existing sampling (25% negatives), coarse patch/edge
losses and JPEG recipe (30%, Q60-99, no grid shift) are preserved. Use identical
server runtime settings in `.env` for both runs.

E1 adds `0.2 * boundary_bce`. The band is dilation minus erosion of `target > 0.5`
on the final 760 x 760 grid, with a square 9 x 9 kernel (Chebyshev radius 4).
This is not the ellipse used in the exploratory spatial diagnostic. BCE retains
the original, potentially soft target. Each image is normalized by its band area,
then all images are averaged, with empty bands contributing zero. Uniform empty
or full masks have no artificial contour at the image frame. Existing full-image
losses still supervise negatives. The extra component is logged as `boundary_bce`.
E0 differs only in run name and `boundary_weight: 0.0`.

No model parameters, forward outputs or inference operations change. The initial
weight 0.2 is an experimental choice, not a tuned or demonstrated optimum.

Checkpoint selection retains the original weighted AIC (small-mask weight 1.6).
Report ordinary development AIC, positive Dice and negative FPR separately.
Compare at the original frozen threshold 0.3984375, cls=0, min-area=0 as well as
the selected threshold. Freeze quality cohorts using source-model development
Dice; report gains and regressions, interior FN and exterior FP. Do not tune on
holdout/test. Compare E1 to E0 to distinguish boundary supervision from extra
training. A single seed is a screening experiment.

CLI alternatives from the project root:

```bash
python -m src.training --config configs/experiments/disentangle_b2_li760_r8_boundary_control_ft.yaml
python -m src.training --config configs/experiments/disentangle_b2_li760_r8_boundary_ft.yaml
```

## Local verification (2026-09-16)

101 focused tests passed (loss, config, fine-tune, disentangle, engine, baseline
defaults and resume). Both notebooks' code cells parse. Independent code review
found no actionable issues. On RTX 3070, a real TRAIN positive/negative batch at
RGB 760 loaded the source EMA strictly and completed BF16 forward/backward:
620 gradient tensors finite, peak allocated memory 3.303 GiB, weighted boundary
BCE 0.03713. No optimizer steps or full training were run. This is a smoke check,
not evidence of quality improvement or a server memory/latency guarantee.
Local artifacts: `runs/boundary_ft_smoke_20260916/{verify.py,result.json}`.
