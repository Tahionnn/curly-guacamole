# Hard-pixel finetune on one H100 NVL

The existing `disentangle_b2_li760_r8_all_data_hard_pixel_ft.yaml` enables:

- Tiled Triton JPEG weight gradients instead of a full native-resolution
  21-channel one-hot tensor and convolution backward.
- Runtime JPEG width to avoid compiling a separate forward kernel for each width.
- Grouped gradient normalization at optimizer updates.

These switches keep checkpoint keys, loss, learning rates, image sizes and batch
settings unchanged. Resume accepts these execution changes; floating-point
reduction order can differ. Restart from the latest saved epoch to apply them;
an already running Python process does not reload source changes. Progress inside
an unfinished epoch is not checkpointed.

## Measure on the server

Run from the repository root, with the training process stopped and the same
environment and batch settings for both variants. Triton must be installed and
usable in the server's PyTorch environment. The profiler uses disposable models
and does not modify training checkpoints.

```bash
export TRITON_CACHE_DIR="$PWD/runs/.triton_cache"
export AIIJC_DEVICE=cuda AIIJC_DEVICES='[0]' AIIJC_AMP=bf16
CFG=configs/experiments/disentangle_b2_li760_r8_all_data_hard_pixel_ft.yaml
python -m src.training.profiling --config "$CFG" --reference-kernels \
  --warmup 16 --batches 64 --output profiles/h100_reference.json
python -m src.training.profiling --config "$CFG" \
  --warmup 16 --batches 64 --output profiles/h100_optimized.json
```

Compare `results.loader.wall_ms_per_batch`, `cuda_ms_per_batch.backward` and
`peak_allocated_gib`. Also inspect `peak_reserved_gib`. Peaks include warmup.
Repeat in reverse order to reduce disk-cache and compilation bias. A single GPU
replay batch does not represent all native JPEG dimensions or worst-case memory.
No H100 speedup or maximum safe batch size has been measured locally.

If loader time substantially exceeds GPU replay, sweep loader workers:

```bash
python -m src.training.profiling --config "$CFG" --workers 2 4 8 16 \
  --repeats 2 --warmup 16 --batches 64 --output profiles/h100_workers.json
```

Set `AIIJC_WORKERS` to the measured winner. Hundreds of available CPU threads do
not imply hundreds of workers are useful. Each worker prefetches one batch of
native frames; too many workers consume RAM and shared memory.

## Batch size and restart

Process environment overrides `.env`, which overrides YAML. Check the startup
log for actual batch size and workers. The local `.env` resolves to batch 2 and
accumulation 8, but the server may differ.

For a **new** run with effective batch 16, benchmark batch/accumulation pairs
4/4, 8/2 and 16/1. Compare images/second, not milliseconds/batch. Native JPEG
activations can dominate memory even with 94 GB, and the JPEG branch still runs
per frame. Larger RGB batches also change BatchNorm statistics; equal effective
batch does not make these runs numerically equivalent.

For **resume**, preserve batch size and accumulation from the saved run.
Changing them is deliberately rejected. Continue the existing run with:

```bash
python -m src.training --config "$CFG"
```

For a fresh batch-size experiment, use a new run name and `finetune_from` pointing
to the chosen saved checkpoint. This loads weights and starts fresh optimizer/EMA
state; it is not an exact continuation. Do not enable global cuDNN benchmarking
blindly: native JPEG shapes vary and per-shape searches can dominate startup.
