# GPU optimization A/B for all-data finetuning

Both switches default to false. No architecture, checkpoint tensor keys, loss,
learning rates, or batch sizes change. The running training process is unchanged.

- `model.jpeg_triton_backward`: accumulate categorical convolution weight
  gradients in FP32 tiles, then reduce bounded partials. No full-image one-hot
  volume or global atomic accumulation. The forward and bias gradient stay the same.
- `train.foreach_grad_normalization`: divide accumulated gradients in groups by
  device/dtype. The sample divisor, clipping order and partial final batch stay the same.

The Triton path needs Triton and a supported CUDA dtype; the existing eager
fallback remains available where the CUDA lookup is unavailable. Different
reduction orders can produce small floating-point differences.

## Server benchmark

Use an idle GPU and the same software environment for every measurement. The
profiler uses disposable random weights and never modifies training checkpoints.
Replace workers=2 if the prior worker sweep found a better setting.

```bash
export AIIJC_BATCH_SIZE=16 AIIJC_ACCUM_STEPS=1 AIIJC_WORKERS=2
export TRITON_CACHE_DIR="$PWD/runs/.triton_cache"

for repeat in 1 2; do
  variants=(baseline foreach jpeg both)
  if (( repeat == 2 )); then variants=(both jpeg foreach baseline); fi
  for variant in "${variants[@]}"; do
    flags=()
    case "$variant" in
      foreach) flags=(--foreach-grad-normalization) ;;
      jpeg) flags=(--jpeg-triton-backward) ;;
      both) flags=(--jpeg-triton-backward --foreach-grad-normalization) ;;
    esac
    python -m src.training.profiling \
      --config configs/experiments/disentangle_b2_li760_r8_all_data_ft.yaml \
      --warmup 32 --batches 256 "${flags[@]}" \
      --output "profiles/all_data_${variant}_${repeat}.json" || exit 1
  done
done
```

Compare `results.loader.wall_ms_per_batch` first, then `backward` and `optimizer`
inside `cuda_ms_per_batch`. GPU replay repeats a fixed frame batch; it is not an
estimate over the full distribution of native JPEG sizes. No server speedup has
been established yet. Validate peak memory and long-run stability on large frames
before selecting the tiled path for a production run.

Local correctness checks cover FP32/FP16/BF16, multi-frame and non-square inputs,
constant categories 0/20, boundary padding, partial tiles, multiple partition
iterations, strided inputs, frozen parameters, and accumulated gradient scaling.
Existing CUDA lookup, config, engine, checkpoint and profiling tests also pass.
