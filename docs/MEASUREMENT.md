# Measurement methodology

## Critical-path rule

Distributed ranks execute concurrently. Phase totals are therefore the maximum
rank-local total, not the sum across ranks. NCCL time is computed as the union of
NCCL kernel intervals in the Chrome trace to avoid double-counting overlapping
collectives.

## Timing hierarchy

1. wall-clock process time determines end-to-end throughput;
2. CUDA events measure GPU phases without synchronizing at every boundary;
3. NVTX ranges make the same phases visible in Nsight Systems;
4. Nsight Compute is used only after the dominant kernel is identified;
5. NVML samples physical utilization, power, clocks, and HBM.

## MFU

The estimator counts dense attention and MLP matrix multiplications for the Wan
DiT backbone and treats one multiply-add as two FLOPs. It omits elementwise work,
RoPE, normalization, the VAE, text encoder, sparsity, padding, and recomputation.
Always publish the formula and sequence length next to MFU.

## Required controls

Before/after runs must hold constant the GPU SKU/count, power mode, model and
checkpoint, prompt, image, action trace, seed, resolution, output frames,
denoising steps, attention backend, and warm-up policy unless the changed field
is the intervention itself.

Run at least one warm-up and five measured repetitions. Report the distribution,
not only the best run.
