# LingBot-World 2.0 systems integration

## Why this model

LingBot-World 2.0 was released in July 2026 and is one of the newest public causal interactive world models with runnable weights. The released `causal-fast` model is a 14B Wan2.2 derivative that generates video chunk by chunk with:

- causal attention;
- self-attention KV caching;
- cross-attention caching;
- local attention windows and persistent sink frames;
- four-step distilled sampling;
- FSDP and Ulysses sequence parallelism;
- action/camera conditioning;
- a shape-matched prewarm path.

This makes it a much richer systems target than a conventional one-shot text-to-video pipeline.

## Upstream bottleneck map

The first benchmark matrix should isolate:

1. cold start versus upstream `prewarm()`;
2. DiT time versus VAE encode/decode;
3. local-attention cache movement and host synchronization;
4. self-attention versus cross-attention;
5. AdaLN/modulation kernels repeated in every block;
6. 1/2/4/8-GPU sequence-parallel scaling;
7. long-horizon cache growth and memory plateau;
8. output quality under numerical changes.

## Implemented pass: fused AdaLN

The causal-fast block applies affine-free LayerNorm followed by per-token scale and shift before self-attention and the FFN. The head repeats the same pattern. The `fused-adaln` integration redirects all three sites to the shared Triton kernel.

Acceptance criteria:

- tensor forward/backward equivalence for FP16/BF16/FP32;
- no change to model arguments, checkpoint or sampling schedule;
- no end-to-end claim until measured with the real 14B checkpoint;
- output comparison on fixed prompt/image/actions/seed;
- report kernel and end-to-end deltas separately.

## Experimental pass: FP32 causal RoPE

Upstream causal RoPE converts Q and K to float64, yielding complex128 multiplication. The experimental path performs the rotation in float32/complex64 and casts back to the original activation dtype.

The CPU contract test measures numerical agreement on representative 3D grids. That is necessary but not sufficient. Long-horizon world models can amplify tiny phase errors, so model-level drift and action fidelity must be evaluated before enabling this pass by default.

## Licensing

The upstream repository advertises CC BY-NC-SA 4.0. The adapter therefore marks commercial use and hosted-service deployment as unavailable by default. World Model Systems does not redistribute source or weights and does not alter the upstream license.
