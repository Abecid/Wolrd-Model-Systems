# Matrix-Game 3.0 Performance Report

## Hardware and workload fingerprint

- GPU / interconnect:
- GPU count:
- CUDA / driver / PyTorch / Triton:
- upstream commit:
- model checkpoint hash:
- resolution / generated frames / denoising steps:
- prompt, image hash, action trace, seed:
- precision / attention backend / sequence-parallel degree:

## Baseline bottleneck decomposition

| Metric | Value |
|---|---:|
| Wall time | |
| DiT forward | |
| VAE encode/decode | |
| Text encoder | |
| NCCL fraction | |
| GPU utilization | |
| Allocated / physical HBM peak | |
| Achieved TFLOP/s / estimated MFU | |
| Frames per GPU-hour | |

## Optimization hypothesis

State the largest measured bottleneck, the mechanism causing it, and why the
proposed change should improve end-to-end time rather than merely a microbenchmark.

## Before / after

Include identical-workload Nsight Systems timelines, profiler traces, kernel
benchmarks, quality checks, and end-to-end deltas.

## Correctness and quality guardrails

- fixed-seed latent/output comparison;
- kernel max/mean error and gradient error;
- long-horizon action consistency;
- output video metrics or blinded human comparison;
- no increased failure/NaN rate.
