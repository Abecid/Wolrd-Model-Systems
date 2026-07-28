# GPU validation roadmap

- [ ] Run BF16 baseline on one H100 with a fixed image/action trace.
- [ ] Export PyTorch Chrome trace and Nsight Systems timeline.
- [ ] Identify the largest measured bottleneck.
- [ ] Validate fused AdaLN forward and backward at width 5120 on H100/A100.
- [ ] Integrate the kernel and verify fixed-seed latent/output error.
- [ ] Compare synchronous versus asynchronous VAE.
- [ ] Compare BF16 versus upstream INT8 Q/K/V/O; add FP8 only after a tested backend exists.
- [ ] Run 1/2/4/8-GPU scaling and quantify NCCL fraction.
- [ ] Run checkpoint foreground-pause and exact-restart tests on eight GPUs.
- [ ] Publish p50/p95 latency and throughput-versus-concurrency service curves.
- [ ] Attach real before/after traces to `reports/`.
