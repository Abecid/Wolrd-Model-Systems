# Open-source systems references

The implementation intentionally borrows architectural patterns rather than
copying code:

- **Matrix-Game 3.0:** target world model, Wan-style DiT, sequence parallelism,
  FlashAttention selection, Q/K/V/O quantization, asynchronous VAE pipeline.
- **PyTorch distributed checkpoint:** sharded and asynchronous state persistence.
- **Megatron-Core / DeepSpeed Ulysses:** sequence/context-parallel all-to-all
  layouts and communication/compute tradeoffs.
- **Triton LayerNorm and fused-attention tutorials:** one-program-per-row
  normalization and explicit backward reductions.
- **vLLM / SGLang / TensorRT-LLM:** bounded queues, continuous scheduling,
  admission control, warm-up, CUDA graphs, and production latency metrics.
- **FastVideo:** distributed video-DiT training, sequence parallelism, profiling,
  and data-pipeline design.
- **NVIDIA Nsight Systems/Compute:** system timeline first, kernel analysis second.

Every borrowed concept should be cited in the final performance report. This
repository's code is original unless a file explicitly says otherwise.
