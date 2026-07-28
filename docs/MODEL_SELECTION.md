# Base-model selection

## Decision

The systems target is **Skywork AI Matrix-Game 3.0 5B**, using its distilled interactive checkpoint for latency work and its base checkpoint as the quality reference.

This is not a generic text-to-video checkpoint. It is an action-conditioned, streaming interactive world model with long-horizon memory. The upstream release exposes the exact systems surfaces this project needs: a few-step autoregressive loop, FlashAttention selection, sequence parallelism, FSDP, INT8 transformer paths, compiled/streaming VAE decoding, and an asynchronous VAE worker.

## Shortlist

| Candidate | Why it is relevant | Why it was or was not selected |
|---|---|---|
| **Matrix-Game 3.0 5B** | March 2026 release; 720p streaming interactive generation; base and distilled weights; long-horizon memory; 5B scale; permissive Apache-2.0 license for the 3.0 subproject. | **Selected.** Best combination of a real task-trained world model, manageable scale, weights, latency-critical code paths, and a clean public systems-extension story. |
| **HY-World 1.5 / WorldPlay 8B** | Richest public training surface in the shortlist: autoregressive memory training, sequence parallelism/HSDP, restart support, RL post-training, and a four-step distilled checkpoint. | Technically the strongest fine-tuning substrate, but its custom community license excludes the EU, UK, and South Korea and constrains redistribution/hosted use. That is a bad foundation for a globally accessible public derivative repository. |
| **HY-World 1.5 / WorldPlay 5B (WAN)** | Lighter checkpoint and related WAN backbone. | The upstream documentation explicitly describes compromised action control and long-term memory relative to the 8B path. It gives up the main reason to choose a task-trained world model. |
| **Matrix-Game 2.0** | Earlier real-time interactive foundation model with causal/model code. | Superseded by 3.0's memory design, distilled runtime, quantization, and streaming VAE path. Useful as an ablation or compatibility target, not the flagship. |
| **Wan/HunyuanVideo base checkpoints** | Strong and widely used open video backbones. | Rejected as the primary target because they are generic video generators, not task-trained interactive world models—the project requirement was explicitly to avoid building around a generic base or a one-paper method tweak. |

## Fine-tuning outlook

There is an important split:

- **Best public systems target:** Matrix-Game 3.0.
- **Best documented training/fine-tuning stack:** HY-World 1.5, subject to its license.

Matrix-Game 3.0 is the better flagship for this repository because latency, streaming, quantization, memory selection, and serving are first-class. Its upstream release is currently much stronger on inference than on official end-to-end training. Therefore, this repository keeps the distributed checkpointing, sampler, deterministic validation, and restart-equivalence components model-agnostic instead of pretending an unofficial full training recipe is authoritative.

A future official Matrix-Game 3.0 training release can plug into the existing `TrainingStepProfiler`, `AsyncDistributedCheckpointer`, stateful batch sampler, and scaling-regression harness without changing the measurement contract.

## Why the AdaLN kernel is the right first kernel

The 5B backbone has 40 transformer blocks with hidden width 5120. In every block, the upstream implementation computes a normalized activation, applies diffusion-time scale and shift, then later applies a gated residual—once for self-attention and once for the feed-forward path. The unfused expression materializes multiple full `[batch, tokens, 5120]` tensors and is repeated throughout every denoising step.

The first custom kernel therefore fuses:

```text
LayerNorm(x) * (1 + scale) + shift
```

with an analytical backward pass, plus a companion gated-residual kernel. This is model-specific enough to matter, repeated enough to be measurable, and narrow enough to validate rigorously before attempting a much riskier sparse-attention kernel.
