# World Model Systems

**A model-agnostic systems optimization library for causal video diffusion and interactive world models.**

World Model Systems separates reusable GPU/runtime machinery from thin upstream adapters. The goal is to make the same profiling, kernel, distributed-training, and serving infrastructure work across several rapidly changing world-model codebases without copying their source or weights.

The repository started as a Matrix-Game 3.0 systems project. Version 0.2 turns it into a registry-driven library and adds a second real optimization target: **LingBot-World 2.0**, a July 2026 causal Wan2.2 world model with chunked generation, KV caching, local attention, sink tokens, few-step inference, FSDP, and Ulysses sequence parallelism.

> This is an independent project. Upstream source and weights remain under their own licenses and are never redistributed here.

## Supported model adapters

| Adapter ID | Model | Why it matters | Integration status |
|---|---|---|---|
| `matrix-game-3` | SkyworkAI Matrix-Game 3.0 5B | 720p streaming interactive generation, memory, few-step inference, INT8, async VAE | Profiling, fused AdaLN patch, serving adapter |
| `lingbot-world-v2` | Robbyant/Ant Group LingBot-World 2.0 14B causal-fast | Unbounded causal rollout, KV cache, local attention + sinks, four-step chunks, Wan2.2 backbone | **New:** reproducible launcher, profiler, fused AdaLN patch, experimental FP32 causal RoPE |
| `hy-worldplay-1.5` | Tencent Hunyuan HY-World 1.5 / WorldPlay | Official Hunyuan causal world model with action control, memory, training code, RL and four-step distillation | Command/profiling adapter; source patching intentionally disabled |

LingBot-World 2.0 is built on Alibaba's Wan2.2, but the released repository is from Robbyant/Ant Group rather than Alibaba Research. Its upstream code and weights are CC BY-NC-SA 4.0, so the adapter is research/non-commercial unless separate rights are obtained. HY-WorldPlay uses a custom Tencent license with territorial and hosted-service restrictions. The registry exposes these policies instead of burying them in a footnote.

## Core architecture

```text
src/world_model_systems/
├── core/                    model specs, capabilities, registry, optimization planner
├── models/                  one thin adapter per upstream model
├── integrations/            revision-pinned, idempotent source patch machinery
├── optimizations/
│   ├── kernels/             reusable Triton/PyTorch kernels
│   ├── precision/           quantization and numerical policies (growing)
│   ├── parallelism/         sequence/context parallel recipes (growing)
│   └── runtime/             cache, compile, CUDA graph, VAE overlap recipes (growing)
├── profiling/               common telemetry and model-command runner
├── distributed/             restartable checkpoints, samplers, scaling utilities
└── serving/                 generic serving contracts and model backends

src/matrixgame_systems/      compatibility package from v0.1; reusable code is retained
configs/models/              declarative model defaults
configs/recipes/             optimization recipes
scripts/                     bootstrap, profile, Nsight, Slurm and service entrypoints
```

The key abstraction is a `ModelAdapter`. It declares:

- upstream repository and pinned revision;
- model family, size, license policy and capabilities;
- a deterministic inference command;
- model-specific source patches;
- supported optimization passes.

Reusable optimizations never import an upstream repository. Upstream adapters import the reusable library.

## Install

```bash
git clone https://github.com/Abecid/Wolrd-Model-Systems.git
cd Wolrd-Model-Systems
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,gpu,service]'
```

List the registered models:

```bash
wms models
wms describe lingbot-world-v2
wms plan lingbot-world-v2 --objective latency
```

## LingBot-World 2.0: first new optimization target

### 1. Bootstrap the pinned upstream source

```bash
bash scripts/bootstrap_model.sh lingbot-world-v2
```

This checks out commit `2648877f763a06cc743bcd919936da4d25f12e7b` under `third_party/lingbot-world-v2`. The directory is gitignored.

Download the official causal-fast checkpoint separately:

```bash
huggingface-cli download robbyant/lingbot-world-v2-14b-causal-fast \
  --local-dir /path/to/lingbot-world-v2-14b-causal-fast
```

### 2. Generate a reproducible command

```bash
wms command lingbot-world-v2 \
  --upstream third_party/lingbot-world-v2 \
  --checkpoint /path/to/lingbot-world-v2-14b-causal-fast \
  --image /path/to/example/image.jpg \
  --prompt 'A character walks forward through the world.' \
  --frames 361 --num-gpus 8
```

### 3. Profile the untouched baseline

```bash
wms-profile lingbot-world-v2 \
  --upstream third_party/lingbot-world-v2 \
  --checkpoint /path/to/lingbot-world-v2-14b-causal-fast \
  --image /path/to/example/image.jpg \
  --prompt 'A character walks forward through the world.' \
  --frames 361 --num-gpus 8 \
  --run-dir runs/lingbot_v2/bf16_baseline
```

The common report stack records wall time, GPU utilization, physical and PyTorch HBM, frames per GPU-hour, and workload fingerprints. Model-internal phase instrumentation is added only when the upstream adapter can do so without changing semantics.

### 4. Apply the fused AdaLN patch

LingBot's causal-fast DiT repeatedly computes

```text
LayerNorm(x) * (1 + scale) + shift
```

before self-attention, the FFN, and the output head. The integration patch replaces those expressions with the same tested Triton AdaLN kernel used by Matrix-Game:

```bash
wms patch lingbot-world-v2 fused-adaln \
  --upstream third_party/lingbot-world-v2
export WMS_USE_FUSED_ADALN=1
```

The patch is:

- pinned to the reviewed upstream revision;
- idempotent;
- fail-closed when source anchors change;
- recorded in a JSON manifest inside the gitignored upstream checkout;
- reversible by resetting the upstream checkout.

### 5. Optional FP32 causal RoPE experiment

Upstream causal RoPE promotes every Q/K tensor to complex128. The library includes a complex64 path and a numerical-equivalence test:

```bash
wms patch lingbot-world-v2 fp32-causal-rope \
  --upstream third_party/lingbot-world-v2
```

This pass is marked **experimental**. It should not be enabled in a published performance claim until long-horizon output quality and drift are checked against the original complex128 path.

### 6. Compare one change at a time

```bash
# Baseline
bash scripts/profile_lingbot_world_v2.sh baseline

# Fused AdaLN only
bash scripts/profile_lingbot_world_v2.sh fused-adaln

# Generate a before/after report
wms-report runs/lingbot_v2/bf16_baseline \
  --candidate runs/lingbot_v2/fused_adaln
```

The report refuses to calculate deltas when the prompt, image, frame count, GPU count, precision, checkpoint or model revision differ.

## Optimization taxonomy

Optimizations are organized by what they change, not by the paper or model where they first appeared.

| Layer | Examples | Reuse boundary |
|---|---|---|
| Algorithm | few-step distillation, causal chunk size, CFG removal | Model/checkpoint capability |
| Kernel | fused AdaLN, fused residual, RoPE, attention | Tensor contract and numerical tolerance |
| Precision | BF16, FP8 GEMM, INT8 linear | Hardware + model quality contract |
| Parallelism | Ulysses sequence parallel, FSDP, context parallel | Process topology and divisibility |
| Runtime | KV cache, prewarm, CUDA graphs, async VAE | Static-shape/state-machine contract |
| Data/training | resumable sampler, async DCP, dataloader overlap | Training-loop contract |
| Serving | queueing, batching, admission control, cancellation | Backend request contract |

`wms plan` selects only passes that are implemented, match the requested objective, and satisfy the model's declared capabilities. Experimental passes are excluded unless explicitly requested.

## Performance and correctness rules

1. **No fabricated benchmark numbers.** GPU claims require real weights and hardware.
2. **One variable per comparison.** Do not bundle quantization, compile, kernel and resolution changes into one “speedup.”
3. **End-to-end first.** Kernel latency is reported alongside total generation latency.
4. **Slowest-rank timing.** Distributed critical path is the maximum rank time, not the sum.
5. **Quality is a constraint.** Every numerical optimization needs tensor tests and model-level output checks.
6. **Upstream revisions are pinned.** Patches fail when reviewed source anchors move.
7. **Licenses are first-class metadata.** Supporting an adapter does not relicense its upstream model.

## Existing systems modules

The original Matrix-Game implementation remains available while the public API migrates:

- phase timers, CUDA events, NVML sampling and trace parsing;
- achieved TFLOP/s and estimated MFU support;
- fused AdaLN Triton forward/backward and benchmarks;
- Ulysses all-to-all transforms;
- sharded asynchronous distributed checkpoints;
- exact-resume distributed sampler;
- deterministic validation and throughput regression gates;
- FastAPI queue, HBM admission control, SSE progress, cancellation and metrics.

The old `mgs-*` commands remain aliases in v0.2. New code should use `wms-*`.

## Add another model

Implement one adapter under `src/world_model_systems/models/` and register it in `core/registry.py`. A useful adapter must provide real contracts, not a name in a table:

1. pin an upstream revision;
2. declare capabilities and license constraints;
3. build a reproducible launch command;
4. define fail-closed source patches where appropriate;
5. add CPU contract tests;
6. add a real GPU benchmark recipe without invented output.

See [`docs/MODEL_ADAPTERS.md`](docs/MODEL_ADAPTERS.md) and [`docs/LINGBOT_WORLD_V2.md`](docs/LINGBOT_WORLD_V2.md).

## Validation boundary

CPU tests cover the registry, planner, patch idempotence, restart logic, profiler/report contracts, serving scheduler and numerical kernel references. CUDA/Triton, NCCL, FP8, CUDA graphs, model quality, and end-to-end speedups require the actual NVIDIA environment and official checkpoints. CI skips those claims rather than manufacturing success.

## License

Original code in this repository is Apache-2.0. Every upstream model remains governed by its own source and weight terms. See [`THIRD_PARTY.md`](THIRD_PARTY.md).
