# MatrixGame Systems

**Unofficial GPU-systems laboratory for Matrix-Game 3.0.**

This repository turns an open interactive world model into a serious ML-systems artifact:

1. an end-to-end video-DiT performance report;
2. a model-specific fused Triton AdaLN kernel;
3. restartable, sharded, asynchronous distributed training utilities;
4. a queueing, admission-controlled inference service with streaming progress and production metrics.

The base target is **SkyworkAI Matrix-Game 3.0 5B**, pinned to upstream commit
`71c3cd7f741311f8100f6cf9cde942b6c1378d11`. It is a 720p streaming interactive world model with autoregressive chunks, long-horizon memory, released 5B base/distilled weights, and an Apache-2.0 code release. It is a much cleaner foundation for a public systems project than source-available models with territorial or hosted-service restrictions.

> This is not an official SkyworkAI project. Model weights remain governed by their upstream terms. No weights are redistributed here.

## Why this model

| Candidate | Why it is interesting | Why it is or is not the base |
|---|---|---|
| **Matrix-Game 3.0 (5B)** | Streaming autoregressive world model; long-horizon memory; 720p; distilled few-step path; existing INT8, FlashAttention, sequence parallelism, and asynchronous VAE hooks | **Selected.** Recent, task-trained, manageable 5B scale, weights released, permissive code license, and an unusually rich latency surface |
| HY-World 1.5 / WorldPlay | Strong training stack, action control, memory, 5B/8B checkpoints, and a four-step distilled model | Excellent reference architecture, but its community license excludes several territories and is not a clean open-source base for a reusable portfolio project |
| Matrix-Game 2.0 | Earlier real-time interactive world model | Useful historical baseline, but 3.0 is the more relevant systems target |
| Generic Wan/Hunyuan video generators | Larger fine-tuning ecosystems | Strong backbones, but they are not themselves task-trained interactive world models |

## What is implemented

### Project 1 — Real performance report

`mgs-profile` runs Matrix-Game or an arbitrary training command while collecting:

- end-to-end and phase latency;
- forward, backward, optimizer, dataloader, VAE, checkpoint, and save time;
- average and peak GPU utilization;
- allocated/reserved/physical HBM;
- estimated forward/training FLOPs, achieved TFLOP/s, and MFU;
- NCCL fraction from a PyTorch Chrome trace;
- dataloader stall and checkpoint-pause fractions;
- frames per GPU-hour;
- machine-readable JSON and a before/after Markdown report.

The repository deliberately ships **no invented benchmark numbers**. The report becomes credible only after the included scripts are run on the target A/H-series GPU node with the real weights and workload.

### Project 2 — Fused AdaLN Triton kernel

Matrix-Game repeatedly computes

```text
LayerNorm(x) * (1 + scale) + shift
```

before attention and the MLP. `matrixgame_systems.kernels.adaln` provides:

- a PyTorch reference;
- a Triton forward kernel with FP32 statistics;
- a Triton analytical backward kernel;
- eager fallback for unsupported/broadcast shapes;
- correctness and gradient tests for FP16/BF16/FP32;
- benchmarks at Matrix-Game's hidden width `5120`;
- an idempotent patcher for the pinned upstream model;
- Nsight Systems and Nsight Compute launch scripts.

### Project 3 — Restartable and scalable training

The distributed package includes:

- Ulysses-style sequence-parallel all-to-all transforms;
- a stateful no-duplication distributed sampler;
- `torch.distributed.checkpoint` sharded save/load;
- asynchronous checkpoint staging with one in-flight checkpoint and atomic commit markers;
- RNG, optimizer, scheduler, dataloader, config, and Git-commit capture;
- deterministic validation seeds and sample selection;
- restart-equivalence and sample-coverage tests;
- throughput-regression gates;
- an eight-GPU launch template and scaling-report schema.

Concrete acceptance target:

> Eight-GPU training resumes from an arbitrary committed checkpoint with matching next-step loss, zero duplicated sample IDs, checkpoint foreground pause below 2% of step time, and a documented 1/2/4/8-GPU parallel-efficiency curve.

### Project 4 — Inference service

The FastAPI service includes:

- bounded request queue;
- compatibility-aware dynamic batch scheduler;
- HBM reservation/admission control;
- model warm-up;
- cancellation and per-request event streams over SSE;
- p50/p95 latency and throughput metrics;
- Prometheus endpoint and health/readiness checks;
- worker failure propagation and restart-safe request states;
- generic fixed-shape CUDA-graph runner;
- BF16/INT8/experimental-FP8 configuration hooks;
- Docker packaging and a concurrency load generator.

The upstream Matrix-Game pipeline assumes batch size one in several memory/control paths. The service therefore defaults to `max_batch_size=1` for the real backend rather than pretending that queue coalescing is true tensor batching. The scheduler and batch API are fully implemented; enabling `max_batch_size>1` is guarded until the upstream state-machine adapter passes the included batch-equivalence contract.

## Quick start

```bash
git clone https://github.com/Abecid/matrixgame-systems.git
cd matrixgame-systems
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,gpu,service]'

# Clone and pin Matrix-Game 3.0 without vendoring its code or weights.
bash scripts/bootstrap_upstream.sh

# Download weights using the upstream instructions, then point this variable at them.
export MATRIX_GAME_CKPT=/path/to/Matrix-Game-3.0
```

### Run a profiled inference

```bash
mgs-profile matrixgame \
  --upstream third_party/Matrix-Game/Matrix-Game-3 \
  --run-dir runs/bf16_baseline \
  --frames 97 --num-gpus 1 --peak-tflops 989 \
  -- \
  --size '704*1280' \
  --ckpt_dir "$MATRIX_GAME_CKPT" \
  --image demo_images/001/image.png \
  --prompt 'A navigable animated city.' \
  --num_iterations 2 --num_inference_steps 3 \
  --fa_version 3 --compile_vae

mgs-report runs/bf16_baseline
```

For a multi-GPU run:

```bash
torchrun --standalone --nproc_per_node=8 -m matrixgame_systems.profiling.matrixgame_runner \
  --upstream third_party/Matrix-Game/Matrix-Game-3 \
  --run-dir runs/8gpu \
  --frames 97 --num-gpus 8 --peak-tflops 989 \
  -- --ulysses_size 8 --dit_fsdp --t5_fsdp \
  --size '704*1280' --ckpt_dir "$MATRIX_GAME_CKPT" \
  --image demo_images/001/image.png --prompt 'A navigable animated city.' \
  --num_iterations 2 --num_inference_steps 3 --fa_version 3
```

### Run and integrate the Triton kernel

```bash
mgs-kernel-bench --hidden-size 5120 --rows 880 3520 8800 --dtype bfloat16
python -m matrixgame_systems.integrations.patch_matrix_game \
  --upstream third_party/Matrix-Game/Matrix-Game-3

export MGS_USE_FUSED_ADALN=1
```

### Profile a training step

```python
from matrixgame_systems.profiling.events import PhaseRecorder

prof = PhaseRecorder(run_dir="runs/train", frames_per_step=16, num_gpus=8)
for step in range(max_steps):
    with prof.phase("dataloader"):
        batch = next(loader)
    with prof.phase("forward"):
        loss = model(batch)
    with prof.phase("backward"):
        loss.backward()
    with prof.phase("optimizer"):
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    prof.mark_step(step)
prof.close()
```

### Start the service

```bash
export MATRIX_GAME_UPSTREAM=$PWD/third_party/Matrix-Game/Matrix-Game-3
export MATRIX_GAME_CKPT=/path/to/Matrix-Game-3.0
mgs-serve --config configs/service.yaml

curl -N -X POST http://localhost:8000/v1/generations \
  -H 'content-type: application/json' \
  -d '{"prompt":"A navigable city","image_path":"/data/start.png","num_iterations":2}'
```

## Reproducible experiment matrix

Run the same prompt/image/seed in this order:

1. BF16 + synchronous VAE + eager AdaLN;
2. BF16 + asynchronous VAE;
3. BF16 + asynchronous VAE + fused AdaLN;
4. upstream INT8 Q/K/V/O + fused AdaLN;
5. 1/2/4/8-GPU sequence-parallel scaling.

The report generator computes deltas only when workload fingerprints match, preventing fake speedups from changed resolution, frame count, steps, prompt, or precision.

## Repository layout

```text
src/matrixgame_systems/profiling    phase timers, NVML sampling, trace parser, reports
src/matrixgame_systems/kernels      fused AdaLN Triton forward/backward + benchmarks
src/matrixgame_systems/distributed  sequence parallelism, sampler, async DCP, regressions
src/matrixgame_systems/serving      queue, batching, admission, SSE API, metrics
src/matrixgame_systems/integrations pinned Matrix-Game patcher
reports/                             report template; real run outputs are gitignored
scripts/                             Nsight, Slurm, bootstrap, and service commands
```

## Validation boundary

CPU unit tests and restart logic can run in ordinary CI. CUDA/Triton, NCCL, FP8, CUDA graphs, Matrix-Game quality equivalence, and the final before/after traces require the actual NVIDIA GPU environment and released weights. CI skips those tests rather than manufacturing success.

## License

This repository's original code is Apache-2.0. Upstream Matrix-Game code and model artifacts are separate works and are not redistributed.


> Full candidate comparison and decision record: [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md)

