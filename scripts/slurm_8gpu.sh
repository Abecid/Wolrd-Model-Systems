#!/usr/bin/env bash
#SBATCH --job-name=mgs-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
: "${MATRIX_GAME_CKPT:?Set MATRIX_GAME_CKPT}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${MATRIX_GAME_UPSTREAM:-$ROOT/third_party/Matrix-Game/Matrix-Game-3}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/8gpu-${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR" "$ROOT/logs"

export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1

srun --ntasks=8 --gpus-per-task=1 bash -lc '
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export WORLD_SIZE=$SLURM_NTASKS
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
  export MASTER_PORT=${MASTER_PORT:-29500}
  python -m matrixgame_systems.profiling.matrixgame_runner \
    --upstream "'"$UPSTREAM"'" --run-dir "'"$RUN_DIR"'" --frames "${FRAMES:-97}" \
    --num-gpus 8 --peak-tflops "${PEAK_TFLOPS:-989}" \
    -- --ulysses_size 8 --dit_fsdp --t5_fsdp --size "${SIZE:-704*1280}" \
    --ckpt_dir "'"$MATRIX_GAME_CKPT"'" \
    --image "${IMAGE:-'"$UPSTREAM"'/demo_images/001/image.png}" \
    --prompt "${PROMPT:-A navigable animated city.}" \
    --num_iterations "${NUM_ITERATIONS:-2}" --num_inference_steps "${NUM_INFERENCE_STEPS:-3}" \
    --fa_version "${FA_VERSION:-3}"
'

mgs-report "$RUN_DIR"
