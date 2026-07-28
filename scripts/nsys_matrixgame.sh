#!/usr/bin/env bash
set -euo pipefail

: "${MATRIX_GAME_CKPT:?Set MATRIX_GAME_CKPT}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${MATRIX_GAME_UPSTREAM:-$ROOT/third_party/Matrix-Game/Matrix-Game-3}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/nsys-matrixgame}"
mkdir -p "$RUN_DIR"

nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --sample=none \
  --force-overwrite=true \
  --output "$RUN_DIR/timeline" \
  python -m matrixgame_systems.profiling.matrixgame_runner \
    --upstream "$UPSTREAM" --run-dir "$RUN_DIR" --frames "${FRAMES:-97}" \
    --num-gpus 1 --peak-tflops "${PEAK_TFLOPS:-989}" \
    -- --size "${SIZE:-704*1280}" --ckpt_dir "$MATRIX_GAME_CKPT" \
    --image "${IMAGE:-$UPSTREAM/demo_images/001/image.png}" \
    --prompt "${PROMPT:-A navigable animated city.}" \
    --num_iterations "${NUM_ITERATIONS:-2}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS:-3}" \
    --fa_version "${FA_VERSION:-3}" --compile_vae

nsys stats --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum "$RUN_DIR/timeline.nsys-rep" \
  > "$RUN_DIR/nsys-summary.txt"
