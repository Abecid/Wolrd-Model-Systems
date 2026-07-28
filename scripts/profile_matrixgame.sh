#!/usr/bin/env bash
set -euo pipefail

: "${MATRIX_GAME_CKPT:?Set MATRIX_GAME_CKPT to the released Matrix-Game 3.0 checkpoint directory}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${MATRIX_GAME_UPSTREAM:-$ROOT/third_party/Matrix-Game/Matrix-Game-3}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/matrixgame-bf16-baseline}"
IMAGE="${IMAGE:-$UPSTREAM/demo_images/001/image.png}"
PROMPT="${PROMPT:-A navigable animated city with moving traffic.}"

mgs-profile matrixgame \
  --upstream "$UPSTREAM" \
  --run-dir "$RUN_DIR" \
  --frames "${FRAMES:-97}" \
  --num-gpus "${WORLD_SIZE:-1}" \
  --peak-tflops "${PEAK_TFLOPS:-989}" \
  -- \
  --size "${SIZE:-704*1280}" \
  --ckpt_dir "$MATRIX_GAME_CKPT" \
  --image "$IMAGE" \
  --prompt "$PROMPT" \
  --num_iterations "${NUM_ITERATIONS:-2}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-3}" \
  --fa_version "${FA_VERSION:-3}" \
  --compile_vae

mgs-report "$RUN_DIR"
