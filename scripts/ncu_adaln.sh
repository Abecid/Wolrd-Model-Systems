#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT/runs/ncu-adaln}"
mkdir -p "$RUN_DIR"

ncu \
  --set full \
  --target-processes all \
  --kernel-name regex:adaln_forward \
  --launch-skip 25 \
  --launch-count 5 \
  --export "$RUN_DIR/adaln" \
  python -m matrixgame_systems.kernels.benchmark \
    --rows "${ROWS:-3520}" --hidden-size 5120 --dtype "${DTYPE:-bfloat16}" \
    --output "$RUN_DIR/benchmark.json"
