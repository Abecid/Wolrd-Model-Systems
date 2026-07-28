#!/usr/bin/env bash
set -euo pipefail

variant=${1:-baseline}
: "${LINGBOT_WORLD_V2_CKPT:?set LINGBOT_WORLD_V2_CKPT}"
: "${LINGBOT_WORLD_V2_IMAGE:?set LINGBOT_WORLD_V2_IMAGE}"
UPSTREAM=${LINGBOT_WORLD_V2_UPSTREAM:-third_party/lingbot-world-v2}
PROMPT=${LINGBOT_WORLD_V2_PROMPT:-A character walks forward through the world.}
NUM_GPUS=${NUM_GPUS:-8}

case "$variant" in
  baseline) run_dir=runs/lingbot_v2/bf16_baseline ;;
  fused-adaln)
    wms patch lingbot-world-v2 fused-adaln --upstream "$UPSTREAM"
    export WMS_USE_FUSED_ADALN=1
    run_dir=runs/lingbot_v2/fused_adaln
    ;;
  fp32-rope)
    wms patch lingbot-world-v2 fp32-causal-rope --upstream "$UPSTREAM"
    run_dir=runs/lingbot_v2/fp32_rope
    ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac

wms-profile lingbot-world-v2 \
  --upstream "$UPSTREAM" \
  --checkpoint "$LINGBOT_WORLD_V2_CKPT" \
  --image "$LINGBOT_WORLD_V2_IMAGE" \
  --prompt "$PROMPT" \
  --frames 361 --num-gpus "$NUM_GPUS" \
  --run-dir "$run_dir"
