#!/usr/bin/env bash
set -euo pipefail
: "${IMAGE:?Set IMAGE to a readable initial-frame path on the service host}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for concurrency in 1 2 4 8; do
  mgs-loadtest \
    --base-url "${BASE_URL:-http://localhost:8000}" \
    --requests "${REQUESTS:-8}" \
    --concurrency "$concurrency" \
    --image "$IMAGE" \
    --output "$ROOT/runs/service-c${concurrency}.json"
done
