#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v gh >/dev/null || { echo "Install GitHub CLI: brew install gh" >&2; exit 1; }
gh auth status >/dev/null || { echo "Authenticate first: gh auth login" >&2; exit 1; }

if gh repo view Abecid/matrixgame-systems >/dev/null 2>&1; then
  echo "Abecid/matrixgame-systems already exists; refusing to overwrite it." >&2
  exit 1
fi

gh repo create Abecid/matrixgame-systems \
  --public \
  --description "GPU profiling, Triton kernels, resilient distributed training, and serving for Matrix-Game 3.0" \
  --source . \
  --remote origin \
  --push
