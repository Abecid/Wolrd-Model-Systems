#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/third_party/Matrix-Game}"
COMMIT="71c3cd7f741311f8100f6cf9cde942b6c1378d11"

if [[ ! -d "$DEST/.git" ]]; then
  mkdir -p "$(dirname "$DEST")"
  git clone https://github.com/SkyworkAI/Matrix-Game.git "$DEST"
fi

git -C "$DEST" fetch --tags origin
git -C "$DEST" checkout --detach "$COMMIT"
actual="$(git -C "$DEST" rev-parse HEAD)"
[[ "$actual" == "$COMMIT" ]] || { echo "Pinned commit mismatch: $actual" >&2; exit 1; }

echo "Matrix-Game pinned at $actual"
echo "Upstream path: $DEST/Matrix-Game-3"
