#!/usr/bin/env bash
set -euo pipefail

model=${1:-}
root=${2:-third_party}
mkdir -p "$root"

case "$model" in
  matrix-game-3)
    repo=https://github.com/SkyworkAI/Matrix-Game.git
    destination="$root/Matrix-Game"
    revision=71c3cd7f741311f8100f6cf9cde942b6c1378d11
    ;;
  lingbot-world-v2)
    repo=https://github.com/Robbyant/lingbot-world-v2.git
    destination="$root/lingbot-world-v2"
    revision=2648877f763a06cc743bcd919936da4d25f12e7b
    ;;
  hy-worldplay-1.5)
    repo=https://github.com/Tencent-Hunyuan/HY-WorldPlay.git
    destination="$root/HY-WorldPlay"
    revision=1588e1336e842b03b0a7860c654ebd7c46bb065e
    echo "HY-WorldPlay uses a custom Tencent license with territorial/use restrictions." >&2
    echo "Review the upstream license before continuing." >&2
    ;;
  *)
    echo "usage: $0 {matrix-game-3|lingbot-world-v2|hy-worldplay-1.5} [third_party_dir]" >&2
    exit 2
    ;;
esac

if [[ ! -d "$destination/.git" ]]; then
  git clone "$repo" "$destination"
fi
git -C "$destination" fetch --all --tags
git -C "$destination" checkout --detach "$revision"
actual=$(git -C "$destination" rev-parse HEAD)
[[ "$actual" == "$revision" ]] || { echo "revision mismatch: $actual" >&2; exit 1; }
echo "$model ready at $destination ($actual)"
