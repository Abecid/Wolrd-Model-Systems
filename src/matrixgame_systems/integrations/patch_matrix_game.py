from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PINNED_COMMIT = "71c3cd7f741311f8100f6cf9cde942b6c1378d11"
IMPORT_LINE = "from matrixgame_systems.kernels.adaln import fused_adaln\n"

REPLACEMENTS = {
    "(self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)).to(x.dtype)": (
        "fused_adaln(x, e[1].squeeze(2), e[0].squeeze(2), eps=self.norm1.eps)"
    ),
    "(self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)).to(self.ffn[0].weight.dtype)": (
        "fused_adaln(x, e[4].squeeze(2), e[3].squeeze(2), eps=self.norm2.eps).to(self.ffn[0].weight.dtype)"
    ),
}


def _git_commit(repo: Path) -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def patch(upstream: Path, *, allow_unpinned: bool = False) -> Path:
    root = upstream.resolve()
    repository_root = root.parent if root.name == "Matrix-Game-3" else root
    commit = _git_commit(repository_root)
    if commit and commit != PINNED_COMMIT and not allow_unpinned:
        raise RuntimeError(
            f"Refusing to patch commit {commit}; expected {PINNED_COMMIT}. "
            "Pass --allow-unpinned only after reviewing the diff."
        )
    model_file = root / "wan" / "modules" / "model.py"
    if not model_file.exists():
        raise FileNotFoundError(model_file)
    source = model_file.read_text(encoding="utf-8")
    original_sha = hashlib.sha256(source.encode()).hexdigest()
    if IMPORT_LINE not in source:
        anchor = "import torch.nn.functional as torch_F\n"
        if anchor not in source:
            raise RuntimeError("Upstream import anchor changed")
        source = source.replace(anchor, anchor + IMPORT_LINE, 1)
    changed = 0
    for old, new in REPLACEMENTS.items():
        if new in source:
            continue
        if old not in source:
            raise RuntimeError(f"Upstream AdaLN expression changed; missing: {old}")
        source = source.replace(old, new)
        changed += 1
    model_file.write_text(source, encoding="utf-8")
    marker = root / ".mgs_adaln_patch"
    marker.write_text(
        f"upstream_commit={commit}\noriginal_sha256={original_sha}\nreplacements={changed}\n",
        encoding="utf-8",
    )
    return model_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch pinned Matrix-Game 3 to call fused AdaLN")
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--allow-unpinned", action="store_true")
    args = parser.parse_args()
    print(patch(Path(args.upstream), allow_unpinned=args.allow_unpinned))


if __name__ == "__main__":
    main()
