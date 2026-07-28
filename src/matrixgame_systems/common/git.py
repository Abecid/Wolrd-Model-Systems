from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


def _run_git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def git_metadata(repo: str | Path = ".") -> dict[str, Any]:
    root = Path(repo).resolve()
    commit = _run_git(root, "rev-parse", "HEAD")
    branch = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git(root, "status", "--porcelain")
    diff = _run_git(root, "diff", "--binary") or ""
    return {
        "root": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None,
    }
