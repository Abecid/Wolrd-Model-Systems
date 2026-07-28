from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .git import git_metadata
from .hardware import torch_hardware_metadata


def stable_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def environment_manifest(
    *,
    command: list[str] | None = None,
    config: dict[str, Any] | None = None,
    repo: str | Path = ".",
) -> dict[str, Any]:
    manifest = {
        "created_unix_s": time.time(),
        "command": command,
        "cwd": os.getcwd(),
        "git": git_metadata(repo),
        "hardware": torch_hardware_metadata(),
        "config": config or {},
    }
    try:
        manifest["pip_freeze"] = subprocess.check_output(
            ["python", "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
        ).splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        manifest["pip_freeze"] = []
    manifest["fingerprint"] = stable_fingerprint(
        {"command": command, "config": config or {}, "git": manifest["git"]}
    )
    return manifest


def write_json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, destination)
