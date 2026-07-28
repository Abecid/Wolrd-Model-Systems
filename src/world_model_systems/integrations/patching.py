from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


class PatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourcePatch:
    path: str
    replacements: tuple[tuple[str, str], ...]
    import_anchor: str | None = None
    import_line: str | None = None


@dataclass(frozen=True)
class PatchManifest:
    model_id: str
    optimization_id: str
    upstream_revision: str | None
    files: tuple[dict[str, str | int], ...]


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def require_revision(upstream: Path, expected: str, *, allow_unpinned: bool) -> str | None:
    revision = git_revision(upstream)
    if revision and revision != expected and not allow_unpinned:
        raise PatchError(
            f"Refusing to patch revision {revision}; expected {expected}. "
            "Use --allow-unpinned only after reviewing the upstream diff."
        )
    return revision


def apply_source_patches(
    *,
    model_id: str,
    optimization_id: str,
    upstream: Path,
    expected_revision: str,
    patches: tuple[SourcePatch, ...],
    marker_name: str,
    allow_unpinned: bool = False,
) -> list[Path]:
    upstream = upstream.resolve()
    revision = require_revision(upstream, expected_revision, allow_unpinned=allow_unpinned)
    changed_paths: list[Path] = []
    records: list[dict[str, str | int]] = []
    for patch in patches:
        target = upstream / patch.path
        if not target.exists():
            raise FileNotFoundError(target)
        source = target.read_text(encoding="utf-8")
        before_sha = hashlib.sha256(source.encode()).hexdigest()
        if patch.import_line and patch.import_line not in source:
            if not patch.import_anchor or patch.import_anchor not in source:
                raise PatchError(f"Import anchor changed in {patch.path}")
            source = source.replace(patch.import_anchor, patch.import_anchor + patch.import_line, 1)
        replacements = 0
        for old, new in patch.replacements:
            if new in source:
                continue
            if old not in source:
                raise PatchError(f"Upstream expression changed in {patch.path}: {old}")
            source = source.replace(old, new)
            replacements += 1
        after_sha = hashlib.sha256(source.encode()).hexdigest()
        if after_sha != before_sha:
            target.write_text(source, encoding="utf-8")
            changed_paths.append(target)
        records.append(
            {
                "path": patch.path,
                "before_sha256": before_sha,
                "after_sha256": after_sha,
                "replacements": replacements,
            }
        )
    manifest = PatchManifest(model_id, optimization_id, revision, tuple(records))
    (upstream / marker_name).write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return changed_paths
