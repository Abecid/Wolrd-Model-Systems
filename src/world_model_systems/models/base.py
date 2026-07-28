from __future__ import annotations

from pathlib import Path

from world_model_systems.core.spec import LaunchRequest, ModelSpec
from world_model_systems.integrations.patching import require_revision


class BaseAdapter:
    spec: ModelSpec

    def validate_upstream(self, upstream: Path, *, allow_unpinned: bool = False) -> None:
        require_revision(upstream.resolve(), self.spec.pinned_revision, allow_unpinned=allow_unpinned)

    def apply_optimization(
        self,
        optimization_id: str,
        upstream: Path,
        *,
        allow_unpinned: bool = False,
    ) -> list[Path]:
        raise KeyError(f"{self.spec.id} does not implement source patch '{optimization_id}'")

    def build_inference_command(self, request: LaunchRequest) -> list[str]:
        raise NotImplementedError
