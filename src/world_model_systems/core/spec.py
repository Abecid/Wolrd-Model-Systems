from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class Capability(str, Enum):
    CAUSAL = "causal"
    STREAMING = "streaming"
    ACTION_CONDITIONED = "action_conditioned"
    CAMERA_CONDITIONED = "camera_conditioned"
    KV_CACHE = "kv_cache"
    FEW_STEP = "few_step"
    TRAINING = "training"
    SEQUENCE_PARALLEL = "sequence_parallel"
    FSDP = "fsdp"
    ASYNC_VAE = "async_vae"
    QUANTIZATION = "quantization"
    DYNAMIC_BATCHING = "dynamic_batching"


@dataclass(frozen=True)
class LicensePolicy:
    identifier: str
    commercial_use: bool
    redistribution: bool
    hosted_service: bool
    notes: str = ""
    acceptance_required: bool = False


@dataclass(frozen=True)
class OptimizationSpec:
    id: str
    category: str
    objective: tuple[str, ...]
    description: str
    implemented: bool = True
    experimental: bool = False
    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    capabilities: tuple[Capability, ...] = ()


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    family: str
    organization: str
    repository: str
    pinned_revision: str
    model_size: str
    license: LicensePolicy
    capabilities: frozenset[Capability]
    default_checkpoint: str | None = None
    default_resolution: tuple[int, int] | None = None
    default_frames: int | None = None
    optimizations: tuple[OptimizationSpec, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["capabilities"] = sorted(item.value for item in self.capabilities)
        for item in payload["optimizations"]:
            item["capabilities"] = [cap.value for cap in item["capabilities"]]
        return payload


@dataclass(frozen=True)
class LaunchRequest:
    upstream: Path
    checkpoint: Path
    prompt: str
    image: Path
    output_dir: Path
    frames: int | None = None
    num_gpus: int = 1
    precision: str = "bf16"
    extra_args: tuple[str, ...] = ()


class ModelAdapter(Protocol):
    spec: ModelSpec

    def build_inference_command(self, request: LaunchRequest) -> list[str]: ...

    def apply_optimization(
        self,
        optimization_id: str,
        upstream: Path,
        *,
        allow_unpinned: bool = False,
    ) -> list[Path]: ...

    def validate_upstream(self, upstream: Path, *, allow_unpinned: bool = False) -> None: ...
