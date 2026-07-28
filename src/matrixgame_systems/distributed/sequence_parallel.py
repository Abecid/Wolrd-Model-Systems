from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


def _world_size(group: dist.ProcessGroup | None = None) -> int:
    return dist.get_world_size(group) if dist.is_available() and dist.is_initialized() else 1


def all_to_all_4d(
    tensor: torch.Tensor,
    *,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Swap one evenly divisible tensor dimension across ranks.

    This is the core Ulysses transform: local sequence shards are exchanged for
    local head shards before attention, then the inverse exchange restores the
    original layout.
    """

    world = _world_size(group)
    if world == 1:
        return tensor
    scatter_dim %= tensor.ndim
    gather_dim %= tensor.ndim
    if tensor.shape[scatter_dim] % world != 0:
        raise ValueError(
            f"Dimension {scatter_dim}={tensor.shape[scatter_dim]} is not divisible by world_size={world}"
        )
    send = [chunk.contiguous() for chunk in tensor.chunk(world, dim=scatter_dim)]
    receive = [torch.empty_like(send[0]) for _ in range(world)]
    dist.all_to_all(receive, send, group=group)
    return torch.cat(receive, dim=gather_dim).contiguous()


def ulysses_pre_attention(
    tensor: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """[B, S/world, H, D] -> [B, S, H/world, D]."""

    if tensor.ndim != 4:
        raise ValueError(f"Expected [B,S,H,D], got {tuple(tensor.shape)}")
    return all_to_all_4d(tensor, scatter_dim=2, gather_dim=1, group=group)


def ulysses_post_attention(
    tensor: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """[B, S, H/world, D] -> [B, S/world, H, D]."""

    if tensor.ndim != 4:
        raise ValueError(f"Expected [B,S,H,D], got {tuple(tensor.shape)}")
    return all_to_all_4d(tensor, scatter_dim=1, gather_dim=2, group=group)


@dataclass(frozen=True)
class ScalingPoint:
    gpus: int
    step_time_s: float
    samples_per_step: int

    @property
    def throughput(self) -> float:
        return self.samples_per_step / self.step_time_s


def parallel_efficiency(single_gpu: ScalingPoint, scaled: ScalingPoint) -> float:
    if single_gpu.gpus != 1:
        raise ValueError("single_gpu point must use exactly one GPU")
    return scaled.throughput / (single_gpu.throughput * scaled.gpus)
