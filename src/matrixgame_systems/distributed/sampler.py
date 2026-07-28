from __future__ import annotations

import math
from collections.abc import Iterator, Sized
from typing import Any

import torch
from torch.utils.data import Sampler


class StatefulDistributedSampler(Sampler[int]):
    """Deterministic distributed sampler with an exact resumable cursor.

    With `drop_last=True`, every sample ID appears on at most one rank per epoch.
    This is the mode used by the no-duplication acceptance test.
    """

    def __init__(
        self,
        dataset: Sized,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} outside [0, {num_replicas})")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.cursor = 0

    def _global_indices(self) -> list[int]:
        size = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(size, generator=generator).tolist()
        else:
            indices = list(range(size))
        if self.drop_last:
            usable = size - size % self.num_replicas
            return indices[:usable]
        total = math.ceil(size / self.num_replicas) * self.num_replicas
        if total > size:
            indices += indices[: total - size]
        return indices

    def rank_indices(self) -> list[int]:
        return self._global_indices()[self.rank :: self.num_replicas]

    def __iter__(self) -> Iterator[int]:
        indices = self.rank_indices()
        while self.cursor < len(indices):
            index = indices[self.cursor]
            self.cursor += 1
            yield index

    def __len__(self) -> int:
        return len(self.rank_indices()) - self.cursor

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.cursor = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "seed": self.seed,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "drop_last": self.drop_last,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("seed", "num_replicas", "rank", "drop_last"):
            if state.get(key) != getattr(self, key):
                raise ValueError(f"Sampler state mismatch for {key}: {state.get(key)} != {getattr(self, key)}")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        if self.cursor > len(self.rank_indices()):
            raise ValueError("Sampler cursor exceeds rank-local epoch length")
