from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def deterministic_seed(base_seed: int, sample_id: str | int, step: int = 0) -> int:
    payload = f"{base_seed}:{sample_id}:{step}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**31)


def deterministic_validation_subset(
    sample_ids: Sequence[T], count: int, *, seed: int
) -> list[T]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count >= len(sample_ids):
        return list(sample_ids)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(sample_ids)), count))
    return [sample_ids[index] for index in indices]
