from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class Reservation:
    token: int
    bytes_reserved: int


class HBMAdmissionController:
    """Conservative HBM reservation independent of allocator-reported free memory."""

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        estimated_model_bytes: int = 0,
        estimated_request_bytes: int = 0,
        safety_fraction: float = 0.12,
        total_bytes_override: int | None = None,
    ) -> None:
        if not 0 <= safety_fraction < 1:
            raise ValueError("safety_fraction must be in [0,1)")
        self.device = device
        self.estimated_model_bytes = int(estimated_model_bytes)
        self.estimated_request_bytes = int(estimated_request_bytes)
        self.safety_fraction = safety_fraction
        self.total_bytes_override = total_bytes_override
        self._reserved = 0
        self._next_token = 0
        self._lock = asyncio.Lock()

    def total_bytes(self) -> int:
        if self.total_bytes_override is not None:
            return self.total_bytes_override
        try:
            import torch

            if torch.cuda.is_available():
                with torch.cuda.device(self.device):
                    _, total = torch.cuda.mem_get_info()
                return int(total)
        except Exception:
            pass
        return 0

    def capacity_bytes(self) -> int:
        total = self.total_bytes()
        return max(0, int(total * (1.0 - self.safety_fraction)) - self.estimated_model_bytes)

    async def reserve(self, batch_size: int) -> Reservation | None:
        requested = self.estimated_request_bytes * batch_size
        async with self._lock:
            if self._reserved + requested > self.capacity_bytes():
                return None
            self._next_token += 1
            reservation = Reservation(self._next_token, requested)
            self._reserved += requested
            return reservation

    async def release(self, reservation: Reservation) -> None:
        async with self._lock:
            self._reserved = max(0, self._reserved - reservation.bytes_reserved)

    @property
    def reserved_bytes(self) -> int:
        return self._reserved
