from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path

from .models import RequestState

ProgressCallback = Callable[[RequestState, str, dict], Awaitable[None]]


class InferenceBackend(ABC):
    max_batch_size: int = 1
    supports_cuda_graphs: bool = False

    async def load(self) -> None:
        return None

    async def warmup(self) -> None:
        return None

    @abstractmethod
    async def generate_batch(
        self, requests: list[RequestState], progress: ProgressCallback
    ) -> list[str]:
        raise NotImplementedError


class MockBackend(InferenceBackend):
    def __init__(self, *, delay_s: float = 0.01, max_batch_size: int = 8, output_dir: str = "/tmp") -> None:
        self.delay_s = delay_s
        self.max_batch_size = max_batch_size
        self.output_dir = Path(output_dir)

    async def generate_batch(
        self, requests: list[RequestState], progress: ProgressCallback
    ) -> list[str]:
        for fraction in (0.25, 0.5, 0.75, 1.0):
            await asyncio.sleep(self.delay_s)
            for request in requests:
                if not request.cancel_event.is_set():
                    await progress(request, "progress", {"fraction": fraction})
        return [str(self.output_dir / f"{request.request_id}.mp4") for request in requests]
