from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RequestStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class GenerationSpec:
    prompt: str
    image_path: str
    height: int = 704
    width: int = 1280
    num_iterations: int = 2
    num_inference_steps: int = 3
    seed: int = 42
    precision: str = "bf16"
    output_name: str | None = None

    def compatibility_key(self) -> tuple[Any, ...]:
        return (
            self.height,
            self.width,
            self.num_iterations,
            self.num_inference_steps,
            self.precision,
        )


@dataclass
class GenerationEvent:
    kind: str
    request_id: str
    timestamp_s: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RequestState:
    spec: GenerationSpec
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: RequestStatus = RequestStatus.QUEUED
    created_s: float = field(default_factory=time.time)
    started_s: float | None = None
    completed_s: float | None = None
    output_path: str | None = None
    error: str | None = None
    events: asyncio.Queue[GenerationEvent] = field(default_factory=asyncio.Queue)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)

    async def emit(self, kind: str, **payload: Any) -> None:
        await self.events.put(GenerationEvent(kind, self.request_id, payload=payload))

    def snapshot(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "created_s": self.created_s,
            "started_s": self.started_s,
            "completed_s": self.completed_s,
            "output_path": self.output_path,
            "error": self.error,
            "spec": asdict(self.spec),
        }
