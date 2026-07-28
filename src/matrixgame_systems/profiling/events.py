from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PhaseEvent:
    name: str
    start_wall_s: float
    end_wall_s: float
    wall_ms: float
    cuda_ms: float | None
    step: int | None
    rank: int


@dataclass
class _PendingCudaEvent:
    event: PhaseEvent
    start: object
    end: object


class PhaseRecorder:
    """Low-overhead nested phase recorder using CUDA events and NVTX ranges.

    CUDA events are synchronized only when `flush()`/`close()` is called, avoiding
    a device-wide synchronization at every phase boundary.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        frames_per_step: int = 0,
        num_gpus: int = 1,
        rank: int | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.frames_per_step = frames_per_step
        self.num_gpus = num_gpus
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else rank
        self.current_step: int | None = None
        self.events: list[PhaseEvent] = []
        self._pending: list[_PendingCudaEvent] = []
        self._lock = threading.Lock()
        self._closed = False
        self._torch = None
        try:
            import torch

            self._torch = torch
            self._cuda = torch.cuda.is_available()
            if self._cuda:
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            self._cuda = False

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start_wall = time.perf_counter()
        start_cuda = end_cuda = None
        if self._cuda and self._torch is not None:
            start_cuda = self._torch.cuda.Event(enable_timing=True)
            end_cuda = self._torch.cuda.Event(enable_timing=True)
            self._torch.cuda.nvtx.range_push(name)
            start_cuda.record()
        try:
            yield
        finally:
            if self._cuda and self._torch is not None and end_cuda is not None:
                end_cuda.record()
                self._torch.cuda.nvtx.range_pop()
            end_wall = time.perf_counter()
            event = PhaseEvent(
                name=name,
                start_wall_s=start_wall,
                end_wall_s=end_wall,
                wall_ms=(end_wall - start_wall) * 1000.0,
                cuda_ms=None,
                step=self.current_step,
                rank=self.rank,
            )
            with self._lock:
                self.events.append(event)
                if start_cuda is not None and end_cuda is not None:
                    self._pending.append(_PendingCudaEvent(event, start_cuda, end_cuda))

    def mark_step(self, step: int) -> None:
        self.current_step = int(step)

    def flush(self) -> Path:
        with self._lock:
            if self._cuda and self._torch is not None and self._pending:
                self._torch.cuda.synchronize()
                for pending in self._pending:
                    pending.event.cuda_ms = float(pending.start.elapsed_time(pending.end))
                self._pending.clear()
            path = self.run_dir / f"phases.rank{self.rank}.json"
            payload = {
                "rank": self.rank,
                "frames_per_step": self.frames_per_step,
                "num_gpus": self.num_gpus,
                "events": [asdict(event) for event in self.events],
            }
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, path)
            return path

    def memory_summary(self) -> dict[str, int | None]:
        if not self._cuda or self._torch is None:
            return {
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
                "allocated_bytes": None,
                "reserved_bytes": None,
            }
        return {
            "peak_allocated_bytes": int(self._torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(self._torch.cuda.max_memory_reserved()),
            "allocated_bytes": int(self._torch.cuda.memory_allocated()),
            "reserved_bytes": int(self._torch.cuda.memory_reserved()),
        }

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        memory_path = self.run_dir / f"torch_memory.rank{self.rank}.json"
        memory_path.write_text(json.dumps(self.memory_summary(), indent=2), encoding="utf-8")
        self._closed = True

    def __enter__(self) -> PhaseRecorder:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
