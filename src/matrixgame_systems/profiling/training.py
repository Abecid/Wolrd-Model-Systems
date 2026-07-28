from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from .events import PhaseRecorder
from .torch_trace import torch_trace


class TrainingProfiler:
    """Named phase profiler plus an optional bounded PyTorch trace window."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        frames_per_step: int,
        num_gpus: int,
        trace_start_step: int | None = None,
        trace_steps: int = 1,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.phases = PhaseRecorder(
            self.run_dir, frames_per_step=frames_per_step, num_gpus=num_gpus
        )
        self.trace_start_step = trace_start_step
        self.trace_steps = trace_steps
        self._trace_context: Any = None
        self._trace_profiler: Any = None

    def begin_step(self, step: int) -> None:
        self.phases.mark_step(step)
        if self.trace_start_step is not None and step == self.trace_start_step:
            self._trace_context = torch_trace(self.run_dir / "trace.json", enabled=True)
            self._trace_profiler = self._trace_context.__enter__()

    def end_step(self, step: int) -> None:
        if (
            self._trace_context is not None
            and self.trace_start_step is not None
            and step + 1 >= self.trace_start_step + self.trace_steps
        ):
            self._trace_context.__exit__(None, None, None)
            self._trace_context = None
            self._trace_profiler = None

    def phase(self, name: str):
        return self.phases.phase(name)

    def close(self) -> None:
        if self._trace_context is not None:
            self._trace_context.__exit__(None, None, None)
            self._trace_context = None
        self.phases.close()

    def __enter__(self) -> "TrainingProfiler":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._trace_context is not None:
            self._trace_context.__exit__(exc_type, exc, traceback)
            self._trace_context = None
        self.phases.close()
