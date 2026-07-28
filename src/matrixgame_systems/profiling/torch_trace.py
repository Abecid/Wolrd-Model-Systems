from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path


@contextlib.contextmanager
def torch_trace(path: str | Path, *, enabled: bool) -> Iterator[object | None]:
    if not enabled:
        yield None
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for --torch-trace") from exc
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
    ) as profiler:
        yield profiler
    profiler.export_chrome_trace(str(destination))
