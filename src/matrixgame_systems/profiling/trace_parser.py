from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


NCCL_MARKERS = ("nccl", "allreduce", "all_reduce", "allgather", "all_gather", "reduce_scatter", "alltoall")


def _union_duration_us(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def parse_chrome_trace(path: str | Path) -> dict[str, float | int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    events = payload.get("traceEvents", payload if isinstance(payload, list) else [])
    gpu_intervals: list[tuple[float, float]] = []
    nccl_intervals: list[tuple[float, float]] = []
    nccl_calls = 0
    for event in events:
        if event.get("ph") != "X" or "dur" not in event or "ts" not in event:
            continue
        name = str(event.get("name", "")).lower()
        category = str(event.get("cat", "")).lower()
        combined = f"{category} {name}"
        start = float(event["ts"])
        end = start + float(event["dur"])
        is_gpu = any(marker in combined for marker in ("kernel", "cuda", "gpu", "nccl"))
        if is_gpu:
            gpu_intervals.append((start, end))
        if any(marker in combined for marker in NCCL_MARKERS):
            nccl_intervals.append((start, end))
            nccl_calls += 1
    gpu_us = _union_duration_us(gpu_intervals)
    nccl_us = _union_duration_us(nccl_intervals)
    return {
        "gpu_busy_ms": gpu_us / 1000.0,
        "nccl_ms": nccl_us / 1000.0,
        "nccl_calls": nccl_calls,
        "nccl_fraction_of_gpu_busy": nccl_us / gpu_us if gpu_us > 0 else 0.0,
    }
