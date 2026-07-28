from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def load_phase_files(run_dir: str | Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(Path(run_dir).glob("phases.rank*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        events.extend(payload.get("events", []))
    return events


def aggregate_phases(events: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    # Distributed ranks execute concurrently. Summing every rank would inflate the
    # critical-path time by world size, so phase totals use the slowest rank.
    values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    wall_values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        name = str(event["name"])
        rank = int(event.get("rank", 0))
        wall_values[name][rank].append(float(event.get("wall_ms", 0.0)))
        duration = event.get("cuda_ms")
        values[name][rank].append(
            float(duration if duration is not None else event.get("wall_ms", 0.0))
        )
    result: dict[str, dict[str, float | int]] = {}
    for name in sorted(values):
        rank_durations = values[name]
        all_durations = [value for durations in rank_durations.values() for value in durations]
        critical_rank = max(rank_durations, key=lambda rank: sum(rank_durations[rank]))
        critical = rank_durations[critical_rank]
        result[name] = {
            "calls": max(len(durations) for durations in rank_durations.values()),
            "total_ms": sum(critical),
            "mean_ms": statistics.fmean(critical),
            "p50_ms": statistics.median(critical),
            "max_ms": max(all_durations),
            "wall_total_ms": max(sum(v) for v in wall_values[name].values()),
            "critical_rank": critical_rank,
        }
    return result


def summarize_gpu_samples(run_dir: str | Path) -> dict[str, float | int | None]:
    samples: list[dict[str, Any]] = []
    for path in sorted(Path(run_dir).glob("gpu.rank*.json")):
        samples.extend(json.loads(path.read_text(encoding="utf-8")))
    utilization = [float(s["utilization_percent"]) for s in samples if s.get("utilization_percent") is not None]
    memory = [int(s["memory_used_bytes"]) for s in samples if s.get("memory_used_bytes") is not None]
    power = [float(s["power_watts"]) for s in samples if s.get("power_watts") is not None]
    return {
        "samples": len(samples),
        "average_utilization_percent": _safe_mean(utilization),
        "p50_utilization_percent": statistics.median(utilization) if utilization else None,
        "peak_physical_hbm_bytes": max(memory) if memory else None,
        "average_power_watts": _safe_mean(power),
    }


def sum_phase(phase_summary: dict[str, dict[str, float | int]], *names: str) -> float:
    return sum(float(phase_summary.get(name, {}).get("total_ms", 0.0)) for name in names)


def derive_metrics(
    *,
    wall_time_s: float,
    frames: int,
    num_gpus: int,
    phase_summary: dict[str, dict[str, float | int]],
    gpu_summary: dict[str, Any],
    forward_flops: float | None = None,
    peak_tflops_per_gpu: float | None = None,
    trace_summary: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    forward_ms = sum_phase(phase_summary, "forward", "dit_forward")
    backward_ms = sum_phase(phase_summary, "backward")
    optimizer_ms = sum_phase(phase_summary, "optimizer")
    dataloader_ms = sum_phase(phase_summary, "dataloader")
    vae_ms = sum_phase(phase_summary, "vae_encode", "vae_decode")
    checkpoint_ms = sum_phase(phase_summary, "checkpoint")
    step_ms = sum_phase(phase_summary, "step")
    denominator_ms = step_ms if step_ms > 0 else wall_time_s * 1000.0
    achieved_tflops = None
    mfu = None
    if forward_flops and forward_ms > 0:
        achieved_tflops = forward_flops / (forward_ms / 1000.0) / 1e12
        if peak_tflops_per_gpu and num_gpus > 0:
            mfu = achieved_tflops / (peak_tflops_per_gpu * num_gpus)
    return {
        "frames_per_gpu_hour": frames * 3600.0 / (wall_time_s * max(num_gpus, 1)) if wall_time_s > 0 else None,
        "forward_fraction": forward_ms / denominator_ms if denominator_ms else None,
        "backward_fraction": backward_ms / denominator_ms if denominator_ms else None,
        "optimizer_fraction": optimizer_ms / denominator_ms if denominator_ms else None,
        "dataloader_stall_fraction": dataloader_ms / denominator_ms if denominator_ms else None,
        "vae_fraction": vae_ms / denominator_ms if denominator_ms else None,
        "checkpoint_pause_fraction": checkpoint_ms / denominator_ms if denominator_ms else None,
        "nccl_fraction": (trace_summary or {}).get("nccl_fraction_of_gpu_busy"),
        "achieved_tflops": achieved_tflops,
        "estimated_mfu": mfu,
        "average_gpu_utilization_percent": gpu_summary.get("average_utilization_percent"),
    }
