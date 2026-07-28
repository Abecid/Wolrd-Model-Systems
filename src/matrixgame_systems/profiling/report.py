from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from matrixgame_systems.common.manifest import stable_fingerprint, write_json_atomic

from .metrics import aggregate_phases, derive_metrics, load_phase_files, summarize_gpu_samples
from .trace_parser import parse_chrome_trace


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "n/a"
        return f"{value:,.{digits}f}"
    return str(value)


def _load_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def build_summary(run_dir: str | Path) -> dict[str, Any]:
    directory = Path(run_dir)
    command = _load_json(directory / "command.json", {})
    manifest = _load_json(directory / "manifest.json", {})
    events = load_phase_files(directory)
    phase_summary = aggregate_phases(events)
    gpu_summary = summarize_gpu_samples(directory)
    torch_memory = _load_json(directory / "torch_memory.rank0.json", {})
    trace_path = directory / "trace.json"
    trace_summary = parse_chrome_trace(trace_path) if trace_path.exists() else {}

    wall_time_s = float(command.get("wall_time_s", 0.0))
    frames = int(command.get("frames", 0))
    num_gpus = int(command.get("num_gpus", 1))
    forward_flops = command.get("forward_flops")
    peak_tflops = command.get("peak_tflops_per_gpu")
    derived = derive_metrics(
        wall_time_s=wall_time_s,
        frames=frames,
        num_gpus=num_gpus,
        phase_summary=phase_summary,
        gpu_summary=gpu_summary,
        forward_flops=float(forward_flops) if forward_flops else None,
        peak_tflops_per_gpu=float(peak_tflops) if peak_tflops else None,
        trace_summary=trace_summary,
    )
    workload = command.get("workload", {})
    workload_fingerprint = command.get("workload_fingerprint") or stable_fingerprint(workload)
    summary = {
        "run_dir": str(directory),
        "workload": workload,
        "workload_fingerprint": workload_fingerprint,
        "wall_time_s": wall_time_s,
        "return_code": command.get("return_code"),
        "frames": frames,
        "num_gpus": num_gpus,
        "phase_ms": phase_summary,
        "gpu": gpu_summary,
        "torch_memory": torch_memory,
        "trace": trace_summary,
        "derived": derived,
        "manifest": manifest,
    }
    write_json_atomic(directory / "summary.json", summary)
    return summary


def render_markdown(summary: dict[str, Any], baseline: dict[str, Any] | None = None) -> str:
    lines = [
        f"# Performance report: `{Path(summary['run_dir']).name}`",
        "",
        "## Run",
        "",
        f"- Workload fingerprint: `{summary['workload_fingerprint']}`",
        f"- Wall time: **{_format_number(summary['wall_time_s'])} s**",
        f"- Frames: **{_format_number(summary['frames'])}**",
        f"- GPUs: **{_format_number(summary['num_gpus'])}**",
        f"- Return code: **{_format_number(summary.get('return_code'))}**",
        "",
        "## Bottleneck decomposition",
        "",
        "| Phase | Calls | Total ms | Mean ms | p50 ms | Max ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in sorted(
        summary["phase_ms"].items(), key=lambda item: float(item[1]["total_ms"]), reverse=True
    ):
        lines.append(
            "| {name} | {calls} | {total} | {mean} | {p50} | {maximum} |".format(
                name=name,
                calls=_format_number(values["calls"]),
                total=_format_number(values["total_ms"]),
                mean=_format_number(values["mean_ms"]),
                p50=_format_number(values["p50_ms"]),
                maximum=_format_number(values["max_ms"]),
            )
        )
    gpu = summary["gpu"]
    memory = summary["torch_memory"]
    trace = summary["trace"]
    derived = summary["derived"]
    lines += [
        "",
        "## GPU and throughput",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Average GPU utilization | {_format_number(gpu.get('average_utilization_percent'))}% |",
        f"| Peak physical HBM | {_format_number((gpu.get('peak_physical_hbm_bytes') or 0) / 2**30)} GiB |",
        f"| Peak allocated HBM | {_format_number((memory.get('peak_allocated_bytes') or 0) / 2**30)} GiB |",
        f"| Peak reserved HBM | {_format_number((memory.get('peak_reserved_bytes') or 0) / 2**30)} GiB |",
        f"| NCCL GPU-busy fraction | {_format_number((trace.get('nccl_fraction_of_gpu_busy') or 0) * 100)}% |",
        f"| Achieved TFLOP/s | {_format_number(derived.get('achieved_tflops'))} |",
        f"| Estimated MFU | {_format_number((derived.get('estimated_mfu') or 0) * 100)}% |",
        f"| Frames / GPU-hour | {_format_number(derived.get('frames_per_gpu_hour'))} |",
        f"| Dataloader stall fraction | {_format_number((derived.get('dataloader_stall_fraction') or 0) * 100)}% |",
        f"| VAE fraction | {_format_number((derived.get('vae_fraction') or 0) * 100)}% |",
        f"| Checkpoint foreground pause | {_format_number((derived.get('checkpoint_pause_fraction') or 0) * 100)}% |",
    ]

    if baseline is not None:
        lines += ["", "## Before / after", ""]
        if baseline["workload_fingerprint"] != summary["workload_fingerprint"]:
            lines.append(
                "**Comparison refused:** workload fingerprints differ. Resolution, frames, steps, "
                "prompt/image/action/seed, precision, or GPU configuration changed."
            )
        else:
            before = float(baseline["wall_time_s"])
            after = float(summary["wall_time_s"])
            speedup = before / after if after > 0 else None
            before_fpg = baseline["derived"].get("frames_per_gpu_hour")
            after_fpg = summary["derived"].get("frames_per_gpu_hour")
            lines += [
                "| Metric | Before | After | Delta |",
                "|---|---:|---:|---:|",
                f"| Wall time (s) | {_format_number(before)} | {_format_number(after)} | {_format_number((after / before - 1) * 100 if before else None)}% |",
                f"| End-to-end speedup | 1.000x | {_format_number(speedup)}x | {_format_number((speedup - 1) * 100 if speedup else None)}% |",
                f"| Frames/GPU-hour | {_format_number(before_fpg)} | {_format_number(after_fpg)} | {_format_number((after_fpg / before_fpg - 1) * 100 if before_fpg and after_fpg else None)}% |",
            ]
    lines += [
        "",
        "## Interpretation checklist",
        "",
        "- Verify that the largest phase, not the flashiest microbenchmark, was optimized.",
        "- Attach matching Nsight Systems timelines and kernel-level Nsight Compute reports.",
        "- Report fixed-seed output/latent error and long-horizon action quality.",
        "- Treat MFU as an estimate because the FLOP model omits elementwise and VAE work.",
        "",
    ]
    return "\n".join(lines)


def generate_report(run_dir: str | Path, baseline_dir: str | Path | None = None) -> Path:
    summary = build_summary(run_dir)
    baseline = build_summary(baseline_dir) if baseline_dir else None
    report = render_markdown(summary, baseline)
    destination = Path(run_dir) / "REPORT.md"
    destination.write_text(report, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a MatrixGame Systems performance report")
    parser.add_argument("run_dir")
    parser.add_argument("--baseline")
    args = parser.parse_args()
    path = generate_report(args.run_dir, args.baseline)
    print(path)


if __name__ == "__main__":
    main()
