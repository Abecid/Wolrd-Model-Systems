from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from matrixgame_systems.common.manifest import environment_manifest, stable_fingerprint, write_json_atomic
from matrixgame_systems.profiling.gpu_monitor import GpuMonitor
from world_model_systems.core.registry import get_model
from world_model_systems.core.spec import LaunchRequest


def main() -> None:
    parser = argparse.ArgumentParser(prog="wms-profile")
    parser.add_argument("model")
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--gpu-sample-ms", type=int, default=100)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    adapter = get_model(args.model)
    extra = tuple(args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra)
    request = LaunchRequest(
        upstream=Path(args.upstream),
        checkpoint=Path(args.checkpoint),
        prompt=args.prompt,
        image=Path(args.image),
        output_dir=Path(args.output_dir),
        frames=args.frames,
        num_gpus=args.num_gpus,
        precision=args.precision,
        extra_args=extra,
    )
    command = adapter.build_inference_command(request)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    workload = {
        "model_id": adapter.spec.id,
        "model_revision": adapter.spec.pinned_revision,
        "frames": request.frames or adapter.spec.default_frames,
        "num_gpus": request.num_gpus,
        "precision": request.precision,
        "command": command,
    }
    write_json_atomic(run_dir / "manifest.json", environment_manifest(command=command, config=workload))
    monitor = GpuMonitor(run_dir / "gpu.rank0.json", interval_ms=args.gpu_sample_ms)
    monitor.start()
    started = time.perf_counter()
    completed = subprocess.run(command, check=False)
    wall = time.perf_counter() - started
    monitor.stop()
    write_json_atomic(run_dir / "command.json", {
        "return_code": completed.returncode,
        "wall_time_s": wall,
        "frames": workload["frames"],
        "num_gpus": request.num_gpus,
        "peak_tflops_per_gpu": args.peak_tflops,
        "forward_flops": None,
        "workload": workload,
        "workload_fingerprint": stable_fingerprint(workload),
    })
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
