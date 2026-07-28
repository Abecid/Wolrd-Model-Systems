from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from matrixgame_systems.common.manifest import (
    environment_manifest,
    stable_fingerprint,
    write_json_atomic,
)

from .gpu_monitor import GpuMonitor


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile an arbitrary training or inference command")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--forward-flops", type=float)
    parser.add_argument("--gpu-sample-ms", type=int, default=100)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("Provide a command after --")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    workload = {"command": command, "frames": args.frames, "num_gpus": args.num_gpus}
    write_json_atomic(
        run_dir / "manifest.json", environment_manifest(command=command, config=workload)
    )
    monitor = GpuMonitor(run_dir / "gpu.rank0.json", interval_ms=args.gpu_sample_ms)
    monitor.start()
    start = time.perf_counter()
    completed = subprocess.run(command, check=False)
    wall = time.perf_counter() - start
    monitor.stop()
    write_json_atomic(
        run_dir / "command.json",
        {
            "return_code": completed.returncode,
            "wall_time_s": wall,
            "frames": args.frames,
            "num_gpus": args.num_gpus,
            "peak_tflops_per_gpu": args.peak_tflops,
            "forward_flops": args.forward_flops,
            "workload": workload,
            "workload_fingerprint": stable_fingerprint(workload),
        },
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
