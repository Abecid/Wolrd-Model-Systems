from __future__ import annotations

import argparse
import builtins
import functools
import json
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Any, Callable

from matrixgame_systems.common.manifest import environment_manifest, stable_fingerprint, write_json_atomic

from .events import PhaseRecorder
from .flops import WanDiTShape, estimate_wan_dit_forward_flops
from .gpu_monitor import GpuMonitor
from .torch_trace import torch_trace


def _wrap_callable(owner: Any, attribute: str, recorder: PhaseRecorder, phase_name: str) -> None:
    original = getattr(owner, attribute, None)
    if original is None or getattr(original, "_mgs_wrapped", False):
        return

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with recorder.phase(phase_name):
            return original(*args, **kwargs)

    wrapped._mgs_wrapped = True  # type: ignore[attr-defined]
    setattr(owner, attribute, wrapped)


class _TimedCallableProxy:
    def __init__(self, target: Any, recorder: PhaseRecorder, phase_name: str) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_recorder", recorder)
        object.__setattr__(self, "_phase_name", phase_name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self._recorder.phase(self._phase_name):
            return self._target(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target, name, value)


def _patch_pipeline_class(cls: type, recorder: PhaseRecorder) -> None:
    original_init = cls.__init__
    if getattr(original_init, "_mgs_wrapped", False):
        return

    @functools.wraps(original_init)
    def instrumented_init(self: Any, *args: Any, **kwargs: Any) -> None:
        with recorder.phase("pipeline_init"):
            original_init(self, *args, **kwargs)
        if getattr(self, "model", None) is not None:
            _wrap_callable(self.model, "forward", recorder, "dit_forward")
        if getattr(self, "vae", None) is not None:
            _wrap_callable(self.vae, "encode", recorder, "vae_encode")
            _wrap_callable(self.vae, "stream_decode", recorder, "vae_decode")
            _wrap_callable(self.vae, "decode", recorder, "vae_decode")
        if getattr(self, "text_encoder", None) is not None and not isinstance(
            self.text_encoder, _TimedCallableProxy
        ):
            self.text_encoder = _TimedCallableProxy(self.text_encoder, recorder, "text_encoder")

    instrumented_init._mgs_wrapped = True  # type: ignore[attr-defined]
    cls.__init__ = instrumented_init
    _wrap_callable(cls, "generate", recorder, "generation")


def _install_matrixgame_instrumentation(upstream: Path, recorder: PhaseRecorder) -> Path:
    matrixgame_root = upstream.resolve()
    if not (matrixgame_root / "generate.py").exists():
        raise FileNotFoundError(f"Expected Matrix-Game-3/generate.py under {matrixgame_root}")
    sys.path.insert(0, str(matrixgame_root))
    from pipeline.inference_pipeline import MatrixGame3Pipeline as BatchPipeline

    _patch_pipeline_class(BatchPipeline, recorder)
    try:
        from pipeline.inference_interactive_pipeline import (
            MatrixGame3Pipeline as InteractivePipeline,
        )
    except ImportError:
        InteractivePipeline = None
    if InteractivePipeline is not None:
        _patch_pipeline_class(InteractivePipeline, recorder)
    return matrixgame_root / "generate.py"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run Matrix-Game 3 with phase and GPU profiling")
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--gpu-sample-ms", type=int, default=100)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--torch-trace", action="store_true", help="Export run_dir/trace.json")
    known, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return known, passthrough


def main() -> None:
    args, passthrough = _parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = PhaseRecorder(run_dir, num_gpus=args.num_gpus, rank=rank)
    monitor = GpuMonitor(run_dir / f"gpu.rank{rank}.json", gpu_index=local_rank, interval_ms=args.gpu_sample_ms)
    generate_path = _install_matrixgame_instrumentation(Path(args.upstream), recorder)

    workload = {
        "upstream": str(Path(args.upstream).resolve()),
        "frames": args.frames,
        "num_gpus": args.num_gpus,
        "matrixgame_args": passthrough,
    }
    forward_flops = None
    if args.sequence_length:
        forward_flops = estimate_wan_dit_forward_flops(
            WanDiTShape(sequence_length=args.sequence_length)
        )

    if rank == 0:
        write_json_atomic(
            run_dir / "manifest.json",
            environment_manifest(
                command=[sys.executable, str(generate_path), *passthrough],
                config=workload,
                repo=args.repo,
            ),
        )

    original_exit = builtins.exit

    def profiled_exit(code: int | None = 0) -> None:
        raise SystemExit(code)

    builtins.exit = profiled_exit
    monitor.start()
    start = time.perf_counter()
    return_code = 0
    try:
        sys.argv = [str(generate_path), *passthrough]
        with torch_trace(run_dir / "trace.json", enabled=args.torch_trace):
            with recorder.phase("end_to_end"):
                runpy.run_path(str(generate_path), run_name="__main__")
    except SystemExit as exc:
        return_code = int(exc.code or 0)
    finally:
        builtins.exit = original_exit
        wall = time.perf_counter() - start
        monitor.stop()
        recorder.close()
        if rank == 0:
            write_json_atomic(
                run_dir / "command.json",
                {
                    "return_code": return_code,
                    "wall_time_s": wall,
                    "frames": args.frames,
                    "num_gpus": args.num_gpus,
                    "peak_tflops_per_gpu": args.peak_tflops,
                    "forward_flops": forward_flops,
                    "workload": workload,
                    "workload_fingerprint": stable_fingerprint(workload),
                },
            )
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
