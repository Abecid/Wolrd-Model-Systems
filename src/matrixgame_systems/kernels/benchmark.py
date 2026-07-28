from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .adaln import adaln_reference, fused_adaln

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _bench(callable_, warmup: int = 25, repetitions: int = 100) -> float:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        callable_()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repetitions


def benchmark_shape(rows: int, hidden_size: int, dtype: torch.dtype) -> dict[str, float | int | str]:
    x = torch.randn(rows, hidden_size, device="cuda", dtype=dtype)
    scale = torch.randn_like(x) * 0.1
    shift = torch.randn_like(x) * 0.1
    eager_ms = _bench(lambda: adaln_reference(x, scale, shift))
    triton_ms = _bench(lambda: fused_adaln(x, scale, shift))
    compiled_ms = None
    if hasattr(torch, "compile"):
        try:
            compiled = torch.compile(adaln_reference, fullgraph=True)
            compiled_ms = _bench(lambda: compiled(x, scale, shift))
        except Exception:
            compiled_ms = None
    bytes_moved = x.numel() * x.element_size() * 4  # x/scale/shift read + output write
    return {
        "rows": rows,
        "hidden_size": hidden_size,
        "dtype": str(dtype).replace("torch.", ""),
        "eager_ms": eager_ms,
        "compiled_ms": compiled_ms,
        "triton_ms": triton_ms,
        "speedup_vs_eager": eager_ms / triton_ms,
        "triton_effective_gbps": bytes_moved / (triton_ms / 1000.0) / 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the Matrix-Game fused AdaLN kernel")
    parser.add_argument("--rows", type=int, nargs="+", default=[880, 3520, 8800])
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--output", default="runs/kernel_benchmark.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Triton benchmark")
    results = [benchmark_shape(rows, args.hidden_size, DTYPES[args.dtype]) for rows in args.rows]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
