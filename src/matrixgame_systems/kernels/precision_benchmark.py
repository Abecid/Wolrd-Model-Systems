from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .fp8 import FP8Linear


def benchmark(callable_, repetitions: int = 100) -> float:
    for _ in range(20):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="BF16 versus native FP8 Linear benchmark")
    parser.add_argument("--rows", type=int, default=3520)
    parser.add_argument("--in-features", type=int, default=5120)
    parser.add_argument("--out-features", type=int, default=5120)
    parser.add_argument("--output", default="runs/precision_benchmark.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    linear = torch.nn.Linear(args.in_features, args.out_features, bias=False, device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(args.rows, args.in_features, device="cuda", dtype=torch.bfloat16)
    fp8 = FP8Linear(linear)
    with torch.inference_mode():
        bf16_output = linear(inputs)
        fp8_output = fp8(inputs)
        bf16_ms = benchmark(lambda: linear(inputs))
        fp8_ms = benchmark(lambda: fp8(inputs))
    error = (bf16_output.float() - fp8_output.float()).abs()
    result = {
        "shape": [args.rows, args.in_features, args.out_features],
        "bf16_ms": bf16_ms,
        "fp8_ms": fp8_ms,
        "speedup": bf16_ms / fp8_ms,
        "mean_absolute_error": float(error.mean()),
        "max_absolute_error": float(error.max()),
        "relative_l2_error": float(error.norm() / bf16_output.float().norm().clamp_min(1e-12)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
