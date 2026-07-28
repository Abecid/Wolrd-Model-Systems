from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _nested(payload: dict[str, Any], key: str) -> float:
    current: Any = payload
    for part in key.split("."):
        current = current[part]
    return float(current)


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_latency_regression: float = 0.03,
    max_memory_regression: float = 0.03,
    min_throughput_ratio: float = 0.97,
) -> list[str]:
    failures: list[str] = []
    if baseline.get("workload_fingerprint") != candidate.get("workload_fingerprint"):
        failures.append("workload fingerprints differ")
        return failures
    before_latency = _nested(baseline, "wall_time_s")
    after_latency = _nested(candidate, "wall_time_s")
    if after_latency > before_latency * (1.0 + max_latency_regression):
        failures.append(
            f"wall time regressed {(after_latency / before_latency - 1) * 100:.2f}%"
        )
    before_memory = _nested(baseline, "gpu.peak_physical_hbm_bytes")
    after_memory = _nested(candidate, "gpu.peak_physical_hbm_bytes")
    if before_memory > 0 and after_memory > before_memory * (1.0 + max_memory_regression):
        failures.append(
            f"peak physical HBM regressed {(after_memory / before_memory - 1) * 100:.2f}%"
        )
    before_throughput = _nested(baseline, "derived.frames_per_gpu_hour")
    after_throughput = _nested(candidate, "derived.frames_per_gpu_hour")
    if before_throughput > 0 and after_throughput < before_throughput * min_throughput_ratio:
        failures.append(
            f"frames/GPU-hour fell {(1 - after_throughput / before_throughput) * 100:.2f}%"
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail CI on throughput/memory regressions")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--max-latency-regression", type=float, default=0.03)
    parser.add_argument("--max-memory-regression", type=float, default=0.03)
    parser.add_argument("--min-throughput-ratio", type=float, default=0.97)
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    failures = compare(
        baseline,
        candidate,
        max_latency_regression=args.max_latency_regression,
        max_memory_regression=args.max_memory_regression,
        min_throughput_ratio=args.min_throughput_ratio,
    )
    if failures:
        raise SystemExit("Performance regression:\n- " + "\n- ".join(failures))
    print("Performance regression gate passed")


if __name__ == "__main__":
    main()
