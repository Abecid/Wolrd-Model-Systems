from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any


async def _one(client, url: str, payload: dict[str, Any], semaphore: asyncio.Semaphore) -> dict[str, Any]:
    async with semaphore:
        start = time.perf_counter()
        request_id = None
        terminal = None
        async with client.stream("POST", url, json=payload, timeout=None) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                request_id = event.get("request_id", request_id)
                if event.get("kind") in {"completed", "failed", "cancelled"}:
                    terminal = event
        return {
            "request_id": request_id,
            "latency_s": time.perf_counter() - start,
            "terminal": terminal,
        }


async def run(args: argparse.Namespace) -> None:
    import httpx

    payload = {
        "prompt": args.prompt,
        "image_path": args.image,
        "height": args.height,
        "width": args.width,
        "num_iterations": args.num_iterations,
        "num_inference_steps": args.steps,
        "seed": args.seed,
        "precision": args.precision,
    }
    semaphore = asyncio.Semaphore(args.concurrency)
    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[
                _one(client, f"{args.base_url.rstrip('/')}/v1/generations", payload, semaphore)
                for _ in range(args.requests)
            ]
        )
    wall = time.perf_counter() - start
    latencies = sorted(result["latency_s"] for result in results)

    def percentile(q: float) -> float:
        if len(latencies) == 1:
            return latencies[0]
        index = (len(latencies) - 1) * q
        lower = int(index)
        upper = min(lower + 1, len(latencies) - 1)
        fraction = index - lower
        return latencies[lower] * (1 - fraction) + latencies[upper] * fraction

    summary = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "wall_time_s": wall,
        "throughput_requests_per_s": args.requests / wall,
        "latency_mean_s": statistics.fmean(latencies),
        "latency_p50_s": percentile(0.50),
        "latency_p95_s": percentile(0.95),
        "latency_p99_s": percentile(0.99),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrency load test for MatrixGame Systems")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--prompt", default="A navigable animated city")
    parser.add_argument("--image", required=True)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-iterations", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--output", default="runs/service_loadtest.json")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
