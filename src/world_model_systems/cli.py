from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from world_model_systems.core.plan import OptimizationPlanner
from world_model_systems.core.registry import get_model, list_models
from world_model_systems.core.spec import LaunchRequest


def _print_models(as_json: bool) -> None:
    payload = [adapter.spec.to_dict() for adapter in list_models()]
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for adapter in list_models():
        spec = adapter.spec
        caps = ",".join(sorted(cap.value for cap in spec.capabilities))
        print(f"{spec.id:22} {spec.display_name:42} {spec.model_size:8} {caps}")


def _launch_request(args: argparse.Namespace, extra: tuple[str, ...]) -> LaunchRequest:
    return LaunchRequest(
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="wms", description="World Model Systems")
    sub = parser.add_subparsers(dest="command", required=True)

    models = sub.add_parser("models", help="List registered model adapters")
    models.add_argument("--json", action="store_true")

    describe = sub.add_parser("describe", help="Describe one model adapter")
    describe.add_argument("model")

    plan = sub.add_parser("plan", help="Build a capability-safe optimization plan")
    plan.add_argument("model")
    plan.add_argument("--objective", default="latency", choices=("latency", "throughput", "memory"))
    plan.add_argument("--passes", nargs="*")
    plan.add_argument("--include-experimental", action="store_true")

    patch = sub.add_parser("patch", help="Apply an idempotent source integration patch")
    patch.add_argument("model")
    patch.add_argument("optimization")
    patch.add_argument("--upstream", required=True)
    patch.add_argument("--allow-unpinned", action="store_true")

    command = sub.add_parser("command", help="Print a reproducible upstream inference command")
    command.add_argument("model")
    command.add_argument("--upstream", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--image", required=True)
    command.add_argument("--prompt", required=True)
    command.add_argument("--output-dir", default="outputs")
    command.add_argument("--frames", type=int)
    command.add_argument("--num-gpus", type=int, default=1)
    command.add_argument("--precision", default="bf16")
    command.add_argument("--execute", action="store_true")
    command.add_argument("extra", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.command == "models":
        _print_models(args.json)
        return
    adapter = get_model(args.model)
    if args.command == "describe":
        print(json.dumps(adapter.spec.to_dict(), indent=2, sort_keys=True))
        return
    if args.command == "plan":
        requested = tuple(args.passes) if args.passes else None
        result = OptimizationPlanner().build(
            adapter.spec,
            objective=args.objective,
            requested=requested,
            include_experimental=args.include_experimental,
        )
        print(json.dumps({
            "model": result.model_id,
            "objective": result.objective,
            "passes": list(result.ids()),
            "skipped": list(result.skipped),
        }, indent=2))
        return
    if args.command == "patch":
        changed = adapter.apply_optimization(
            args.optimization,
            Path(args.upstream),
            allow_unpinned=args.allow_unpinned,
        )
        print(json.dumps({"changed": [str(path) for path in changed]}, indent=2))
        return
    extra = tuple(args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra)
    request = _launch_request(args, extra)
    cmd = adapter.build_inference_command(request)
    print(shlex.join(cmd))
    if args.execute:
        raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
