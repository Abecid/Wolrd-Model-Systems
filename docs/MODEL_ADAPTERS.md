# Model adapter contract

World Model Systems treats upstream repositories as plugins. The library owns reusable systems code; each adapter owns only the knowledge required to drive one upstream model safely.

## Required metadata

Every adapter publishes a `ModelSpec` with:

- stable adapter ID;
- upstream repository and pinned revision;
- organization, family and model size;
- license policy, including commercial/hosted-service constraints;
- capabilities such as causal generation, KV cache, few-step inference, sequence parallelism and training;
- supported optimization passes;
- default checkpoint, resolution and frame count where available.

This metadata is executable. The optimization planner uses capabilities to reject invalid passes, and the CLI exposes license constraints before bootstrap or patching.

## Required methods

```python
class ModelAdapter(Protocol):
    spec: ModelSpec

    def build_inference_command(self, request: LaunchRequest) -> list[str]: ...

    def apply_optimization(
        self,
        optimization_id: str,
        upstream: Path,
        *,
        allow_unpinned: bool = False,
    ) -> list[Path]: ...
```

An adapter should not import a heavy upstream package at library import time. Commands and source patches are built lazily so `wms models` works on a laptop without CUDA or model dependencies.

## Patch rules

Source patches must be:

1. **revision-pinned** — the reviewed upstream commit is explicit;
2. **fail-closed** — missing anchors raise rather than silently patching the wrong code;
3. **idempotent** — a second invocation changes nothing;
4. **recorded** — a JSON marker contains before/after hashes;
5. **minimal** — patch call sites, not entire vendored files;
6. **optional** — the untouched upstream baseline is always runnable.

## New adapter checklist

- Add `src/world_model_systems/models/<model>.py`.
- Register the adapter in `core/registry.py`.
- Add `configs/models/<model>.yaml`.
- Add a bootstrap entry in `scripts/bootstrap_model.sh`.
- Add command-construction and patch-idempotence tests.
- Add a model document describing architecture, bottlenecks and honest validation boundaries.
- Never add upstream checkpoints or copied source to this repository.
