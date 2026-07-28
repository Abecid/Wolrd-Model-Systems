# Architecture

World Model Systems has four layers.

## 1. Core

`world_model_systems.core` defines model capabilities, license policy, launch requests, the adapter registry and the optimization planner. It has no CUDA dependency.

## 2. Model adapters

`world_model_systems.models` contains thin, lazy adapters. They know upstream paths, revisions, command-line flags and safe patch anchors. They do not own shared kernels or profiler logic.

## 3. Reusable systems modules

Optimizations are grouped by mechanism:

- kernels;
- precision;
- parallelism;
- runtime/cache;
- distributed training;
- profiling;
- serving.

The original `matrixgame_systems` package remains as a compatibility layer while reusable modules move behind the generic namespace.

## 4. Experiment artifacts

Configs and scripts define reproducible workloads. Runs write manifests and workload fingerprints. Reports compare only matching fingerprints.

## Dependency direction

```text
upstream model source
        ↓ imports after optional patch
world_model_systems.optimizations
        ↑ selected by
model adapter → registry → planner/CLI/profiler
```

Reusable code never imports an upstream model at module import time. This keeps the library installable and testable without every model's dependency stack.
