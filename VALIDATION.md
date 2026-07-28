# Validation status

**CPU contract suite: PASS**

Executed locally after the multi-model refactor:

```text
23 passed, 3 skipped
```

Skipped tests require CUDA/Triton or native Hopper FP8. The new registry, optimization planner, LingBot source patcher, causal RoPE numerical test, checkpointing, profiler/report contracts, distributed sampler, sequence-parallel utilities, and serving scheduler all passed.

`ruff` was not installed in the execution environment, so lint was not claimed locally. GitHub Actions installs the declared development dependencies and runs Ruff plus the full CPU suite.

No end-to-end GPU speedup or model-quality claim is made without the official checkpoints and target NVIDIA hardware.
