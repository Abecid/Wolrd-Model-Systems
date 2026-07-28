# Architecture

```text
Client -> bounded queue -> compatibility batcher -> HBM admission -> backend
                                      |                    |
                                      v                    v
                                  Prometheus          cancellation
                                      |
                                      v
                               SSE progress/output
```

The Matrix-Game adapter is intentionally batch-size one until its autoregressive
memory and action state are vectorized and validated. The scheduler is backend-
agnostic and is tested with true multi-request batches through the mock backend.

Training utilities are similarly model-agnostic: the profiler records named
phases in any loop, while the sequence-parallel transform and DCP checkpointer
operate on PyTorch tensors/stateful objects. The pinned integration patch changes
only Matrix-Game's repeated LayerNorm-plus-modulation expression.
