from __future__ import annotations

from collections.abc import Callable
from typing import Any


class CUDAGraphRunner:
    """Capture a fixed-shape tensor-only callable and replay with copied inputs."""

    def __init__(self, function: Callable[..., Any], warmup: int = 3) -> None:
        self.function = function
        self.warmup = warmup
        self.graph = None
        self.static_inputs = None
        self.static_output = None

    def capture(self, *example_inputs):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require CUDA")
        if not all(isinstance(value, torch.Tensor) for value in example_inputs):
            raise TypeError("CUDAGraphRunner currently supports tensor-only positional inputs")
        self.static_inputs = tuple(value.clone() for value in example_inputs)
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(self.warmup):
                self.function(*self.static_inputs)
        torch.cuda.current_stream().wait_stream(side_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.function(*self.static_inputs)
        return self.static_output

    def __call__(self, *inputs):
        if self.graph is None or self.static_inputs is None:
            raise RuntimeError("Call capture() first")
        if len(inputs) != len(self.static_inputs):
            raise ValueError("Input count changed")
        for static, incoming in zip(self.static_inputs, inputs, strict=True):
            if static.shape != incoming.shape or static.dtype != incoming.dtype:
                raise ValueError("CUDA graph input shape/dtype changed")
            static.copy_(incoming)
        self.graph.replay()
        return self.static_output
