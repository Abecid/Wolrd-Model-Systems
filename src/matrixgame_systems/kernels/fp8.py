from __future__ import annotations

import re
from collections.abc import Iterable

import torch


def _require_native_fp8() -> None:
    if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("Native FP8 requires a PyTorch build exposing float8_e4m3fn and _scaled_mm")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("The current FP8 path requires Hopper-class compute capability >= 9.0")


def quantize_per_tensor_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _require_native_fp8()
    dtype = torch.float8_e4m3fn
    maximum = torch.finfo(dtype).max
    absolute_max = tensor.detach().abs().amax().float().clamp_min(1e-12)
    scale = absolute_max / maximum
    quantized = (tensor.float() / scale).clamp(-maximum, maximum).to(dtype)
    return quantized, scale


class FP8Linear(torch.nn.Module):
    """Experimental per-tensor FP8 inference linear using native scaled GEMM.

    The activation scale is computed dynamically for every call; the weight is
    quantized once. This is intentionally explicit and benchmarkable, not a
    claim that FP8 is automatically faster for every Matrix-Game shape.
    """

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        _require_native_fp8()
        weight_fp8, weight_scale = quantize_per_tensor_fp8(linear.weight.detach())
        self.register_buffer("weight_fp8", weight_fp8.contiguous())
        self.register_buffer("weight_scale", weight_scale.reshape(1))
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().clone())
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_shape = inputs.shape
        matrix = inputs.reshape(-1, original_shape[-1])
        activation_fp8, activation_scale = quantize_per_tensor_fp8(matrix)
        result = torch._scaled_mm(
            activation_fp8,
            self.weight_fp8.t(),
            scale_a=activation_scale.reshape(1),
            scale_b=self.weight_scale,
            out_dtype=inputs.dtype,
        )
        if isinstance(result, tuple):  # compatibility with older PyTorch return conventions
            result = result[0]
        if self.bias is not None:
            result = result + self.bias.to(result.dtype)
        return result.reshape(*original_shape[:-1], self.out_features)


def convert_named_linears_to_fp8(
    module: torch.nn.Module,
    *,
    name_patterns: Iterable[str] = (r"(^|\.)(q|k|v|o|q_proj|k_proj|v_proj|o_proj)$",),
) -> list[str]:
    """Replace matching child Linear modules in place and return changed names."""

    _require_native_fp8()
    patterns = [re.compile(pattern) for pattern in name_patterns]
    changed: list[str] = []

    def visit(parent: torch.nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, torch.nn.Linear) and any(
                pattern.search(full_name) for pattern in patterns
            ):
                setattr(parent, name, FP8Linear(child))
                changed.append(full_name)
            else:
                visit(child, full_name)

    visit(module)
    return changed
