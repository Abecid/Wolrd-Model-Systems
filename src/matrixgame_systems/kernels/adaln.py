from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on GPU environment
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


def adaln_reference(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    normalized = F.layer_norm(x.float(), (x.shape[-1],), eps=eps)
    return (normalized * (1.0 + scale.float()) + shift.float()).to(x.dtype)


def _can_use_triton(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> bool:
    return (
        _TRITON_AVAILABLE
        and x.is_cuda
        and x.is_contiguous()
        and scale.is_contiguous()
        and shift.is_contiguous()
        and x.shape == scale.shape == shift.shape
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.shape[-1] <= 65536
    )


if _TRITON_AVAILABLE:

    @triton.jit
    def _adaln_forward_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        y_ptr,
        mean_ptr,
        rstd_ptr,
        row_stride: tl.constexpr,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        base = row * row_stride + offsets
        x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / hidden_size
        centered = tl.where(mask, x - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / hidden_size
        rstd = tl.rsqrt(variance + eps)
        scale = tl.load(scale_ptr + base, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + base, mask=mask, other=0.0).to(tl.float32)
        output = centered * rstd * (1.0 + scale) + shift
        tl.store(y_ptr + base, output, mask=mask)
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, rstd)

    @triton.jit
    def _adaln_backward_kernel(
        grad_y_ptr,
        x_ptr,
        scale_ptr,
        mean_ptr,
        rstd_ptr,
        grad_x_ptr,
        grad_scale_ptr,
        grad_shift_ptr,
        row_stride: tl.constexpr,
        hidden_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        base = row * row_stride + offsets
        grad_y = tl.load(grad_y_ptr + base, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + base, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row).to(tl.float32)
        rstd = tl.load(rstd_ptr + row).to(tl.float32)
        z = tl.where(mask, (x - mean) * rstd, 0.0)
        grad_z = tl.where(mask, grad_y * (1.0 + scale), 0.0)
        mean_grad = tl.sum(grad_z, axis=0) / hidden_size
        mean_grad_z = tl.sum(grad_z * z, axis=0) / hidden_size
        grad_x = rstd * (grad_z - mean_grad - z * mean_grad_z)
        tl.store(grad_x_ptr + base, grad_x, mask=mask)
        tl.store(grad_scale_ptr + base, grad_y * z, mask=mask)
        tl.store(grad_shift_ptr + base, grad_y, mask=mask)

    @triton.jit
    def _gated_residual_forward_kernel(
        residual_ptr,
        branch_ptr,
        gate_ptr,
        output_ptr,
        numel: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        residual = tl.load(residual_ptr + offsets, mask=mask)
        branch = tl.load(branch_ptr + offsets, mask=mask)
        gate = tl.load(gate_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, residual + branch * gate, mask=mask)


class _FusedAdaLNFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        rows = x.numel() // x.shape[-1]
        hidden_size = x.shape[-1]
        x_2d = x.reshape(rows, hidden_size)
        scale_2d = scale.reshape(rows, hidden_size)
        shift_2d = shift.reshape(rows, hidden_size)
        output = torch.empty_like(x_2d)
        mean = torch.empty((rows,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((rows,), device=x.device, dtype=torch.float32)
        block_size = triton.next_power_of_2(hidden_size)
        num_warps = 8 if block_size >= 4096 else 4
        _adaln_forward_kernel[(rows,)](
            x_2d,
            scale_2d,
            shift_2d,
            output,
            mean,
            rstd,
            x_2d.stride(0),
            hidden_size,
            eps,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        ctx.save_for_backward(x_2d, scale_2d, mean, rstd)
        ctx.original_shape = x.shape
        ctx.hidden_size = hidden_size
        return output.reshape_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        x_2d, scale_2d, mean, rstd = ctx.saved_tensors
        grad_2d = grad_output.contiguous().reshape_as(x_2d)
        grad_x = torch.empty_like(x_2d)
        grad_scale = torch.empty_like(scale_2d)
        grad_shift = torch.empty_like(scale_2d)
        rows, hidden_size = x_2d.shape
        block_size = triton.next_power_of_2(hidden_size)
        num_warps = 8 if block_size >= 4096 else 4
        _adaln_backward_kernel[(rows,)](
            grad_2d,
            x_2d,
            scale_2d,
            mean,
            rstd,
            grad_x,
            grad_scale,
            grad_shift,
            x_2d.stride(0),
            hidden_size,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        shape = ctx.original_shape
        return grad_x.reshape(shape), grad_scale.reshape(shape), grad_shift.reshape(shape), None


def fused_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
    *,
    force_reference: bool = False,
) -> torch.Tensor:
    """Fused affine-free LayerNorm + AdaLN modulation.

    Triton is used only for contiguous CUDA tensors where x, scale, and shift
    have identical shapes. Broadcasted conditioning falls back to the reference
    implementation rather than silently returning wrong gradients.
    """

    use_kernel = os.environ.get("MGS_USE_FUSED_ADALN", "1") != "0"
    if not force_reference and use_kernel and _can_use_triton(x, scale, shift):
        return _FusedAdaLNFunction.apply(x, scale, shift, float(eps))
    return adaln_reference(x, scale, shift, eps)


def gated_residual_reference(
    residual: torch.Tensor, branch: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return residual + branch * gate


def fused_gated_residual(
    residual: torch.Tensor, branch: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    if not (
        _TRITON_AVAILABLE
        and residual.is_cuda
        and residual.shape == branch.shape == gate.shape
        and residual.is_contiguous()
        and branch.is_contiguous()
        and gate.is_contiguous()
    ):
        return gated_residual_reference(residual, branch, gate)
    output = torch.empty_like(residual)
    numel = residual.numel()
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    _gated_residual_forward_kernel[grid](
        residual,
        branch,
        gate,
        output,
        numel,
        BLOCK_SIZE=256,
    )
    return output
