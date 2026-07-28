from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from matrixgame_systems.kernels.adaln import adaln_reference, fused_adaln

@pytest.mark.parametrize("shape", [(3, 8), (2, 4, 16)])
def test_reference_matches_definition(shape):
    torch.manual_seed(0)
    x = torch.randn(shape)
    scale = torch.randn(shape)
    shift = torch.randn(shape)
    expected = F.layer_norm(x.float(), (shape[-1],), eps=1e-6) * (1 + scale) + shift
    actual = adaln_reference(x, scale, shift)
    torch.testing.assert_close(actual, expected)


def test_broadcast_falls_back_and_gradients_exist():
    x = torch.randn(2, 3, 8, requires_grad=True)
    scale = torch.randn(8, requires_grad=True)
    shift = torch.randn(8, requires_grad=True)
    fused_adaln(x, scale, shift).sum().backward()
    assert x.grad is not None
    assert scale.grad is not None
    assert shift.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_forward_and_backward(dtype):
    pytest.importorskip("triton")
    torch.manual_seed(0)
    shape = (32, 5120)
    x_ref = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    s_ref = (torch.randn(shape, device="cuda", dtype=dtype) * 0.1).requires_grad_()
    b_ref = (torch.randn(shape, device="cuda", dtype=dtype) * 0.1).requires_grad_()
    x_tri = x_ref.detach().clone().requires_grad_()
    s_tri = s_ref.detach().clone().requires_grad_()
    b_tri = b_ref.detach().clone().requires_grad_()
    grad = torch.randn(shape, device="cuda", dtype=dtype)
    y_ref = adaln_reference(x_ref, s_ref, b_ref)
    y_tri = fused_adaln(x_tri, s_tri, b_tri)
    y_ref.backward(grad)
    y_tri.backward(grad)
    tolerance = 3e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(y_tri, y_ref, rtol=tolerance, atol=tolerance)
    for actual, expected in zip((x_tri.grad, s_tri.grad, b_tri.grad), (x_ref.grad, s_ref.grad, b_ref.grad)):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
