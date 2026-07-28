import pytest
import torch

from matrixgame_systems.kernels.fp8 import FP8Linear


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 9
    or not hasattr(torch, "_scaled_mm"),
    reason="Native Hopper FP8 required",
)
def test_fp8_linear_error_is_bounded():
    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 384, device="cuda", dtype=torch.bfloat16)
    fp8 = FP8Linear(linear)
    inputs = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    expected = linear(inputs).float()
    actual = fp8(inputs).float()
    relative = (expected - actual).norm() / expected.norm()
    assert float(relative) < 0.15
