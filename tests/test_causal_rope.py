import pytest

torch = pytest.importorskip("torch")
causal_rope = pytest.importorskip(
    "world_model_systems.optimizations.kernels.causal_rope"
)
causal_rope_apply_fp32 = causal_rope.causal_rope_apply_fp32
causal_rope_reference = causal_rope.causal_rope_reference


def test_fp32_causal_rope_matches_reference():
    torch.manual_seed(0)
    batch, frames, height, width, heads, head_dim = 1, 3, 2, 2, 2, 12
    sequence = frames * height * width
    x = torch.randn(batch, sequence, heads, head_dim, dtype=torch.float32)
    grid = torch.tensor([[frames, height, width]])
    half = head_dim // 2
    positions = torch.arange(32, dtype=torch.float64)
    inv = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float64) / half))
    freqs = torch.polar(
        torch.ones((32, half), dtype=torch.float64),
        positions[:, None] * inv[None, :],
    )
    reference = causal_rope_reference(x, grid, freqs)
    actual = causal_rope_apply_fp32(x, grid, freqs)
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
