import torch

from matrixgame_systems.distributed.sequence_parallel import (
    ScalingPoint,
    parallel_efficiency,
    ulysses_post_attention,
    ulysses_pre_attention,
)


def test_world_size_one_is_identity():
    tensor = torch.randn(2, 4, 8, 16)
    assert torch.equal(ulysses_pre_attention(tensor), tensor)
    assert torch.equal(ulysses_post_attention(tensor), tensor)


def test_parallel_efficiency():
    one = ScalingPoint(gpus=1, step_time_s=8, samples_per_step=8)
    eight = ScalingPoint(gpus=8, step_time_s=1.25, samples_per_step=8)
    assert abs(parallel_efficiency(one, eight) - 0.8) < 1e-9
