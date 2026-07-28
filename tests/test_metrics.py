from matrixgame_systems.profiling.metrics import aggregate_phases, derive_metrics


def test_distributed_phase_uses_slowest_rank_not_sum():
    events = [
        {"name": "forward", "rank": 0, "wall_ms": 10, "cuda_ms": 9},
        {"name": "forward", "rank": 1, "wall_ms": 12, "cuda_ms": 11},
    ]
    summary = aggregate_phases(events)
    assert summary["forward"]["total_ms"] == 11
    assert summary["forward"]["critical_rank"] == 1


def test_derived_metrics():
    phases = {
        "step": {"total_ms": 100},
        "forward": {"total_ms": 50},
        "dataloader": {"total_ms": 10},
        "checkpoint": {"total_ms": 1},
    }
    result = derive_metrics(
        wall_time_s=1,
        frames=10,
        num_gpus=2,
        phase_summary=phases,
        gpu_summary={"average_utilization_percent": 80},
        forward_flops=100e12,
        peak_tflops_per_gpu=1000,
    )
    assert result["frames_per_gpu_hour"] == 18000
    assert result["dataloader_stall_fraction"] == 0.1
    assert result["checkpoint_pause_fraction"] == 0.01
    assert result["achieved_tflops"] == 2000
    assert result["estimated_mfu"] == 1.0
