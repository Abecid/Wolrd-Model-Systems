from matrixgame_systems.distributed.regression import compare


def payload(wall, memory, throughput):
    return {
        "workload_fingerprint": "same",
        "wall_time_s": wall,
        "gpu": {"peak_physical_hbm_bytes": memory},
        "derived": {"frames_per_gpu_hour": throughput},
    }


def test_regression_gate():
    baseline = payload(10, 100, 10)
    assert compare(baseline, payload(9, 99, 11)) == []
    failures = compare(baseline, payload(11, 120, 8))
    assert len(failures) == 3
