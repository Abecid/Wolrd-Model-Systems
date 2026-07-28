import json

from matrixgame_systems.profiling.trace_parser import parse_chrome_trace


def test_trace_union_and_nccl_fraction(tmp_path):
    trace = {
        "traceEvents": [
            {"ph": "X", "cat": "cuda_kernel", "name": "gemm", "ts": 0, "dur": 100},
            {"ph": "X", "cat": "cuda_kernel", "name": "ncclAllReduce", "ts": 50, "dur": 100},
            {"ph": "X", "cat": "cuda_kernel", "name": "other", "ts": 160, "dur": 40},
        ]
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))
    result = parse_chrome_trace(path)
    assert abs(result["gpu_busy_ms"] - 0.19) < 1e-9
    assert abs(result["nccl_ms"] - 0.1) < 1e-9
    assert abs(result["nccl_fraction_of_gpu_busy"] - 100 / 190) < 1e-9
