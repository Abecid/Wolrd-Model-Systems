from __future__ import annotations

import platform
import socket
from typing import Any

# Dense BF16/FP16 tensor-core peaks in TFLOP/s. These are convenience defaults,
# not a substitute for recording the exact SKU, power mode, clock, and sparsity mode.
GPU_PEAK_TFLOPS: dict[str, float] = {
    "NVIDIA H100 80GB HBM3": 989.0,
    "NVIDIA H100 PCIe": 756.0,
    "NVIDIA H200": 989.0,
    "NVIDIA A100-SXM4-80GB": 312.0,
    "NVIDIA A100-PCIE-80GB": 312.0,
    "NVIDIA L40S": 362.0,
}


def torch_hardware_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
        )
        if torch.cuda.is_available():
            result["gpus"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        result["torch"] = None
        result["cuda_available"] = False
        result["gpu_count"] = 0
    return result


def infer_peak_tflops(gpu_name: str | None) -> float | None:
    if not gpu_name:
        return None
    if gpu_name in GPU_PEAK_TFLOPS:
        return GPU_PEAK_TFLOPS[gpu_name]
    normalized = gpu_name.lower()
    for name, peak in GPU_PEAK_TFLOPS.items():
        if name.lower() in normalized or normalized in name.lower():
            return peak
    return None
