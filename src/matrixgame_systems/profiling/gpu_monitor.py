from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class GpuSample:
    timestamp_s: float
    gpu_index: int
    utilization_percent: float | None
    memory_utilization_percent: float | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    power_watts: float | None
    sm_clock_mhz: int | None


class GpuMonitor:
    def __init__(
        self,
        output_path: str | Path,
        *,
        gpu_index: int | None = None,
        interval_ms: int = 100,
    ) -> None:
        self.output_path = Path(output_path)
        self.gpu_index = (
            int(os.environ.get("LOCAL_RANK", "0")) if gpu_index is None else int(gpu_index)
        )
        self.interval_s = max(interval_ms, 20) / 1000.0
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None

    def _initialize(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception:
            self._nvml = None
            self._handle = None

    def _sample(self) -> GpuSample:
        now = time.time()
        if self._nvml is not None and self._handle is not None:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                power = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                clock = self._nvml.nvmlDeviceGetClockInfo(
                    self._handle, self._nvml.NVML_CLOCK_SM
                )
                return GpuSample(
                    timestamp_s=now,
                    gpu_index=self.gpu_index,
                    utilization_percent=float(util.gpu),
                    memory_utilization_percent=float(util.memory),
                    memory_used_bytes=int(memory.used),
                    memory_total_bytes=int(memory.total),
                    power_watts=float(power),
                    sm_clock_mhz=int(clock),
                )
            except Exception:
                pass
        try:
            import torch

            used = int(torch.cuda.memory_allocated(self.gpu_index)) if torch.cuda.is_available() else None
            total = (
                int(torch.cuda.get_device_properties(self.gpu_index).total_memory)
                if torch.cuda.is_available()
                else None
            )
        except Exception:
            used = total = None
        return GpuSample(now, self.gpu_index, None, None, used, total, None, None)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.samples.append(self._sample())

    def start(self) -> None:
        if self._thread is not None:
            return
        self._initialize()
        self.samples.append(self._sample())
        self._thread = threading.Thread(target=self._loop, name="mgs-gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> Path:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.samples.append(self._sample())
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps([asdict(sample) for sample in self.samples], indent=2), encoding="utf-8"
        )
        return self.output_path

    def __enter__(self) -> "GpuMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
