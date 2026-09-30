"""Peak GPU memory measurement.

Method ``nvml_device_used_delta``: poll the *device-wide* used memory through
NVML in a background thread and report ``peak - baseline``. Per-process
accounting is not available on Windows (WDDM), so this method is used on every
platform for consistency. It includes the CUDA context and runtime
workspaces, and is disturbed by other processes using the same GPU, so run
benchmarks on an otherwise idle GPU.
"""

from __future__ import annotations

import threading
import time

METHOD = "nvml_device_used_delta"


class PeakVramMonitor:
    def __init__(self, index: int = 0, interval_s: float = 0.005):
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_mb = self._used_mb()
        self.peak_mb = self.baseline_mb

    def _used_mb(self) -> float:
        return self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20

    def _run(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._used_mb())
            time.sleep(self._interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
        self.peak_mb = max(self.peak_mb, self._used_mb())

    @property
    def delta_mb(self) -> float:
        return self.peak_mb - self.baseline_mb
