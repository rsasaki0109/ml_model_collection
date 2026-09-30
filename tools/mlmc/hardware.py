"""Hardware description and VRAM tiers."""

from __future__ import annotations

import platform
import subprocess

# (name, upper bound in MB). Measured peak VRAM is mapped onto these tiers.
VRAM_TIERS = [
    ("Tiny", 2 * 1024),
    ("Light", 4 * 1024),
    ("Consumer", 8 * 1024),
    ("Performance", 16 * 1024),
    ("Large", 24 * 1024),
    ("Huge", float("inf")),
]

# Hardware classes used to tag benchmark records. The class of a machine is
# chosen by whoever runs the benchmark (see docs/hardware.md); it is never
# derived from a model.
HARDWARE_CLASSES = (
    "edge_device",
    "igpu",
    "low_end_gpu",
    "affordable_laptop",
    "gaming_laptop",
    "consumer_desktop_gpu",
    "high_end_gpu",
    "workstation",
    "cpu_only",
)


def vram_tier(peak_mb: float | None) -> str | None:
    if peak_mb is None:
        return None
    for name, bound in VRAM_TIERS:
        if peak_mb <= bound:
            return name
    return None


def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def gpu_info(index: int = 0) -> dict | None:
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        return {
            "name": pynvml.nvmlDeviceGetName(h),
            "vram_total_mb": pynvml.nvmlDeviceGetMemoryInfo(h).total // 2**20,
            "driver": pynvml.nvmlSystemGetDriverVersion(),
        }
    except Exception:
        return None
