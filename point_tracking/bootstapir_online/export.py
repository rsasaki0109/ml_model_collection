"""Export online BootsTAPIR (causal, 4 refinement iteration(s), 256x256) to ONNX.

Usage: python point_tracking/bootstapir_online/export.py
Requires: torch (the inference port has no other dependencies for export).
See tools/mlmc/export_tapir.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc import export_tapir  # noqa: E402

HERE = Path(__file__).parent
RESOLUTION, ITERS = 256, 4

if __name__ == "__main__":
    export_tapir.export(HERE, "bootstapir_online", RESOLUTION, ITERS)
