"""Export SEA-RAFT (S, TartanAir + Chairs + Things checkpoint) to ONNX from the official code.

Usage: python optical_flow/sea_raft_s/export.py
Requires: torch, huggingface_hub, safetensors.

Config: upstream config/eval/sintel-S.json (iters 4). Fixed 432x768 graph,
see tools/mlmc/export_searaft.py for details.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc import export_searaft  # noqa: E402

CONFIG = "config/eval/sintel-S.json"
WEIGHTS = "MemorySlices/Tartan-C-T432x960-S"
REVISION = "a8173ddd35833392fae9465606430401a47e64ea"
H, W = 432, 768
HERE = Path(__file__).parent


def build():
    return export_searaft.build(HERE, CONFIG, WEIGHTS, REVISION)


if __name__ == "__main__":
    export_searaft.export(HERE, "sea_raft_s", CONFIG, WEIGHTS, REVISION, H, W)
