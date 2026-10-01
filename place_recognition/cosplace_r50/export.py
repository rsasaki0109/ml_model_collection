"""Export CosPlace-R50 (ResNet-50, 2048-D) to ONNX from the official code.

Usage: python place_recognition/cosplace_r50/export.py
Requires: torch, torchvision. See tools/mlmc/export_vpr.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc import export_vpr  # noqa: E402

HERE = Path(__file__).parent
REPO = "https://github.com/gmberton/CosPlace"
COMMIT = "52b56e95ea62245281281f3bafd7b9390d19a0fd"
CKPT_URL = "https://github.com/gmberton/CosPlace/releases/download/v1.0/resnet50_2048_cosplace.pth"
CKPT_SHA256 = "1083431a283b755fd9b49302711fe1d31b061a73d3b743c2971e895224ecd89a"


def build():
    return export_vpr.build(HERE, REPO, COMMIT, "cosplace_model", CKPT_URL, CKPT_SHA256)


if __name__ == "__main__":
    export_vpr.export(build(), HERE / "weights" / "cosplace_r50.onnx")
