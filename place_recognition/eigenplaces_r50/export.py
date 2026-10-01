"""Export EigenPlaces-R50 (ResNet-50, 2048-D) to ONNX from the official code.

Usage: python place_recognition/eigenplaces_r50/export.py
Requires: torch, torchvision. See tools/mlmc/export_vpr.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc import export_vpr  # noqa: E402

HERE = Path(__file__).parent
REPO = "https://github.com/gmberton/EigenPlaces"
COMMIT = "a2969f71d5ea31017443af490b15273ca4c50af1"
CKPT_URL = "https://github.com/gmberton/EigenPlaces/releases/download/v1.0/ResNet50_2048_eigenplaces.pth"
CKPT_SHA256 = "98e8e21e9364a7e0990a9d7e1de5853b6d830f7f73d6bd9ee4f397575e0d41d1"


def build():
    return export_vpr.build(HERE, REPO, COMMIT, "eigenplaces_model", CKPT_URL, CKPT_SHA256)


if __name__ == "__main__":
    export_vpr.export(build(), HERE / "weights" / "eigenplaces_r50.onnx")
