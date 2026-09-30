"""Export D-FINE-S (COCO-only weights, Hugging Face port) to ONNX.

Usage: python object_detection/dfine_s/export.py
Requires: torch, transformers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.detectors.hf_detr import export_hf_detr  # noqa: E402

REPO = "ustc-community/dfine-small-coco"
REVISION = "f79e65b5fbb33ceb9d3ebba042955d7410c608f8"

if __name__ == "__main__":
    export_hf_detr(REPO, REVISION, Path(__file__).parent / "weights" / "dfine_s.onnx")
