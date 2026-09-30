"""Export RT-DETR-R18 (Hugging Face transformers port) to ONNX.

Usage: python object_detection/rtdetr_r18vd/export.py
Requires: torch, transformers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.detectors.hf_detr import export_hf_detr  # noqa: E402

REPO = "PekingU/rtdetr_r18vd"
REVISION = "ac77a11ff0170a41b771c03264987f8ce2b0d753"

if __name__ == "__main__":
    export_hf_detr(REPO, REVISION, Path(__file__).parent / "weights" / "rtdetr_r18vd.onnx")
