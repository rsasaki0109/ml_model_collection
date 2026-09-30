"""Export D-FINE-N (COCO-only weights, Hugging Face port) to ONNX.

Usage: python object_detection/dfine_n/export.py
Requires: torch, transformers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.detectors.hf_detr import export_hf_detr  # noqa: E402

REPO = "ustc-community/dfine-nano-coco"
REVISION = "066438d3d8f0da137a37b38fdf3368fd4afceced"

if __name__ == "__main__":
    export_hf_detr(REPO, REVISION, Path(__file__).parent / "weights" / "dfine_n.onnx")
