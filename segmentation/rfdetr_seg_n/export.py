"""Export RF-DETR-Seg-N (COCO) to ONNX with the rfdetr package's exporter.

Usage: python segmentation/rfdetr_seg_n/export.py
Requires: rfdetr==1.11.0 (Apache-2.0); see object_detection/rfdetr_n/export.py
for the recommended virtualenv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.detectors.rfdetr import export_rfdetr  # noqa: E402
from tools.mlmc.export import download  # noqa: E402

# URL and MD5 (99954977...) are listed in rfdetr/assets/model_weights.py (1.11.0).
WEIGHTS_URL = "https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth"
WEIGHTS_SHA256 = "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"

HERE = Path(__file__).parent

if __name__ == "__main__":
    pth = download(WEIGHTS_URL, HERE / "weights" / "rfdetr_seg_n.pth", WEIGHTS_SHA256)
    export_rfdetr("RFDETRSegNano", pth, HERE / "weights" / "rfdetr_seg_n.onnx")
