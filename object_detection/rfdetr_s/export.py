"""Export RF-DETR-S (COCO) to ONNX with the rfdetr package's exporter.

Usage: python object_detection/rfdetr_s/export.py
Requires: rfdetr==1.11.0 (Apache-2.0). A dedicated virtualenv is recommended:

    python -m venv --system-site-packages .venv-rfdetr
    .venv-rfdetr/Scripts/pip install rfdetr==1.11.0
    python tools/fetch_model.py rfdetr_s --python .venv-rfdetr/Scripts/python
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.detectors.rfdetr import export_rfdetr  # noqa: E402
from tools.mlmc.export import download  # noqa: E402

# URL and MD5 (fb...) are listed in rfdetr/assets/model_weights.py (1.11.0).
WEIGHTS_URL = "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth"
WEIGHTS_SHA256 = "d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33"

HERE = Path(__file__).parent

if __name__ == "__main__":
    pth = download(WEIGHTS_URL, HERE / "weights" / "rfdetr_s.pth", WEIGHTS_SHA256)
    export_rfdetr("RFDETRSmall", pth, HERE / "weights" / "rfdetr_s.onnx")
