"""Export RF-DETR-N (COCO) to ONNX with the rfdetr package's exporter.

Usage: python object_detection/rfdetr_n/export.py
Requires: rfdetr==1.11.0 (Apache-2.0). A dedicated virtualenv is recommended:

    python -m venv --system-site-packages .venv-rfdetr
    .venv-rfdetr/Scripts/pip install rfdetr==1.11.0
    python tools/fetch_model.py rfdetr_n --python .venv-rfdetr/Scripts/python
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.detectors.rfdetr import export_rfdetr  # noqa: E402
from tools.mlmc.export import download  # noqa: E402

# URL and MD5 (fb...) are listed in rfdetr/assets/model_weights.py (1.11.0).
WEIGHTS_URL = "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth"
WEIGHTS_SHA256 = "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d"

HERE = Path(__file__).parent

if __name__ == "__main__":
    pth = download(WEIGHTS_URL, HERE / "weights" / "rfdetr_n.pth", WEIGHTS_SHA256)
    export_rfdetr("RFDETRNano", pth, HERE / "weights" / "rfdetr_n.onnx")
