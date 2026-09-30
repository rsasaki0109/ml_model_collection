"""Export YOLO11n to ONNX with the Ultralytics exporter.

Usage: python object_detection/yolo11n/export.py
Requires: ultralytics (AGPL-3.0). A dedicated virtualenv is recommended:

    python -m venv --system-site-packages .venv-ultralytics
    .venv-ultralytics/Scripts/pip install ultralytics   # or bin/pip
    python tools/fetch_model.py yolo11n --python .venv-ultralytics/Scripts/python
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
OUT_DIR = Path(__file__).parent / "weights"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pt = OUT_DIR / "yolo11n.pt"
    if not pt.exists():
        import urllib.request
        urllib.request.urlretrieve(WEIGHTS_URL, pt)
    onnx = YOLO(str(pt)).export(format="onnx", imgsz=640, opset=17,
                               simplify=False, dynamic=False, nms=False)
    if Path(onnx).resolve() != (OUT_DIR / "yolo11n.onnx").resolve():
        shutil.move(onnx, OUT_DIR / "yolo11n.onnx")
    print(f"wrote {OUT_DIR / 'yolo11n.onnx'}")


if __name__ == "__main__":
    main()
