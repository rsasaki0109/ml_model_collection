"""Export DEIM-D-FINE-S to ONNX from the official repository and checkpoint.

Usage: python object_detection/deim_dfine_s/export.py
Requires: torch, torchvision, gdown, plus upstream's import-time dependencies
(tensorboard, faster-coco-eval, calflops, scipy). A dedicated virtualenv is
recommended:

    python -m venv --system-site-packages .venv-dfine
    .venv-dfine/Scripts/pip install tensorboard "faster-coco-eval>=1.6.5" calflops scipy gdown
    python tools/fetch_model.py deim_dfine_s --python .venv-dfine/Scripts/python
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.export import clone_at, download, export_dfine_family  # noqa: E402

REPO = "https://github.com/Intellindust-AI-Lab/DEIM"
COMMIT = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"
CONFIG = "configs/deim_dfine/deim_hgnetv2_s_coco.yml"
CHECKPOINT = "gdrive:1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC"  # linked from the upstream README model table
CHECKPOINT_SHA256 = "a089c9eb798f8a0322e05f757fe8e8188eee34aa578788a35ec253bcec8a6b38"

HERE = Path(__file__).parent

if __name__ == "__main__":
    src = clone_at(REPO, COMMIT, HERE / "weights" / "_src")
    ckpt = download(CHECKPOINT, HERE / "weights" / "deim_dfine_s.pth", CHECKPOINT_SHA256)
    export_dfine_family(src, CONFIG, ckpt, HERE / "weights" / "deim_dfine_s.onnx")
