"""Export RT-DETRv4-S to ONNX from the official repository and checkpoint.

Usage: python object_detection/rtdetrv4_s/export.py
Requires: torch, torchvision, gdown, plus upstream's import-time dependencies
(tensorboard, faster-coco-eval, calflops, scipy). A dedicated virtualenv is
recommended:

    python -m venv --system-site-packages .venv-dfine
    .venv-dfine/Scripts/pip install tensorboard "faster-coco-eval>=1.6.5" calflops scipy gdown
    python tools/fetch_model.py rtdetrv4_s --python .venv-dfine/Scripts/python
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.export import clone_at, download, export_dfine_family  # noqa: E402

REPO = "https://github.com/RT-DETRs/RT-DETRv4"
COMMIT = "55fefaaed7efe2a5f72d0a18fd4e05965e35c292"
CONFIG = "configs/rtv4/rtv4_hgnetv2_s_coco.yml"
CHECKPOINT = "gdrive:1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z"  # linked from the upstream README model table
CHECKPOINT_SHA256 = "238a3f6537bf3b75b55e73f91f9d4cec8d21259b4908b3f21896f3e038b5a3ee"

HERE = Path(__file__).parent

if __name__ == "__main__":
    src = clone_at(REPO, COMMIT, HERE / "weights" / "_src")
    ckpt = download(CHECKPOINT, HERE / "weights" / "rtdetrv4_s.pth", CHECKPOINT_SHA256)
    export_dfine_family(src, CONFIG, ckpt, HERE / "weights" / "rtdetrv4_s.onnx")
