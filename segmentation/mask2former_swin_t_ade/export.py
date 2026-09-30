"""Export Mask2Former-SwinT (ADE20K) (Hugging Face transformers) to ONNX at 512x512.

Usage: python segmentation/mask2former_swin_t_ade/export.py
Requires: torch, transformers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.segmenters.hf_semantic import export_hf_semantic  # noqa: E402

REPO = "facebook/mask2former-swin-tiny-ade-semantic"
REVISION = "c8cf1b5e823aee214d937d0d001c1850ba44ef6a"

if __name__ == "__main__":
    export_hf_semantic(REPO, REVISION, Path(__file__).parent / "weights" / "mask2former_swin_t_ade.onnx")
