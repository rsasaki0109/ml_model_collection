"""Export SegFormer-B0 (ADE20K) (Hugging Face transformers) to ONNX at 512x512.

Usage: python segmentation/segformer_b0_ade/export.py
Requires: torch, transformers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.segmenters.hf_semantic import export_hf_semantic  # noqa: E402

REPO = "nvidia/segformer-b0-finetuned-ade-512-512"
REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"

if __name__ == "__main__":
    export_hf_semantic(REPO, REVISION, Path(__file__).parent / "weights" / "segformer_b0_ade.onnx")
