"""Export Depth-Anything-V2-S (Hugging Face transformers port) to ONNX.

Usage: python depth_estimation/depth_anything_v2_small/export.py
Requires: torch, transformers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.estimators.hf_depth import export_hf_depth  # noqa: E402

REPO = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"

if __name__ == "__main__":
    export_hf_depth(REPO, REVISION, Path(__file__).parent / "weights" / "depth_anything_v2_small.onnx")
