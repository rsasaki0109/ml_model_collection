"""Export Depth-Anything-V2-B (Hugging Face transformers port) to ONNX.

Usage: python depth_estimation/depth_anything_v2_base/export.py
Requires: torch, transformers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.estimators.hf_depth import export_hf_depth  # noqa: E402

REPO = "depth-anything/Depth-Anything-V2-Base-hf"
REVISION = "b1958afc87fb45a9e3746cb387596094de553ed8"

if __name__ == "__main__":
    export_hf_depth(REPO, REVISION, Path(__file__).parent / "weights" / "depth_anything_v2_base.onnx")
