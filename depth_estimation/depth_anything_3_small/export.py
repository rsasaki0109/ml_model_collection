"""Export Depth Anything 3 Small (monocular use) to ONNX from the official code.

Usage: python depth_estimation/depth_anything_3_small/export.py
Requires: torch, huggingface_hub, safetensors, plus the model code's imports
(omegaconf, einops, addict). The official package's full dependency list
(xformers, open3d, pycolmap, numpy<2, ...) is not needed for export; a
dedicated virtualenv is recommended:

    python -m venv --system-site-packages .venv-da3
    .venv-da3/Scripts/pip install omegaconf einops addict
    python tools/fetch_model.py depth_anything_3_small --python .venv-da3/Scripts/python

The network is built from the checkpoint's own config.json with the
upstream ``create_object`` helper, and loaded with ``strict=False`` exactly
like upstream (``PyTorchModelHubMixin`` / ``utils/model_loading.py``); the
6 missing keys belong to the auxiliary ray head, which the depth output does
not use. Graph: x [1,1,3,280,504] (one view) -> depth [1,1,280,504],
depth_conf [1,1,280,504]. 280x504 is what upstream's
``upper_bound_resize`` (process_res=504) produces for 16:9 input.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.export import clone_at, patch_for_ort  # noqa: E402

CODE = "https://github.com/ByteDance-Seed/Depth-Anything-3"
COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
WEIGHTS = "depth-anything/DA3-SMALL"
REVISION = "e08cab65ca0ec38e7826075418411ab90cab4da3"
H, W = 280, 504
HERE = Path(__file__).parent


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return out["depth"], out["depth_conf"]


def _cartesian_prod(*tensors):
    """ONNX-exportable torch.cartesian_prod (used for RoPE grid positions)."""
    grids = torch.meshgrid(*tensors, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=-1)


def main():
    torch.cartesian_prod = _cartesian_prod  # aten::cartesian_prod has no ONNX symbolic
    src = clone_at(CODE, COMMIT, HERE / "weights" / "_src")
    sys.path.insert(0, str(src / "src"))
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    from depth_anything_3.cfg import create_object  # upstream module

    cfg = json.loads(Path(hf_hub_download(WEIGHTS, "config.json", revision=REVISION)).read_text())
    model = create_object(OmegaConf.create(cfg["config"])).eval()
    state = load_file(hf_hub_download(WEIGHTS, "model.safetensors", revision=REVISION))
    missing, unexpected = model.load_state_dict(
        {k.removeprefix("model."): v for k, v in state.items()}, strict=False)
    if unexpected or any("aux" not in k for k in missing):
        raise SystemExit(f"unexpected weight mismatch: missing={missing} unexpected={unexpected}")
    out = HERE / "weights" / "depth_anything_3_small.onnx"
    with torch.no_grad():
        torch.onnx.export(Wrapper(model), torch.rand(1, 1, 3, H, W), str(out),
                          input_names=["x"], output_names=["depth", "depth_conf"],
                          opset_version=17, dynamo=False)
    changes = patch_for_ort(out)
    print(f"wrote {out}" + (f" (patched for ONNX Runtime: {changes})" if changes else ""))


if __name__ == "__main__":
    main()
