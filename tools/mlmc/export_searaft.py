"""Shared SEA-RAFT ONNX export (used by optical_flow/sea_raft_*/export.py).

Builds ``core/raft.py::RAFT`` from the upstream eval config and loads the
Hugging Face ``model.safetensors`` strictly. Upstream's constructor would
download torchvision ImageNet ResNet weights to initialise the encoders; that
initialisation is skipped because every tensor is overwritten by the
checkpoint. Graph: image1, image2 [1,3,H,W] RGB float 0-255 -> flow
[1,2,H,W] (pixels), ``test_mode=True``, ``iters`` from the config (4).
The model's own normalisation (2*x/255-1) and /8 padding are inside the graph.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from tools.mlmc.export import clone_at, patch_for_ort

CODE = "https://github.com/princeton-vl/SEA-RAFT"
COMMIT = "9137517ba24e628442aec097d3afe71d03503b75"


class Wrapper(torch.nn.Module):
    def __init__(self, m, iters):
        super().__init__()
        self.m, self.iters = m, iters

    def forward(self, image1, image2):
        return self.m(image1, image2, iters=self.iters, test_mode=True)["flow"][-1]


def build(here: Path, config: str, repo: str, revision: str):
    src = clone_at(CODE, COMMIT, here / "weights" / "_src")
    sys.path.insert(0, str(src / "core"))
    sys.path.insert(0, str(src))
    import extractor  # upstream modules
    from config.parser import json_to_args
    from huggingface_hub import hf_hub_download
    from raft import RAFT
    from safetensors.torch import load_file

    args = json_to_args(str(src / config))
    orig = extractor.ResNetFPN.__init__

    def no_download(self, *a, **kw):
        kw["init_weight"] = False
        orig(self, *a, **kw)

    extractor.ResNetFPN.__init__ = no_download
    try:
        model = RAFT(args)
    finally:
        extractor.ResNetFPN.__init__ = orig
    state = load_file(hf_hub_download(repo, "model.safetensors", revision=revision))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # ``downsample.1`` is the same BatchNorm object as ``bn3`` (upstream
    # registers it twice); safetensors stores shared tensors once.
    aliased = all(".downsample.1." in k and model.get_submodule(k.rsplit(".downsample.1.", 1)[0]).bn3
                  is model.get_submodule(k.rsplit(".", 1)[0]) for k in missing)
    if unexpected or not aliased:
        raise SystemExit(f"unexpected weight mismatch: missing={missing} unexpected={unexpected}")
    return Wrapper(model.eval(), args.iters)


def export(here: Path, name: str, config: str, repo: str, revision: str, h: int, w: int):
    model = build(here, config, repo, revision)
    out = here / "weights" / f"{name}.onnx"
    x = torch.rand(1, 3, h, w) * 255
    with torch.no_grad():
        torch.onnx.export(model, (x, x.flip(-1)), str(out),
                          input_names=["image1", "image2"], output_names=["flow"],
                          opset_version=17, dynamo=False)
    changes = patch_for_ort(out)
    print(f"wrote {out}" + (f" (patched for ONNX Runtime: {changes})" if changes else ""))
