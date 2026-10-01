"""Shared export for CosPlace / EigenPlaces (``place_recognition/*/export.py``).

Builds the official network class from the pinned repository and loads the
release checkpoint (SHA-256 pinned). Two upstream side effects are avoided:
constructing the network would download ImageNet (CosPlace) or CosPlace
(EigenPlaces) backbone weights that the checkpoint overwrites anyway, so the
backbone is built without them; and GeM's ``avg_pool2d(kernel = H x W)`` is
exported as the equivalent ``adaptive_avg_pool2d(1)`` so the graph accepts any
input size (both evaluate at the native image resolution).
Graph: images [B,3,H,W] (ImageNet-normalised RGB) -> descriptor [B,D]
(L2-normalised).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

from tools.mlmc.export import clone_at, download, patch_for_ort


def _gem(x, p=3, eps=1e-6):
    return F.adaptive_avg_pool2d(x.clamp(min=eps).pow(p), 1).pow(1.0 / p)


def build(here: Path, repo: str, commit: str, package: str, ckpt_url: str, ckpt_sha256: str,
          backbone: str = "ResNet50", dim: int = 2048):
    src = clone_at(repo, commit, here / "weights" / "_src")
    sys.path.insert(0, str(src))
    import importlib
    layers = importlib.import_module(f"{package}.layers")
    net = importlib.import_module(f"{package}.{package.split('_')[0]}_network")
    layers.gem = _gem
    # no pretrained-backbone downloads: the checkpoint overwrites every tensor
    if hasattr(net, "get_pretrained_torchvision_model"):
        net.get_pretrained_torchvision_model = lambda name: getattr(torchvision.models, name.lower())()
    orig_hub_load = torch.hub.load
    torch.hub.load = lambda *a, **k: getattr(torchvision.models, backbone.lower())()
    try:
        model = net.GeoLocalizationNet(backbone, dim) if hasattr(net, "GeoLocalizationNet") \
            else net.GeoLocalizationNet_(backbone, dim)
    finally:
        torch.hub.load = orig_hub_load
    ckpt = download(ckpt_url, here / "weights" / Path(ckpt_url).name, ckpt_sha256)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return model.eval()


def export(model, out: Path, size=(480, 640)):
    x = torch.randn(1, 3, *size)
    with torch.no_grad():
        torch.onnx.export(model, x, str(out), input_names=["images"], output_names=["descriptor"],
                          dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}, "descriptor": {0: "batch"}},
                          opset_version=17, dynamo=False)
    changes = patch_for_ort(out)
    print(f"wrote {out}" + (f" (patched for ONNX Runtime: {changes})" if changes else ""))
