"""Export SAFMN x4 (DF2K) to ONNX.

Usage: python super_resolution/safmn_x4/export.py
Requires: torch, torchvision (imported, unused, by the upstream arch file).

Architecture: upstream ``basicsr/archs/safmn_arch.py`` with the test config
``options/test/SAFMN/test_benchmark_x4.yml``: SAFMN(dim=36, n_blocks=8,
ffn_scale=2.0, upscaling_factor=4); weights ``params``.
``F.adaptive_max_pool2d(x, (h // 2**i, w // 2**i))`` has no dynamic-shape
ONNX form; it is exported as ``max_pool2d(kernel = stride = 2**i)``, which
is identical when h and w are multiples of 8 — the runner reflect-pads the
input to a multiple of 8 and crops the output (upstream's own ONNX script
also requires multiples of 8). Graph: lr [1,3,h,w] RGB [0,1] (dynamic,
multiples of 8) -> sr [1,3,4h,4w].
"""

import types

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.export import clone_at, download  # noqa: E402
from tools.mlmc.export_sr import export, load_arch  # noqa: E402

HERE = Path(__file__).parent
CODE = "https://github.com/sunny2109/SAFMN"
COMMIT = "1a412066e927da640bc7e2a53ea60f9ffc134241"
CKPT_URL = "https://github.com/sunny2109/SAFMN/releases/download/v0/SAFMN_DF2K_x4.pth"
CKPT_SHA256 = "fe4f38fd6c1e3692bfbc6ab94f44adb815bfb54984b8886547657727af67f49d"


def _max_pool_as_adaptive(x, size):
    k = x.shape[-2] // size[0]  # = 2**i for inputs that are multiples of 8
    return torch.nn.functional.max_pool2d(x, k, k)


def build(onnx_friendly=True):
    src = clone_at(CODE, COMMIT, HERE / "weights" / "_src")
    arch = load_arch(src / "basicsr/archs/safmn_arch.py")
    if onnx_friendly:
        arch.F = types.SimpleNamespace(**{**vars(torch.nn.functional),
                                          "adaptive_max_pool2d": _max_pool_as_adaptive})
    model = arch.SAFMN(dim=36, n_blocks=8, ffn_scale=2.0, upscaling_factor=4)
    ckpt = download(CKPT_URL, HERE / "weights" / "SAFMN_DF2K_x4.pth", CKPT_SHA256)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["params"], strict=True)
    return model.eval()


if __name__ == "__main__":
    export(build(), HERE / "weights" / "safmn_x4.onnx")
