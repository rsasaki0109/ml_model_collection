"""Export Real-ESRGAN realesr-general-x4v3 (SRVGGNetCompact) to ONNX.

Usage: python super_resolution/realesr_general_x4v3/export.py
Requires: torch.

Architecture: Real-ESRGAN ``realesrgan/archs/srvgg_arch.py`` with the
arguments from ``inference_realesrgan.py``: SRVGGNetCompact(3, 3,
num_feat=64, num_conv=32, upscale=4, act_type='prelu'); weights ``params``.
Upstream's CLI blends this checkpoint with ``realesr-general-wdn-x4v3`` when
``--denoise_strength`` < 1 (default 0.5); this export is the plain model
(denoise strength 1). Graph: lr [1,3,h,w] RGB [0,1] (dynamic) -> sr [1,3,4h,4w].
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.export import clone_at, download  # noqa: E402
from tools.mlmc.export_sr import export, load_arch  # noqa: E402

HERE = Path(__file__).parent
CODE = "https://github.com/xinntao/Real-ESRGAN"
COMMIT = "a4abfb2979a7bbff3f69f58f58ae324608821e27"
CKPT_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
CKPT_SHA256 = "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"


def build():
    src = clone_at(CODE, COMMIT, HERE / "weights" / "_src")
    arch = load_arch(src / "realesrgan/archs/srvgg_arch.py")
    model = arch.SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
                                 upscale=4, act_type="prelu")
    ckpt = download(CKPT_URL, HERE / "weights" / "realesr-general-x4v3.pth", CKPT_SHA256)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["params"], strict=True)
    return model.eval()


if __name__ == "__main__":
    export(build(), HERE / "weights" / "realesr_general_x4v3.onnx")
