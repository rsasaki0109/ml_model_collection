"""Export Real-ESRGAN x4plus (RRDBNet, GAN-trained) to ONNX.

Usage: python super_resolution/real_esrgan_x4plus/export.py
Requires: torch.

Architecture: BasicSR ``rrdbnet_arch.py`` (the module Real-ESRGAN's
``inference_realesrgan.py`` imports), built with the arguments from that
script: RRDBNet(3, 3, num_feat=64, num_block=23, num_grow_ch=32, scale=4);
weights ``params_ema``. Graph: lr [1,3,h,w] RGB [0,1] (dynamic) ->
sr [1,3,4h,4w].
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.export import clone_at, download  # noqa: E402
from tools.mlmc.export_sr import export, load_arch  # noqa: E402

HERE = Path(__file__).parent
BASICSR = "https://github.com/XPixelGroup/BasicSR"
BASICSR_COMMIT = "8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a"
CKPT_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
CKPT_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"


def build():
    src = clone_at(BASICSR, BASICSR_COMMIT, HERE / "weights" / "_basicsr")
    arch = load_arch(src / "basicsr/archs/rrdbnet_arch.py", src / "basicsr/archs/arch_util.py")
    model = arch.RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    ckpt = download(CKPT_URL, HERE / "weights" / "RealESRGAN_x4plus.pth", CKPT_SHA256)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["params_ema"], strict=True)
    return model.eval()


if __name__ == "__main__":
    export(build(), HERE / "weights" / "real_esrgan_x4plus.onnx")
