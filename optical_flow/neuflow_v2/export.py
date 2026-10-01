"""Export NeuFlow v2 (FlyingThings checkpoint) to ONNX from the official code.

Usage: python optical_flow/neuflow_v2/export.py
Requires: torch (the model code has no other dependencies).

Follows upstream ``infer.py``: Conv+BN of every ``ConvBlock`` fused, 1 global
refinement iteration at 1/16 and 8 at 1/8 (the paper default), but FP32
(``init_bhwd(amp=False)``) so ONNX Runtime / TensorRT pick the precision.
``init_bhwd`` bakes batch and resolution into buffers, so the graph is fixed
at 432x768 (upstream ``infer.py`` resolution; H and W must be multiples of 16).
Graph: image1, image2 [1,3,432,768] RGB float 0-255 -> flow [1,2,432,768]
(pixels, image1 -> image2). Upstream divides the inputs by 255 in place;
the wrapper clones them first.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.export import clone_at, patch_for_ort, sha256  # noqa: E402

CODE = "https://github.com/neufieldrobotics/NeuFlow_v2"
COMMIT = "204b5e3744461d90303b9ff82caa7a1bb56a2ca2"
CKPT = "neuflow_things.pth"  # in the repository at COMMIT
CKPT_SHA256 = "733d13b1b2202adefcc99bd1f0fceb89fc90da5479f9826fa3f17ff42c4bdbe0"
H, W = 432, 768
HERE = Path(__file__).parent


def fuse_conv_and_bn(conv, bn):
    """Same arithmetic as upstream infer.py."""
    fused = torch.nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                            stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                            groups=conv.groups, bias=True).requires_grad_(False)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))
    b_conv = torch.zeros(conv.weight.shape[0]) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fused


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image1, image2):
        return self.m(image1.clone(), image2.clone())[-1]


def build():
    src = clone_at(CODE, COMMIT, HERE / "weights" / "_src")
    if sha256(src / CKPT) != CKPT_SHA256:
        raise SystemExit(f"sha256 mismatch for {CKPT}")
    sys.path.insert(0, str(src))
    from NeuFlow.backbone_v7 import ConvBlock  # upstream modules
    from NeuFlow.neuflow import NeuFlow

    model = NeuFlow()
    model.load_state_dict(torch.load(src / CKPT, map_location="cpu")["model"], strict=True)
    for m in model.modules():
        if type(m) is ConvBlock:
            m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)
            m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)
            delattr(m, "norm1")
            delattr(m, "norm2")
            m.forward = m.forward_fuse
    model.eval()
    model.init_bhwd(1, H, W, "cpu", amp=False)
    return Wrapper(model)


def main():
    model = build()
    out = HERE / "weights" / "neuflow_v2.onnx"
    x = torch.rand(1, 3, H, W) * 255
    with torch.no_grad():
        torch.onnx.export(model, (x, x.flip(-1)), str(out),
                          input_names=["image1", "image2"], output_names=["flow"],
                          opset_version=17, dynamo=False)
    changes = patch_for_ort(out)
    print(f"wrote {out}" + (f" (patched for ONNX Runtime: {changes})" if changes else ""))


if __name__ == "__main__":
    main()
