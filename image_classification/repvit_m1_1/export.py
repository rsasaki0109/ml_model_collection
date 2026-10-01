"""Export RepViT-M1.1 (timm, distilled, ImageNet-1k) to ONNX.

Usage: python image_classification/repvit_m1_1/export.py
Requires: torch, timm, huggingface_hub, safetensors.

timm ``repvit_m1_1`` with the pinned Hugging Face checkpoint, structurally
re-parameterised for inference (``timm.utils.reparameterize_model``, as
timm's ``onnx_export.py --reparam``). Graph: input0 [B,3,224,224] -> output0
[B,1000] logits.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.export import patch_for_ort  # noqa: E402

REPO = "timm/repvit_m1_1.dist_450e_in1k"
REVISION = "ff441038a42982fff1c8c75be7c794f9ba76db72"
SHA256 = "071d227f69657c02d1bee8ae3317772ce7f9540af6f7865bf27e3f051e2cd160"
HERE = Path(__file__).parent


def build():
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from timm.utils.model import reparameterize_model

    from tools.mlmc.export import sha256
    ckpt = hf_hub_download(REPO, "model.safetensors", revision=REVISION)
    if sha256(Path(ckpt)) != SHA256:
        raise SystemExit("sha256 mismatch for model.safetensors")
    model = timm.create_model("repvit_m1_1", pretrained=False, exportable=True)
    model.load_state_dict(load_file(ckpt), strict=True)
    return reparameterize_model(model.eval())


if __name__ == "__main__":
    model = build()
    out = HERE / "weights" / "repvit_m1_1.onnx"
    out.parent.mkdir(exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, torch.randn(1, 3, 224, 224), str(out), input_names=["input0"],
                          output_names=["output0"], dynamic_axes={"input0": {0: "batch"}, "output0": {0: "batch"}},
                          opset_version=17, dynamo=False)
    changes = patch_for_ort(out)
    print(f"wrote {out}" + (f" (patched for ONNX Runtime: {changes})" if changes else ""))
