"""SigLIP 2 B/16-224 zero-shot classifier: image tower ONNX + ImageNet text embeddings.

Usage: python image_classification/siglip2_b16_224/export.py
Requires: torch, transformers, huggingface_hub, safetensors.

1. Downloads the image tower ``onnx/vision_model.onnx`` from
   onnx-community/siglip2-base-patch16-224-ONNX (pinned revision, SHA-256).
2. Computes the 1000 ImageNet class text embeddings once with the official
   google/siglip2-base-patch16-224 checkpoint (pinned revision):
   open_clip ``IMAGENET_CLASSNAMES`` x ``SIMPLE_IMAGENET_TEMPLATES`` (7
   prompts, pinned file), lower-cased, padded to 64 tokens (as SigLIP 2 was
   trained), ``get_text_features`` -> L2-normalise -> mean over templates ->
   L2-normalise. Saved as ``imagenet_text_embeddings.npy`` (1000 x 768,
   float32), so the 1.1 GB text tower is not needed at run time.
"""

import ast
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.export import download, sha256  # noqa: E402

ONNX_URL = ("https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/"
            "ba1f3b0843f24bc5417d38e19c37b287d719b2f4/onnx/vision_model.onnx")
ONNX_SHA256 = "c0573e3f4140c3a7c4e9cc5912bd6b26a033b46a6a8e8af26cbea262b163bcad"
MODEL = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
PROMPTS_URL = ("https://raw.githubusercontent.com/mlfoundations/open_clip/"
               "7067b350050dc409bfdfdbb3f0cc6c9a1651c000/src/open_clip/zero_shot_metadata.py")
PROMPTS_SHA256 = "3bd8adb168bdf80fd681a71258e67f2c4d0f8ed73eae9b504ed5063e1726bbe9"
HERE = Path(__file__).parent


def prompts():
    src = urllib.request.urlopen(PROMPTS_URL).read()
    import hashlib
    if hashlib.sha256(src).hexdigest() != PROMPTS_SHA256:
        raise SystemExit("sha256 mismatch for zero_shot_metadata.py")
    vals = {}
    for n in ast.parse(src.decode("utf-8")).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id == "IMAGENET_CLASSNAMES":
                vals["names"] = ast.literal_eval(n.value)
            elif n.targets[0].id == "SIMPLE_IMAGENET_TEMPLATES":
                # lambda c: f'... {c}.' -> keep the f-string pattern
                vals["templates"] = [ast.unparse(e.body) for e in n.value.elts]
    return vals["names"], vals["templates"]


def fmt(template: str, name: str) -> str:
    """Evaluate one ``f'...{c}...'`` template from the pinned open_clip file."""
    return eval(template, {"c": name})  # noqa: S307 (pinned, sha-checked source)


def main():
    out_dir = HERE / "weights"
    out_dir.mkdir(exist_ok=True)
    download(ONNX_URL, out_dir / "siglip2_b16_224.onnx", ONNX_SHA256)
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).eval()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    names, templates = prompts()
    embs = []
    with torch.no_grad():
        for name in names:
            texts = [fmt(t, name).lower() for t in templates]
            ids = tok(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            f = model.get_text_features(input_ids=ids["input_ids"])
            f = getattr(f, "pooler_output", f)
            f = torch.nn.functional.normalize(f, dim=-1).mean(0)
            embs.append(torch.nn.functional.normalize(f, dim=0).numpy())
    emb = np.stack(embs).astype(np.float32)
    np.save(out_dir / "imagenet_text_embeddings.npy", emb)
    print("wrote", out_dir / "siglip2_b16_224.onnx", "and text embeddings", emb.shape,
          sha256(out_dir / "imagenet_text_embeddings.npy"))


if __name__ == "__main__":
    main()
