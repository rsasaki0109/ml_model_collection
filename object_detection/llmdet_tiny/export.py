"""Export LLMDet (Swin-T) to ONNX with a fixed COCO-80 text prompt.

Usage: python object_detection/llmdet_tiny/export.py
Requires: torch, transformers (MMGroundingDinoForObjectDetection)

Static shapes, so the graph runs on ONNX Runtime / TensorRT:
  pixel_values [1,3,800,1333]  image resized (shortest edge 800, longest
                               <= 1333) and padded bottom/right
  pixel_mask   [1,800,1333]    1 = valid pixel, 0 = padding
  input_ids / attention_mask / token_type_ids [1,256]
-> logits [1,900,256] (per-token), pred_boxes [1,900,4] (cxcywh, relative to
   the valid, unpadded image)

weights/llmdet_tiny_prompt_coco80.npz stores the tokenized prompt
("person. bicycle. ... toothbrush.") and the class->token positive map used
to turn token logits into class scores.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.detection import COCO80  # noqa: E402
from tools.mlmc.export import patch_for_ort  # noqa: E402

REPO = "iSEE-Laboratory/llmdet_tiny"
REVISION = "d05199165a19320a9236396e20fca0a5e065189c"
OUT = Path(__file__).parent / "weights"
H, W, L = 800, 1333, 256


def tokenize(tokenizer, names):
    text = ". ".join(names) + "."
    tok = tokenizer(text, padding="max_length", max_length=L, truncation=True,
                    return_offsets_mapping=True, return_tensors="np")
    offsets = tok.pop("offset_mapping")[0]
    pos = np.zeros((len(names), L), np.float32)
    start = 0
    for c, name in enumerate(names):
        end = start + len(name)
        for t, (a, b) in enumerate(offsets):
            if b > a and a < end and b > start:
                pos[c, t] = 1
        start = end + 2  # ". "
    if (pos.sum(1) == 0).any():
        raise SystemExit("a class has no tokens (prompt truncated?)")
    return {k: v.astype(np.int64) for k, v in tok.items()}, pos


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values, pixel_mask, input_ids, attention_mask, token_type_ids):
        o = self.m(pixel_values=pixel_values, pixel_mask=pixel_mask, input_ids=input_ids,
                   attention_mask=attention_mask, token_type_ids=token_type_ids)
        return o.logits, o.pred_boxes


def _isin(elements, test_elements, *, assume_unique=False, invert=False):
    """ONNX-exportable torch.isin (aten::isin has no ONNX symbolic)."""
    test = torch.as_tensor(test_elements, device=elements.device).reshape(-1)
    out = (elements.unsqueeze(-1) == test).any(-1)
    return ~out if invert else out


def _cumulative(x, dim, largest):
    """cummax/cummin via a triangular mask (O(n^2), fine for n = 256)."""
    x = x.transpose(dim, -1)
    n = x.shape[-1]
    tri = torch.tril(torch.ones(n, n, dtype=torch.bool, device=x.device))
    info = torch.finfo(x.dtype) if x.is_floating_point() else torch.iinfo(x.dtype)
    fill = info.min if largest else info.max
    expanded = x.unsqueeze(-2).expand(*x.shape[:-1], n, n).masked_fill(~tri, fill)
    values, indices = expanded.max(-1) if largest else expanded.min(-1)
    return values.transpose(dim, -1), indices.transpose(dim, -1)


def _cummax(x, dim):
    return torch.return_types.cummax(_cumulative(x, dim, largest=True))


def _cummin(x, dim):
    return torch.return_types.cummin(_cumulative(x, dim, largest=False))


def main():
    # aten::isin / cummax / cummin have no ONNX symbolic; swap in equivalents
    # built from exportable ops (only affects this export process).
    torch.isin = _isin
    torch.cummax = _cummax
    torch.cummin = _cummin
    proc = AutoProcessor.from_pretrained(REPO, revision=REVISION)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(REPO, revision=REVISION).eval()
    tok, pos = tokenize(proc.tokenizer, list(COCO80))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "llmdet_tiny_prompt_coco80.npz", positive_map=pos, **tok)
    pixel_mask = torch.zeros(1, H, W, dtype=torch.int64)
    pixel_mask[:, :750, :] = 1  # exercise the padding path during tracing
    args = (torch.rand(1, 3, H, W), pixel_mask,
            *(torch.from_numpy(tok[k]) for k in ("input_ids", "attention_mask", "token_type_ids")))
    with torch.no_grad():
        torch.onnx.export(Wrapper(model), args, str(OUT / "llmdet_tiny.onnx"),
                          input_names=["pixel_values", "pixel_mask", "input_ids",
                                       "attention_mask", "token_type_ids"],
                          output_names=["logits", "pred_boxes"],
                          opset_version=17, dynamo=False)
    changes = patch_for_ort(OUT / "llmdet_tiny.onnx")
    print(f"wrote {OUT / 'llmdet_tiny.onnx'} (patched for ONNX Runtime: {changes})")


if __name__ == "__main__":
    main()
