"""Export OWLv2 B/16 (ensemble checkpoint) to ONNX, plus tokenized COCO prompts.

Usage: python object_detection/owlv2_b16/export.py
Requires: torch, transformers

Outputs (weights/):
  owlv2_b16.onnx               pixel_values [1,3,960,960], input_ids [Q,16],
                               attention_mask [Q,16] -> logits [1,3600,Q],
                               pred_boxes [1,3600,4] (Q is dynamic)
  owlv2_b16_prompts_coco80.npz input_ids / attention_mask for the 80 COCO
                               class names ("a photo of a {name}"), so the
                               default detector needs no tokenizer at runtime
"""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Owlv2ForObjectDetection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mlmc.detection import COCO80  # noqa: E402

REPO = "google/owlv2-base-patch16-ensemble"
REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
OUT = Path(__file__).parent / "weights"
PROMPT = "a photo of a {}"


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values, input_ids, attention_mask):
        o = self.m(pixel_values=pixel_values, input_ids=input_ids,
                   attention_mask=attention_mask)
        return o.logits, o.pred_boxes


def main():
    proc = AutoProcessor.from_pretrained(REPO, revision=REVISION)
    model = Owlv2ForObjectDetection.from_pretrained(REPO, revision=REVISION).eval()
    tok = proc.tokenizer([PROMPT.format(n) for n in COCO80], padding="max_length",
                         max_length=16, truncation=True, return_tensors="np")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "owlv2_b16_prompts_coco80.npz",
             input_ids=tok["input_ids"].astype(np.int64),
             attention_mask=tok["attention_mask"].astype(np.int64))
    ids = torch.from_numpy(tok["input_ids"].astype(np.int64))
    mask = torch.from_numpy(tok["attention_mask"].astype(np.int64))
    with torch.no_grad():
        torch.onnx.export(
            Wrapper(model), (torch.rand(1, 3, 960, 960), ids, mask), str(OUT / "owlv2_b16.onnx"),
            input_names=["pixel_values", "input_ids", "attention_mask"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={"input_ids": {0: "Q"}, "attention_mask": {0: "Q"}, "logits": {2: "Q"}},
            opset_version=17, dynamo=False)
    print(f"wrote {OUT / 'owlv2_b16.onnx'}")


if __name__ == "__main__":
    main()
