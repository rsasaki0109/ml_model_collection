"""Semantic segmentation models exported from Hugging Face transformers.

Two graph families, detected from the ONNX output names:

* Mask2Former-style ("universal") — ``class_queries_logits [1,Q,C+1]`` and
  ``masks_queries_logits [1,Q,h,w]``. Semantic scores follow transformers'
  ``post_process_semantic_segmentation``: softmax over classes (drop the
  no-object column) x sigmoid(mask), summed over queries.
* SegFormer-style — ``logits [1,C,h,w]``.

Both are resized (bilinear, on class scores) to the source size, then argmax.
Pre-processing: RGB, resize to the artifact's fixed input size, ImageNet
mean/std. Class names are saved at export time to ``weights/labels.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from tools.mlmc.detection import ort_session
from tools.mlmc.segmentation import SemanticMap

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Segmenter:
    def __init__(self, model, provider="cuda"):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        *_, self.th, self.tw = model.meta["artifacts"]["onnx"]["input_shape"]
        self.names = json.loads((model.weights_dir / "labels.json").read_text(encoding="utf-8"))
        self.outputs = [o.name for o in self.sess.get_outputs()]

    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img = cv2.cvtColor(cv2.resize(frame_bgr, (self.tw, self.th),
                                      interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(x), (w, h)

    def infer(self, x):
        return self.sess.run(None, {"pixel_values": x})

    def class_scores(self, out) -> np.ndarray:
        """(C, h, w) per-class scores at the model's output resolution."""
        if "class_queries_logits" in self.outputs:
            cls = out[self.outputs.index("class_queries_logits")][0]     # (Q, C+1)
            masks = out[self.outputs.index("masks_queries_logits")][0]   # (Q, h, w)
            e = np.exp(cls - cls.max(-1, keepdims=True))
            prob = (e / e.sum(-1, keepdims=True))[:, :-1]
            sig = 1 / (1 + np.exp(-np.clip(masks, -88, 88)))
            q, h, w = sig.shape  # sum over queries as a matrix product (BLAS), = einsum qc,qhw->chw
            return (prob.T @ sig.reshape(q, h * w)).reshape(-1, h, w)
        return out[self.outputs.index("logits")][0]

    def postprocess(self, out, wh):
        w, h = wh
        scores = self.class_scores(out)
        # Upsampling all 150 ADE20K score maps to full resolution dominates the
        # run time, so only classes that are top-2 somewhere at low resolution
        # are upsampled; a pixel's full-resolution argmax is almost always one
        # of them (see tests). Resized in groups of 4 channels because
        # OpenCV 5 rejects arrays with many channels.
        top2 = np.argpartition(-scores, 1, axis=0)[:2]
        cand = np.unique(top2)
        hwc = np.ascontiguousarray(scores[cand].transpose(1, 2, 0))
        up = np.concatenate(
            [cv2.resize(hwc[..., i:i + 4], (w, h), interpolation=cv2.INTER_LINEAR).reshape(h, w, -1)
             for i in range(0, hwc.shape[-1], 4)], axis=-1)
        return SemanticMap(cand[up.argmax(-1)].astype(np.int32), self.names)

    def __call__(self, frame_bgr):
        x, wh = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), wh)


def export_hf_semantic(repo_id: str, revision: str, out: Path, size=(512, 512), opset: int = 17):
    """Export a transformers semantic / universal segmentation model."""
    import torch
    from transformers import AutoConfig, AutoModelForSemanticSegmentation, \
        AutoModelForUniversalSegmentation

    from tools.mlmc.export import patch_for_ort

    config = AutoConfig.from_pretrained(repo_id, revision=revision)
    universal = config.model_type in ("mask2former", "maskformer", "oneformer")
    cls = AutoModelForUniversalSegmentation if universal else AutoModelForSemanticSegmentation
    model = cls.from_pretrained(repo_id, revision=revision).eval()
    names = [config.id2label[i] for i in range(len(config.id2label))]

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            o = self.m(pixel_values=pixel_values)
            return (o.class_queries_logits, o.masks_queries_logits) if universal else o.logits

    output_names = ["class_queries_logits", "masks_queries_logits"] if universal else ["logits"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(Wrapper(model), torch.rand(1, 3, *size), str(out),
                          input_names=["pixel_values"], output_names=output_names,
                          opset_version=opset, dynamo=False)
    (out.parent / "labels.json").write_text(json.dumps(names), encoding="utf-8", newline="\n")
    changes = patch_for_ort(out)
    print(f"wrote {out}" + (f" (patched for ONNX Runtime: {changes})" if changes else ""))
