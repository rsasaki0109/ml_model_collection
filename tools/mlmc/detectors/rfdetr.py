"""RF-DETR ONNX exports (``rfdetr`` package exporter).

Mirrors ``rfdetr/export/_runtime`` (preprocess.py / decode.py) of rfdetr 1.11:
    input [1,3,S,S]: RGB, bilinear resize to S×S (no letterbox, no antialias),
                     ImageNet mean/std normalisation
    dets [1,300,4]:  cxcywh, normalised
    labels [1,300,91]: per-class logits; for COCO checkpoints the slot index
                     is the original COCO category id (no background slot)
Decode: sigmoid, top-300 over (query, class) pairs, threshold. NMS-free.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tools.mlmc.detection import COCO91_ID_TO_NAME, Detections, ort_session

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Detector:
    def __init__(self, model, provider="cuda", score_thr=0.5, top_k=300):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.size = model.meta["artifacts"]["onnx"]["input_shape"][2]
        self.score_thr, self.top_k = score_thr, top_k

    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img = cv2.cvtColor(cv2.resize(frame_bgr, (self.size, self.size),
                                      interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(x), (w, h)

    def infer(self, x):
        return self.sess.run(["dets", "labels"], {"input": x})

    def postprocess(self, out, wh):
        boxes, logits = out[0][0], out[1][0]
        prob = 1 / (1 + np.exp(-np.clip(logits, -88, 88)))  # (Q, 91)
        flat = prob.reshape(-1)
        idx = np.argsort(-flat)[: self.top_k]
        scores = flat[idx]
        q, cls = idx // prob.shape[1], idx % prob.shape[1]
        m = (scores > self.score_thr) & np.isin(cls, list(COCO91_ID_TO_NAME))
        scores, q, cls = scores[m], q[m], cls[m]
        cx, cy, bw, bh = boxes[q].T
        w, h = wh
        xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                         (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        return Detections(xyxy.astype(np.float32), scores.astype(np.float32),
                          [COCO91_ID_TO_NAME[int(c)] for c in cls])

    def __call__(self, frame_bgr):
        x, wh = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), wh)


def export_rfdetr(variant_cls: str, weights: Path, out: Path):
    """Export with the rfdetr package's own ONNX exporter (batch 1, static)."""
    import shutil
    import tempfile

    import rfdetr

    model = getattr(rfdetr, variant_cls)(pretrain_weights=str(Path(weights).resolve()))
    with tempfile.TemporaryDirectory() as tmp:
        onnx = model.export(output_dir=tmp, format="onnx", verbose=False)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(onnx), out)
    print(f"wrote {out}")
