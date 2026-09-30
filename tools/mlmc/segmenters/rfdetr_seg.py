"""RF-DETR-Seg ONNX exports (``rfdetr`` package exporter).

Same input and box/class decode as RF-DETR detection
(tools/mlmc/detectors/rfdetr.py), plus a mask head:
    masks [1,Q,Hm,Wm]: per-query mask logits at stride 4
Mask post-processing mirrors rfdetr ``PostProcess._postprocess_masks``:
gather the selected queries' masks, bilinearly upsample to the source image
size, foreground = logit > 0.
"""

from __future__ import annotations

import cv2
import numpy as np

from tools.mlmc.detection import COCO91_ID_TO_NAME, ort_session
from tools.mlmc.detectors.rfdetr import MEAN, STD
from tools.mlmc.segmentation import InstanceMasks


class Segmenter:
    def __init__(self, model, provider="cuda", score_thr=0.5, top_k=100):
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
        return self.sess.run(["dets", "labels", "masks"], {"input": x})

    def postprocess(self, out, wh):
        boxes, logits, masks = out[0][0], out[1][0], out[2][0]
        prob = 1 / (1 + np.exp(-np.clip(logits, -88, 88)))   # (Q, 91)
        flat = prob.reshape(-1)
        idx = np.argsort(-flat)[: self.top_k]
        scores = flat[idx]
        q, cls = idx // prob.shape[1], idx % prob.shape[1]
        keep = (scores > self.score_thr) & np.isin(cls, list(COCO91_ID_TO_NAME))
        scores, q, cls = scores[keep], q[keep], cls[keep]
        w, h = wh
        cx, cy, bw, bh = boxes[q].T
        xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                         (cx + bw / 2) * w, (cy + bh / 2) * h], 1).astype(np.float32)
        full = np.zeros((len(q), h, w), bool)
        for i, qi in enumerate(q):
            full[i] = cv2.resize(masks[qi], (w, h), interpolation=cv2.INTER_LINEAR) > 0
        return InstanceMasks(xyxy, scores.astype(np.float32),
                             [COCO91_ID_TO_NAME[int(c)] for c in cls], full)

    def __call__(self, frame_bgr):
        x, wh = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), wh)
