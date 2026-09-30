"""YOLO11n (Ultralytics ONNX export) — pre/post-processing.

This file is independent of the ultralytics package: centered letterbox with
pad 114, RGB in [0, 1]; output rows are cx, cy, w, h followed by 80 class
scores, followed by class-aware NMS.
"""

import cv2
import numpy as np

from tools.mlmc.detection import (COCO80, Detections, letterbox,
                                  nms_per_class, ort_session)


class Detector:
    def __init__(self, model, provider="cuda", score_thr=0.35, iou_thr=0.45):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = model.meta["artifacts"]["onnx"]["input_shape"][2]
        self.score_thr, self.iou_thr = score_thr, iou_thr

    def preprocess(self, frame_bgr):
        img, r, pad = letterbox(frame_bgr, self.size, center=True)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        return x, (r, pad)

    def infer(self, x):
        return self.sess.run(None, {self.input_name: x})[0][0]

    def postprocess(self, out, meta):
        r, (px, py) = meta
        out = out.T                                    # (8400, 84)
        cls = out[:, 4:].argmax(1)
        scores = out[np.arange(len(cls)), 4 + cls]
        m = scores > self.score_thr
        out, cls, scores = out[m], cls[m], scores[m]
        cx, cy, w, h = out[:, :4].T
        boxes = np.stack([cx - w / 2 - px, cy - h / 2 - py,
                          cx + w / 2 - px, cy + h / 2 - py], 1) / r
        keep = nms_per_class(boxes, scores, cls, self.iou_thr)
        return Detections(boxes[keep].astype(np.float32), scores[keep],
                          [COCO80[i] for i in cls[keep]])

    def __call__(self, frame_bgr):
        x, meta = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), meta)
