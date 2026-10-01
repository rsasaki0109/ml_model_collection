"""SCRFD-10GF with keypoints (InsightFace buffalo_l ``det_10g.onnx``).

As upstream ``detection/scrfd/tools/scrfd.py``: **RGB**, (x - 127.5) / 128,
aspect-preserving resize into the top-left of a 640x640 canvas (the graph's
output shapes are fixed for 640x640). Strides (8, 16, 32), 2 anchors per
cell, anchor centres (x, y) * stride; boxes = centre -/+ distances * stride,
landmarks = centre + offsets * stride; scores already sigmoid. NMS IoU 0.4.
"""

import cv2
import numpy as np

from tools.mlmc.detection import ort_session
from tools.mlmc.face import Faces, select

STRIDES = (8, 16, 32)
SIZE = 640


class FaceDetector:
    def __init__(self, model, provider="cuda", score_thr=0.5, nms_iou=0.4):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.input_name = self.sess.get_inputs()[0].name
        self.score_thr, self.nms_iou = score_thr, nms_iou
        self.centers = []
        for s in STRIDES:
            n = SIZE // s
            c = np.stack(np.mgrid[:n, :n][::-1], -1).reshape(-1, 2).astype(np.float32) * s
            self.centers.append(np.repeat(c, 2, axis=0))

    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        r = min(SIZE / h, SIZE / w)
        nh, nw = int(h * r), int(w * r)
        canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame_bgr, (nw, nh))
        x = (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 128.0
        return x.transpose(2, 0, 1)[None], r

    def infer(self, x):
        return self.sess.run(None, {self.input_name: x})

    def postprocess(self, out, r):
        boxes, scores, lms = [], [], []
        for i, (s, c) in enumerate(zip(STRIDES, self.centers)):
            sc, bb, kp = out[i][:, 0], out[i + 3] * s, out[i + 6].reshape(-1, 5, 2) * s
            boxes.append(np.concatenate([c - bb[:, :2], c + bb[:, 2:]], 1))
            scores.append(sc)
            lms.append(c[:, None] + kp)
        return select(np.concatenate(boxes) / r, np.concatenate(scores), np.concatenate(lms) / r,
                      self.score_thr, self.nms_iou)

    def __call__(self, frame_bgr):
        x, r = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), r)
