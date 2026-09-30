"""YOLOX-S (official ONNX) — pre/post-processing.

Follows YOLOX/demo/ONNXRuntime/onnx_inference.py: letterbox to the top-left
with pad 114, BGR, no mean/std normalisation, then decode grid outputs.
"""

import numpy as np

from tools.mlmc.detection import (COCO80, Detections, letterbox,
                                  nms_per_class, ort_session)

STRIDES = (8, 16, 32)


def _grids(size):
    grids, strides = [], []
    for s in STRIDES:
        n = size // s
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids).astype(np.float32), \
        np.concatenate(strides).astype(np.float32)


class Detector:
    def __init__(self, model, provider="cuda", score_thr=0.35, iou_thr=0.45):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = model.meta["artifacts"]["onnx"]["input_shape"][2]
        self.grid, self.stride = _grids(self.size)
        self.score_thr, self.iou_thr = score_thr, iou_thr

    def preprocess(self, frame_bgr):
        img, r, _ = letterbox(frame_bgr, self.size, center=False)
        x = img.transpose(2, 0, 1)[None].astype(np.float32)
        return x, r

    def infer(self, x):
        return self.sess.run(None, {self.input_name: x})[0][0]

    def postprocess(self, out, r):
        xy = (out[:, :2] + self.grid) * self.stride
        wh = np.exp(out[:, 2:4]) * self.stride
        scores_all = out[:, 4:5] * out[:, 5:]
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(cls)), cls]
        m = scores > self.score_thr
        xy, wh, cls, scores = xy[m], wh[m], cls[m], scores[m]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r
        keep = nms_per_class(boxes, scores, cls, self.iou_thr)
        return Detections(boxes[keep].astype(np.float32), scores[keep],
                          [COCO80[i] for i in cls[keep]])

    def __call__(self, frame_bgr):
        x, r = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), r)
