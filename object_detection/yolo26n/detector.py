"""YOLO26n (Ultralytics ONNX export, end-to-end head) — pre/post-processing.

Independent of the ultralytics package: centered letterbox with pad 114,
RGB in [0, 1]. The graph already selects the top 300 detections without NMS;
rows are x1, y1, x2, y2, score, class in letterboxed input pixels.
"""

import cv2
import numpy as np

from tools.mlmc.detection import COCO80, Detections, letterbox, ort_session


class Detector:
    def __init__(self, model, provider="cuda", score_thr=0.35):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = model.meta["artifacts"]["onnx"]["input_shape"][2]
        self.score_thr = score_thr

    def preprocess(self, frame_bgr):
        img, r, pad = letterbox(frame_bgr, self.size, center=True)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        return x, (r, pad)

    def infer(self, x):
        return self.sess.run(None, {self.input_name: x})[0][0]

    def postprocess(self, out, meta):
        r, (px, py) = meta
        out = out[out[:, 4] > self.score_thr]
        boxes = (out[:, :4] - np.array([px, py, px, py], np.float32)) / r
        return Detections(boxes.astype(np.float32), out[:, 4].astype(np.float32),
                          [COCO80[int(c)] for c in out[:, 5]])

    def __call__(self, frame_bgr):
        x, meta = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), meta)
