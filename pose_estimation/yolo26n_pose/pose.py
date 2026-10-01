"""YOLO26n-pose (Ultralytics official ONNX, end-to-end head) — AGPL-3.0.

Independent of the ultralytics package: centered letterbox with pad 114,
RGB in [0, 1]. ``output0 [1,300,57]`` rows are x1, y1, x2, y2, score, class,
then 17 x (x, y, conf) in letterboxed input pixels. NMS-free.
"""

import cv2
import numpy as np

from tools.mlmc.detection import letterbox, ort_session
from tools.mlmc.pose import Poses


class PoseEstimator:
    def __init__(self, model, provider="cuda", score_thr=0.3):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = model.meta["artifacts"]["onnx"]["input_shape"][2]
        self.score_thr = score_thr

    def preprocess(self, frame_bgr):
        img, r, pad = letterbox(frame_bgr, self.size, center=True)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None], (r, pad)

    def infer(self, x):
        return self.sess.run(None, {self.input_name: x})[0][0]

    def postprocess(self, out, meta):
        r, (px, py) = meta
        out = out[out[:, 4] > self.score_thr]
        off = np.array([px, py], np.float32)
        boxes = (out[:, :4] - np.tile(off, 2)) / r
        kps = out[:, 6:].reshape(-1, 17, 3)
        xy = (kps[..., :2] - off) / r
        return Poses(xy.astype(np.float32), kps[..., 2].astype(np.float32),
                     boxes.astype(np.float32), out[:, 4].astype(np.float32))

    def __call__(self, frame_bgr):
        x, meta = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), meta)
