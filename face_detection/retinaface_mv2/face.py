"""RetinaFace MobileNetV2 (yakhyo/retinaface-pytorch), release ONNX.

As upstream ``onnx_inference.py`` / ``layers/functions/prior_box.py``:
**BGR** float minus (104, 117, 123), source resolution (no resize). Priors
per stride (8, 16, 32) with min sizes ((16, 32), (64, 128), (256, 512)),
centre ((j + 0.5) * step, (i + 0.5) * step) for each row i, column j, min
size; variances (0.1, 0.2). ``conf`` is already softmaxed (column 1 = face).
NMS IoU 0.4, at most 750 faces.
"""

import numpy as np

from tools.mlmc.detection import ort_session
from tools.mlmc.face import Faces, select

MEAN = np.array([104, 117, 123], np.float32)
STEPS = (8, 16, 32)
MIN_SIZES = ((16, 32), (64, 128), (256, 512))


def priors(h, w):
    out = []
    for step, sizes in zip(STEPS, MIN_SIZES):
        fh, fw = -(-h // step), -(-w // step)
        i, j = np.meshgrid(np.arange(fh), np.arange(fw), indexing="ij")
        cx = ((j + 0.5) * step / w).reshape(-1, 1).repeat(len(sizes), 1)
        cy = ((i + 0.5) * step / h).reshape(-1, 1).repeat(len(sizes), 1)
        sw = np.tile(np.array(sizes, np.float32) / w, (fh * fw, 1))
        sh = np.tile(np.array(sizes, np.float32) / h, (fh * fw, 1))
        out.append(np.stack([cx, cy, sw, sh], -1).reshape(-1, 4))
    return np.concatenate(out).astype(np.float32)


class FaceDetector:
    def __init__(self, model, provider="cuda", score_thr=0.6, nms_iou=0.4):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.score_thr, self.nms_iou = score_thr, nms_iou
        self._priors = {}

    def preprocess(self, frame_bgr):
        x = (frame_bgr.astype(np.float32) - MEAN).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(x), frame_bgr.shape[:2]

    def infer(self, x):
        return self.sess.run(None, {"input": x})

    def postprocess(self, out, size):
        h, w = size
        if size not in self._priors:
            self._priors[size] = priors(h, w)
        p = self._priors[size]
        loc, conf, lm = out[0][0], out[1][0], out[2][0]
        cxcy = p[:, :2] + loc[:, :2] * 0.1 * p[:, 2:]
        wh = p[:, 2:] * np.exp(loc[:, 2:] * 0.2)
        scale = np.array([w, h], np.float32)
        boxes = np.concatenate([cxcy - wh / 2, cxcy + wh / 2], 1) * np.tile(scale, 2)
        lms = (p[:, None, :2] + lm.reshape(-1, 5, 2) * 0.1 * p[:, None, 2:]) * scale
        return select(boxes, conf[:, 1], lms, self.score_thr, self.nms_iou)

    def __call__(self, frame_bgr):
        x, size = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), size)
