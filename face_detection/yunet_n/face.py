"""YuNet-n (libfacedetection.train), dynamic-input ONNX.

Decoding follows OpenCV ``FaceDetectorYN`` (modules/objdetect/src/face_detect.cpp):
raw **BGR** float 0-255 at the source resolution, zero-padded at the bottom /
right to a multiple of 32; per stride s in (8, 16, 32) one prediction per
cell (row-major), score = sqrt(cls * obj), box centre (c + b0, r + b1) * s,
size exp(b2, b3) * s, landmarks (k + c, k + r) * s. NMS IoU 0.3.
"""

import numpy as np

from tools.mlmc.detection import ort_session
from tools.mlmc.face import Faces, select

STRIDES = (8, 16, 32)


class FaceDetector:
    def __init__(self, model, provider="cuda", score_thr=0.6, nms_iou=0.3):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.names = [o.name for o in self.sess.get_outputs()]
        self.score_thr, self.nms_iou = score_thr, nms_iou

    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        H, W = -(-h // 32) * 32, -(-w // 32) * 32
        x = np.zeros((1, 3, H, W), np.float32)
        x[0, :, :h, :w] = frame_bgr.transpose(2, 0, 1)
        return x, (H, W)

    def infer(self, x):
        return dict(zip(self.names, self.sess.run(None, {"input": x})))

    def postprocess(self, out, size):
        H, W = size
        boxes, scores, lms = [], [], []
        for s in STRIDES:
            rows, cols = H // s, W // s
            cls = np.clip(out[f"cls_{s}"][0, :, 0], 0, 1)
            obj = np.clip(out[f"obj_{s}"][0, :, 0], 0, 1)
            b, k = out[f"bbox_{s}"][0], out[f"kps_{s}"][0]
            r, c = np.divmod(np.arange(rows * cols), cols)
            cx, cy = (c + b[:, 0]) * s, (r + b[:, 1]) * s
            bw, bh = np.exp(b[:, 2]) * s, np.exp(b[:, 3]) * s
            boxes.append(np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1))
            scores.append(np.sqrt(cls * obj))
            lms.append(np.stack([(k[:, 0::2] + c[:, None]) * s, (k[:, 1::2] + r[:, None]) * s], -1))
        return select(np.concatenate(boxes), np.concatenate(scores), np.concatenate(lms),
                      self.score_thr, self.nms_iou)

    def __call__(self, frame_bgr):
        x, size = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), size)
