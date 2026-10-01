"""Robust Video Matting (MobileNetV3), official ONNX — recurrent video runner.

As upstream ``documentation/inference.md``: RGB in [0, 1] at the source
resolution, recurrent states ``r1i``-``r4i`` start as zeros and every call
feeds back ``r1o``-``r4o``; ``downsample_ratio`` follows upstream's automatic
rule ``min(512 / max(h, w), 1)``. Only ``pha`` (alpha) is used here.
"""

import numpy as np

from tools.mlmc.detection import ort_session
from tools.mlmc.matting import AlphaMatte


class Remover:
    def __init__(self, model, provider="cuda"):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.reset()

    def reset(self):
        self.rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4

    def preprocess(self, frame_bgr):
        x = frame_bgr[..., ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        h, w = frame_bgr.shape[:2]
        return (np.ascontiguousarray(x), np.array([min(512 / max(h, w), 1.0)], np.float32)), None

    def infer(self, x):
        src, ratio = x
        outs = self.sess.run(None, {"src": src, "r1i": self.rec[0], "r2i": self.rec[1],
                                    "r3i": self.rec[2], "r4i": self.rec[3], "downsample_ratio": ratio})
        self.rec = outs[2:]
        return outs[1]

    def postprocess(self, pha, meta=None):
        return AlphaMatte(np.clip(pha[0, 0], 0, 1).astype(np.float32))

    def __call__(self, frame_bgr):
        x, _ = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x))
