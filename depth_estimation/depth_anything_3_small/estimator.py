"""Depth Anything 3 Small (monocular) — pre/post-processing.

Mirrors upstream ``InputProcessor`` with ``process_res=504,
upper_bound_resize``: resize the longest side to 504 (INTER_AREA when
shrinking), then round each side to the nearest multiple of 14 with a second
small resize, ImageNet mean/std. The graph is fixed at 280x504 (16:9 input);
other aspect ratios are stretched to it. Output is relative *depth*
(larger = farther), resized back to the source size.
"""

import cv2
import numpy as np

from tools.mlmc.depth import DEPTH, DepthMap
from tools.mlmc.detection import ort_session

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Estimator:
    def __init__(self, model, provider="cuda", process_res=504):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        *_, self.th, self.tw = model.meta["artifacts"]["onnx"]["input_shape"]
        self.process_res = process_res

    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        s = self.process_res / max(h, w)
        mid = (max(1, int(round(w * s))), max(1, int(round(h * s))))
        rgb = cv2.resize(rgb, mid, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        if mid != (self.tw, self.th):
            up = self.tw > mid[0] or self.th > mid[1]
            rgb = cv2.resize(rgb, (self.tw, self.th),
                             interpolation=cv2.INTER_CUBIC if up else cv2.INTER_AREA)
        x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None, None]
        return np.ascontiguousarray(x, dtype=np.float32), (w, h)

    def infer(self, x):
        return self.sess.run(["depth"], {"x": x})[0]

    def postprocess(self, out, wh):
        w, h = wh
        d = cv2.resize(out[0, 0], (w, h), interpolation=cv2.INTER_LINEAR)
        return DepthMap(d.astype(np.float32), DEPTH)

    def __call__(self, frame_bgr):
        x, wh = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), wh)
