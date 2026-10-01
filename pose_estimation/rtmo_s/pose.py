"""RTMO-s (mmpose, one-stage) — official ONNX SDK file.

Pre-processing (bundled pipeline.json: ``to_rgb: false``, mean 0, std 1):
**BGR**, 0-255 float, resize keeping aspect ratio and pad with 114 to
640x640 (top-left, as rtmlib; the inverse mapping matches). The graph
includes NMS and returns ``dets [1,N,5]`` (x1,y1,x2,y2,score) and
``keypoints [1,N,17,3]`` (x, y, score) in input pixels.
"""

import cv2
import numpy as np

from tools.mlmc.detection import ort_session
from tools.mlmc.pose import Poses


class PoseEstimator:
    def __init__(self, model, provider="cuda", score_thr=0.3):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.size = model.meta["artifacts"]["onnx"]["input_shape"][2]
        self.score_thr = score_thr

    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = int(h * r), int(w * r)
        img = np.full((self.size, self.size, 3), 114, np.uint8)
        img[:nh, :nw] = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return img.transpose(2, 0, 1)[None].astype(np.float32), r

    def infer(self, x):
        return self.sess.run(None, {"input": x})

    def postprocess(self, out, r):
        dets, kps = out[0][0], out[1][0]
        m = dets[:, 4] > self.score_thr
        dets, kps = dets[m], kps[m]
        return Poses((kps[..., :2] / r).astype(np.float32), kps[..., 2].astype(np.float32),
                     (dets[:, :4] / r).astype(np.float32), dets[:, 4].astype(np.float32))

    def __call__(self, frame_bgr):
        x, r = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), r)
