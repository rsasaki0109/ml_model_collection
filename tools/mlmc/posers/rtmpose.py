"""RTMPose (mmpose, SimCC) top-down pose estimation, official ONNX SDK files.

Person boxes come from a detector in this collection (``person_detector`` in
model.yaml, default D-FINE-N, class "person"), as in mmpose's top-down
evaluation with detector boxes.

Per box (mmpose ``GetBBoxCenterScale`` padding 1.25 + ``TopdownAffine``,
same as rtmlib): center / scale from the box, scale expanded to the model's
192:256 aspect ratio, affine warp (no rotation) to 192x256, **RGB**
(``bgr_to_rgb=True`` in the mmpose config), mean (123.675, 116.28, 103.53),
std (58.395, 57.12, 57.375).

Decode (``SimCCLabel``, ``simcc_split_ratio=2.0``): x = argmax(simcc_x) / 2,
y = argmax(simcc_y) / 2, score = mean of the two maxima (as rtmlib), then
mapped back through the affine transform.
"""

from __future__ import annotations

import cv2
import numpy as np

from tools.mlmc.detection import ort_session
from tools.mlmc.pose import Poses

MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


class PoseEstimator:
    def __init__(self, model, provider="cuda", score_thr=0.3, max_people=20, split_ratio=2.0):
        from tools.mlmc.catalog import get_model

        self.sess = ort_session(model.artifact_path("onnx"), provider)
        _, _, self.h, self.w = model.meta["artifacts"]["onnx"]["input_shape"]
        self.detector = get_model(model.meta.get("person_detector", "dfine_n")).load_runner(
            provider=provider, score_thr=score_thr)
        self.max_people, self.split = max_people, split_ratio

    def _crop(self, img, box):
        x1, y1, x2, y2 = box
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
        sw, sh = (x2 - x1) * 1.25, (y2 - y1) * 1.25
        ar = self.w / self.h
        sw, sh = (sw, sw / ar) if sw > sh * ar else (sh * ar, sh)
        s = self.w / sw  # isotropic scale after aspect correction
        mat = np.array([[s, 0, self.w / 2 - s * center[0]],
                        [0, s, self.h / 2 - s * center[1]]], np.float32)
        crop = cv2.warpAffine(img, mat, (self.w, self.h), flags=cv2.INTER_LINEAR)
        return crop, center, np.array([sw, sh], np.float32)

    def preprocess(self, frame_bgr):
        det = self.detector(frame_bgr)
        keep = [i for i in np.argsort(-det.scores) if det.labels[i] == "person"][: self.max_people]
        boxes, scores = det.boxes[keep], det.scores[keep]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        crops, centers, scales = [], [], []
        for b in boxes:
            c, ctr, sc = self._crop(rgb, b)
            crops.append(((c.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1))
            centers.append(ctr)
            scales.append(sc)
        x = np.stack(crops).astype(np.float32) if crops else np.zeros((0, 3, self.h, self.w), np.float32)
        return x, (boxes, scores, np.array(centers).reshape(-1, 2), np.array(scales).reshape(-1, 2))

    def infer(self, x):
        if not len(x):
            return None
        return self.sess.run(None, {"input": x})

    def postprocess(self, out, meta):
        boxes, scores, centers, scales = meta
        if out is None:
            return Poses(np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32),
                         np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
        sx, sy = out
        kp = np.stack([sx.argmax(-1), sy.argmax(-1)], -1).astype(np.float32) / self.split
        kscore = 0.5 * (sx.max(-1) + sy.max(-1))
        kp = kp / np.array([self.w, self.h], np.float32) * scales[:, None] \
            + centers[:, None] - scales[:, None] / 2
        return Poses(kp.astype(np.float32), kscore.astype(np.float32),
                     boxes.astype(np.float32), scores.astype(np.float32))

    def __call__(self, frame_bgr):
        x, meta = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), meta)
