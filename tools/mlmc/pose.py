"""Common pieces for human pose estimation models.

``pose_estimation/<model>/pose.py`` exposes a ``PoseEstimator`` whose
``__call__(frame_bgr)`` returns :class:`Poses` — one entry per person, COCO-17
keypoints in source-image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

COCO17 = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle")

SKELETON = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
            (1, 3), (2, 4), (3, 5), (4, 6))

# limb colour by body side (BGR): left = orange, right = blue, centre = green
_LEFT, _RIGHT, _MID = (0, 140, 255), (255, 120, 0), (0, 200, 0)


def _limb_color(a: int, b: int):
    left = {1, 3, 5, 7, 9, 11, 13, 15}
    right = {2, 4, 6, 8, 10, 12, 14, 16}
    if a in left and b in left:
        return _LEFT
    if a in right and b in right:
        return _RIGHT
    return _MID


@dataclass
class Poses:
    keypoints: np.ndarray   # (N, 17, 2) x, y in source pixels
    scores: np.ndarray      # (N, 17) per-keypoint confidence
    boxes: np.ndarray       # (N, 4) person boxes xyxy (from the detector or the model)
    box_scores: np.ndarray  # (N,) person confidence

    def __len__(self):
        return len(self.keypoints)


def render(frame_bgr: np.ndarray, poses: Poses, kpt_thr: float = 0.3) -> np.ndarray:
    out = frame_bgr.copy()
    t = max(1, round(max(out.shape[:2]) / 640))
    for kps, sc in zip(poses.keypoints, poses.scores):
        for a, b in SKELETON:
            if sc[a] > kpt_thr and sc[b] > kpt_thr:
                cv2.line(out, tuple(int(v) for v in kps[a]), tuple(int(v) for v in kps[b]),
                         _limb_color(a, b), 2 * t, cv2.LINE_AA)
        for (x, y), s in zip(kps, sc):
            if s > kpt_thr:
                cv2.circle(out, (int(x), int(y)), 2 * t, (255, 255, 255), -1, cv2.LINE_AA)
    return out
