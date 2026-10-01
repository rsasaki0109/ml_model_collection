"""Whole-body pose (COCO-WholeBody, 133 keypoints): layout and rendering.

Runners return ``tools.mlmc.pose.Poses`` with 133 keypoints per person:
body 0-16 (COCO-17), feet 17-22, face 23-90, left hand 91-111, right hand
112-132 — the layout of mmpose ``configs/_base_/datasets/coco_wholebody.py``
(open-mmlab/mmpose 759b39c13fea6ba094afc1fa932f51dc1b11cbf9, Apache-2.0),
whose 65 skeleton links are reproduced below.
"""

from __future__ import annotations

import cv2
import numpy as np

from tools.mlmc.pose import Poses

PARTS = {"body": (0, 17), "foot": (17, 23), "face": (23, 91), "lefthand": (91, 112), "righthand": (112, 133)}

SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
            (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17), (15, 18), (15, 19),
            (16, 20), (16, 21), (16, 22), (91, 92), (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
            (98, 99), (91, 100), (100, 101), (101, 102), (102, 103), (91, 104), (104, 105), (105, 106),
            (106, 107), (91, 108), (108, 109), (109, 110), (110, 111), (112, 113), (113, 114), (114, 115),
            (115, 116), (112, 117), (117, 118), (118, 119), (119, 120), (112, 121), (121, 122), (122, 123),
            (123, 124), (112, 125), (125, 126), (126, 127), (127, 128), (112, 129), (129, 130), (130, 131),
            (131, 132)]

COLORS = {"body": (0, 200, 255), "foot": (255, 128, 0), "face": (255, 255, 255),
          "lefthand": (0, 255, 0), "righthand": (255, 0, 255)}


def _part(i: int) -> str:
    return next(p for p, (a, b) in PARTS.items() if a <= i < b)


def render(frame: np.ndarray, poses: Poses, kpt_thr: float = 0.3) -> np.ndarray:
    out = frame.copy()
    t = max(1, round(frame.shape[1] / 640))
    for kp, sc in zip(poses.keypoints, poses.scores):
        for a, b in SKELETON:
            if sc[a] > kpt_thr and sc[b] > kpt_thr:
                cv2.line(out, tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b]),
                         COLORS[_part(b)], t, cv2.LINE_AA)
        for i in np.flatnonzero(sc > kpt_thr):
            r = 1 if 23 <= i < 133 else t + 1  # face / hand points small
            cv2.circle(out, tuple(int(v) for v in kp[i]), r, COLORS[_part(i)], -1, cv2.LINE_AA)
    return out
