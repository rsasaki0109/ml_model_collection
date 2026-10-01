"""Face detection: shared result type, NMS and rendering.

Every face-detection runner returns ``Faces``: boxes ``(N, 4)`` x1 y1 x2 y2,
scores ``(N,)`` and five landmarks ``(N, 5, 2)`` (right eye, left eye, nose
tip, right / left mouth corner — the subject's right, i.e. image left), all
in source pixels. Runners take ``score_thr`` (rendering default) and
``nms_iou``; evaluation lowers ``score_thr``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Faces:
    boxes: np.ndarray      # (N, 4) float32
    scores: np.ndarray     # (N,) float32
    landmarks: np.ndarray  # (N, 5, 2) float32

    @staticmethod
    def empty():
        return Faces(np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                     np.zeros((0, 5, 2), np.float32))


def nms(boxes: np.ndarray, scores: np.ndarray, score_thr: float, iou: float,
        top_k: int = 5000, keep_k: int = 750) -> np.ndarray:
    """Indices kept by greedy NMS (cv2.dnn.NMSBoxes on x, y, w, h)."""
    m = np.flatnonzero(scores > score_thr)
    if not len(m):
        return m
    m = m[np.argsort(-scores[m])][:top_k]
    xywh = np.concatenate([boxes[m, :2], boxes[m, 2:] - boxes[m, :2]], 1)
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores[m].tolist(), score_thr, iou)
    return m[np.array(keep, dtype=int).reshape(-1)][:keep_k]


def select(boxes, scores, landmarks, score_thr, iou, **kw) -> Faces:
    k = nms(boxes, scores, score_thr, iou, **kw)
    return Faces(boxes[k].astype(np.float32), scores[k].astype(np.float32),
                 landmarks[k].reshape(-1, 5, 2).astype(np.float32))


LANDMARK_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)]


def render(frame: np.ndarray, faces: Faces, thumbs: int = 6) -> np.ndarray:
    """Boxes and landmarks on the frame, plus a bottom strip with the
    ``thumbs`` highest-scoring faces enlarged (faces in street scenes are
    often only ~10 px tall)."""
    out = frame.copy()
    H, W = frame.shape[:2]
    t = max(1, round(W / 400))
    for b in faces.boxes:
        x1, y1, x2, y2 = b.round().astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), t)
    tw = W // thumbs
    strip = np.full((tw, W, 3), 32, np.uint8)
    for k, i in enumerate(np.argsort(-faces.scores)[:thumbs]):
        x1, y1, x2, y2 = faces.boxes[i]
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.6
        x0, y0 = cx - side / 2, cy - side / 2
        m = np.array([[tw / side, 0, -x0 * tw / side], [0, tw / side, -y0 * tw / side]], np.float32)
        thumb = cv2.warpAffine(frame, m, (tw, tw), flags=cv2.INTER_LINEAR)
        for p, c in zip(faces.landmarks[i], LANDMARK_COLORS):
            q = ((p - [x0, y0]) * tw / side).round().astype(int)
            cv2.circle(thumb, tuple(int(v) for v in q), max(2, tw // 40), c, -1)
        cv2.putText(thumb, f"{faces.scores[i]:.2f}", (4, tw - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    tw / 160, (0, 255, 0), max(1, tw // 80), cv2.LINE_AA)
        strip[:, k * tw:(k + 1) * tw] = thumb
    out[H - tw:] = strip
    return out
