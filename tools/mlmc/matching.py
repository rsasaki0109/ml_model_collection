"""Local feature matching: shared pair runner, rendering and homography tools.

Every feature-matching runner returns ``Matches``: matched keypoints in the
previous frame ``kpts0 (M, 2)`` and the current frame ``kpts1 (M, 2)`` (source
pixels) with match scores. Runners are stateful video runners (each call
matches the frame against the previous one; the first frame is matched with
itself).

The graphs are fabio-sim/LightGlue-ONNX end-to-end pipelines (extractor +
LightGlue, top-k keypoints, matching inside the graph):
``images [2B, C, H, W]`` (pairs interleaved) -> ``keypoints [2B, K, 2]``
(input pixels), ``matches [M, 3]`` (batch, idx0, idx1), ``mscores [M]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tools.mlmc.detection import ort_session


@dataclass
class Matches:
    kpts0: np.ndarray   # (M, 2) float32, previous frame
    kpts1: np.ndarray   # (M, 2) float32, current frame
    scores: np.ndarray  # (M,) float32


def to_tensor(img_bgr: np.ndarray, channels: str) -> np.ndarray:
    """``rgb``: RGB /255 (ALIKED / RaCo, DISK); ``gray``: 0.299 R + 0.587 G + 0.114 B, /255 (SuperPoint)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if channels == "gray":
        return (rgb @ np.array([0.299, 0.587, 0.114], np.float32))[None]
    return rgb.transpose(2, 0, 1)


class PairMatcher:
    def __init__(self, model, provider="cuda"):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        cfg = model.meta["matching"]
        self.channels = cfg["channels"]
        self.h, self.w = model.meta["artifacts"]["onnx"]["input_shape"][-2:]
        self.prev = None

    def preprocess(self, frame_bgr):
        cur = to_tensor(cv2.resize(frame_bgr, (self.w, self.h), interpolation=cv2.INTER_AREA), self.channels)
        prev = self.prev if self.prev is not None else cur
        self.prev = cur
        sx, sy = frame_bgr.shape[1] / self.w, frame_bgr.shape[0] / self.h
        return np.ascontiguousarray(np.stack([prev, cur])), np.array([sx, sy], np.float32)

    def infer(self, x):
        return self.sess.run(None, {"images": x})

    def match_pair(self, x):
        """Run one pre-built pair tensor; return (kpts0, kpts1, scores) in tensor pixels."""
        kpts, matches, scores = self.infer(x)
        m = matches[matches[:, 0] == 0]
        s = scores[matches[:, 0] == 0]
        return kpts[0][m[:, 1]].astype(np.float32), kpts[1][m[:, 2]].astype(np.float32), s.astype(np.float32)

    def postprocess(self, out, scale):
        kpts, matches, scores = out
        sel = matches[:, 0] == 0
        m, s = matches[sel], scores[sel]
        return Matches(kpts[0][m[:, 1]].astype(np.float32) * scale,
                       kpts[1][m[:, 2]].astype(np.float32) * scale, s.astype(np.float32))

    def __call__(self, frame_bgr):
        x, scale = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), scale)


def render(frame: np.ndarray, m: Matches, max_lines: int = 600) -> np.ndarray:
    """Current frame with each match drawn as a motion track (previous -> current),
    colour by match score (red low, green high)."""
    out = (frame * 0.7).astype(np.uint8)
    t = max(1, round(frame.shape[1] / 640))
    order = np.argsort(-m.scores)[:max_lines]
    for i in order:
        s = float(np.clip(m.scores[i], 0, 1))
        c = (0, int(255 * s), int(255 * (1 - s)))
        p0, p1 = m.kpts0[i].round().astype(int), m.kpts1[i].round().astype(int)
        cv2.line(out, tuple(p0), tuple(p1), c, t, cv2.LINE_AA)
        cv2.circle(out, tuple(p1), t + 1, c, -1, cv2.LINE_AA)
    cv2.putText(out, f"{len(m.scores)} matches", (10, 40 * t), cv2.FONT_HERSHEY_SIMPLEX,
                1.2 * t, (255, 255, 255), 2 * t, cv2.LINE_AA)
    return out


def homography_dlt(p0: np.ndarray, p1: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted DLT with Hartley normalisation (all matches, no RANSAC)."""
    if len(p0) < 4:
        raise ValueError("need at least 4 matches")
    w = np.ones(len(p0)) if w is None else np.asarray(w, np.float64)

    def norm(p):
        c = p.mean(0)
        s = np.sqrt(2) / max(np.sqrt(((p - c) ** 2).sum(1)).mean(), 1e-12)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    T0, T1 = norm(p0), norm(p1)
    a = (np.c_[p0, np.ones(len(p0))] @ T0.T)[:, :2]
    b = (np.c_[p1, np.ones(len(p1))] @ T1.T)[:, :2]
    rows = []
    for (x, y), (u, v), wi in zip(a, b, w):
        rows.append(wi * np.array([-x, -y, -1, 0, 0, 0, u * x, u * y, u]))
        rows.append(wi * np.array([0, 0, 0, -x, -y, -1, v * x, v * y, v]))
    _, _, vt = np.linalg.svd(np.array(rows))
    H = np.linalg.inv(T1) @ vt[-1].reshape(3, 3) @ T0
    return H / H[2, 2]


def corner_error(H_est: np.ndarray, H_gt: np.ndarray, w: int, h: int) -> float:
    """Mean distance of the four image corners mapped by the two homographies."""
    c = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]], np.float64).T
    a, b = H_est @ c, H_gt @ c
    return float(np.linalg.norm(a[:2] / a[2] - b[:2] / b[2], axis=0).mean())


def error_auc(errors, thresholds) -> list[float]:
    """Area under the cumulative error curve up to each threshold (glue-factory ``cal_error_auc``)."""
    errors = np.sort(np.asarray(errors, np.float64))
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors, recall = np.r_[0.0, errors], np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last = np.searchsorted(errors, t)
        r = np.r_[recall[:last], recall[last - 1]]
        e = np.r_[errors[:last], t]
        aucs.append(float(np.trapezoid(r, x=e) / t))
    return aucs
