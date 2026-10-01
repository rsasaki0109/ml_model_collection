"""Point tracking (tracking any point, TAP): online causal runner, rendering, TAP-Vid metrics.

Every point-tracking runner returns ``PointTracks``: the position of each
tracked point ``(N, 2)`` (x, y, source pixels) in the current frame and
whether it is visible. Runners are stateful video runners: on the first frame
(or after ``reset``) they start tracking a regular grid of points.

The graphs follow the two-part export of ibaiGorordo/Tapir-Pytorch-Inference
(Apache-2.0) of DeepMind's causal (online) TAPIR / BootsTAPIR:

* ``point_encoder``: query_points [1,N,2] (y, x normalised to [0, 1]),
  feature_grid [1,R/8,R/8,256], hires_feats_grid [1,R/4,R/4,128] ->
  query_feats [1,N,256], hires_query_feats [1,N,128];
* ``tapir`` (per frame): input_frame [1,3,R,R] RGB in [-1, 1], the query
  features and causal_state [iters,12,N,2,2560] -> tracks [1,N,2] (x, y in
  R x R pixels), visibles [1,N] (bool), new_causal_state and this frame's
  feature grids.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tools.mlmc.detection import ort_session


@dataclass
class PointTracks:
    points: np.ndarray   # (N, 2) float32, x y in source pixels
    visible: np.ndarray  # (N,) bool


class OnlineTapir:
    def __init__(self, model, provider="cuda", grid=16):
        cfg = model.meta["point_tracking"]
        self.sess = ort_session(model.artifact_path("onnx"), provider)  # per-frame graph
        self.encoder = ort_session(model.weights_dir / cfg["encoder_file"], provider)
        self.res = cfg["resolution"]
        self.iters = cfg["pips_iters"]
        self.grid = grid
        self.reset()

    def reset(self):
        self.qf = self.hqf = self.state = None

    # -- graph helpers -------------------------------------------------------
    def tensor(self, frame_bgr):
        x = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (self.res, self.res))
        return np.ascontiguousarray((x.astype(np.float32) / 255 * 2 - 1).transpose(2, 0, 1)[None])

    def zero_state(self, n):
        return np.zeros((self.iters, 12, n, 2, 2560), np.float32)

    def step(self, x, qf, hqf, state):
        return self.sess.run(None, {"input_frame": x, "query_feats": qf,
                                    "hires_query_feats": hqf, "causal_state": state})

    def encode(self, query_yx_norm, x):
        """Query features of points (y, x in [0, 1]) sampled from frame tensor ``x``."""
        n = len(query_yx_norm)
        _, _, _, fg, hg = self.step(x, np.zeros((1, n, 256), np.float32),
                                    np.zeros((1, n, 128), np.float32), self.zero_state(n))
        return self.encoder.run(None, {"query_points": query_yx_norm[None].astype(np.float32),
                                       "feature_grid": fg, "hires_feats_grid": hg})

    # -- runner interface ----------------------------------------------------
    def preprocess(self, frame_bgr):
        x = self.tensor(frame_bgr)
        if self.qf is None:  # start: a regular grid over the frame
            g = (np.arange(self.grid) + 0.5) / self.grid
            yy, xx = np.meshgrid(g, g, indexing="ij")
            self.qf, self.hqf = self.encode(np.stack([yy.ravel(), xx.ravel()], 1), x)
            self.state = self.zero_state(self.grid * self.grid)
        return x, frame_bgr.shape[:2]

    def infer(self, x):
        tracks, vis, self.state, _, _ = self.step(x, self.qf, self.hqf, self.state)
        return tracks, vis

    def postprocess(self, out, size):
        tracks, vis = out
        h, w = size
        pts = tracks[0] * np.array([w / self.res, h / self.res], np.float32)
        return PointTracks(pts.astype(np.float32), vis[0].astype(bool))

    def __call__(self, frame_bgr):
        x, size = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), size)


class TrackTrails:
    """Points with trails of their last positions (stateful, one per video)."""

    def __init__(self, length=12):
        self.length, self.hist = length, []

    def __call__(self, frame, tr: PointTracks):
        out = (frame * 0.75).astype(np.uint8)
        self.hist.append((tr.points.copy(), tr.visible.copy()))
        del self.hist[:-self.length]
        n = len(tr.points)
        hue = (np.arange(n) * 180 // max(n, 1)).astype(np.uint8)
        colors = cv2.cvtColor(np.stack([hue, np.full(n, 255, np.uint8), np.full(n, 255, np.uint8)], 1)[None],
                              cv2.COLOR_HSV2BGR)[0]
        t = max(1, round(frame.shape[1] / 400))
        for (p0, v0), (p1, v1) in zip(self.hist[:-1], self.hist[1:]):
            for i in np.flatnonzero(v0 & v1):
                cv2.line(out, tuple(int(c) for c in p0[i]), tuple(int(c) for c in p1[i]),
                         tuple(int(c) for c in colors[i]), t, cv2.LINE_AA)
        for i in np.flatnonzero(tr.visible):
            cv2.circle(out, tuple(int(c) for c in tr.points[i]), t + 2, tuple(int(c) for c in colors[i]), -1, cv2.LINE_AA)
        return out


def tapvid_metrics(query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, query_mode="first"):
    """TAP-Vid metrics for one video, after tapnet ``compute_tapvid_metrics``
    (Apache-2.0): occlusion accuracy, fraction of visible points within
    1/2/4/8/16 px (delta_avg) and average Jaccard, over the frames after each
    query ('first') or all frames but the query frame ('strided').

    query_points [N,3] (t, y, x); gt / pred tracks [N,T,2] (x, y); occluded [N,T] bool.
    """
    T = gt_tracks.shape[1]
    eye = np.eye(T, dtype=np.int32)
    q2e = (np.cumsum(eye, axis=1) - eye) if query_mode == "first" else (1 - eye)
    ev = q2e[np.round(query_points[:, 0]).astype(int)] > 0
    visible, pred_visible = ~gt_occluded, ~pred_occluded
    out = {"occlusion_accuracy": np.sum((pred_occluded == gt_occluded) & ev) / np.sum(ev)}
    fracs, jaccs = [], []
    for thr in (1, 2, 4, 8, 16):
        within = np.sum((pred_tracks - gt_tracks) ** 2, axis=-1) < thr ** 2
        correct = within & visible
        fracs.append(np.sum(correct & ev) / np.sum(visible & ev))
        tp = np.sum(correct & pred_visible & ev)
        fp = np.sum(((~visible) & pred_visible | (~within) & pred_visible) & ev)
        jaccs.append(tp / (np.sum(visible & ev) + fp))
    out["delta_avg"] = float(np.mean(fracs))
    out["average_jaccard"] = float(np.mean(jaccs))
    return out
