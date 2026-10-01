"""Multi-object tracking by detection: ByteTrack, BoT-SORT (motion only) and OC-SORT.

Every tracking runner returns ``Tracks``: boxes ``(N, 4)`` x1 y1 x2 y2 in
source pixels, integer track ids, scores and class names, for the tracks
reported at the current frame.

The trackers are written from scratch for this repository from the papers
and the published default parameters (no code copied):

* ByteTrack — Zhang et al., ECCV 2022, arXiv 2110.06864 (reference
  implementation ifzhang/ByteTrack, MIT).
* BoT-SORT — Aharon et al., arXiv 2206.14651 (NirAharon/BoT-SORT, MIT);
  here without the ReID branch: ByteTrack association with an x-y-w-h
  Kalman filter and camera-motion compensation (sparse optical flow).
* OC-SORT — Cao et al., CVPR 2023, arXiv 2203.14360 (noahcao/OC_SORT, MIT):
  observation-centric re-update, momentum and recovery.

The constant-velocity Kalman filters use the noise model of the SORT /
DeepSORT papers (position / velocity standard deviations proportional to
the box size).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Tracks:
    boxes: np.ndarray   # (N, 4) float32
    ids: np.ndarray     # (N,) int64
    scores: np.ndarray  # (N,) float32
    labels: list


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def assign(cost: np.ndarray, thresh: float):
    """Minimum-cost matching keeping pairs with cost <= thresh."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] <= thresh
    matches = list(zip(r[ok], c[ok]))
    mr, mc = {i for i, _ in matches}, {j for _, j in matches}
    return (matches, [i for i in range(cost.shape[0]) if i not in mr],
            [j for j in range(cost.shape[1]) if j not in mc])


# --------------------------------------------------------------------------
# Kalman filters
# --------------------------------------------------------------------------
class CVKalman:
    """Constant-velocity Kalman filter on a 4-d box measurement.

    ``mode="xyah"`` (centre, aspect ratio, height; ByteTrack) or
    ``"xywh"`` (centre, width, height; BoT-SORT). Noise std = 1/20 of the
    box size for position, 1/160 for velocity.
    """

    W_POS, W_VEL = 1 / 20, 1 / 160

    def __init__(self, mode: str):
        self.mode = mode
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def _scale(self, z):
        """Per-dimension box size the noise scales with (aspect ratio: fixed, set by callers)."""
        if self.mode == "xyah":
            return np.array([z[3], z[3], z[3], z[3]])
        return np.array([z[2], z[3], z[2], z[3]])

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        s = self._scale(z)
        pos, vel = 2 * self.W_POS * s, 10 * self.W_VEL * s
        if self.mode == "xyah":
            pos[2], vel[2] = 1e-2, 1e-5
        return mean, np.diag(np.r_[pos, vel] ** 2)

    def predict(self, mean, cov):
        s = self._scale(mean[:4])
        pos, vel = self.W_POS * s, self.W_VEL * s
        if self.mode == "xyah":
            pos[2], vel[2] = 1e-2, 1e-5
        Q = np.diag(np.r_[pos, vel] ** 2)
        return self.F @ mean, self.F @ cov @ self.F.T + Q

    def update(self, mean, cov, z):
        s = self._scale(mean[:4])
        r = self.W_POS * s
        if self.mode == "xyah":
            r[2] = 1e-1
        S = self.H @ cov @ self.H.T + np.diag(r ** 2)
        K = np.linalg.solve(S, self.H @ cov).T
        return mean + K @ (z - self.H @ mean), cov - K @ S @ K.T


def to_meas(box, mode):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if mode == "xyah":
        return np.array([x1 + w / 2, y1 + h / 2, w / max(h, 1e-6), h])
    return np.array([x1 + w / 2, y1 + h / 2, w, h])


def from_meas(z, mode):
    cx, cy = z[0], z[1]
    w, h = (z[2] * z[3], z[3]) if mode == "xyah" else (z[2], z[3])
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# --------------------------------------------------------------------------
# ByteTrack / BoT-SORT
# --------------------------------------------------------------------------
class _Track:
    _next_id = 1

    def __init__(self, box, score, label, kf, frame, activated):
        self.kf = kf
        self.mean, self.cov = kf.initiate(to_meas(box, kf.mode))
        self.score, self.label = score, label
        self.id = None
        self.state = "tracked"
        self.activated = activated
        self.start = self.last = frame
        if activated:
            self._new_id()

    def _new_id(self):
        self.id = _Track._next_id
        _Track._next_id += 1

    @property
    def box(self):
        return from_meas(self.mean[:4], self.kf.mode)

    def predict(self):
        if self.state != "tracked" and self.kf.mode == "xyah":
            self.mean[7] = 0.0  # lost tracks keep their height
        elif self.state != "tracked":
            self.mean[6:] = 0.0
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)

    def update(self, box, score, frame):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, to_meas(box, self.kf.mode))
        self.score, self.last, self.state = score, frame, "tracked"
        if not self.activated:
            self.activated = True
            self._new_id()


class ByteTracker:
    """ByteTrack; with ``mode="xywh"`` and ``gmc=True`` it is BoT-SORT without ReID."""

    def __init__(self, track_thresh=0.5, low_thresh=0.1, new_thresh=None, match_thresh=0.8,
                 second_thresh=0.5, unconfirmed_thresh=0.7, track_buffer=30, fps=30,
                 min_box_area=10, mode="xyah", gmc=False):
        self.track_thresh, self.low_thresh = track_thresh, low_thresh
        self.new_thresh = track_thresh + 0.1 if new_thresh is None else new_thresh
        self.match_thresh, self.second_thresh, self.unconf_thresh = match_thresh, second_thresh, unconfirmed_thresh
        self.max_lost = int(fps / 30.0 * track_buffer)
        self.min_box_area = min_box_area
        self.kf = CVKalman(mode)
        self.gmc = SparseFlowGMC() if gmc else None
        self.reset()

    def reset(self):
        self.tracked, self.lost, self.frame = [], [], 0
        _Track._next_id = 1
        if self.gmc:
            self.gmc.reset()

    @staticmethod
    def _fused_cost(tracks, boxes, scores):
        iou = iou_matrix(np.array([t.box for t in tracks]).reshape(-1, 4), boxes)
        return 1 - iou * scores[None, :]

    def update(self, boxes, scores, labels, frame_bgr=None) -> list:
        self.frame += 1
        f = self.frame
        hi = scores >= self.track_thresh
        lo = (scores > self.low_thresh) & ~hi
        unconfirmed = [t for t in self.tracked if not t.activated]
        active = [t for t in self.tracked if t.activated]
        pool = active + self.lost
        for t in pool + unconfirmed:
            t.predict()
        if self.gmc is not None and frame_bgr is not None:
            A = self.gmc.apply(frame_bgr)
            for t in pool + unconfirmed:
                gmc_warp(t, A)

        # 1st association: high-score detections, IoU fused with score
        hb, hs, hl = boxes[hi], scores[hi], [l for l, k in zip(labels, hi) if k]
        m1, um_trk, um_det = assign(self._fused_cost(pool, hb, hs), self.match_thresh)
        for i, j in m1:
            pool[i].update(hb[j], hs[j], f)
        # 2nd association: remaining tracked (not lost) tracks with low-score detections
        rest = [pool[i] for i in um_trk if pool[i].state == "tracked"]
        lb, ls = boxes[lo], scores[lo]
        cost2 = 1 - iou_matrix(np.array([t.box for t in rest]).reshape(-1, 4), lb)
        m2, um_rest, _ = assign(cost2, self.second_thresh)
        for i, j in m2:
            rest[i].update(lb[j], ls[j], f)
        for i in um_rest:
            rest[i].state = "lost"
        # unconfirmed tracks with the remaining high-score detections
        rb, rs = hb[um_det], hs[um_det]
        rl = [hl[j] for j in um_det]
        m3, um_unc, um_det2 = assign(self._fused_cost(unconfirmed, rb, rs), self.unconf_thresh)
        for i, j in m3:
            unconfirmed[i].update(rb[j], rs[j], f)
        for i in um_unc:
            unconfirmed[i].state = "removed"
        # new tracks
        new = [_Track(rb[j], rs[j], rl[j], self.kf, f, activated=(f == 1))
               for j in um_det2 if rs[j] >= self.new_thresh]

        everyone = pool + unconfirmed + new
        self.tracked = [t for t in everyone if t.state == "tracked"]
        self.lost = [t for t in everyone if t.state == "lost" and f - t.last <= self.max_lost]
        self._dedup()
        out = []
        for t in self.tracked:
            b = t.box
            if t.activated and (b[2] - b[0]) * (b[3] - b[1]) > self.min_box_area:
                out.append(t)
        return out

    def _dedup(self):
        """Drop the shorter of a tracked / lost pair that overlap (IoU > 0.85)."""
        if not self.tracked or not self.lost:
            return
        iou = iou_matrix(np.array([t.box for t in self.tracked]), np.array([t.box for t in self.lost]))
        drop_t, drop_l = set(), set()
        for i, j in zip(*np.nonzero(iou > 0.85)):
            a, b = self.tracked[i], self.lost[j]
            if a.last - a.start > b.last - b.start:
                drop_l.add(j)
            else:
                drop_t.add(i)
        self.tracked = [t for k, t in enumerate(self.tracked) if k not in drop_t]
        self.lost = [t for k, t in enumerate(self.lost) if k not in drop_l]


class SparseFlowGMC:
    """Global motion: affine (rotation, scale, translation) from sparse
    Lucas-Kanade flow on the previous frame's corners, at half resolution."""

    def __init__(self, downscale=2):
        self.downscale = downscale
        self.reset()

    def reset(self):
        self.prev, self.prev_pts = None, None

    def apply(self, frame_bgr) -> np.ndarray:
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        d = self.downscale
        g = cv2.resize(g, (g.shape[1] // d, g.shape[0] // d))
        A = np.eye(2, 3)
        pts = cv2.goodFeaturesToTrack(g, maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3)
        if self.prev is not None and self.prev_pts is not None and len(self.prev_pts) >= 4:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, self.prev_pts, None)
            ok = st.reshape(-1) == 1
            if ok.sum() >= 4:
                M, _ = cv2.estimateAffinePartial2D(self.prev_pts[ok], nxt[ok], method=cv2.RANSAC)
                if M is not None:
                    A = M
                    A[:, 2] *= d
        self.prev, self.prev_pts = g, pts
        return A


def gmc_warp(t: _Track, A: np.ndarray):
    """Apply the camera-motion affine to an x-y-(w-h | a-h) state."""
    R, tr = A[:, :2], A[:, 2]
    if t.kf.mode == "xywh":
        R8 = np.kron(np.eye(4), R)
        t.mean = R8 @ t.mean
        t.mean[:2] += tr
        t.cov = R8 @ t.cov @ R8.T
    else:  # only the centre and its velocity
        t.mean[:2] = R @ t.mean[:2] + tr
        t.mean[4:6] = R @ t.mean[4:6]


# --------------------------------------------------------------------------
# OC-SORT
# --------------------------------------------------------------------------
def _xysr(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)])


def _box_from_xysr(x):
    w = np.sqrt(max(x[2] * x[3], 0))
    h = x[2] / max(w, 1e-6)
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class _OCTrack:
    """SORT-style Kalman filter on (x, y, s, r) with observation-centric re-update."""
    _next_id = 1

    def __init__(self, box, score, label, delta_t):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1
        self.H = np.eye(4, 7)
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0  # unobserved velocities
        self.P *= 10.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.x = np.r_[_xysr(box), 0, 0, 0]
        self.id = _OCTrack._next_id
        _OCTrack._next_id += 1
        self.score, self.label, self.delta_t = score, label, delta_t
        self.age = self.hits = self.hit_streak = self.time_since_update = 0
        self.observations = {}   # age -> observed box (the creating detection only seeds the filter)
        self.last_obs = None
        self.velocity = np.zeros(2)
        self.frozen = None  # (x, P) saved at the last observation

    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _box_from_xysr(self.x)

    def _kf_update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S, self.H @ self.P).T
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def _prev_obs(self):
        """Observation ``delta_t`` steps back (else the closest one within that
        window, else the latest); None before the first observation."""
        for dt in range(self.delta_t, 0, -1):
            if self.age - dt in self.observations:
                return self.observations[self.age - dt]
        return self.last_obs

    def update(self, box, score):
        box = np.asarray(box, float)
        prev = self._prev_obs()
        if prev is not None:
            v = (box[:2] + box[2:]) / 2 - (prev[:2] + prev[2:]) / 2
            self.velocity = v / (np.linalg.norm(v) + 1e-6)
        if self.frozen is not None and self.last_obs is not None:
            # observation-centric re-update: from the state saved at the first
            # miss, update with a linear virtual trajectory between the last
            # and the new observation (predicting in between)
            self.x, self.P = self.frozen
            gap = self.time_since_update
            for i in range(gap):
                self._kf_update(_xysr(self.last_obs + (box - self.last_obs) * (i + 1) / gap))
                if i < gap - 1:
                    self.x = self.F @ self.x
                    self.P = self.F @ self.P @ self.F.T + self.Q
        self._kf_update(_xysr(box))
        self.frozen = None
        self.last_obs = box
        self.observations[self.age] = box
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = score

    def miss(self):
        if self.frozen is None:
            self.frozen = (self.x.copy(), self.P.copy())


class OCSortTracker:
    def __init__(self, det_thresh=0.6, max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3,
                 inertia=0.2, min_box_area=10):
        self.det_thresh, self.max_age, self.min_hits = det_thresh, max_age, min_hits
        self.iou_threshold, self.delta_t, self.inertia = iou_threshold, delta_t, inertia
        self.min_box_area = min_box_area
        self.reset()

    def reset(self):
        self.trackers, self.frame = [], 0
        _OCTrack._next_id = 1

    def update(self, boxes, scores, labels, frame_bgr=None) -> list:
        self.frame += 1
        keep = scores > self.det_thresh
        dets, dsc = boxes[keep], scores[keep]
        dl = [l for l, k in zip(labels, keep) if k]
        trks = [t for t in self.trackers]
        pred = np.array([t.predict() for t in trks]).reshape(-1, 4)
        ok = np.isfinite(pred).all(1)
        trks, pred = [t for t, k in zip(trks, ok) if k], pred[ok]
        self.trackers = trks

        # first round: IoU + observation-centric momentum (direction consistency)
        iou = iou_matrix(pred, dets)
        if len(trks) and len(dets):
            prev = np.array([p if (p := t._prev_obs()) is not None else np.zeros(4) for t in trks])
            pc = (prev[:, :2] + prev[:, 2:]) / 2
            dc = (dets[:, :2] + dets[:, 2:]) / 2
            dirs = dc[None, :, :] - pc[:, None, :]
            dirs /= np.linalg.norm(dirs, axis=2, keepdims=True) + 1e-6
            vel = np.array([t.velocity for t in trks])
            cos = np.clip((dirs * vel[:, None, :]).sum(2), -1, 1)
            diff = (np.pi / 2 - np.abs(np.arccos(cos))) / np.pi
            valid = (np.linalg.norm(vel, axis=1) > 0)[:, None]
            angle_cost = valid * diff * self.inertia * dsc[None, :]
            m, um_t, um_d = assign(-(iou + angle_cost), 0.0 + 1e9)
            m = [(i, j) for i, j in m if iou[i, j] >= self.iou_threshold]
            mt, md = {i for i, _ in m}, {j for _, j in m}
            um_t = [i for i in range(len(trks)) if i not in mt]
            um_d = [j for j in range(len(dets)) if j not in md]
        else:
            m, um_t, um_d = [], list(range(len(trks))), list(range(len(dets)))
        for i, j in m:
            trks[i].update(dets[j], dsc[j])
        # second round (observation-centric recovery): last observations vs. leftovers
        if um_t and um_d:
            last = np.array([trks[i].last_obs if trks[i].last_obs is not None else -np.ones(4)
                             for i in um_t])
            iou2 = iou_matrix(last, dets[um_d])
            m2, _, _ = assign(-iou2, -self.iou_threshold)
            done_t, done_d = set(), set()
            for a, b in m2:
                trks[um_t[a]].update(dets[um_d[b]], dsc[um_d[b]])
                done_t.add(um_t[a])
                done_d.add(um_d[b])
            um_t = [i for i in um_t if i not in done_t]
            um_d = [j for j in um_d if j not in done_d]
        for i in um_t:
            trks[i].miss()
        for j in um_d:
            self.trackers.append(_OCTrack(dets[j], dsc[j], dl[j], self.delta_t))

        out = []
        for t in self.trackers:
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits or self.frame <= self.min_hits):
                b = t.last_obs if t.last_obs is not None else _box_from_xysr(t.x)
                if (b[2] - b[0]) * (b[3] - b[1]) > self.min_box_area:
                    out.append(t)
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return out


# --------------------------------------------------------------------------
# Runner and rendering
# --------------------------------------------------------------------------
class TrackingRunner:
    """Detector from this collection (``detector`` in model.yaml) + tracker.

    ``infer`` runs the detector graph; the tracker update happens in
    ``postprocess`` (host side), so benchmark latency = detector only and
    end-to-end = detection + tracking.
    """

    def __init__(self, model, provider="cuda", classes=("person",)):
        from tools.mlmc.catalog import get_model
        cfg = model.meta["tracker"]
        self.det = get_model(model.meta["detector"]).load_runner(provider=provider, score_thr=cfg.get("det_score_floor", 0.1))
        self.sess = self.det.sess
        self.classes = set(classes) if classes else None
        params = dict(cfg.get("params", {}))
        self.params = params
        self.aspect_max = cfg.get("aspect_ratio_max")
        algo = cfg["algorithm"]
        if algo == "bytetrack":
            self.tracker = ByteTracker(**params)
        elif algo == "botsort":
            self.tracker = ByteTracker(mode="xywh", gmc=True, **params)
        elif algo == "ocsort":
            self.tracker = OCSortTracker(**params)
        else:
            raise ValueError(f"unknown tracker {algo!r}")

    def reset(self, fps: float | None = None):
        """New video. ByteTrack / BoT-SORT scale their track buffer by fps / 30 (as upstream)."""
        self.tracker.reset()
        if fps and isinstance(self.tracker, ByteTracker):
            buf = self.params.get("track_buffer", 30)
            self.tracker.max_lost = int(fps / 30.0 * buf)

    def preprocess(self, frame_bgr):
        x, meta = self.det.preprocess(frame_bgr)
        return x, (meta, frame_bgr)

    def infer(self, x):
        return self.det.infer(x)

    def postprocess(self, out, meta):
        dmeta, frame = meta
        d = self.det.postprocess(out, dmeta)
        keep = [i for i, l in enumerate(d.labels) if self.classes is None or l in self.classes]
        boxes = np.asarray(d.boxes, np.float64).reshape(-1, 4)[keep]
        scores = np.asarray(d.scores, np.float64)[keep]
        labels = [d.labels[i] for i in keep]
        tracks = self.tracker.update(boxes, scores, labels, frame)
        if not tracks:
            return Tracks(np.zeros((0, 4), np.float32), np.zeros(0, np.int64), np.zeros(0, np.float32), [])
        boxes = np.array([(t.last_obs if t.last_obs is not None else _box_from_xysr(t.x))
                          if isinstance(t, _OCTrack) else t.box for t in tracks], np.float32)
        if self.aspect_max:  # pedestrians: drop boxes wider than aspect_max x height
            ok = (boxes[:, 2] - boxes[:, 0]) / np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6) <= self.aspect_max
            boxes, tracks = boxes[ok], [t for t, k in zip(tracks, ok) if k]
            if not tracks:
                return Tracks(np.zeros((0, 4), np.float32), np.zeros(0, np.int64), np.zeros(0, np.float32), [])
        return Tracks(boxes, np.array([t.id for t in tracks], np.int64),
                      np.array([t.score for t in tracks], np.float32), [t.label for t in tracks])

    def __call__(self, frame_bgr):
        x, meta = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), meta)


def id_color(i: int):
    rng = np.random.default_rng(int(i) * 7919)
    return tuple(int(c) for c in rng.integers(64, 256, 3))


class TrackRenderer:
    """Boxes coloured by id with short trails (stateful, one per video)."""

    def __init__(self, trail=20):
        self.trail, self.hist = trail, {}

    def __call__(self, frame, tr: Tracks):
        out = frame.copy()
        t = max(2, round(frame.shape[1] / 320))  # readable after downscaling to a ~400 px tile
        for b, i in zip(tr.boxes, tr.ids):
            c = id_color(i)
            x1, y1, x2, y2 = b.round().astype(int)
            h = self.hist.setdefault(int(i), [])
            h.append(((x1 + x2) // 2, y2))
            del h[:-self.trail]
            for p, q in zip(h[:-1], h[1:]):
                cv2.line(out, p, q, c, t, cv2.LINE_AA)
            cv2.rectangle(out, (x1, y1), (x2, y2), c, t)
            cv2.putText(out, str(i), (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, c, t, cv2.LINE_AA)
        return out
