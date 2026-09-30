"""MoGe-2 ViT-S (official ONNX) — pre/post-processing to metric depth.

The ONNX graph is MoGe's raw ``forward()``: an affine-invariant point map,
normals, a validity mask and a metric scale. ``MoGeModel.infer()`` does the
rest in PyTorch; this file ports it (upstream moge/model/v2.py and
moge/utils/geometry_numpy.py @ 74fbce0):

1. mask = sigmoid mask > 0.5
2. recover the focal length and z-shift by least squares on a 64x64
   nearest-neighbour downsample (torch ``F.interpolate`` sampling) of the
   valid points
   (``solve_optimal_focal_shift``: min |f * xy / (z + shift) - uv|)
3. depth = (z + shift) * metric_scale, invalid pixels (sky, etc.) = inf

``num_tokens`` trades speed for detail (upstream suggests 1200-2500;
``infer()`` defaults to 3600). 1800 is used here, as in upstream's own
ONNX export example.
"""

from functools import partial

import cv2
import numpy as np

from tools.mlmc.depth import METRIC, DepthMap
from tools.mlmc.detection import ort_session


def view_plane_uv(width: int, height: int) -> np.ndarray:
    aspect = width / height
    span_x = aspect / (1 + aspect ** 2) ** 0.5
    span_y = 1 / (1 + aspect ** 2) ** 0.5
    u = np.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=np.float32)
    v = np.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=np.float32)
    return np.stack(np.meshgrid(u, v, indexing="xy"), axis=-1)


def recover_shift(points: np.ndarray, mask: np.ndarray, size=(64, 64)) -> float:
    from scipy.optimize import least_squares

    h, w = points.shape[:2]
    uv = view_plane_uv(w, h)
    # torch F.interpolate(mode="nearest") sampling, as in upstream geometry_torch
    ys = np.minimum(np.floor(np.arange(size[0]) * (h / size[0])).astype(int), h - 1)
    xs = np.minimum(np.floor(np.arange(size[1]) * (w / size[1])).astype(int), w - 1)
    grid = np.ix_(ys, xs)
    keep = mask[grid]
    p, q = points[grid][keep], uv[grid][keep]
    if len(p) < 2:
        return 0.0
    xy, z = p[:, :2], p[:, 2]

    def residual(shift):
        proj = xy / (z + shift)[:, None]
        f = (proj * q).sum() / np.square(proj).sum()
        return (f * proj - q).ravel()

    return float(least_squares(residual, x0=0, ftol=1e-3, method="lm")["x"][0])


class Estimator:
    def __init__(self, model, provider="cuda", num_tokens=1800):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.num_tokens = np.array(num_tokens, dtype=np.int64)

    def preprocess(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return (np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]), self.num_tokens), None

    def infer(self, x):
        image, n = x
        return self.sess.run(None, {"image": image, "num_tokens": n})

    def postprocess(self, out, _meta=None):
        points, _normal, mask, scale = out
        points, mask = points[0], mask[0] > 0.5
        shift = recover_shift(points, mask)
        depth = (points[..., 2] + shift) * float(np.asarray(scale).reshape(-1)[0])
        mask &= depth > 0
        return DepthMap(np.where(mask, depth, np.inf).astype(np.float32), METRIC)

    def __call__(self, frame_bgr):
        x, meta = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), meta)
