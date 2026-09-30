"""Common pieces for depth estimation models.

Every ``depth_estimation/<model>/estimator.py`` exposes an ``Estimator`` whose
``__call__(frame_bgr)`` returns a :class:`DepthMap` at the source resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# What the numbers in a depth map mean. Tools must not mix these up.
DISPARITY = "relative_disparity"   # affine-invariant inverse depth, larger = nearer
DEPTH = "relative_depth"           # affine-invariant depth, larger = farther
METRIC = "metric_depth_m"          # metres


@dataclass
class DepthMap:
    values: np.ndarray   # (H, W) float32 at source resolution
    kind: str            # DISPARITY / DEPTH / METRIC


def colorize(d: DepthMap, cmap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Near = bright, far = dark, whatever the model outputs.

    Relative outputs are normalised per frame (2nd-98th percentile), so
    colours compare structure, not absolute scale.
    """
    v = d.values.astype(np.float32)
    if d.kind in (DEPTH, METRIC):
        v = 1.0 / np.maximum(v, 1e-6)  # to disparity-like: near = large
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cmap)
