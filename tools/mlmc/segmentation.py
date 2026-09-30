"""Common pieces for segmentation models.

``segmentation/<model>/segmenter.py`` exposes a ``Segmenter`` whose
``__call__(frame_bgr)`` returns one of:

* :class:`InstanceMasks` — instance segmentation (boxes + per-instance masks
  + class names), at source resolution.
* :class:`SemanticMap`  — semantic segmentation (a class index per pixel).

Both render with :func:`render` for videos and comparison GIFs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from tools.mlmc.detection import Detections, draw


@dataclass
class InstanceMasks:
    boxes: np.ndarray    # (N, 4) xyxy, source pixels
    scores: np.ndarray   # (N,)
    labels: list[str]    # (N,)
    masks: np.ndarray    # (N, H, W) bool, source resolution

    def __len__(self):
        return len(self.labels)

    def as_detections(self) -> Detections:
        return Detections(self.boxes, self.scores, self.labels)


@dataclass
class SemanticMap:
    classes: np.ndarray                # (H, W) int class index, source resolution
    names: list[str] = field(default_factory=list)


def _palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)  # fixed, so colours are stable across runs
    return rng.integers(40, 255, size=(n, 3), dtype=np.uint8)


def _label_color(label: str) -> tuple[int, int, int]:
    h = sum(map(ord, label)) * 47 % 180
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def render(frame_bgr: np.ndarray, result, alpha: float = 0.5) -> np.ndarray:
    out = frame_bgr.copy()
    if isinstance(result, InstanceMasks):
        order = np.argsort(result.scores)  # draw confident instances last (on top)
        for i in order:
            m = result.masks[i]
            if m.any():
                color = np.array(_label_color(result.labels[i]), np.float32)
                out[m] = (out[m] * (1 - alpha) + color * alpha).astype(np.uint8)
        return draw(out, result.as_detections())
    if isinstance(result, SemanticMap):
        pal = _palette(max(int(result.classes.max()) + 1, len(result.names), 1))
        color = pal[result.classes]
        return (out * (1 - alpha) + color * alpha).astype(np.uint8)
    raise TypeError(f"unsupported result {type(result).__name__}")
