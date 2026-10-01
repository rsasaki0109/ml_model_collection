"""Background removal / matting: shared runners and compositing.

Every background-removal runner returns an ``AlphaMatte``: per-pixel
foreground opacity ``(H, W)`` float32 in [0, 1] at the source resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tools.mlmc.detection import ort_session

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
BACKGROUND = (90, 200, 90)  # BGR, used by render()


@dataclass
class AlphaMatte:
    alpha: np.ndarray  # (H, W) float32 in [0, 1]


def _resize(img, size):
    """cv2 resize to (w, h): INTER_AREA when shrinking (antialiased like PIL), else bilinear."""
    shrink = size[0] < img.shape[1] or size[1] < img.shape[0]
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


class SquareMatting:
    """Fixed-size single-image graph ``[1,3,S,S] RGB -> [1,1,S,S]``.

    ``normalize``: ImageNet mean/std after /255 (BiRefNet) or /255 only (BEN2).
    ``output``: ``"logits"`` (sigmoid applied here) or ``"prob"``.
    ``minmax``: rescale the upsampled alpha to [min, max] = [0, 1] per image
    (BEN2's official ``onnx_run.py`` does this).
    """

    def __init__(self, model, provider="cuda", normalize=True, output="logits", minmax=False):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = model.meta["artifacts"]["onnx"]["input_shape"][-1]
        self.normalize, self.output, self.minmax = normalize, output, minmax

    def preprocess(self, frame_bgr):
        img = cv2.cvtColor(_resize(frame_bgr, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        x = img.astype(np.float32) / 255.0
        if self.normalize:
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x.transpose(2, 0, 1)[None], frame_bgr.shape[:2]

    def infer(self, x):
        return self.sess.run(None, {self.input_name: x})[0]

    def postprocess(self, out, size):
        a = out[0, 0].astype(np.float32)
        if not np.isfinite(a).all():
            # e.g. BEN2's mixed-precision graph on some CUDA EP setups; never
            # turn that into an alpha matte or an accuracy number.
            raise RuntimeError(f"non-finite model output ({int((~np.isfinite(a)).sum())} values)")
        if self.output == "logits":
            a = 1.0 / (1.0 + np.exp(-a))
        a = cv2.resize(a, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        if self.minmax:
            lo, hi = float(a.min()), float(a.max())
            a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
        return AlphaMatte(np.clip(a, 0, 1))

    def __call__(self, frame_bgr):
        x, size = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), size)


def render(frame: np.ndarray, matte: AlphaMatte, background=BACKGROUND) -> np.ndarray:
    """Composite the frame over a solid background with the predicted alpha."""
    a = matte.alpha[..., None]
    bg = np.empty_like(frame)
    bg[:] = background
    return (frame * a + bg * (1 - a)).round().astype(np.uint8)
