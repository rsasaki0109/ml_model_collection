"""Optical flow: shared two-frame ONNX runner and flow visualisation.

Every optical-flow runner returns a ``Flow``: per-pixel displacement
``(H, W, 2)`` float32 (u, v) in source pixels, from the previous frame to the
current one. Runners are stateful video runners: each call pairs the frame
with the previous one (the first frame is paired with itself, i.e. zero
motion is expected).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tools.mlmc.detection import ort_session


@dataclass
class Flow:
    uv: np.ndarray  # (H, W, 2) float32, pixels


class TwoFrameFlow:
    """ONNX graph ``(image1, image2) [1,3,H,W] RGB float 0-255 -> flow [1,2,H,W]``.

    Frames are resized to the fixed export resolution (``input_shape`` in
    model.yaml); the flow is resized back and scaled to source pixels.
    """

    def __init__(self, model, provider="cuda"):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.names = [i.name for i in self.sess.get_inputs()]
        self.h, self.w = model.meta["artifacts"]["onnx"]["input_shape"][-2:]
        self.prev = None

    def _tensor(self, frame_bgr):
        img = cv2.resize(frame_bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32).transpose(2, 0, 1)[None]

    def preprocess(self, frame_bgr):
        cur = self._tensor(frame_bgr)
        prev = self.prev if self.prev is not None else cur
        self.prev = cur
        return (prev, cur), frame_bgr.shape[:2]

    def infer(self, x):
        return self.sess.run(None, dict(zip(self.names, x)))[0]

    def postprocess(self, out, size):
        h, w = size
        uv = cv2.resize(out[0].transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LINEAR)
        uv *= np.array([w / self.w, h / self.h], np.float32)
        return Flow(uv.astype(np.float32))

    def __call__(self, frame_bgr):
        x, size = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), size)


def flow_to_color(uv: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    """Standard HSV flow colouring (hue = direction, saturation = magnitude), BGR.

    ``max_mag`` defaults to the 99th percentile magnitude of the frame.
    """
    u, v = uv[..., 0], uv[..., 1]
    mag, ang = cv2.cartToPolar(u, v)
    if max_mag is None:
        max_mag = float(np.percentile(mag, 99))
    max_mag = max(max_mag, 1.0)
    hsv = np.zeros(uv.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)  # 0..180 in OpenCV
    hsv[..., 1] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    hsv[..., 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def render(frame: np.ndarray, flow: Flow, max_mag: float = 20.0, alpha: float = 0.6) -> np.ndarray:
    """Flow colours (white = static, |flow| >= ``max_mag`` px fully saturated)
    blended over the grayscale frame so the scene stays visible."""
    gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(flow_to_color(flow.uv, max_mag), alpha, gray, 1 - alpha, 0)


def read_flo(path) -> np.ndarray:
    """Middlebury ``.flo`` (tag PIEH, w, h, interleaved u, v float32) -> (H, W, 2)."""
    with open(path, "rb") as f:
        tag = np.fromfile(f, np.float32, 1)[0]
        if tag != 202021.25:
            raise ValueError(f"{path}: not a .flo file")
        w, h = np.fromfile(f, np.int32, 2)
        return np.fromfile(f, np.float32, 2 * w * h).reshape(h, w, 2)
