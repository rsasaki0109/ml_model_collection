"""Super-resolution: shared x4 runner and side-by-side rendering.

Every super-resolution runner returns an ``SRImage``: the upscaled BGR image
plus the low-resolution input it was computed from.

Graphs take ``lr [1,3,h,w]`` RGB float in [0, 1] (dynamic h, w) and return
``sr [1,3,4h,4w]``. The deployment setting in ``model.yaml`` (``input_shape``)
is what video runs and benchmarks use: every frame is first downscaled to
that size (bicubic, as the LR input), then upscaled x4 — e.g. 320x180 ->
1280x720. ``upscale(lr)`` runs on an LR image as-is (used by evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tools.mlmc.detection import ort_session


@dataclass
class SRImage:
    image: np.ndarray  # (4h, 4w, 3) uint8 BGR
    lr: np.ndarray     # (h, w, 3) uint8 BGR input


class SuperRes:
    scale = 4

    def __init__(self, model, provider="cuda", pad_multiple: int = 1):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.h, self.w = model.meta["artifacts"]["onnx"]["input_shape"][-2:]
        self.pad_multiple = pad_multiple

    def _tensor(self, lr_bgr):
        h, w = lr_bgr.shape[:2]
        m = self.pad_multiple
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            lr_bgr = cv2.copyMakeBorder(lr_bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        x = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return x.transpose(2, 0, 1)[None], (h, w)

    def preprocess(self, frame_bgr):
        if frame_bgr.shape[:2] != (self.h, self.w):
            lr = cv2.resize(frame_bgr, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        else:
            lr = frame_bgr
        x, size = self._tensor(lr)
        return x, (lr, size)

    def infer(self, x):
        return self.sess.run(None, {"lr": x})[0]

    def postprocess(self, out, meta):
        lr, (h, w) = meta
        sr = out[0, :, : h * self.scale, : w * self.scale].transpose(1, 2, 0)
        sr = (np.clip(sr, 0, 1) * 255).round().astype(np.uint8)
        return SRImage(cv2.cvtColor(sr, cv2.COLOR_RGB2BGR), lr)

    def upscale(self, lr_bgr):
        x, size = self._tensor(lr_bgr)
        return self.postprocess(self.infer(x), (lr_bgr, size)).image

    def __call__(self, frame_bgr):
        x, meta = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), meta)


# Region shown in the comparison (fractions of the frame): building facade,
# street signs and the van in assets/demo.mp4.
CROP = (0.30, 0.03, 0.35, 0.35)  # x, y, w, h


def render(frame: np.ndarray, res: SRImage, crop=CROP) -> np.ndarray:
    """Zoomed crop, left half bicubic upscale of the LR input, right half model output."""
    sr = res.image
    H, W = sr.shape[:2]
    bic = cv2.resize(res.lr, (W, H), interpolation=cv2.INTER_CUBIC)
    x0, y0 = int(crop[0] * W), int(crop[1] * H)
    cw, ch = int(crop[2] * W), int(crop[3] * H)
    view = sr[y0:y0 + ch, x0:x0 + cw].copy()
    half = cw // 2
    view[:, :half] = bic[y0:y0 + ch, x0:x0 + half]
    out = cv2.resize(view, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    mid = out.shape[1] // 2
    fs = out.shape[1] / 400  # readable after downscaling to a ~400 px tile
    t = max(1, round(fs * 2))
    cv2.line(out, (mid, 0), (mid, out.shape[0]), (255, 255, 255), t)
    for text, x in (("bicubic", 10), ("model", mid + 10)):
        y = int(30 * fs)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3 * t, cv2.LINE_AA)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), t, cv2.LINE_AA)
    return out


def _cubic(x):
    ax = np.abs(x)
    ax2, ax3 = ax ** 2, ax ** 3
    return ((1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1)
            + (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2)))


def _resize_weights(in_len: int, out_len: int, scale: float):
    """MATLAB imresize contributions (bicubic, antialiasing when shrinking)."""
    kernel_width = 4.0 / scale if scale < 1 else 4.0
    x = np.arange(1, out_len + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    p = int(np.ceil(kernel_width)) + 2
    idx = left[:, None] + np.arange(p)[None, :]
    dist = u[:, None] - idx
    w = scale * _cubic(dist * scale) if scale < 1 else _cubic(dist)
    w /= w.sum(1, keepdims=True)
    aux = np.r_[np.arange(in_len), np.arange(in_len)[::-1]]  # MATLAB's symmetric border
    idx = aux[np.mod(idx.astype(int) - 1, len(aux))]
    return w, idx


def imresize_matlab(img: np.ndarray, scale: float) -> np.ndarray:
    """MATLAB ``imresize(img, scale, 'bicubic')`` (with antialiasing) for uint8
    images, as used to make the x4 low-resolution benchmark inputs."""
    x = img.astype(np.float64)
    h, w = x.shape[:2]
    oh, ow = int(np.ceil(h * scale)), int(np.ceil(w * scale))
    wh, ih = _resize_weights(h, oh, scale)
    ww, iw = _resize_weights(w, ow, scale)
    x = np.einsum("op,op...->o...", wh, x[ih])               # rows
    x = np.einsum("op,xop...->xo...", ww, x[:, iw])          # columns
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def psnr_y(a_bgr: np.ndarray, b_bgr: np.ndarray, crop: int) -> float:
    """PSNR on the BT.601 Y channel (MATLAB ``rgb2ycbcr``), ``crop`` px border removed."""
    def y(img):
        img = img.astype(np.float64)
        return 16.0 + (24.966 * img[..., 0] + 128.553 * img[..., 1] + 65.481 * img[..., 2]) / 255.0
    ya, yb = y(a_bgr), y(b_bgr)
    if crop:
        ya, yb = ya[crop:-crop, crop:-crop], yb[crop:-crop, crop:-crop]
    mse = np.mean((ya - yb) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
