"""Image classification (ImageNet-1k classes): supervised and zero-shot runners.

Every classification runner returns ``Classes``: the top-k ImageNet class
indices (0-999), their probabilities / similarities and names.

* ``TimmClassifier`` — timm-style evaluation transform: resize the shorter
  side to ``size / crop_pct`` (PIL, bicubic), centre crop ``size``,
  ImageNet mean / std; graph outputs 1000 logits.
* ``ZeroShotClassifier`` — image-text models: the graph returns the image
  embedding; it is compared (cosine) with precomputed, L2-normalised text
  embeddings of the 1000 class prompts (``text_embeddings`` artifact).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tools.mlmc.detection import ort_session

CLASSNAMES = (Path(__file__).parent / "data" / "imagenet_classnames.txt").read_text(encoding="utf-8").splitlines()
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
_INTERP = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}


@dataclass
class Classes:
    indices: np.ndarray  # (k,) int64, ImageNet class index
    scores: np.ndarray   # (k,) float32, softmax probability or cosine similarity
    names: list


def _to_pil(frame_bgr):
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def _topk(scores: np.ndarray, k: int) -> Classes:
    idx = np.argsort(-scores)[:k]
    return Classes(idx.astype(np.int64), scores[idx].astype(np.float32), [CLASSNAMES[i] for i in idx])


class TimmClassifier:
    def __init__(self, model, provider="cuda", top_k=5):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.input_name = self.sess.get_inputs()[0].name
        cfg = model.meta["classification"]
        self.size, self.crop_pct = cfg["size"], cfg["crop_pct"]
        self.interp = _INTERP[cfg.get("interpolation", "bicubic")]
        self.top_k = top_k

    def preprocess(self, frame_bgr):
        img = _to_pil(frame_bgr)
        short = int(self.size / self.crop_pct)  # math.floor, as timm's transforms_factory
        w, h = img.size
        if w < h:
            nw, nh = short, int(short * h / w)
        else:
            nh, nw = short, int(short * w / h)
        img = img.resize((nw, nh), self.interp)
        left, top = int(round((nw - self.size) / 2.0)), int(round((nh - self.size) / 2.0))
        img = img.crop((left, top, left + self.size, top + self.size))
        x = (np.asarray(img, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), None

    def infer(self, x):
        return self.sess.run(None, {self.input_name: x})[0]

    def postprocess(self, logits, meta=None):
        z = logits[0].astype(np.float64)
        p = np.exp(z - z.max())
        return _topk((p / p.sum()).astype(np.float32), self.top_k)

    def __call__(self, frame_bgr):
        x, meta = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), meta)


class ZeroShotClassifier:
    def __init__(self, model, provider="cuda", top_k=5):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.input_name = self.sess.get_inputs()[0].name
        self.output = model.meta["classification"].get("image_output", "pooler_output")
        cfg = model.meta["classification"]
        self.size = cfg["size"]
        self.mean, self.std = np.float32(cfg["mean"]), np.float32(cfg["std"])
        self.interp = _INTERP[cfg.get("interpolation", "bilinear")]
        self.text = np.load(model.weights_dir / cfg["text_embeddings"]).astype(np.float32)  # (1000, D), normalised
        self.top_k = top_k

    def preprocess(self, frame_bgr):
        img = _to_pil(frame_bgr).resize((self.size, self.size), self.interp)  # no crop (SigLIP processor)
        x = (np.asarray(img, np.float32) / 255.0 - self.mean) / self.std
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), None

    def infer(self, x):
        names = [o.name for o in self.sess.get_outputs()]
        return self.sess.run([self.output if self.output in names else names[-1]], {self.input_name: x})[0]

    def postprocess(self, emb, meta=None):
        e = emb[0] / np.linalg.norm(emb[0])
        return _topk(self.text @ e, self.top_k)

    def __call__(self, frame_bgr):
        x, meta = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), meta)


def render(frame: np.ndarray, res: Classes, center_crop: bool = True) -> np.ndarray:
    """Top-5 classes as a panel over the frame."""
    out = frame.copy()
    H, W = out.shape[:2]
    s = W / 640
    lines = [f"{n[:28]}  {sc:.2f}" for n, sc in zip(res.names, res.scores)]
    lh = int(34 * s)
    cv2.rectangle(out, (0, 0), (int(W * 0.62), lh * len(lines) + int(14 * s)), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(out, t, (int(10 * s), lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.85 * s,
                    (255, 255, 255) if i else (80, 255, 80), max(1, round(2 * s)), cv2.LINE_AA)
    return out
