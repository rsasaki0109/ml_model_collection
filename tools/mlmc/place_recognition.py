"""Visual place recognition: global image descriptors, retrieval rendering, Recall@N.

Every place-recognition runner returns ``Descriptor``: one L2-normalised
global descriptor per image; places are matched by cosine similarity (inner
product). ``preprocess`` follows each model's evaluation setting
(``place_recognition:`` in model.yaml): ImageNet mean / std at the native
resolution (CosPlace / EigenPlaces, optionally capped by ``max_side``) or a
fixed square resize (MegaLoc, 322x322).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from tools.mlmc.detection import ort_session

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Descriptor:
    vector: np.ndarray  # (D,) float32, L2-normalised


class GlobalDescriptor:
    def __init__(self, model, provider="cuda"):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.input_name = self.sess.get_inputs()[0].name
        cfg = model.meta["place_recognition"]
        self.size = cfg.get("resize")        # fixed square side, or None for native resolution
        self.max_side = cfg.get("max_side")  # cap for native-resolution models (video frames)

    def preprocess(self, frame_bgr):
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if self.size:
            img = img.resize((self.size, self.size), Image.BILINEAR)  # antialiased, as torchvision Resize
        elif self.max_side and max(img.size) > self.max_side:
            s = self.max_side / max(img.size)
            img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.BILINEAR)
        x = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), None

    def infer(self, x):
        return self.sess.run(None, {self.input_name: x})[0]

    def postprocess(self, out, meta=None):
        v = out[0].astype(np.float32)
        return Descriptor(v / max(float(np.linalg.norm(v)), 1e-12))

    def __call__(self, frame_bgr):
        x, meta = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), meta)


class RetrievalView:
    """Stateful renderer: the current frame with the most similar earlier frame
    (at least ``gap`` frames back) inset, plus the cosine similarity."""

    def __init__(self, gap=25, every=5):
        self.gap, self.every, self.db, self.n = gap, every, [], 0

    def __call__(self, frame, d: Descriptor):
        out = frame.copy()
        H, W = out.shape[:2]
        cand = [(i, f, v) for i, f, v in self.db if self.n - i >= self.gap]
        if cand:
            sims = [float(v @ d.vector) for _, _, v in cand]
            k = int(np.argmax(sims))
            thumb = cv2.resize(cand[k][1], (W // 2, H // 2))
            out[H - H // 2:, W - W // 2:] = thumb
            t = max(1, round(W / 320))
            cv2.rectangle(out, (W - W // 2, H - H // 2), (W - 1, H - 1), (0, 255, 255), t)
            label = f"match #{cand[k][0]}  cos {sims[k]:.2f}"
            cv2.putText(out, label, (W - W // 2 + 8, H - H // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        W / 900, (0, 0, 0), 4 * t, cv2.LINE_AA)
            cv2.putText(out, label, (W - W // 2 + 8, H - H // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        W / 900, (0, 255, 255), t, cv2.LINE_AA)
        if self.n % self.every == 0:
            self.db.append((self.n, frame.copy(), d.vector))
        self.n += 1
        return out


def recall_at_n(q: np.ndarray, db: np.ndarray, positives: list, ns=(1, 5, 10)) -> dict:
    """% of queries with a positive among the top-N database items (cosine on
    L2-normalised descriptors, as faiss IndexFlatL2 ranking)."""
    order = np.argsort(-(q @ db.T), axis=1)[:, :max(ns)]
    return {f"R@{n}": round(100 * float(np.mean([len(set(order[i, :n]) & set(p)) > 0
                                                   for i, p in enumerate(positives)])), 1) for n in ns}
