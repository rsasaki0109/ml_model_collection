"""Scene-text OCR (detection + recognition): PP-OCR-style pipeline runner.

Every OCR runner returns ``OcrResult``: text-line quadrilaterals ``(N, 4, 2)``
(top-left, top-right, bottom-right, bottom-left, source pixels), recognised
strings and recognition confidences.

Re-implements PaddleOCR's inference path at commit
dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf (Apache-2.0):

* detection — ``DetResizeForTest`` (type 0: ``limit_type``/``limit_side_len``,
  sides rounded to multiples of 32; or ``resize_long``: multiples of 128),
  BGR, /255, ImageNet mean/std; ``DBPostProcess`` "fast" box score,
  ``unclip`` with pyclipper (round joins), min side 3 / 5;
* ``sorted_boxes`` + ``get_rotate_crop_image`` (perspective crop,
  INTER_CUBIC, replicate border, rotate if h / w >= 1.5);
* recognition — ``resize_norm_img`` (height 48, batch width from the widest
  crop, (x / 255 - 0.5) / 0.5, BGR, right zero padding), batches of 6 sorted
  by aspect ratio, ``CTCLabelDecode`` (blank + dictionary + space).

Parameters come from the ``ocr:`` section of model.yaml (each model's
``inference.yml``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from tools.mlmc.detection import ort_session

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class OcrResult:
    polys: np.ndarray                 # (N, 4, 2) float32
    texts: list = field(default_factory=list)
    scores: np.ndarray = None         # (N,) recognition confidence


def _det_resize(img, cfg):
    h, w = img.shape[:2]
    if "resize_long" in cfg:
        ratio = cfg["resize_long"] / max(h, w)
        rh, rw = int(h * ratio), int(w * ratio)
        rh, rw = (rh + 127) // 128 * 128, (rw + 127) // 128 * 128
    else:
        side, kind = cfg.get("limit_side_len", 736), cfg.get("limit_type", "min")
        if kind == "max":
            ratio = side / max(h, w) if max(h, w) > side else 1.0
        elif kind == "min":
            ratio = side / min(h, w) if min(h, w) < side else 1.0
        else:  # resize_long
            ratio = side / max(h, w)
        rh, rw = int(h * ratio), int(w * ratio)
        limit = cfg.get("max_side_limit", 4000)
        if max(rh, rw) > limit:
            r2 = limit / max(rh, rw)
            rh, rw = int(rh * r2), int(rw * r2)
        rh, rw = max(int(round(rh / 32) * 32), 32), max(int(round(rw / 32) * 32), 32)
    return cv2.resize(img, (rw, rh))


def _mini_box(contour):
    rect = cv2.minAreaRect(contour)
    pts = sorted(list(cv2.boxPoints(rect)), key=lambda p: p[0])
    i1, i4 = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    i2, i3 = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
    return np.array([pts[i1], pts[i2], pts[i3], pts[i4]], np.float32), min(rect[1])


def _box_score_fast(prob, box):
    h, w = prob.shape
    xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
    b = box.copy()
    b[:, 0] -= xmin
    b[:, 1] -= ymin
    cv2.fillPoly(mask, b.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(prob[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def _unclip(box, ratio):
    import pyclipper
    area = cv2.contourArea(box)
    length = cv2.arcLength(box.reshape(-1, 1, 2), True)
    off = pyclipper.PyclipperOffset()
    off.AddPath([tuple(p) for p in box.tolist()], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    return off.Execute(area * ratio / length)


def db_boxes(prob, src_w, src_h, thresh, box_thresh, unclip_ratio, max_candidates, min_size=3):
    """``DBPostProcess.boxes_from_bitmap`` (score_mode fast, no dilation)."""
    h, w = prob.shape
    contours, _ = cv2.findContours(((prob > thresh) * 255).astype(np.uint8),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes, scores = [], []
    for c in contours[:max_candidates]:
        pts, sside = _mini_box(c)
        if sside < min_size:
            continue
        score = _box_score_fast(prob, pts)
        if box_thresh > score:
            continue
        exp = _unclip(pts, unclip_ratio)
        if len(exp) != 1:
            continue
        box, sside = _mini_box(np.array(exp[0], np.float32).reshape(-1, 1, 2))
        if sside < min_size + 2:
            continue
        box[:, 0] = np.clip(np.round(box[:, 0] / w * src_w), 0, src_w)
        box[:, 1] = np.clip(np.round(box[:, 1] / h * src_h), 0, src_h)
        boxes.append(box.astype(np.int32))
        scores.append(score)
    return boxes, scores


def _order_clockwise(pts):
    """``TextDetector.order_points_clockwise``."""
    s = pts.sum(1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    rest = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
    d = np.diff(rest, axis=1)
    return np.array([tl, rest[np.argmin(d)], br, rest[np.argmax(d)]], np.float32)


def _sorted_boxes(boxes):
    """``sorted_boxes``: top-to-bottom, then left-to-right within 10 px."""
    b = sorted(boxes, key=lambda x: (x[0][1], x[0][0]))
    for i in range(len(b) - 1):
        for j in range(i, -1, -1):
            if abs(b[j + 1][0][1] - b[j][0][1]) < 10 and b[j + 1][0][0] < b[j][0][0]:
                b[j], b[j + 1] = b[j + 1], b[j]
            else:
                break
    return b


def rotate_crop(img, pts):
    """``get_rotate_crop_image``."""
    pts = pts.astype(np.float32)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    crop = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (w, h),
                               borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if crop.shape[0] / max(crop.shape[1], 1) >= 1.5:
        crop = np.rot90(crop)
    return crop


class PPOCR:
    rec_batch = 6

    def __init__(self, model, provider="cuda"):
        cfg = model.meta["ocr"]
        wd = model.weights_dir
        self.sess = ort_session(model.artifact_path("onnx"), provider)  # detector
        self.rec = ort_session(wd / cfg["rec_file"], provider)
        self.det_in = self.sess.get_inputs()[0].name
        self.rec_in = self.rec.get_inputs()[0].name
        self.det_resize = cfg.get("det_resize", {})
        self.db = cfg["db"]
        self.drop_score = cfg.get("drop_score", 0.5)
        _, self.rec_h, self.rec_w = cfg.get("rec_image_shape", [3, 48, 320])
        chars = (wd / cfg["dict_file"]).read_text(encoding="utf-8").splitlines()
        self.labels = ["blank"] + [c.strip("\r\n") for c in chars] + [" "]

    # -- detection ---------------------------------------------------------
    def preprocess(self, frame_bgr):
        img = _det_resize(frame_bgr, self.det_resize)
        x = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(x), frame_bgr

    def infer(self, x):
        return self.sess.run(None, {self.det_in: x})[0]

    def detect(self, prob, frame):
        h, w = frame.shape[:2]
        boxes, _ = db_boxes(prob[0, 0], w, h, self.db["thresh"], self.db["box_thresh"],
                            self.db["unclip_ratio"], self.db.get("max_candidates", 1000))
        kept = []
        for b in boxes:
            b = _order_clockwise(b.astype(np.float32))
            b[:, 0] = np.clip(b[:, 0], 0, w - 1)
            b[:, 1] = np.clip(b[:, 1], 0, h - 1)
            if int(np.linalg.norm(b[0] - b[1])) <= 3 or int(np.linalg.norm(b[0] - b[3])) <= 3:
                continue
            kept.append(b)
        return _sorted_boxes(kept)

    # -- recognition -------------------------------------------------------
    def _norm(self, crop, img_w):
        h, w = crop.shape[:2]
        rw = img_w if math.ceil(self.rec_h * w / h) > img_w else int(math.ceil(self.rec_h * w / h))
        x = cv2.resize(crop, (rw, self.rec_h)).astype(np.float32).transpose(2, 0, 1) / 255.0
        out = np.zeros((3, self.rec_h, img_w), np.float32)
        out[:, :, :rw] = (x - 0.5) / 0.5
        return out

    def recognize(self, crops):
        ratios = [c.shape[1] / float(c.shape[0]) for c in crops]
        order = np.argsort(ratios)
        res = [("", 0.0)] * len(crops)
        for s in range(0, len(crops), self.rec_batch):
            idx = order[s:s + self.rec_batch]
            max_ratio = max([self.rec_w / self.rec_h] + [ratios[i] for i in idx])
            img_w = int(self.rec_h * max_ratio)
            batch = np.stack([self._norm(crops[i], img_w) for i in idx])
            probs = self.rec.run(None, {self.rec_in: batch})[0]
            for i, p in zip(idx, probs):
                ids, conf = p.argmax(-1), p.max(-1)
                keep = np.ones(len(ids), bool)
                keep[1:] = ids[1:] != ids[:-1]
                keep &= ids != 0
                text = "".join(self.labels[k] for k in ids[keep])
                res[i] = (text, float(conf[keep].mean()) if keep.any() else 0.0)
        return res

    def postprocess(self, prob, frame):
        boxes = self.detect(prob, frame)
        if not boxes:
            return OcrResult(np.zeros((0, 4, 2), np.float32), [], np.zeros(0, np.float32))
        texts = self.recognize([rotate_crop(frame, b) for b in boxes])
        keep = [i for i, (_, s) in enumerate(texts) if s >= self.drop_score]
        return OcrResult(np.array([boxes[i] for i in keep], np.float32).reshape(-1, 4, 2),
                         [texts[i][0] for i in keep], np.array([texts[i][1] for i in keep], np.float32))

    def __call__(self, frame_bgr):
        x, frame = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), frame)


def render(frame: np.ndarray, res: OcrResult) -> np.ndarray:
    """Text-line quads and recognised strings (non-ASCII shown as '?')."""
    out = frame.copy()
    t = max(1, round(frame.shape[1] / 640))
    for poly, text in zip(res.polys, res.texts):
        p = poly.round().astype(np.int32)
        cv2.polylines(out, [p], True, (0, 255, 255), t)
        label = "".join(c if 32 <= ord(c) < 127 else "?" for c in text)
        org = (int(p[:, 0].min()), max(12, int(p[:, 1].min()) - 4))
        fs = 0.5 * t
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        cv2.rectangle(out, (org[0], org[1] - th - 2), (org[0] + tw, org[1] + 2), (0, 0, 0), -1)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), t, cv2.LINE_AA)
    return out
