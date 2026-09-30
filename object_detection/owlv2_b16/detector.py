"""OWLv2 B/16 (open-vocabulary) — pre/post-processing.

Mirrors transformers' Owlv2ImageProcessor: rescale to [0,1], pad bottom/right
to a square with 0, Gaussian anti-alias blur (sigma = (factor - 1) / 2),
bilinear resize to 960x960, CLIP mean/std normalisation. Boxes are relative
to the padded square. Post-processing: sigmoid of per-query logits, best
query per box, threshold, class-aware NMS (OWLv2 predicts one box per image
patch, so near-duplicates are common). OWLv2 scores are low in absolute
terms (e.g. 0.2 for a clearly visible person), hence the 0.1 default.

``prompts`` defaults to the 80 COCO class names, tokenized at export time
(weights/owlv2_b16_prompts_coco80.npz). Custom prompts are tokenized with
the transformers CLIP tokenizer (only then is transformers needed).
"""

import math

import cv2
import numpy as np

from tools.mlmc.detection import COCO80, Detections, nms_per_class, ort_session

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
REPO = "google/owlv2-base-patch16-ensemble"
REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"


class Detector:
    def __init__(self, model, provider="cuda", score_thr=0.1, iou_thr=0.5, prompts=None):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.size = model.meta["artifacts"]["onnx"]["input_shape"][2]
        self.score_thr, self.iou_thr = score_thr, iou_thr
        if prompts is None:
            z = np.load(model.weights_dir / "owlv2_b16_prompts_coco80.npz")
            self.ids, self.mask, self.names = z["input_ids"], z["attention_mask"], list(COCO80)
        else:
            from transformers import AutoProcessor
            tok = AutoProcessor.from_pretrained(REPO, revision=REVISION).tokenizer(
                [f"a photo of a {p}" for p in prompts], padding="max_length",
                max_length=16, truncation=True, return_tensors="np")
            self.ids = tok["input_ids"].astype(np.int64)
            self.mask = tok["attention_mask"].astype(np.int64)
            self.names = list(prompts)

    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        s = max(h, w)
        img = np.zeros((s, s, 3), np.float32)
        img[:h, :w] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        sigma = max((s / self.size - 1) / 2, 0)
        if sigma > 0:
            k = 2 * math.ceil(3 * sigma) + 1
            img = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma,
                                   borderType=cv2.BORDER_REFLECT)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        x = ((img - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        return x, s

    def infer(self, x):
        return self.sess.run(None, {"pixel_values": x, "input_ids": self.ids,
                                    "attention_mask": self.mask})

    def postprocess(self, out, s):
        logits, boxes = out[0][0], out[1][0]          # (P, Q), (P, 4)
        prob = 1 / (1 + np.exp(-logits))
        cls = prob.argmax(1)
        scores = prob[np.arange(len(cls)), cls]
        m = scores > self.score_thr
        cls, scores, boxes = cls[m], scores[m], boxes[m]
        cx, cy, bw, bh = boxes.T * s
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        keep = nms_per_class(xyxy, scores, cls, self.iou_thr)
        return Detections(xyxy[keep].astype(np.float32), scores[keep].astype(np.float32),
                          [self.names[i] for i in cls[keep]])

    def __call__(self, frame_bgr):
        x, s = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), s)
