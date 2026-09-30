"""SAM 2-family promptable segmentation (onnx-community ONNX exports).

Two graphs, as exported by onnx-community for transformers' Sam2 / EdgeTam:
    vision_encoder:  pixel_values [1,3,1024,1024] -> image_embeddings.0/.1/.2
    prompt_encoder_mask_decoder:
        input_points [1,1,N,2], input_labels [1,1,N], input_boxes [1,nb,4]
        (coordinates in the 1024x1024 input frame) + the three embeddings
        -> iou_scores [1,nb,3], pred_masks [1,nb,3,256,256], object_score_logits

Pre-processing follows ``Sam2ImageProcessor``: RGB, resize (no padding) to
1024x1024, ImageNet mean/std. For each prompt the mask with the highest
predicted IoU is kept (multimask output), bilinearly upsampled from 256x256
to the source size and thresholded at 0.

Prompts: by default boxes come from a detector in this collection
(``prompt_model``, e.g. D-FINE-N), so the model behaves as "detect, then
segment" and returns :class:`InstanceMasks` with the detector's labels and
scores. ``boxes=`` can be passed to ``segment()`` for user prompts.
"""

from __future__ import annotations

import cv2
import numpy as np

from tools.mlmc.detection import ort_session
from tools.mlmc.segmentation import InstanceMasks

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
S = 1024


class Segmenter:
    def __init__(self, model, provider="cuda", prompt_model="dfine_n", score_thr=0.5,
                 max_prompts=100):
        from tools.mlmc.catalog import get_model

        self.enc = ort_session(model.artifact_path("onnx"), provider)
        self.dec = ort_session(model.weights_dir / "prompt_encoder_mask_decoder.onnx", provider)
        self.sess = self.enc  # benchmark.py reads the execution provider from .sess
        self.detector = get_model(prompt_model).load_runner(provider=provider, score_thr=score_thr)
        self.max_prompts = max_prompts
        self.emb_names = [o.name for o in self.enc.get_outputs()]

    def preprocess(self, frame_bgr):
        """Runs the prompt detector too (its time counts as pre-processing)."""
        h, w = frame_bgr.shape[:2]
        det = self.detector(frame_bgr)
        order = np.argsort(-det.scores)[: self.max_prompts]
        boxes = det.boxes[order] * np.array([S / w, S / h, S / w, S / h], np.float32)
        img = cv2.cvtColor(cv2.resize(frame_bgr, (S, S), interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_BGR2RGB)
        x = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        meta = (w, h, det.boxes[order], det.scores[order], [det.labels[i] for i in order])
        return (np.ascontiguousarray(x), boxes.astype(np.float32)), meta

    def infer(self, x):
        pixels, boxes = x
        emb = self.enc.run(None, {"pixel_values": pixels})
        feeds = {name: e for name, e in zip(self.emb_names, emb)}
        feeds.update({
            "input_points": np.zeros((1, 1, 0, 2), np.float32),
            "input_labels": np.zeros((1, 1, 0), np.int64),
            "input_boxes": boxes[None] if len(boxes) else np.zeros((1, 0, 4), np.float32),
        })
        if not len(boxes):
            return None
        return self.dec.run(["iou_scores", "pred_masks"], feeds)

    def postprocess(self, out, meta):
        w, h, boxes, scores, labels = meta
        masks = np.zeros((len(boxes), h, w), bool)
        if out is not None:
            iou, pred = out[0][0], out[1][0]                 # (nb, 3), (nb, 3, 256, 256)
            best = iou.argmax(-1)
            for i in range(len(boxes)):
                masks[i] = cv2.resize(pred[i, best[i]], (w, h), interpolation=cv2.INTER_LINEAR) > 0
        return InstanceMasks(boxes.astype(np.float32), scores.astype(np.float32), labels, masks)

    def __call__(self, frame_bgr):
        x, meta = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), meta)
