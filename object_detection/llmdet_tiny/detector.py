"""LLMDet Swin-T (open-vocabulary, Grounding-DINO style) — pre/post-processing.

Pre-processing follows GroundingDinoImageProcessor: resize so the shortest
edge is 800 and the longest <= 1333 (keeping aspect ratio), ImageNet mean/std;
then pad bottom/right into the fixed 800x1333 canvas with a pixel mask.
Boxes are relative to the valid (unpadded) image.

Token logits are converted to class scores with the positive map saved at
export time: score(class) = mean over the class's tokens of sigmoid(logit),
as in mmdetection's Grounding DINO evaluation. Then top-300 over
(query, class) pairs; NMS-free.

The prompt is fixed at export time (80 COCO class names); changing the
vocabulary means re-running export.py with different names.
"""

import cv2
import numpy as np

from tools.mlmc.detection import COCO80, Detections, ort_session

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
H, W = 800, 1333


class Detector:
    def __init__(self, model, provider="cuda", score_thr=0.3, top_k=300):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        z = np.load(model.weights_dir / "llmdet_tiny_prompt_coco80.npz")
        self.text = {k: z[k] for k in ("input_ids", "attention_mask", "token_type_ids")}
        pos = z["positive_map"]
        self.pos = pos / pos.sum(1, keepdims=True)           # (C, 256), rows mean
        self.score_thr, self.top_k = score_thr, top_k

    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        # Upstream: shortest edge 800, longest <= 1333. The static canvas is
        # landscape (800x1333), so portrait images are fit inside it instead
        # (height capped at 800), which makes them smaller than upstream.
        r = min(H / min(h, w), W / max(h, w), H / h, W / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        img = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (nw, nh),
                         interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        canvas = np.zeros((H, W, 3), np.float32)
        canvas[:nh, :nw] = (img - MEAN) / STD
        mask = np.zeros((1, H, W), np.int64)
        mask[:, :nh, :nw] = 1
        return (canvas.transpose(2, 0, 1)[None], mask), (w, h)

    def infer(self, x):
        pixels, mask = x
        return self.sess.run(None, {"pixel_values": pixels, "pixel_mask": mask, **self.text})

    def postprocess(self, out, wh):
        logits, boxes = out[0][0], out[1][0]                  # (Q, 256), (Q, 4)
        prob = 1 / (1 + np.exp(-logits)) @ self.pos.T         # (Q, C)
        flat = prob.reshape(-1)
        idx = np.argsort(-flat)[: self.top_k]
        scores = flat[idx]
        q, cls = idx // prob.shape[1], idx % prob.shape[1]
        m = scores > self.score_thr
        scores, q, cls = scores[m], q[m], cls[m]
        cx, cy, bw, bh = boxes[q].T
        w, h = wh
        xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                         (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        return Detections(xyxy.astype(np.float32), scores.astype(np.float32),
                          [COCO80[i] for i in cls])

    def __call__(self, frame_bgr):
        x, wh = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), wh)
