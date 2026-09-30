"""D-FINE-style deploy graphs (D-FINE / DEIM / RT-DETRv4 official exporters).

Graph (see tools/mlmc/export.py: export_dfine_family):
    images [1,3,S,S] RGB in [0,1], plain resize (no letterbox, no mean/std)
    orig_target_sizes [1,2] int64 (w, h)
 -> labels [1,300] (0..79), boxes [1,300,4] xyxy in original pixels,
    scores [1,300]. Top-k is inside the graph; NMS-free.
"""

import cv2
import numpy as np

from tools.mlmc.detection import COCO80, Detections, ort_session


class Detector:
    def __init__(self, model, provider="cuda", score_thr=0.5):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.size = model.meta["artifacts"]["onnx"]["input_shape"][2]
        self.score_thr = score_thr

    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img = cv2.cvtColor(cv2.resize(frame_bgr, (self.size, self.size)),
                           cv2.COLOR_BGR2RGB)
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        return (x, np.array([[w, h]], dtype=np.int64)), None

    def infer(self, x):
        images, sizes = x
        return self.sess.run(None, {"images": images, "orig_target_sizes": sizes})

    def postprocess(self, out, _meta=None):
        labels, boxes, scores = out[0][0], out[1][0], out[2][0]
        m = scores > self.score_thr
        return Detections(boxes[m].astype(np.float32), scores[m].astype(np.float32),
                          [COCO80[int(i)] for i in labels[m]])

    def __call__(self, frame_bgr):
        x, meta = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), meta)
