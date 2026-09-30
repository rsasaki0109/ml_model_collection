"""SSDLite320-MobileNetV3 (torchvision export) — pre/post-processing.

The graph already normalises and runs NMS; here we only resize, convert to
RGB [0, 1], rescale boxes and map torchvision's 91-id COCO labels to names.
"""

import cv2
import numpy as np

from tools.mlmc.detection import COCO91_ID_TO_NAME as ID_TO_NAME
from tools.mlmc.detection import Detections, ort_session


class Detector:
    def __init__(self, model, provider="cuda", score_thr=0.35):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.size = model.meta["artifacts"]["onnx"]["input_shape"][2]
        self.score_thr = score_thr

    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img = cv2.cvtColor(cv2.resize(frame_bgr, (self.size, self.size)),
                           cv2.COLOR_BGR2RGB)
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        return x, (w, h)

    def infer(self, x):
        return self.sess.run(None, {"image": x})

    def postprocess(self, out, wh):
        boxes, scores, labels = out
        m = scores > self.score_thr
        boxes, scores, labels = boxes[m], scores[m], labels[m]
        w, h = wh
        boxes = boxes * np.array([w, h, w, h], np.float32) / self.size
        return Detections(boxes.astype(np.float32), scores.astype(np.float32),
                          [ID_TO_NAME.get(int(l), str(l)) for l in labels])

    def __call__(self, frame_bgr):
        x, wh = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), wh)
