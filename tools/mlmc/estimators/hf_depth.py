"""Depth Anything V2 style models exported from Hugging Face transformers.

Pre-processing follows ``DPTImageProcessor`` as configured by the
Depth-Anything-V2 checkpoints (bicubic resize, ImageNet mean/std), but the
exported graph has a FIXED input size: both exporters bake the DPT head's
interpolation sizes into the graph, so a dynamic-shape graph silently
produces wrong outputs for other sizes. The artifact is exported at 518x924,
which is exactly what the upstream processor produces for 16:9 input
(keep_aspect_ratio, multiple of 14); other aspect ratios are stretched to
518x924. Output ``predicted_depth [1, 518, 924]`` is relative, affine-
invariant disparity (larger = nearer); it is resized back to the source size.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tools.mlmc.depth import DISPARITY, DepthMap
from tools.mlmc.detection import ort_session

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Estimator:
    def __init__(self, model, provider="cuda"):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        _, _, self.th, self.tw = model.meta["artifacts"]["onnx"]["input_shape"]

    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        th, tw = self.th, self.tw
        img = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (tw, th),
                         interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = ((img - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        return np.ascontiguousarray(x), (w, h)

    def infer(self, x):
        return self.sess.run(None, {"pixel_values": x})[0]

    def postprocess(self, out, wh):
        w, h = wh
        d = cv2.resize(out[0], (w, h), interpolation=cv2.INTER_LINEAR)
        return DepthMap(d.astype(np.float32), DISPARITY)

    def __call__(self, frame_bgr):
        x, wh = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(x), wh)


def export_hf_depth(repo_id: str, revision: str, out: Path, size=(518, 924), opset: int = 17):
    """Export a transformers depth-estimation model at a fixed input size."""
    import torch
    from transformers import AutoModelForDepthEstimation

    model = AutoModelForDepthEstimation.from_pretrained(repo_id, revision=revision).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).predicted_depth

    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(Wrapper(model), torch.rand(1, 3, *size), str(out),
                          input_names=["pixel_values"], output_names=["predicted_depth"],
                          opset_version=opset, dynamo=False)
    print(f"wrote {out}")
