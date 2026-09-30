"""DETR-style models exported from Hugging Face transformers.

Used by RT-DETR and D-FINE exports made with ``export_hf_detr`` below.
Graph: pixel_values [1,3,S,S] -> logits [1,Q,80], pred_boxes [1,Q,4]
(cxcywh, normalised). Pre-processing matches ``RTDetrImageProcessor``
defaults (also used by D-FINE): plain resize to S×S without keeping the
aspect ratio, RGB, rescale to [0, 1], no mean/std normalisation.
Post-processing is sigmoid + top-k over (query, class), NMS-free.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tools.mlmc.detection import COCO80, Detections, ort_session


class Detector:
    def __init__(self, model, provider="cuda", score_thr=0.5, top_k=300):
        self.sess = ort_session(model.artifact_path("onnx"), provider)
        self.size = model.meta["artifacts"]["onnx"]["input_shape"][2]
        self.score_thr, self.top_k = score_thr, top_k

    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img = cv2.resize(frame_bgr, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        return x, (w, h)

    def infer(self, x):
        return self.sess.run(None, {"pixel_values": x})

    def postprocess(self, out, wh):
        logits, boxes = out[0][0], out[1][0]
        prob = 1 / (1 + np.exp(-logits))                     # (Q, 80)
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


def export_hf_detr(repo_id: str, revision: str, out: Path, size: int = 640,
                   opset: int = 17):
    """Export a transformers ``AutoModelForObjectDetection`` DETR to ONNX."""
    import torch
    from transformers import AutoModelForObjectDetection

    model = AutoModelForObjectDetection.from_pretrained(repo_id, revision=revision).eval()
    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    if len(labels) != 80:
        raise SystemExit(f"expected 80 COCO classes, got {len(labels)}")

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            o = self.m(pixel_values=pixel_values)
            return o.logits, o.pred_boxes

    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(Wrapper(model), torch.rand(1, 3, size, size), str(out),
                          input_names=["pixel_values"],
                          output_names=["logits", "pred_boxes"],
                          opset_version=opset, dynamo=False)
    n = _float32_trig(out)
    print(f"wrote {out}" + (f" ({n} float64 Sin/Cos wrapped in float32 casts)" if n else ""))


def _float32_trig(path: Path) -> int:
    """Run float64 Sin/Cos in float32.

    transformers' D-FINE computes sin-cos position embeddings in float64,
    which ONNX Runtime's CPU provider does not implement for Sin/Cos. The
    original D-FINE code computes them in float32, so casting around these
    nodes restores upstream behaviour and CPU compatibility.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(path))
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {v.name: v.type.tensor_type.elem_type
             for v in list(inferred.graph.value_info) + list(inferred.graph.input)}
    nodes, count = [], 0
    for node in model.graph.node:
        if node.op_type in ("Sin", "Cos") and types.get(node.input[0]) == TensorProto.DOUBLE:
            x32, y32 = node.input[0] + "_f32_" + node.name, node.output[0] + "_f32"
            nodes.append(helper.make_node("Cast", [node.input[0]], [x32], to=TensorProto.FLOAT))
            nodes.append(helper.make_node(node.op_type, [x32], [y32], name=node.name))
            nodes.append(helper.make_node("Cast", [y32], [node.output[0]], to=TensorProto.DOUBLE))
            count += 1
        else:
            nodes.append(node)
    if count:
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        onnx.save(model, str(path))
    return count
