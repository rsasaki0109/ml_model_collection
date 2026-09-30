# Object detection

All models here take a BGR frame and return boxes in source-image pixels with
COCO class names (`tools/mlmc/detection.py: Detections`), so they can be run,
compared and benchmarked by the same tools.

![comparison](../assets/object_detection_comparison.gif)

## Comparison

<!-- BEGIN:object_detection_table -->
| Model | Code license | Weights license | Input | COCO mAP<br>(reported) | COCO mAP<br>(measured, ONNX) | Peak VRAM<br>(measured) | GTX 1660 Ti Laptop<br>onnxruntime-cuda FP32 · ms / FPS | Tesla T4 (Colab)<br>onnxruntime-cuda FP32 · ms / FPS | Tesla T4 (Colab)<br>onnxruntime-tensorrt FP16 · ms / FPS |
|---|---|---|---|---|---|---|---|---|---|
| [DEIM-D-FINE-S](deim_dfine_s) | 🟢 Apache-2.0 | 🟢 Apache-2.0* | 640×640 | [49.0](https://github.com/Intellindust-AI-Lab/DEIM/blob/09d35d53d39ee3145a1e61e3a989b28b9468d1dd/README.md) | **48.7** | 299 MB (Tiny, CUDA FP32)<br>335 MB (Tiny, CUDA FP32)<br>415 MB (Tiny, TRT FP16) | 21.7 / 46 | 16.7 / 60 | 6.2 / 162 |
| [D-FINE-N](dfine_n) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 640×640 | [42.8](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/README.md) | **42.6** | 177 MB (Tiny, CUDA FP32)<br>219 MB (Tiny, CUDA FP32)<br>405 MB (Tiny, TRT FP16) | 13.4 / 74 | 9.8 / 102 | 5.4 / 184 |
| [D-FINE-S](dfine_s) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 640×640 | [48.5](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/README.md) | **48.3** | 299 MB (Tiny, CUDA FP32)<br>335 MB (Tiny, CUDA FP32)<br>425 MB (Tiny, TRT FP16) | 22.3 / 45 | 17.3 / 58 | 6.8 / 148 |
| [LLMDet-T](llmdet_tiny) 🔤 ⚠️ | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 800×1333 | – | **1.5** | 5939 MB (Consumer, CUDA FP32)<br>6257 MB (Consumer, CUDA FP32)<br>1435 MB (Tiny, TRT FP16) | 2655.8 / 0 | 728.6 / 1 | 172.0 / 6 |
| [OWLv2-B/16](owlv2_b16) 🔤 | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 960×960 | – | **45.7** | 3477 MB (Light, CUDA FP32)<br>3519 MB (Light, CUDA FP32)<br>763 MB (Tiny, TRT FP16) | 660.1 / 2 | 510.2 / 2 | 84.5 / 12 |
| [RF-DETR-N](rfdetr_n) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 384×384 | [48.4](https://github.com/roboflow/rf-detr/blob/5f441831aaf23a68f40128ad0a2e27cb44e52640/README.md) | **47.9** | 355 MB (Tiny, CUDA FP32)<br>423 MB (Tiny, TRT FP16) | – | 13.2 / 76 | 3.7 / 270 |
| [RF-DETR-S](rfdetr_s) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 512×512 | [53.0](https://github.com/roboflow/rf-detr/blob/5f441831aaf23a68f40128ad0a2e27cb44e52640/README.md) | **52.6** | 489 MB (Tiny, CUDA FP32)<br>433 MB (Tiny, TRT FP16) | – | 24.7 / 40 | 5.6 / 179 |
| [RT-DETR-R18](rtdetr_r18vd) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 640×640 | [46.5](https://huggingface.co/PekingU/rtdetr_r18vd/blob/ac77a11ff0170a41b771c03264987f8ce2b0d753/README.md) | **46.2** | 361 MB (Tiny, CUDA FP32)<br>467 MB (Tiny, TRT FP16) | – | 23.1 / 43 | 6.7 / 149 |
| [RT-DETRv4-M](rtdetrv4_m) | 🟢 Apache-2.0 | 🟢 Apache-2.0* | 640×640 | [53.7](https://github.com/RT-DETRs/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/README.md) | **53.5** | 383 MB (Tiny, CUDA FP32)<br>441 MB (Tiny, TRT FP16) | – | 26.9 / 37 | 8.7 / 115 |
| [RT-DETRv4-S](rtdetrv4_s) | 🟢 Apache-2.0 | 🟢 Apache-2.0* | 640×640 | [49.8](https://github.com/RT-DETRs/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/README.md) | **49.6** | 335 MB (Tiny, CUDA FP32)<br>415 MB (Tiny, TRT FP16) | – | 16.8 / 59 | 6.8 / 148 |
| [SSDLite320-MobileNetV3](ssdlite320_mobilenet_v3_large) | 🟢 BSD-3-Clause | ⚪ unknown | 320×320 | [21.3](https://github.com/pytorch/vision/blob/6da25ff876100d36f23472f5762d5f306c47d735/torchvision/models/detection/ssdlite.py) | **21.1** | 249 MB (Tiny, CUDA FP32) | – | 20.9 / 48 | – |
| [YOLO11n](yolo11n) | 🟡 AGPL-3.0 | 🟡 AGPL-3.0* | 640×640 | [39.5](https://github.com/ultralytics/ultralytics/blob/50ca85e03bc669c694c474ab91168f9b5425d9a8/docs/en/models/yolo11.md) | **38.6** | 239 MB (Tiny, CUDA FP32)<br>403 MB (Tiny, TRT FP16) | – | 7.2 / 138 | 4.9 / 205 |
| [YOLO26n](yolo26n) | 🟡 AGPL-3.0 | 🟡 AGPL-3.0 | 640×640 | [40.1](https://github.com/ultralytics/ultralytics/blob/50ca85e03bc669c694c474ab91168f9b5425d9a8/README.md) | **40.0** | 247 MB (Tiny, CUDA FP32)<br>401 MB (Tiny, TRT FP16) | – | 8.3 / 121 | 4.5 / 223 |
| [YOLOX-S](yolox_s) | 🟢 Apache-2.0 | 🟢 Apache-2.0* | 640×640 | [40.5](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/README.md) | **40.3** | 249 MB (Tiny, CUDA FP32)<br>387 MB (Tiny, TRT FP16) | – | 11.2 / 89 | 5.4 / 184 |
<!-- END:object_detection_table -->

See the [top-level README](../README.md#object-detection) for how to read each column.

## Provenance

<!-- BEGIN:object_detection_provenance -->
| Model | Source (pinned) | Artifact | How it is produced |
|---|---|---|---|
| DEIM-D-FINE-S | [Intellindust-AI-Lab/DEIM@09d35d5](https://github.com/Intellindust-AI-Lab/DEIM/tree/09d35d53d39ee3145a1e61e3a989b28b9468d1dd) | `deim_dfine_s.onnx` | [`export.py`](deim_dfine_s/export.py) |
| D-FINE-N | [Peterande/D-FINE@956d170](https://github.com/Peterande/D-FINE/tree/956d1709314c2c6a4df6f34de232054578a7449f) | `dfine_n.onnx` | [`export.py`](dfine_n/export.py) |
| D-FINE-S | [Peterande/D-FINE@956d170](https://github.com/Peterande/D-FINE/tree/956d1709314c2c6a4df6f34de232054578a7449f) | `dfine_s.onnx` | [`export.py`](dfine_s/export.py) |
| LLMDet-T | [iSEE-Laboratory/LLMDet@5336624](https://github.com/iSEE-Laboratory/LLMDet/tree/53366243fba7) | `llmdet_tiny.onnx` | [`export.py`](llmdet_tiny/export.py) |
| OWLv2-B/16 | [google-research/scenic@8c113c5](https://github.com/google-research/scenic/tree/8c113c501c9f700b69899c55a69e65bb46727da6) | `owlv2_b16.onnx` | [`export.py`](owlv2_b16/export.py) |
| RF-DETR-N | [roboflow/rf-detr@5f44183](https://github.com/roboflow/rf-detr/tree/5f441831aaf23a68f40128ad0a2e27cb44e52640) | `rfdetr_n.onnx` | [`export.py`](rfdetr_n/export.py) |
| RF-DETR-S | [roboflow/rf-detr@5f44183](https://github.com/roboflow/rf-detr/tree/5f441831aaf23a68f40128ad0a2e27cb44e52640) | `rfdetr_s.onnx` | [`export.py`](rfdetr_s/export.py) |
| RT-DETR-R18 | [lyuwenyu/RT-DETR@29320b6](https://github.com/lyuwenyu/RT-DETR/tree/29320b6fd828f8e0987a71426cf2d961b09dfed7) | `rtdetr_r18vd.onnx` | [`export.py`](rtdetr_r18vd/export.py) |
| RT-DETRv4-M | [RT-DETRs/RT-DETRv4@55fefaa](https://github.com/RT-DETRs/RT-DETRv4/tree/55fefaaed7efe2a5f72d0a18fd4e05965e35c292) | `rtdetrv4_m.onnx` | [`export.py`](rtdetrv4_m/export.py) |
| RT-DETRv4-S | [RT-DETRs/RT-DETRv4@55fefaa](https://github.com/RT-DETRs/RT-DETRv4/tree/55fefaaed7efe2a5f72d0a18fd4e05965e35c292) | `rtdetrv4_s.onnx` | [`export.py`](rtdetrv4_s/export.py) |
| SSDLite320-MobileNetV3 | [pytorch/vision@6da25ff](https://github.com/pytorch/vision/tree/6da25ff876100d36f23472f5762d5f306c47d735) | `ssdlite320_mobilenet_v3_large.onnx` | [`export.py`](ssdlite320_mobilenet_v3_large/export.py) |
| YOLO11n | [ultralytics/ultralytics@50ca85e](https://github.com/ultralytics/ultralytics/tree/50ca85e03bc669c694c474ab91168f9b5425d9a8) | `yolo11n.onnx` | [`export.py`](yolo11n/export.py) |
| YOLO26n | [ultralytics/ultralytics@50ca85e](https://github.com/ultralytics/ultralytics/tree/50ca85e03bc669c694c474ab91168f9b5425d9a8) | `yolo26n.onnx` | [`export.py`](yolo26n/export.py) |
| YOLOX-S | [Megvii-BaseDetection/YOLOX@6ddff48](https://github.com/Megvii-BaseDetection/YOLOX/tree/6ddff4824372906469a7fae2dc3206c7aa4bbaee) | `yolox_s.onnx` | download `yolox_s.onnx` (sha256 pinned) |
<!-- END:object_detection_provenance -->

`weights/provenance.json` (created by `tools/fetch_model.py`) additionally
records the artifact's SHA-256 and the package versions used for export.
Exports are not guaranteed to be byte-identical across torch/onnx versions;
benchmark records therefore store the SHA-256 of the exact file measured.

## Per-model notes

Current real-time SOTA (see [docs/sota/object_detection.md](../docs/sota/object_detection.md)):

- **D-FINE-N / D-FINE-S** — exported from the Hugging Face port
  (`ustc-community/dfine-*-coco`, model card: apache-2.0) at a pinned
  revision. **COCO-only** weights on purpose: upstream warns that Objects365
  checkpoints should not be assumed commercially cleared. transformers
  computes the sin-cos position embedding in float64; the exporter wraps
  those Sin/Cos nodes in float32 casts (as in the original D-FINE code) so
  the graph also runs on ONNX Runtime's CPU provider. Parity with
  `transformers` post-processing checked on the demo clip (scores within
  ±0.002).
- **DEIM-D-FINE-S** — official DEIM repository + checkpoint (Google Drive,
  SHA-256 pinned). Same deploy graph as D-FINE.
- **RT-DETRv4-S / M** — official repository + checkpoints. Trained with a
  DINOv3 teacher; whether that affects the weights' license is unresolved
  upstream (noted in `model.yaml`).
- **RF-DETR-N / S** — exported with `rfdetr==1.11.0`. ~30M parameters even for
  N (DINOv2 ViT backbone); Objects365-pretrained. Pre/post-processing mirrors
  `rfdetr/export/_runtime`; parity with `RFDETR.predict()` checked on the
  demo clip.
- **YOLO26n** — Ultralytics export with the end-to-end (NMS-free) head
  (`nms=False`). AGPL-3.0.

Open-vocabulary (🔤; prompted with the 80 COCO class names by default):

- **OWLv2-B/16** — exported from `google/owlv2-base-patch16-ensemble`
  (Apache-2.0, explicit). The COCO prompts are tokenized at export time, so
  the default detector needs no tokenizer; custom prompts use the
  transformers CLIP tokenizer. Parity with transformers checked (image
  tensor ≤ 1.7e-4, probabilities ≤ 5e-5). COCO is not in its training
  data, so COCO AP is zero-shot. Scores are low in absolute terms, hence the
  0.1 default threshold.
- **LLMDet-T** — Grounding-DINO-style, exported from
  `iSEE-Laboratory/llmdet_tiny` (Apache-2.0) with a fixed 80-class prompt
  and static shapes (800×1333 canvas + pixel mask, 256 text tokens).
  `aten::isin/cummax/cummin` are replaced by exportable equivalents during
  export and a bool `EyeLike` is rewritten for ONNX Runtime; top detections
  match PyTorch (score ≤ 4e-4, box ≤ 0.003). COCO is part of its training
  data, so its COCO AP is **not** zero-shot. ⚠️ **Known issue:** measured COCO AP is 1.5 — boxes are
  badly localised by the transformers `MMGroundingDino` port itself (the official transformers pipeline
  gives the same boxes; Grounding-DINO-T in the same pipeline is correct). Kept for transparency; see
  `known_issue` in its `model.yaml`.

Earlier baselines, kept for comparison:

- **YOLOX-S** — official ONNX from the 0.1.1rc0 release (BGR 0–255, top-left
  letterbox).
- **RT-DETR-R18** — Hugging Face port (`PekingU/rtdetr_r18vd`).
- **YOLO11n** — Ultralytics export (`nms=False` on YOLO11 gives raw outputs;
  NMS in `detector.py`). AGPL-3.0.
- **SSDLite320-MobileNetV3** — torchvision export; the graph contains
  torchvision's transform and NMS, so benchmark latency includes them.
  torchvision does not state a license for its pretrained weights, hence
  *unknown*. Not shown in the GIF.

Shared pre/post-processing lives in `tools/mlmc/detectors/`
(`hf_detr.py`, `dfine_deploy.py`, `rfdetr.py`); a model's `detector.py`
re-exports one of them when its ONNX interface matches.

## Thresholds used for the demo

| Model family | score threshold | NMS |
|---|---|---|
| D-FINE, DEIM, RT-DETRv4, RT-DETR, RF-DETR | 0.5 | none (set prediction) |
| YOLO26n | 0.35 | none (end-to-end head) |
| YOLOX-S, YOLO11n | 0.35 | class-aware, IoU 0.45 |
| SSDLite320-MobileNetV3 | 0.35 | in-graph (torchvision default) |

Thresholds are defaults in each detector; they affect how busy the GIF
looks, not the benchmark numbers. The GIF line-up is set in
[`comparison.yaml`](comparison.yaml).
