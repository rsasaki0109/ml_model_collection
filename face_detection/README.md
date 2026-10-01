# Face detection

Every model returns `Faces` (`tools/mlmc/face.py`): boxes x1 y1 x2 y2,
scores and five landmarks (right eye, left eye, nose tip, right / left mouth
corner) in source pixels.

![comparison](../assets/face_detection_comparison.gif)

## Comparison

<!-- BEGIN:face_detection_table -->
| Model | Code license | Weights license | Input | WIDER FACE val AP E / M / H<br>(reported) | WIDER FACE val AP E / M / H<br>(measured, ONNX) | Peak VRAM<br>(measured) |
|---|---|---|---|---|---|---|
| [RetinaFace-MV2](retinaface_mv2) | 🟢 MIT | 🟢 MIT* | source size | [94.0 / 92.3 / 83.6](https://github.com/yakhyo/retinaface-pytorch/blob/4cd6e3471e5bac794637290a530566f463db4762/README.md) | – | not measured |
| [SCRFD-10G](scrfd_10g) | 🟢 MIT | 🔴 InsightFace-NC | 640×640 | [95.4 / 94.0 / 82.8](https://github.com/deepinsight/insightface/blob/1480e705287bc5d59f923b46c260ec6e3e4150f6/detection/scrfd/README.md) | – | not measured |
| [YuNet-n](yunet_n) | 🟢 BSD-3-Clause | 🟢 BSD-3-Clause* | source size | [89.2 / 88.3 / 81.1](https://github.com/ShiqiYu/libfacedetection.train/blob/02246e79b1e976c83d1e135a85e0628120c93769/README.md) | – | not measured |
<!-- END:face_detection_table -->

## Provenance

<!-- BEGIN:face_detection_provenance -->
| Model | Source (pinned) | Artifact | How it is produced |
|---|---|---|---|
| RetinaFace-MV2 | [yakhyo/retinaface-pytorch@4cd6e34](https://github.com/yakhyo/retinaface-pytorch/tree/4cd6e3471e5bac794637290a530566f463db4762) | `retinaface_mv2.onnx` | download `retinaface_mv2.onnx` (sha256 pinned) |
| SCRFD-10G | [deepinsight/insightface@1480e70](https://github.com/deepinsight/insightface/tree/1480e705287bc5d59f923b46c260ec6e3e4150f6) | `scrfd_10g.onnx` | download `buffalo_l.zip`, extract `det_10g.onnx` (sha256 pinned) |
| YuNet-n | [ShiqiYu/libfacedetection.train@dca340a](https://github.com/ShiqiYu/libfacedetection.train/tree/dca340aa082c71081a68d17db8e58b33a58a914b) | `yunet_n.onnx` | download `yunet_n_dynamic.onnx` (sha256 pinned) |
<!-- END:face_detection_provenance -->

## Per-model notes

Official / release ONNX files (SHA-256 pinned); decoding is reimplemented
from the upstream sources named in each `face.py`. On the demo clip the
three agree on face positions within a few pixels.

- **YuNet-n** — `onnx/yunet_n_dynamic.onnx` from libfacedetection.train
  (dynamic input). Decoding as OpenCV `FaceDetectorYN`: BGR 0-255 at the
  source size padded to /32, score = sqrt(cls * obj). NMS IoU 0.3 for
  rendering, 0.45 for WIDER FACE (as upstream).
- **RetinaFace-MV2** — yakhyo/retinaface-pytorch release ONNX (dynamic
  input). BGR minus (104, 117, 123), priors and variances as upstream.
- **SCRFD-10G** — `det_10g.onnx` from InsightFace `buffalo_l.zip` (archive
  and member SHA-256 pinned). The graph's output shapes are frozen for
  640x640, so inputs are letterboxed (top-left) to 640. **Non-commercial
  weights**; included as the accuracy reference.

Evaluation: WIDER FACE val (3226 images) with a NumPy port of the official
`evaluation.m` (score min-max normalisation over the set, IoU 0.5, faces
outside each setting ignored, 1000 thresholds, VOC AP); score threshold
0.02.

## Training data

| Dataset | Terms |
|---|---|
| WIDER FACE | CC BY-NC-ND (official page; HF mirror: CC BY-NC-ND 4.0) |
| RetinaFace / SCRFD landmark annotations | no license stated (unverified) |
