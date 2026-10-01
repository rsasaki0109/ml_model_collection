# Pose estimation

Every model returns `Poses` (`tools/mlmc/pose.py`): per person, COCO-17
keypoints in source pixels with per-keypoint scores, plus a person box and
score.

![comparison](../assets/pose_estimation_comparison.gif)

## Comparison

<!-- BEGIN:pose_estimation_table -->
| Model | Kind | Code license | Weights license | Input | COCO kpt AP<br>(reported) | COCO kpt AP<br>(measured, ONNX) | Peak VRAM<br>(measured) | Tesla T4 (Colab)<br>onnxruntime-cuda FP32 · ms / FPS | Tesla T4 (Colab)<br>onnxruntime-tensorrt FP16 · ms / FPS |
|---|---|---|---|---|---|---|---|---|---|
| [RTMO-s](rtmo_s) | one-stage | 🟢 Apache-2.0 | 🟢 Apache-2.0* | 640×640 | [68.6](https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/projects/rtmo/README.md) | **67.9** | 269 MB (Tiny, T4 CUDA FP32)<br>1183 MB (Tiny, T4 TRT FP16) | 13.6 / 74<br>e2e 15.5 ms | 5.5 / 183<br>e2e 6.7 ms |
| [RTMPose-s](rtmpose_s) | top-down | 🟢 Apache-2.0 | 🟢 Apache-2.0* | 256×192 | [69.7](https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/projects/rtmpose/README.md) | **68.0** (persons from dfine_n) | 361 MB (Tiny, T4 CUDA FP32)<br>475 MB (Tiny, T4 TRT FP16) | 13.0 / 77 (14 crops)<br>e2e 56.5 ms | 6.3 / 158 (14 crops)<br>e2e 35.6 ms |
| [YOLO26n-pose](yolo26n_pose) | one-stage | 🟡 AGPL-3.0 | 🟡 AGPL-3.0 | 640×640 | [57.2](https://docs.ultralytics.com/tasks/pose/) | **57.0** | 237 MB (Tiny, T4 CUDA FP32)<br>401 MB (Tiny, T4 TRT FP16) | 10.0 / 100<br>e2e 11.3 ms | 4.5 / 225<br>e2e 6.2 ms |
<!-- END:pose_estimation_table -->

## Provenance

<!-- BEGIN:pose_estimation_provenance -->
| Model | Source (pinned) | Artifact | How it is produced |
|---|---|---|---|
| RTMO-s | [open-mmlab/mmpose@759b39c](https://github.com/open-mmlab/mmpose/tree/759b39c13fea6ba094afc1fa932f51dc1b11cbf9) | `rtmo_s.onnx` | download `rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip`, extract `end2end.onnx` (sha256 pinned) |
| RTMPose-s | [open-mmlab/mmpose@759b39c](https://github.com/open-mmlab/mmpose/tree/759b39c13fea6ba094afc1fa932f51dc1b11cbf9) | `rtmpose_s.onnx` | download `rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip`, extract `end2end.onnx` (sha256 pinned) |
| YOLO26n-pose | [ultralytics/ultralytics@50ca85e](https://github.com/ultralytics/ultralytics/tree/50ca85e03bc669c694c474ab91168f9b5425d9a8) | `yolo26n_pose.onnx` | download `yolo26n-pose.onnx` (sha256 pinned) |
<!-- END:pose_estimation_provenance -->

## Per-model notes

- **RTMPose-s** — mmpose's official ONNX SDK archive (archive and extracted
  `end2end.onnx` both SHA-256 pinned). Top-down: person boxes come from
  **D-FINE-N** (`person_detector` in `model.yaml`), so speed and accuracy
  include that detector. Crop / normalisation follow the mmpose config
  (RGB); the crop is pixel-identical to rtmlib's `top_down_affine` on a
  COCO test image. SimCC decode with split ratio 2.
- **RTMO-s** — mmpose's official ONNX SDK archive; one-stage, NMS inside the
  graph, BGR 0-255 input per the bundled `pipeline.json`.
- **YOLO26n-pose** — Ultralytics' release ONNX (AGPL-3.0 code and weights;
  the ONNX metadata embeds the license string). NMS-free end-to-end head.

On COCO image 785 (mmpose test data) the three models land within 4.8-5.7 px
mean keypoint error of the ground truth. On full COCO val2017 (T4, ONNX
Runtime CUDA) the measured AP is within 0.2 (YOLO26n-pose), 0.7 (RTMO-s) and
1.7 (RTMPose-s) of the reported values. RTMPose's number depends on the person
boxes: upstream evaluates with its own detection results, here the boxes come
from D-FINE-N.

The mmpose ONNX releases are **Body7** checkpoints (AIC, COCO, CrowdPose,
MPII, sub-JHMDB, Halpe, PoseTrack18). They score slightly lower on COCO than
the AIC+COCO checkpoints, which have no official ONNX; each dataset has its
own terms. See [docs/sota/pose_estimation.md](../docs/sota/pose_estimation.md).
