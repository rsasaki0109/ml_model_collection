# Human pose estimation (2D, COCO-17) — state of the art survey

**Surveyed: 2026-10-01.** Sources: upstream READMEs / LICENSE files at
HEAD, Hugging Face API license tags and revisions, model-zoo tables.
Numbers are **upstream-reported** COCO val2017 keypoint AP (top-down models
with detector boxes); *unverified* marks items not confirmed from a primary
source.

| Model | Type | Input | Params | COCO AP | Training data | Code / weights | Official ONNX |
|---|---|---|---|---|---|---|---|
| [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) t/s/m/l (AIC+COCO) | top-down, SimCC | 256x192 | 3.3/5.5/13.6/27.7 M | 68.5/72.2/75.8/76.5 | AIC+COCO | Apache-2.0 / repo license | no (.pth) |
| RTMPose t/s/m/l (Body7) | top-down, SimCC | 256x192 | same | 65.9/69.7/74.9/76.7 | 7 datasets | Apache-2.0 / repo license | **yes** (onnx_sdk zips) |
| [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo) t/s/m/l | one-stage | 416/640 | –/9.9/22.6/44.8 M | Body7 57.4/68.6/72.6/74.8 (COCO-only s/m/l 67.7/70.9/72.4) | Body7 or COCO | Apache-2.0 / repo license | **yes** (Body7) |
| [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) S/B/L/H | top-down heatmap | 256x192 | ViT | 73.8/75.8/78.3/79.1 | COCO | Apache-2.0 / OneDrive, unstated | no (third-party easy_ViTPose, untagged) |
| ViTPose++ S/B/L/H | top-down, MoE | 256x192 | | 75.8/77.0/78.6/79.4 | 6 datasets | Apache-2.0 (HF tag) | no |
| [DWPose](https://github.com/IDEA-Research/DWPose) / RTMW | top-down whole-body (133 kpt) | 256x192 / 384x288 | | Whole AP 48.5–66.5 / 58.2–70.2 | COCO-WholeBody + UBody / 14 datasets | Apache-2.0 | yes |
| [DETRPose](https://github.com/SebastianJanampa/DETRPose) N/S/M/L/X | one-stage DETR | 640 | 4.1/11.5/20.8/32.8/73.3 M | 57.2/67.0/69.4/72.5/73.3 | COCO | Apache-2.0 (HF tag) | export script |
| [RF-DETR keypoint preview](https://github.com/roboflow/rf-detr) | one-stage DETR | 576 | 40.7 M | 71.8 | COCO | Apache-2.0, **preview** (weights may change) | via rfdetr |
| YOLO11-pose n..x | one-stage + NMS | 640 | 2.9–58.8 M | 50.0–69.5 | COCO | **AGPL-3.0** | yes |
| YOLO26-pose n..x | one-stage, NMS-free | 640 | 2.9–57.6 M | 57.2–71.6 | COCO | **AGPL-3.0** | yes |
| [ECPose](https://github.com/Intellindust-AI-Lab/EdgeCrafter) S/M/L/X | one-stage DETR | 640 | 10–51 M | 68.9–74.8 (O365 69.7–75.9) | COCO (/O365) | **non-commercial** | *unverified* |
| [ED-Pose](https://github.com/IDEA-Research/ED-Pose) | one-stage DETR | | | 71.6 / 74.3 | COCO | **IDEA License, non-commercial** | no |
| [GroupPose](https://github.com/Michel-liu/GroupPose) | one-stage DETR | | | 72.0–74.8 | COCO | Apache-2.0 | no |
| HRNet-W32/W48 | top-down heatmap | 256x192 | 28.5/63.6 M | 74.4/75.1 | COCO | MIT code / weights unstated | no |
| SimCC (original repo) | top-down | | | 75.4 (W48) | COCO | **no license** | no |
| [Sapiens](https://github.com/facebookresearch/sapiens) 0.3b–1b | top-down, 308 kpt | 1024x768 | 0.3–1 B | not COCO-17 | Meta internal | **CC-BY-NC-4.0** | TorchScript |
| Sapiens2 | top-down, 308 kpt | | 0.4–5 B | – | Meta | **Sapiens2 License** (use restrictions) | no |

## License traps

- **AGPL-3.0**: Ultralytics YOLO11/26-pose code and weights (embedded in the
  ONNX metadata).
- **Non-commercial**: ECPose / EdgeCrafter (weights hosted in a separately
  licensed repo), ED-Pose (IDEA License), Sapiens v1 (CC-BY-NC-4.0).
- **Sapiens2 License**: commercial use allowed but bans surveillance and
  biometric processing; redistribution must carry the license.
- **No / unstated license**: SimCC repo; HRNet weights (MIT covers code
  only); openmmlab and ViTPose checkpoints are only implicitly covered by the
  repository license; easy_ViTPose ONNX is untagged.
- **Training data mixtures**: Body7 / Cocktail14 / ViTPose++ combine several
  datasets with their own terms — the official mmpose ONNX files are exactly
  these mixed-data checkpoints.
- **Preview churn**: RF-DETR keypoint weights may change before release.

## Implications for this collection

| Model | Status | Why |
|---|---|---|
| RTMPose-s (Body7) | **added** | permissive, small, official ONNX; top-down with D-FINE-N boxes |
| RTMO-s (Body7) | **added** | permissive one-stage real-time, official ONNX with NMS |
| YOLO26n-pose | **added** | AGPL reference, NMS-free |
| DETRPose-S | next | Apache, COCO-only one-stage DETR; needs export from the official code |
| DWPose / RTMW | later | whole-body (133 keypoints) |
| ECPose, ED-Pose, Sapiens | not planned | non-commercial / restricted |
