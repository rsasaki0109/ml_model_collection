# Segmentation — state of the art survey

**Surveyed: 2026-10-01.** Sources: upstream READMEs / LICENSE files at HEAD,
Hugging Face API license tags and revisions, model-zoo tables. Numbers are
**upstream-reported**; *unverified* marks items not confirmed from a primary
source.

## Real-time instance segmentation (COCO val2017 mask AP)

| Model | Code | Weights | Mask AP | Params | Input | ONNX |
|---|---|---|---|---|---|---|
| [RF-DETR-Seg](https://github.com/roboflow/rf-detr) N / S / M | Apache-2.0 | Apache-2.0 (README table) | 40.3 / 43.1 / 45.3 | 33.6 / 33.7 / 35.7 M | 312 / 384 / 432 | official exporter |
| YOLO26-seg n / s | AGPL-3.0 | AGPL-3.0 | 33.9 / 40.0 | 2.7 / 10.4 M | 640 | ultralytics |
| YOLO11-seg n / s | AGPL-3.0 | AGPL-3.0 | 32.0 / 37.8 | 2.9 / 10.1 M | 640 | ultralytics |
| [ECSeg](https://github.com/Intellindust-AI-Lab/EdgeCrafter) S / M | **non-commercial** | **non-commercial** (explicit) | 43.0 / 45.2 (COCO only) | 10 / 20 M | 640 | *unverified* |
| RTMDet-Ins (mmdet) tiny / s | Apache-2.0 | Apache-2.0 | 35.4 / 38.7 | 5.6 / 10.2 M | 640 | mmdeploy (*unverified*) |
| Mask2Former Swin-T (COCO instance) | MIT (majority) | HF tag "other" | 45.0 | ~47 M | ~1024 | see below |
| EoMT-L (DINOv2) | MIT | MIT | 45.2 @640 | ~300 M | 640 | *unverified* |

## Semantic segmentation

| Model | Code | Weights | mIoU | Params | ONNX |
|---|---|---|---|---|---|
| [Mask2Former](https://github.com/facebookresearch/Mask2Former) Swin-T ADE | MIT (majority) | HF tag "other", points to MIT | ADE20K 47.7 | ~48 M | exported here (GridSample cast fix) |
| [SegFormer](https://github.com/NVlabs/SegFormer)-B0 ADE | **NVIDIA SCL, non-commercial** | same | ADE20K 37.4 | 3.7 M | exported here; Xenova ONNX re-upload has no license tag |
| EoMT-S / -L | MIT | MIT (DINOv2 variants) | ADE 58.4 (L @512) | 24 M / ~300 M | *unverified* |
| PIDNet-S / DDRNet-23-slim (mmseg) | MIT / Apache-2.0 | openmmlab-hosted | Cityscapes 78.7 / 77.8 | small | *unverified* |
| PP-LiteSeg-T/B | Apache-2.0 | Apache-2.0 | Cityscapes 73–79 | small | paddle2onnx (*unverified*) |
| YOLO26-sem | AGPL-3.0 | AGPL-3.0 | ADE n 38.8 / s 45.6 | 1.6 / 6.5 M | ultralytics |

## Promptable segmentation

| Model | Code | Weights | ONNX |
|---|---|---|---|
| [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) | Apache-2.0 | Apache-2.0 | `onnx-community/EdgeTAM-ONNX` (encoder + prompt decoder) |
| [SAM 2.1](https://github.com/facebookresearch/sam2) hiera-tiny / small | Apache-2.0 | Apache-2.0 | `onnx-community/sam2.1-hiera-*-ONNX` (no license tag on the ONNX repos) |
| [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit) L0 | Apache-2.0 | Apache-2.0 | official encoder / decoder ONNX |
| EfficientSAM ti / s | Apache-2.0 | Apache-2.0 | official ONNX |
| MobileSAM | Apache-2.0 | HF card says MIT (conflict) | third-party |
| EdgeSAM | **S-Lab License, non-commercial** | untagged | official ONNX |
| SAM 3 / 3.1 | **SAM License** (custom) | gated | `onnx-community/sam3-tracker-ONNX` is ungated and untagged |

## License traps

- **SegFormer** — NVIDIA non-commercial on code and weights; ONNX
  re-uploads and mmsegmentation's Apache repository do not restate it.
- **EdgeCrafter / ECSeg** — project license (non-commercial, explicitly
  covers weights) vs. an Apache-2.0 weight-hosting repo vs. untagged HF repos.
- **SAM 3** — custom license with trade-control / military clauses and a
  redistribution notice requirement; ungated ONNX re-uploads drop it.
- **EoMT DINOv3 variants** — tagged MIT but derived from DINOv3 (DINOv3
  License); DINOv2-based EoMT is clean.
- **EdgeSAM** — non-commercial, untagged on the Hub.
- **Ultralytics seg / sem / YOLOE** — AGPL-3.0 code and weights.

## Implications for this collection

| Model | Status | Why |
|---|---|---|
| RF-DETR-Seg-N | **added** | strongest permissive real-time instance segmentation, official exporter |
| Mask2Former Swin-T ADE | **added** | permissive (MIT) semantic segmentation baseline |
| SegFormer-B0 ADE | **added (non-commercial)** | license-trap example |
| EdgeTAM / SAM 2.1-tiny | next | Apache promptable segmentation; needs a point/box prompt interface |
| ECSeg, EdgeSAM, SAM 3 | not planned | non-commercial / custom licenses |
