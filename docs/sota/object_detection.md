# Object detection — state of the art survey

**Surveyed: 2026-09-30.** Sources are upstream READMEs, LICENSE files at
HEAD, GitHub/HF APIs and arXiv. All numbers are **upstream-reported** (not
measured here). AP = COCO val2017 AP@[.5:.95] unless noted. "T4" = NVIDIA T4,
TensorRT, FP16, batch 1. Items that could not be confirmed from a primary
source are marked *unverified*.

This page decides what goes into the collection; it is not a leaderboard.
Licenses change — re-check before relying on this page (see
[License events](#license-events-2026)).

## 1. Real-time closed-set detection

### Best per size class

| Class | Best AP overall | License | Best permissive (code **and** weights) |
|---|---|---|---|
| Nano (≤5M) | DEIMv2-N 43.0 (3.6M) | non-commercial | DEIM-D-FINE-N 43.0 / D-FINE-N 42.8 (4M, 2.12 ms T4) — Apache-2.0 |
| Small (~10M) | ECDet-S 53.6 (O365) | non-commercial | RF-DETR-S 53.0 (32M!, 3.5 ms) · D-FINE-S obj2coco 50.7 · RT-DETRv4-S 49.8 (COCO only) |
| Medium (~20M) | ECDet-M 56.7 (O365) | non-commercial | D-FINE-M obj2coco 55.1 · RF-DETR-M 54.7 · RT-DETRv4-M 53.7 |
| Large (~30M) | ECDet-L 59.0 (O365) | non-commercial | D-FINE-L obj2coco 57.3 · RF-DETR-L 56.5 · RT-DETRv4-L 55.4 |
| XLarge | RF-DETR-2XL 60.1 (127M) | PML 1.0 | DEIM-D-FINE-X O365 59.5 · D-FINE-X obj2coco 59.3 · RT-DETRv4-X 57.0 |

AGPL reference (Ultralytics YOLO26, COCO only, T4): n 40.9 · s 48.6 · m 53.1 · l 55.0 · x 57.5.

### Families

| Family | Latest | Code | Weights | COCO AP range | Pretrain | ONNX |
|---|---|---|---|---|---|---|
| [D-FINE](https://github.com/Peterande/D-FINE) (ICLR'25) | HEAD `956d170` 2026-08 | Apache-2.0 | Apache-2.0 (repo); README: O365 checkpoints "should not be assumed to be commercially cleared" | N 42.8 … X 55.8 (COCO) / S 50.7 … X 59.3 (obj2coco) | COCO / O365 | official `export_onnx.py`; [onnx-community/dfine_*](https://huggingface.co/onnx-community) ; in transformers |
| [DEIM](https://github.com/Intellindust-AI-Lab/DEIM) (CVPR'25) | HEAD `09d35d5` | Apache-2.0 | Apache-2.0 (repo) | D-FINE arch: N 43.0 … X 56.5 | COCO (+X O365 59.5) | official `export_onnx.py` |
| [RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) (ECCV'26) | code 2025-11 | Apache-2.0 | Apache-2.0 (repo) | S 49.8 · M 53.7 · L 55.4 · X 57.0 | COCO (DINOv3 teacher at train time) | official `export_onnx.py` |
| [RF-DETR](https://github.com/roboflow/rf-detr) | `rfdetr` 1.11.0 (2026-09-24) | Apache-2.0 | N–L Apache-2.0 (explicit); **XL/2XL PML 1.0** | N 48.4 · S 53.0 · M 54.7 · L 56.5 · XL 58.6 · 2XL 60.1 | O365 → COCO, DINOv2 backbone | built-in exporter (ONNX/TRT/OpenVINO/TFLite) |
| [LW-DETR](https://github.com/Atten4Vis/LW-DETR) | dormant since 2025-02 | Apache-2.0 | Apache-2.0 | tiny 42.6 … xlarge 58.3 | O365 | scripts |
| [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR) | 2024-07 | Apache-2.0 | Apache-2.0 (HF explicit) | S 48.1 … X 54.3 | COCO / O365 | official; onnx-community |
| [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | HEAD `1d2ca42` | **non-commercial** (since 2026-08-24) | **non-commercial (explicit)** | Atto 23.8 … N 43.0 · S 50.9 · X 57.8 | COCO; DINOv3-based backbones | official |
| [EdgeCrafter / ECDet](https://github.com/Intellindust-AI-Lab/EdgeCrafter) (TMLR'26) | 2026-03 | **non-commercial** (since 2026-08) | non-commercial; weight host repo still says Apache-2.0 (conflict) | S 51.7/53.6 … X 57.9/59.9 (COCO/O365) | COCO / O365 | official |
| [Ultralytics YOLO26 / YOLO11](https://github.com/ultralytics/ultralytics) | 8.4.166 | AGPL-3.0 | AGPL-3.0 / Enterprise | YOLO26 n 40.9 … x 57.5 | COCO | built-in |
| [YOLOv12](https://github.com/sunsmarterjie/yolov12) / [YOLOv13](https://github.com/iMoonLab/yolov13) / [YOLOv10](https://github.com/THU-MIG/yolov10) / [YOLO-Master](https://github.com/Tencent/YOLO-Master) | 2024–2026 | AGPL-3.0 | AGPL-3.0 | v12 n 40.4 … x 55.4; v13 N 41.6 … X 54.8; Master-N 42.4 | COCO | via Ultralytics |
| [YOLOv9](https://github.com/WongKinYiu/yolov9), [Gold-YOLO](https://github.com/huawei-noah/Efficient-Computing), MMYOLO | — | GPL-3.0 | GPL-3.0 | ≤ 55.6 | COCO | varies |
| [YOLO-NAS](https://github.com/Deci-AI/super-gradients) | 2024-04 | Apache-2.0 | **non-commercial, no modification** | S 47.5 … L 52.2 | *unverified* O365 | yes |
| [RTMDet](https://github.com/open-mmlab/mmdetection) (mmdetection) | dormant | Apache-2.0 | Apache-2.0 (repo) | tiny 41.1 … x 52.8 | COCO | mmdeploy |
| [DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO), [PP-YOLOE+](https://github.com/PaddlePaddle/PaddleDetection), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | 2021–2023 | Apache-2.0 | Apache-2.0 (repo) | YOLOX-s 40.5 · DAMO-L 51.9 · PP-YOLOE+ x 54.7 | COCO / O365 (PP-YOLOE+) | official |

Notes

- RF-DETR "N" is **30.5M params** (DINOv2 ViT); it is fast on a T4 but not
  small. Its comparison table re-measures others in its own harness and
  compares against D-FINE's *obj2coco* weights.
- RT-DETRv4 params/GFLOPs are not stated; configs include D-FINE's HGNetv2
  configs, so the deployed architecture is presumably D-FINE's (*inferred*).
- Latency hardware is not stated for DEIMv2, YOLOv13, YOLOv10, LW-DETR
  (*unverified*).

## 2. Highest-accuracy closed-set detection (non-real-time)

| Model | COCO AP | Size | Code | Weights |
|---|---|---|---|---|
| DINOv3 ViT-7B + Plain-DETR (Meta, 2025-08) | 65.6 val / 66.1 TTA | 7B frozen + 100M | DINOv3 License | gated, DINOv3 License |
| [PE-spatial-G + DETA](https://github.com/facebookresearch/perception_models) (Meta, 2025-04) | 65.2 val / 66.0 TTA | ~1.9B | Apache-2.0 | **Apache-2.0, ungated** ([facebook/PE-Detection](https://huggingface.co/facebook/PE-Detection)) |
| [Co-DINO ViT-L](https://github.com/Sense-X/Co-DETR) (ICCV'23) | 65.9 val / 66.0 test-dev | 304M backbone | MIT | HF tag apache-2.0 |
| CB-InternImage-H + DINO | 64.5 val | 2.18B | MIT | MIT (G variant unreleased) |
| EVA-02-L Cascade | 64.1 val | ~300M | MIT | apache-2.0 (HF) |

No 2026 result above 66.1 COCO val was found. None of these has an official
ONNX export or published inference VRAM; all are implausible on a 6 GB GPU
by parameter count alone (*inferred*, not measured).

## 3. Open-vocabulary / promptable detection

| Model | COCO zero-shot AP | LVIS minival AP | Code | Weights |
|---|---|---|---|---|
| DINO-X Pro (IDEA) | 56.0 | 59.8 | SDK only | **API only** |
| Grounding DINO 1.5 / 1.6 Pro, T-Rex2 | 54.3–55.4 / 52.2 | 55.7–57.7 / 54.9 | SDK only | **API only** |
| [SAM 3 / 3.1](https://github.com/facebookresearch/sam3) (Meta, 2025-11 / 2026-03) | 56.4 (box) | LVIS 53.6 box (split not stated) | SAM License | gated, SAM License (custom; commercial allowed, ITAR/military clauses) |
| [WeDetect-L/B/T](https://github.com/WeChatCV/WeDetect) (CVPR'26) | 54.5 / 52.1 / 44.9 | 55.0 / 47.3 / 37.4 | **no LICENSE file** | HF tag GPL-3.0; Chinese class names |
| [LLMDet](https://github.com/iSEE-Laboratory/LLMDet) L/B/T (CVPR'25) | COCO seen in training | 51.1 / 48.3 / 44.7 | Apache-2.0 | **Apache-2.0**, in transformers |
| [MM-Grounding-DINO](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino) T/B/L | 50.4 / 52.5 / 53.0 | T 41.4 | Apache-2.0 | Apache-2.0 |
| [OWLv2](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) L/14 | – | 44.6 (ens.) | Apache-2.0 | Apache-2.0 (explicit); onnx-community exports |
| [OmDet-Turbo-Tiny](https://github.com/om-ai-lab/OmDet) | 42.5 | – | Apache-2.0 | Apache-2.0; official ONNX export |
| [YOLOE](https://github.com/THU-MIG/yoloe) / YOLOE-26 | – | 35.9 (v8-L) / 40.6 (26x) | AGPL-3.0 | AGPL-3.0 |
| [YOLO-World v2](https://github.com/AILab-CVC/YOLO-World) | – | 35.4 | GPL-3.0 | GPL-3.0 |
| [Florence-2-L](https://huggingface.co/microsoft/Florence-2-large) | 37.5 | – | MIT | MIT |
| [Rex-Omni](https://github.com/IDEA-Research/Rex-Omni) | reports F1, not AP | – | IDEA License (**non-commercial**) | + Qwen Research License |

Zero-shot caveats: LLMDet and MM-GDINO `*_all` include COCO in training;
OWLv2 "ST+FT" is fine-tuned on LVIS base classes.

## License events (2026)

| Date | Project | Change |
|---|---|---|
| 2026-08-19 | D-FINE | README adds: Objects365 checkpoints "should not be assumed to be commercially cleared" |
| 2026-08-20/24 | EdgeCrafter | Apache-2.0 → non-commercial (code + weights) |
| 2026-08-24 | DEIMv2 | Apache-2.0 → non-commercial, weights explicitly included (commit `bb64e5e`). History: GPL-3.0 at initial commit `6de930b` (2025-09-19) → Apache-2.0 at `3e491c6` (2025-09-25) → non-commercial |

Recurring traps: Objects365 pretraining (RF-DETR all sizes, LW-DETR,
obj2coco weights, PP-YOLOE+); DINOv3 License on DINOv3-derived backbones
(whether distillation from a DINOv3 teacher makes weights "derivative" is
**unresolved**); PML 1.0 for RF-DETR XL/2XL; YOLO-NAS weights; API-only
IDEA models.

## Implications for this collection

The initial line-up (YOLOX-S 2021, SSDLite 2019, RT-DETR-R18 2023, YOLO11n
2024) was a working baseline but not current SOTA. Status after this survey:

| Model | Status | Why | License here |
|---|---|---|---|
| D-FINE-N, D-FINE-S (COCO-only) | **added** | best permissive accuracy per parameter | Apache-2.0 (explicit, HF card) |
| DEIM-D-FINE-S | **added** | +0.5 AP over D-FINE-S, same deploy graph | Apache-2.0* |
| RT-DETRv4-S, RT-DETRv4-M | **added** | newest permissive real-time DETR, COCO only | Apache-2.0* (DINOv3-teacher question noted) |
| RF-DETR-N, RF-DETR-S | **added** | strongest permissive AP at a given T4 latency; ~30M params | Apache-2.0 (explicit); O365 pretrain noted |
| YOLO26n | **added** | current AGPL reference, NMS-free | AGPL-3.0 |
| RT-DETR-R18, YOLOX-S, YOLO11n | kept | older baselines for CPU/edge comparison | – |
| SSDLite320 | kept, **removed from GIF** | weights license unknown, low accuracy | ⚪ unknown |
| LLMDet-T, OWLv2-B | next | first open-vocabulary entries; need a prompt-aware detector interface | Apache-2.0 (explicit) |

Not planned: DEIMv2, ECDet, YOLO-NAS, RF-DETR XL/2XL (non-commercial / PML);
API-only models; ≥1B-param accuracy leaders (do not fit the collection's
hardware focus; may be listed as references later).
