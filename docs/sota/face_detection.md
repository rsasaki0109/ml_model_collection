# Face detection — state of the art survey

**Surveyed: 2026-10-01.** Sources: GitHub API (HEAD commits, license files),
upstream READMEs, model cards. AP = WIDER FACE val easy / medium / hard as
each source reports it (protocols differ: original size, VGA 640, multi-scale).
*unverified* marks items not confirmed.

| Model | Code | Weights | AP E / M / H | Params | Official ONNX | Landmarks |
|---|---|---|---|---|---|---|
| [YuNet](https://github.com/ShiqiYu/libfacedetection.train)-n / -s | BSD-3 at HEAD (MIT earlier); opencv_zoo dir MIT | in repo | n .892/.883/.811; s .887/.871/.768 (original size) | 76K / 55K | yes (fixed and dynamic) | 5 |
| [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) 10G / 2.5G / 0.5G (KPS) | MIT (README) | **non-commercial research only** | 95.40/94.01/82.80; 93.80/92.02/77.13; 90.97/88.44/69.49 (VGA) | 4.2M (10G) | in model packs | 5 |
| [RetinaFace (yakhyo)](https://github.com/yakhyo/retinaface-pytorch) MV2 / MV1-0.25 | MIT | release, no separate license | 94.04/92.26/83.59; 90.70/88.12/73.82 (original size) | 3.1M | yes | 5 |
| RetinaFace (biubug6) R50 / mnet0.25 | MIT | Google Drive, not stated | 95.48/94.04/84.43; 90.70/88.16/73.82 | – | export script | 5 |
| RetinaFace (official, InsightFace) | MIT | **non-commercial** | – | – | no | 5 |
| BlazeFace (MediaPipe) | Apache-2.0 | Apache-2.0 (model card) | not on WIDER (in-house data) | 128² / 192² input | TFLite only | 6 |
| YOLOv5-Face | **GPL-3.0** | Google Drive | v5n 93.61/91.52/80.53 | 1.7M | export script | 5 |
| YOLOv8-face / yolo-face (v6-v12, YOLO26) | **GPL-3.0** (Ultralytics AGPL base) | Drive / release | v8n 94.6/92.3/79.6 | – | some | 5 / none |
| Ultra-Light-Fast-Generic-1MB | MIT | in repo | RFB .855/.822/.579 (VGA) | ~1 MB | yes | none |
| CenterFace | MIT | in repo | .935/.924/.875 (multi-scale) | 1.9M | opset 9, fixed 32x32 input (needs re-export) | 5 |
| TinaFace | Apache-2.0 | Google Drive | .963/.957/.930 | 173 GFLOPs | no | none |
| DSFD | **academic research only** | – | 94.29/91.47/71.39 | 120M | no | none |
| DamoFD | ModelScope metadata: MIT | ModelScope | *unverified* | – | *unverified* | 5 |

No widely adopted new face detector with permissive weights appeared in
2025-2026 (search on GitHub / web); recent activity is GPL YOLO-face
variants and the BSD-3 relicensing of libfacedetection.train.

## License traps

1. **WIDER FACE is CC BY-NC-ND** — the training set of almost every
   detector above (BlazeFace excepted).
2. **InsightFace** (SCRFD, official RetinaFace, buffalo packs): pretrained
   models are "available for non-commercial research purposes only";
   ModelScope and UniFace label SCRFD as MIT, contradicting upstream.
3. **YuNet license drift**: MIT at earlier commits, BSD-3 at HEAD; pin the
   commit you copy from.
4. **YOLO-face family**: GPL-3.0 on top of AGPL Ultralytics.
5. **DSFD**: academic research only.

## Implications for this collection

| Model | Status | Why |
|---|---|---|
| YuNet-n | **added** | BSD-3, 76K params, official dynamic ONNX |
| RetinaFace-MV2 | **added** | MIT code, release ONNX, strong hard AP |
| SCRFD-10G | **added (flagged)** | accuracy reference; non-commercial weights |
| BlazeFace | candidate | only permissive weights not trained on WIDER; TFLite -> ONNX conversion needed |
| YOLO-face, DSFD | not planned | GPL / academic-only |

Evaluation: WIDER FACE val images (HF CUHK-CSE/wider_face `data/WIDER_val.zip`,
363 MB) + official `eval_tools.zip` ground truth.
