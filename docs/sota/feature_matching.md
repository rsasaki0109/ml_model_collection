# Local feature matching — state of the art survey

**Surveyed: 2026-10-01.** Sources: GitHub API (HEAD, licenses, release
assets), READMEs, ONNX graphs inspected for I/O. *unverified* marks items
not confirmed. MegaDepth-1500: pose AUC@5/10/20 (2048 kp, long side 1600);
HPatches: homography AUC@1/3/5 px (DLT, 1024 kp) — glue-factory / RaCo READMEs.

| Model | Code | Weights | Training data | MD1500 (OpenCV) | HPatches DLT | ONNX |
|---|---|---|---|---|---|---|
| SuperPoint + SuperGlue | **Magic Leap NC** | **NC** | COCO + synthetic | 48.7 / 65.6 / 79.0 | 32.1 / 65.0 / 75.7 | 3rd-party |
| SuperPoint + LightGlue | LightGlue Apache-2.0; SuperPoint **NC** | SP **NC**, LG Apache-2.0 | COCO; MegaDepth | 51.0 / 68.1 / 80.7 | 35.1 / 67.2 / 77.6 | fabio-sim |
| ALIKED + LightGlue | BSD-3 / Apache-2.0 | in repos | MegaDepth + R2D2 synthetic pairs | 52.3 / 68.8 / 81.0 | 33.8 / 66.1 / 76.6 | DeformConv rewrite needed |
| **RaCo + ALIKED + LightGlue+** (3DV 2026) | Apache-2.0 | release, project license | Oxford-Paris 1M synthetic pairs | 51.9 / 68.3 / 80.8 | **40.4 / 71.1 / 80.5** | fabio-sim v3.0 (TensorRT-friendly) |
| DISK + LightGlue | Apache-2.0 | Apache-2.0 | MegaDepth | – | – | fabio-sim |
| SIFT + LightGlue | Apache-2.0 | Apache-2.0 | – | 49.9 / 67.3 / 80.3 | – | – |
| XFeat (+ LighterGlue) (CVPR 2024) | Apache-2.0 | in repo | MegaDepth + COCO warps | sparse 42.6 / 56.4 / 67.7 (paper) | – | community |
| DeDoDe v1/v2 | MIT (DINOv2 Apache) | release | MegaDepth | – | – | community (MIT) |
| RDD (CVPR 2025) | Apache-2.0 | Google Drive (*unverified*) | MegaDepth (+ A2G) | 55.1 / 71.2 / 82.5 (+LG) | – | – |
| RoMa / RoMa v2 (dense) | MIT (DINOv3 custom for v2) | release | MegaDepth (+) | v2 62.8 / 76.8 / 86.5 | – | none; not real-time |
| LoFTR | Apache-2.0 | Google Drive (*unverified*); indoor = ScanNet NC | MegaDepth / ScanNet | – | – | – |
| **EfficientLoFTR** | **changed 2026-09-15 to a registration license** | Google Drive | MegaDepth | – | – | – |
| SiLK | **GPL-3.0** | – | – | – | – | – |
| R2D2 | **CC BY-NC-SA 3.0** | – | – | – | – | – |
| MASt3R | **CC BY-NC-SA 4.0** | – | – | – | – | – |
| DKM | MIT (MegaDepth models only; synthetic **NC**) | – | – | – | – | – |
| LiftFeat | **no LICENSE file** (README badge Apache) | – | MegaDepth + COCO | – | – | – |

## License traps

- **SuperPoint / SuperGlue**: Magic Leap noncommercial research only — this
  also covers LightGlue's bundled `superpoint.py`.
- **EfficientLoFTR** relicensed on 2026-09-15 (registration required for
  organisational use).
- **GPL / NC**: SiLK, R2D2, MASt3R, DKM synthetic models.
- **ScanNet** (indoor training / ScanNet-1500) requires signing its terms.
- MegaDepth: CC BY 4.0 for the reconstructions; the photos keep their own
  licenses.

## Implications for this collection

| Model | Status | Why |
|---|---|---|
| RaCo-ALIKED + LightGlue | **added** | permissive, best HPatches, TensorRT-friendly official ONNX |
| DISK + LightGlue | **added** | permissive, official ONNX |
| SuperPoint + LightGlue | **added (flagged)** | classic reference; non-commercial |
| XFeat | candidate | CPU real-time; needs export + host post-processing |
| RoMa v2, RDD | candidate | accuracy references (dense / weights hosting) |
