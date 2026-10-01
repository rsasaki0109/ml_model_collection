# Whole-body pose (COCO-WholeBody, 133 keypoints) — state of the art survey

**Surveyed: 2026-10-02.** Sources: mmpose 759b39c13fea6ba094afc1fa932f51dc1b11cbf9
configs / READMEs, DWPose / ViTPose / CIGPose READMEs, ONNX graphs inspected.
COCO-WholeBody v1.0 val, top-down with AP_H_56 person detections.

| Model | Input | Body / Foot / Face / Hand / **Whole** AP | Official ONNX | Training data |
|---|---|---|---|---|
| DWPose-t | 256x192 | 58.5 / 46.5 / 73.5 / 35.7 / **48.5** | yes | COCO-WB + UBody |
| DWPose-s | 256x192 | 63.3 / 53.3 / 77.6 / 42.7 / **53.8** | yes | COCO-WB + UBody |
| DWPose-m | 256x192 | 68.5 / 63.6 / 82.8 / 52.7 / **60.6** | yes | COCO-WB + UBody |
| DWPose-l | 384x288 | 72.2 / 70.4 / 88.7 / 62.1 / **66.5** | yes | COCO-WB + UBody |
| RTMW-m (real file `rtmw-dw-l-m`) | 256x192 | 67.6 / 67.1 / 78.3 / 49.1 / **58.2** | yes (unlisted) | Cocktail14 |
| RTMW-l | 384x288 | 76.1 / 79.3 / 88.4 / 66.3 / **70.1** | yes | Cocktail14 |
| RTMW-x | 384x288 | 76.3 / 79.6 / 88.4 / 66.4 / **70.2** | cocktail13 zip only | Cocktail14 |
| RTMPose-m / l / x (COCO-WB) | 256 / 384 | Whole 58.2 / 61.1 / 65.3 | no (pth) | COCO-WB |
| ViTPose++-S / B / L / H | 256x192 | Whole 54.4 / 57.4 / 60.6 / 61.2 | third-party, no license | 6 datasets |
| CIGPose (CVPR 2026) | 256 / 384 | Whole up to 67.5 | third-party bundle | COCO-WB (+UBody) |
| **Sapiens** | 1024x768 | 308 keypoints | TorchScript | **CC BY-NC 4.0** |
| **Sapiens2** | - | 308 keypoints | - | **Sapiens2 License (no biometric processing)** |

## License traps

- **COCO-WholeBody: research / non-commercial only** — every model above.
- Cocktail14 adds Human-Art and LaPa (non-commercial) and InterHand2.6M
  (CC BY-NC 4.0); several members unverified.
- Sapiens (CC BY-NC), Sapiens2 (use restrictions).
- mmpose's README RTMW-m link is mislabeled (s-width model); `pipeline.json`
  sizes in the RTMW archives are stale.

## Implications for this collection

| Model | Status | Why |
|---|---|---|
| DWPose-s / -m | **added** | small, official ONNX, same runner as RTMPose |
| RTMW-l-384 | **added** | best official ONNX (whole AP 70.1) |
| RTMPose-m Hand5 | candidate | hand-only 21 keypoints |
| Sapiens / Sapiens2 | not planned | license |
