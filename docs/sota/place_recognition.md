# Visual place recognition — state of the art survey

**Surveyed: 2026-10-02.** Sources: pinned repositories (licenses via the
GitHub API), papers (MegaLoc Table 1, EigenPlaces, SALAD, BoQ README).
Recall@1 as reported (MSLS-val numbers differ between papers).

| Model | Code | Weights | Training data | Dim / input | Pitts30k | MSLS-val | Tokyo24/7 |
|---|---|---|---|---|---|---|---|
| NetVLAD | MIT (MATLAB) | mirror, *unverified* | Pitts30k / TokyoTM | 4096 | 85.0 | 54.5 | 69.8 |
| pytorch-NetVlad | **no LICENSE** | - | Pitts30k | 4096 | - | - | - |
| CosPlace R50 | MIT | release (repo license) | SF-XL (research) | 2048 / native | 90.9 | 85.0 | 87.3 |
| EigenPlaces R50 | MIT | release (repo license) | SF-XL (research) | 2048 / native | 92.5 | 85.9 | 93.0 |
| MixVPR | **no LICENSE** | Google Drive | GSV-Cities (**CC BY-NC-ND**) | 4096 / 320 | 91.6 | 83.2 | 87.0 |
| SALAD | **GPL-3.0** | release | GSV-Cities | 8448 / 322 | 92.3 | 88.2 | 94.6 |
| CliqueMining | **GPL-3.0** | Google Drive | GSV-Cities + MSLS | 8448 / 322 | 92.6 | 91.6 | 96.8 |
| BoQ R50 / DINOv2 | MIT | release | GSV-Cities (**CC BY-NC-ND**) | 16384 / 12288 | 92.4 / 93.7 | 91.2 / 93.8 | - |
| AnyLoc | BSD-3 | DINOv2 ViT-G + VLAD centres | training-free | 49152 | 86.3 | 58.7 | 87.6 |
| **MegaLoc** | MIT | HF, mit | SF-XL, GSV-Cities, MSLS, MegaScenes, ScanNet | 8448 / 322 | **94.1** | 91.0 | **96.5** |

## License traps

- **Training data**: SF-XL (research-purpose form), GSV-Cities (Kaggle:
  CC BY-NC-ND 4.0), MSLS (login; non-commercial claimed, *unverified*),
  ScanNet (gated terms), Pittsburgh / Tokyo (Street View, *unverified*).
- **Code**: SALAD and CliqueMining GPL-3.0; MixVPR and pytorch-NetVlad
  without LICENSE.
- Redistributed benchmark copies (e.g. OpenVPRLab's msls-val zip) do not
  carry the datasets' rights.

## Implications for this collection

| Model | Status | Why |
|---|---|---|
| CosPlace / EigenPlaces R50-2048 | **added** | MIT code + weights, exportable at any resolution |
| MegaLoc | **added** | best reported recall, official ONNX |
| BoQ | candidate | MIT; GSV-Cities NC data |
| SALAD / CliqueMining | not planned | GPL-3.0 |
