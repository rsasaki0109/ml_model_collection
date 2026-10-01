# Multi-object tracking — state of the art survey

**Surveyed: 2026-10-01.** Sources: GitHub API (HEAD, licenses), READMEs and
papers. Numbers are MOT17 / MOT20 **test** (HOTA / MOTA / IDF1) with each
repository's MOT-trained private detector, DanceTrack test HOTA — not
comparable with a COCO detector. *unverified* marks items not confirmed.

| Tracker | Code | MOT17 | MOT20 | DanceTrack | ReID | Camera motion |
|---|---|---|---|---|---|---|
| SORT | **GPL-3.0** | – | – | – | no | no |
| DeepSORT | **GPL-3.0** | – | – | – | yes | no |
| ByteTrack (ECCV'22) | MIT | 63.1 / 80.3 / 77.3 | 61.3 / 77.8 / 75.2 | 47.7 | no | no |
| OC-SORT (CVPR'23) | MIT | 63.2 / 78.0 / 77.5 | 62.4 / 75.9 / 76.4 | 55.1 | no | no |
| Deep OC-SORT | MIT | 64.9 / 79.4 / 80.6 | 63.9 / 75.6 / 79.2 | 61.3 | yes | yes |
| BoT-SORT | MIT | 64.6 / 80.6 / 79.5 (ReID 65.0 / 80.5 / 80.2) | 62.6 / 77.7 / 76.3 | – | optional | yes |
| StrongSORT++ | **GPL-3.0** | 64.4 / 79.6 / 79.5 | 62.6 / 73.8 / 77.0 | 55.6 | yes (DukeMTMC-pretrained) | ECC |
| Hybrid-SORT (AAAI'24) | MIT | 63.6 / 79.3 / 78.4 | 62.5 / 76.4 / 76.2 | 62.2 (ReID 65.7) | optional | – |
| SparseTrack | MIT | 65.1 / 81.0 / 80.1 | 63.4 / 78.2 / 77.3 | 55.5 | no | yes |
| UCMCTrack (AAAI'24) | MIT | 65.8 / 80.6 / 81.0 | 62.8 / 75.6 / 77.4 | – | no | needs camera parameters |
| DiffMOT (CVPR'24) | MIT | 64.5 / 79.8 / 79.3 | 61.7 / 76.7 / 74.9 | 62.3 | yes | – |
| MOTR / MOTRv2 / CO-MOT (end-to-end) | MIT (+Apache) | MOTR 57.8 / 73.4 / 68.6 | – | 54.2 / 69.9 / 69.9 | – | – |
| MOTIP (CVPR'25), SambaMOTR (ICLR'25) | Apache-2.0 / MIT | – | – | 69.6 / 69.0 | – | – |

Libraries: roboflow/trackers (Apache-2.0, clean-room SORT / ByteTrack /
OC-SORT / BoT-SORT-CMC / C-BIoU + TrackEval port), norfair (BSD-3), motpy
(MIT); **boxmot and Ultralytics trackers are AGPL-3.0**. supervision's
ByteTrack moved to `trackers`.

## License traps

- **GPL-3.0**: SORT, DeepSORT, StrongSORT — and the Kalman filter in the MIT
  ByteTrack / BoT-SORT code derives from DeepSORT.
- **AGPL-3.0**: boxmot, Ultralytics trackers.
- **ReID data**: DukeMTMC was withdrawn (2019); MSMT17 is research-only
  (OSNet MSMT17 weights); FastReID SBS weights trained on MOT inherit
  CC BY-NC-SA.
- **Benchmarks**: MOTChallenge CC BY-NC-SA 3.0; DanceTrack non-commercial
  research only (annotations CC BY 4.0).

## Implications for this collection

| Tracker | Status | Why |
|---|---|---|
| ByteTrack, BoT-SORT (no ReID), OC-SORT | **added** (own clean implementation) | detector-agnostic, motion only, permissive |
| Deep OC-SORT / BoT-SORT + ReID | candidate | needs a ReID ONNX with clean training data |
| MOTIP / SambaMOTR | candidate | end-to-end; ONNX export *unverified* |
