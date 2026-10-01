# Dense optical flow — state of the art survey

**Surveyed: 2026-10-01.** Sources: upstream repositories at the listed
commit, Hugging Face API, papers. Numbers are upstream-reported; S-c / S-f =
Sintel **test** EPE clean / final, K15 = KITTI-15 test Fl-all (%), C+T+S+K+H
setting unless noted. *unverified* marks items not confirmed from a primary
source.

| Model | Venue | Code | Weights | S-c / S-f | K15 | Params | ONNX |
|---|---|---|---|---|---|---|---|
| [RAFT](https://github.com/princeton-vl/RAFT) | ECCV'20 | BSD-3 | Dropbox, no license stated | 1.61 / 2.86 | 5.10 | 5.3M | no; `grid_sample` on all-pairs volume |
| [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) S/M/L | ECCV'24 | BSD-3 | HF, bsd-3-clause | M 1.44/2.86, L 1.31/2.60 | M 4.64, L 4.30 | S 8.9M, M 19.7M | export feasible (**added**) |
| [NeuFlow v2](https://github.com/neufieldrobotics/NeuFlow_v2) | 2024 | Apache-2.0 | in repo | no test submission; Things-only Sintel-train 1.24/2.67 | KITTI-train Fl 15.3 | 9.0M | export feasible (**added**); community MIT ONNX exists |
| NeuFlow v1 | 2024 | Apache-2.0 | in repo | C+T Sintel-train 1.66/3.13 | – | – | blocked: custom CUDA correlation op |
| [GMFlow / UniMatch](https://github.com/autonomousvision/unimatch) | CVPR'22 / TPAMI'23 | MIT | S3, no license stated | 1.74/2.90; GMFlow+ 1.03/2.37 | 9.32; 4.49 | 4.7M / 7.4M | not checked |
| [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official) | ECCV'22 | Apache-2.0 | Google Drive, not stated | 1.16/2.09 | 4.68 | 16.2M | none found |
| FlowFormer++ | CVPR'23 | **no LICENSE file** | Google Drive | 1.07/1.94 | 4.52 | 16.2M | none found |
| VideoFlow (multi-frame) | ICCV'23 | **no LICENSE file** | Google Drive | 0.99/1.65 | 3.65 | 13.5M | none found |
| [MEMFOF](https://github.com/msu-video-group/memfof) (3-frame) | ICCV'25 | BSD-3 (GMA module copied from unlicensed VideoFlow) | HF, bsd-3-clause | 0.963/1.907 | 2.94 | 75.8M | none found |
| [WAFT](https://github.com/princeton-vl/WAFT) | ICLR'26 | BSD-3 | Google Drive, not stated; DINOv3 variant under Meta's DINOv3 license | 0.94/2.02 | 3.56 / 3.31 | 35-39M | none found |
| DPFlow | CVPR'25 | Apache-2.0 | **academic research only** | 1.04/1.97 | 3.56 | 10.0M | – |
| RAPIDFlow | ICRA'24 | Apache-2.0 | **academic research only** | *unverified* | KITTI-train Fl 17.7 | 1.65M | official converter ("not optimal") |
| FlowSeek | ICCV'25 | Apache-2.0 | Google Drive; M variant uses DAv2-Base (CC BY-NC 4.0) | zero-shot only | – | – | – |
| FlowIt | BMVC'26 | **CC BY-NC 4.0** | Google Drive | 0.85/1.84 | 3.59 | – | – |
| FreeFlow | ECCV'26 | **CC BY-NC-SA 4.0** | HF, cc-by-nc-sa-4.0 | L 0.68/1.48 | 3.23 | 35-231M | – |

## License traps

- **Training data**: every released checkpoint above has seen FlyingChairs
  and/or FlyingThings3D ("research purposes only ... Any commercial use is
  prohibited"); fine-tuned ones add KITTI (CC BY-NC-SA 3.0). TartanAir is
  CC BY 4.0; Spring CC BY 4.0 (per its B2FIND record, *unverified* on the
  project site); Sintel and HD1K state no license (*unverified*).
- **Academic-only weights under permissive code**: DPFlow, RAPIDFlow
  (ptlflow).
- **Non-commercial code**: FlowIt, FreeFlow.
- **Missing LICENSE**: FlowFormer++, VideoFlow (and MEMFOF's GMA module,
  copied from VideoFlow).
- **Backbone licenses**: WAFT-DINOv3 (DINOv3 license), FlowSeek-M (Depth
  Anything V2 Base, CC BY-NC 4.0).

## Export notes

RAFT-family models look up correlations with `F.grid_sample(align_corners=True)`
(opset ≥ 16; TensorRT supports GridSample). Fixed refinement iterations
unroll under tracing; padding is computed from the input shape, so one graph
per resolution.

## Implications for this collection

| Model | Status | Why |
|---|---|---|
| NeuFlow v2 (Things) | **added** | Apache, real-time |
| SEA-RAFT S / M (Tartan-C-T) | **added** | BSD-3 code and weights, strong zero-shot |
| MEMFOF | candidate | best accuracy with BSD-3 weights; 303 MB, 3-frame |
| DPFlow, RAPIDFlow, FlowIt, FreeFlow | not planned | non-commercial weights or code |

Evaluation: MPI-Sintel training split (`MPI-Sintel-training_images.zip`
1.8 GB + `MPI-Sintel-training_extras.zip` 3.3 GB from files.is.tue.mpg.de),
1041 pairs per pass; EPE over all pixels.
