# Point tracking (TAP) — state of the art survey

**Surveyed: 2026-10-02.** Sources: pinned repositories, Hugging Face
revisions, papers (TAPNext Table 1, Track-On2 / Track-On-R, CoTracker3).
TAP-Vid DAVIS AJ / delta_avg / OA, 256x256 unless noted.

| Model | Code | Weights | Online | DAVIS first | DAVIS strided | Kinetics first |
|---|---|---|---|---|---|---|
| TAPIR | Apache-2.0 | Apache-2.0 | offline | 58.5 / 70.0 / 86.5 | 61.3 / 73.6 / 88.8 | 49.6 / 64.2 / 85.0 |
| Online TAPIR | Apache-2.0 | Apache-2.0 (JAX only) | causal | 56.2 / 70.0 / 86.5 | AJ 58.3 | AJ 51.2 |
| BootsTAPIR | Apache-2.0 | Apache-2.0 | offline | 61.4 / 74.0 / 88.4 | 66.2 / 78.5 / 90.7 | 54.6 / 68.4 / 86.5 |
| **Online BootsTAPIR** | Apache-2.0 | Apache-2.0 (JAX + PyTorch) | **causal** | 59.7 / 72.3 / 86.9 | AJ 61.2 | 55.1 / 67.5 / 86.3 |
| BootsTAPNext-B (ICCV 2025) | Apache-2.0 | Apache-2.0 | per-frame recurrent | 65.2 / 78.5 / 91.2 | 68.9 / 82.4 / 91.6 | 57.3 / 70.6 / 87.4 |
| TAPNext++ (CVPR 2026 Findings) | Apache-2.0 | Apache-2.0 (2.5 GB) | per-frame | 65.6 / 79.0 / 92.0 | 512 model: AJ 71.2 | 53.9 / 68.4 / 88.7 |
| Track-On2 / Track-On-R | MIT | MIT (DINOv3 backbone gated; DINOv2 variant) | causal (memory) | 67.0 / 79.9 / 92.0; R 68.1 / 80.3 / 92.5 (384x512) | - | 55.3 / 69.3 / 89.6 |
| **CoTracker3** | **CC BY-NC 4.0** | **CC BY-NC 4.0** | sliding window | online 63.8 / 76.3 / 90.2 (384x512) | - | 55.8 / 68.5 / 88.3 |
| LocoTrack | Apache-2.0 | Apache-2.0 | offline | 63.0 / 75.3 / 87.2 | 67.8 / 79.6 / 89.9 | 52.9 / 66.8 / 85.3 |
| AllTracker | MIT | MIT | windowed / dense | 63.3 / 76.3 / 90.0 | - | 56.8 / 69.3 / 89.1 |
| **TAPTR v1-v3** | **IDEA License (non-commercial)** | - | window | v3 63.2 / 76.7 / 91.0 | - | v3 54.5 / 67.5 / 88.2 |
| **DELTA** | **Snap non-commercial** | Google Drive | dense, offline | - | - | - |

## License traps

- **CoTracker 2/3** (code and weights CC BY-NC 4.0); Track-On-R uses
  CoTracker3 among its pseudo-label teachers (NC carry-over unclear).
- **TAPTR** (IDEA License), **DELTA** (Snap) non-commercial.
- Track-On2's DINOv3 backbone is gated under Meta's DINOv3 license.
- Training data: Kubric (no explicit data license found); BootsTAP's ~15M
  internet clips and CoTracker3's videos are not released.
- TAP-Vid annotations CC BY 4.0; DAVIS frames per creators (*unverified*).

## Export notes

TAPIR: per-frame global cost volume, PIPs-style iterative refinement with
causal-conv state, grid sampling everywhere. TAPNext: 12 recurrent blocks
with RG-LRU / conv1d caches; the number of queries must be fixed at export.
Track-On: mmcv multi-scale deformable attention (custom op).

## Implications for this collection

| Model | Status | Why |
|---|---|---|
| Online BootsTAPIR (4 / 1 iterations) | **added** | Apache code + weights, causal, exportable |
| BootsTAPNext-B | candidate | best permissive accuracy; recurrent-state export to do |
| Track-On2 (DINOv2) | candidate | MIT; deformable-attention op blocks ONNX |
| CoTracker3, TAPTR, DELTA | not planned | non-commercial |
