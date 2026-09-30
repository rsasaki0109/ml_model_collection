# Monocular depth estimation — state of the art survey

**Surveyed: 2026-09-30.** Sources: upstream READMEs / LICENSE files at HEAD,
Hugging Face API license tags and revisions, arXiv tables. Numbers are
**upstream-reported**. *unverified* marks items not confirmed from a primary
source.

## Summary

| Model | Output | Code | Weights (per variant) | Params | ONNX |
|---|---|---|---|---|---|
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (2406.09414) | relative disparity; metric variants | Apache-2.0 | **S: Apache-2.0; B/L/G: CC-BY-NC-4.0** (README) | S 24.8M · B 97.5M · L 335M | onnx-community, fabio-sim; exported here |
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (2511.10647) | any-view depth + pose; MONO / METRIC variants | Apache-2.0 | SMALL, BASE, METRIC-LARGE, MONO-LARGE: Apache-2.0; LARGE / GIANT / NESTED: CC-BY-NC-4.0 | 0.08B – 1.4B | onnx-community (no base_model / export script — provenance *unverified*) |
| [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) (2501.12375) | temporally consistent relative / metric | Apache-2.0 | S: Apache-2.0; B/L: CC-BY-NC-4.0 | L 382M | third-party only |
| [MoGe-2 / MoGe-3](https://github.com/microsoft/MoGe) (2507.02546) | metric point map, depth, normals, FOV | MIT | MIT (all) | S 35M – G 1.25B | official ONNX for MoGe-2 S/B/L |
| [Metric3D v2](https://github.com/YvanYin/Metric3D) (2404.15506) | metric depth + normals | BSD-2-Clause | **not stated** (HF repo untagged; asks for commercial inquiries) | S – G2 | onnx-community (tagged cc0 — conflicts with upstream) |
| [UniDepth v2](https://github.com/lpiccinelli-eth/UniDepth) (2502.20110) | metric depth + camera | **CC-BY-NC-4.0** | covered by the NC repo | S 34M – L 354M | *unverified* |
| [Depth Pro](https://github.com/apple/ml-depth-pro) (2410.02073) | metric depth + focal | Apple sample-code license | **research only** (apple-amlr) | 952M | onnx-community (tagged apple-ascl — conflicts) |
| [Marigold](https://github.com/prs-eth/Marigold) | affine-invariant depth (diffusion) | Apache-2.0 | README: RAIL++-M; v1-0 tagged Apache, v1-1 OpenRAIL++ | ~1B (SD2) *unverified* | none |
| [Lotus](https://github.com/EnVision-Research/Lotus) | disparity / depth (diffusion) | Apache-2.0 | Apache-2.0 | SD-based | none |
| [DepthCrafter](https://github.com/Tencent/DepthCrafter) | video relative depth | **academic/research only** (incl. weights) | same | 2.2B | none |
| [Distill-Any-Depth](https://github.com/Westlake-AGI-Lab/Distill-Any-Depth) | relative disparity | MIT | Apache-2.0 / MIT (inconsistent tags) | S–L | none |
| [DepthAnything-AC](https://github.com/HVision-NKU/DepthAnythingAC) | relative disparity (adverse conditions) | **CC-BY-NC-4.0** (README) | CC-BY-NC-4.0 (fine-tuned from Apache DA-V2-S) | ViT-S | none |
| [MiDaS v3.1](https://github.com/isl-org/MiDaS) / [ZoeDepth](https://github.com/isl-org/ZoeDepth) | relative / metric | MIT | MIT / Apache (HF) | – | legacy ONNX |
| [VGGT](https://github.com/facebookresearch/vggt) | multi-view depth + pose | VGGT License v1 | VGGT-1B: CC-BY-NC; VGGT-1B-Commercial: gated | 1B | none |

Selected upstream numbers (zero-shot, relative; AbsRel ↓ / δ1 ↑):
DA-V2-S NYUv2 0.053 / 0.973, KITTI 0.078 / 0.936; DA-V2-B NYUv2 0.049;
DA-V2-L NYUv2 0.045 / 0.979 (arXiv 2406.09414, Table 2).

## License traps

- **One family, split licenses:** DA-V2, Video-DA and DA3 publish small
  variants under Apache-2.0 and larger ones under CC-BY-NC-4.0 — the same
  code, the same wrapper, different rights.
- **Metadata conflicts:** HF tags on `DA3-LARGE-1.1` (Apache) and the DA-V2
  Metric Base/Large repos (Apache) contradict the upstream READMEs
  (CC-BY-NC). Record the upstream statement, not the tag.
- **Conversions re-tagging licenses:** `onnx-community/DepthPro-ONNX`
  (apple-ascl vs upstream research-only), `onnx-community/metric3d-*` (cc0 vs
  unstated upstream), `onnx-community/depth-anything-v3-large` (Apache vs
  CC-BY-NC source).
- **Fine-tunes more restrictive than the parent:** DepthAnything-AC is
  CC-BY-NC although derived from Apache DA-V2-S.
- **Code itself non-commercial:** UniDepth, DepthCrafter.

## Implications for this collection

| Model | Status | Why |
|---|---|---|
| Depth-Anything-V2-S | **added** | Apache-2.0 code + weights, 25M params, the baseline everyone compares to |
| Depth-Anything-V2-B | **added** | license-trap example: same code, CC-BY-NC-4.0 weights |
| Depth Anything 3 Small | **added** | newest (2025-11), Apache weights; exported from the official code (the community ONNX lacks provenance) |
| MoGe-2 ViT-S | **added** | MIT, metric geometry, official ONNX; focal/shift recovery ported from upstream |
| Metric3D v2, Depth Pro, UniDepth, DepthCrafter | not planned | weights license unstated, research-only or non-commercial |
