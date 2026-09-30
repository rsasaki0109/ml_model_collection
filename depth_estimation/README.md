# Depth estimation

All models here take a BGR frame and return a `DepthMap`
(`tools/mlmc/depth.py`) at the source resolution, tagged with what the
numbers mean:

| `kind` | Meaning |
|---|---|
| `relative_disparity` | affine-invariant inverse depth — larger = nearer, no unit |
| `relative_depth` | affine-invariant depth — larger = farther, no unit |
| `metric_depth_m` | metres |

Relative outputs are **not** comparable across models or frames without
alignment; the comparison GIF normalises each frame independently.

![comparison](../assets/depth_estimation_comparison.gif)

## Comparison

<!-- BEGIN:depth_estimation_table -->
| Model | Code license | Weights license | Output | Input | NYUv2 AbsRel ↓<br>(reported) | Peak VRAM<br>(measured) |
|---|---|---|---|---|---|---|
| [Depth-Anything-V2-B](depth_anything_v2_base) | 🟢 Apache-2.0 | 🔴 CC-BY-NC-4.0 | relative (disparity) | 518×924 | [0.049](https://arxiv.org/abs/2406.09414) | not measured |
| [Depth-Anything-V2-S](depth_anything_v2_small) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | relative (disparity) | 518×924 | [0.053](https://arxiv.org/abs/2406.09414) | not measured |
<!-- END:depth_estimation_table -->

## Provenance

<!-- BEGIN:depth_estimation_provenance -->
| Model | Source (pinned) | Artifact | How it is produced |
|---|---|---|---|
| Depth-Anything-V2-B | [DepthAnything/Depth-Anything-V2@a561b84](https://github.com/DepthAnything/Depth-Anything-V2/tree/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf) | `depth_anything_v2_base.onnx` | [`export.py`](depth_anything_v2_base/export.py) |
| Depth-Anything-V2-S | [DepthAnything/Depth-Anything-V2@a561b84](https://github.com/DepthAnything/Depth-Anything-V2/tree/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf) | `depth_anything_v2_small.onnx` | [`export.py`](depth_anything_v2_small/export.py) |
<!-- END:depth_estimation_provenance -->

## Per-model notes

- **Depth-Anything-V2-S / -B** — exported from the Hugging Face ports at pinned
  revisions. Same code base (Apache-2.0), **different weight licenses**:
  Small is Apache-2.0, Base is CC-BY-NC-4.0 (non-commercial). Both exporters
  available in torch 2.11 bake the DPT head's interpolation sizes into the
  graph, so the artifacts are fixed at 518×924 — exactly what the upstream
  processor produces for 16:9 input; other aspect ratios are stretched.
  ONNX output equals the PyTorch model on the same input (relative
  difference 4e-7); against the full transformers pipeline (PIL bicubic vs
  OpenCV bicubic resize) the correlation is 0.99998.

See [docs/sota/depth_estimation.md](../docs/sota/depth_estimation.md) for how these
were chosen and which models are *not* included.
