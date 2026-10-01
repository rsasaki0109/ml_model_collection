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
| Model | Code license | Weights license | Output | Input | NYUv2 AbsRel ↓<br>(reported) | NYUv2 AbsRel ↓ / δ1<br>(measured, ONNX, aligned) | Peak VRAM<br>(measured) | Tesla T4 (Colab)<br>onnxruntime-cuda FP32 · ms / FPS | Tesla T4 (Colab)<br>onnxruntime-tensorrt FP16 · ms / FPS |
|---|---|---|---|---|---|---|---|---|---|
| [Depth-Anything-3-S](depth_anything_3_small) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | relative (depth) | 280×504 | – | – | 507 MB (Tiny, T4 CUDA FP32)<br>447 MB (Tiny, T4 TRT FP16) | 22.1 / 45 | 5.2 / 192 |
| [Depth-Anything-V2-B](depth_anything_v2_base) | 🟢 Apache-2.0 | 🔴 CC-BY-NC-4.0 | relative (disparity) | 518×924 | [0.049](https://arxiv.org/abs/2406.09414) | – | 2099 MB (Light, T4 CUDA FP32)<br>697 MB (Tiny, T4 TRT FP16) | 297.3 / 3 | 48.1 / 21 |
| [Depth-Anything-V2-S](depth_anything_v2_small) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | relative (disparity) | 518×924 | [0.053](https://arxiv.org/abs/2406.09414) | – | 1013 MB (Tiny, T4 CUDA FP32)<br>495 MB (Tiny, T4 TRT FP16) | 119.4 / 8 | 18.9 / 53 |
| [MoGe-2-S](moge2_vits) | 🟢 MIT | 🟢 MIT | metric (m) | 720×1280 | – | – | 1503 MB (Tiny, T4 CUDA FP32) | 203.4 / 5 | – |
<!-- END:depth_estimation_table -->

## Provenance

<!-- BEGIN:depth_estimation_provenance -->
| Model | Source (pinned) | Artifact | How it is produced |
|---|---|---|---|
| Depth-Anything-3-S | [ByteDance-Seed/Depth-Anything-3@3d835ec](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4) | `depth_anything_3_small.onnx` | [`export.py`](depth_anything_3_small/export.py) |
| Depth-Anything-V2-B | [DepthAnything/Depth-Anything-V2@a561b84](https://github.com/DepthAnything/Depth-Anything-V2/tree/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf) | `depth_anything_v2_base.onnx` | [`export.py`](depth_anything_v2_base/export.py) |
| Depth-Anything-V2-S | [DepthAnything/Depth-Anything-V2@a561b84](https://github.com/DepthAnything/Depth-Anything-V2/tree/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf) | `depth_anything_v2_small.onnx` | [`export.py`](depth_anything_v2_small/export.py) |
| MoGe-2-S | [microsoft/MoGe@74fbce0](https://github.com/microsoft/MoGe/tree/74fbce054ebed49800de42d0ad0e83495065719a) | `moge2_vits.onnx` | download `model.onnx` (sha256 pinned) |
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

- **Depth-Anything-3-S** — exported from the official ByteDance-Seed code
  (pinned commit) and `depth-anything/DA3-SMALL` weights, not from the
  community ONNX (which does not state its source). Only the model code's
  imports are needed (omegaconf, einops, addict), not the full package.
  `torch.cartesian_prod` (RoPE grid) is replaced by an exportable
  equivalent. Pre-processing is identical to upstream `InputProcessor`
  (pixel difference 0.0); ONNX equals PyTorch (relative difference 4e-7).
  Outputs relative **depth** (larger = farther), unlike DA-V2's disparity.

- **MoGe-2-S** — the author's official ONNX (SHA-256 pinned), MIT. The graph
  is MoGe's raw forward pass; focal/shift recovery and metric scaling from
  `MoGeModel.infer()` are ported to NumPy/SciPy in `estimator.py`. Output is
  **metric depth in metres**, with sky/invalid pixels = inf. The ONNX
  equals upstream PyTorch in its `onnx_compatible_mode` (2e-6); that mode is
  ~2-3% off default PyTorch inference (see `model.yaml`).

See [docs/sota/depth_estimation.md](../docs/sota/depth_estimation.md) for how these
were chosen and which models are *not* included.
