# Feature matching

Every model returns `Matches` (`tools/mlmc/matching.py`): matched keypoints
in the previous and current frame (source pixels) with LightGlue match
scores. Runners are stateful video runners.

![comparison](../assets/feature_matching_comparison.gif)

## Comparison

<!-- BEGIN:feature_matching_table -->
| Model | Code license | Weights license | Training data | HPatches H-AUC @1/3/5 px<br>(reported, DLT) | HPatches H-AUC @1/3/5 px<br>(measured, ONNX) | Peak VRAM<br>(measured) | Tesla T4 (Colab)<br>onnxruntime-cuda FP32 · ms / FPS | Tesla T4 (Colab)<br>onnxruntime-tensorrt FP16 · ms / FPS |
|---|---|---|---|---|---|---|---|---|
| [DISK+LightGlue](disk_lightglue) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | MegaDepth | – | **28.6** / **60.5** / **72.9** | 2723 MB (Light, T4 CUDA FP32)<br>1091 MB (Tiny, T4 TRT FP16) | 214.3 / 5 | 63.8 / 16 |
| [RaCo-ALIKED+LightGlue](raco_aliked_lightglue) | 🟢 Apache-2.0 | 🟢 Apache-2.0* | Oxford-Paris 1M distractors (synthetic homographies), MegaDepth | [40.4 / 71.1 / 80.5](https://github.com/cvg/RaCo/blob/35790eb48074ed14839d0fb496b8806caa4e766b/README.md) | **38.5** / **70.1** / **80.3** | 4355 MB (Consumer, T4 CUDA FP32)<br>993 MB (Tiny, T4 TRT FP16) | 1808.2 / 1 | 71.3 / 14 |
| [SuperPoint+LightGlue](superpoint_lightglue) | 🟢 Apache-2.0 | 🔴 MagicLeap-NC | MS-COCO 2014 + synthetic shapes (SuperPoint); MegaDepth (LightGlue) | [35.1 / 67.2 / 77.6](https://github.com/cvg/glue-factory/blob/2d17e3b3bd7d30f0c828d4c4d3eac4ecefbf283d/README.md) | **36.2** / **68.0** / **78.0** | 905 MB (Tiny, T4 CUDA FP32)<br>865 MB (Tiny, T4 TRT FP16) | 82.3 / 12 | 30.8 / 32 |
<!-- END:feature_matching_table -->

## Provenance

<!-- BEGIN:feature_matching_provenance -->
| Model | Source (pinned) | Artifact | How it is produced |
|---|---|---|---|
| DISK+LightGlue | [fabio-sim/LightGlue-ONNX@d12b4ba](https://github.com/fabio-sim/LightGlue-ONNX/tree/d12b4ba1632f558234e3f084e1f3d8bdf9147890) | `disk_lightglue.onnx` | download `disk_lightglue_pipeline.ort.onnx` (sha256 pinned) |
| RaCo-ALIKED+LightGlue | [fabio-sim/LightGlue-ONNX@d12b4ba](https://github.com/fabio-sim/LightGlue-ONNX/tree/d12b4ba1632f558234e3f084e1f3d8bdf9147890) | `raco_aliked_lightglue.onnx` | download `raco_aliked_lightglue_pipeline_k1024.onnx` (sha256 pinned) |
| SuperPoint+LightGlue | [fabio-sim/LightGlue-ONNX@d12b4ba](https://github.com/fabio-sim/LightGlue-ONNX/tree/d12b4ba1632f558234e3f084e1f3d8bdf9147890) | `superpoint_lightglue.onnx` | download `superpoint_lightglue_pipeline.ort.onnx` (sha256 pinned) |
<!-- END:feature_matching_provenance -->

## Per-model notes

All three are end-to-end graphs released by fabio-sim/LightGlue-ONNX
(extractor, top-1024 keypoints, LightGlue and match filtering inside one
ONNX file; SHA-256 pinned). Input: one pair `[2, C, H, W]`.

- **RaCo-ALIKED+LightGlue** (release v3.0, `k1024`) — RGB /255, sides
  multiples of 32; plain ONNX ops, so TensorRT can take the whole graph.
- **DISK+LightGlue** (release v2.0, `.ort.onnx`) — RGB /255, multiples of 16.
  fabio-sim's `DISKPreprocessor` passes OpenCV BGR unchanged; DISK and
  LightGlue's own loader use RGB, which is what this runner feeds. Contains
  `com.microsoft.MultiHeadAttention`, so the TensorRT EP runs those nodes on
  CUDA.
- **SuperPoint+LightGlue** (release v2.0, `.ort.onnx`) — grayscale
  (0.299 R + 0.587 G + 0.114 B) /255, multiples of 8. **Magic Leap
  non-commercial SuperPoint weights.**

A synthetic check (a demo frame warped by five random homographies) gives
sub-pixel DLT corner errors for all three.

## Evaluation

HPatches sequences (Hugging Face vbalnt/hpatches, MIT), glue-factory
protocol: image 1 vs. 2-6 of every scene except the 8 large ones, short side
480 (then rounded to the graph multiple and padded to a common size), DLT on
all matches weighted by score, mean corner error, AUC at 1 / 3 / 5 px.
