# Background removal / matting

Every model returns `AlphaMatte` (`tools/mlmc/matting.py`): foreground
opacity `(H, W)` in [0, 1] at the source resolution.

![comparison](../assets/background_removal_comparison.gif)

## Comparison

<!-- BEGIN:background_removal_table -->
| Model | Kind | Code license | Weights license | Training data | Input | DIS-VD S<sub>α</sub> / wF<br>(reported) | DIS-VD S<sub>α</sub> / wF / MAE<br>(measured, ONNX) | Peak VRAM<br>(measured) | Tesla T4 (Colab)<br>onnxruntime-cuda FP32 · ms / FPS | Tesla T4 (Colab)<br>onnxruntime-tensorrt FP16 · ms / FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| [BEN2-Base](ben2_base) | image (dichotomous segmentation) | 🟢 MIT | 🟢 MIT | DIS5K + 22K proprietary images | 1024×1024 | – | – | 4543 MB (Consumer, T4 CUDA FP32)<br>1525 MB (Tiny, T4 TRT FP16) | 559.9 / 2 | 161.8 / 6 |
| [BiRefNet-lite](birefnet_lite) | image (dichotomous segmentation) | 🟢 MIT | 🟢 MIT | DIS5K, P3M-10k, DUTS, HRSOD, UHRSD, HRS10K and others (general model) | 1024×1024 | [0.882 / 0.83](https://github.com/ZhengPeng7/BiRefNet/blob/ebcc0bc8ec7fe919cec829f2dea656b3078acddc/README.md) | – | 10749 MB (Performance, T4 CUDA FP32)<br>1471 MB (Tiny, T4 TRT FP16) | 591.9 / 2 | 93.2 / 11 |
| [RVM-MobileNetV3](rvm_mobilenetv3) | video (recurrent human matting) | 🟡 GPL-3.0 | 🟡 GPL-3.0* | VideoMatte240K, Distinctions-646, Adobe Image Matting, COCO, YouTubeVIS 2021, Supervisely Person | 720×1280 | – | – | 337 MB (Tiny, T4 CUDA FP32) | 16.6 / 60 | – |
<!-- END:background_removal_table -->

## Provenance

<!-- BEGIN:background_removal_provenance -->
| Model | Source (pinned) | Artifact | How it is produced |
|---|---|---|---|
| BEN2-Base | [PramaLLC/BEN2@2c99a5d](https://github.com/PramaLLC/BEN2/tree/2c99a5da477b5523585bfa5c893888a6e818a8f6) | `ben2_base.onnx` | download `BEN2_Base.onnx` (sha256 pinned) |
| BiRefNet-lite | [ZhengPeng7/BiRefNet@ebcc0bc](https://github.com/ZhengPeng7/BiRefNet/tree/ebcc0bc8ec7fe919cec829f2dea656b3078acddc) | `birefnet_lite.onnx` | download `BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx` (sha256 pinned) |
| RVM-MobileNetV3 | [PeterL1n/RobustVideoMatting@53d74c6](https://github.com/PeterL1n/RobustVideoMatting/tree/53d74c6826735f01f4406b5ca9075eee27bec094) | `rvm_mobilenetv3.onnx` | download `rvm_mobilenetv3_fp32.onnx` (sha256 pinned) |
<!-- END:background_removal_provenance -->

## Per-model notes

All three use the official ONNX files (SHA-256 pinned).

- **BiRefNet-lite** — upstream GitHub release `v1`
  `BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx`. Pre-processing as the
  model card: RGB, 1024x1024, /255, ImageNet mean/std; output logits ->
  sigmoid -> bilinear resize. Upstream notes the ONNX graph decomposes
  deformable convolutions and is slower than PyTorch.
- **BEN2-Base** — `BEN2_Base.onnx` from the Hugging Face repository. As the
  official `onnx_run.py`: RGB, 1024x1024, /255 without mean/std; the output
  is float16 alpha; upsampled bilinearly and min-max rescaled per image.
  (`refine_foreground` is a PyTorch-only extra and not part of the ONNX.)
  **Known issue:** on a GTX 1660 Ti with ONNX Runtime 1.22 CUDA EP the graph
  returns only NaN (CPU is fine; the comparison GIF frames for BEN2 were
  rendered on CPU). The runner raises on non-finite output instead of
  producing a matte.
- **RVM-MobileNetV3** — release `rvm_mobilenetv3_fp32.onnx`. Recurrent: the
  runner keeps `r1`-`r4` across frames (reset per image in evaluation);
  `downsample_ratio = min(512 / max(h, w), 1)` (upstream's automatic rule;
  0.4 for 720p). TensorRT needs a fixed ratio upstream (issue #52); with the
  official graph the TensorRT EP fails at engine build on a T4 (TensorRT 10.9
  `graphShapeAnalyzer` assertion), so there is no TensorRT record.
- On a T4 the CUDA EP needs 10.7 GB for BiRefNet-lite (the ONNX decomposes
  deformable convolutions into gather / scatter ops); TensorRT FP16 needs
  1.5 GB and is 6x faster.

## Training data

| Dataset | Terms |
|---|---|
| DIS5K | non-commercial research / education only, no redistribution |
| P3M-10k | P3M-10k release agreement (MIT-style per survey; *unverified*) |
| VideoMatte240K | commercial and non-commercial use |
| Distinctions-646, Adobe Image Matting | on request (*unverified*) |
