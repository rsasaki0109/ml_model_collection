# Segmentation

Every model here takes a BGR frame and returns either `InstanceMasks`
(boxes + per-instance masks + COCO names) or a `SemanticMap` (class index per
pixel, ADE20K names), both at source resolution (`tools/mlmc/segmentation.py`).

![comparison](../assets/segmentation_comparison.gif)

## Comparison

<!-- BEGIN:segmentation_table -->
| Model | Kind | Code license | Weights license | Input | Accuracy (reported) | Peak VRAM<br>(measured) | Tesla T4 (Colab)<br>onnxruntime-cuda FP32 · ms / FPS | Tesla T4 (Colab)<br>onnxruntime-tensorrt FP16 · ms / FPS |
|---|---|---|---|---|---|---|---|---|
| [Mask2Former-SwinT (ADE20K)](mask2former_swin_t_ade) | semantic | 🟢 MIT | 🟢 MIT* | 512×512 | [47.7](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/MODEL_ZOO.md) ADE20K val mIoU | 1375 MB (Tiny, CUDA FP32)<br>573 MB (Tiny, TRT FP16) | 108.2 / 9 | 42.2 / 24 |
| [RF-DETR-Seg-N](rfdetr_seg_n) | instance | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 312×312 | [40.3](https://github.com/roboflow/rf-detr/blob/5f441831aaf23a68f40128ad0a2e27cb44e52640/README.md) COCO val2017 mask AP | 407 MB (Tiny, CUDA FP32)<br>431 MB (Tiny, TRT FP16) | 21.0 / 48 | 5.1 / 196 |
| [SegFormer-B0 (ADE20K)](segformer_b0_ade) | semantic | 🔴 NVIDIA-NC | 🔴 NVIDIA-NC* | 512×512 | [37.4](https://arxiv.org/abs/2105.15203) ADE20K val mIoU | 485 MB (Tiny, CUDA FP32)<br>461 MB (Tiny, TRT FP16) | 13.8 / 72 | 6.3 / 159 |
<!-- END:segmentation_table -->

## Provenance

<!-- BEGIN:segmentation_provenance -->
| Model | Source (pinned) | Artifact | How it is produced |
|---|---|---|---|
| Mask2Former-SwinT (ADE20K) | [facebookresearch/Mask2Former@9b0651c](https://github.com/facebookresearch/Mask2Former/tree/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a) | `mask2former_swin_t_ade.onnx` | [`export.py`](mask2former_swin_t_ade/export.py) |
| RF-DETR-Seg-N | [roboflow/rf-detr@5f44183](https://github.com/roboflow/rf-detr/tree/5f441831aaf23a68f40128ad0a2e27cb44e52640) | `rfdetr_seg_n.onnx` | [`export.py`](rfdetr_seg_n/export.py) |
| SegFormer-B0 (ADE20K) | [NVlabs/SegFormer@65fa8cf](https://github.com/NVlabs/SegFormer/tree/65fa8cfa9b52b6ee7e8897a98705abf8570f9e32) | `segformer_b0_ade.onnx` | [`export.py`](segformer_b0_ade/export.py) |
<!-- END:segmentation_provenance -->

## Per-model notes

- **RF-DETR-Seg-N** — exported with `rfdetr==1.11.0` (Apache-2.0 code and
  weights). Mask post-processing mirrors rfdetr's `PostProcess`
  (bilinear upsample, logit > 0); on the demo frame the top-5 masks match
  `RFDETRSegNano.predict()` with IoU 0.995–0.999.
- **Mask2Former-SwinT (ADE20K)** — exported from the Hugging Face port.
  transformers builds `GridSample` with a float64 grid, which ONNX Runtime
  does not implement; the exporter casts it to float32
  (`tools/mlmc/export.py: patch_for_ort`). 99.3% pixel agreement with
  transformers' `post_process_semantic_segmentation` on the demo frame.
  Input is a fixed 512x512 resize (the upstream evaluation resizes the
  shortest side to 512), so measure mIoU before relying on the reported
  47.7.
- **SegFormer-B0 (ADE20K)** — ⚠️ **non-commercial** (NVIDIA Source Code
  License §3.3, code and weights). Exported from `nvidia/segformer-b0-*` at a
  pinned revision rather than from third-party ONNX re-uploads, which drop
  the license. 100% pixel agreement with transformers.

Not yet included: promptable models (EdgeTAM, SAM 2.1-tiny, EfficientViT-SAM)
need a point/box prompt interface; see
[docs/sota/segmentation.md](../docs/sota/segmentation.md).
