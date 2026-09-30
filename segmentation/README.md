# Segmentation

Every model here takes a BGR frame and returns either `InstanceMasks`
(boxes + per-instance masks + COCO names) or a `SemanticMap` (class index per
pixel, ADE20K names), both at source resolution (`tools/mlmc/segmentation.py`).

![comparison](../assets/segmentation_comparison.gif)

## Comparison

<!-- BEGIN:segmentation_table -->
| Model | Kind | Code license | Weights license | Input | Accuracy (reported) | Accuracy<br>(measured, ONNX) | Peak VRAM<br>(measured) | Tesla T4 (Colab)<br>onnxruntime-cuda FP32 · ms / FPS | Tesla T4 (Colab)<br>onnxruntime-tensorrt FP16 · ms / FPS |
|---|---|---|---|---|---|---|---|---|---|
| [EdgeTAM](edgetam) | promptable | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 1024×1024 | [71.7](https://github.com/facebookresearch/EdgeTAM/blob/7711e012a30a2402c4eaab637bdb00a521302c91/README.md) SA-V test J&F | **33.9** COCO mask AP (boxes from dfine_n) | 1393 MB (Tiny, CUDA FP32)<br>801 MB (Tiny, TRT FP16) | 79.3 / 13 | 28.8 / 35 |
| [Mask2Former-SwinT (ADE20K)](mask2former_swin_t_ade) | semantic | 🟢 MIT | 🟢 MIT* | 512×512 | [47.7](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/MODEL_ZOO.md) ADE20K val mIoU | **46.5** ADE20K mIoU | 1375 MB (Tiny, CUDA FP32)<br>573 MB (Tiny, TRT FP16) | 108.2 / 9 | 42.2 / 24 |
| [RF-DETR-Seg-N](rfdetr_seg_n) | instance | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 312×312 | [40.3](https://github.com/roboflow/rf-detr/blob/5f441831aaf23a68f40128ad0a2e27cb44e52640/README.md) COCO val2017 mask AP | **40.1** COCO mask AP | 407 MB (Tiny, CUDA FP32)<br>431 MB (Tiny, TRT FP16) | 21.0 / 48 | 5.1 / 196 |
| [SAM2.1-Hiera-T](sam21_hiera_tiny) | promptable | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 1024×1024 | [76.5](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/README.md) SA-V test J&F | **35.5** COCO mask AP (boxes from dfine_n) | 3359 MB (Light, CUDA FP32)<br>1123 MB (Tiny, TRT FP16) | 169.1 / 6 | 55.3 / 18 |
| [SegFormer-B0 (ADE20K)](segformer_b0_ade) | semantic | 🔴 NVIDIA-NC | 🔴 NVIDIA-NC* | 512×512 | [37.4](https://arxiv.org/abs/2105.15203) ADE20K val mIoU | **35.9** ADE20K mIoU | 485 MB (Tiny, CUDA FP32)<br>461 MB (Tiny, TRT FP16) | 13.8 / 72 | 6.3 / 159 |
<!-- END:segmentation_table -->

## Provenance

<!-- BEGIN:segmentation_provenance -->
| Model | Source (pinned) | Artifact | How it is produced |
|---|---|---|---|
| EdgeTAM | [facebookresearch/EdgeTAM@7711e01](https://github.com/facebookresearch/EdgeTAM/tree/7711e012a30a2402c4eaab637bdb00a521302c91) | `vision_encoder.onnx` | download 4 files (sha256 pinned) |
| Mask2Former-SwinT (ADE20K) | [facebookresearch/Mask2Former@9b0651c](https://github.com/facebookresearch/Mask2Former/tree/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a) | `mask2former_swin_t_ade.onnx` | [`export.py`](mask2former_swin_t_ade/export.py) |
| RF-DETR-Seg-N | [roboflow/rf-detr@5f44183](https://github.com/roboflow/rf-detr/tree/5f441831aaf23a68f40128ad0a2e27cb44e52640) | `rfdetr_seg_n.onnx` | [`export.py`](rfdetr_seg_n/export.py) |
| SAM2.1-Hiera-T | [facebookresearch/sam2@2b90b9f](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4) | `vision_encoder.onnx` | download 4 files (sha256 pinned) |
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
  shortest side to 512); measured ADE20K val mIoU of this artifact is
  **46.5** vs. 47.7 reported.
- **SegFormer-B0 (ADE20K)** — ⚠️ **non-commercial** (NVIDIA Source Code
  License §3.3, code and weights). Exported from `nvidia/segformer-b0-*` at a
  pinned revision rather than from third-party ONNX re-uploads, which drop
  the license. 100% pixel agreement with transformers.

- **SAM2.1-Hiera-T / EdgeTAM** (promptable) — onnx-community ONNX exports
  (encoder + prompt decoder), pinned by SHA-256; Apache-2.0 upstream. By
  default they are prompted with **D-FINE-N boxes** ("detect, then
  segment"), so they return instance masks with D-FINE-N's labels and
  scores; `segment()` accepts user boxes. The mask with the highest
  predicted IoU is kept. SAM2.1 masks match transformers' `Sam2Model` on the
  demo frame (IoU ≥ 0.99). EdgeTAM could **not** be parity-checked:
  `facebook/EdgeTAM` has no transformers-format checkpoint, and
  onnx-community does not state which checkpoint it converted.
  COCO mask AP of these pipelines depends on the prompt detector: with
  D-FINE-N boxes (box AP 42.6) SAM2.1-T reaches 35.5 and EdgeTAM 33.9 mask
  AP, below RF-DETR-Seg-N's 40.1 — box-prompted SAM masks are class-agnostic
  and the scores come from the detector.
