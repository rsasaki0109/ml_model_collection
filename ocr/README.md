# OCR (scene text)

Every model returns `OcrResult` (`tools/mlmc/ocr.py`): text-line
quadrilaterals (source pixels), recognised strings and confidences.

![comparison](../assets/ocr_comparison.gif)

## Comparison

<!-- BEGIN:ocr_table -->
| Model | Code license | Weights license | Languages | PaddleOCR benchmark det Hmean / rec acc<br>(reported, not ICDAR) | ICDAR2015 det H-mean / end-to-end H-mean<br>(measured, ONNX) | Peak VRAM<br>(measured) | Tesla T4 (Colab)<br>onnxruntime-cuda FP32 · ms / FPS | Tesla T4 (Colab)<br>onnxruntime-tensorrt FP16 · ms / FPS |
|---|---|---|---|---|---|---|---|---|
| [PP-OCRv5-mobile-EN](ppocrv5_mobile_en) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | English (436-entry dictionary) | [75.2](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/blob/28fe5895c24fd108c19eb3e8479f4ab385fbfc62/README.md) / – | **43.2 / 25.8** | 353 MB (Tiny, T4 CUDA FP32)<br>445 MB (Tiny, T4 TRT FP16) | 13.0 / 77 | 5.0 / 201 |
| [PP-OCRv6-small](ppocrv6_small) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | multilingual (rec card: 50 languages) | [84.1](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/blob/28fe5895c24fd108c19eb3e8479f4ab385fbfc62/README.md) / [81.3](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/blob/b8f84f0b80c529de40b4fbb3544b84fa7233a513/README.md) | **47.3 / 30.5** | 465 MB (Tiny, T4 CUDA FP32)<br>429 MB (Tiny, T4 TRT FP16) | 21.0 / 48 | 9.0 / 112 |
| [PP-OCRv6-tiny](ppocrv6_tiny) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | multilingual (6904-entry dictionary) | [80.6](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/blob/2ba1506c0380b8f0b03dd142459aac66d4421f6c/README.md) / [73.5](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/blob/2612ab37152ae0a677521bae4e1e3d4fb4cf7c30/README.md) | **43.1 / 21.7** | 371 MB (Tiny, T4 CUDA FP32)<br>459 MB (Tiny, T4 TRT FP16) | 12.8 / 78 | 6.8 / 148 |
<!-- END:ocr_table -->

## Provenance

<!-- BEGIN:ocr_provenance -->
| Model | Source (pinned) | Artifact | How it is produced |
|---|---|---|---|
| PP-OCRv5-mobile-EN | [PaddlePaddle/PaddleOCR@dab3fe3](https://github.com/PaddlePaddle/PaddleOCR/tree/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf) | `det.onnx` | download 3 files (sha256 pinned) |
| PP-OCRv6-small | [PaddlePaddle/PaddleOCR@dab3fe3](https://github.com/PaddlePaddle/PaddleOCR/tree/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf) | `det.onnx` | download 3 files (sha256 pinned) |
| PP-OCRv6-tiny | [PaddlePaddle/PaddleOCR@dab3fe3](https://github.com/PaddlePaddle/PaddleOCR/tree/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf) | `det.onnx` | download 3 files (sha256 pinned) |
<!-- END:ocr_provenance -->

## Pipeline

Official PaddlePaddle ONNX files from Hugging Face (`*_onnx` repositories,
pinned revisions and SHA-256) plus the character dictionary from the
PaddleOCR repository. `tools/mlmc/ocr.py` re-implements PaddleOCR's
inference path (commit dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf):

1. `DetResizeForTest` — PP-OCRv6: the operator defaults (short side at
   least 736, sides rounded to multiples of 32; the ONNX repositories'
   `inference.yml` gives no arguments); PP-OCRv5 mobile: `resize_long 960`
   (multiples of 128), as its `inference.yml`.
2. BGR, /255, ImageNet mean/std -> DB probability map.
3. `DBPostProcess` with each model's `inference.yml` thresholds (v6 small:
   0.2 / 0.45 / unclip 1.4; v6 tiny: 0.2 / 0.4 / 1.4; v5 mobile 0.3 / 0.6 /
   1.5), "fast" box score, pyclipper unclip.
4. `sorted_boxes`, perspective crop (`get_rotate_crop_image`).
5. Recognition in batches of 6 sorted by aspect ratio, height 48, CTC
   decoding with blank + dictionary + space; lines below 0.5 confidence are
   dropped (PaddleOCR `drop_score`). No text-line orientation classifier.

On a synthetic image all three read "Hello World 2026" /
"ml_model_collection" correctly (v6 small splits the first line into words).

## Evaluation

ICDAR2015 Task 4.1 test (500 images), evaluated with the logic of
PaddleOCR's `DetectionIoUEvaluator` (IoU > 0.5, '###' don't care);
end-to-end = matched detection whose string equals the transcription
(case-insensitive, no lexicon). The data comes from a Hugging Face mirror
(dlxjj/ICDAR2015, pinned revision) of the RRC files, whose official download
needs a registration and states no license (unverified). ICDAR2015 labels
words, PP-OCR detects text lines, and none of these checkpoints is fine-tuned
on ICDAR2015 — the numbers are a zero-shot reference, not comparable with
ICDAR-trained detectors.
