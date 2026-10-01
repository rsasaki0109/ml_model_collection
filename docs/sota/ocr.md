# Scene-text OCR — state of the art survey

**Surveyed: 2026-10-01.** Sources: GitHub API (HEAD commits, licenses),
Hugging Face model cards and `inference.yml` files, ONNX graph headers.
*unverified* marks items not confirmed.

| Model | Code | Weights | Official ONNX | Notes |
|---|---|---|---|---|
| **PP-OCRv6** tiny / small / medium (2026, arXiv 2606.13108) | Apache-2.0 | HF `PaddlePaddle/PP-OCRv6_*_onnx`, apache-2.0 | **yes** | 1.5M-34.5M params; det Hmean 80.6 / 84.1 / 86.2, rec W-Avg 73.5 / 81.3 / 83.2 on PaddleOCR's multi-scenario benchmark (not ICDAR) |
| PP-OCRv5 mobile / server (+ 12 language recognizers) | Apache-2.0 | HF apache-2.0 | **yes** | det 75.2 / 81.6, rec 73.7 / 78.1 (same benchmark) |
| RapidOCR | Apache-2.0 | Apache-2.0 (own conversions, not byte-identical to official) | yes | packaging of PP-OCR |
| EasyOCR (CRAFT + CRNN) | Apache-2.0 | jaided.ai, **no license stated** | export script | |
| CRAFT | MIT | Google Drive, *unverified* | no | |
| DBNet / DBNet++ (MMOCR) | Apache-2.0 | download.openmmlab.com, *unverified* | no | DBNet++ R50 IC15 H 86.2 |
| PARSeq | Apache-2.0 | GitHub release, *unverified* | community | IIIT5k 99.0 / IC15 89.2 (word crops) |
| ABINet | **non-commercial** | – | no | |
| MaskTextSpotterV3 | **CC BY-NC 4.0** | – | no | |
| docTR / OnnxTR | Apache-2.0 | OnnxTR release assets, *unverified* | community | document-focused |
| Surya | Apache-2.0 code | **modified OpenRAIL-M (revenue / funding cap)** | no | |
| TrOCR | MIT | HF card has no license | no | |
| Doc VLMs: GOT-OCR2, Florence-2, PaddleOCR-VL, DeepSeek-OCR, olmOCR-2, Chandra | mixed | apache / MIT / **openrail** | no | not real-time |

## License traps

- **Surya** weights: modified OpenRAIL-M with commercial caps.
- **ABINet** (non-commercial), **MaskTextSpotterV3** (CC BY-NC 4.0).
- **Weights without a license**: EasyOCR, CRAFT, TrOCR, MMOCR checkpoints.
- **Datasets**: Total-Text is non-commercial academic use only; ICDAR2015
  (RRC) needs registration and states no license (*unverified*).
- PP-OCR training data is not disclosed.

## Implications for this collection

| Pipeline | Status | Why |
|---|---|---|
| PP-OCRv6 small | **added** | Apache, official ONNX, multilingual, balanced |
| PP-OCRv6 tiny | **added** | 1.5M params total, edge |
| PP-OCRv5 mobile det + EN rec | **added** | baseline, English dictionary |
| PP-OCRv6 medium | candidate | best of the family (34.5M) |
| PARSeq, docTR | candidate | permissive code, weights licensing unclear |
| Surya, ABINet, MaskTextSpotterV3 | not planned | restricted |
