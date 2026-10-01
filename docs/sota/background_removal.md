# Background removal / matting / dichotomous segmentation — state of the art survey

**Surveyed: 2026-10-01.** Sources: GitHub and Hugging Face APIs (license
files, tags, revisions), ONNX graphs inspected for I/O. *unverified* marks
items not confirmed from a primary source.

| Model | Code | Weights | Train data flags | Size / input | Official ONNX | Video |
|---|---|---|---|---|---|---|
| [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (general / lite / HR / matting / portrait) | MIT | HF, mit | DIS5K (**NC**), others *unverified* | 220.7M (L), 44.4M (lite); 1024² (HR 2048²) | yes, GitHub release v1 | no |
| [BEN2](https://github.com/PramaLLC/BEN2) | MIT | HF, mit | DIS5K + 22K proprietary | 94.6M; 1024² | yes (HF) | frame-wise |
| [RVM](https://github.com/PeterL1n/RobustVideoMatting) | **GPL-3.0** | release (repo license) | VideoMatte240K (commercial OK), D646 / AIM on request | 3.7M (mbv3); any size | yes, fp32/fp16 | **recurrent** |
| RMBG-2.0 (BRIA, BiRefNet arch.) | – | **CC BY-NC 4.0**, gated | private | ~220M; 1024² | yes | no |
| RMBG-1.4 (IS-Net based) | – | **bria-rmbg-1.4, non-commercial** | 12k licensed images | 44.1M; 1024² | yes | no |
| [MODNet](https://github.com/ZHKKKe/MODNet) | Apache-2.0 ("code, models, and demos") | Google Drive; third-party ONNX | *unverified* | 512 | Drive link only | frame-wise |
| BackgroundMattingV2 | MIT | Google Drive | VideoMatte240K, PhotoMatte85 | needs a background plate | *unverified* | no |
| MatAnyone / MatAnyone2 | **S-Lab License 1.0 (non-commercial)** | HF, no tag | – | 35.3M; needs first-frame mask | no | yes |
| ViTMatte | MIT | HF apache-2.0 | Composition-1k (*unverified*) | 25.8M; **needs trimap** | third-party | no |
| InSPyReNet | MIT | Google Drive | DIS5K / DUTS | – | *unverified* | no |
| U²-Net / IS-Net | Apache-2.0 | Google Drive; rembg re-hosts ONNX | IS-Net: DIS5K (**NC**) | ~44M | third-party | no |
| MediaPipe Selfie Segmenter | Apache-2.0 | model card: Apache-2.0 | – | 256² | TFLite only | frame-wise |
| withoutBG open weights | Apache-2.0 (SDK) | **mixed: Apache + Meta DINOv3 license** | *unverified* | – | yes | no |
| ZIM | **CC BY-NC 4.0** code; HF says cc-by-4.0 (**conflict**) | HF | SA-1B derived | prompt-based | yes | no |
| FlowDIS | **non-commercial** | 23.8 GB + T5-XXL | – | text-guided | no | no |

Reported: BiRefNet (DIS-VD, S / wF / HCE) general 0.911 / 0.875 / 1069,
lite 0.882 / 0.830 / 1175, HR 0.927 / 0.894 / 881; RVM mbv3 VideoMatte240K
512x288 MAD 6.08, MSE 1.47 (paper Table 1). BEN2 publishes only an image
comparison.

## License traps

- **DIS5K is non-commercial and may not be redistributed** — BiRefNet,
  BEN2 and IS-Net are trained on it.
- **BRIA RMBG**: 1.4 non-commercial, 2.0 CC BY-NC 4.0 and gated; rembg
  re-hosts the gated 2.0 file publicly.
- **GPL-3.0**: RVM.
- **Non-commercial code**: MatAnyone (S-Lab), ZIM, FlowDIS.
- **Mixed licenses**: withoutBG (DINOv3 backbone license).

## Implications for this collection

| Model | Status | Why |
|---|---|---|
| BiRefNet-lite | **added** | MIT code and weights, official ONNX |
| BEN2-Base | **added** | MIT, official ONNX, matting-quality edges |
| RVM-MobileNetV3 | **added** | real-time recurrent video matting reference (GPL-3.0) |
| MODNet | candidate | Apache incl. models; ONNX only on Google Drive |
| RMBG-1.4 / 2.0, MatAnyone, ZIM | not planned | non-commercial |

Evaluation: DIS5K validation split (DIS-VD, 470 images; Google Drive id
1O1eIuXX1hlGsV7qx4eSkjH231q7G1by1 from the DIS README; terms forbid
redistribution), PySODMetrics (MIT). VideoMatte240K composited test sets for
RVM are available from robustvideomatting.blob.core.windows.net.
