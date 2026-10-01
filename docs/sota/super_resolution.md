# Single-image super-resolution x4 — state of the art survey

**Surveyed: 2026-10-01.** Sources: upstream repositories at the listed
commit (GitHub API), papers (arXiv). PSNR / SSIM are upstream-reported, Y
channel, x4, bicubic degradation. *unverified* marks items not confirmed.

| Model | Code | Weights | Train data | Params | Set5 / Set14 / Urban100 / Manga109 PSNR |
|---|---|---|---|---|---|
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) x4plus | BSD-3 | GitHub release | DF2K + OST | 16.7M | GAN, no PSNR table |
| realesr-general-x4v3 / animevideov3 | BSD-3 | GitHub release | *unverified* | 1.21M / 0.62M | – |
| [SwinIR](https://github.com/JingyunLiang/SwinIR)-M | Apache-2.0 | GitHub release | DF2K | – | 32.92 / 29.09 / 27.45 / 32.03 |
| SwinIR-light | Apache-2.0 | GitHub release | DIV2K | 0.90M | 32.44 / 28.77 / 26.47 / 30.92 |
| [HAT](https://github.com/XPixelGroup/HAT) / HAT-L | Apache-2.0 | Google Drive | DF2K (HAT-L: + ImageNet) | 20.8M | 33.04 / 29.23 / 27.97 / 32.48; L 33.30 / 29.47 / 28.60 / 33.09 |
| [DAT](https://github.com/zhengchen1999/DAT) | Apache-2.0 | Google Drive | DF2K | 14.8M | 33.08 / 29.23 / 27.87 / 32.51 |
| [DRCT](https://github.com/ming053l/DRCT) | MIT | Google Drive | DF2K | 14.1M | 33.11 / 29.35 / 28.06 / 32.59 |
| [ATD](https://github.com/LabShuHangGU/Adaptive-Token-Dictionary) / light | Apache-2.0 | Google Drive | DF2K | 20.3M / 0.77M | 33.10 / 29.24 / 28.17 / 32.62; light 32.62 / 28.87 / 26.97 / 31.47 |
| [MambaIR](https://github.com/csguoh/MambaIR) / v2-light | Apache-2.0 | Google Drive | DF2K | – / 0.79M | 33.03 / 29.20 / 27.68 / 32.32; v2-light 32.51 / 28.84 / 26.82 / 31.24 |
| [Swin2SR](https://github.com/mv-lab/swin2sr) | Apache-2.0 | GitHub release | DF2K | – | 32.92 / 29.06 / 27.51 / 31.03 |
| [Omni-SR](https://github.com/Francis0625/Omni-SR) | **no LICENSE file** (README: Apache 2.0) | Drive / Baidu | DIV2K or DF2K | 0.79M | 32.49 / 28.78 / 26.64 / 31.02 |
| [SAFMN](https://github.com/sunny2109/SAFMN) | Apache-2.0 (inherited BasicSR LICENSE) | GitHub release; HF apache-2.0 | DF2K | 0.24M | 32.18 / 28.60 / 25.97 / 30.43 |
| [SPAN](https://github.com/hongyuanyu/SPAN) (NTIRE24 ESR 1st) | Apache-2.0 | 1.3 GB Drive zip | DF2K | 0.50M | 32.20 / 28.66 / 26.18 / 30.66 |
| [RLFN](https://github.com/bytedance/RLFN) (NTIRE22 runtime 1st) | Apache-2.0 | in repo | DIV2K | 0.54M | 32.24 / 28.62 / 26.17 / – |
| ESRT | MIT | none found | DIV2K | 0.75M | 32.19 / 28.69 / 26.39 / 30.75 |
| OSEDiff (one-step diffusion) | Apache-2.0 | LoRA on Drive; needs SD-2.1-base + RAM | – | – | real-world, no x4 PSNR table |

NTIRE 2025 efficient SR winners (SPAN-based, 0.13-0.17M params, trained on
DIV2K + LSDIR) are in Amazingren/NTIRE2025_ESR (MIT code); per-submission
weight licenses *unverified*.

## License traps

- **Training data**: DIV2K and LSDIR are "for academic research purpose
  only"; ImageNet (HAT-L pretraining) is research / non-commercial. Almost
  every model here is permissive code + research-only training data.
- **No explicit weights license** anywhere: checkpoints ship under the repo
  license without a separate statement.
- **Missing LICENSE**: Omni-SR.
- **Diffusion SR** inherits Stable Diffusion base-model terms (*unverified*
  mirror for OSEDiff).

## Export notes

Conv + PixelShuffle models (RRDBNet, SRVGGNetCompact, SPAN, RLFN) export
with dynamic H/W. SAFMN needs its adaptive max-pool replaced (multiples of 8).
Window-attention models (SwinIR, HAT, DAT, DRCT, ATD) bake mask computation
at trace time: one graph per resolution, pad to the window size. MambaIR
needs a custom CUDA selective-scan kernel.

## Implications for this collection

| Model | Status | Why |
|---|---|---|
| SAFMN x4 | **added** | Apache, 0.24M params, paper PSNR to verify the export |
| realesr-general-x4v3 | **added** | BSD-3, tiny real-world model |
| Real-ESRGAN x4plus | **added** | BSD-3, the standard real-world GAN model |
| SwinIR-light / HAT | candidate | transformer reference; fixed-shape graphs |
| SPAN / NTIRE ESR | candidate | fastest; weights hosting / licensing to check |

Evaluation: Hugging Face `eugenesiow/Set5`, `Set14`, `Urban100` (HR and LR_x4
archives); PSNR on BT.601 Y with a 4 px border (BasicSR).
