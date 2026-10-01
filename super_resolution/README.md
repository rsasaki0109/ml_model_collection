# Super-resolution (x4)

Every model returns `SRImage` (`tools/mlmc/sr.py`): the x4 upscaled BGR
image plus the low-resolution input. Graphs take `lr [1,3,h,w]` RGB in
[0, 1] with dynamic h, w. Video runs and benchmarks use the deployment
setting in `model.yaml` (`input_shape` 180x320: every frame is downscaled to
320x180 with bicubic, then upscaled to 1280x720); evaluation feeds the
benchmark LR images at their own sizes.

![comparison](../assets/super_resolution_comparison.gif)

## Comparison

<!-- BEGIN:super_resolution_table -->
| Model | Kind | Code license | Weights license | Training data | PSNR-Y x4 Set5 / Set14 / Urban100<br>(reported) | PSNR-Y x4 Set5 / Set14 / Urban100<br>(measured, ONNX) | Peak VRAM<br>(measured) |
|---|---|---|---|---|---|---|---|
| [Real-ESRGAN-x4plus](real_esrgan_x4plus) | real-world (GAN) | 🟢 BSD-3-Clause | 🟢 BSD-3-Clause* | DF2K (DIV2K + Flickr2K) + OST | – | – | not measured |
| [realesr-general-x4v3](realesr_general_x4v3) | real-world (GAN) | 🟢 BSD-3-Clause | 🟢 BSD-3-Clause* | not documented upstream | – | – | not measured |
| [SAFMN-x4](safmn_x4) | PSNR-oriented | 🟢 Apache-2.0 | 🟢 Apache-2.0* | DF2K (DIV2K + Flickr2K) | [32.18 / 28.60 / 25.97](https://arxiv.org/abs/2302.13800) | – | not measured |
<!-- END:super_resolution_table -->

## Provenance

<!-- BEGIN:super_resolution_provenance -->
| Model | Source (pinned) | Artifact | How it is produced |
|---|---|---|---|
| Real-ESRGAN-x4plus | [xinntao/Real-ESRGAN@a4abfb2](https://github.com/xinntao/Real-ESRGAN/tree/a4abfb2979a7bbff3f69f58f58ae324608821e27) | `real_esrgan_x4plus.onnx` | [`export.py`](real_esrgan_x4plus/export.py) |
| realesr-general-x4v3 | [xinntao/Real-ESRGAN@a4abfb2](https://github.com/xinntao/Real-ESRGAN/tree/a4abfb2979a7bbff3f69f58f58ae324608821e27) | `realesr_general_x4v3.onnx` | [`export.py`](realesr_general_x4v3/export.py) |
| SAFMN-x4 | [sunny2109/SAFMN@1a41206](https://github.com/sunny2109/SAFMN/tree/1a412066e927da640bc7e2a53ea60f9ffc134241) | `safmn_x4.onnx` | [`export.py`](safmn_x4/export.py) |
<!-- END:super_resolution_provenance -->

## Per-model notes

All three are exported from the official architecture code and release
checkpoints (SHA-256 pinned in `export.py`). BasicSR-style architecture
files are loaded without the `basicsr` package (stub registry; helper
functions taken verbatim from upstream `arch_util.py`). ONNX Runtime matches
PyTorch to within 1.5e-5 on a 176x320 frame.

- **SAFMN-x4** — `SAFMN(dim=36, n_blocks=8, ffn_scale=2.0, upscaling_factor=4)`
  from the upstream x4 test config. `adaptive_max_pool2d` has no
  dynamic-shape ONNX form and is exported as `max_pool2d(kernel = stride = 2^i)`,
  identical for sides that are multiples of 8; the runner reflect-pads to
  multiples of 8 and crops (upstream's ONNX script also requires multiples
  of 8). Set5 on a local check: 32.15 dB vs 32.18 reported.
- **realesr-general-x4v3** — `SRVGGNetCompact(num_conv=32, act_type='prelu')`.
  Upstream's CLI blends it with the `-wdn` (denoise) checkpoint at
  `--denoise_strength 0.5` by default; this is the plain model (strength 1).
- **Real-ESRGAN-x4plus** — BasicSR `RRDBNet(num_block=23)`, `params_ema`.

## Training data

| Dataset | Terms |
|---|---|
| DIV2K | "made available for academic research purpose only" |
| Flickr2K, OST | no license found (unverified) |
| Set5 / Set14 (evaluation) | HF cards: "academic use only" / license other |
| Urban100 (evaluation) | HF card: images from Flickr, CC BY 4.0 |
