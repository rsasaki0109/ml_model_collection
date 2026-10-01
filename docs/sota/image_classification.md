# Image classification — state of the art survey

**Surveyed: 2026-10-02.** Sources: Hugging Face API (tags, revisions, LFS
hashes), timm results CSVs (`results-imagenet.csv`,
`results-imagenetv2-matched-frequency.csv` at
92dbd9e3f9b5d61c4d008223410781da983239fc), papers, open_clip results CSV.

## Supervised ImageNet-1k (timm hub weights)

| Model | Weights tag | Params / GMACs | Eval size | IN-1k top-1 | IN-v2 top-1 |
|---|---|---|---|---|---|
| MobileNetV4-Conv-S (e2400) | apache-2.0 | 3.8 / 0.2 | 224 | 73.76 | 60.90 |
| MobileNetV4-Conv-M (e500) | apache-2.0 | 9.7 / 1.1 | 256 | 79.92 | 69.00 |
| MobileNetV4-Hybrid-M | apache-2.0 | 11.1 / 1.3 | 256 | 81.48 | 70.57 |
| EfficientNet-B0 (ra) | apache-2.0 | 5.3 / 0.4 | 224 | 77.70 | 66.15 |
| EfficientNetV2-S (tf) | apache-2.0 | 21.5 / 5.4 | 300 | 83.16 | 72.76 |
| RepViT-M1.1 (dist) | apache-2.0 | 8.8 / 1.4 | 224 | 81.31 | 70.37 |
| EfficientViT-B1 | apache-2.0 | 9.1 / 0.5 | 224 | 79.25 | 67.85 |
| LeViT-128S | apache-2.0 | 7.8 / 0.3 | 224 | 76.52 | 64.45 |
| EdgeNeXt-S | mit | 5.6 / 1.3 | 256 | 81.07 | 70.00 |
| ConvNeXt-T (fb) | apache-2.0 | 28.6 / 4.5 | 224 | 82.07 | 70.93 |
| **ConvNeXt V2-T** | **cc-by-nc-4.0** | 28.6 | 224 | 82.95 | 72.03 |
| **FastViT-T8** | **Apple license (not OSI)** | 4.0 / 0.7 | 256 | 76.18 | 64.69 |
| DeiT III-S | apache-2.0 | 22.1 / 4.6 | 224 | 81.38 | 70.77 |
| ViT-B/16 augreg2 (IN-21k -> 1k) | apache-2.0 | 86.6 | 224 | 85.11 | 75.32 |

Newer timm backbones (iFormer, LowFormer, EfficientViM, CSATv2) have no
timm-measured results yet; MobileNetV5 carries the **Gemma** license.

## Zero-shot image-text models

| Model | Weights license | Training data | ZS IN-1k / IN-v2 |
|---|---|---|---|
| **SigLIP 2 B/16-224** | apache-2.0 | WebLI (not released) | 78.2 / 71.4 (paper) |
| PE-Core B16-224 (Meta) | apache-2.0 | Meta-curated (*unverified*) | 78.4 / 71.7 (card) |
| MobileCLIP2-S0, DFN CLIP | **apple-amlr: research only** | DFN data CC-BY-NC-ND | 71.5 / -; 76.2 / 68.2 |
| OpenAI CLIP B/32 | no tag; model card: **any deployed use out of scope** | WIT-400M | 63.3 / 55.9 |
| EVA02-B/16 (EVA-CLIP) | mit | LAION-based "Merged-2B" | 74.7 / 67.0 |
| OpenCLIP B/32 LAION-2B | mit | LAION-2B (taken down 2023, Re-LAION 2024) | 66.6 / 58.1 |
| MetaCLIP 2 | **cc-by-nc-4.0** | - | - |

## License traps

- **ImageNet-1k** terms are non-commercial research; timm: "assume that the
  original dataset license applies to the weights". The HF `imagenet-1k`
  dataset is gated (terms acceptance).
- torchvision weights: no explicit license except SWAG (CC BY-NC).
- ConvNeXt V2 and MetaCLIP 2: CC BY-NC. FastViT, MobileCLIP, DFN: Apple
  research licenses. OpenAI CLIP: model card rules out deployed use.
- LAION-5B was withdrawn (CSAM links); Re-LAION metadata is Apache-2.0,
  image copyright *unverified*.

## Implications for this collection

| Model | Status | Why |
|---|---|---|
| MobileNetV4-Conv-S / -M | **added** | Apache, official ONNX, real-time |
| RepViT-M1.1 | **added** | Apache code and weights, 81.3 % at 1.4 GMACs |
| SigLIP 2 B/16-224 | **added** (zero-shot) | Apache, strongest permissive zero-shot base model |
| PE-Core B16 | candidate | Apache; preprocessing / tokenizer to verify |
| ConvNeXt V2, FastViT, MobileCLIP, CLIP | not planned | license |
