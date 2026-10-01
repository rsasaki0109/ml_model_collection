# Small vision-language models for captioning — state of the art survey

**Surveyed: 2026-10-02.** Sources: Hugging Face API (revisions, license
tags, LFS hashes), model cards, papers; ONNX graphs inspected and greedy
decoding smoke-tested on one image (no benchmarks).

| Model | Params | Code | Weights (inherited) | COCO Karpathy CIDEr | ONNX |
|---|---|---|---|---|---|
| **Florence-2-base / -large** | 0.23B / 0.77B | MIT | MIT | zero-shot 133.0 / 135.6; -ft 140.0 / 143.3 | onnx-community (fp16 merged decoder broken in ORT 1.22) |
| **SmolVLM-256M / 500M-Instruct** | 256M / 500M | Apache-2.0 | Apache-2.0 | not reported | **official** `onnx/` |
| SmolVLM2-256M / 500M-Video | 256M / 500M | Apache-2.0 | Apache-2.0 | not reported | official |
| Moondream 2 | - | Apache-2.0 | Apache-2.0 (**moondream3: BSL 1.1**) | not checked | outdated Xenova port |
| Qwen3-VL-2B-Instruct | 2B | Apache-2.0 | Apache-2.0 | not checked | onnx-community (no README / license tag), M-RoPE |
| **Qwen2.5-VL-3B** | 3B | Apache-2.0 | **Qwen research license (non-commercial)** | - | onnx-community |
| InternVL3-1B / 2B | 1B / 2B | MIT | **conflicting tags** (Apache / Qwen license) | - | none |
| **PaliGemma 2** | 3B | Apache | **Gemma terms** (gated) | - | onnx-community |
| Phi-3.5-vision / Phi-4-multimodal | > 2B | MIT | MIT | - | int4 for ORT-GenAI only |
| BLIP base / large | - | BSD-3 | BSD-3 | 136.7 (ViT-L, COCO fine-tuned) | non-standard split |
| BLIP-2 OPT-2.7B | > 2B | BSD-3 | MIT tag, OPT base **license other** | 145.8 fine-tuned | none |
| GIT base / large (COCO) | 0.1B / 0.3B | MIT | MIT | 131.4 / 138.5 (CE) | none |
| **FastVLM-0.5B** | 0.5B | Apple custom | **apple-amlr research only** | - | onnx-community |
| **LFM2-VL-450M** | 450M | - | **LFM Open License (>= $10M revenue limited)** | - | onnx-community |

## License traps

Qwen2.5-VL-3B (research license), PaliGemma (Gemma terms), FastVLM (Apple
research), LFM2-VL (revenue cap), Moondream 3 (BSL), InternVL's conflicting
tags, BLIP-2's OPT base. ONNX mirrors without a license tag inherit nothing
explicit.

## Implications for this collection

| Model | Status | Why |
|---|---|---|
| Florence-2-base | **added** | MIT, best CIDEr per parameter, standard encoder-decoder ONNX |
| SmolVLM-256M-Instruct | **added** | Apache-2.0, official ONNX, chat VLM |
| Florence-2-large-ft, SmolVLM-500M | candidate | same code paths |
| Qwen3-VL-2B | candidate | Apache; large ONNX, M-RoPE |
