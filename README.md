# ml_model_collection

> A collection of reproducible, comparable, and deployment-ready ML models.

Pick a model by **Task × License × Hardware requirements × Runtime** — not by scrolling through hundreds of files.
Every number below was measured by a script in this repo, under the conditions recorded next to it.

![Object detection comparison: D-FINE-N, D-FINE-S, DEIM-D-FINE-S, RT-DETRv4-S, RF-DETR-S, YOLO26n on the same clip](assets/object_detection_comparison.gif)

<sub>Current real-time SOTA, small sizes: same frames, same tile size, six models ([line-up](object_detection/comparison.yaml), [why these](docs/sota/object_detection.md)). Regenerate with `python tools/make_comparison.py --task object_detection --input assets/demo.mp4`.
Video: "Road traffic on Stritarjeva street" by Sounds of Changes, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0), via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Road_traffic_on_Stritarjeva_street.webm) — trimmed, scaled, and annotated with model outputs ([details](assets/README.md)).</sub>

## Object detection

<!-- BEGIN:object_detection_table -->
| Model | Code license | Weights license | Input | COCO mAP<br>(reported) | Peak VRAM<br>(measured) |
|---|---|---|---|---|---|
| [DEIM-D-FINE-S](object_detection/deim_dfine_s) | 🟢 Apache-2.0 | 🟢 Apache-2.0* | 640×640 | [49.0](https://github.com/Intellindust-AI-Lab/DEIM/blob/09d35d53d39ee3145a1e61e3a989b28b9468d1dd/README.md) | not measured |
| [D-FINE-N](object_detection/dfine_n) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 640×640 | [42.8](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/README.md) | not measured |
| [D-FINE-S](object_detection/dfine_s) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 640×640 | [48.5](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/README.md) | not measured |
| [RF-DETR-N](object_detection/rfdetr_n) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 384×384 | [48.4](https://github.com/roboflow/rf-detr/blob/5f441831aaf23a68f40128ad0a2e27cb44e52640/README.md) | not measured |
| [RF-DETR-S](object_detection/rfdetr_s) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 512×512 | [53.0](https://github.com/roboflow/rf-detr/blob/5f441831aaf23a68f40128ad0a2e27cb44e52640/README.md) | not measured |
| [RT-DETR-R18](object_detection/rtdetr_r18vd) | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 640×640 | [46.5](https://huggingface.co/PekingU/rtdetr_r18vd/blob/ac77a11ff0170a41b771c03264987f8ce2b0d753/README.md) | not measured |
| [RT-DETRv4-M](object_detection/rtdetrv4_m) | 🟢 Apache-2.0 | 🟢 Apache-2.0* | 640×640 | [53.7](https://github.com/RT-DETRs/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/README.md) | not measured |
| [RT-DETRv4-S](object_detection/rtdetrv4_s) | 🟢 Apache-2.0 | 🟢 Apache-2.0* | 640×640 | [49.8](https://github.com/RT-DETRs/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/README.md) | not measured |
| [SSDLite320-MobileNetV3](object_detection/ssdlite320_mobilenet_v3_large) | 🟢 BSD-3-Clause | ⚪ unknown | 320×320 | [21.3](https://github.com/pytorch/vision/blob/6da25ff876100d36f23472f5762d5f306c47d735/torchvision/models/detection/ssdlite.py) | not measured |
| [YOLO11n](object_detection/yolo11n) | 🟡 AGPL-3.0 | 🟡 AGPL-3.0* | 640×640 | [39.5](https://github.com/ultralytics/ultralytics/blob/50ca85e03bc669c694c474ab91168f9b5425d9a8/docs/en/models/yolo11.md) | not measured |
| [YOLO26n](object_detection/yolo26n) | 🟡 AGPL-3.0 | 🟡 AGPL-3.0 | 640×640 | [40.1](https://github.com/ultralytics/ultralytics/blob/50ca85e03bc669c694c474ab91168f9b5425d9a8/README.md) | not measured |
| [YOLOX-S](object_detection/yolox_s) | 🟢 Apache-2.0 | 🟢 Apache-2.0* | 640×640 | [40.5](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/README.md) | not measured |
<!-- END:object_detection_table -->

- 🟢 permissive · 🟡 copyleft · 🔴 restricted · ⚪ unknown. `*` = the weights are published from a repository/release under that license, but upstream does not state a separate license for the weights. See [docs/licenses.md](docs/licenses.md).
- **COCO mAP** is copied from upstream (linked) and was *not* re-evaluated here. Different models report slightly different metrics — follow the link.
- **ms / FPS**: mean `session.run` latency, batch 1, FP32, ONNX Runtime 1.22 (full conditions in each model's `benchmarks.yaml`). Graphs differ in how much post-processing they contain: DEIM / RT-DETRv4 / YOLO26n include top-k selection, SSDLite includes resize + NMS, the others end at raw predictions (decoded in NumPy, see `e2e_ms_mean` in `benchmarks.yaml`).
- **Peak VRAM**: device memory delta during session creation + inference, CUDA context included ([method](docs/hardware.md#how-vram-is-measured)). It is valid only for the listed hardware/runtime/precision/batch/input.

Tables are generated from metadata by `python tools/build_readme.py`; do not edit them by hand.

How these models relate to the current state of the art, and which models are planned next: [docs/sota/object_detection.md](docs/sota/object_detection.md) (surveyed 2026-09-30).

## Find a model

```console
$ python tools/find_models.py --task object_detection --license permissive --max-vram-mb 4096
model         license (code / weights)      hardware            runtime           latency  peak VRAM
rtdetr_r18vd  🟢 Apache-2.0 / 🟢 Apache-2.0   GTX 1660 Ti Laptop  onnxruntime-cuda  27.4 ms  319 MB
yolox_s       🟢 Apache-2.0 / 🟢 Apache-2.0*  GTX 1660 Ti Laptop  onnxruntime-cuda  14.5 ms  199 MB

* = license inherited from the repository/release; weights have no separate license statement.
```

Filters: `--task`, `--license {permissive,copyleft,restricted}` (applies to code **and** weights; `unknown` never matches), `--max-vram-mb` / `--vram-tier`, `--hardware-class`, `--runtime`, `--max-latency-ms`.
Hardware filters only match **measured** records — nothing is assumed to fit without a benchmark.

## Principles

1. **Compare, don't hoard.** A small number of models with comparable, trustworthy information beats a large number of opaque files.
2. **License per component.** Source code, pretrained weights, training data and runtime dependencies are recorded separately, each with an evidence link. When upstream says nothing, we write `unknown` — we never infer "commercial use OK".
3. **No number without conditions.** VRAM, latency and FPS are only written by [`tools/benchmark.py`](tools/benchmark.py), together with hardware, runtime, precision, batch size, input shape and the artifact's SHA-256. Nothing is estimated.
4. **Reproducible from source.** Each model pins the upstream commit and either downloads a hash-checked file or runs an `export.py`. `weights/provenance.json` records where every artifact came from.
5. **Attributes, not folders.** License, VRAM, hardware, runtime and precision are metadata, not directory levels, so a model never has to be in two places.

### VRAM tiers

| Tier | Peak VRAM | Typical hardware |
|---|---|---|
| Tiny | ≤ 2 GB | Edge devices, iGPU, any discrete GPU |
| Light | ≤ 4 GB | Low-end GPUs, affordable laptops |
| Consumer | ≤ 8 GB | Gaming laptops, mainstream desktop GPUs |
| Performance | ≤ 16 GB | Upper consumer desktop GPUs |
| Large | ≤ 24 GB | High-end consumer GPUs |
| Huge | > 24 GB | Workstation / data center |

A tier is a property of a **measurement**, not of a model: the same model can be Tiny at FP16/batch 1 and Consumer at FP32/batch 16. See [docs/hardware.md](docs/hardware.md).

## Quick start

```bash
pip install -r requirements.txt          # onnxruntime-gpu, opencv, pyyaml, nvidia-ml-py, torch...

# 1. get the ONNX files (download + sha256 check, or export from pinned upstream)
python tools/fetch_model.py --task object_detection
#    Some exporters need extra packages; keep each in its own venv and pass --python
#    (instructions at the top of each model's export.py):
#      yolo11n, yolo26n            -> ultralytics (AGPL-3.0)
#      deim_dfine_s, rtdetrv4_*    -> upstream repo deps (tensorboard, faster-coco-eval, calflops, gdown)
#      rfdetr_*                    -> rfdetr==1.11.0
#    e.g. python tools/fetch_model.py yolo26n --python .venv-ultralytics/Scripts/python

# 2. run one model on a video -> outputs/object_detection/<model>/<clip>/{annotated.mp4,detections.jsonl}
python tools/run_video.py --model yolox_s --input assets/demo.mp4

# 3. rebuild the comparison GIF
python tools/make_comparison.py --task object_detection --input assets/demo.mp4

# 4. benchmark on your machine and contribute the record
python tools/benchmark.py --task object_detection --provider cuda \
    --hardware-label "RTX 4060 Laptop" --hardware-class gaming_laptop

# 5. regenerate tables, validate metadata
python tools/build_readme.py && python tools/validate.py
```

## Layout

```text
ml_model_collection/
├── assets/                  demo clip + generated comparison GIF (with attribution)
├── docs/                    licenses, hardware/VRAM, metadata, adding a model
├── object_detection/
│   └── <model>/
│       ├── model.yaml       curated metadata: source, licenses, artifacts, reported accuracy
│       ├── benchmarks.yaml  measured records (generated by tools/benchmark.py)
│       ├── detector.py      ONNX pre/post-processing -> common Detections output
│       ├── export.py        how the ONNX file is produced (if not downloaded)
│       └── weights/         fetched artifacts + provenance.json (git-ignored)
└── tools/                   fetch, run, compare, benchmark, find, validate, build_readme
```

Other tasks (`segmentation/`, `depth_estimation/`, `pose_estimation/`, `optical_flow/`) will be added when there are models for them. The inner structure of `object_detection/` is deliberately flat for now and will be revisited once more models show what is actually shared ([docs/metadata.md](docs/metadata.md)).

## Contributing

Adding a model or a benchmark on your hardware is the most useful contribution — see [docs/adding_a_model.md](docs/adding_a_model.md).

## License

The code and documentation in this repository are licensed under [Apache-2.0](LICENSE).
**Models are not.** Each model keeps its upstream licenses — check the code *and* weights license of every model before use. Nothing in this repository is legal advice.
