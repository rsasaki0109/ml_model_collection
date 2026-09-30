# Metadata

The schema is intentionally **not frozen**. Fields are added when a real
model needs them, and only the fields in `tools/validate.py` are required.

## `model.yaml` (curated by hand)

```yaml
name: yolox_s                 # == directory name
display_name: YOLOX-S         # shown in tables and the comparison GIF
task: object_detection        # == parent directory
summary: ...

source:
  repository: https://github.com/...
  commit: <full sha>          # upstream HEAD when the model was added
  paper: https://arxiv.org/...
  # optional: weights_repository, weights_revision, weights_url

license:                      # see docs/licenses.md
  code:    {spdx: Apache-2.0, status: explicit, evidence: <url>}
  weights: {spdx: Apache-2.0, status: repository_license, evidence: <url>, note: ...}
  dataset: {name: COCO 2017, note: ...}
  export_dependencies: [{name: transformers, spdx: Apache-2.0}]

artifacts:
  onnx:
    file: yolox_s.onnx
    fetch: download | export
    url: ...                  # download
    sha256: ...               # download (pinned)
    script: export.py         # export
    input_shape: [1, 3, 640, 640]
    input_format: ...         # colour order, resize/letterbox, normalisation
    output: ...

runtimes:
  onnxruntime: verified       # verified = run in this repo

reported_accuracy:            # copied from upstream, never re-labelled as ours
  - {metric: ..., value: 40.5, input_size: 640, source: <url>}
```

## `benchmarks.yaml` (generated)

Written only by `tools/benchmark.py`. One record per
`(hardware label, runtime, precision, batch)`; re-running replaces the record.
See [hardware.md](hardware.md) for what each number means.

## Things deliberately left open

- **Directory structure below `object_detection/`.** Flat for now. Grouping by
  license, VRAM, hardware, runtime or precision as directories is avoided
  because those attributes are independent and would put one model in several
  places.
- **Multiple variants / precisions per model** (e.g. FP16, INT8, TensorRT
  engines). The `artifacts` map is keyed by format so more entries can be
  added; whether variants become separate model directories will be decided
  once there is a second variant.
- **Common detector base class.** The four detectors share helpers
  (`tools/mlmc/detection.py`) but not a base class yet.
