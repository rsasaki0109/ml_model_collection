# Hardware, VRAM and benchmarks

## A measurement, not a model property

Peak VRAM, latency and FPS depend on **hardware, runtime (and its version),
execution provider, precision, batch size, input resolution and model
variant**. `tools/benchmark.py` therefore always writes all of these into the
record, plus the SHA-256 of the artifact that was measured. Numbers that were
not produced this way are not accepted.

## VRAM tiers

| Tier | Peak VRAM |
|---|---|
| Tiny | ≤ 2 GB |
| Light | ≤ 4 GB |
| Consumer | ≤ 8 GB |
| Performance | ≤ 16 GB |
| Large | ≤ 24 GB |
| Huge | > 24 GB |

The tier is computed from a record's `peak_vram_mb` (`tools/mlmc/hardware.py`).

## Hardware classes

Set by the person running the benchmark with `--hardware-class`. Guidelines:

| class | Typical examples |
|---|---|
| `edge_device` | Jetson, Raspberry Pi + accelerator, phones |
| `igpu` | Intel UHD/Iris Xe, AMD Radeon integrated graphics |
| `low_end_gpu` | Entry discrete GPUs with ≤ 4 GB |
| `affordable_laptop` | Thin-and-light / budget laptops with entry dGPU |
| `gaming_laptop` | Laptops with GTX 16xx / RTX xx50–xx70 class mobile GPUs |
| `consumer_desktop_gpu` | Desktop RTX xx60–xx70 class |
| `high_end_gpu` | Desktop RTX xx80–xx90 class |
| `workstation` | RTX A-series / data-center GPUs |
| `cpu_only` | Records measured on the CPU execution provider |

Current records: NVIDIA Tesla T4 on Google Colab (`workstation`), measured
with `tools/colab_benchmark.ipynb`. Records for a GTX 1660 Ti laptop
(`gaming_laptop`) and its Core i7-9750H CPU (`cpu_only`) are pending: the
first attempt was discarded because other workloads were running on the
machine (see the idle guard in `tools/benchmark.py`).

## How VRAM is measured

`tools/mlmc/vram.py`, method `nvml_device_used_delta`:

1. Read device-wide used memory via NVML → baseline.
2. Poll it every 5 ms in a background thread while the ONNX Runtime session is
   created, warmed up and run.
3. Report `peak − baseline`.

Consequences:

- The CUDA context and runtime workspaces are included (this is what you need
  to have free to run the model).
- Other processes allocating GPU memory during the run distort the number —
  benchmark on an idle GPU.
- Per-process accounting is not used because it is unavailable on Windows
  (WDDM). Sanity check on the reference machine: allocating 1 GiB with
  PyTorch reports 1089 MB (1024 MB + CUDA context).
- ONNX Runtime's CUDA arena is configured with
  `arena_extend_strategy=kSameAsRequested` to avoid power-of-two
  over-allocation.

## TensorRT

`--provider tensorrt` / `tensorrt-fp16` use ONNX Runtime's TensorRT execution
provider (nodes TensorRT cannot take run on CUDA). The engine is built and
cached (`weights/trt_cache/`) in a separate process *before* the measured
run, so builder workspace memory is not counted as inference VRAM. Records
carry `precision: fp16` and a `notes` field. These are the closest
equivalent to upstream "T4 / TensorRT / FP16" tables, but ORT's partitioning
means they are not identical to a pure `trtexec` engine.

## Latency

- `latency_ms`: `session.run` only, after 20 warm-up runs, 200 timed runs
  (CPU records may use fewer; stored in the record).
- `e2e_ms_mean`: preprocess + inference + postprocess on the first frame of
  `assets/demo.mp4` (1280×720), in Python/NumPy. Useful as a rough "what you
  get in practice" number, but it depends on the CPU too.

## Benchmarking on Google Colab

For GPUs you do not own (or when your machine is busy):

1. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rsasaki0109/ml_model_collection/blob/main/tools/colab_benchmark.ipynb)
   — select a GPU runtime and run all cells (it clones this repository; for
   a fork or local changes, set `REPO_URL` or upload the zip from
   `python tools/pack_for_colab.py`).
2. The notebook exports every artifact from
   the pinned sources inside Colab, benchmarks it, and downloads
   `benchmarks_colab.zip`.
3. `python tools/merge_benchmarks.py benchmarks_colab.zip && python tools/build_readme.py`

Records are labelled e.g. `Tesla T4 (Colab)` / `workstation`. Colab CPUs are
shared vCPUs, so CPU records are disabled by default.
