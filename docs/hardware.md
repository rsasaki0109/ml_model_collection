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

The first records in this repo were measured on a laptop with an
NVIDIA GeForce GTX 1660 Ti (6 GB) and an Intel Core i7-9750H, Windows 11,
labelled `gaming_laptop` (GPU) and `cpu_only` (CPU).

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

## Latency

- `latency_ms`: `session.run` only, after 20 warm-up runs, 200 timed runs
  (CPU records may use fewer; stored in the record).
- `e2e_ms_mean`: preprocess + inference + postprocess on the first frame of
  `assets/demo.mp4` (1280×720), in Python/NumPy. Useful as a rough "what you
  get in practice" number, but it depends on the CPU too.

## Benchmarking on Google Colab

For GPUs you do not own (or when your machine is busy):

1. `python tools/pack_for_colab.py` → `dist/ml_model_collection.zip`
   (skip if the repository can be `git clone`d; set `REPO_URL` instead).
2. Open [`tools/colab_benchmark.ipynb`](../tools/colab_benchmark.ipynb) in
   Colab with a GPU runtime and run all cells. It exports every artifact from
   the pinned sources inside Colab, benchmarks it, and downloads
   `benchmarks_colab.zip`.
3. `python tools/merge_benchmarks.py benchmarks_colab.zip && python tools/build_readme.py`

Records are labelled e.g. `Tesla T4 (Colab)` / `workstation`. Colab CPUs are
shared vCPUs, so CPU records are disabled by default.
