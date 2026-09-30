"""Measure latency and peak VRAM of a model and record the conditions.

    python tools/benchmark.py --model yolox_s --provider cuda \
        --hardware-label "GTX 1660 Ti Laptop" --hardware-class gaming_laptop
    python tools/benchmark.py --task object_detection --provider cpu \
        --hardware-label "Core i7-9750H" --hardware-class cpu_only

A record is appended to ``<task>/<model>/benchmarks.yaml`` (replacing a
previous record with the same id). Numbers are only ever written by this
tool, never by hand, and always together with the conditions they were
measured under:

  * latency_ms      model forward only (``session.run``), batch 1
  * e2e_ms          preprocess + forward + postprocess on a real frame
  * peak_vram_mb    see tools/mlmc/vram.py; covers session creation, warm-up
                    and all timed runs; the CUDA context is included

Each model runs in its own subprocess so VRAM baselines do not leak between
models. The machine must be otherwise idle: before each model, CPU load and
GPU utilisation are sampled, recorded in the record (``system_load``), and
the run is refused above ``--max-cpu-load`` / ``--max-gpu-util`` (ORT CUDA
latency is also CPU-bound, so CPU contention distorts GPU numbers too).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.mlmc import REPO_ROOT, hardware, vram  # noqa: E402
from tools.mlmc.catalog import Model, all_models, get_model  # noqa: E402
from tools.mlmc.detection import PROVIDERS, provider_precision  # noqa: E402


def ort_package() -> str:
    for name in ("onnxruntime-gpu", "onnxruntime"):
        try:
            return f"{name} {md.version(name)}"
        except md.PackageNotFoundError:
            continue
    return "unknown"


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")


def system_load(seconds: float = 3.0) -> dict:
    load = {"cpu_percent": None, "gpu_util_percent": None}
    try:
        import psutil
        load["cpu_percent"] = round(psutil.cpu_percent(interval=seconds), 1)
    except ImportError:
        pass
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        load["gpu_util_percent"] = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
    except Exception:
        pass
    return load


def check_idle(args) -> dict:
    load = system_load()
    busy = []
    if load["cpu_percent"] is None:
        busy.append("cannot measure CPU load (pip install psutil)")
    elif load["cpu_percent"] > args.max_cpu_load:
        busy.append(f"CPU load {load['cpu_percent']}% > {args.max_cpu_load}%")
    if load["gpu_util_percent"] is not None and load["gpu_util_percent"] > args.max_gpu_util:
        busy.append(f"GPU utilisation {load['gpu_util_percent']}% > {args.max_gpu_util}%")
    if busy and not args.allow_busy:
        raise SystemExit("machine is not idle: " + "; ".join(busy) +
                         " — stop other workloads, or pass --allow-busy (not for published numbers)")
    return load


def measure(model: Model, provider: str, frame_path: Path, warmup: int,
            iters: int, args) -> dict:
    if provider.startswith("tensorrt"):
        # Build (or load) the TensorRT engine in a separate process first, so
        # the builder's temporary workspace is not counted as inference VRAM.
        subprocess.run([sys.executable, __file__, "--model", model.name,
                        "--provider", provider, "--prebuild-only",
                        "--hardware-label", "-", "--hardware-class", args.hardware_class],
                       check=True)
    load = check_idle(args)
    frame = read_frame(frame_path)
    gpu = provider != "cpu"
    mon = vram.PeakVramMonitor() if gpu else None
    if mon:
        mon.__enter__()
    try:
        det = model.load_runner(provider=provider)
        x, meta = det.preprocess(frame)
        for _ in range(warmup):
            det.infer(x)
        lat = []
        for _ in range(iters):
            t0 = time.perf_counter()
            det.infer(x)
            lat.append((time.perf_counter() - t0) * 1000)
        e2e = []
        for _ in range(max(iters // 4, 10)):
            t0 = time.perf_counter()
            det.postprocess(det.infer(det.preprocess(frame)[0]), meta)
            e2e.append((time.perf_counter() - t0) * 1000)
    finally:
        if mon:
            mon.__exit__(None, None, None)

    lat.sort()
    mean = statistics.fmean(lat)
    art = model.meta["artifacts"]["onnx"]
    prov_file = model.weights_dir / "provenance.json"
    sha = json.loads(prov_file.read_text()).get("sha256") if prov_file.exists() else None
    gpu_info = hardware.gpu_info() if gpu else None
    runtime = "onnxruntime-" + provider.split("-")[0]
    precision = provider_precision(provider)
    rec = {
        "id": f"{slug(args.hardware_label)}_{runtime}_{precision}_b1",
        "date": dt.date.today().isoformat(),
        "hardware": {
            "label": args.hardware_label,
            "class": args.hardware_class,
            "gpu": gpu_info["name"] if gpu_info else None,
            "gpu_vram_total_mb": gpu_info["vram_total_mb"] if gpu_info else None,
            "gpu_driver": gpu_info["driver"] if gpu_info else None,
            "cpu": hardware.cpu_name(),
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        },
        "runtime": runtime,
        "runtime_version": ort_package(),
        "execution_provider": det.sess.get_providers()[0],
        "precision": precision,
        "batch_size": 1,
        "input_shape": list((x[0] if isinstance(x, tuple) else x).shape),
        "artifact": {"file": art["file"], "sha256": sha},
        "warmup_iters": warmup,
        "timed_iters": iters,
        "latency_ms": {"mean": round(mean, 2),
                       "p50": round(lat[len(lat) // 2], 2),
                       "p90": round(lat[int(len(lat) * 0.9)], 2)},
        "fps": round(1000 / mean, 1),
        "e2e_ms_mean": round(statistics.fmean(e2e), 2),
        "e2e_frame": f"{frame_path.name} ({frame.shape[1]}x{frame.shape[0]})",
        "peak_vram_mb": round(mon.delta_mb) if mon else None,
        "vram_method": vram.METHOD if mon else None,
        "vram_tier": hardware.vram_tier(mon.delta_mb) if mon else None,
        "system_load": load,  # sampled right before the run
    }
    if provider.startswith("tensorrt"):
        # ORT may run unsupported nodes on CUDA; engine built beforehand.
        rec["notes"] = "ONNX Runtime TensorRT EP; engine built and cached before timing"
    return rec


def read_frame(path: Path):
    if path.suffix.lower() in (".mp4", ".webm", ".avi", ".mov", ".mkv"):
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"cannot read {path}")
        return frame
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"cannot read {path}")
    return frame


def save(model: Model, rec: dict):
    path = model.dir / "benchmarks.yaml"
    records = []
    if path.exists():
        records = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    records = [r for r in records if r.get("id") != rec["id"]] + [rec]
    records.sort(key=lambda r: r["id"])
    header = ("# GENERATED by tools/benchmark.py — do not edit by hand.\n"
              "# Every number is tied to the conditions recorded next to it.\n")
    path.write_text(header + yaml.safe_dump(records, sort_keys=False,
                                            allow_unicode=True), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--task")
    ap.add_argument("--provider", default="cuda", choices=list(PROVIDERS))
    ap.add_argument("--prebuild-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--hardware-label", required=True,
                    help='human-readable name, e.g. "GTX 1660 Ti Laptop"')
    ap.add_argument("--hardware-class", required=True, choices=hardware.HARDWARE_CLASSES)
    ap.add_argument("--input", type=Path, default=REPO_ROOT / "assets" / "demo.mp4",
                    help="image or video (first frame) used for e2e timing")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true", help="print, do not save")
    ap.add_argument("--max-cpu-load", type=float, default=15.0,
                    help="refuse to run above this CPU load (%%)")
    ap.add_argument("--max-gpu-util", type=float, default=10.0,
                    help="refuse to run above this GPU utilisation (%%)")
    ap.add_argument("--allow-busy", action="store_true",
                    help="run anyway; the load is still recorded")
    args = ap.parse_args()

    if args.model and args.prebuild_only:
        runner = get_model(args.model).load_runner(provider=args.provider)
        x, _ = runner.preprocess(read_frame(args.input))
        runner.infer(x)
        return
    if args.model:
        model = get_model(args.model)
        rec = measure(model, args.provider, args.input, args.warmup, args.iters, args)
        print(yaml.safe_dump(rec, sort_keys=False))
        if not args.dry_run:
            save(model, rec)
        return

    # One subprocess per model so each gets a clean VRAM baseline.
    models = [m for m in all_models(args.task) if m.artifact_path("onnx").exists()]
    passthrough = [a for a in sys.argv[1:]]
    i = passthrough.index("--task") if "--task" in passthrough else None
    if i is not None:
        del passthrough[i:i + 2]
    for m in models:
        print(f"== {m.name}")
        subprocess.run([sys.executable, __file__, "--model", m.name, *passthrough],
                       check=True)


if __name__ == "__main__":
    main()
