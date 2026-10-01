"""Regenerate the model tables in README files from model.yaml/benchmarks.yaml.

    python tools/build_readme.py          # rewrite tables in place
    python tools/build_readme.py --check  # exit 1 if a table is out of date (CI)

Tables are written between ``<!-- BEGIN:<name> -->`` / ``<!-- END:<name> -->``
markers so that the rest of each README stays hand-written.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.mlmc import REPO_ROOT, licenses  # noqa: E402
from tools.mlmc.catalog import Model, all_models, get_model  # noqa: E402


def rel(model: Model, start: Path) -> str:
    return os.path.relpath(model.dir, start).replace("\\", "/")


def bench_columns(models: list[Model]) -> list[tuple[str, str]]:
    """Distinct (hardware label, runtime) pairs, GPU runs first."""
    seen = {}
    for m in models:
        for b in m.benchmarks:
            key = (b["hardware"]["label"], b["runtime"], b.get("precision", "fp32"))
            seen[key] = b["peak_vram_mb"] is None
    return sorted(seen, key=lambda k: (seen[k], k))


def find_bench(model: Model, col):
    for b in model.benchmarks:
        if (b["hardware"]["label"], b["runtime"], b.get("precision", "fp32")) == col:
            return b
    return None


def accuracy(model: Model, metric_prefix: str) -> str:
    """First upstream-reported value whose metric name starts with the column's metric."""
    acc = [a for a in model.meta.get("reported_accuracy") or []
           if a["metric"].startswith(metric_prefix)]
    if not acc:
        return "–"
    a = acc[0]
    return f"[{a['value']}]({a['source']})"


def measured_coco(model: Model) -> str:
    """COCO val2017 AP measured by tools/evaluate.py on the exported artifact."""
    recs = [r for r in model.accuracy if r.get("dataset") == "COCO val2017"]
    if not recs:
        return "–"
    r = sorted(recs, key=lambda r: r.get("precision") != "fp32")[0]
    return f"**{r['metrics']['AP']}**"


def measured_seg(model: Model) -> str:
    """COCO mask AP (instance / promptable) or ADE20K mIoU (semantic), measured on the artifact."""
    for r in model.accuracy:
        if r.get("dataset") == "COCO val2017":
            extra = f" ({r['settings']['prompts']})" if r["settings"].get("prompts") else ""
            return f"**{r['metrics']['AP']}** COCO mask AP{extra}"
        if r.get("dataset") == "ADE20K val":
            return f"**{r['metrics']['mIoU']}** ADE20K mIoU"
    return "–"


def vram_cell(m: Model) -> str:
    vram = [b for b in m.benchmarks if b.get("peak_vram_mb") is not None]
    short = {"onnxruntime-cuda": "CUDA", "onnxruntime-tensorrt": "TRT", "onnxruntime-cpu": "CPU"}
    def hw(b):  # short hardware name, e.g. "T4" / "GTX 1660 Ti"
        return b["hardware"]["label"].replace(" (Colab)", "").replace(" Laptop", "").replace("Tesla ", "")

    cells = [f"{b['peak_vram_mb']} MB ({b['vram_tier']}, {hw(b)} "
             f"{short.get(b['runtime'], b['runtime'])} {b.get('precision', 'fp32').upper()})"
             for b in sorted(vram, key=lambda b: (b["hardware"]["label"], b["runtime"]))]
    return "<br>".join(cells) or "not measured"


def bench_cells(m: Model, cols) -> list[str]:
    out = []
    for col in cols:
        b = find_bench(m, col)
        out.append(f"{b['latency_ms']['mean']:.1f} / {b['fps']:.0f}" if b else "–")
    return out


def bench_head(cols) -> list[str]:
    return [f"{hw}<br>{rt} {prec.upper()} · ms / FPS" for hw, rt, prec in cols]


def comparison_table(models: list[Model], start: Path) -> str:
    cols = bench_columns(models)
    head = ["Model", "Code license", "Weights license", "Input",
            "COCO mAP<br>(reported)", "COCO mAP<br>(measured, ONNX)", "Peak VRAM<br>(measured)"]
    head += bench_head(cols)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in models:
        lic = m.meta.get("license", {})
        shape = m.meta["artifacts"]["onnx"]["input_shape"]
        name = f"[{m.display_name}]({rel(m, start)})"
        if m.meta.get("open_vocabulary"):
            name += " 🔤"
        if m.meta.get("known_issue"):
            name += " ⚠️"
        row = [name, licenses.describe(lic.get("code")), licenses.describe(lic.get("weights")),
               f"{shape[-2]}×{shape[-1]}", accuracy(m, "COCO"), measured_coco(m), vram_cell(m)]
        lines.append("| " + " | ".join(row + bench_cells(m, cols)) + " |")
    return "\n".join(lines)


OUTPUT_LABEL = {"relative_disparity": "relative (disparity)",
                "relative_depth": "relative (depth)", "metric_depth_m": "metric (m)"}


def depth_table(models: list[Model], start: Path) -> str:
    cols = bench_columns(models)
    head = ["Model", "Code license", "Weights license", "Output", "Input",
            "NYUv2 AbsRel ↓<br>(reported)", "Peak VRAM<br>(measured)"] + bench_head(cols)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in models:
        lic = m.meta.get("license", {})
        shape = m.meta["artifacts"]["onnx"]["input_shape"]
        row = [f"[{m.display_name}]({rel(m, start)})", licenses.describe(lic.get("code")),
               licenses.describe(lic.get("weights")),
               OUTPUT_LABEL.get(m.meta.get("output"), "?"), f"{shape[-2]}×{shape[-1]}",
               accuracy(m, "NYUv2 AbsRel"), vram_cell(m)]
        lines.append("| " + " | ".join(row + bench_cells(m, cols)) + " |")
    return "\n".join(lines)


def provenance_table(models: list[Model], start: Path) -> str:
    lines = ["| Model | Source (pinned) | Artifact | How it is produced |",
             "|---|---|---|---|"]
    for m in models:
        src = m.meta["source"]
        commit = src.get("commit", "")
        repo = src["repository"]
        art = m.meta["artifacts"]["onnx"]
        how = (f"download `{art['url'].rsplit('/', 1)[-1]}`, extract `{art['member'].rsplit('/', 1)[-1]}` (sha256 pinned)"
               if art["fetch"] == "download_zip"
               else f"download `{art['url'].rsplit('/', 1)[-1]}` (sha256 pinned)"
               if art["fetch"] == "download"
               else f"download {len(art['files'])} files (sha256 pinned)"
               if art["fetch"] == "download_files"
               else f"[`{art['script']}`]({rel(m, start)}/{art['script']})")
        lines.append(f"| {m.display_name} | [{repo.split('github.com/')[-1]}@{commit[:7]}]"
                     f"({repo}/tree/{commit}) | `{art['file']}` | {how} |")
    return "\n".join(lines)


def segmentation_table(models: list[Model], start: Path) -> str:
    cols = bench_columns(models)
    head = ["Model", "Kind", "Code license", "Weights license", "Input",
            "Accuracy (reported)", "Accuracy<br>(measured, ONNX)", "Peak VRAM<br>(measured)"] + bench_head(cols)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in models:
        lic = m.meta.get("license", {})
        shape = m.meta["artifacts"]["onnx"]["input_shape"]
        acc = (m.meta.get("reported_accuracy") or [None])[0]
        acc_cell = f"[{acc['value']}]({acc['source']}) {acc['metric'].split(' (')[0]}" if acc else "–"
        row = [f"[{m.display_name}]({rel(m, start)})", m.meta.get("kind", "?"),
               licenses.describe(lic.get("code")), licenses.describe(lic.get("weights")),
               f"{shape[-2]}×{shape[-1]}", acc_cell, measured_seg(m), vram_cell(m)]
        lines.append("| " + " | ".join(row + bench_cells(m, cols)) + " |")
    return "\n".join(lines)


def pose_table(models: list[Model], start: Path) -> str:
    cols = bench_columns(models)
    head = ["Model", "Kind", "Code license", "Weights license", "Input",
            "COCO kpt AP<br>(reported)", "COCO kpt AP<br>(measured, ONNX)", "Peak VRAM<br>(measured)"] + bench_head(cols)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in models:
        lic = m.meta.get("license", {})
        shape = m.meta["artifacts"]["onnx"]["input_shape"]
        meas = next((r for r in m.accuracy if r.get("dataset") == "COCO val2017 keypoints"), None)
        meas_cell = f"**{meas['metrics']['AP']}**" if meas else "–"
        if meas and meas["settings"].get("person_boxes"):
            meas_cell += f" (persons {meas['settings']['person_boxes']})"
        row = [f"[{m.display_name}]({rel(m, start)})", m.meta.get("kind", "?"),
               licenses.describe(lic.get("code")), licenses.describe(lic.get("weights")),
               f"{shape[-2]}×{shape[-1]}", accuracy(m, "COCO"), meas_cell, vram_cell(m)]
        cells = []
        for col in cols:
            b = find_bench(m, col)
            if not b:
                cells.append("–")
                continue
            crops = f" ({b['input_shape'][0]} crops)" if b["input_shape"][0] > 1 else ""
            cells.append(f"{b['latency_ms']['mean']:.1f} / {b['fps']:.0f}{crops}<br>e2e {b['e2e_ms_mean']:.1f} ms")
        lines.append("| " + " | ".join(row + cells) + " |")
    return "\n".join(lines)


def flow_table(models: list[Model], start: Path) -> str:
    cols = bench_columns(models)
    head = ["Model", "Code license", "Weights license", "Training data", "Input",
            "Sintel train EPE<br>clean / final (reported)", "Sintel train EPE<br>clean / final (measured, ONNX)",
            "Peak VRAM<br>(measured)"] + bench_head(cols)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in models:
        lic = m.meta.get("license", {})
        shape = m.meta["artifacts"]["onnx"]["input_shape"]
        rep = {r["metric"].split(" EPE")[0].rsplit(" ", 1)[-1]: r for r in m.meta.get("reported_accuracy", [])}
        rep_cell = (f"[{rep['clean']['value']} / {rep['final']['value']}]({rep['clean']['source']})"
                    if "clean" in rep and "final" in rep else "–")
        meas = next((r for r in m.accuracy if r.get("dataset") == "MPI-Sintel train"), None)
        meas_cell = (f"**{meas['metrics']['clean_EPE']:.2f} / {meas['metrics']['final_EPE']:.2f}**"
                     if meas else "–")
        row = [f"[{m.display_name}]({rel(m, start)})",
               licenses.describe(lic.get("code")), licenses.describe(lic.get("weights")),
               lic.get("dataset", {}).get("name", "?"), f"{shape[-2]}×{shape[-1]}",
               rep_cell, meas_cell, vram_cell(m)]
        lines.append("| " + " | ".join(row + bench_cells(m, cols)) + " |")
    return "\n".join(lines)


def sr_table(models: list[Model], start: Path) -> str:
    cols = bench_columns(models)
    sets = ("Set5", "Set14", "Urban100")
    head = ["Model", "Kind", "Code license", "Weights license", "Training data",
            "PSNR-Y x4 Set5 / Set14 / Urban100<br>(reported)",
            "PSNR-Y x4 Set5 / Set14 / Urban100<br>(measured, ONNX)",
            "Peak VRAM<br>(measured)"] + bench_head(cols)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in models:
        lic = m.meta.get("license", {})
        rep = {r["metric"].split()[0]: r for r in m.meta.get("reported_accuracy", [])}
        rep_cell = "–"
        if rep:
            vals = " / ".join(f"{rep[s]['value']:.2f}" if s in rep else "–" for s in sets)
            rep_cell = f"[{vals}]({next(iter(rep.values()))['source']})"
        meas = next((r for r in m.accuracy if r.get("dataset", "").startswith("SR benchmarks x4")), None)
        meas_cell = (" / ".join(f"**{meas['metrics'][f'{s}_PSNR_Y']:.2f}**" if f"{s}_PSNR_Y" in meas["metrics"]
                                else "–" for s in sets) if meas else "–")
        row = [f"[{m.display_name}]({rel(m, start)})", m.meta.get("kind", "?"),
               licenses.describe(lic.get("code")), licenses.describe(lic.get("weights")),
               lic.get("dataset", {}).get("name", "?"), rep_cell, meas_cell, vram_cell(m)]
        lines.append("| " + " | ".join(row + bench_cells(m, cols)) + " |")
    return "\n".join(lines)


def bg_table(models: list[Model], start: Path) -> str:
    cols = bench_columns(models)
    head = ["Model", "Kind", "Code license", "Weights license", "Training data", "Input",
            "DIS-VD S<sub>α</sub> / wF<br>(reported)", "DIS-VD S<sub>α</sub> / wF / MAE<br>(measured, ONNX)",
            "Peak VRAM<br>(measured)"] + bench_head(cols)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in models:
        lic = m.meta.get("license", {})
        shape = m.meta["artifacts"]["onnx"]["input_shape"]
        rep = m.meta.get("reported_accuracy", [])
        rep_cell = f"[{' / '.join(str(r['value']) for r in rep)}]({rep[0]['source']})" if rep else "–"
        meas = next((r for r in m.accuracy if r.get("dataset") == "DIS5K DIS-VD"), None)
        meas_cell = (f"**{meas['metrics']['S_measure']:.3f} / {meas['metrics']['weighted_F']:.3f}"
                     f" / {meas['metrics']['MAE']:.3f}**" if meas else "–")
        row = [f"[{m.display_name}]({rel(m, start)})", m.meta.get("kind", "?"),
               licenses.describe(lic.get("code")), licenses.describe(lic.get("weights")),
               lic.get("dataset", {}).get("name", "?"), f"{shape[-2]}×{shape[-1]}",
               rep_cell, meas_cell, vram_cell(m)]
        lines.append("| " + " | ".join(row + bench_cells(m, cols)) + " |")
    return "\n".join(lines)


def face_table(models: list[Model], start: Path) -> str:
    cols = bench_columns(models)
    head = ["Model", "Code license", "Weights license", "Input",
            "WIDER FACE val AP E / M / H<br>(reported)", "WIDER FACE val AP E / M / H<br>(measured, ONNX)",
            "Peak VRAM<br>(measured)"] + bench_head(cols)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in models:
        lic = m.meta.get("license", {})
        art = m.meta["artifacts"]["onnx"]
        rep = m.meta.get("reported_accuracy", [])
        rep_cell = f"[{' / '.join(f'{r['value']:.1f}' for r in rep)}]({rep[0]['source']})" if rep else "–"
        meas = next((r for r in m.accuracy if r.get("dataset") == "WIDER FACE val"), None)
        meas_cell = (" / ".join(f"**{meas['metrics'][f'AP_{s}']:.1f}**" for s in ("easy", "medium", "hard"))
                     if meas else "–")
        inp = "640×640" if art["input_shape"][-1] == 640 else "source size"
        row = [f"[{m.display_name}]({rel(m, start)})",
               licenses.describe(lic.get("code")), licenses.describe(lic.get("weights")),
               inp, rep_cell, meas_cell, vram_cell(m)]
        lines.append("| " + " | ".join(row + bench_cells(m, cols)) + " |")
    return "\n".join(lines)


def ocr_table(models: list[Model], start: Path) -> str:
    cols = bench_columns(models)
    head = ["Model", "Code license", "Weights license", "Languages",
            "PaddleOCR benchmark det Hmean / rec acc<br>(reported, not ICDAR)",
            "ICDAR2015 det H-mean / end-to-end H-mean<br>(measured, ONNX)",
            "Peak VRAM<br>(measured)"] + bench_head(cols)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in models:
        lic = m.meta.get("license", {})
        rep = m.meta.get("reported_accuracy", [])
        det = next((r for r in rep if "det" in r["metric"]), None)
        rec = next((r for r in rep if "rec" in r["metric"]), None)
        rep_cell = " / ".join(f"[{r['value']}]({r['source']})" if r else "–" for r in (det, rec))
        meas = next((r for r in m.accuracy if r.get("dataset") == "ICDAR2015 test"), None)
        meas_cell = (f"**{meas['metrics']['det_hmean']} / {meas['metrics']['e2e_hmean']}**" if meas else "–")
        row = [f"[{m.display_name}]({rel(m, start)})",
               licenses.describe(lic.get("code")), licenses.describe(lic.get("weights")),
               m.meta.get("languages", "?"), rep_cell, meas_cell, vram_cell(m)]
        lines.append("| " + " | ".join(row + bench_cells(m, cols)) + " |")
    return "\n".join(lines)


def matching_table(models: list[Model], start: Path) -> str:
    cols = bench_columns(models)
    head = ["Model", "Code license", "Weights license", "Training data",
            "HPatches H-AUC @1/3/5 px<br>(reported, DLT)", "HPatches H-AUC @1/3/5 px<br>(measured, ONNX)",
            "Peak VRAM<br>(measured)"] + bench_head(cols)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in models:
        lic = m.meta.get("license", {})
        rep = m.meta.get("reported_accuracy", [])
        rep_cell = f"[{' / '.join(str(r['value']) for r in rep)}]({rep[0]['source']})" if rep else "–"
        meas = next((r for r in m.accuracy if r.get("dataset") == "HPatches"), None)
        meas_cell = (" / ".join(f"**{meas['metrics'][f'H_AUC@{t}px']}**" for t in (1, 3, 5)) if meas else "–")
        row = [f"[{m.display_name}]({rel(m, start)})",
               licenses.describe(lic.get("code")), licenses.describe(lic.get("weights")),
               lic.get("dataset", {}).get("name", "?"), rep_cell, meas_cell, vram_cell(m)]
        lines.append("| " + " | ".join(row + bench_cells(m, cols)) + " |")
    return "\n".join(lines)


def tracking_table(models: list[Model], start: Path) -> str:
    cols = bench_columns(models)
    head = ["Tracker", "Code license", "Detector", "Appearance / camera motion",
            "MOT17 test HOTA / MOTA / IDF1<br>(reported, MOT-trained detector)",
            "MOT17 train HOTA / MOTA / IDF1<br>(measured, our detector)"] + bench_head(cols)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in models:
        lic = m.meta.get("license", {})
        rep = m.meta.get("reported_accuracy", [])
        rep_cell = f"[{' / '.join(str(r['value']) for r in rep)}]({rep[0]['source']})" if rep else "–"
        meas = next((r for r in m.accuracy if r.get("dataset") == "MOT17 train"), None)
        meas_cell = (" / ".join(f"**{meas['metrics'][k]}**" for k in ("HOTA", "MOTA", "IDF1")) if meas else "–")
        algo = m.meta["tracker"]["algorithm"]
        extra = {"botsort": "no ReID; sparse-flow camera motion", "bytetrack": "motion only",
                 "ocsort": "motion only (observation-centric)"}.get(algo, "?")
        det = get_model(m.meta["detector"])
        row = [f"[{m.display_name}]({rel(m, start)})", licenses.describe(lic.get("code")),
               f"[{det.display_name}]({rel(det, start)})", extra, rep_cell, meas_cell]
        lines.append("| " + " | ".join(row + bench_cells(m, cols)) + " |")
    return "\n".join(lines)


# task -> (section title, measured metric: dataset prefix, key, higher is better, label)
TASK_INDEX = {
    "object_detection": ("Object detection", ("COCO val2017", "AP", True, "COCO AP")),
    "segmentation": ("Segmentation", ("COCO val2017", "AP", True, "COCO mask AP")),
    "depth_estimation": ("Depth estimation", None),
    "pose_estimation": ("Pose estimation", ("COCO val2017 keypoints", "AP", True, "COCO keypoint AP")),
    "optical_flow": ("Optical flow", ("MPI-Sintel train", "final_EPE", False, "Sintel final EPE")),
    "super_resolution": ("Super-resolution (x4)", ("SR benchmarks x4", "Urban100_PSNR_Y", True, "Urban100 PSNR-Y")),
    "background_removal": ("Background removal / matting", ("DIS5K DIS-VD", "S_measure", True, "DIS-VD S-measure")),
    "face_detection": ("Face detection", ("WIDER FACE val", "AP_hard", True, "WIDER FACE hard AP")),
    "ocr": ("OCR (scene text)", ("ICDAR2015 test", "e2e_hmean", True, "ICDAR2015 end-to-end H-mean")),
    "feature_matching": ("Feature matching", ("HPatches", "H_AUC@3px", True, "HPatches H-AUC@3px")),
    "tracking": ("Multi-object tracking", ("MOT17 train", "HOTA", True, "MOT17-train HOTA")),
}


def _anchor(title: str) -> str:
    return "#" + "".join(c for c in title.lower().replace(" ", "-") if c.isalnum() or c == "-")


def task_index(models: list[Model], start: Path) -> str:
    """One row per task: model count, weights-license mix, best measured
    accuracy and lowest T4 TensorRT FP16 latency (each from recorded files)."""
    head = ["Task", "Models", "Weights licenses", "Best measured accuracy", "Fastest on T4 TensorRT FP16<br>(model-only latency)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for task, (title, spec) in TASK_INDEX.items():
        ms = [m for m in models if m.task == task]
        if not ms:
            continue
        cats = {}
        for m in ms:
            c = licenses.category(m.meta.get("license", {}).get("weights", {}).get("spdx"))
            cats[c] = cats.get(c, 0) + 1
        mix = " ".join(f"{licenses.BADGE[c]}{n}" for c, n in sorted(cats.items(), key=lambda kv: -kv[1]))
        best = "–"
        if spec:
            ds, key, higher, label = spec
            vals = [(r["metrics"][key], m) for m in ms for r in m.accuracy
                    if key in r["metrics"] and (r.get("dataset") == ds
                                                or r.get("dataset", "").startswith(ds + " ("))]
            if vals:
                v, m = (max if higher else min)(vals, key=lambda t: t[0])
                best = f"{label} **{v}** — {m.display_name}"
        trt = [(b["latency_ms"]["mean"], m) for m in ms for b in m.benchmarks
               if b.get("hardware", {}).get("gpu") == "Tesla T4" and b.get("runtime") == "onnxruntime-tensorrt"
               and b.get("precision") == "fp16"]
        fast = "–"
        if trt:
            t, m = min(trt, key=lambda x: x[0])
            fast = f"{m.display_name} — {t:.1f} ms"
        lines.append(f"| [{title}]({_anchor(title)}) | {len(ms)} | {mix} | {best} | {fast} |")
    return "\n".join(lines)


OD, DE, SG, PE, OF, SR, BG = ("object_detection", "depth_estimation", "segmentation",
                              "pose_estimation", "optical_flow", "super_resolution",
                              "background_removal")
TABLES = {
    (REPO_ROOT / "README.md", "object_detection_table"): (OD, comparison_table),
    (REPO_ROOT / "README.md", "depth_estimation_table"): (DE, depth_table),
    (REPO_ROOT / OD / "README.md", "object_detection_table"): (OD, comparison_table),
    (REPO_ROOT / OD / "README.md", "object_detection_provenance"): (OD, provenance_table),
    (REPO_ROOT / DE / "README.md", "depth_estimation_table"): (DE, depth_table),
    (REPO_ROOT / DE / "README.md", "depth_estimation_provenance"): (DE, provenance_table),
    (REPO_ROOT / "README.md", "segmentation_table"): (SG, segmentation_table),
    (REPO_ROOT / SG / "README.md", "segmentation_table"): (SG, segmentation_table),
    (REPO_ROOT / SG / "README.md", "segmentation_provenance"): (SG, provenance_table),
    (REPO_ROOT / "README.md", "task_index"): (None, task_index),
    (REPO_ROOT / "README.md", "pose_estimation_table"): (PE, pose_table),
    (REPO_ROOT / PE / "README.md", "pose_estimation_table"): (PE, pose_table),
    (REPO_ROOT / PE / "README.md", "pose_estimation_provenance"): (PE, provenance_table),
    (REPO_ROOT / "README.md", "optical_flow_table"): (OF, flow_table),
    (REPO_ROOT / OF / "README.md", "optical_flow_table"): (OF, flow_table),
    (REPO_ROOT / OF / "README.md", "optical_flow_provenance"): (OF, provenance_table),
    (REPO_ROOT / "README.md", "super_resolution_table"): (SR, sr_table),
    (REPO_ROOT / SR / "README.md", "super_resolution_table"): (SR, sr_table),
    (REPO_ROOT / SR / "README.md", "super_resolution_provenance"): (SR, provenance_table),
    (REPO_ROOT / "README.md", "background_removal_table"): (BG, bg_table),
    (REPO_ROOT / BG / "README.md", "background_removal_table"): (BG, bg_table),
    (REPO_ROOT / BG / "README.md", "background_removal_provenance"): (BG, provenance_table),
    (REPO_ROOT / "README.md", "face_detection_table"): ("face_detection", face_table),
    (REPO_ROOT / "face_detection" / "README.md", "face_detection_table"): ("face_detection", face_table),
    (REPO_ROOT / "face_detection" / "README.md", "face_detection_provenance"):
        ("face_detection", provenance_table),
    (REPO_ROOT / "README.md", "ocr_table"): ("ocr", ocr_table),
    (REPO_ROOT / "ocr" / "README.md", "ocr_table"): ("ocr", ocr_table),
    (REPO_ROOT / "ocr" / "README.md", "ocr_provenance"): ("ocr", provenance_table),
    (REPO_ROOT / "README.md", "feature_matching_table"): ("feature_matching", matching_table),
    (REPO_ROOT / "feature_matching" / "README.md", "feature_matching_table"): ("feature_matching", matching_table),
    (REPO_ROOT / "feature_matching" / "README.md", "feature_matching_provenance"):
        ("feature_matching", provenance_table),
    (REPO_ROOT / "README.md", "tracking_table"): ("tracking", tracking_table),
    (REPO_ROOT / "tracking" / "README.md", "tracking_table"): ("tracking", tracking_table),
}


def render(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN:{name} -->\n).*?(<!-- END:{name} -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"marker {name} not found")
    return pat.sub(lambda mt: mt.group(1) + body + "\n" + mt.group(2), text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    stale = []
    files: dict[Path, str] = {}
    for (path, name), (task, fn) in TABLES.items():
        text = files.get(path) or path.read_text(encoding="utf-8")
        files[path] = render(text, name, fn(all_models(task), path.parent))
    for path, new in files.items():
        if new != path.read_text(encoding="utf-8"):
            stale.append(path)
            if not args.check:
                path.write_text(new, encoding="utf-8", newline="\n")
    if args.check and stale:
        print("out of date:", *[p.relative_to(REPO_ROOT) for p in stale])
        sys.exit(1)
    print("updated:" if stale else "up to date", *[p.relative_to(REPO_ROOT) for p in stale])


if __name__ == "__main__":
    main()
