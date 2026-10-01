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
from tools.mlmc.catalog import Model, all_models  # noqa: E402


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


OD, DE, SG, PE, OF, SR = ("object_detection", "depth_estimation", "segmentation", "pose_estimation",
                          "optical_flow", "super_resolution")
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
    (REPO_ROOT / "README.md", "pose_estimation_table"): (PE, pose_table),
    (REPO_ROOT / PE / "README.md", "pose_estimation_table"): (PE, pose_table),
    (REPO_ROOT / PE / "README.md", "pose_estimation_provenance"): (PE, provenance_table),
    (REPO_ROOT / "README.md", "optical_flow_table"): (OF, flow_table),
    (REPO_ROOT / OF / "README.md", "optical_flow_table"): (OF, flow_table),
    (REPO_ROOT / OF / "README.md", "optical_flow_provenance"): (OF, provenance_table),
    (REPO_ROOT / "README.md", "super_resolution_table"): (SR, sr_table),
    (REPO_ROOT / SR / "README.md", "super_resolution_table"): (SR, sr_table),
    (REPO_ROOT / SR / "README.md", "super_resolution_provenance"): (SR, provenance_table),
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
