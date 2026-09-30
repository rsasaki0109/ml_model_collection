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
            key = (b["hardware"]["label"], b["runtime"])
            seen[key] = b["peak_vram_mb"] is None
    return sorted(seen, key=lambda k: (seen[k], k))


def find_bench(model: Model, col):
    for b in model.benchmarks:
        if (b["hardware"]["label"], b["runtime"]) == col:
            return b
    return None


def accuracy(model: Model) -> str:
    acc = model.meta.get("reported_accuracy") or []
    if not acc:
        return "–"
    a = acc[0]
    return f"[{a['value']}]({a['source']})"


def comparison_table(models: list[Model], start: Path) -> str:
    cols = bench_columns(models)
    head = ["Model", "Code license", "Weights license", "Input",
            "COCO mAP<br>(reported)", "Peak VRAM<br>(measured)"]
    head += [f"{hw}<br>{rt} · ms / FPS" for hw, rt in cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in models:
        lic = m.meta.get("license", {})
        shape = m.meta["artifacts"]["onnx"]["input_shape"]
        vram = [b for b in m.benchmarks if b.get("peak_vram_mb") is not None]
        vram_cell = "<br>".join(
            f"{b['peak_vram_mb']} MB ({b['vram_tier']})" for b in vram) or "not measured"
        row = [f"[{m.display_name}]({rel(m, start)})",
               licenses.describe(lic.get("code")),
               licenses.describe(lic.get("weights")),
               f"{shape[2]}×{shape[3]}",
               accuracy(m), vram_cell]
        for col in cols:
            b = find_bench(m, col)
            row.append(f"{b['latency_ms']['mean']:.1f} / {b['fps']:.0f}" if b else "–")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def provenance_table(models: list[Model], start: Path) -> str:
    lines = ["| Model | Source (pinned) | Artifact | How it is produced |",
             "|---|---|---|---|"]
    for m in models:
        src = m.meta["source"]
        commit = src.get("commit", "")
        repo = src["repository"]
        art = m.meta["artifacts"]["onnx"]
        how = (f"download `{art['url'].rsplit('/', 1)[-1]}` (sha256 pinned)"
               if art["fetch"] == "download"
               else f"[`{art['script']}`]({rel(m, start)}/{art['script']})")
        lines.append(f"| {m.display_name} | [{repo.split('github.com/')[-1]}@{commit[:7]}]"
                     f"({repo}/tree/{commit}) | `{art['file']}` | {how} |")
    return "\n".join(lines)


TABLES = {
    (REPO_ROOT / "README.md", "object_detection_table"):
        lambda ms, s: comparison_table(ms, s),
    (REPO_ROOT / "object_detection" / "README.md", "object_detection_table"):
        lambda ms, s: comparison_table(ms, s),
    (REPO_ROOT / "object_detection" / "README.md", "object_detection_provenance"):
        lambda ms, s: provenance_table(ms, s),
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
    models = all_models("object_detection")
    stale = []
    files: dict[Path, str] = {}
    for (path, name), fn in TABLES.items():
        text = files.get(path) or path.read_text(encoding="utf-8")
        files[path] = render(text, name, fn(models, path.parent))
    for path, new in files.items():
        if new != path.read_text(encoding="utf-8"):
            stale.append(path)
            if not args.check:
                path.write_text(new, encoding="utf-8")
    if args.check and stale:
        print("out of date:", *[p.relative_to(REPO_ROOT) for p in stale])
        sys.exit(1)
    print("updated:" if stale else "up to date", *[p.relative_to(REPO_ROOT) for p in stale])


if __name__ == "__main__":
    main()
