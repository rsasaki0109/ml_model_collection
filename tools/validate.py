"""Check model.yaml / benchmarks.yaml files for the fields tools rely on.

    python tools/validate.py

Intentionally minimal: it only enforces what is needed to keep license and
measurement information honest. Everything else in model.yaml is free-form
until real models show that a field should be required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml  # noqa: E402

from tools.mlmc import REPO_ROOT, hardware, licenses  # noqa: E402
from tools.mlmc.catalog import TASKS, all_models  # noqa: E402

BENCH_REQUIRED = ("id", "date", "hardware", "runtime", "runtime_version",
                  "precision", "batch_size", "input_shape", "latency_ms",
                  "peak_vram_mb", "vram_method")


def check_model(m) -> list[str]:
    errs = []
    meta = m.meta
    for key in ("name", "task", "source", "license", "artifacts"):
        if key not in meta:
            errs.append(f"missing '{key}'")
    if errs:
        return errs
    if meta["name"] != m.dir.name:
        errs.append(f"name {meta['name']!r} != directory {m.dir.name!r}")
    if meta["task"] not in TASKS or m.dir.parent.name != meta["task"]:
        errs.append(f"task {meta['task']!r} does not match directory")
    if "repository" not in meta["source"]:
        errs.append("source.repository missing")
    for part in ("code", "weights"):
        lic = meta["license"].get(part)
        if not lic:
            errs.append(f"license.{part} missing (use spdx: null, status: unknown)")
            continue
        status = lic.get("status")
        if status not in licenses.STATUSES:
            errs.append(f"license.{part}.status must be one of {licenses.STATUSES}")
        if lic.get("spdx") and licenses.category(lic["spdx"]) == licenses.UNKNOWN:
            errs.append(f"license.{part}.spdx {lic['spdx']!r} not in tools/mlmc/licenses.py")
        if status != "unknown" and not lic.get("spdx"):
            errs.append(f"license.{part}: status {status!r} requires spdx")
        if not lic.get("evidence"):
            errs.append(f"license.{part}.evidence (URL) missing")
    art = meta["artifacts"].get("onnx")
    if art:
        for key in ("file", "fetch", "input_shape"):
            if key not in art:
                errs.append(f"artifacts.onnx.{key} missing")
        if art.get("fetch") == "export" and not (m.dir / art.get("script", "")).is_file():
            errs.append("artifacts.onnx.script does not exist")
        if art.get("fetch") == "download" and not art.get("sha256"):
            errs.append("artifacts.onnx.sha256 missing for a download")
    if not (m.dir / "detector.py").is_file():
        errs.append("detector.py missing")
    for b in m.benchmarks:
        miss = [k for k in BENCH_REQUIRED if k not in b]
        if miss:
            errs.append(f"benchmark {b.get('id')}: missing {miss}")
        elif b["hardware"].get("class") not in hardware.HARDWARE_CLASSES:
            errs.append(f"benchmark {b['id']}: unknown hardware.class")
    return errs


def main():
    bad = 0
    models = all_models()
    for m in models:
        for e in check_model(m):
            print(f"{m.dir.parent.name}/{m.dir.name}: {e}")
            bad += 1
    names = {m.name for m in models}
    for task in TASKS:
        cfg_file = REPO_ROOT / task / "comparison.yaml"
        if cfg_file.exists():
            cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
            for n in cfg.get("models", []):
                if n not in names:
                    print(f"{task}/comparison.yaml: unknown model {n!r}")
                    bad += 1
    print(f"{len(models)} models checked, {bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
