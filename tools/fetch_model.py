"""Download or export a model's ONNX artifact into ``<model>/weights/``.

    python tools/fetch_model.py yolox_s
    python tools/fetch_model.py --task object_detection
    python tools/fetch_model.py yolo11n --python .venv-ultralytics/Scripts/python

For ``fetch: download`` the file's SHA-256 is checked against model.yaml.
For ``fetch: export`` the model's ``export.py`` is run. In both cases a
``weights/provenance.json`` records where the file came from, its hash and
the package versions used, so a result can be traced back to its source.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.mlmc.catalog import Model, all_models, get_model  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions(python: str) -> dict:
    code = (
        "import importlib.metadata as m, json\n"
        "out = {}\n"
        "for p in ['torch', 'torchvision', 'transformers', 'ultralytics', 'onnx']:\n"
        "    try: out[p] = m.version(p)\n"
        "    except m.PackageNotFoundError: pass\n"
        "print(json.dumps(out))\n"
    )
    res = subprocess.run([python, "-c", code], capture_output=True, text=True)
    return json.loads(res.stdout) if res.returncode == 0 else {}


def write_hash_into_yaml(model: Model, digest: str):
    path = model.dir / "model.yaml"
    text = path.read_text(encoding="utf-8")
    new = re.sub(r"(?m)^(\s*sha256:)\s*null.*$", rf"\1 {digest}", text, count=1)
    path.write_text(new, encoding="utf-8")


def fetch(model: Model, python: str, force: bool, update_hash: bool):
    art = model.meta["artifacts"]["onnx"]
    out = model.artifact_path("onnx")
    if out.exists() and not force:
        print(f"[{model.name}] already present: {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    prov = {"model": model.name, "file": out.name, "method": art["fetch"],
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "source": model.meta["source"]}

    if art["fetch"] == "download":
        print(f"[{model.name}] downloading {art['url']}")
        urllib.request.urlretrieve(art["url"], out)
        digest = sha256(out)
        expected = art.get("sha256")
        if expected and expected != digest:
            out.unlink()
            raise SystemExit(f"[{model.name}] sha256 mismatch: {digest} != {expected}")
        if not expected:
            if update_hash:
                write_hash_into_yaml(model, digest)
                print(f"[{model.name}] recorded sha256 in model.yaml")
            else:
                print(f"[{model.name}] WARNING: no sha256 in model.yaml "
                      f"(got {digest}); rerun with --update-hash to pin it")
        prov["url"] = art["url"]
    elif art["fetch"] == "export":
        script = model.dir / art["script"]
        print(f"[{model.name}] exporting with {python} {script}")
        subprocess.run([python, str(script)], check=True)
        prov["script"] = str(script.relative_to(model.dir.parents[1])).replace("\\", "/")
        prov["packages"] = package_versions(python)
        prov["python"] = platform.python_version()
    else:
        raise SystemExit(f"unknown fetch method {art['fetch']!r}")

    prov["sha256"] = sha256(out)
    (out.parent / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"[{model.name}] ok -> {out} ({out.stat().st_size / 2**20:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*")
    ap.add_argument("--task")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter used to run export.py scripts")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--update-hash", action="store_true",
                    help="write the downloaded file's sha256 into model.yaml")
    args = ap.parse_args()
    models = [get_model(n) for n in args.models] or all_models(args.task)
    for m in models:
        fetch(m, args.python, args.force, args.update_hash)


if __name__ == "__main__":
    main()
