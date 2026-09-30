"""Helpers used by ``<task>/<model>/export.py`` scripts.

Only standard library + torch at import time, so export scripts can run in a
separate virtualenv that has the upstream project's dependencies.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clone_at(url: str, commit: str, dest: Path) -> Path:
    """Shallow-fetch ``url`` at exactly ``commit`` into ``dest``."""
    dest = Path(dest)
    if (dest / ".git").exists():
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        if head == commit:
            return dest
    dest.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(["git", "-C", str(dest), *a], check=True)  # noqa: E731
    run("init", "-q")
    subprocess.run(["git", "-C", str(dest), "remote", "remove", "origin"],
                   capture_output=True)
    run("remote", "add", "origin", url)
    run("fetch", "-q", "--depth", "1", "origin", commit)
    run("checkout", "-q", "FETCH_HEAD")
    return dest


def download(url: str, dest: Path, expected_sha256: str | None) -> Path:
    """Download ``url`` (or ``gdrive:<file id>``) and verify its SHA-256."""
    dest = Path(dest)
    if not (dest.exists() and expected_sha256 and sha256(dest) == expected_sha256):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.startswith("gdrive:"):
            try:
                import gdown
            except ImportError:
                raise SystemExit("Google Drive download needs `pip install gdown`")
            gdown.download(id=url.split(":", 1)[1], output=str(dest), quiet=True)
        else:
            urllib.request.urlretrieve(url, dest)
    digest = sha256(dest)
    if expected_sha256 and digest != expected_sha256:
        raise SystemExit(f"sha256 mismatch for {dest.name}: {digest} != {expected_sha256}")
    return dest


def export_dfine_family(repo_dir: Path, config: str, checkpoint: Path, out: Path,
                        size: int = 640, opset: int = 17):
    """Export a D-FINE-style model (D-FINE / DEIM / RT-DETRv4 code bases).

    Mirrors upstream ``tools/deployment/export_onnx.py`` (deploy-mode model +
    post-processor) but with a fixed batch of 1 and no onnxsim dependency.
    Graph: images [1,3,S,S] RGB in [0,1], orig_target_sizes [1,2] (w, h)
    -> labels [1,300] (0..79), boxes [1,300,4] xyxy in original pixels,
    scores [1,300].
    """
    import torch
    import torch.nn as nn

    sys.path.insert(0, str(repo_dir))
    from engine.core import YAMLConfig  # upstream module

    cfg = YAMLConfig(str(repo_dir / config), resume=str(checkpoint))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state["ema"]["module"] if "ema" in state else state["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    model = Model().eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model, (torch.rand(1, 3, size, size), torch.tensor([[size, size]])),
            str(out), input_names=["images", "orig_target_sizes"],
            output_names=["labels", "boxes", "scores"],
            opset_version=opset, dynamo=False, do_constant_folding=True)
    print(f"wrote {out}")
