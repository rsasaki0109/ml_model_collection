"""Helpers for super-resolution exports (``super_resolution/<model>/export.py``).

BasicSR-style architecture files import ``basicsr.utils.registry`` and
``basicsr.archs.arch_util``; importing the real ``basicsr`` package pulls in
data pipelines and compiled ops. Instead, the architecture file is loaded on
its own with a stub registry, and the few helpers it needs are taken
verbatim from upstream ``arch_util.py`` (extracted with ``ast``).
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm


class _Registry:
    def register(self, *a, **kw):
        return lambda obj: obj


def _module(name: str, **attrs) -> types.ModuleType:
    mod = sys.modules.get(name) or types.ModuleType(name)
    mod.__path__ = getattr(mod, "__path__", [])
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def load_arch(arch_file: Path, arch_util: Path | None = None,
              helpers=("default_init_weights", "make_layer", "pixel_unshuffle")):
    """Import a BasicSR-style ``*_arch.py`` without the basicsr package."""
    _module("basicsr")
    _module("basicsr.utils")
    _module("basicsr.utils.registry", ARCH_REGISTRY=_Registry())
    _module("basicsr.archs")
    if arch_util is not None:
        tree = ast.parse(Path(arch_util).read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in helpers]
        ns = {"torch": torch, "nn": nn, "init": init, "_BatchNorm": _BatchNorm, "math": math}
        exec(compile(tree, str(arch_util), "exec"), ns)
        _module("basicsr.archs.arch_util", **{k: ns[k] for k in helpers if k in ns})
    name = f"basicsr.archs.{Path(arch_file).stem}"
    spec = importlib.util.spec_from_file_location(name, arch_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export(model: torch.nn.Module, out: Path, size=(64, 64), opset: int = 17):
    """Dynamic-shape x4 graph: lr [1,3,h,w] RGB [0,1] -> sr [1,3,4h,4w]."""
    from tools.mlmc.export import patch_for_ort
    x = torch.rand(1, 3, *size)
    with torch.no_grad():
        torch.onnx.export(model.eval(), x, str(out), input_names=["lr"], output_names=["sr"],
                          dynamic_axes={"lr": {2: "h", 3: "w"}, "sr": {2: "H", 3: "W"}},
                          opset_version=opset, dynamo=False)
    changes = patch_for_ort(out)
    print(f"wrote {out}" + (f" (patched for ONNX Runtime: {changes})" if changes else ""))
