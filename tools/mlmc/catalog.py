"""Discover models in the collection.

A model is any directory ``<task>/<name>/`` that contains ``model.yaml``.
Benchmark records live next to it in ``benchmarks.yaml`` (written by
``tools/benchmark.py``), so curated metadata and measured data stay separate.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import REPO_ROOT

TASKS = ("object_detection", "segmentation", "depth_estimation",
         "pose_estimation", "optical_flow")


@dataclass
class Model:
    dir: Path
    meta: dict
    benchmarks: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.meta["name"]

    @property
    def task(self) -> str:
        return self.meta["task"]

    @property
    def display_name(self) -> str:
        return self.meta.get("display_name", self.name)

    @property
    def weights_dir(self) -> Path:
        return self.dir / "weights"

    def artifact_path(self, fmt: str = "onnx") -> Path:
        return self.weights_dir / self.meta["artifacts"][fmt]["file"]

    def load_detector(self, **kwargs):
        """Import ``detector.py`` from the model dir and build its Detector."""
        path = self.dir / "detector.py"
        spec = importlib.util.spec_from_file_location(
            f"mlmc_models.{self.task}.{self.name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Detector(self, **kwargs)


def load_model(path: Path) -> Model:
    path = Path(path)
    if path.name == "model.yaml":
        path = path.parent
    with open(path / "model.yaml", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    bench = []
    bench_file = path / "benchmarks.yaml"
    if bench_file.exists():
        with open(bench_file, encoding="utf-8") as f:
            bench = yaml.safe_load(f) or []
    return Model(dir=path, meta=meta, benchmarks=bench)


def all_models(task: str | None = None) -> list[Model]:
    tasks = [task] if task else TASKS
    models = []
    for t in tasks:
        for meta in sorted((REPO_ROOT / t).glob("*/model.yaml")):
            models.append(load_model(meta))
    return models


def get_model(name: str) -> Model:
    for m in all_models():
        if m.name == name:
            return m
    raise SystemExit(f"unknown model: {name!r} "
                     f"(available: {', '.join(m.name for m in all_models())})")
