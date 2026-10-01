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
         "pose_estimation", "optical_flow", "super_resolution")

# task -> (runner module in the model directory, class name)
RUNNERS = {
    "object_detection": ("detector.py", "Detector"),
    "depth_estimation": ("estimator.py", "Estimator"),
    "segmentation": ("segmenter.py", "Segmenter"),
    "pose_estimation": ("pose.py", "PoseEstimator"),
    "optical_flow": ("flow.py", "FlowEstimator"),
    "super_resolution": ("upscaler.py", "Upscaler"),
}


@dataclass
class Model:
    dir: Path
    meta: dict
    benchmarks: list = field(default_factory=list)
    accuracy: list = field(default_factory=list)

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

    def load_runner(self, **kwargs):
        """Build the model's runner (``Detector`` / ``Estimator``).

        Every runner exposes ``preprocess(frame) -> (x, meta)``,
        ``infer(x)``, ``postprocess(out, meta)`` and ``__call__(frame)``, so
        benchmarking and evaluation tools are task-agnostic.
        """
        filename, cls = RUNNERS[self.task]
        path = self.dir / filename
        spec = importlib.util.spec_from_file_location(
            f"mlmc_models.{self.task}.{self.name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, cls)(self, **kwargs)

    load_detector = load_runner


def load_model(path: Path) -> Model:
    path = Path(path)
    if path.name == "model.yaml":
        path = path.parent
    with open(path / "model.yaml", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    def generated(name):
        f = path / name
        return (yaml.safe_load(f.read_text(encoding="utf-8")) or []) if f.exists() else []
    return Model(dir=path, meta=meta, benchmarks=generated("benchmarks.yaml"),
                 accuracy=generated("accuracy.yaml"))


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
