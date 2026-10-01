"""Fast checks for the shared helpers (no model weights, no GPU needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.mlmc import hardware, licenses  # noqa: E402
from tools.mlmc.catalog import RUNNERS, TASKS, all_models  # noqa: E402
from tools.mlmc.depth import DEPTH, DISPARITY, METRIC, DepthMap, colorize  # noqa: E402
from tools.mlmc.detection import (COCO80, COCO91_ID_TO_NAME, Detections,  # noqa: E402
                                  letterbox, nms_per_class)


def test_license_categories():
    assert licenses.category("Apache-2.0") == licenses.PERMISSIVE
    assert licenses.category("AGPL-3.0") == licenses.COPYLEFT
    assert licenses.category("CC-BY-NC-4.0") == licenses.RESTRICTED
    assert licenses.category(None) == licenses.UNKNOWN
    assert licenses.category("Some-Custom-License") == licenses.UNKNOWN
    assert licenses.describe({"spdx": "Apache-2.0", "status": "repository_license"}).endswith("*")


@pytest.mark.parametrize("mb,tier", [(0, "Tiny"), (2048, "Tiny"), (2049, "Light"),
                                     (8192, "Consumer"), (30000, "Huge"), (None, None)])
def test_vram_tier(mb, tier):
    assert hardware.vram_tier(mb) == tier


def test_coco_maps():
    assert len(COCO80) == 80 and len(set(COCO80)) == 80
    assert COCO91_ID_TO_NAME[1] == "person" and COCO91_ID_TO_NAME[90] == "toothbrush"
    assert len(COCO91_ID_TO_NAME) == 80


def test_letterbox_center_and_corner():
    img = np.zeros((100, 200, 3), np.uint8)
    out, r, (px, py) = letterbox(img, 64, center=True)
    assert out.shape == (64, 64, 3) and r == pytest.approx(0.32) and py > 0 and px == 0
    _, _, pad = letterbox(img, 64, center=False)
    assert pad == (0, 0)


def test_nms_is_class_aware():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    classes = np.array([0, 0, 1])
    keep = set(nms_per_class(boxes, scores, classes, 0.5).tolist())
    assert keep == {0, 2}  # same-class duplicate removed, other class kept


def test_detections_json_roundtrip():
    d = Detections(np.array([[1, 2, 3, 4]], np.float32), np.array([0.5], np.float32), ["car"])
    assert d.to_json() == [{"label": "car", "score": 0.5, "box": [1.0, 2.0, 3.0, 4.0]}]


@pytest.mark.parametrize("kind", [DISPARITY, DEPTH, METRIC])
def test_colorize_near_is_bright(kind):
    near_top = np.linspace(1, 10, 50, dtype=np.float32)[:, None].repeat(20, 1)
    values = 1 / near_top if kind == DISPARITY else near_top  # top row is nearest
    img = colorize(DepthMap(values, kind))
    assert img.shape == (50, 20, 3)
    assert img[0].mean() > img[-1].mean()


def test_colorize_handles_inf():
    v = np.full((10, 10), 5.0, np.float32)
    v[0] = np.inf  # e.g. sky in metric depth
    assert np.isfinite(colorize(DepthMap(v, METRIC)).astype(float)).all()


def test_catalog_consistency():
    models = all_models()
    assert models, "no models found"
    for m in models:
        assert m.task in TASKS
        assert (m.dir / RUNNERS[m.task][0]).is_file()


def test_segmentation_render():
    from tools.mlmc.segmentation import InstanceMasks, SemanticMap, render
    frame = np.zeros((20, 30, 3), np.uint8)
    masks = np.zeros((1, 20, 30), bool)
    masks[0, 5:10, 5:10] = True
    inst = InstanceMasks(np.array([[5, 5, 10, 10]], np.float32), np.array([0.9], np.float32),
                         ["person"], masks)
    out = render(frame, inst)
    assert out.shape == frame.shape and out[7, 7].any() and not out[15, 25].any()
    sem = SemanticMap(np.ones((20, 30), np.int64), ["bg", "road"])
    assert render(frame, sem).any()


def test_pose_render():
    from tools.mlmc.pose import COCO17, SKELETON, Poses, render
    assert len(COCO17) == 17 and all(0 <= a < 17 and 0 <= b < 17 for a, b in SKELETON)
    kps = np.tile(np.linspace(5, 25, 17)[:, None], (1, 2))[None].astype(np.float32)
    p = Poses(kps, np.ones((1, 17), np.float32), np.array([[0, 0, 30, 30]], np.float32),
              np.array([0.9], np.float32))
    out = render(np.zeros((30, 30, 3), np.uint8), p)
    assert out.any()
