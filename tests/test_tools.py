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


def test_flow_render_and_flo(tmp_path):
    from tools.mlmc.flow import Flow, read_flo, render
    uv = np.zeros((8, 10, 2), np.float32)
    uv[..., 0] = 30  # rightwards, beyond saturation
    out = render(np.full((8, 10, 3), 128, np.uint8), Flow(uv))
    assert out.shape == (8, 10, 3) and out[..., 2].mean() > out[..., 0].mean()  # red-ish
    f = tmp_path / "a.flo"
    with open(f, "wb") as fh:
        np.array([202021.25], np.float32).tofile(fh)
        np.array([10, 8], np.int32).tofile(fh)
        uv.tofile(fh)
    assert np.array_equal(read_flo(f), uv)


def test_sr_psnr_and_render():
    from tools.mlmc.sr import SRImage, psnr_y, render
    a = np.full((16, 16, 3), 100, np.uint8)
    b = a.copy()
    b[8:, :] = 110
    assert psnr_y(a, a, crop=4) == float("inf")
    assert 20 < psnr_y(a, b, crop=4) < 40
    out = render(np.zeros((40, 80, 3), np.uint8), SRImage(np.zeros((40, 80, 3), np.uint8), np.zeros((10, 20, 3), np.uint8)))
    assert out.shape == (40, 80, 3)


def test_face_render_and_matte():
    from tools.mlmc.face import Faces, render
    from tools.mlmc.matting import AlphaMatte, render as render_matte
    f = Faces(np.array([[10, 10, 30, 34]], np.float32), np.array([0.9], np.float32),
              np.full((1, 5, 2), 20, np.float32))
    out = render(np.zeros((120, 160, 3), np.uint8), f)
    assert out.shape == (120, 160, 3) and out.any()
    frame = np.full((4, 4, 3), 200, np.uint8)
    a = np.zeros((4, 4), np.float32)
    a[:2] = 1
    comp = render_matte(frame, AlphaMatte(a))
    assert (comp[:2] == 200).all() and (comp[2:] != 200).any()


def test_matching_dlt_and_auc():
    from tools.mlmc.matching import corner_error, error_auc, homography_dlt
    H = np.array([[1.1, 0.05, 10], [-0.02, 0.95, 5], [1e-4, 2e-4, 1]])
    p0 = np.random.default_rng(0).uniform(0, 400, (50, 2))
    q = np.c_[p0, np.ones(50)] @ H.T
    p1 = q[:, :2] / q[:, 2:]
    assert corner_error(homography_dlt(p0, p1), H, 400, 300) < 1e-6
    assert error_auc([0.0, 0.0], [1])[0] > 0.99 and error_auc([10.0, 10.0], [1, 3, 5]) == [0.0, 0.0, 0.0]
