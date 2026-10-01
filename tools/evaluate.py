"""Measure accuracy of a model's *exported artifact* on a standard dataset.

    python tools/evaluate.py --model yolox_s --coco-root /data/coco
    python tools/evaluate.py --task object_detection --coco-root /data/coco
    python tools/evaluate.py --model rfdetr_seg_n --coco-root /data/coco
    python tools/evaluate.py --model mask2former_swin_t_ade --ade-root /data/ADEChallengeData2016

Always uses exactly the ONNX file and runner (``detector.py`` /
``segmenter.py``) of the collection — so the number checks the export and
our pre/post-processing, not the upstream PyTorch model.

* Object detection: COCO val2017 box AP (pycocotools).
* Instance / promptable segmentation: COCO val2017 mask AP (segm) plus box AP.
  Promptable models use their default prompt detector's boxes.
* Pose estimation: COCO val2017 keypoint AP (OKS, pycocotools, needs
  ``annotations/person_keypoints_val2017.json``). Top-down models use their
  default person detector, so the number covers the whole pipeline.
* Optical flow: MPI-Sintel training split EPE (clean + final),
  ``--sintel-root`` = directory with ``training/{clean,final,flow}``
  (https://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip and
  MPI-Sintel-training_extras.zip).
* Super-resolution: x4 PSNR-Y on Set5 / Set14 / Urban100, ``--sr-root`` =
  directory with ``<Set>_HR/`` and ``<Set>_LR_x4/`` (Hugging Face
  eugenesiow/Set5, eugenesiow/Set14, eugenesiow/Urban100 ``data/*.tar.gz``).
* Background removal: DIS5K validation split (DIS-VD) S-measure, weighted F,
  max F, MAE (PySODMetrics), ``--dis-root`` = the extracted ``DIS5K``
  directory (Google Drive id 1O1eIuXX1hlGsV7qx4eSkjH231q7G1by1, from the DIS
  README; non-commercial terms, do not redistribute).
* Face detection: WIDER FACE val AP easy / medium / hard (NumPy port of the
  official ``evaluation.m``), ``--wider-root`` = directory with
  ``WIDER_val/images/`` (Hugging Face CUHK-CSE/wider_face ``data/WIDER_val.zip``)
  and ``eval_tools/ground_truth/`` (official ``eval_tools.zip``).
* Semantic segmentation: ADE20K val mIoU (150 classes, label 0 ignored),
  ``--ade-root`` = the extracted ADEChallengeData2016 directory
  (http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).

``--coco-root`` must contain ``val2017/`` and
``annotations/instances_val2017.json`` (download:
http://images.cocodataset.org/zips/val2017.zip and
http://images.cocodataset.org/annotations/annotations_trainval2017.zip).

Evaluation settings (recorded in the result): score threshold 0.001, at most
100 detections per image (COCO maxDets), NMS IoU 0.65 for models that run NMS
in ``detector.py``; each model's own input size and resize policy.

Results go to ``<task>/<model>/accuracy.yaml`` (generated, one record per
dataset/runtime/precision).
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark import collection_commit  # noqa: E402
from tools.mlmc.catalog import Model, all_models, get_model  # noqa: E402
from tools.mlmc.detection import COCO80, COCO91_IDS, PROVIDERS, provider_precision  # noqa: E402

NAME_TO_CAT = dict(zip(COCO80, COCO91_IDS))
SCORE_THR = 0.001
MAX_DETS = 100
NMS_IOU = 0.65


def eval_coco_detection(model: Model, coco_root: Path, provider: str,
                        limit: int | None) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(coco_root / "annotations" / "instances_val2017.json"))
    img_ids = sorted(gt.getImgIds())[:limit] if limit else sorted(gt.getImgIds())

    kwargs = {"provider": provider}
    params = inspect.signature(_runner_cls(model)).parameters
    if "score_thr" in params:
        kwargs["score_thr"] = SCORE_THR
    if "iou_thr" in params:
        kwargs["iou_thr"] = NMS_IOU
    det = model.load_runner(**kwargs)
    from pycocotools import mask as mask_utils

    from tools.mlmc.segmentation import InstanceMasks

    results, t0, has_masks = [], time.perf_counter(), False
    for k, img_id in enumerate(img_ids):
        info = gt.loadImgs(img_id)[0]
        img = cv2.imread(str(coco_root / "val2017" / info["file_name"]))
        d = det(img)
        masks = d.masks if isinstance(d, InstanceMasks) else None
        has_masks = has_masks or masks is not None
        order = np.argsort(-d.scores)[:MAX_DETS]
        for i in order:
            x1, y1, x2, y2 = (float(v) for v in d.boxes[i])
            r = {"image_id": img_id, "category_id": NAME_TO_CAT[d.labels[i]],
                 "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                 "score": round(float(d.scores[i]), 5)}
            if masks is not None:
                rle = mask_utils.encode(np.asfortranarray(masks[i].astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                r["segmentation"] = rle
            results.append(r)
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{len(img_ids)} images", flush=True)
    elapsed = time.perf_counter() - t0

    if not results:
        raise SystemExit("no detections")
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
    dt_ = gt.loadRes(results)
    stats = {}
    for iou_type in (["segm", "bbox"] if has_masks else ["bbox"]):
        ev = COCOeval(gt, dt_, iou_type)
        ev.params.imgIds = img_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
        stats[iou_type] = {n: round(float(v) * 100, 1) for n, v in zip(names, ev.stats[:6])}
    out = {
        "dataset": "COCO val2017" + (f" (first {limit} images)" if limit else ""),
        "images": len(img_ids),
        "metrics": stats["segm"] if has_masks else stats["bbox"],
    }
    if has_masks:
        out["box_metrics"] = stats["bbox"]
    out.update({
        "settings": {"score_threshold": SCORE_THR, "max_dets": MAX_DETS,
                     "nms_iou": NMS_IOU if "iou_thr" in params else None,
                     "evaluator": "pycocotools COCOeval (" + ("segm" if has_masks else "bbox") + ")"},
        "wall_time_s": round(elapsed, 1),
    })
    if model.meta.get("prompt_model"):
        out["settings"]["prompts"] = f"boxes from {model.meta['prompt_model']}"
    return out


def eval_coco_keypoints(model: Model, coco_root: Path, provider: str,
                        limit: int | None) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(coco_root / "annotations" / "person_keypoints_val2017.json"))
    img_ids = sorted(gt.getImgIds())[:limit] if limit else sorted(gt.getImgIds())
    params = inspect.signature(_runner_cls(model)).parameters
    kwargs = {"provider": provider}
    if "score_thr" in params:
        kwargs["score_thr"] = SCORE_THR
    est = model.load_runner(**kwargs)
    results, t0 = [], time.perf_counter()
    for k, img_id in enumerate(img_ids):
        info = gt.loadImgs(img_id)[0]
        poses = est(cv2.imread(str(coco_root / "val2017" / info["file_name"])))
        order = np.argsort(-poses.box_scores)[:20]   # COCO keypoint maxDets = 20
        for i in order:
            kps = np.concatenate([poses.keypoints[i], poses.scores[i][:, None]], 1)
            # instance score as in mmpose: box score x mean confident keypoint score
            conf = poses.scores[i][poses.scores[i] > 0.2]
            score = float(poses.box_scores[i]) * (float(conf.mean()) if len(conf) else 0.0)
            results.append({"image_id": img_id, "category_id": 1,
                            "keypoints": [round(float(v), 2) for v in kps.reshape(-1)],
                            "score": round(score, 5)})
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{len(img_ids)} images", flush=True)
    elapsed = time.perf_counter() - t0
    if not results:
        raise SystemExit("no poses")
    ev = COCOeval(gt, gt.loadRes(results), "keypoints")
    ev.params.imgIds = img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    names = ["AP", "AP50", "AP75", "AP_medium", "AP_large"]
    out = {
        "dataset": "COCO val2017 keypoints" + (f" (first {limit} images)" if limit else ""),
        "images": len(img_ids),
        "metrics": {n: round(float(v) * 100, 1) for n, v in zip(names, ev.stats[:5])},
        "settings": {"score_threshold": SCORE_THR, "max_dets": 20,
                     "instance_score": "box score x mean keypoint score (> 0.2)",
                     "evaluator": "pycocotools COCOeval (keypoints, OKS)"},
        "wall_time_s": round(elapsed, 1),
    }
    if model.meta.get("person_detector"):
        out["settings"]["person_boxes"] = f"from {model.meta['person_detector']}"
    return out


def eval_ade20k(model: Model, ade_root: Path, provider: str, limit: int | None) -> dict:
    """ADE20K val mIoU: 150 classes, label 0 (other / unlabelled) ignored."""
    seg = model.load_runner(provider=provider)
    imgs = sorted((ade_root / "images" / "validation").glob("*.jpg"))[:limit] if limit else \
        sorted((ade_root / "images" / "validation").glob("*.jpg"))
    n = 150
    conf = np.zeros((n, n), np.int64)
    t0 = time.perf_counter()
    for k, p in enumerate(imgs):
        gt = cv2.imread(str(ade_root / "annotations" / "validation" / (p.stem + ".png")),
                        cv2.IMREAD_GRAYSCALE).astype(np.int64)
        pred = seg(cv2.imread(str(p))).classes.astype(np.int64) + 1   # 0..149 -> 1..150
        valid = gt > 0
        conf += np.bincount((gt[valid] - 1) * n + (pred[valid] - 1),
                            minlength=n * n).reshape(n, n)
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(imgs)} images", flush=True)
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    iou = inter[union > 0] / union[union > 0]
    return {
        "dataset": "ADE20K val" + (f" (first {limit} images)" if limit else ""),
        "images": len(imgs),
        "metrics": {"mIoU": round(float(iou.mean()) * 100, 1),
                    "aAcc": round(float(inter.sum() / conf.sum()) * 100, 1)},
        "settings": {"classes": n, "ignore_label": 0,
                     "evaluator": "confusion matrix over all pixels (mmseg-style mIoU)"},
        "wall_time_s": round(time.perf_counter() - t0, 1),
    }


def eval_sintel(model: Model, root: Path, provider: str, limit: int | None) -> dict:
    """MPI-Sintel training split, clean and final passes, end-point error.

    Frames of each scene are fed in order to the (stateful) runner, so every
    flow is frame t -> t+1 against ``flow/<scene>/frame_<t>.flo``. EPE is the
    mean over all pixels of all pairs (RAFT convention; Sintel frames all
    have the same size, so it equals the per-image average).
    """
    from tools.mlmc.flow import read_flo

    t0 = time.perf_counter()
    metrics, pairs = {}, 0
    for pas in ("clean", "final"):
        runner = model.load_runner(provider=provider)
        epe_sum, n_px, outl = 0.0, 0, {1: 0, 3: 0, 5: 0}
        pairs = 0
        for scene in sorted((root / "training" / pas).iterdir()):
            runner.prev = None
            frames = sorted(scene.glob("frame_*.png"))
            for k, f in enumerate(frames):
                flow = runner(cv2.imread(str(f)))
                if k == 0:
                    continue
                gt = read_flo(root / "training" / "flow" / scene.name / f"{frames[k - 1].stem}.flo")
                epe = np.linalg.norm(flow.uv - gt, axis=-1)
                epe_sum += float(epe.sum())
                n_px += epe.size
                for t in outl:
                    outl[t] += int((epe > t).sum())
                pairs += 1
                if limit and pairs >= limit:
                    break
            if limit and pairs >= limit:
                break
        metrics[f"{pas}_EPE"] = round(epe_sum / n_px, 3)
        for t in outl:
            metrics[f"{pas}_{t}px_pct"] = round(100 * outl[t] / n_px, 2)
    shape = model.meta["artifacts"]["onnx"]["input_shape"]
    return {
        "dataset": "MPI-Sintel train" + (f" (first {limit} pairs)" if limit else ""),
        "images": pairs,
        "metrics": metrics,
        "settings": {"input": f"frames 1024x436 resized to the graph's {shape[-1]}x{shape[-2]}, "
                              "flow resized back and rescaled",
                     "evaluator": "end-point error over all pixels (RAFT convention); "
                                  "Npx = % of pixels with EPE > N"},
        "wall_time_s": round(time.perf_counter() - t0, 1),
    }


SR_SETS = ("Set5", "Set14", "Urban100")


def eval_sr(model: Model, root: Path, provider: str, limit: int | None) -> dict:
    """x4 PSNR on the Y channel (BasicSR convention: BT.601 Y, 4 px border
    cropped, HR cropped to the SR size). ``root`` holds ``<Set>_HR/`` and
    ``<Set>_LR_x4/`` (Hugging Face eugenesiow/<Set> archives); the LR images
    are used as provided, each runs through the graph at its own size."""
    from tools.mlmc.sr import psnr_y

    t0 = time.perf_counter()
    runner = model.load_runner(provider=provider)
    metrics, n_img = {}, 0
    for name in SR_SETS:
        hr_files = sorted((root / f"{name}_HR").glob("*.png"))[:limit]
        if not hr_files:
            continue
        vals = []
        for f in hr_files:
            hr = cv2.imread(str(f))
            sr = runner.upscale(cv2.imread(str(root / f"{name}_LR_x4" / f.name)))
            h, w = min(hr.shape[0], sr.shape[0]), min(hr.shape[1], sr.shape[1])
            vals.append(psnr_y(sr[:h, :w], hr[:h, :w], crop=4))
        metrics[f"{name}_PSNR_Y"] = round(float(np.mean(vals)), 2)
        n_img += len(vals)
    if not metrics:
        raise SystemExit(f"no <Set>_HR/ directories under {root}")
    return {
        "dataset": "SR benchmarks x4 (" + ", ".join(k.split("_")[0] for k in metrics) + ")"
                   + (f" (first {limit} images each)" if limit else ""),
        "images": n_img,
        "metrics": metrics,
        "settings": {"lr_source": "Hugging Face eugenesiow/<Set> LR_x4 archives, used as provided",
                     "evaluator": "PSNR on BT.601 Y, 4 px border cropped, mean over images (BasicSR)"},
        "wall_time_s": round(time.perf_counter() - t0, 1),
    }


def eval_dis(model: Model, root: Path, provider: str, limit: int | None) -> dict:
    """DIS5K validation split (DIS-VD): S-measure, weighted F, max F, MAE with
    PySODMetrics (the toolkit DIS / BiRefNet results are computed with).
    ``root`` = the extracted ``DIS5K`` directory (contains ``DIS-VD/im``, ``DIS-VD/gt``)."""
    from py_sod_metrics import MAE, Fmeasure, Smeasure, WeightedFmeasure

    t0 = time.perf_counter()
    runner = model.load_runner(provider=provider)
    sm, wfm, fm, mae = Smeasure(), WeightedFmeasure(), Fmeasure(), MAE()
    gts = sorted((root / "DIS-VD" / "gt").glob("*.png"))[:limit]
    for g in gts:
        if hasattr(runner, "reset"):
            runner.reset()  # recurrent video models: every image is a new clip
        img = cv2.imread(str(root / "DIS-VD" / "im" / f"{g.stem}.jpg"))
        gt = cv2.imread(str(g), cv2.IMREAD_GRAYSCALE)
        alpha = runner(img).alpha
        if alpha.shape != gt.shape:
            alpha = cv2.resize(alpha, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        pred = (alpha * 255).round().astype(np.uint8)
        for m in (sm, wfm, fm, mae):
            m.step(pred=pred, gt=gt)
    return {
        "dataset": "DIS5K DIS-VD" + (f" (first {limit} images)" if limit else ""),
        "images": len(gts),
        "metrics": {"S_measure": round(float(sm.get_results()["sm"]), 3),
                    "weighted_F": round(float(wfm.get_results()["wfm"]), 3),
                    "max_F": round(float(fm.get_results()["fm"]["curve"].max()), 3),
                    "MAE": round(float(mae.get_results()["mae"]), 4)},
        "settings": {"evaluator": "PySODMetrics (Smeasure, WeightedFmeasure, Fmeasure curve max, MAE); "
                                  "alpha at the image resolution, 8-bit"},
        "wall_time_s": round(time.perf_counter() - t0, 1),
    }


def _wider_overlaps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of x1 y1 x2 y2 boxes with the +1 pixel convention of the official tool."""
    area_a = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    area_b = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + 1
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + 1
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    return np.where((iw > 0) & (ih > 0), inter / (area_a[:, None] + area_b[None, :] - inter), 0)


def eval_widerface(model: Model, root: Path, provider: str, limit: int | None) -> dict:
    """WIDER FACE val AP (easy / medium / hard), a NumPy port of the official
    ``evaluation.m`` (as in the widely used WiderFace-Evaluation Python port):
    scores min-max normalised over the whole set, greedy matching at IoU 0.5
    in descending score order, faces outside the setting's ``gt_list`` are
    ignored, 1000 thresholds, VOC all-point AP.

    ``root`` contains ``WIDER_val/images/<event>/*.jpg`` and the official
    ``eval_tools/ground_truth/wider_{face,easy,medium,hard}_val.mat``.
    """
    from scipy.io import loadmat

    t0 = time.perf_counter()
    gt_dir = root / "eval_tools" / "ground_truth"
    face = loadmat(str(gt_dir / "wider_face_val.mat"))
    events = [str(e[0][0]) for e in face["event_list"]]
    eval_kw = dict(model.meta.get("eval_settings", {}))
    runner = model.load_runner(provider=provider, score_thr=eval_kw.pop("score_thr", 0.02), **eval_kw)

    preds, n_img = {}, 0
    for ei, ev in enumerate(events):
        for fi, f in enumerate(face["file_list"][ei][0]):
            if limit and n_img >= limit:
                break
            name = str(f[0][0])
            res = runner(cv2.imread(str(root / "WIDER_val" / "images" / ev / f"{name}.jpg")))
            order = np.argsort(-res.scores)
            preds[(ei, fi)] = np.concatenate([res.boxes[order], res.scores[order, None]], 1)
            n_img += 1
    allscores = np.concatenate([p[:, 4] for p in preds.values() if len(p)]) if preds else np.zeros(1)
    lo, hi = float(allscores.min()), float(allscores.max())
    for p in preds.values():
        p[:, 4] = (p[:, 4] - lo) / max(hi - lo, 1e-12)

    thresh_num, metrics = 1000, {}
    for setting in ("easy", "medium", "hard"):
        sub = loadmat(str(gt_dir / f"wider_{setting}_val.mat"))["gt_list"]
        pr_curve, count_face = np.zeros((thresh_num, 2)), 0
        for (ei, fi), pred in preds.items():
            gt = face["face_bbx_list"][ei][0][fi][0].astype(np.float64)
            keep = sub[ei][0][fi][0].reshape(-1)
            count_face += len(keep)
            if len(gt) == 0 or len(pred) == 0:
                continue
            ignore = np.zeros(len(gt))
            if len(keep):
                ignore[keep - 1] = 1
            gtb = gt.copy()
            gtb[:, 2:] += gtb[:, :2]  # x y w h -> x1 y1 x2 y2
            ov = _wider_overlaps(pred[:, :4], gtb)
            recall = np.zeros(len(gt))
            proposal = np.ones(len(pred))
            pred_recall = np.zeros(len(pred))
            for h in range(len(pred)):
                j = int(ov[h].argmax())
                if ov[h, j] >= 0.5:
                    if ignore[j] == 0:
                        recall[j], proposal[h] = -1, -1
                    elif recall[j] == 0:
                        recall[j] = 1
                pred_recall[h] = (recall == 1).sum()
            for t in range(thresh_num):
                idx = np.flatnonzero(pred[:, 4] >= 1 - (t + 1) / thresh_num)
                if len(idx):
                    r = idx[-1]
                    pr_curve[t, 0] += (proposal[:r + 1] == 1).sum()
                    pr_curve[t, 1] += pred_recall[r]
        prec = np.divide(pr_curve[:, 1], pr_curve[:, 0], out=np.zeros(thresh_num), where=pr_curve[:, 0] > 0)
        rec = pr_curve[:, 1] / max(count_face, 1)
        mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.concatenate([[0.0], prec, [0.0]])
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        i = np.flatnonzero(mrec[1:] != mrec[:-1])
        metrics[f"AP_{setting}"] = round(float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])) * 100, 2)
    return {
        "dataset": "WIDER FACE val" + (f" (first {limit} images)" if limit else ""),
        "images": n_img,
        "metrics": metrics,
        "settings": {"score_threshold": runner.score_thr, "nms_iou": runner.nms_iou,
                     "input": model.meta["artifacts"]["onnx"]["input_format"],
                     "evaluator": "NumPy port of the official evaluation.m (IoU 0.5, 1000 thresholds, VOC AP)"},
        "wall_time_s": round(time.perf_counter() - t0, 1),
    }


def _runner_cls(model: Model):
    import importlib.util
    from tools.mlmc.catalog import RUNNERS
    filename, cls = RUNNERS[model.task]
    spec = importlib.util.spec_from_file_location(f"_probe_{model.name}", model.dir / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls)


def ort_version() -> str:
    import importlib.metadata as md
    for name in ("onnxruntime-gpu", "onnxruntime"):
        try:
            return f"{name} {md.version(name)}"
        except md.PackageNotFoundError:
            pass
    return "unknown"


def save(model: Model, rec: dict):
    path = model.dir / "accuracy.yaml"
    records = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else []
    records = [r for r in (records or []) if r.get("id") != rec["id"]] + [rec]
    records.sort(key=lambda r: r["id"])
    path.write_text("# GENERATED by tools/evaluate.py — do not edit by hand.\n"
                    + yaml.safe_dump(records, sort_keys=False, allow_unicode=True),
                    encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--task", default="object_detection")
    ap.add_argument("--coco-root", type=Path)
    ap.add_argument("--ade-root", type=Path)
    ap.add_argument("--sintel-root", type=Path, help="extracted MPI-Sintel (contains training/)")
    ap.add_argument("--sr-root", type=Path, help="directory with <Set>_HR/ and <Set>_LR_x4/")
    ap.add_argument("--dis-root", type=Path, help="extracted DIS5K (contains DIS-VD/)")
    ap.add_argument("--wider-root", type=Path,
                    help="directory with WIDER_val/images/ and eval_tools/ground_truth/")
    ap.add_argument("--provider", default="cuda", choices=list(PROVIDERS))
    ap.add_argument("--limit", type=int, help="evaluate only the first N images (smoke test)")
    args = ap.parse_args()

    if not args.model:
        for m in all_models(args.task):
            if m.artifact_path("onnx").exists():
                print(f"== {m.name}", flush=True)
                roots = (["--coco-root", str(args.coco_root)] if args.coco_root else []) + \
                    (["--ade-root", str(args.ade_root)] if args.ade_root else []) + \
                    (["--sintel-root", str(args.sintel_root)] if args.sintel_root else []) + \
                    (["--sr-root", str(args.sr_root)] if args.sr_root else []) + \
                    (["--dis-root", str(args.dis_root)] if args.dis_root else []) + \
                    (["--wider-root", str(args.wider_root)] if args.wider_root else [])
                subprocess.run([sys.executable, __file__, "--model", m.name, *roots,
                                "--provider", args.provider]
                               + (["--limit", str(args.limit)] if args.limit else []))
        return

    model = get_model(args.model)
    if model.task == "segmentation" and model.meta.get("kind") == "semantic":
        if not args.ade_root:
            raise SystemExit("--ade-root is required for semantic segmentation")
        res = eval_ade20k(model, args.ade_root, args.provider, args.limit)
        ds_id = "ade20k-val"
    elif model.task == "pose_estimation":
        if not args.coco_root:
            raise SystemExit("--coco-root is required")
        res = eval_coco_keypoints(model, args.coco_root, args.provider, args.limit)
        ds_id = "coco-val2017-keypoints"
    elif model.task == "optical_flow":
        if not args.sintel_root:
            raise SystemExit("--sintel-root is required")
        res = eval_sintel(model, args.sintel_root, args.provider, args.limit)
        ds_id = "sintel-train"
    elif model.task == "super_resolution":
        if not args.sr_root:
            raise SystemExit("--sr-root is required")
        res = eval_sr(model, args.sr_root, args.provider, args.limit)
        ds_id = "sr-x4"
    elif model.task == "background_removal":
        if not args.dis_root:
            raise SystemExit("--dis-root is required")
        res = eval_dis(model, args.dis_root, args.provider, args.limit)
        ds_id = "dis5k-vd"
    elif model.task == "face_detection":
        if not args.wider_root:
            raise SystemExit("--wider-root is required")
        res = eval_widerface(model, args.wider_root, args.provider, args.limit)
        ds_id = "widerface-val"
    elif model.task in ("object_detection", "segmentation"):
        if not args.coco_root:
            raise SystemExit("--coco-root is required")
        res = eval_coco_detection(model, args.coco_root, args.provider, args.limit)
        ds_id = "coco-val2017"
    else:
        raise SystemExit(f"no evaluator for task {model.task!r} yet")
    prov_file = model.weights_dir / "provenance.json"
    sha = json.loads(prov_file.read_text()).get("sha256") if prov_file.exists() else None
    runtime = "onnxruntime-" + args.provider.split("-")[0]
    precision = provider_precision(args.provider)
    rec = {
        "id": f"{ds_id}{'-limit' + str(args.limit) if args.limit else ''}_{runtime}_{precision}",
        "date": dt.date.today().isoformat(),
        **res,
        "runtime": runtime,
        "runtime_version": ort_version(),
        "precision": precision,
        "artifact": {"file": model.meta["artifacts"]["onnx"]["file"], "sha256": sha},
        "collection_commit": collection_commit(),
    }
    print(yaml.safe_dump(rec, sort_keys=False))
    if not args.limit:
        save(model, rec)


if __name__ == "__main__":
    main()
