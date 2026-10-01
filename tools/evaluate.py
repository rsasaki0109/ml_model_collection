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
    ap.add_argument("--provider", default="cuda", choices=list(PROVIDERS))
    ap.add_argument("--limit", type=int, help="evaluate only the first N images (smoke test)")
    args = ap.parse_args()

    if not args.model:
        for m in all_models(args.task):
            if m.artifact_path("onnx").exists():
                print(f"== {m.name}", flush=True)
                roots = (["--coco-root", str(args.coco_root)] if args.coco_root else []) + \
                    (["--ade-root", str(args.ade_root)] if args.ade_root else []) + \
                    (["--sintel-root", str(args.sintel_root)] if args.sintel_root else [])
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
