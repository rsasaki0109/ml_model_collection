"""Run one model over a video and save annotated video + raw detections.

    python tools/run_video.py --model yolox_s --input assets/demo.mp4
    python tools/run_video.py --model owlv2_b16 --input assets/demo.mp4 --prompts "van,pedestrian,flag"

Outputs (under ``outputs/<task>/<model>/<input stem>/``):
    annotated.mp4     every input frame, same fps / size: boxes drawn
                      (object detection), a colourised depth map (depth) or
                      mask overlays (segmentation)
    detections.jsonl  object detection only: one JSON line per frame,
                      {"frame": i, "detections": [...]}
    run.json          settings used (provider, thresholds, input, model hash)

Frame i of annotated.mp4 always corresponds to frame i of the input, which is
what lets tools/make_comparison.py line different models up exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.mlmc import REPO_ROOT  # noqa: E402
from tools.mlmc.catalog import Model, get_model  # noqa: E402
from tools.mlmc.depth import colorize  # noqa: E402
from tools.mlmc.detection import draw  # noqa: E402
from tools.mlmc.face import render as render_faces  # noqa: E402
from tools.mlmc.flow import render as render_flow  # noqa: E402
from tools.mlmc.matching import render as render_matches  # noqa: E402
from tools.mlmc.matting import render as render_matte  # noqa: E402
from tools.mlmc.ocr import render as render_ocr  # noqa: E402
from tools.mlmc.pose import render as render_pose  # noqa: E402
from tools.mlmc.segmentation import render as render_seg  # noqa: E402
from tools.mlmc.sr import render as render_sr  # noqa: E402


# task -> render(source frame, runner result) -> BGR frame of the same size
RENDER = {
    "depth_estimation": lambda frame, res: colorize(res),
    "segmentation": render_seg,
    "pose_estimation": render_pose,
    "optical_flow": render_flow,
    "super_resolution": render_sr,
    "background_removal": render_matte,
    "face_detection": render_faces,
    "ocr": render_ocr,
    "feature_matching": render_matches,
}


def output_dir(model: Model, video: Path) -> Path:
    return REPO_ROOT / "outputs" / model.task / model.name / video.stem


def run(model: Model, video: Path, provider: str = "cuda",
        max_frames: int | None = None, prompts: list[str] | None = None,
        stride: int = 1) -> Path:
    """``stride`` > 1 runs the model on every ``stride``-th frame only and
    repeats its rendering in between (for slow models when only sampled frames
    are needed, e.g. the comparison GIF). Not used for object detection."""
    out_dir = output_dir(model, video)
    if prompts:
        out_dir = out_dir.with_name(out_dir.name + "_prompts")
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"provider": provider}
    if prompts:
        if not model.meta.get("open_vocabulary") or model.meta.get("prompts") == "fixed":
            raise SystemExit(f"{model.name} does not take runtime prompts")
        kwargs["prompts"] = prompts
    runner = model.load_runner(**kwargs)
    is_det = model.task == "object_detection"

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n, n_run, t_total, last = 0, 0, 0.0, None
    stride = 1 if is_det else max(1, stride)
    jf = open(out_dir / "detections.jsonl", "w") if is_det else None
    while max_frames is None or n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if n % stride:
            writer.write(last)
            n += 1
            continue
        t0 = time.perf_counter()
        res = runner(frame)
        t_total += time.perf_counter() - t0
        n_run += 1
        if is_det:
            writer.write(draw(frame, res))
            jf.write(json.dumps({"frame": n, "detections": res.to_json()}) + "\n")
        else:
            last = RENDER[model.task](frame, res)
            writer.write(last)
        n += 1
    if jf:
        jf.close()
    cap.release()
    writer.release()

    prov_file = model.weights_dir / "provenance.json"
    prov = json.loads(prov_file.read_text()) if prov_file.exists() else {}
    (out_dir / "run.json").write_text(json.dumps({
        "model": model.name, "input": video.name, "frames": n, "stride": stride, "provider": provider,
        "artifact_sha256": prov.get("sha256"),
        # Wall time incl. pre/post-processing; NOT a benchmark (see benchmark.py).
        "mean_wall_ms_per_frame": round(1000 * t_total / max(n_run, 1), 2),
    }, indent=2))
    print(f"[{model.name}] {n} frames -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--provider", default="cuda", choices=["cuda", "cpu", "tensorrt", "tensorrt-fp16"])
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--prompts", help="comma-separated class prompts (open-vocabulary models only), "
                                      "e.g. 'delivery van,pedestrian,street lamp'")
    args = ap.parse_args()
    prompts = [s.strip() for s in args.prompts.split(",") if s.strip()] if args.prompts else None
    run(get_model(args.model), args.input.resolve(), args.provider, args.max_frames, prompts)


if __name__ == "__main__":
    main()
