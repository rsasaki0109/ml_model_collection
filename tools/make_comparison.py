"""Build a side-by-side comparison GIF/MP4 from several models' outputs.

    python tools/make_comparison.py --task object_detection --input assets/demo.mp4

Steps:
  1. For each model (default: every model of --task with a fetched artifact),
     reuse ``outputs/<task>/<model>/<clip>/detections.jsonl`` or create it with
     tools/run_video.py (``--rerun`` forces this).
  2. Take the same frame indices from the input video, resize them to one
     tile size and draw each model's detections for exactly that frame
     (other tasks, e.g. depth: take the same frames from each model's
     rendered annotated.mp4).
     Drawing at tile resolution (instead of downscaling annotated.mp4) keeps
     labels readable in the GIF.
  3. Add a header bar with the model name and licenses, arrange tiles in a
     grid and encode a GIF (ffmpeg palette) and optionally an MP4.

Defaults (model line-up, columns, tile width, fps) are read from
``<task>/comparison.yaml`` when it exists; command-line flags override them.
Without either, every fetched model of the task is used, alphabetically.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.mlmc import REPO_ROOT  # noqa: E402
from tools.mlmc.catalog import Model, all_models, get_model  # noqa: E402
from tools.mlmc.detection import Detections, draw  # noqa: E402
from tools.run_video import output_dir, run  # noqa: E402

HEADER_H = 44
BG = (24, 24, 24)


def license_text(model: Model) -> str:
    lic = model.meta.get("license", {})
    code = (lic.get("code") or {}).get("spdx") or "unknown"
    weights = (lic.get("weights") or {}).get("spdx") or "unknown"
    return f"code: {code}   weights: {weights}"


def header(model: Model, width: int) -> np.ndarray:
    bar = np.full((HEADER_H, width, 3), BG, np.uint8)
    cv2.putText(bar, model.display_name, (8, 21), cv2.FONT_HERSHEY_DUPLEX,
                0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, license_text(model), (8, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (185, 185, 185), 1, cv2.LINE_AA)
    return bar


def read_frames(path: Path, indices: list[int], size: tuple[int, int]):
    cap = cv2.VideoCapture(str(path))
    wanted, frames, i = set(indices), {}, 0
    while i <= max(indices):
        ok, f = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
        i += 1
    cap.release()
    missing = wanted - frames.keys()
    if missing:
        raise SystemExit(f"{path} is missing frames {sorted(missing)[:5]}...")
    return [frames[k] for k in indices]


def read_detections(path: Path, indices: list[int], scale: float):
    by_frame = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            by_frame[rec["frame"]] = rec["detections"]
    out = []
    for i in indices:
        if i not in by_frame:
            raise SystemExit(f"{path} has no record for frame {i}; use --rerun")
        dets = by_frame[i]
        out.append(Detections(
            boxes=np.array([d["box"] for d in dets], np.float32).reshape(-1, 4) * scale,
            scores=np.array([d["score"] for d in dets], np.float32),
            labels=[d["label"] for d in dets]))
    return out


def encode(frames_dir: Path, fps: float, gif: Path | None, mp4: Path | None):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required (https://ffmpeg.org)")
    src = ["-framerate", f"{fps}", "-i", str(frames_dir / "%05d.png")]
    if gif:
        gif.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", *src, "-vf",
                        "split[a][b];[a]palettegen=max_colors=128[p];"
                        "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
                        "-loop", "0", str(gif)], check=True)
        print(f"wrote {gif} ({gif.stat().st_size / 2**20:.1f} MB)")
    if mp4:
        mp4.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", *src, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "23", str(mp4)], check=True)
        print(f"wrote {mp4} ({mp4.stat().st_size / 2**20:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="object_detection")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--models", nargs="+", help="model names, in display order")
    ap.add_argument("--cols", type=int)
    ap.add_argument("--tile-width", type=int)
    ap.add_argument("--fps", type=float, help="output frame rate")
    ap.add_argument("--start", type=float, default=0, help="seconds")
    ap.add_argument("--duration", type=float, help="seconds (default: whole clip)")
    ap.add_argument("--provider", default="cuda", choices=["cuda", "cpu", "tensorrt", "tensorrt-fp16"])
    ap.add_argument("--rerun", action="store_true", help="re-run inference")
    ap.add_argument("--gif", type=Path, help="default: assets/<task>_comparison.gif")
    ap.add_argument("--mp4", type=Path, help="also write an MP4")
    args = ap.parse_args()

    cfg_file = REPO_ROOT / args.task / "comparison.yaml"
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
    args.models = args.models or cfg.get("models")
    args.cols = args.cols or cfg.get("cols", 2)
    args.tile_width = args.tile_width or cfg.get("tile_width", 480)
    args.fps = args.fps or cfg.get("fps", 5)

    video = args.input.resolve()
    if args.models:
        models = [get_model(n) for n in args.models]
    else:
        models = sorted((m for m in all_models(args.task)
                         if m.artifact_path("onnx").exists()),
                        key=lambda m: m.display_name.lower())
    if not models:
        raise SystemExit("no models with fetched artifacts; run tools/fetch_model.py")

    is_det = args.task == "object_detection"
    marker = "detections.jsonl" if is_det else "annotated.mp4"
    for m in models:
        if args.rerun or not (output_dir(m, video) / marker).exists():
            run(m, video, args.provider)

    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    first = int(args.start * src_fps)
    last = n_src if args.duration is None else min(n_src, first + int(args.duration * src_fps))
    step = src_fps / args.fps
    indices = [int(first + k * step) for k in range(int((last - first) / step))]

    tw = args.tile_width
    th = int(round(h * tw / w / 2)) * 2
    frames = read_frames(video, indices, (tw, th))
    if is_det:
        dets = [read_detections(output_dir(m, video) / "detections.jsonl", indices, tw / w)
                for m in models]
    else:  # e.g. depth: use each model's rendered frames directly
        rendered = [read_frames(output_dir(m, video) / "annotated.mp4", indices, (tw, th))
                    for m in models]
    headers = [header(m, tw) for m in models]

    cols = min(args.cols, len(models))
    rows = math.ceil(len(models) / cols)
    gap = 2
    cell_w, cell_h = tw + gap, HEADER_H + th + gap
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k, frame in enumerate(frames):
            canvas = np.full((rows * cell_h - gap, cols * cell_w - gap, 3), BG, np.uint8)
            for j in range(len(models)):
                r, c = divmod(j, cols)
                y, x = r * cell_h, c * cell_w
                canvas[y:y + HEADER_H, x:x + tw] = headers[j]
                canvas[y + HEADER_H:y + HEADER_H + th, x:x + tw] = \
                    draw(frame, dets[j][k], thickness=1) if is_det else rendered[j][k]
            cv2.imwrite(str(tmp / f"{k:05d}.png"), canvas)
        gif = args.gif or REPO_ROOT / "assets" / f"{args.task}_comparison.gif"
        encode(tmp, args.fps, gif, args.mp4)


if __name__ == "__main__":
    main()
