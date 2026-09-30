"""Run one model over a video and save annotated video + raw detections.

    python tools/run_video.py --model yolox_s --input assets/demo.mp4

Outputs (under ``outputs/<task>/<model>/<input stem>/``):
    annotated.mp4     every input frame, same fps / size: boxes drawn
                      (object detection) or a colourised depth map (depth)
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


def output_dir(model: Model, video: Path) -> Path:
    return REPO_ROOT / "outputs" / model.task / model.name / video.stem


def run(model: Model, video: Path, provider: str = "cuda",
        max_frames: int | None = None) -> Path:
    out_dir = output_dir(model, video)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = model.load_runner(provider=provider)
    is_det = model.task == "object_detection"

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n, t_total = 0, 0.0
    jf = open(out_dir / "detections.jsonl", "w") if is_det else None
    while max_frames is None or n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = runner(frame)
        t_total += time.perf_counter() - t0
        if is_det:
            writer.write(draw(frame, res))
            jf.write(json.dumps({"frame": n, "detections": res.to_json()}) + "\n")
        else:
            writer.write(colorize(res))
        n += 1
    if jf:
        jf.close()
    cap.release()
    writer.release()

    prov_file = model.weights_dir / "provenance.json"
    prov = json.loads(prov_file.read_text()) if prov_file.exists() else {}
    (out_dir / "run.json").write_text(json.dumps({
        "model": model.name, "input": video.name, "frames": n, "provider": provider,
        "artifact_sha256": prov.get("sha256"),
        # Wall time incl. pre/post-processing; NOT a benchmark (see benchmark.py).
        "mean_wall_ms_per_frame": round(1000 * t_total / max(n, 1), 2),
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
    args = ap.parse_args()
    run(get_model(args.model), args.input.resolve(), args.provider, args.max_frames)


if __name__ == "__main__":
    main()
