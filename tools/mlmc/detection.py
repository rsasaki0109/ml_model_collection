"""Common pieces for object detection models.

Every ``object_detection/<model>/detector.py`` exposes a ``Detector`` whose
``__call__(frame_bgr)`` returns :class:`Detections` in original-image pixel
coordinates with COCO class names, so tools can treat all models the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

COCO80 = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# Original (sparse, 1..90) COCO category ids, in COCO80 order. Used by models
# whose class slots are category ids (torchvision, RF-DETR).
COCO91_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
              20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
              39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
              56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75,
              76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)
COCO91_ID_TO_NAME = dict(zip(COCO91_IDS, COCO80))


@dataclass
class Detections:
    boxes: np.ndarray    # (N, 4) float32, x1 y1 x2 y2 in source pixels
    scores: np.ndarray   # (N,) float32
    labels: list[str]    # (N,) COCO class names

    def __len__(self):
        return len(self.labels)

    def to_json(self) -> list[dict]:
        return [
            {"label": l, "score": round(float(s), 4),
             "box": [round(float(v), 1) for v in b]}
            for b, s, l in zip(self.boxes, self.scores, self.labels)
        ]


_CUDA = ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})

# provider name -> (ORT providers, precision). TensorRT engines are cached next
# to the ONNX file (weights/trt_cache/) so the build cost is paid once.
PROVIDERS = {
    "cpu": (["CPUExecutionProvider"], "fp32"),
    "cuda": ([_CUDA, "CPUExecutionProvider"], "fp32"),
    "tensorrt": (["TensorrtExecutionProvider", _CUDA, "CPUExecutionProvider"], "fp32"),
    "tensorrt-fp16": (["TensorrtExecutionProvider", _CUDA, "CPUExecutionProvider"], "fp16"),
}


def provider_precision(provider: str) -> str:
    return PROVIDERS[provider][1]


def ort_session(path, provider: str = "cuda"):
    import onnxruntime as ort

    if provider != "cpu" and hasattr(ort, "preload_dlls"):
        # Load CUDA/cuDNN DLLs shipped with PyTorch or nvidia-* wheels, if any.
        ort.preload_dlls()
    providers, precision = PROVIDERS[provider]
    if provider.startswith("tensorrt"):
        from pathlib import Path
        cache = Path(path).parent / "trt_cache"
        cache.mkdir(exist_ok=True)
        providers = [("TensorrtExecutionProvider", {
            "trt_fp16_enable": precision == "fp16",
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache),
            "trt_timing_cache_enable": True,
        })] + providers[1:]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(path), opts, providers=providers)
    wanted = "TensorrtExecutionProvider" if provider.startswith("tensorrt") else None
    if provider != "cpu" and sess.get_providers()[0] == "CPUExecutionProvider":
        raise RuntimeError(f"provider {provider!r} unavailable; got "
                           f"{sess.get_providers()}")
    if wanted and sess.get_providers()[0] != wanted:
        raise RuntimeError(f"TensorRT unavailable; got {sess.get_providers()}")
    return sess


def letterbox(img: np.ndarray, size: int, center: bool, pad: int = 114):
    """Resize keeping aspect ratio and pad to ``size`` x ``size``.

    Returns (padded image, scale, (pad_x, pad_y)).
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), pad, dtype=np.uint8)
    px, py = ((size - nw) // 2, (size - nh) // 2) if center else (0, 0)
    out[py:py + nh, px:px + nw] = resized
    return out, r, (px, py)


def nms_per_class(boxes, scores, class_ids, iou_thr: float):
    """Class-aware NMS via OpenCV. boxes are xyxy."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    # Offset boxes per class so one NMS call never mixes classes.
    offset = class_ids[:, None].astype(np.float32) * 4096.0
    b = boxes + offset
    xywh = np.concatenate([b[:, :2], b[:, 2:] - b[:, :2]], axis=1)
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, iou_thr)
    return np.asarray(keep, dtype=np.int64).reshape(-1)


def _color(label: str):
    # Deterministic per-label colour (str hash() is salted per process).
    h = sum(map(ord, label)) * 47
    hsv = np.uint8([[[h % 180, 200, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw(img: np.ndarray, det: Detections, thickness: int = 2) -> np.ndarray:
    out = img.copy()
    fs = max(0.38, 0.5 * max(out.shape[:2]) / 1280)
    for (x1, y1, x2, y2), s, l in zip(det.boxes, det.scores, det.labels):
        c = _color(l)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, c, thickness, cv2.LINE_AA)
        text = f"{l} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ty = max(p1[1], th + 4)
        cv2.rectangle(out, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty), c, -1)
        cv2.putText(out, text, (p1[0] + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (0, 0, 0), 1, cv2.LINE_AA)
    return out
