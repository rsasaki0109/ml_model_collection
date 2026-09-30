"""Export torchvision SSDLite320-MobileNetV3-Large (COCO) to ONNX.

Usage: python object_detection/ssdlite320_mobilenet_v3_large/export.py
Requires: torch, torchvision

The exported graph contains torchvision's own resize/normalize and NMS, so it
takes a single RGB image in [0, 1] and returns final detections.
"""

from pathlib import Path

import torch
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large)

OUT = Path(__file__).parent / "weights" / "ssdlite320_mobilenet_v3_large.onnx"


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):  # (1, 3, 320, 320)
        out = self.model([image[0]])[0]
        return out["boxes"], out["scores"], out["labels"]


def main():
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    model = ssdlite320_mobilenet_v3_large(weights=weights).eval()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        Wrapper(model), torch.rand(1, 3, 320, 320), str(OUT),
        input_names=["image"], output_names=["boxes", "scores", "labels"],
        opset_version=17, dynamo=False,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
