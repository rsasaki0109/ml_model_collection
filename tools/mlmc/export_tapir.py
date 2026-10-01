"""Shared export for online (causal) BootsTAPIR (``point_tracking/*/export.py``).

Uses the PyTorch inference port ibaiGorordo/Tapir-Pytorch-Inference
(Apache-2.0; derived from google-deepmind/tapnet's PyTorch TAPIR, Apache-2.0)
at a pinned commit and DeepMind's ``causal_bootstapir_checkpoint.pt``
(Apache-2.0, SHA-256 pinned). Two graphs with a fixed number of points N (the port's reshapes bake N in;
runners pad or chunk query sets to N): ``<name>_encoder.onnx`` (query
features) and ``<name>.onnx`` (one frame).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from tools.mlmc.export import clone_at, download, patch_for_ort

CODE = "https://github.com/ibaiGorordo/Tapir-Pytorch-Inference"
COMMIT = "3b6e84523011976d25f37365e12881f16216003e"
CKPT_URL = "https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt"
CKPT_SHA256 = "87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f"


def build(here: Path, resolution: int, iters: int):
    src = clone_at(CODE, COMMIT, here / "weights" / "_src")
    sys.path.insert(0, str(src))
    from tapnet.tapir_inference import TapirPointEncoder, TapirPredictor, build_model
    ckpt = download(CKPT_URL, here / "weights" / "causal_bootstapir_checkpoint.pt", CKPT_SHA256)
    model = build_model(str(ckpt), (resolution, resolution), iters, True, torch.device("cpu")).eval()
    return model, TapirPredictor(model).eval(), TapirPointEncoder(model).eval()


def export(here: Path, name: str, resolution: int, iters: int, n: int = 256):
    model, predictor, encoder = build(here, resolution, iters)
    r = resolution
    state = torch.zeros((iters, model.num_mixer_blocks, n, 2, 512 + 2048))
    fg = torch.zeros((1, r // 8, r // 8, 256))
    hg = torch.zeros((1, r // 4, r // 4, 128))
    q = torch.rand((1, n, 2))
    frame = torch.rand((1, 3, r, r)) * 2 - 1
    qf, hqf = encoder(q, fg, hg)
    enc_out = here / "weights" / f"{name}_encoder.onnx"
    out = here / "weights" / f"{name}.onnx"
    with torch.no_grad():
        torch.onnx.export(encoder, (q, fg, hg), str(enc_out),
                          input_names=["query_points", "feature_grid", "hires_feats_grid"],
                          output_names=["query_feats", "hires_query_feats"],
                          opset_version=17, dynamo=False)
        torch.onnx.export(predictor, (frame, qf, hqf, state), str(out),
                          input_names=["input_frame", "query_feats", "hires_query_feats", "causal_state"],
                          output_names=["tracks", "visibles", "new_causal_state", "feature_grid", "hires_feats_grid"],
                          opset_version=17, dynamo=False)
    for f in (enc_out, out):
        changes = patch_for_ort(f)
        print(f"wrote {f}" + (f" (patched for ONNX Runtime: {changes})" if changes else ""))
