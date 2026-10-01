# Optical flow

Every model returns `Flow` (`tools/mlmc/flow.py`): per-pixel displacement
`(H, W, 2)` in source pixels from the previous frame to the current one.
Runners are stateful video runners (each call pairs the frame with the
previous one; the first frame is paired with itself).

![comparison](../assets/optical_flow_comparison.gif)

## Comparison

<!-- BEGIN:optical_flow_table -->
| Model | Code license | Weights license | Training data | Input | Sintel train EPE<br>clean / final (reported) | Sintel train EPE<br>clean / final (measured, ONNX) | Peak VRAM<br>(measured) |
|---|---|---|---|---|---|---|---|
| [NeuFlow-v2](neuflow_v2) | 🟢 Apache-2.0 | 🟢 Apache-2.0* | FlyingThings3D | 432×768 | [1.24 / 2.67](https://arxiv.org/abs/2408.10161) | – | not measured |
| [SEA-RAFT-M](sea_raft_m) | 🟢 BSD-3-Clause | 🟢 BSD-3-Clause | TartanAir, FlyingChairs, FlyingThings3D | 432×768 | – | – | not measured |
| [SEA-RAFT-S](sea_raft_s) | 🟢 BSD-3-Clause | 🟢 BSD-3-Clause | TartanAir, FlyingChairs, FlyingThings3D | 432×768 | [1.27 / 3.74](https://arxiv.org/abs/2405.14793) | – | not measured |
<!-- END:optical_flow_table -->

## Provenance

<!-- BEGIN:optical_flow_provenance -->
| Model | Source (pinned) | Artifact | How it is produced |
|---|---|---|---|
| NeuFlow-v2 | [neufieldrobotics/NeuFlow_v2@204b5e3](https://github.com/neufieldrobotics/NeuFlow_v2/tree/204b5e3744461d90303b9ff82caa7a1bb56a2ca2) | `neuflow_v2.onnx` | [`export.py`](neuflow_v2/export.py) |
| SEA-RAFT-M | [princeton-vl/SEA-RAFT@9137517](https://github.com/princeton-vl/SEA-RAFT/tree/9137517ba24e628442aec097d3afe71d03503b75) | `sea_raft_m.onnx` | [`export.py`](sea_raft_m/export.py) |
| SEA-RAFT-S | [princeton-vl/SEA-RAFT@9137517](https://github.com/princeton-vl/SEA-RAFT/tree/9137517ba24e628442aec097d3afe71d03503b75) | `sea_raft_s.onnx` | [`export.py`](sea_raft_s/export.py) |
<!-- END:optical_flow_provenance -->

## Per-model notes

All three are exported from the official code with fixed 432x768 inputs
(`image1`, `image2`: RGB float 0-255; normalisation is inside the graph) and
checked against PyTorch on two demo frames (max |Δflow| ≤ 0.005 px).

- **NeuFlow-v2** — `neuflow_things.pth` from the repository (SHA-256 pinned
  in `export.py`). Conv+BN fused as in upstream `infer.py`; 1 refinement
  iteration at 1/16 and 8 at 1/8 (paper default); FP32 buffers
  (`init_bhwd(amp=False)`). Upstream bakes the resolution into buffers, so
  the graph is fixed.
- **SEA-RAFT-S / -M** — Hugging Face `MemorySlices/Tartan-C-T432x960-{S,M}`
  (pinned revisions), upstream `config/eval/sintel-{S,M}.json` (4
  iterations). Upstream's constructor would download torchvision ImageNet
  ResNet weights for initialisation; the export skips that because the
  checkpoint overwrites every tensor. The checkpoints store the
  `downsample.1` BatchNorm only once (it is the same module as `bn3`).

## Training data

| Dataset | Terms |
|---|---|
| FlyingChairs, FlyingThings3D | "research purposes only ... Any commercial use is prohibited" |
| TartanAir | CC BY 4.0 |
| MPI-Sintel (evaluation only here) | no license text found (copyright MPI) |

Fine-tuned "C+T+S+K+H" checkpoints also include KITTI (CC BY-NC-SA 3.0).
