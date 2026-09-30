# Licenses

**Not legal advice.** This page explains how license information in this
repository is recorded so that it can be compared. Always read the linked
upstream text before using a model.

## Components are licensed separately

Each `model.yaml` records these independently:

| Field | What it covers | Example of divergence |
|---|---|---|
| `license.code` | Upstream training / model source code | Apache-2.0 repo… |
| `license.weights` | The pretrained parameters we fetch or export | …whose weights ship under a different or unstated license |
| `license.dataset` | Training data the weights were produced from | COCO: annotations CC BY 4.0, images under Flickr terms |
| `license.export_dependencies` | Packages needed to *produce* the artifact | `ultralytics` (AGPL-3.0) is needed to export YOLO11n |

Runtime dependencies used by this repo at inference time (ONNX Runtime: MIT,
OpenCV: Apache-2.0, NumPy: BSD-3-Clause) are the same for every model and are
not repeated per model.

## Categories

Used only to make tables scannable (`tools/mlmc/licenses.py`):

| Category | Meaning here | Examples |
|---|---|---|
| 🟢 permissive | Few conditions beyond attribution/notice | Apache-2.0, MIT, BSD-3-Clause |
| 🟡 copyleft | Derivatives / network use may have to be released under the same license | GPL-3.0, AGPL-3.0, CC-BY-SA |
| 🔴 restricted | Use itself is limited (e.g. non-commercial, research-only, custom terms) | CC-BY-NC-4.0, custom model licenses |
| ⚪ unknown | No license statement found | — |

A category is **not** a conclusion like "commercial use OK". Permissive
licenses still carry obligations, and the dataset terms may add others.

## How a weights license is established (`status`)

| status | Meaning | Table marker |
|---|---|---|
| `explicit` | Upstream states a license for exactly these weights (e.g. model-card metadata) | `Apache-2.0` |
| `repository_license` | Weights are published as a release asset of a repository under that license, with no separate statement | `Apache-2.0*` |
| `unknown` | No statement found; `spdx: null` | `unknown` |

Rules:

- Every entry has an `evidence` URL, pinned to a commit/revision where possible.
- If nothing official can be found, record `unknown`. Do not infer from
  blog posts, forks, or "everyone uses it commercially".
- Quote the upstream sentence in `note` when it is ambiguous.
