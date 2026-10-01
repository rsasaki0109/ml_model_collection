# Multi-object tracking

Every tracker returns `Tracks` (`tools/mlmc/tracking.py`): boxes, persistent
integer ids, scores and labels for the tracks reported at the current frame.
A tracker "model" here is an algorithm plus a detector from this collection
(`detector:` in model.yaml; the ONNX artifact is reused, `fetch: reuse`).

![comparison](../assets/tracking_comparison.gif)

## Comparison

<!-- BEGIN:tracking_table -->
| Tracker | Code license | Detector | Appearance / camera motion | MOT17 test HOTA / MOTA / IDF1<br>(reported, MOT-trained detector) | MOT17 train HOTA / MOTA / IDF1<br>(measured, our detector) | Tesla T4 (Colab)<br>onnxruntime-cuda FP32 · ms / FPS | Tesla T4 (Colab)<br>onnxruntime-tensorrt FP16 · ms / FPS |
|---|---|---|---|---|---|---|---|
| [BoT-SORT](botsort) | 🟢 Apache-2.0 | [D-FINE-S](../object_detection/dfine_s) | no ReID; sparse-flow camera motion | [64.6 / 80.6 / 79.5](https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/README.md) | **39.6** / **33.8** / **46.0** | 17.9 / 56 | 6.1 / 163 |
| [ByteTrack](bytetrack) | 🟢 Apache-2.0 | [D-FINE-S](../object_detection/dfine_s) | motion only | [63.1 / 80.3 / 77.3](https://github.com/ifzhang/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/README.md) | **40.6** / **36.9** / **48.6** | 18.2 / 55 | 6.1 / 164 |
| [OC-SORT](ocsort) | 🟢 Apache-2.0 | [D-FINE-S](../object_detection/dfine_s) | motion only (observation-centric) | [63.2 / 78.0 / 77.5](https://github.com/noahcao/OC_SORT/blob/8462e7e729a93ccd3bd995c0a79a890336cb3a0b/README.md) | **37.0** / **31.8** / **42.6** | 18.1 / 55 | 6.1 / 163 |
<!-- END:tracking_table -->

## Implementation

Written from scratch for this repository (Apache-2.0) from the papers and the
published default parameters; no code copied. Reference implementations
(MIT): ifzhang/ByteTrack, NirAharon/BoT-SORT, noahcao/OC_SORT — whose Kalman
filters derive from nwojke/deep_sort (GPL-3.0), which is why nothing is
copied. Uses NumPy, SciPy `linear_sum_assignment` and OpenCV.

- **ByteTrack** — x-y-aspect-height Kalman filter; first association of all
  tracks with high-score detections (IoU x score), second association of the
  remaining tracked tracks with low-score detections (IoU 0.5), unconfirmed
  tracks (0.7), new tracks above track_thresh + 0.1, 30-frame buffer scaled by
  fps / 30.
- **BoT-SORT** (no ReID) — the same association with an x-y-w-h Kalman
  filter, BoT-SORT's thresholds (0.6 / 0.1 / 0.7) and camera-motion
  compensation: affine from sparse Lucas-Kanade flow at half resolution,
  applied to every track's state and covariance.
- **OC-SORT** — SORT Kalman filter on (x, y, area, ratio) with
  observation-centric momentum (direction consistency cost, inertia 0.2,
  delta_t 3), observation-centric recovery (second round on last
  observations) and re-update along a virtual trajectory when a lost track is
  found again; reports the matched detection box.

All three drop boxes wider than 1.6 x their height and smaller than 10 px²
(the pedestrian setting of the references). Detections: D-FINE-S, class
person, score >= 0.1.

## Evaluation

MOT17 train (the 7 sequences, FRCNN copy — images are identical across the
DPM / FRCNN / SDP copies; only the public detections differ, and they are
not used), HOTA / MOTA / IDF1 with TrackEval (MIT). MOTChallenge data is
CC BY-NC-SA 3.0 (archived site; the evaluation server is offline since
2026-04-16). The reported numbers are MOT17 *test* with MOT17-trained
YOLOX-X detectors — not comparable with the measured column.
