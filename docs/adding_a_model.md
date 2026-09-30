# Adding a model

1. **Create `<task>/<name>/model.yaml`.** Copy an existing one. Pin
   `source.commit` to the upstream commit you used.
2. **Record licenses with evidence.** Code and weights separately, each with
   an `evidence` URL. If you cannot find an official statement for the
   weights, write `spdx: null, status: unknown`. See [licenses.md](licenses.md).
3. **Make the artifact reproducible.**
   - Upstream publishes ONNX → `fetch: download` + `url`, then
     `python tools/fetch_model.py <name> --update-hash` pins the SHA-256.
   - Otherwise → write `export.py` that writes `weights/<file>.onnx` from a
     pinned upstream revision, and set `fetch: export`.
4. **Write `detector.py`** with a `Detector(model, provider=...)` class that
   has `preprocess`, `infer`, `postprocess` and `__call__(frame_bgr) ->
   Detections`. Match the upstream pre-processing exactly (colour order,
   letterbox vs. resize, normalisation) and say so in the docstring.
5. **Check it visually.**
   `python tools/run_video.py --model <name> --input assets/demo.mp4`
6. **Benchmark** (idle GPU):
   `python tools/benchmark.py --model <name> --provider cuda --hardware-label "..." --hardware-class ...`
   and optionally `--provider cpu`.
7. **Regenerate and validate.**
   ```bash
   python tools/make_comparison.py --task <task> --input assets/demo.mp4
   python tools/build_readme.py
   python tools/validate.py
   ```

Do not add numbers you did not measure with `tools/benchmark.py`, and do not
upgrade a license from `unknown` without an upstream source.

## Adding benchmarks for your hardware

Only step 6 and `tools/build_readme.py`. Use a descriptive
`--hardware-label` (e.g. `"RTX 4060 Laptop"`, `"Jetson Orin Nano 8GB"`), and
pick the closest `--hardware-class` from [hardware.md](hardware.md).
