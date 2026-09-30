# Assets

## `demo.mp4`

- Source: ["Road traffic on Stritarjeva street"](https://commons.wikimedia.org/wiki/File:Road_traffic_on_Stritarjeva_street.webm)
  by **Sounds of Changes**, recorded 2019-07-18 in Ljubljana.
- License: [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) (attribution required).
- Original file SHA-1 (Wikimedia): `ab2454025725bc59364b19398d25dd7ff16fd54e`
- Changes: trimmed to 3.0 s – 8.0 s (skipping the fade-in), scaled from
  1920×1080 to 1280×720, audio removed, re-encoded as H.264:

  ```bash
  ffmpeg -ss 3 -t 5 -i Road_traffic_on_Stritarjeva_street.webm \
      -vf scale=1280:720 -an -c:v libx264 -pix_fmt yuv420p -crf 24 assets/demo.mp4
  ```

The CC BY 3.0 license of the video is independent of this repository's
Apache-2.0 license.

## `object_detection_comparison.gif`

Generated, not hand-made:

```bash
python tools/make_comparison.py --task object_detection --input assets/demo.mp4
```

It is a derivative of `demo.mp4` (same attribution applies) with detections
drawn by the models listed in the header of each tile. Model outputs follow
the respective model licenses; see each `model.yaml`.
