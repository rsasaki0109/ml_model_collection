"""Find models by Task × License × Hardware × Runtime.

    python tools/find_models.py --task object_detection --license permissive
    python tools/find_models.py --max-vram-mb 4096 --hardware-class gaming_laptop
    python tools/find_models.py --license permissive --max-latency-ms 20 --runtime onnxruntime-cuda

``--license permissive`` requires BOTH the code and the weights license to be
permissive; a weights license of ``unknown`` never matches. Hardware filters
only match *measured* records — a model that has not been benchmarked on a
matching configuration is not assumed to fit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.mlmc import hardware, licenses  # noqa: E402
from tools.mlmc.catalog import all_models  # noqa: E402

ORDER = [licenses.PERMISSIVE, licenses.COPYLEFT, licenses.RESTRICTED]


def license_ok(meta: dict, wanted: str | None) -> bool:
    if not wanted:
        return True
    lic = meta.get("license", {})
    cats = [licenses.category((lic.get(k) or {}).get("spdx")) for k in ("code", "weights")]
    if licenses.UNKNOWN in cats:
        return False
    # "copyleft" means "copyleft or more permissive", etc.
    return all(ORDER.index(c) <= ORDER.index(wanted) for c in cats)


def bench_ok(b: dict, args) -> bool:
    if args.hardware_class and b["hardware"]["class"] != args.hardware_class:
        return False
    if args.runtime and b["runtime"] != args.runtime:
        return False
    if args.max_vram_mb is not None:
        if b.get("peak_vram_mb") is None or b["peak_vram_mb"] > args.max_vram_mb:
            return False
    if args.max_latency_ms is not None and b["latency_ms"]["mean"] > args.max_latency_ms:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task")
    ap.add_argument("--license", choices=ORDER,
                    help="most restrictive category allowed for code AND weights")
    ap.add_argument("--max-vram-mb", type=float)
    ap.add_argument("--vram-tier", choices=[t for t, _ in hardware.VRAM_TIERS],
                    help="shortcut for --max-vram-mb")
    ap.add_argument("--hardware-class", choices=hardware.HARDWARE_CLASSES)
    ap.add_argument("--runtime", help="e.g. onnxruntime-cuda, onnxruntime-cpu")
    ap.add_argument("--max-latency-ms", type=float)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # emoji badges on Windows consoles
    if args.vram_tier:
        args.max_vram_mb = dict(hardware.VRAM_TIERS)[args.vram_tier]
    hw_filter = any(v is not None for v in (args.max_vram_mb, args.hardware_class,
                                            args.runtime, args.max_latency_ms))

    rows = []
    for m in all_models(args.task):
        if not license_ok(m.meta, args.license):
            continue
        lic = m.meta.get("license", {})
        lic_s = f"{licenses.describe(lic.get('code'))} / {licenses.describe(lic.get('weights'))}"
        benches = [b for b in m.benchmarks if bench_ok(b, args)]
        if hw_filter and not benches:
            continue
        if not benches:
            rows.append((m.name, lic_s, "(no benchmark)", "", "", ""))
        for b in benches:
            vram = f"{b['peak_vram_mb']} MB" if b.get("peak_vram_mb") is not None else "-"
            rows.append((m.name, lic_s, b["hardware"]["label"], b["runtime"],
                         f"{b['latency_ms']['mean']:.1f} ms", vram))

    if not rows:
        print("no matching models")
        return
    head = ("model", "license (code / weights)", "hardware", "runtime", "latency", "peak VRAM")
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    for r in [head, *rows]:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    print("\n* = license inherited from the repository/release; "
          "weights have no separate license statement.")


if __name__ == "__main__":
    main()
