"""Merge benchmark records produced elsewhere (e.g. Colab) into this repo.

    python tools/merge_benchmarks.py benchmarks_colab.zip
    python tools/build_readme.py

The zip is the one downloaded from tools/colab_benchmark.ipynb: it contains
``<task>/<model>/benchmarks.yaml`` files. Records are merged by ``id``
(an incoming record replaces a local record with the same id).
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark import save  # noqa: E402
from tools.mlmc.catalog import get_model  # noqa: E402
from tools.validate import BENCH_REQUIRED  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", type=Path)
    args = ap.parse_args()
    merged = 0
    with zipfile.ZipFile(args.zip) as z:
        for name in z.namelist():
            if not name.endswith("benchmarks.yaml"):
                continue
            model = get_model(Path(name).parent.name)
            for rec in yaml.safe_load(z.read(name)) or []:
                missing = [k for k in BENCH_REQUIRED if k not in rec]
                if missing:
                    raise SystemExit(f"{name}: record {rec.get('id')} lacks {missing}")
                save(model, rec)
                merged += 1
                print(f"{model.name}: {rec['id']}")
    print(f"merged {merged} record(s); now run tools/build_readme.py")


if __name__ == "__main__":
    main()
