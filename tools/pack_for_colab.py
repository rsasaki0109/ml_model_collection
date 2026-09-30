"""Zip the repository (without weights / outputs) for running on Colab.

    python tools/pack_for_colab.py            # -> dist/ml_model_collection.zip

Upload the zip in tools/colab_benchmark.ipynb when the repository is not
reachable by ``git clone`` (e.g. not pushed yet).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.mlmc import REPO_ROOT  # noqa: E402

EXCLUDE_DIRS = {".git", "weights", "outputs", "dist", "__pycache__"}


def main():
    out = REPO_ROOT / "dist" / "ml_model_collection.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(REPO_ROOT.rglob("*")):
            rel = p.relative_to(REPO_ROOT)
            if p.is_dir() or EXCLUDE_DIRS & set(rel.parts) or rel.parts[0].startswith(".venv"):
                continue
            z.write(p, Path("ml_model_collection") / rel)
            n += 1
    print(f"wrote {out} ({n} files, {out.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
