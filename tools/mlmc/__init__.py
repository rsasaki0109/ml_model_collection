"""Shared helpers for ml_model_collection tools.

Kept deliberately small: only code that is already shared by more than one
model or tool lives here. Model-specific pre/post-processing stays in each
model directory (``<task>/<model>/detector.py``).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
