"""License categorisation.

Categories are a *reading aid* for comparison tables, not legal advice.
Anything not listed here is reported as ``unknown`` rather than guessed.
"""

PERMISSIVE = "permissive"
COPYLEFT = "copyleft"
RESTRICTED = "restricted"
UNKNOWN = "unknown"

# SPDX identifier -> category
CATEGORIES = {
    "Apache-2.0": PERMISSIVE,
    "MIT": PERMISSIVE,
    "BSD-2-Clause": PERMISSIVE,
    "BSD-3-Clause": PERMISSIVE,
    "CC0-1.0": PERMISSIVE,
    "CC-BY-4.0": PERMISSIVE,
    "LGPL-3.0-only": COPYLEFT,
    "GPL-2.0-only": COPYLEFT,
    "GPL-3.0-only": COPYLEFT,
    "GPL-3.0-or-later": COPYLEFT,
    "GPL-3.0": COPYLEFT,  # as reported by GitHub when only/or-later is not stated
    "AGPL-3.0-only": COPYLEFT,
    "AGPL-3.0": COPYLEFT,  # as reported by GitHub when only/or-later is not stated
    "CC-BY-SA-4.0": COPYLEFT,
    "CC-BY-NC-4.0": RESTRICTED,
    "CC-BY-NC-SA-4.0": RESTRICTED,
    "LicenseRef-NVIDIA-SCL-NC": RESTRICTED,  # NVIDIA Source Code License (non-commercial)
    "LicenseRef-InsightFace-NC": RESTRICTED,  # InsightFace models: non-commercial research only
    "LicenseRef-MagicLeap-NC": RESTRICTED,  # Magic Leap SuperPoint / SuperGlue: noncommercial research only
}

# How the license of a component was established.
#   explicit            upstream states a license for exactly this component
#   repository_license  shipped from a repository/release under that license,
#                       but upstream does not separately state a weights license
#   unknown             no statement found
STATUSES = ("explicit", "repository_license", "unknown")


# Display names for non-SPDX identifiers.
SHORT = {"LicenseRef-NVIDIA-SCL-NC": "NVIDIA-NC", "LicenseRef-InsightFace-NC": "InsightFace-NC",
         "LicenseRef-MagicLeap-NC": "MagicLeap-NC"}


def short(spdx: str | None) -> str:
    return SHORT.get(spdx, spdx) if spdx else "unknown"


def category(spdx: str | None) -> str:
    if not spdx:
        return UNKNOWN
    return CATEGORIES.get(spdx, UNKNOWN)


BADGE = {PERMISSIVE: "🟢", COPYLEFT: "🟡", RESTRICTED: "🔴", UNKNOWN: "⚪"}


def describe(entry: dict | None) -> str:
    """Short table cell for a license entry from model.yaml."""
    if not entry:
        return f"{BADGE[UNKNOWN]} unknown"
    spdx = entry.get("spdx")
    status = entry.get("status", "unknown")
    text = short(spdx)
    if status == "repository_license":
        text += "*"
    return f"{BADGE[category(spdx)]} {text}"
