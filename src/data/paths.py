"""Folder of the committed manifests every script reads by default.

``data`` holds the original dataset (30 Sep 2026 labels, frozen split). Setting ``WOUND_DATA_DIR``
points every default at another dataset version with the same layout, e.g.
``data/relabel_2026_10_23`` for the T23 rerun with the 23 Oct 2026 labels.
"""

from __future__ import annotations

import os
from pathlib import Path


DATA_DIR = Path(os.environ.get("WOUND_DATA_DIR", "data"))
# Number of wound classes: 7, or 6 when cut and penetration wounds are merged into sharp injury (T39)
EXPECTED_CLASSES = int(os.environ.get("WOUND_N_CLASSES", "7"))
