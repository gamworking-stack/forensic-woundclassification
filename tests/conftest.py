"""Tests build their own 7-class data, so a job's dataset settings must not reach them.

A T39 job exports WOUND_N_CLASSES=6 (cut and penetration merged); the modules read it at import, which
happens after this file runs.
"""

import os

os.environ.pop("WOUND_N_CLASSES", None)
