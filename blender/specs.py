"""Patch specs for the tools in this folder: ``envgen.primitives`` under the call signature the batch
planner and the gates were written against, ``build_spec(test, tier, family, seed, split)``. Runs in the
CLI's Python (numpy, scipy, scikit-fmm), not inside Blender: ``build_course.py`` only reads spec.json."""

from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envgen.primitives import build_spec as _build_spec  # noqa: E402


def build_spec(test: int, tier: str, family: str, seed: int, split: str = "train"):
    return _build_spec(family, seed, tier=tier, split=split, test=test)
