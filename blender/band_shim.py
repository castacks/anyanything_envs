"""``envgen.band`` for the Blender side: the package import when it works (the CLI), else the module
loaded by file path (Blender's Python may lack the package's dependencies). Either way the same module
object is shared by everything in the process, so ``band.use()`` rebinding reaches every bound constant."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_PATH = Path(__file__).resolve().parent.parent / "envgen" / "band.py"


def load():
    m = sys.modules.get("envgen.band") or sys.modules.get("_envgen_band")
    if m is not None:
        return m
    try:
        from envgen import band as m  # noqa: PLC0415
        return m
    except Exception:  # noqa: BLE001  (the package's dependencies missing: Blender)
        spec = importlib.util.spec_from_file_location("_envgen_band", _PATH)
        m = importlib.util.module_from_spec(spec)
        sys.modules["_envgen_band"] = m
        spec.loader.exec_module(m)
        return m


band = load()
