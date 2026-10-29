"""Place kits: each module turns a patch into a realistic place.

A kit module defines ``NAME``, ``INDOOR`` (bool), ``DESCRIPTION`` (one line)
and ``build(ctx) -> dict`` (``kits.base.Ctx``); ``build`` must light the scene
(``ctx.looks.sky`` and/or lamps), lay the ground, call
``ctx.dress_primitives`` exactly once, and may add structure, props and
scenery. It returns a dict of what it chose (recorded in course.json).
"""

from __future__ import annotations

import importlib

#: kit name -> module (under blender/kits/); a kit missing its module is skipped by the planner
KITS = ["street", "alley", "port", "industrial", "construction", "garage", "warehouse", "farm",
        "forest", "park", "ruins", "backyard", "coast"]
#: backdrop kits: they open a delivered scene instead of building a place (``BACKDROP`` in the module).
#: Not in ``KITS``, so plan2/plan3 never plan them. None ship with this repo.
BACKDROPS: list[str] = []


def load(name: str):
    return importlib.import_module(f"kits.{name}")


def available() -> list[str]:
    out = []
    for k in KITS:
        try:
            load(k)
            out.append(k)
        except ModuleNotFoundError:
            pass
    return out
