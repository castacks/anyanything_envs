"""Build, save and load patch specs (the ``spec.json`` a Blender course is built from)."""

from __future__ import annotations

import json
from pathlib import Path

from . import families as ds
from . import tiers

#: tier-A families: one obstacle (or one obstacle type) across the heading
TIER_A_FAMILIES = tuple(ds.FAMILIES_T5) + tuple(ds.FAMILIES_CLUTTER) + tuple(ds.FAMILIES_OPTIN)


def build_spec(family: str, seed: int, tier: str = "A", split: str = "train", test: int = 5) -> ds.PatchSpec:
    """One patch. Tier A: ``family`` names the obstacle; tier B (``corridor``) and tier C (``arena``) compose
    several. ``test`` 3/4 are the earlier fixed-goal series (wall_offset, wall_sym, float_box, stand_*)."""
    if test != 5:
        return ds.build_patch(test, family, seed, split)
    if tier == "B":
        return tiers.corridor_patch(seed, split)
    if tier == "C":
        return tiers.arena_patch(seed, split)
    return ds.build_patch(5, family, seed, split)


def save_spec(spec: ds.PatchSpec, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_json())
    return path


def load_spec(path) -> ds.PatchSpec:
    d = json.loads(Path(path).read_text())
    return ds.PatchSpec(**{k: d[k] for k in ds.PatchSpec.__dataclass_fields__})
