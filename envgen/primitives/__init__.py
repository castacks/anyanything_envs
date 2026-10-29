"""Primitive obstacle patches: the layouts every Blender course is built on.

    from envgen.primitives import build_spec, patch_cloud
    spec = build_spec("wall_slot", seed=2000)              # tier A, one family
    spec = build_spec("corridor", seed=3007, tier="B")     # tier B: several obstacles in sequence
    spec = build_spec("arena", seed=3001, tier="C")        # tier C: many families on one map
    points, region = patch_cloud(spec)                     # surfaces at 0.1 m, floor grid at 0.25 m (region -1)
"""

from .families import (FAMILIES_CLUTTER, FAMILIES_OPTIN, FAMILIES_T5, PatchSpec, build_patch,  # noqa: F401
                       patch_cloud)
from .spec import TIER_A_FAMILIES, build_spec, load_spec, save_spec  # noqa: F401
