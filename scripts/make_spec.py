#!/usr/bin/env python3
"""Write one patch spec (the spec.json a Blender course is built from).

    python scripts/make_spec.py --family wall_slot --seed 2000 --out out/wall_slot_2000/spec.json
    python scripts/make_spec.py --tier B --seed 3007 --out out/corridor_3007/spec.json     # a corridor
    python scripts/make_spec.py --tier C --seed 3001 --out out/arena_3001/spec.json        # an arena
    python scripts/make_spec.py --list                                                     # the tier-A families
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envgen.primitives import TIER_A_FAMILIES, build_spec, save_spec  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", default=None, help="tier-A family (see --list)")
    ap.add_argument("--tier", choices=["A", "B", "C"], default="A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(TIER_A_FAMILIES))
        return 0
    family = {"B": "corridor", "C": "arena"}.get(a.tier, a.family)
    if family is None or a.out is None:
        ap.error("--family (tier A) and --out are required")
    spec = build_spec(family, a.seed, tier=a.tier, split=a.split)
    print(save_spec(spec, a.out), f"({len(spec.primitives)} primitives)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
