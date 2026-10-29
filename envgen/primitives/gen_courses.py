#!/usr/bin/env python3
"""Hard obstacle courses for a zero-shot generalisation test of a trained student.

Two suites, both written as a sim_run episode list, neither seen in training
(training was three single-obstacle families -- offset wall, symmetric wall,
floating box -- with the goal 4-5 m past the obstacle):

  family    nine patch families (three tree-stand densities, chicane, wall slot,
            window, room with a doorway, bars+lintel, ring with one gap,
            wide floating bar) plus random arena maps whose route crosses
            several families in sequence.
  corridor  corridors of increasing length with obstacles in sequence:
            15 m/1, 25 m/2, 40 m/3, 60 m/5, mixed families, spaced so
            consecutive obstacles are not visible as one.

    ./scripts/gen_courses.py --suite family   --patches 25 --out tmp/gen/family
    ./scripts/gen_courses.py --suite corridor --patches 25 --out tmp/gen/corridor
"""
from __future__ import annotations

import argparse
import json
import zlib
import math
import sys
from pathlib import Path

import numpy as np

from . import arena as A
from . import patch as P

CRUISE = 3.0
PASS_SHARE = 0.26          # share of goals that are pass-through at 2 m/s
GOAL_PAST = 4.5            # metres past the last obstacle's far face
Y_HALF = P.Y_HALF
Z = P.FLIGHT_Z


# --- one obstacle, placed at x = d, returning (prims, far_face_x, face_x) ------

def _stand(rng, d, spacing):
    prims = []
    P.stand(prims, rng, d, depth=10.0, spacing=spacing)
    return prims, d + 10.0, d

def _chicane(rng, d):
    prims = []
    P.chicane(prims, d, gap=1.8, depth=4.0)
    return prims, d + 4.0, d

def _wall_slot(rng, d):
    prims = []
    P.wall_slot(prims, d, width=1.4, y=float(rng.uniform(-2.5, 2.5)))
    return prims, d + 0.3, d

def _window(rng, d):
    prims = []
    P.window(prims, d, width=1.2, height=0.9, sill=1.3, y=float(rng.uniform(-1.5, 1.5)))
    return prims, d + 0.25, d

def _room(rng, d):
    prims = []
    P.room(prims, d, w=6.0, depth=6.0, door_w=1.2, door_y=float(rng.uniform(-1.2, 1.2)))
    return prims, d + 6.0, d

def _bars_lintel(rng, d):
    prims = []
    P.lintel(prims, d, z_under=1.3)
    P.bars(prims, rng, d + 3.0, n=3, depth=6.0)
    return prims, d + 9.0, d

def _ring(rng, d):
    prims = []
    P.ring(prims, (d + 2.5, 0.0), r=2.5, gap_deg=70.0, gap_at_deg=180.0)
    return prims, d + 5.0, d

def _float_bar(rng, d):
    """A wide slab floating at flight height: over, under, or round 6 m of it."""
    prims = [A.box((d + 0.3, 0.0, Z), (0.3, 3.0, 0.55), (0, 0, 0), "float_bar")]
    return prims, d + 0.6, d

def _float_box(rng, d):
    prims = [A.box((d + 0.25, float(rng.uniform(-1.0, 1.0)), Z), (0.25, 1.0, 0.8), (0, 0, 0), "float_box")]
    return prims, d + 0.5, d

def _wall(rng, d):
    prims = []
    P.wall_slot(prims, d, width=1.4, y=float(rng.uniform(-3.0, 3.0)))
    return prims, d + 0.3, d

FAMILY = {
    "stand_sparse": lambda r, d: _stand(r, d, 3.5),
    "stand_medium": lambda r, d: _stand(r, d, 2.5),
    "stand_dense":  lambda r, d: _stand(r, d, 2.0),
    "chicane":      _chicane,
    "wall_slot":    _wall_slot,
    "window":       _window,
    "room":         _room,
    "bars_lintel":  _bars_lintel,
    "ring":         _ring,
    "float_bar":    _float_bar,
}
#: the mix a corridor draws its legs from
def _seed(*parts) -> int:
    """Stable across processes, unlike hash() on strings."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


CORRIDOR_MIX = ["wall_slot", "float_box", "stand_medium", "window", "float_bar"]
CORRIDOR_LEG = {"wall_slot": _wall_slot, "float_box": _float_box, "stand_medium": lambda r, d: _stand(r, d, 2.5),
                "window": _window, "float_bar": _float_bar}
LENGTHS = [(15.0, 1), (25.0, 2), (40.0, 3), (60.0, 5)]


def cloud(prims, bounds, spacing=0.1, floor_spacing=0.25):
    names = sorted({p["region"] for p in prims})
    scene = A.Arena(seed=0, size=0.0, flight_band=P.FLIGHT_BAND, robot_radius=P.ROBOT_RADIUS,
                    primitives=prims, regions={n: list(bounds) for n in names})
    if prims:
        pts, ids = A.sample_surfaces(scene, spacing=spacing, floor_spacing=1e9, floor_margin=0.0)
        pts = pts[ids >= 0]
    else:
        pts = np.zeros((0, 3))
    x0, y0, x1, y1 = bounds
    xs = np.arange(x0, x1 + floor_spacing / 2, floor_spacing)
    ys = np.arange(y0, y1 + floor_spacing / 2, floor_spacing)
    X, Y = np.meshgrid(xs, ys)
    floor = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    return np.concatenate([pts, floor])


def family_patch(fam, seed):
    rng = np.random.default_rng(_seed(fam, seed))
    d = 5.0
    prims, far, face = FAMILY[fam](rng, d)
    goal = [far + GOAL_PAST, float(rng.uniform(-1.0, 1.0)), Z]
    bounds = [-8.0, -Y_HALF, goal[0] + 6.0, Y_HALF]
    return dict(family=fam, seed=seed, prims=prims, bounds=bounds, goal=goal,
                faces=[face], start=[0.0, 0.0, Z], length=goal[0])


def corridor_patch(length, n_obs, seed):
    rng = np.random.default_rng(_seed("corr", length, seed))
    prims, faces = [], []
    span = length - 6.0
    step = span / n_obs
    fams = list(rng.choice(CORRIDOR_MIX, size=n_obs, replace=n_obs > len(CORRIDOR_MIX)))
    x = 5.0
    for i, fam in enumerate(fams):
        pr, far, face = CORRIDOR_LEG[fam](rng, x)
        prims += pr; faces.append(face)
        x = max(far + 8.0, 5.0 + (i + 1) * step)     # >= 8 m apart: not one obstacle
    goal = [length, float(rng.uniform(-1.5, 1.5)), Z]
    bounds = [-8.0, -Y_HALF, length + 6.0, Y_HALF]
    return dict(family=f"corr_{int(length)}m", seed=seed, prims=prims, bounds=bounds, goal=goal,
                faces=faces, start=[0.0, 0.0, Z], length=length, legs=[str(f) for f in fams])


def arena_patch(seed):
    sc = A.generate_random(seed, size=float(np.random.default_rng(seed).uniform(30.0, 40.0)))
    start = list(sc.start); goal = list(sc.goal)
    return dict(family="arena", seed=seed, prims=sc.primitives, bounds=[-sc.size / 2, -sc.size / 2, sc.size / 2, sc.size / 2],
                goal=goal, faces=[], start=start, length=float(np.linalg.norm(np.array(goal) - np.array(start))))
