"""Tier B (sequences) and tier C (random arena maps) patches for test 5
(deliverables/2026-09-11/test5_primitives_finale.md section 3).

Tier A -- one obstacle in front of the state -- is ``draw_states.build_patch``
and stays there. The two tiers here are the compositional ones:

    corridor   15-40 m holding 2-4 obstacles drawn from the twelve tier-A
               families, the clear gap between consecutive obstacles 4-10 m so
               that two are usually in view at once. ``scripts/gen_courses.py``
               builds corridors already but spaces legs >= 8 m apart precisely
               so consecutive obstacles are *not* seen as one -- an evaluation
               choice, and the opposite of what training wants (finding 2).
               Its leg builders are reused; its spacing rule is not.
    arena      ``arena.generate_random`` at 30-45 m, cell target 10-13 m, the
               held-out families dropped, with a start and a goal 8-25 m apart
               across 1-3 cells and reachable.

Both return a ``draw_states.PatchSpec`` in the patch frame the rest of the
pipeline assumes (``PatchSpec.as_patch`` hard-codes it): the state at the
origin flying +x, floor at z = 0, flight altitude ``FLIGHT_Z``. A corridor is
built in that frame. An arena is drawn on its own map and then moved into it
-- a quarter turn about z, chosen so the route runs roughly +x, then a
translation putting the start at the origin. Quarter turns are exact and keep
the map's bounding box square, so the grid does not grow; the +-25 deg of
bearing jitter is what gives the route its angle to the cell walls.

Goals are validated, not hoped for: the builder samples the cloud -- always the
primitives', never a Blender course's (``draw_states.primitive_clouds``), since
the course is built from the spec this returns -- builds the
``PatchGrid`` the draw uses, and redraws while the start is not free, the goal
has less than 1 m of clearance, or the goal is not reachable from the start
*inside the flight band*. ``params['attempts']`` and ``params['rejected']``
record what that took; ``PatchUnbuildable`` is what a seed that never came
good raises, so a worker can write a skip record instead of dying.

``params['face_records']`` is the tier's contract with ``draw_states``: one
entry per leg (tier B) or per obstruction along the start->goal chord (tier C),
in route order, each ``{face_x, far_x, y_c, half_width, family, kind}``. It is
written through ``draw_states.face_params``, which projects the same list into
``face_spans`` (``[[face_x, far_x, y_c, half_width], ...]``, what
``draw_states.faces`` reads), ``faces`` (the bare near-face x) and
``far_faces``. One list is built; the four keys are its projections, so they
cannot disagree, and ``faces`` is a list of floats here exactly as it is in
tier A -- it used to be this list of dicts, and a shard mixing the tiers then
had both types under the one name. Two rules the records exist to keep:

  * every leg is in it, not only the first. A corridor holds 2-4 obstacles and
    the zones have to be drawn among them, not all in front of leg 0.
  * ``y_c`` and ``half_width`` are a *centred* pair -- the obstruction spans
    ``[y_c - half_width, y_c + half_width]`` -- because that is what
    ``draw_states._span`` needs. An outer extent measured from y = 0 makes an
    offset wall's span cover the whole corridor and empties ``beside``.

Faces are measured off the same 0.1 m surface cloud the ``PatchGrid`` is built
from, inside the flight band, never taken from the leg builder's nominal ``d``:
that is 0.34 m out on a ring (the 70 deg near-side gap), 0.15 m on a slotted
wall (half its thickness), 0.05-0.31 m on a stand (the trunk radius), and up to
5 m on bars+lintel (the last bar is drawn anywhere in a 6 m window).

``spec.face_x``/``spec.far_x`` stay the first entry's, for everything that
reads them today.

``split`` is recorded, not seeded, exactly as ``build_patch`` records it: the
train/eval separation is the seed offset of ``draw_states.seeds_for``.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import zlib

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree

from . import families as ds

from . import arena as A
from . import patch as PT


TEST = 5
FLIGHT_Z = ds.FLIGHT_Z
FLIGHT_BAND = ds.FLIGHT_BAND              # the band states are drawn in and routes must stay in
Y_HALF = PT.Y_HALF                        # corridor half width, as gen_courses

#: tier B: corridor length, obstacle count, and the clear gap between the far
#: face of one obstacle and the near face of the next.
CORRIDOR_LEN = (15.0, 40.0)
CORRIDOR_N = (2, 4)
CORRIDOR_GAP = (4.0, 10.0)
#: tier C: map side, cell target, goal distance from the start, cells crossed.
ARENA_SIZE = (30.0, 45.0)
ARENA_CELL = (10.0, 13.0)
ARENA_GOAL_D = (8.0, 25.0)
ARENA_CELLS = (1, 3)
#: never trained on (plan section 3, "held out"): the live zero-shot probe.
ARENA_HELD_OUT = ("gate", "box_yard")
#: ``chicane`` is held out at *family* level pending its label QA (plan section
#: 3, "held out", conditional), so it is out of a tier-B leg and out of a tier-C
#: cell alike -- an arena cell is the same geometry under another name.
ARENA_EXCLUDE = ARENA_HELD_OUT + ("chicane",)

#: the goal must be this clear of geometry, the start at least this clear.
GOAL_CLEAR = 1.0
START_CLEAR = 0.5
#: margins used while drawing, above the acceptance thresholds, so that a draw
#: accepted on the point cloud is still accepted on the 0.15 m voxel grid.
_GOAL_CLEAR_DRAW = 1.25
_START_CLEAR_DRAW = 0.8
_MAP_MARGIN = 2.5                         # start and goal this far inside the map
#: tier C: the first obstruction must be this far ahead of the start, or the
#: approach band ([face - 5, face - 1] m) sits behind the origin and the zone
#: is drawn out of the map. Wave 1, seeds 0-59: faces at 0.00-1.88 m.
_FACE_MIN = 3.0
#: below this the chord counts as narrowed ("squeeze") rather than open.
_SQUEEZE_CLR = 1.0

_MAX_ATTEMPTS = 8                         # whole-patch redraws before giving up
_ARENA_TRIES = 40                         # start/goal draws inside one map
_CLOUD_SPACING = 0.1                      # as ``draw_states.patch_cloud``: measure what it will see


class PatchUnbuildable(RuntimeError):
    """No draw for this seed passed validation in ``_MAX_ATTEMPTS`` tries.

    A subclass of ``RuntimeError`` so nothing that catches the old raise
    breaks, and its own type so a worker can catch *this* and write the skip
    record the dataset builder's other failure paths write, rather than taking
    down the whole shard. ``.reasons`` is the per-attempt rejection list, which
    is what says whether a family is unlucky or unbuildable.
    """

    def __init__(self, what: str, reasons: list):
        super().__init__(f"{what}: {reasons}")
        self.what, self.reasons = what, list(reasons)


# --- seeding ---------------------------------------------------------------------

def _seed(*parts) -> int:
    """Stable across processes, unlike ``hash()`` on strings (as
    ``gen_courses._seed``)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng(_seed(*parts))


# --- rigid motion of a primitive list ---------------------------------------------
# Only quarter turns about z are used, so the rotation is exact (no 1e-17 dust
# in the spec) and a square map stays axis-aligned.

_QUAD = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


def _rpy_after_yaw(rpy, c, s) -> list:
    """Intrinsic-XYZ euler of ``Rz(psi) @ Rx Ry Rz``, the convention
    ``arena._euler_xyz`` samples boxes with."""
    M = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ A._euler_xyz(*rpy)
    p = math.asin(float(np.clip(M[0, 2], -1.0, 1.0)))
    if abs(M[0, 2]) < 1.0 - 1e-9:                       # cos(pitch) != 0
        return [math.atan2(-M[1, 2], M[2, 2]), p, math.atan2(-M[0, 1], M[0, 0])]
    return [0.0, p, math.atan2(M[1, 0], M[1, 1])]       # gimbal: fold roll into yaw


def _place(prims, quad: int, dx: float, dy: float) -> list:
    """``prims`` turned ``quad`` quarter turns about z, then translated."""
    c, s = _QUAD[quad % 4]

    def xy(p):
        return [c * p[0] - s * p[1] + dx, s * p[0] + c * p[1] + dy]

    def xyz(p):
        return xy(p) + [float(p[2])]

    out = []
    for p in prims:
        k = p["kind"]
        if k == "sphere":
            out.append(A.sphere(xyz(p["c"]), p["r"], p["region"]))
        elif k == "box":
            out.append(A.box(xyz(p["c"]), p["half"], _rpy_after_yaw(p["rpy"], c, s), p["region"]))
        elif k == "cyl_v":
            q = xy(p["c"])
            out.append(A.cyl_v(q[0], q[1], p["r"], p["z0"], p["z1"], p["region"]))
        elif k == "cyl_h":
            out.append(A.cyl_h(xyz(p["a"]), xyz(p["b"]), p["r"], p["region"]))
        elif k == "wall":
            w = A.wall(xy(p["p0"]), xy(p["p1"]), p["z0"], p["z1"], p["t"], p["region"], slots=p["slots"])
            out.append(w)
        else:
            raise ValueError(f"unknown primitive kind {k!r}")
    return out


# --- what a leg actually occupies --------------------------------------------------

def _extent(prims, band=FLIGHT_BAND) -> tuple[float, float, float, float]:
    """(x0, x1, y_c, half_width) of ``prims``' surfaces inside ``band``.

    Measured off the same 0.1 m cloud ``patch_cloud`` samples and the grid is
    voxelised from, so a face means "where the robot meets it", not "the d the
    builder was called with" -- those differ by up to 0.34 m, and by 5 m for
    bars+lintel, whose last bar is drawn anywhere in a 6 m window. Outside the
    band the geometry is there but cannot be flown into, so it does not set the
    face; a leg with nothing in the band (none today) falls back to all of it.
    """
    names = sorted({p["region"] for p in prims})
    sc = A.Arena(seed=0, size=0.0, flight_band=tuple(band), robot_radius=A.ROBOT_RADIUS,
                 primitives=list(prims), regions={n: [0.0, 0.0, 0.0, 0.0] for n in names})
    pts, ids = A.sample_surfaces(sc, spacing=_CLOUD_SPACING, floor_spacing=1e9, floor_margin=0.0)
    p = pts[ids >= 0]
    inb = p[(p[:, 2] >= band[0]) & (p[:, 2] <= band[1])]
    if len(inb):
        p = inb
    y0, y1 = float(p[:, 1].min()), float(p[:, 1].max())
    return float(p[:, 0].min()), float(p[:, 0].max()), 0.5 * (y0 + y1), 0.5 * (y1 - y0)


def _face_keys(records: list) -> dict:
    """Every face key of ``params``, projected from the one list of records by
    the one writer both tiers and tier A use (``draw_states.face_params``).
    Never a second draw, and never a second type under a shared name."""
    return ds.face_params(records)


# --- tier B: the legs -------------------------------------------------------------
# Every leg is built around a nominal near face at x = 0 and is then shifted so
# that its *measured* near face lands where the layout put it.

_GC = None


def _gen_courses():
    """The legacy leg builders (``gen_courses``)."""
    from . import gen_courses
    return gen_courses

def _gc_leg(rng, attr: str, name: str):
    prims, _far, _face = getattr(_gen_courses(), attr)(rng, 0.0)
    return prims, name


def _wall_leg(rng, sym: bool):
    """A wall across part of the corridor: around it, one side clearly better
    unless it is centred (as ``draw_states._wall``, swept wider)."""
    w = float(rng.uniform(1.0, 8.0))
    t = 0.3
    name = "wall_sym" if sym else "wall_offset"
    y_c = 0.0 if sym else float(rng.uniform(0.5, 3.0) * rng.choice([-1.0, 1.0]))
    prims = [A.wall((t / 2, y_c - w / 2), (t / 2, y_c + w / 2), 0.0, A.TALL, t, name)]
    return prims, name


def _pillar_leg(rng):
    r = float(rng.uniform(0.4, 1.0))
    y = float(rng.uniform(-1.0, 1.0))
    prims = []
    PT.pillar(prims, 0.0, r=r, y=y)
    return prims, "pillar"


def _slab_leg(rng):
    hw = float(rng.uniform(1.0, 2.5))
    y = float(rng.uniform(-1.0, 1.0))
    prims = []
    PT.slab(prims, 0.0, half=(0.3, hw, 2.0), y=y)
    return prims, "slab"


def _stand_leg(rng):
    spacing = float(rng.choice([3.5, 2.5, 2.0]))
    name = {3.5: "stand_sparse", 2.5: "stand_medium", 2.0: "stand_dense"}[spacing]
    prims, _far, _face = _gen_courses()._stand(rng, 0.0, spacing)
    return prims, name


#: bars+lintel: the underside the plan sweeps over (1.1-1.8 m), floored so the
#: leg is passable *inside the flight band*. The beam spans the full corridor
#: and is 2.6 m thick, so under it is the only way through, and the band floor
#: is 1.0 m: measured on the 0.15 m grid, a full-span lintel seals the band up
#: to 1.30 m and opens at 1.35 (one free layer, z = 1.125) and 1.60 (two).
#: ``gen_courses`` fixes it at 1.3, which is why every corridor with no route
#: inside the band held one of these legs.
_LINTEL_Z = (1.5, 1.8)


def _bars_lintel_leg(rng):
    prims = []
    PT.lintel(prims, 0.0, z_under=float(rng.uniform(*_LINTEL_Z)))
    PT.bars(prims, rng, 3.0, n=int(rng.integers(2, 5)), depth=6.0)
    return prims, "bars_lintel"


#: the twelve tier-A families a corridor leg is drawn from (plan section 3,
#: tier A table). ``chicane`` is not among them: it is held out at family level
#: pending its label QA, in tier C as well (``ARENA_EXCLUDE``). Values are
#: (builder, greatest *measured* depth it can draw) -- the bound the layout
#: budgets against, so it is the extent the cloud shows (a room's walls are
#: 0.2 m thick, a stand's trunks stick 0.35 m out of the disc, a bar's radius
#: out of its 6 m window), not the nominal one.
LEGS = {
    "wall_offset": (lambda rng: _wall_leg(rng, False), 0.3),
    "wall_sym":    (lambda rng: _wall_leg(rng, True), 0.3),
    "float_box":   (lambda rng: _gc_leg(rng, "_float_box", "float_box"), 0.5),
    "float_bar":   (lambda rng: _gc_leg(rng, "_float_bar", "float_bar"), 0.6),
    "wall_slot":   (lambda rng: _gc_leg(rng, "_wall_slot", "wall_slot"), 0.3),
    "window":      (lambda rng: _gc_leg(rng, "_window", "window"), 0.25),
    "room":        (lambda rng: _gc_leg(rng, "_room", "room"), 6.3),
    "bars_lintel": (_bars_lintel_leg, 9.2),
    "ring":        (lambda rng: _gc_leg(rng, "_ring", "ring"), 5.0),
    "pillar":      (_pillar_leg, 2.0),
    "slab":        (_slab_leg, 0.6),
    "stand":       (_stand_leg, 11.0),
}
_MIN_DEPTH = min(d for _, d in LEGS.values())
_THIN = tuple(k for k, (_, d) in LEGS.items() if d <= 0.6)


def _spread(vals, need: float, lo: float, hi: float):
    """Push ``need`` (signed) metres into ``vals``, each staying in [lo, hi],
    proportional to the headroom. Returns (vals, what did not fit)."""
    vals = [float(v) for v in vals]
    room = [(hi - v) if need > 0 else (v - lo) for v in vals]
    total = sum(room)
    if total <= 1e-9 or abs(need) < 1e-9:
        return vals, need
    take = min(abs(need), total)
    for i, r in enumerate(room):
        vals[i] += math.copysign(take * r / total, need)
    return vals, math.copysign(abs(need) - take, need)


def _draw_corridor(rng, seed: int, split: str) -> ds.PatchSpec:
    """One corridor: gaps first, then legs under what depth is left, so the
    length lands in [15, 40] m by construction rather than by rejection."""
    n = int(rng.integers(CORRIDOR_N[0], CORRIDOR_N[1] + 1))
    first = float(rng.uniform(4.0, 7.0))
    past = float(rng.uniform(4.0, 5.0))
    gaps = [float(v) for v in rng.uniform(*CORRIDOR_GAP, size=n - 1)]
    budget = CORRIDOR_LEN[1] - first - sum(gaps) - past
    legs, depths, ext, prim0 = [], [], [], []
    for i in range(n):
        left = budget - sum(depths) - (n - 1 - i) * _MIN_DEPTH
        pool = [k for k, (_, d) in LEGS.items() if d <= left] or list(_THIN)
        fam = pool[int(rng.integers(len(pool)))]
        pr, name = LEGS[fam][0](rng)
        x0, x1, y_c, hw = _extent(pr)          # what it occupies, not what it was asked for
        prim0.append(pr); ext.append((x0, y_c, hw)); depths.append(x1 - x0); legs.append(name)

    length = first + sum(depths) + sum(gaps) + past
    if length < CORRIDOR_LEN[0] or length > CORRIDOR_LEN[1]:
        target = float(np.clip(length, CORRIDOR_LEN[0] + rng.uniform(0.0, 1.5), CORRIDOR_LEN[1]))
        gaps, rest = _spread(gaps, target - length, *CORRIDOR_GAP)
        first = float(np.clip(first + rest, 3.0, 12.0))
        length = first + sum(depths) + sum(gaps) + past

    # place each leg so that its measured near face lands on the planned x: the
    # gaps are then clear gaps between surfaces, not between nominal faces.
    prims, records = [], []
    x = first
    for i in range(n):
        x0, y_c, hw = ext[i]
        prims += _place(prim0[i], 0, x - x0, 0.0)
        records.append({"face_x": round(x, 4), "far_x": round(x + depths[i], 4),
                        "y_c": round(y_c, 4), "half_width": round(hw, 4),
                        "family": legs[i], "kind": "leg"})
        if i < n - 1:
            x = records[i]["far_x"] + gaps[i]
    goal = [records[-1]["far_x"] + past, float(rng.uniform(-1.5, 1.5)), FLIGHT_Z]
    v_goal = 0.0 if rng.random() < 0.7 else 2.0
    params = {
        "length": round(float(goal[0]), 4), "n_obstacles": n, "legs": legs,
        **_face_keys(records),
        "spacings": [round(float(v), 4) for v in gaps], "depths": [round(float(v), 4) for v in depths],
        "first_face": round(float(first), 4), "goal_past": round(float(past), 4),
        "half_width": records[0]["half_width"], "y_c": records[0]["y_c"],
        "y_half": float(Y_HALF), "spacing_rule": "clear gap, far face to next near face",
        "face_rule": "measured on the 0.1 m cloud inside the flight band",
    }
    return ds.PatchSpec(test=TEST, family="corridor", seed=int(seed), split=split,
                        bounds=[-8.0, -Y_HALF, round(float(goal[0]) + 6.0, 4), Y_HALF],
                        flight_z=FLIGHT_Z, params=params, face_x=records[0]["face_x"],
                        far_x=records[0]["far_x"], goal_pos=[round(float(v), 4) for v in goal],
                        goal_speed=float(v_goal), goal_radius=ds.GOAL_RADIUS, primitives=prims)


# --- tier C: the arena ------------------------------------------------------------

def _cell_rects(scene) -> list:
    """(name, family, [x0, y0, x1, y1]) per map cell, read off the scene.

    ``generate_random`` already records every cell's rectangle; re-deriving the
    tiling from ``size / cell_target`` and addressing cells by a ``f"{c}{r}"``
    name is a second source of truth that happens to agree only while the cell
    count stays one digit.
    """
    return [(nm, scene.families[nm], [float(v) for v in scene.regions[nm]])
            for nm in sorted(scene.regions) if nm != "perimeter"]


def _cell_at(rects, p):
    """The (name, family) of the cell holding ``p``, or ``None`` outside them."""
    x, y = float(p[0]), float(p[1])
    for nm, fam, (x0, y0, x1, y1) in rects:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return nm, fam
    return None


def _cells_crossed(rects, a, b, cell: float) -> list:
    """The cells the straight a->b chord touches, in route order."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    steps = max(int(np.linalg.norm(b - a) / (0.25 * cell)) * 4, 8)
    out = []
    for t in np.linspace(0.0, 1.0, steps):
        hit = _cell_at(rects, a + t * (b - a))
        if hit is not None and hit not in out:
            out.append(hit)
    return out


def _place_rect(rect, quad: int, dx: float, dy: float) -> list:
    """``rect`` under the same quarter turn + shift ``_place`` applies; a
    quarter turn keeps an axis-aligned rectangle axis-aligned."""
    c, s = _QUAD[quad % 4]
    x0, y0, x1, y1 = rect
    xs = [c * x - s * y + dx for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    ys = [s * x + c * y + dy for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    return [min(xs), min(ys), max(xs), max(ys)]


def _chord_span(grid: ds.PatchGrid, i, j):
    """(y_c, half_width) of the unflyable lateral run through grid column ``i``
    at row ``j``, at flight altitude -- the span a state can be "in front of".

    Clearance, not occupancy: what bounds the span is where the robot cannot
    go, which is what the zone masks mean by in-span. It is therefore a robot
    radius wider on each side than the surface, where tier B's leg spans are
    the surface itself.
    """
    k = int(np.clip(round((FLIGHT_Z - grid.lo[2]) / grid.h - 0.5), 0, grid.dims[2] - 1))
    col = grid.clearance[i, :, k] < A.ROBOT_RADIUS
    if not col[j]:                                  # blocked above or below, not here
        return float(grid.lo[1] + (j + 0.5) * grid.h), 2.0
    j0 = j1 = int(j)
    while j0 > 0 and col[j0 - 1]:
        j0 -= 1
    while j1 + 1 < grid.dims[1] and col[j1 + 1]:
        j1 += 1
    y0 = grid.lo[1] + j0 * grid.h
    y1 = grid.lo[1] + (j1 + 1) * grid.h
    return float(0.5 * (y0 + y1)), float(0.5 * (y1 - y0))


def _chord_obstructions(grid: ds.PatchGrid, a, b, cells=()) -> list:
    """Every obstruction the route meets, in route order, read off the straight
    start->goal chord: ``{face_x, far_x, y_c, half_width, family, kind}``.

    An arena has no single obstacle, so "an obstruction" needs a convention:
    a maximal stretch of the chord the robot cannot fly -- too little
    clearance, or cut off from the goal (the sealed inside of a room or a
    closed ring reads as free space in a surface cloud, and is not). Each near
    face is refined to the first cell that actually holds surface, so it means
    the same thing as a tier-A ``face_x`` rather than sitting a robot radius in
    front of it. ``kind``: ``blocked`` for such a stretch, and when there is
    none, one entry -- ``squeeze`` where the chord narrows below
    ``_SQUEEZE_CLR``, else
    ``clear`` at the goal, which is what the lead budget wants for a state with
    nothing in its way. ``family`` is the map cell the face falls in, so the
    tier-C faces name a family the way tier-B legs do (``unknown`` only off
    the tiling, which the map's own perimeter makes unreachable).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(int(np.linalg.norm(b - a) / (grid.h / 2)), 4)
    ps = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
    idx = np.clip(np.floor((ps - grid.lo) / grid.h).astype(int), 0, grid.dims - 1)
    clr = grid.clearance[tuple(idx.T)]
    blocked = (clr < A.ROBOT_RADIUS) | ~np.isfinite(grid.T[tuple(idx.T)])

    def entry(i_face, i_far, kind):
        y_c, hw = (_chord_span(grid, idx[i_face, 0], idx[i_face, 1]) if kind == "blocked"
                   else (float(ps[i_face, 1]), 2.0))
        hit = _cell_at(cells, ps[i_face])
        return {"face_x": round(float(ps[i_face, 0]), 4), "far_x": round(float(ps[i_far, 0]), 4),
                "y_c": round(y_c, 4), "half_width": round(hw, 4),
                "family": hit[1] if hit else "unknown", "kind": kind}

    out, i = [], 0
    while i < n:
        if not blocked[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and blocked[j + 1]:
            j += 1
        solid = np.flatnonzero(clr[i:j + 1] <= 0.5 * grid.h)
        out.append(entry(i + int(solid[0]) if len(solid) else i, j, "blocked"))
        i = j + 1
    if not out:
        if clr.min() < _SQUEEZE_CLR:
            out.append(entry(int(np.argmin(clr)), int(np.argmin(clr)), "squeeze"))
        else:
            out.append(entry(n - 1, n - 1, "clear"))
    return out


def _clear_ahead(tree, s_xy, d_hat, z: float, reach: float = _FACE_MIN) -> bool:
    """The first ``reach`` metres of the chord clear of everything, by more
    than the squeeze threshold.

    A cheap stand-in, on the cloud, for the grid test ``arena_patch`` applies
    afterwards: clear by ``_SQUEEZE_CLR`` means neither a blocked stretch nor a
    squeeze can be found there, so the first face is at least ``_FACE_MIN``
    ahead. Without it a start drawn 0.8 m off a wall gives a first obstruction
    at 0.6-2.0 m and an approach band ([face - 5, face - 1] m) behind the
    origin. The extra 0.15 m is the 0.15 m voxel grid's diagonal: what the
    cloud says here, the grid's clearance has to agree with there.
    """
    t = np.arange(0.0, reach + 1e-9, 0.25)
    q = np.column_stack([s_xy[0] + t * d_hat[0], s_xy[1] + t * d_hat[1], np.full(len(t), z)])
    return float(tree.query(q)[0].min()) >= _SQUEEZE_CLR + 0.15


def _draw_arena(rng, seed: int, split: str, attempt: int):
    """One random map, one start/goal pair on it, moved into the patch frame.
    Returns ``(spec, cells_in_patch_frame)``, or ``None`` when no pair on this
    map met the draw's margins."""
    size = float(rng.uniform(*ARENA_SIZE))
    cell_target = float(rng.uniform(*ARENA_CELL))
    map_seed = _seed("t5-arena-map", seed, attempt)
    fams = [f for f in A.FAMILIES if f not in ARENA_EXCLUDE]
    scene = A.generate_random(map_seed, size=size, cell_target=cell_target,
                              nominal_goal=False, families=fams)
    rects = _cell_rects(scene)
    cell = min(r[2] - r[0] for _, _, r in rects)
    n = int(round(math.sqrt(len(rects))))
    lo = -size / 2.0
    z = FLIGHT_Z

    probe = ds.PatchSpec(test=TEST, family="arena", seed=int(seed), split=split,
                         bounds=[lo, lo, -lo, -lo], flight_z=z, params={}, face_x=0.0, far_x=0.0,
                         goal_pos=[0.0, 0.0, z], goal_speed=0.0, goal_radius=ds.GOAL_RADIUS,
                         primitives=scene.primitives)
    pts, reg = ds.patch_cloud(probe)
    tree = cKDTree(pts[reg >= 0])
    edge = lo + _MAP_MARGIN

    pair = None
    for _ in range(_ARENA_TRIES):
        s_xy = rng.uniform(edge, -edge, size=2)
        if float(tree.query([s_xy[0], s_xy[1], z])[0]) < _START_CLEAR_DRAW:
            continue
        quad = int(rng.integers(4))
        jitter = float(rng.uniform(-25.0, 25.0))
        th = math.radians(90.0 * quad + jitter)
        d_hat = np.array([math.cos(th), math.sin(th)])
        # how far the chord can run before it leaves the map's inner square
        lim = [((-edge if d_hat[i] > 0 else edge) - s_xy[i]) / d_hat[i] for i in range(2) if abs(d_hat[i]) > 1e-6]
        d_max = min(lim) if lim else 0.0
        if d_max < ARENA_GOAL_D[0]:
            continue
        d = float(rng.uniform(ARENA_GOAL_D[0], min(ARENA_GOAL_D[1], d_max)))
        g_xy = s_xy + d * d_hat
        if float(tree.query([g_xy[0], g_xy[1], z])[0]) < _GOAL_CLEAR_DRAW:
            continue
        if not _clear_ahead(tree, s_xy, d_hat, z):
            continue
        cells = _cells_crossed(rects, s_xy, g_xy, cell)
        if not (ARENA_CELLS[0] <= len(cells) <= ARENA_CELLS[1]):
            continue
        pair = (s_xy, g_xy, quad, jitter, d, cells)
        break
    if pair is None:
        return None
    s_xy, g_xy, quad, jitter, d, cells = pair

    # into the patch frame: unturn the quadrant, then put the start at the origin
    k = (-quad) % 4
    c, s = _QUAD[k]
    off = (-(c * s_xy[0] - s * s_xy[1]), -(s * s_xy[0] + c * s_xy[1]))
    prims = _place(scene.primitives, k, off[0], off[1])
    in_patch = [(nm, fam, _place_rect(r, k, off[0], off[1])) for nm, fam, r in rects]
    bounds = [round(lo + off[0], 4), round(lo + off[1], 4), round(-lo + off[0], 4), round(-lo + off[1], 4)]
    goal = [c * g_xy[0] - s * g_xy[1] + off[0], s * g_xy[0] + c * g_xy[1] + off[1], z]
    v_goal = 0.0 if rng.random() < 0.7 else 2.0
    params = {
        "size": round(size, 4), "cell_target": round(cell_target, 4), "n_cells": n,
        "cell": round(cell, 4), "map_seed": int(map_seed), "held_out": list(ARENA_EXCLUDE),
        "cell_families": {nm: scene.families[nm] for nm in scene.regions},
        "route_cells": [nm for nm, _ in cells], "n_route_cells": len(cells),
        "route_families": [fam for _, fam in cells],
        "goal_dist": round(d, 4), "bearing_deg": round(jitter, 4), "quarter_turns": int(k),
        "start_map": [round(float(v), 4) for v in s_xy], "goal_map": [round(float(v), 4) for v in g_xy],
    }
    spec = ds.PatchSpec(test=TEST, family="arena", seed=int(seed), split=split, bounds=bounds,
                        flight_z=z, params=params, face_x=0.0, far_x=0.0,
                        goal_pos=[round(float(v), 4) for v in goal], goal_speed=float(v_goal),
                        goal_radius=ds.GOAL_RADIUS, primitives=prims)
    return spec, in_patch


# --- validation and the two entry points ------------------------------------------

def patch_grid(spec: ds.PatchSpec):
    """(points, region, grid) exactly as the dataset builder makes them."""
    pts, reg = ds.patch_cloud(spec)
    return pts, reg, ds.PatchGrid(pts, reg, spec)


def band_route(grid: ds.PatchGrid, a, b) -> bool:
    """Is there a route from ``a`` to ``b`` *inside the flight band*?

    ``PatchGrid.T`` marches over ``free``, which spans the grid's whole z range
    (0.3-3.5 m), so a patch reads as reachable when the only route is a dive
    under the band the states are drawn in and the mission stays in. Measured
    on the wave-1 builder, seeds 0-59: 55 of 60 corridors reachable under the
    band against 60 of 60 under ``free``, every failure a bars+lintel leg.

    6-connected components of ``grid.free_band``, which is the connectivity
    ``skfmm.travel_time`` propagates over -- one label pass on the array the
    grid already holds, no second march.
    """
    lbl, _ = ndi.label(grid.free_band, structure=ndi.generate_binary_structure(3, 1))
    ia, ib = tuple(grid.ijk(a)), tuple(grid.ijk(b))
    return bool(lbl[ia] != 0 and lbl[ia] == lbl[ib])


def _check(spec: ds.PatchSpec):
    """(reason, grid): ``''`` when the patch is usable. A patch whose goal is
    unreachable is a bug in the builder, not a draw to hand downstream."""
    _, _, grid = patch_grid(spec)
    start = np.array([0.0, 0.0, spec.flight_z])
    if grid.clear_at(start) < START_CLEAR:
        return "start_blocked", grid
    if grid.clear_at(spec.goal_pos) < GOAL_CLEAR:
        return "goal_clearance", grid
    if not np.isfinite(grid.T[tuple(grid.ijk(start))]):
        return "unreachable", grid
    if not band_route(grid, start, spec.goal_pos):
        return "unreachable_band", grid
    return "", grid


def corridor_patch(seed: int, split: str = "train", *, validate: bool = True) -> ds.PatchSpec:
    """Tier B: a 15-40 m corridor of 2-4 obstacles 4-10 m apart.

    Built on the primitives whatever ``SUPERFLY_SCENE_CLOUDS`` says
    (``draw_states.primitive_clouds``): every attempt is validated against
    ``patch_cloud``, and a scene course's cloud -- clutter included -- rejected
    attempts the primitives pass, so the corridor came out different from the
    one the course was built from (batch 1: corridors 3002 and 3005)."""
    with ds.primitive_clouds():
        return _corridor_patch(seed, split, validate=validate)


def _corridor_patch(seed: int, split: str, *, validate: bool) -> ds.PatchSpec:
    rejected = []
    for attempt in range(_MAX_ATTEMPTS):
        spec = _draw_corridor(_rng("t5-corridor", seed, attempt), seed, split)
        if validate:
            why, _ = _check(spec)
            if why:
                rejected.append(why)
                continue
        spec.params.update(attempts=attempt + 1, rejected=rejected, validated=bool(validate))
        return spec
    raise PatchUnbuildable(f"corridor_patch({seed}, {split!r})", rejected)


def arena_patch(seed: int, split: str = "train", *, validate: bool = True) -> ds.PatchSpec:
    """Tier C: a random 30-45 m map with a real start/goal pair on it.

    Built on the primitives whatever ``SUPERFLY_SCENE_CLOUDS`` says
    (``draw_states.primitive_clouds``): the start/goal pair is drawn on a
    *map-frame* probe of the patch id, and the finished course's cloud is in
    the *patch* frame -- shifted and turned -- so with the hook live the pair
    search ran against misplaced geometry and every batch-1 arena (3000-3005)
    came out a different map from the one its course was built from."""
    with ds.primitive_clouds():
        return _arena_patch(seed, split, validate=validate)


def _arena_patch(seed: int, split: str, *, validate: bool) -> ds.PatchSpec:
    rejected = []
    for attempt in range(_MAX_ATTEMPTS):
        drawn = _draw_arena(_rng("t5-arena", seed, attempt), seed, split, attempt)
        if drawn is None:
            rejected.append("no_pair")
            continue
        spec, cells = drawn
        why, grid = _check(spec)                    # the grid is needed for the faces either way
        if why and validate:
            rejected.append(why)
            continue
        records = _chord_obstructions(grid, [0.0, 0.0, spec.flight_z], spec.goal_pos, cells)
        if validate and records[0]["kind"] != "clear" and records[0]["face_x"] < _FACE_MIN:
            rejected.append("face_too_near")        # approach band behind the origin
            continue
        spec.face_x, spec.far_x = records[0]["face_x"], records[0]["far_x"]
        spec.params.update(**_face_keys(records), n_obstructions=len(records),
                           first_obstruction=records[0]["kind"], half_width=records[0]["half_width"],
                           y_c=records[0]["y_c"], attempts=attempt + 1, rejected=rejected,
                           validated=bool(validate))
        return spec
    raise PatchUnbuildable(f"arena_patch({seed}, {split!r})", rejected)


# --- scene lists: patch ids -------------------------------------------------------
# A scene run (a Blender course set) is a list of patch ids ``<family>_<seed>``
# rather than a family x contiguous seed range: kit scenes have sparse seeds and
# mix tiers. The id names its tier through its family.

#: tiers B and C are one family each (``corridor_patch``, ``arena_patch``)
TIER_FAMILY = {"B": "corridor", "C": "arena"}


def split_pid(pid: str) -> tuple[str, int]:
    """``"wall_slot_3022"`` -> ``("wall_slot", 3022)``. Families contain
    underscores, seeds do not."""
    fam, sep, seed = str(pid).strip().rpartition("_")
    if not sep or not fam or not seed.isdigit():
        raise ValueError(f"patch id {pid!r} is not <family>_<seed>")
    return fam, int(seed)


def tier_of(family: str) -> str:
    """The test-5 tier a family belongs to: ``corridor`` B, ``arena`` C, the
    tier-A families (``draw_states.FAMILIES_T5``) A."""
    for tier, fam in TIER_FAMILY.items():
        if family == fam:
            return tier
    if family in ds.FAMILIES_T5 or family in getattr(ds, "tier_a_known", lambda: ())():
        return "A"                  # + the opt-in tier-A families, named explicitly (round-2 v9 hosts)
    raise ValueError(f"unknown test-5 family {family!r} (tier A: {', '.join(ds.FAMILIES_T5)}; "
                     f"tier B: corridor; tier C: arena)")


def read_pids(arg: str) -> list[str]:
    """``--pids``: ``a,b,c``, or ``@file`` with ids separated by commas,
    whitespace or newlines (``#`` starts a comment). Order kept; a repeated or
    malformed id, or an unknown family, is refused rather than skipped."""
    text = str(arg)
    if text.startswith("@"):
        text = "\n".join(line.split("#", 1)[0] for line in Path(text[1:]).expanduser().read_text().splitlines())
    pids = [t for t in text.replace(",", " ").split() if t]
    if not pids:
        raise ValueError(f"--pids {arg!r} names no patch")
    seen, dup = set(), []
    for p in pids:
        tier_of(split_pid(p)[0])
        if p in seen:
            dup.append(p)
        seen.add(p)
    if dup:
        raise ValueError(f"--pids names {sorted(set(dup))} more than once")
    return pids


def pid_rows(pids: list[str]) -> list[tuple[str, str, int, str]]:
    """``(pid, family, seed, tier)`` per id."""
    out = []
    for p in pids:
        fam, seed = split_pid(p)
        out.append((p, fam, seed, tier_of(fam)))
    return out
