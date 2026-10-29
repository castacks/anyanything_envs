"""Patches, goals, states and attitudes for tests 3 and 4
(deliverables/2026-09-07/test_plan_tests_3_4.md, sections 2-3).

Patch frame: x forward, z up, floor at z = 0, flight altitude 1.75 m, the
nominal state at the origin flying +x. A patch is one family drawn from a
seed; ``build_patch`` returns the primitives, the goal tuple and the family
parameters, ``patch_cloud`` samples the surfaces the labeller reads, and
``draw_states`` fills the pair sampler's quota grid (geodesic distance x
speed x heading offset x clearance, distance bins extended to 26 m) with
states toward the patch's goal. ``eval_starts`` draws the three start types
of an evaluation patch.

Attitude synthesis (the labeller is a point mass, the student's input has a
rotation matrix and body rates): yaw = velocity heading, drawn within
+-30 deg of +x at rest; pitch and roll from a drawn acceleration in the
state's speed band through the flatness map (thrust = a + g z, body z along
the thrust, body x the heading projected onto the plane normal to it); body
rates N(0, 0.3 rad/s).

Families:
    test 3   wall_offset   wall across the heading, width {1,2,4} m, tall,
                           face 3-8 m ahead, centre y in +-[0.5, 1.5] m
             wall_sym      the same, centred (y = 0)
             float_box     2.0 x 1.6 x 0.5 m box centred on the heading at
                           flight altitude, 3-8 m ahead (over/under/left/right)
    test 4   stand_sparse / stand_medium / stand_dense
                           Poisson-disc trunks, spacing 3.5 / 2.5 / 2.0 m,
                           radius 0.2-0.5 m, 30 x 12 m starting 3 m ahead
    test 5   the twelve tier-A families of
             deliverables/2026-09-11/test5_primitives_finale.md section 3:
             the three above plus float_bar, wall_slot, window, room,
             bars_lintel, pillar, slab and ring, each swept over a range test 3
             held fixed (``T5_SWEEP``). Tests 3 and 4 are a separate branch and
             are byte-identical to what the measured series was built on.

Faces: the v3 zones are defined against a *list* of obstructions, which a spec
declares as ``params['face_spans'] = [[face_x, far_x, y_centre, half_width],
...]``, near face first -- one entry for a tier-A patch, one per leg for a
tier-B corridor or a tier-C route. A state is attributed to the nearest face
ahead of it, so a corridor's approach states land in front of every obstacle
rather than piling up on the first. With one entry every mask is bit for bit
the single-face one. ``face_x`` is measured off the geometry (``band_faces``),
not taken from the placement argument.

Every tier writes those keys through ``face_params`` and nowhere else, so one
key means one thing in every shard: ``faces`` is the bare near-face x of each
obstruction (floats), ``far_faces`` its far x, ``face_spans`` the four-tuples
above, and ``face_records`` the dicts that additionally name the ``family``
and ``kind`` the QA reads. Tier A used to write ``faces`` as floats while the
compositional tiers wrote dicts under the same key -- a mixed shard then had
both, and anything iterating ``params['faces']`` got whichever the tier felt
like. The types cannot diverge now: the four keys are one function's return
value, projected from one list of records.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
import threading
import warnings
from pathlib import Path
import sys

import numpy as np
import scipy.ndimage as ndi
import skfmm

from . import arena as A
from . import patch as PT

GRAVITY = 9.81
FLIGHT_Z = PT.FLIGHT_Z
FLIGHT_BAND = PT.FLIGHT_BAND
GOAL_RADIUS = PT.GOAL_RADIUS
ROBOT_RADIUS = PT.ROBOT_RADIUS
TALL = A.TALL
CRUISE = 3.0

#: test 5 tier A: the three test-3 families kept (so the series stays
#: comparable) plus nine more, the three stand densities counting as one family
#: of the plan's twelve.
FAMILIES_T5 = ("wall_offset", "wall_sym", "float_box", "float_bar", "wall_slot", "window",
               "room", "bars_lintel", "pillar", "slab", "ring",
               "stand_sparse", "stand_medium", "stand_dense")
#: ``data_make_dataset`` reads ``FAMILIES[test]``, so test 5 belongs in here.
#: Anything that iterates the whole dict now sees the test-5 sweeps as well:
#: ``tests/test_data_pipeline.py::DrawTest`` does, and asserts the test-3
#: ranges (width in {1, 2, 4}, goal past the far face), so it has to be scoped
#: to ``(3, 4)`` -- see the report's handoff.
FAMILIES = {3: ("wall_offset", "wall_sym", "float_box"),
            4: ("stand_sparse", "stand_medium", "stand_dense"),
            5: FAMILIES_T5}
STAND_SPACING = {"stand_sparse": 3.5, "stand_medium": 2.5, "stand_dense": 2.0}
#: ids 1-6 are frozen: they seed every test-3/4 draw (``data_make_dataset``
#: seeds on ``[test, FAMILY_ID[family], seed, 7]``). New families take new ids.
FAMILY_ID = {"wall_offset": 1, "wall_sym": 2, "float_box": 3, "stand_sparse": 4, "stand_medium": 5, "stand_dense": 6,
             "float_bar": 7, "wall_slot": 8, "window": 9, "room": 10, "bars_lintel": 11,
             "pillar": 12, "slab": 13, "ring": 14, "clutter_mixed": 15, "clutter_tilted": 16,
             "low_line": 17, "wall_wide": 18, "bar_eye": 19,
             "twig_wire": 20, "low_post": 21, "reveal": 22}
#: test 6, tier A, **opt-in**: generic clutter (``_patch_clutter``). Not in
#: ``FAMILIES[5]``: that tuple is the fourteen-family yardstick every test-5
#: default, plan and test iterates, and adding to it would silently put clutter
#: into every run that names no families. A run asks for these by name
#: (``--families clutter_mixed``); ``tier_a_known`` is what a family name is
#: validated against.
FAMILIES_CLUTTER = ("clutter_mixed", "clutter_tilted", "low_line")
#: test 6, tier A, **opt-in**, single obstacle (``_patch_t5``): ``wall_wide``
#: (DAgger round 3, 2026-09-30) -- one tall wall 6-24 m wide, yawed up to 20 deg,
#: its nearer end 1.5 m .. w/2 off the start->goal line, the goal behind it; the
#: patch widened to the wall (``WALL_WIDE_MARGIN``). The yardstick ``wall_*``
#: families stop at 12 m and the patch at +-8 m, so no training state saw a wall
#: filling the FOV. Deliberately NOT the superfly Isaac wall suite (held out:
#: widths {6, 10, 16, 24} exactly, square to the line, centred, 0.3 m thick,
#: 8 m tall, near face 4/6 m, goal on the line at 14.3 m): widths continuous,
#: yawed, offset, 0.2-0.6 m thick, 5 m tall, near face 3-10 m, goals from the
#: measured goal cases. By name only (``--families wall_wide``).
#: ``bar_eye`` (2026-09-30, over/under investigation): thin horizontal bars and branch-like cylinders
#: (r 0.04-0.15 m) as the FIRST obstruction toward the goal, near flight height, so approach states sit
#: roughly level with a bar 1.5-6 m ahead. Four shapes (``BAR_EYE_KINDS``): a wide bar (4-16 m, over or
#: under), the same with a fence under it (over only) or a screen over it (under only), a short bar
#: (0.8-3 m, off-centre: lateral genuinely best), and a branch cluster (2-3 yawed/tilted bars + twigs).
#: Training geometry coverage only: widths/heights/tilts continuous and randomised, NOT the fixed
#: eye-level probe bar of the OU-B bench.
#: Round-2 RGB geometry (2026-10-01, ``deliverables/2026-10-01/RGB_DATASET_V1.md`` sections 3-4), opt-in,
#: the geometry halves of the v9 state-draw families (``draw_v9``):
#: ``twig_wire`` (family F): 1-3 cm DIAMETER wires and twigs (r 0.005-0.015 m) -- sagging wires between thin
#: poles, guy wires from a pole top to the ground, a tree with low limbs branching into twigs, a hanging twig/vine
#: curtain. Nothing in training was thinner than 4 cm (bar_eye twigs) and nothing in the superfly Isaac fields is
#: thinner than r 0.05 m (``draw_v9.held_out_report``).
#: ``low_post`` (family L): bollards 0.3-1.3 m, low rails / curbs, tilted ramps and sunk mounds (terrain rises).
#: ``reveal`` (family V): an occluder edge, a corner or a door frame with an obstacle 1-3 m behind it, so the
#: obstacle enters the 87 deg camera late.
FAMILIES_OPTIN = ("wall_wide", "bar_eye", "twig_wire", "low_post", "reveal")
BAR_EYE_KINDS = (("wide", 0.30), ("fence", 0.12), ("screen", 0.12), ("short", 0.22), ("branches", 0.24))
TWIG_WIRE_KINDS = (("wires", 0.30), ("guy", 0.15), ("branches", 0.35), ("curtain", 0.20))
LOW_POST_KINDS = (("bollards", 0.35), ("rail", 0.25), ("ramp", 0.20), ("mound", 0.20))
REVEAL_KINDS = (("occluder", 0.40), ("corner", 0.30), ("doorframe", 0.30))
WALL_WIDE_MARGIN = 5.0


#: the whole-scene course family (``whole_scene.py``): a Wenshan scene's full 50 m core, no primitives of its own,
#: seed = the scene's seq (pid ``open_<seq>``). Its spec is only ever the course's own ``spec.json``.
WHOLE_SCENE_FAMILY = "open"


def tier_a_known() -> tuple:
    """Every tier-A family ``build_patch(5, ...)`` can build: the yardstick
    fourteen plus the opt-in ones (and the whole-scene ``open`` family, whose spec is its course's)."""
    return FAMILIES_T5 + FAMILIES_CLUTTER + FAMILIES_OPTIN + (WHOLE_SCENE_FAMILY,)
EVAL_SEED0 = 1000

# the pair sampler's bins, with the distance axis extended to the test-4 starts
DIST_BINS = {"near": (0.6, 2.5), "mid": (2.5, 6.0), "far": (6.0, 14.0), "xfar": (14.0, 26.0)}
SPEED_BINS = {"v0": 0.0, "v1p5": 1.5, "v3": 3.0}
HEADING_BINS = {"aligned": (0.0, 30.0), "oblique": (30.0, 90.0)}
CLR_BINS = {"corridor": (0.5, 1.0), "open": (1.0, np.inf)}
GRID_H = 0.15

#: Test 5's tier-A sweeps, one source of truth for the builder and its tests
#: (plan section 3; test 3's fixed values are in the comments). Distances are
#: metres, angles degrees, all inclusive uniform ranges.
#: ``face`` is the plan's 2-12 m face sweep, shortened to 2-8 m for the deep
#: families (room, bars+lintel, ring) so the patch does not run past 30 m for
#: nothing, held at 3 m for the stands (test 4's placement, only depth sweeps)
#: and started at 4 m for the slab, whose goal is *short of* the face.
T5_SWEEP = {
    "wall_offset": {"face": (2.0, 12.0), "width": (1.0, 12.0), "offset": (0.5, 3.0)},   # was 3-8 / {1,2,4} / 0.5-1.5
    "wall_sym":    {"face": (2.0, 12.0), "width": (1.0, 12.0)},
    # opt-in (FAMILIES_OPTIN): face = the wall centre's x, near_end = the nearer
    # end's distance from the line (capped at w/2 = centred), yaw in degrees
    "wall_wide":   {"face": (4.0, 11.0), "width": (6.0, 24.0), "near_end": (1.5, 12.0),
                    "yaw": (-20.0, 20.0), "t": (0.2, 0.6), "face_min": 3.0},
    "float_box":   {"face": (2.0, 12.0), "width": (1.0, 3.0), "height": (0.6, 2.0),     # was one 0.5x2.0x1.6 box
                    "depth": (0.3, 0.8), "centre_z": FLIGHT_BAND},
    "float_bar":   {"face": (2.0, 12.0), "width": (4.0, 10.0)},                         # over/under only
    # opt-in (FAMILIES_OPTIN): thin bars; z = the bar axis at the start->goal line
    "bar_eye":     {"face": (3.0, 10.0), "r": (0.04, 0.15), "z": (1.2, 2.4), "wide": (4.0, 16.0),
                    "short": (0.8, 3.0), "tilt_deg": (-12.0, 12.0), "yaw_deg": (-30.0, 30.0)},
    # opt-in round-2 geometry (FAMILIES_OPTIN, draw_v9): face = the first element's x
    "twig_wire":   {"face": (3.0, 9.0), "r_thin": (0.005, 0.015), "pole_r": (0.04, 0.10),
                    "wire_z": (0.9, 2.8), "span": (4.0, 14.0), "sag": (0.05, 0.5)},
    "low_post":    {"face": (3.0, 8.0), "post_r": (0.05, 0.20), "post_h": (0.3, 1.3), "depth": (4.0, 10.0),
                    "rail_z": (0.4, 1.1), "ramp_h": (0.4, 1.5), "mound_h": (0.3, 1.2)},
    "reveal":      {"face": (4.0, 10.0), "edge": (0.6, 1.6), "behind": (1.0, 3.0), "door_w": (1.2, 2.0),
                    "door_h": (2.0, 2.4)},
    "wall_slot":   {"face": (2.0, 12.0), "gap": (0.9, 2.5), "lateral": (-4.0, 4.0)},
    "window":      {"face": (2.0, 12.0), "width": (1.0, 1.8), "height": (0.8, 1.4),
                    "sill": (0.9, 1.6), "lateral": (-2.0, 2.0)},
    "room":        {"face": (2.0, 8.0), "door_w": (1.2, 1.8), "door_y": (-1.2, 1.2)},
    "bars_lintel": {"face": (2.0, 8.0), "z_under": (1.1, 1.8), "n_bars": (2, 4)},
    "pillar":      {"face": (2.0, 12.0), "r": (0.4, 1.0), "lateral": (-1.0, 1.0)},
    "slab":        {"face": (4.0, 12.0), "width": (2.0, 5.0), "lateral": (-1.0, 1.0)},
    "ring":        {"face": (2.0, 8.0), "gap_deg": (50.0, 90.0), "bearing": (0.0, 360.0), "closed_share": 0.25},
    "stand_sparse": {"face": (3.0, 3.0), "depth": (10.0, 30.0)},                        # was depth 30 exactly
    "stand_medium": {"face": (3.0, 3.0), "depth": (10.0, 30.0)},
    "stand_dense":  {"face": (3.0, 3.0), "depth": (10.0, 30.0)},
    # generic clutter (``_patch_clutter``): ``face`` is where the clutter
    # rectangle starts, ``depth`` its length, ``goal_past`` the goal's x past
    # its end (negative = among the clutter). The density knob is per patch.
    "clutter_mixed":  {"face": (3.0, 8.0), "depth": (8.0, 26.0), "spacing": (1.6, 3.6),
                       "goal_past": (-3.0, 4.0), "goal_y": (-1.5, 1.5)},
    "clutter_tilted": {"face": (3.0, 8.0), "depth": (8.0, 26.0), "density": (0.07, 0.20),
                       "goal_past": (-3.0, 4.0), "goal_y": (-1.5, 1.5)},
    # low obstacles on the start->goal line (``PT.low_line``): n_line on the
    # chord, n_off scattered, per patch
    "low_line":       {"face": (3.0, 8.0), "depth": (6.0, 20.0), "n_line": (2, 5), "n_off": (0, 6),
                       "goal_past": (-1.0, 4.0), "goal_y": (-1.5, 1.5)},
}
#: clutter: sites are drawn in |y| <= this, and an object whose surface leaves
#: |y| <= T5_Y_HALF - CLUTTER_EDGE is dropped whole (never cut).
CLUTTER_HALF_Y = 6.5
CLUTTER_EDGE = 0.2
#: the start and goal keep-out: every object's exact surface distance from the
#: vertical column over the start and over the nominal goal, for z in
#: ``CLUTTER_KEEPOUT_Z`` (floor to the top of the grid), is at least this. An
#: object that violates it is dropped whole.
CLUTTER_KEEPOUT = 1.0
CLUTTER_KEEPOUT_Z = (0.0, FLIGHT_BAND[1] + 1.0)
#: whole-patch redraws before a seed is declared unbuildable.
CLUTTER_ATTEMPTS = 8
#: two blocked stretches of the chord closer than this are one obstruction.
CLUTTER_MERGE = 0.5
#: How the two clutter families differ from the Isaac evaluation fields
#: (``superfly/sim/obstacle_field.py``: ``generate`` = "diffphys", scale 5;
#: ``generate_diffaero`` = "diffaero"), which are the held-out test and must not
#: be copied. Measured numbers: ``scripts/clutter_vs_isaac.py``.
CLUTTER_VS_ISAAC = {
    "placement": {"clutter_mixed": "Poisson-disc sites, spacing U(1.6, 3.6) m drawn per patch",
                  "clutter_tilted": "complete spatial randomness, Poisson count, density U(0.07, 0.20) /m^2 per patch",
                  "diffphys": "fixed counts (30 balls, 30 cubes, 10 ground boxes, 30 poles, 2 bars) uniform over 40 x 34.2 m",
                  "diffaero": "fixed 30 (9 spheres, 21 boxes), Gaussian perpendicular scatter std 6-7 m about a 47.9 m start-goal line"},
    "density": {"clutter_mixed": "varies per patch", "clutter_tilted": "varies per patch",
                "diffphys": "fixed", "diffaero": "fixed, peaks on the start-goal line"},
    "low_line": "2-5 low objects on the start->goal chord (lateral U(-0.8, 0.8) m) + 0-6 off it: spheres r U(0.3, 0.8) "
                "centre z U(0.4, 2.6), crates 0.3-1.3 m tall, boulders r U(0.6, 1.4) sunk to z = r U(-0.3, 0.5); "
                "the Isaac fields place nothing deliberately on the line (diffphys uniform, diffaero scattered std 6-7 m)",
    "spheres": {"clutter_mixed": "boulders r U(0.3, 1.6) sunk to z = r U(0.4, 1); floaters r U(0.2, 0.8) at z U(0.9, 3.2)",
                "clutter_tilted": "boulders r U(0.3, 1.2) resting on the floor, 15 % of objects",
                "diffphys": "r U(0.4, 0.6), centre z U(-1, 5)",
                "diffaero": "r in {0.6, 1.0, 1.4, 1.8} resting on the floor, 30 % of objects"},
    "boxes": {"clutter_mixed": "crates 0.5-3.0 m wide, 0.3-3.5 m tall, yaw only; slabs 0.16-0.6 m thick overhead at 1.6-3.0 m",
              "clutter_tilted": "blocks 0.3-2.0 m footprint, 1.2-9.0 m tall, lean 0-35 deg; panels 0.08-0.24 m thick, 1-4 m wide, lean 0-45 deg",
              "diffphys": "floating cubes 0.4-0.6 m, ground boxes 0.2-6.0 m wide, top -0.8..3 m, axis-aligned",
              "diffaero": "0.6-1.2 m footprint (4 values), 10-14 m tall, roll/pitch U(-15, 15) deg each"},
    "cylinders": {"clutter_mixed": "poles r U(0.08, 0.6), 1.3-6.5 m tall; bars r U(0.04, 0.2), 1.5-6 m, any yaw, z 0.5-3 m",
                  "clutter_tilted": "struts r U(0.06, 0.25), 2.5-7 m, leaning 20-60 deg off vertical",
                  "diffphys": "poles r U(0.05, 0.4), 12 m tall; 2 bars r U(0.05, 0.15), 6 m, across the corridor",
                  "diffaero": "none"},
}


class ClutterUnbuildable(RuntimeError):
    """No clutter draw for this seed passed validation in ``CLUTTER_ATTEMPTS``
    tries (``tiers.PatchUnbuildable``'s tier-A counterpart: a worker can catch
    it and write a skip record). ``.reasons`` is the per-attempt list."""

    def __init__(self, what: str, reasons: list):
        super().__init__(f"{what}: {reasons}")
        self.what, self.reasons = what, list(reasons)
#: patch half width for test 5; a wall this wide seals the grid, which is how
#: "full span" families deny going around (``window``, ``wall_slot``).
T5_Y_HALF = 8.0
T5_RING_R = 2.5


@dataclass
class PatchSpec:
    test: int
    family: str
    seed: int
    split: str
    bounds: list                      # x0, y0, x1, y1
    flight_z: float
    params: dict                      # the drawn family parameters
    face_x: float                     # x of the obstruction's near face (start placement)
    far_x: float                      # x of its far face (goal placement)
    goal_pos: list
    goal_speed: float
    goal_radius: float
    primitives: list = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    def as_patch(self) -> PT.Patch:
        """The scenes/patch.py object, for its ``sample`` and ``route``."""
        return PT.Patch(seed=self.seed, regime="", family=self.family, bounds=list(self.bounds),
                        flight_z=self.flight_z, state_pos=[0.0, 0.0, self.flight_z],
                        state_vel=[CRUISE, 0.0, 0.0], goal_pos=list(self.goal_pos),
                        goal_speed=self.goal_speed, goal_radius=self.goal_radius,
                        primitives=self.primitives)


# --- patches --------------------------------------------------------------------

def _wall(prims, d, width, y_c, fam, t=0.3, height=TALL):
    prims.append(A.wall((d, y_c - width / 2), (d, y_c + width / 2), 0.0, height, t, fam))


_GC = None


def _gen_courses():
    """The legacy leg builders (``gen_courses``): ``_float_bar`` lives only there."""
    from . import gen_courses
    return gen_courses


_BOX_CORNERS = np.array([[i, j, k] for i in (-1.0, 1.0) for j in (-1.0, 1.0) for k in (-1.0, 1.0)])


def _prim_bbox(p: dict) -> tuple[float, float, float, float, float, float]:
    """(x0, x1, y0, y1, z0, z1) of one primitive's surface, in the frame
    ``arena.sample_surfaces`` samples it in."""
    k = p["kind"]
    if k == "sphere":
        c, r = p["c"], p["r"]
        return c[0] - r, c[0] + r, c[1] - r, c[1] + r, c[2] - r, c[2] + r
    if k == "box":
        c = np.asarray(p["c"], float)
        q = c + (_BOX_CORNERS * np.asarray(p["half"], float)) @ A._euler_xyz(*p["rpy"]).T
        lo, hi = q.min(0), q.max(0)
        return lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]
    if k == "cyl_v":
        (cx, cy), r = p["c"], p["r"]
        return cx - r, cx + r, cy - r, cy + r, p["z0"], p["z1"]
    if k == "cyl_h":
        a, b, r = p["a"], p["b"], p["r"]
        return tuple(v for i in range(3) for v in (min(a[i], b[i]) - r, max(a[i], b[i]) + r))
    if k == "wall":
        p0, p1, t = np.asarray(p["p0"], float), np.asarray(p["p1"], float), p["t"]
        d = p1 - p0
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d) * t / 2      # the slab's half-thickness
        q = np.array([p0 - n, p0 + n, p1 - n, p1 + n])
        return q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max(), p["z0"], p["z1"]
    raise ValueError(f"unknown primitive kind {k!r}")


def _band_bboxes(prims: list, band) -> list:
    return [e for e in (_prim_bbox(p) for p in prims) if e[5] >= band[0] and e[4] <= band[1]]


def band_faces(prims: list, band=FLIGHT_BAND) -> tuple[float, float] | None:
    """(near, far) x of the surfaces that reach into the altitude band, i.e.
    the first and last thing a state flying the band can hit; ``None`` when
    nothing does.

    ``face_x`` drives the whole v3 approach zone, and the nominal conventions
    the earlier tests carried are the *placement* argument, not the surface:
    wave 1's verifier measured them off by -0.25 to +0.34 m (a ring's near
    pillars stop 0.34 m short of its nominal face because the gap lattice has
    no pillar there; a stand's trunk radius puts real bark 0.20-0.25 m in front
    of its). Test 5 measures the face instead of assuming it.
    """
    ext = _band_bboxes(prims, band)
    if not ext:
        return None
    return float(min(e[0] for e in ext)), float(max(e[1] for e in ext))


def band_span(prims: list, band=FLIGHT_BAND) -> tuple[float, float] | None:
    """(y centre, half-width) of the same surfaces -- the lateral span the v3
    zones budget against, measured rather than declared.

    Declared spans are where the empty-``beside`` bugs come from: a centre of 0
    with an outer extent for a half-width stretches the span across the whole
    corridor. A leg builder that hands this its own primitives cannot get that
    wrong, and a ring whose gap faces sideways gets the asymmetric span it
    really has.
    """
    ext = _band_bboxes(prims, band)
    if not ext:
        return None
    lo, hi = min(e[2] for e in ext), max(e[3] for e in ext)
    return float((lo + hi) / 2), float((hi - lo) / 2)


def _patch_t5(family: str, seed: int, split: str) -> PatchSpec:
    """Test 5, tier A: one obstacle in front of the state, swept (``T5_SWEEP``).

    Geometry is never re-implemented here -- every family calls the builder
    ``scenes/patch.py`` or ``gen_courses`` already ships, so a training patch
    and the evaluation suite draw the same shape. What is new is the sweep, the
    measured ``face_x``/``far_x`` (``band_faces``) and the per-family goal rule:
    a goal *past* the obstruction is the test-3 rule and wrong for the families
    whose point is a goal inside (room, ring), behind (pillar) or short of
    (slab) the geometry. Wave 2's ``goal_cases`` will draw those against this
    geometry; until it lands the rule below keeps every patch usable.
    """
    if family in FAMILIES_CLUTTER:
        return _patch_clutter(family, seed, split)
    rng = np.random.default_rng([5, FAMILY_ID[family], seed])
    S = T5_SWEEP[family]
    z = FLIGHT_Z
    prims: list = []
    goal_rule = "past"
    d = float(rng.uniform(*S["face"]))
    if family in ("wall_offset", "wall_sym"):
        w = float(rng.uniform(*S["width"]))
        y_c = 0.0 if family == "wall_sym" else float(rng.uniform(*S["offset"]) * rng.choice([-1.0, 1.0]))
        t = 0.3
        _wall(prims, d + t / 2, w, y_c, family, t=t)
        params = {"width": w, "d": d, "y_c": y_c, "t": t, "height": TALL}
    elif family == "wall_wide":
        w = float(rng.uniform(*S["width"]))
        near = min(float(rng.uniform(*S["near_end"])), w / 2)
        y_c = float(rng.choice([-1.0, 1.0])) * (w / 2 - near)
        t = float(rng.uniform(*S["t"]))
        # yaw bounded so the wall's nearest corner stays >= face_min ahead of the start
        ymax = math.degrees(math.asin(min(1.0, max(0.0, (d - S["face_min"] - t) / (w / 2)))))
        yaw = float(np.clip(rng.uniform(*S["yaw"]), -ymax, ymax))
        u = np.array([-math.sin(math.radians(yaw)), math.cos(math.radians(yaw))])     # along the wall
        c = np.array([d, y_c])
        p0, p1 = c - u * w / 2, c + u * w / 2
        prims.append(A.wall((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])), 0.0, TALL, t, family))
        params = {"width": w, "d": d, "y_c": y_c, "near_end": near, "yaw_deg": yaw, "t": t, "height": TALL}
    elif family == "bar_eye":
        prims, params = _bar_eye(rng, d, S, family)
    elif family == "twig_wire":
        prims, params = _twig_wire(rng, d, S, family)
    elif family == "low_post":
        prims, params = _low_post(rng, d, S, family)
    elif family == "reveal":
        prims, params = _reveal(rng, d, S, family)
        goal_rule = "reveal"
    elif family == "float_box":
        w, hgt = float(rng.uniform(*S["width"])), float(rng.uniform(*S["height"]))
        dep, z_c = float(rng.uniform(*S["depth"])), float(rng.uniform(*S["centre_z"]))
        prims.append(A.box((d + dep / 2, 0.0, z_c), (dep / 2, w / 2, hgt / 2), (0.0, 0.0, 0.0), family))
        params = {"d": d, "width": w, "height": hgt, "depth": dep, "centre_z": z_c}
    elif family == "float_bar":
        w = float(rng.uniform(*S["width"]))
        prims, _, _ = _gen_courses()._float_bar(rng, d)      # the canonical bar: 0.6 deep, 1.1 tall, at flight height
        prims[0]["half"][1] = w / 2                          # the sweep is its width alone -- around must stay denied
        params = {"d": d, "width": w, "height": 2 * prims[0]["half"][2], "depth": 2 * prims[0]["half"][0]}
    elif family == "wall_slot":
        gap, y = float(rng.uniform(*S["gap"])), float(rng.uniform(*S["lateral"]))
        PT.wall_slot(prims, d, width=gap, y=y, fam=family, half=T5_Y_HALF)
        params = {"d": d, "gap": gap, "slot_y": y, "half_span": T5_Y_HALF, "t": 0.3}
    elif family == "window":
        w, hgt = float(rng.uniform(*S["width"])), float(rng.uniform(*S["height"]))
        sill, y = float(rng.uniform(*S["sill"])), float(rng.uniform(*S["lateral"]))
        PT.window(prims, d, width=w, height=hgt, sill=sill, y=y, fam=family)
        params = {"d": d, "width": w, "height": hgt, "sill": sill, "window_y": y, "t": 0.25}
    elif family == "room":
        door_w, door_y = float(rng.uniform(*S["door_w"])), float(rng.uniform(*S["door_y"]))
        PT.room(prims, d, w=6.0, depth=6.0, door_w=door_w, door_h=2.2, door_y=door_y, fam=family)
        params = {"d": d, "door_w": door_w, "door_y": door_y, "w": 6.0, "depth": 6.0}
        goal_rule = "inside"
    elif family == "bars_lintel":
        z_under = float(rng.uniform(*S["z_under"]))
        n_bars = int(rng.integers(S["n_bars"][0], S["n_bars"][1] + 1))
        PT.lintel(prims, d, z_under=z_under, fam=family)
        PT.bars(prims, rng, d + 3.0, n=n_bars, depth=6.0, fam=family)
        params = {"d": d, "z_under": z_under, "n_bars": n_bars, "bar_depth": 6.0}
    elif family == "pillar":
        r, y = float(rng.uniform(*S["r"])), float(rng.uniform(*S["lateral"]))
        PT.pillar(prims, d, r=r, y=y, fam=family)
        params = {"d": d, "r": r, "y_c": y}
        goal_rule = "behind"
    elif family == "slab":
        w, y = float(rng.uniform(*S["width"])), float(rng.uniform(*S["lateral"]))
        PT.slab(prims, d, half=(0.3, w / 2, 2.0), y=y, fam=family)
        params = {"d": d, "width": w, "y_c": y, "depth": 0.6}
        goal_rule = "short"
    elif family == "ring":
        closed = bool(rng.random() < S["closed_share"])
        gap_deg = 0.0 if closed else float(rng.uniform(*S["gap_deg"]))
        bearing = float(rng.uniform(*S["bearing"]))
        PT.ring(prims, (d + T5_RING_R, 0.0), r=T5_RING_R, gap_deg=gap_deg, gap_at_deg=bearing, fam=family)
        params = {"d": d, "r": T5_RING_R, "gap_deg": gap_deg, "gap_at_deg": bearing, "closed": closed,
                  "centre": [d + T5_RING_R, 0.0]}
        goal_rule = "past" if closed else "inside"          # a closed ring has no route in: go round it
    elif family in STAND_SPACING:
        s = STAND_SPACING[family]
        depth, half = float(rng.uniform(*S["depth"])), 6.0
        for cx, cy in A._poisson_disc(rng, d, -half, d + depth, half, s, margin=0.3):
            prims.append(A.cyl_v(cx, cy, float(rng.uniform(0.2, 0.5)), 0.0, TALL, family))
        params = {"spacing": s, "d": d, "depth": depth, "half_width": half, "n_trees": len(prims)}
    else:
        raise ValueError(f"unknown family {family!r}")

    # both the face and the lateral span are measured off the primitives that
    # reach the flight band, never off the placement arguments
    face_x, far_x = band_faces(prims) or (d, d)
    span = band_span(prims) or (0.0, 1.0)
    if goal_rule == "inside":
        c = params["centre"] if family == "ring" else [d + 3.0, params["door_y"]]
        goal = [c[0], float(c[1] + rng.uniform(-0.4, 0.4)), z]
        v_goal = 0.0
    elif goal_rule == "behind":
        # far enough behind that the goal cell is free at 0.15 m voxels (0.5 m
        # put it at 0.45 m of clearance), close enough that the pillar hides it
        goal = [far_x + float(rng.uniform(0.8, 1.8)), params["y_c"], z]
        v_goal = 0.0
    elif goal_rule == "reveal":                             # past the hidden obstacle, on the side the kind opens to
        gx, gy = params["goal_xy"]
        goal = [float(gx), float(gy), z]
        v_goal = 0.0 if rng.random() < 0.6 else 2.0
    elif goal_rule == "short":                              # stop 1-2 m before the face (G3/G4)
        goal = [face_x - float(rng.uniform(1.0, 2.0)), params["y_c"] + float(rng.uniform(-0.5, 0.5)), z]
        v_goal = 0.0
    else:
        goal = [far_x + float(rng.uniform(4.0, 5.0)), float(rng.uniform(-1.0, 1.0)), z]
        v_goal = (CRUISE if rng.random() < 0.7 else 0.0) if family in STAND_SPACING \
            else (0.0 if rng.random() < 0.7 else 2.0)
    # the one obstruction, written through the same projection the
    # compositional tiers use, so ``faces`` is a list of floats in every tier
    params.update(goal_rule=goal_rule, span=[float(span[0]), float(span[1])],
                  **face_params([{"face_x": float(face_x), "far_x": float(far_x),
                                  "y_c": float(span[0]), "half_width": float(span[1]),
                                  "family": family, "kind": "obstacle"}]))
    y_half = T5_Y_HALF
    if family == "wall_wide":                                 # the patch spans the wall + a margin
        y_half = float(max(T5_Y_HALF, abs(span[0]) + span[1] + WALL_WIDE_MARGIN))
    bounds = [-8.0, -y_half, max(goal[0] + 8.0, far_x + 6.0, 20.0), y_half]
    return PatchSpec(test=5, family=family, seed=seed, split=split, bounds=bounds, flight_z=z,
                     params=params, face_x=float(face_x), far_x=float(far_x), goal_pos=goal,
                     goal_speed=float(v_goal), goal_radius=GOAL_RADIUS, primitives=prims)


def _bar_eye(rng, d: float, S: dict, family: str) -> tuple[list, dict]:
    """``bar_eye``'s primitives: one of ``BAR_EYE_KINDS`` with its first bar's near surface at x ~ ``d``."""
    kinds, w8 = zip(*BAR_EYE_KINDS)
    kind = str(rng.choice(kinds, p=np.asarray(w8) / sum(w8)))
    prims: list = []

    def bar(x, yc, z, w, r, tilt_deg=0.0, yaw_deg=0.0):
        u = np.array([-math.sin(math.radians(yaw_deg)), math.cos(math.radians(yaw_deg)), 0.0])
        u[2] = math.tan(math.radians(tilt_deg))
        c = np.array([x, yc, z])
        a, b = c - u * w / 2, c + u * w / 2
        a[2], b[2] = max(a[2], r + 0.05), max(b[2], r + 0.05)           # never through the ground
        prims.append(A.cyl_h(a, b, r, family))
        return a, b

    def twigs(a, b, n, r_main):
        for _ in range(n):
            t = float(rng.uniform(0.1, 0.9)); p = a + t * (b - a)
            ln = float(rng.uniform(0.4, 1.5)); th = float(rng.uniform(0, 2 * math.pi)); el = float(rng.uniform(-0.6, 0.9))
            q = p + ln * np.array([math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el)])
            q[2] = max(q[2], 0.1)
            prims.append(A.cyl_h(p, q, float(rng.uniform(0.02, max(0.03, 0.6 * r_main))), family))

    r = float(rng.uniform(*S["r"])); z = float(rng.uniform(*S["z"]))
    params = {"kind": kind, "d": d, "r": r, "z": z}
    x = d + r
    if kind in ("wide", "fence", "screen"):
        w = float(rng.uniform(*S["wide"])); yc = float(rng.uniform(-0.25, 0.25) * w)
        tilt = float(rng.uniform(*S["tilt_deg"])) if rng.random() < 0.3 else 0.0
        a, b = bar(x, yc, z, w, r, tilt_deg=tilt, yaw_deg=float(rng.uniform(-10.0, 10.0)))
        if rng.random() < 0.3:
            twigs(a, b, int(rng.integers(2, 6)), r)
        t = 0.06
        if kind == "fence":                  # a panel from the ground to the bar: under is shut, over is not
            prims.append(A.box((x, yc, (z - r) / 2), (t / 2, w / 2, (z - r) / 2), (0.0, 0.0, 0.0), family))
        elif kind == "screen":               # a panel from the bar to 5 m: over is shut, under is not
            top = TALL
            prims.append(A.box((x, yc, (z + r + top) / 2), (t / 2, w / 2, (top - z - r) / 2), (0.0, 0.0, 0.0), family))
        params.update(width=w, y_c=yc, tilt_deg=tilt)
    elif kind == "short":
        w = float(rng.uniform(*S["short"])); yc = float(rng.uniform(-1.2, 1.2))
        a, b = bar(x, yc, z, w, r, yaw_deg=float(rng.uniform(-25.0, 25.0)))
        if rng.random() < 0.3:
            twigs(a, b, int(rng.integers(1, 4)), r)
        params.update(width=w, y_c=yc)
    else:                                    # branches: 2-3 bars, the first nearest, yawed and tilted
        n = int(rng.integers(2, 4)); xs = [x] + sorted(x + float(rng.uniform(1.5, 4.5)) for _ in range(n - 1))
        for i, xi in enumerate(xs):
            ri = r if i == 0 else float(rng.uniform(*S["r"]))
            zi = z if i == 0 else float(rng.uniform(0.9, 2.8))
            wi = float(rng.uniform(3.0, 14.0))
            a, b = bar(xi, float(rng.uniform(-2.0, 2.0)), zi, wi, ri, tilt_deg=float(rng.uniform(*S["tilt_deg"])),
                       yaw_deg=float(rng.uniform(*S["yaw_deg"])))
            if rng.random() < 0.5:
                twigs(a, b, int(rng.integers(1, 5)), ri)
        params.update(n_bars=n, width=None, y_c=0.0)
    return prims, params


def _seg_chain(prims, pts, r, family):
    """``cyl_h`` segments along a polyline (any direction; ``cyl_h`` is a general segment)."""
    for a, b in zip(pts[:-1], pts[1:]):
        if float(np.linalg.norm(np.asarray(b) - np.asarray(a))) > 1e-3:
            prims.append(A.cyl_h(np.asarray(a, float), np.asarray(b, float), float(r), family))


def _sag_wire(a, b, sag, n=8):
    """A wire from ``a`` to ``b`` hanging ``sag`` metres at mid-span (a parabola, n segments)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.linspace(0.0, 1.0, n + 1)
    p = a[None] + t[:, None] * (b - a)[None]
    p[:, 2] -= 4.0 * sag * t * (1.0 - t)
    return p


def _branch(rng, prims, p0, direction, length, r, depth, family, z_min=0.15):
    """A limb from ``p0`` along ``direction`` that forks into thinner limbs ``depth`` times; the last level is
    twigs of r 0.005-0.012 m. Bent: two segments with a kink."""
    d = np.asarray(direction, float); d /= max(np.linalg.norm(d), 1e-9)
    kink = d + rng.normal(0.0, 0.25, 3); kink /= np.linalg.norm(kink)
    mid = p0 + d * length * 0.5
    end = mid + kink * length * 0.5
    mid[2] = max(mid[2], z_min); end[2] = max(end[2], z_min)
    _seg_chain(prims, [p0, mid, end], r, family)
    if depth <= 0:
        return
    for _ in range(int(rng.integers(2, 4))):
        base = p0 + (end - p0) * float(rng.uniform(0.35, 1.0))
        nd = kink + rng.normal(0.0, 0.7, 3); nd[2] += float(rng.uniform(-0.2, 0.4))
        r_child = max(0.005, r * float(rng.uniform(0.45, 0.7))) if depth > 1 else float(rng.uniform(0.005, 0.012))
        _branch(rng, prims, base, nd, length * float(rng.uniform(0.4, 0.65)), r_child, depth - 1, family, z_min)


def _twig_wire(rng, d: float, S: dict, family: str) -> tuple[list, dict]:
    """``twig_wire``: 1-3 cm (diameter) wires and twigs in front of the state, one of ``TWIG_WIRE_KINDS``.

    Every thin element is r U(0.005, 0.015) m; the carriers (poles, a trunk, first limbs) are thicker, as
    real ones are. Continuous random placement, yaw and sag -- deliberately unlike the superfly Isaac fields
    (full-height poles r >= 0.05, two straight bars r >= 0.05 across the corridor)."""
    kinds, w8 = zip(*TWIG_WIRE_KINDS)
    kind = str(rng.choice(kinds, p=np.asarray(w8) / sum(w8)))
    prims: list = []
    rt = lambda: float(rng.uniform(*S["r_thin"]))
    params = {"kind": kind, "d": d}
    yaw = math.radians(float(rng.uniform(-30.0, 30.0)))
    across = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    if kind == "wires":
        n_sets = 1 + int(rng.random() < 0.4)
        n_wires = []
        for k in range(n_sets):
            x = d + (0.0 if k == 0 else float(rng.uniform(2.0, 5.0)))
            span = float(rng.uniform(*S["span"])); yc = float(rng.uniform(-0.3, 0.3) * span)
            ph = float(rng.uniform(2.6, 4.5)); pr = float(rng.uniform(*S["pole_r"]))
            c = np.array([x, yc, 0.0])
            a2, b2 = c - across * span / 2, c + across * span / 2
            for q in (a2, b2):
                prims.append(A.cyl_v(float(q[0]), float(q[1]), pr, 0.0, ph, family))
            n = int(rng.integers(1, 5)); n_wires.append(n)
            for _ in range(n):
                zw = float(rng.uniform(S["wire_z"][0], min(S["wire_z"][1], ph - 0.1)))
                sag = float(rng.uniform(*S["sag"]))
                zw = max(zw, sag + 0.5)
                _seg_chain(prims, _sag_wire(a2 + [0, 0, zw], b2 + [0, 0, zw], sag), rt(), family)
        params.update(n_sets=n_sets, n_wires=n_wires)
    elif kind == "guy":
        y0 = float(rng.uniform(-2.5, 2.5)); ph = float(rng.uniform(3.0, 6.0)); pr = float(rng.uniform(0.06, 0.12))
        top = np.array([d + 1.0, y0, ph])
        prims.append(A.cyl_v(d + 1.0, y0, pr, 0.0, ph, family))
        n = int(rng.integers(2, 4)); th0 = float(rng.uniform(0, 2 * math.pi))
        for i in range(n):
            th = th0 + 2 * math.pi * i / n + float(rng.uniform(-0.4, 0.4)); rr = float(rng.uniform(2.0, 5.0))
            anchor_ = np.array([top[0] + rr * math.cos(th), top[1] + rr * math.sin(th), 0.05])
            _seg_chain(prims, np.linspace(top - [0, 0, 0.2], anchor_, 4), rt(), family)
        params.update(n_guys=n, pole_h=ph, y_c=y0)
    elif kind == "branches":
        side = float(rng.choice([-1.0, 1.0])); off = float(rng.uniform(0.8, 3.0))
        tx, ty = d + float(rng.uniform(0.5, 2.0)), side * off
        tr = float(rng.uniform(0.08, 0.25)); th = float(rng.uniform(3.0, 6.0))
        prims.append(A.cyl_v(tx, ty, tr, 0.0, th, family))
        n_limbs = int(rng.integers(2, 6))
        for _ in range(n_limbs):
            z0 = float(rng.uniform(1.0, min(2.8, th - 0.3)))
            ang = math.atan2(-ty, -0.5) + float(rng.uniform(-1.6, 1.6))       # mostly toward the line
            el = math.radians(float(rng.uniform(-15.0, 30.0)))
            dvec = np.array([math.cos(ang) * math.cos(el), math.sin(ang) * math.cos(el), math.sin(el)])
            _branch(rng, prims, np.array([tx, ty, z0]) + dvec * tr, dvec, float(rng.uniform(1.5, 4.0)),
                    float(rng.uniform(0.02, 0.05)), 2, family)
        params.update(n_limbs=n_limbs, trunk_r=tr, trunk_y=ty)
    else:                                       # curtain: a high support with twigs / vines hanging from it
        span = float(rng.uniform(3.0, 8.0)); yc = float(rng.uniform(-1.0, 1.0)); zs = float(rng.uniform(2.4, 3.2))
        c = np.array([d, yc, zs]); a2, b2 = c - across * span / 2, c + across * span / 2
        _seg_chain(prims, _sag_wire(a2, b2, float(rng.uniform(0.0, 0.25))), float(rng.uniform(0.01, 0.02)), family)
        for q in (a2, b2):
            prims.append(A.cyl_v(float(q[0]), float(q[1]), float(rng.uniform(0.05, 0.1)), 0.0, zs + 0.1, family))
        n, t = 0, float(rng.uniform(0.1, 0.6))
        while t < span - 0.1:
            top = a2 + across * t
            ln = float(rng.uniform(0.6, min(2.0, zs - 0.5)))
            lean = rng.normal(0.0, 0.12, 3); lean[2] = 0.0
            mid = top + lean - np.array([0, 0, ln * 0.5]); bot = mid + lean - np.array([0, 0, ln * 0.5])
            _seg_chain(prims, [top, mid, bot], rt(), family)
            n += 1; t += float(rng.uniform(0.3, 0.9))
        params.update(n_hanging=n, support_z=zs, span=span, y_c=yc)
    return prims, params


def _low_post(rng, d: float, S: dict, family: str) -> tuple[list, dict]:
    """``low_post``: low obstacles and terrain rises in front of the state (one of ``LOW_POST_KINDS``), all
    topping out at 0.2-1.5 m, so the cue is low in the image and the label flies over or around low."""
    kinds, w8 = zip(*LOW_POST_KINDS)
    kind = str(rng.choice(kinds, p=np.asarray(w8) / sum(w8)))
    prims: list = []
    depth = float(rng.uniform(*S["depth"]))
    params = {"kind": kind, "d": d, "depth": depth}
    if kind == "bollards":
        n = int(rng.integers(3, 11))
        for _ in range(n):
            x = d + float(rng.uniform(0.0, depth)); y = float(rng.uniform(-2.5, 2.5))
            prims.append(A.cyl_v(x, y, float(rng.uniform(*S["post_r"])), 0.0, float(rng.uniform(*S["post_h"])), family))
        params.update(n_posts=n)
    elif kind == "rail":
        n_posts = int(rng.integers(2, 6)); yaw = math.radians(float(rng.uniform(-35.0, 35.0)))
        u = np.array([-math.sin(yaw), math.cos(yaw), 0.0]); c = np.array([d, float(rng.uniform(-1.5, 1.5)), 0.0])
        span = float(rng.uniform(3.0, 9.0)); zr = float(rng.uniform(*S["rail_z"]))
        ends = [c + u * (span * (i / (n_posts - 1) - 0.5)) for i in range(n_posts)]
        for q in ends:
            prims.append(A.cyl_v(float(q[0]), float(q[1]), float(rng.uniform(0.03, 0.08)), 0.0, zr + 0.05, family))
        _seg_chain(prims, [q + [0, 0, zr] for q in ends], float(rng.uniform(0.02, 0.05)), family)
        if rng.random() < 0.5:                  # a low curb / wall a few metres on
            h = float(rng.uniform(0.2, 0.9)); x2 = d + float(rng.uniform(2.0, depth))
            w = float(rng.uniform(2.0, 8.0)); t = float(rng.uniform(0.15, 0.4)); yc = float(rng.uniform(-2.0, 2.0))
            prims.append(A.box((x2, yc, h / 2), (t / 2, w / 2, h / 2), (0.0, 0.0, float(rng.uniform(-0.5, 0.5))), family))
            params.update(curb_h=h)
        params.update(rail_z=zr, n_rail_posts=n_posts)
    elif kind == "ramp":                        # a tilted slab rising from the floor: the ground comes up
        h = float(rng.uniform(*S["ramp_h"])); L = float(rng.uniform(3.0, 8.0)); w = float(rng.uniform(4.0, 10.0))
        th = math.atan2(h, L); t = 0.3
        hl = math.hypot(L, h) / 2
        cx = d + L / 2; cz = h / 2 - t / 2 * math.cos(th)
        prims.append(A.box((cx, float(rng.uniform(-1.5, 1.5)), cz), (hl, w / 2, t / 2), (0.0, -th, 0.0), family))
        if rng.random() < 0.6:                  # posts on or past the top
            for _ in range(int(rng.integers(1, 5))):
                x = d + L + float(rng.uniform(-1.0, 3.0)); y = float(rng.uniform(-2.0, 2.0))
                base = h if x <= d + L else 0.0
                prims.append(A.cyl_v(x, y, float(rng.uniform(*S["post_r"])), 0.0, base + float(rng.uniform(*S["post_h"])), family))
        params.update(ramp_h=h, ramp_len=L, ramp_w=w)
    else:                                       # mound: sunk spheres, tops 0.3-1.2 m
        n = int(rng.integers(1, 4))
        for _ in range(n):
            r = float(rng.uniform(1.5, 4.0)); top = float(rng.uniform(*S["mound_h"]))
            prims.append(A.sphere((d + float(rng.uniform(0.0, depth)) + r * 0.3, float(rng.uniform(-3.0, 3.0)), top - r),
                                  r, family))
        if rng.random() < 0.5:
            for _ in range(int(rng.integers(1, 4))):
                prims.append(A.cyl_v(d + float(rng.uniform(0.0, depth)), float(rng.uniform(-2.0, 2.0)),
                                     float(rng.uniform(*S["post_r"])), 0.0, float(rng.uniform(*S["post_h"])), family))
        params.update(n_mounds=n)
    return prims, params


def _hidden_obstacle(rng, prims, x, y, family) -> str:
    k = str(rng.choice(["pillar", "crate", "bar"], p=[0.45, 0.35, 0.20]))
    if k == "pillar":
        prims.append(A.cyl_v(x, y, float(rng.uniform(0.15, 0.5)), 0.0, TALL, family))
    elif k == "crate":
        h = float(rng.uniform(1.0, 2.6))
        prims.append(A.box((x, y, h / 2), (float(rng.uniform(0.3, 0.8)), float(rng.uniform(0.3, 0.8)), h / 2),
                           (0.0, 0.0, float(rng.uniform(0.0, math.pi))), family))
    else:
        z = float(rng.uniform(1.2, 2.3)); w = float(rng.uniform(1.5, 4.0))
        prims.append(A.cyl_h((x, y - w / 2, z), (x, y + w / 2, z), float(rng.uniform(0.05, 0.15)), family))
    return k


def _reveal(rng, d: float, S: dict, family: str) -> tuple[list, dict]:
    """``reveal``: geometry that hides an obstacle until it is 2-6 m away (one of ``REVEAL_KINDS``). Returns
    ``params['goal_xy']``, the goal past the hidden obstacle on the side the kind opens to."""
    kinds, w8 = zip(*REVEAL_KINDS)
    kind = str(rng.choice(kinds, p=np.asarray(w8) / sum(w8)))
    prims: list = []
    side = float(rng.choice([-1.0, 1.0]))
    t = float(rng.uniform(0.2, 0.4))
    behind = float(rng.uniform(*S["behind"]))
    params = {"kind": kind, "d": d, "side": side, "t": t}
    if kind == "occluder":                      # a wall from the side to an edge near the line; obstacle behind it
        edge = side * float(rng.uniform(*S["edge"]))
        far = side * float(rng.uniform(4.0, 7.0))
        _wall(prims, d + t / 2, abs(far - edge), (far + edge) / 2, family, t=t)
        oy = edge + side * float(rng.uniform(-0.6, 0.8))
        ok = _hidden_obstacle(rng, prims, d + t + behind, oy, family)
        gxy = (d + t + behind + float(rng.uniform(3.0, 6.0)), float(rng.uniform(-1.5, 1.5)) - side * 0.5)
        params.update(edge_y=edge, obstacle=ok, obstacle_xy=[d + t + behind, oy])
    elif kind == "corner":                      # fly along a wall, turn round its end: the obstacle is just round it
        wy = side * float(rng.uniform(1.2, 2.5))
        prims.append(A.wall((-2.0, wy), (d, wy), 0.0, TALL, t, family))
        prims.append(A.wall((d, wy), (d, wy + side * float(rng.uniform(3.5, 5.5))), 0.0, TALL, t, family))
        ox, oy = d + float(rng.uniform(0.8, 2.5)), wy + side * float(rng.uniform(0.5, 2.5))
        ok = _hidden_obstacle(rng, prims, ox, oy, family)
        gxy = (d + float(rng.uniform(2.0, 4.0)), wy + side * float(rng.uniform(3.0, 5.0)))
        params.update(wall_y=wy, obstacle=ok, obstacle_xy=[ox, oy])
    else:                                       # a door frame in a full-span wall; obstacle just past the door
        dw = float(rng.uniform(*S["door_w"])); dh = float(rng.uniform(*S["door_h"])); dy = float(rng.uniform(-1.5, 1.5))
        prims.append(A.wall((d, -T5_Y_HALF), (d, T5_Y_HALF), 0.0, 3.2, t, family,
                            slots=[[dy + T5_Y_HALF, dw, 0.0, dh]]))
        oy = dy + side * float(rng.uniform(0.3, 1.5))
        ok = _hidden_obstacle(rng, prims, d + t + behind, oy, family)
        gxy = (d + t + behind + float(rng.uniform(3.0, 6.0)), dy + float(rng.uniform(-1.5, 1.5)))
        params.update(door_w=dw, door_h=dh, door_y=dy, obstacle=ok, obstacle_xy=[d + t + behind, oy])
    params["goal_xy"] = [float(gxy[0]), float(gxy[1])]
    return prims, params


# --- test 6 tier A: generic clutter ------------------------------------------------
#
# One patch = a rectangle of clutter ahead of the state, the goal past it or
# among its last objects. Where a single-obstacle family has one face, a clutter
# patch has one per obstruction the straight start->goal line meets (tier C's
# convention, ``tiers._chord_obstructions``, measured here on the exact SDF
# instead of the voxel grid): the v3 zones then put approach states in front of
# every obstacle in the way, *inside* the clutter, rather than all of them in
# front of its first row.

def _column(xy, zr=CLUTTER_KEEPOUT_Z, step=0.1) -> np.ndarray:
    z = np.arange(zr[0], zr[1] + 1e-9, step)
    return np.column_stack([np.full(len(z), float(xy[0])), np.full(len(z), float(xy[1])), z])


def _lateral_run(sdf, ps, lim=T5_Y_HALF, step=0.05) -> tuple[float, float]:
    """(y centre, half-width) of the obstruction the chord points ``ps`` (one
    blocked stretch) pass through: at each point, the run along y of positions
    the robot cannot occupy (SDF < robot radius) that contains it, and the
    union of those runs over the stretch. Clearance, not occupancy -- tier C's
    ``_chord_span`` -- so a robot radius wider each side than the surface; a
    union over the stretch, not one row at the face, so a round obstacle's
    span is its widest, not its leading edge's."""
    ys = np.arange(-lim, lim + 1e-9, step)
    ps = np.atleast_2d(ps)
    y0 = y1 = None
    for p in ps:
        q = np.column_stack([np.full(len(ys), p[0]), ys, np.full(len(ys), p[2])])
        bad = sdf(q) < ROBOT_RADIUS
        j = int(np.clip(round((p[1] + lim) / step), 0, len(ys) - 1))
        if not bad[j]:
            continue
        j0 = j1 = j
        while j0 > 0 and bad[j0 - 1]:
            j0 -= 1
        while j1 + 1 < len(ys) and bad[j1 + 1]:
            j1 += 1
        a, b = ys[j0] - step / 2, ys[j1] + step / 2
        y0, y1 = (a, b) if y0 is None else (min(y0, a), max(y1, b))
    if y0 is None:
        return float(ps[0, 1]), 2.0
    return float(0.5 * (y0 + y1)), float(0.5 * (y1 - y0))


def clutter_records(prims: list, goal, family: str, z: float = FLIGHT_Z, step: float = 0.05) -> list[dict]:
    """The obstructions the straight start->goal chord meets at flight
    altitude, in route order: ``{face_x, far_x, y_c, half_width, family, kind}``.

    A stretch of the chord where the exact SDF is below the robot radius is an
    obstruction (stretches closer than ``CLUTTER_MERGE`` are one); its face is
    the first chord point inside a surface (SDF <= 0), its far face the last,
    or the closest approach when the chord only grazes. ``kind`` as tier C:
    ``blocked``; with none, ``squeeze`` where the chord passes within 1 m of a
    surface, else one ``clear`` record at the goal.
    """
    from .sim_sdf import PrimitiveSdf
    sdf = PrimitiveSdf(prims)
    a = np.array([0.0, 0.0, z])
    b = np.array([float(goal[0]), float(goal[1]), z])
    n = max(int(np.linalg.norm(b - a) / step) + 1, 2)
    ps = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
    ds_ = float(np.linalg.norm(b - a)) / (n - 1)          # metres between chord samples
    d = sdf(ps)
    blocked = d < ROBOT_RADIUS
    runs, i = [], 0
    while i < n:
        if not blocked[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and blocked[j + 1]:
            j += 1
        if runs and (i - runs[-1][1]) * ds_ < CLUTTER_MERGE:
            runs[-1][1] = j
        else:
            runs.append([i, j])
        i = j + 1

    def rec(i_face, i_far, kind, run=None):
        if kind == "blocked":
            y_c, hw = _lateral_run(sdf, ps[run[0]:run[1] + 1:max(int(round(0.1 / ds_)), 1)])
        else:
            y_c, hw = float(ps[i_face, 1]), 2.0
        return {"face_x": round(float(ps[i_face, 0]), 4), "far_x": round(float(ps[i_far, 0]), 4),
                "y_c": round(y_c, 4), "half_width": round(hw, 4), "family": family, "kind": kind}

    out = []
    for i, j in runs:
        seg = d[i:j + 1]
        solid = np.flatnonzero(seg <= 0.0)
        f = i + int(solid[0] if len(solid) else np.argmin(seg))
        l = i + int(solid[-1] if len(solid) else np.argmin(seg))
        out.append(rec(f, l, "blocked", (i, j)))
    if not out:
        k = int(np.argmin(d))
        out.append(rec(k, k, "squeeze") if d[k] < 1.0 else rec(n - 1, n - 1, "clear"))
    return out


def _clutter_check(spec: PatchSpec) -> str:
    """'' when the patch is usable: start and goal free on the grid the draw
    uses, and a route between them *inside the flight band* (6-connected
    components of ``free_band``, ``tiers.band_route``'s test)."""
    pts, reg = patch_cloud(spec)
    g = _BandGrid(pts, reg, spec)
    start = np.array([0.0, 0.0, spec.flight_z])
    if g.clear_at(start) < 0.5:
        return "start_blocked"
    if g.clear_at(spec.goal_pos) < 0.75:              # 1.0 m exact, less the 0.15 m voxel's diagonal
        return "goal_clearance"
    lbl, _ = ndi.label(g.free_band, structure=ndi.generate_binary_structure(3, 1))
    la, lb = lbl[tuple(g.ijk(start))], lbl[tuple(g.ijk(spec.goal_pos))]
    return "" if la != 0 and la == lb else "unreachable_band"


def _draw_clutter(family: str, seed: int, split: str, rng: np.random.Generator) -> PatchSpec:
    from .sim_sdf import PrimitiveSdf
    S = T5_SWEEP[family]
    z = FLIGHT_Z
    d = float(rng.uniform(*S["face"]))
    depth = float(rng.uniform(*S["depth"]))
    x1 = d + depth
    goal = [x1 + float(rng.uniform(*S["goal_past"])), float(rng.uniform(*S["goal_y"])), z]
    v_goal = CRUISE if rng.random() < 0.7 else 0.0
    if family == "clutter_mixed":
        knob = {"spacing": float(rng.uniform(*S["spacing"]))}
        sites = PT.clutter_mixed(rng, d, x1, CLUTTER_HALF_Y, knob["spacing"], family)
    elif family == "low_line":
        knob = {"n_line": int(rng.integers(S["n_line"][0], S["n_line"][1] + 1)),
                "n_off": int(rng.integers(S["n_off"][0], S["n_off"][1] + 1))}
        sites = PT.low_line(rng, d, x1, goal[:2], CLUTTER_HALF_Y, knob["n_line"], knob["n_off"], family)
    else:
        knob = {"density": float(rng.uniform(*S["density"]))}
        sites = PT.clutter_tilted(rng, d, x1, CLUTTER_HALF_Y, knob["density"], family)

    # whole objects out: the ones inside a keep-out column, and the ones that
    # leave the patch sideways
    cols = np.concatenate([_column((0.0, 0.0)), _column(goal[:2])])
    ylim = T5_Y_HALF - CLUTTER_EDGE
    prims, kinds, dropped = [], {}, {"keepout": 0, "edge": 0}
    for kind, obj in sites:
        if float(PrimitiveSdf(obj)(cols).min()) < CLUTTER_KEEPOUT:
            dropped["keepout"] += 1
            continue
        bb = [_prim_bbox(p) for p in obj]
        if min(e[2] for e in bb) < -ylim or max(e[3] for e in bb) > ylim:
            dropped["edge"] += 1
            continue
        prims += obj
        kinds[kind] = kinds.get(kind, 0) + 1

    records = clutter_records(prims, goal, family)
    ext = band_faces(prims) or (d, x1)
    area = depth * 2.0 * CLUTTER_HALF_Y
    n_obj = int(sum(kinds.values()))
    params = {"d": d, "depth": depth, "x_end": x1, **knob, "half_y": CLUTTER_HALF_Y,
              "n_sites": len(sites), "n_objects": n_obj, "kinds": kinds, "dropped": dropped,
              "objects_per_m2": round(n_obj / area, 5), "keepout": CLUTTER_KEEPOUT,
              "goal_rule": "clutter", "span": [records[0]["y_c"], records[0]["half_width"]],
              "face_rule": "exact SDF along the start->goal chord at flight altitude (tier C convention)",
              **face_params(records)}
    bounds = [-8.0, -T5_Y_HALF, round(max(goal[0] + 8.0, ext[1] + 6.0, 20.0), 4), T5_Y_HALF]
    return PatchSpec(test=5, family=family, seed=seed, split=split, bounds=bounds, flight_z=z,
                     params=params, face_x=float(records[0]["face_x"]), far_x=float(records[0]["far_x"]),
                     goal_pos=[float(v) for v in goal], goal_speed=float(v_goal), goal_radius=GOAL_RADIUS,
                     primitives=prims)


def _patch_clutter(family: str, seed: int, split: str) -> PatchSpec:
    """Test 6, tier A: a rectangle of generic clutter (``PT.clutter_mixed`` /
    ``PT.clutter_tilted``) swept by ``T5_SWEEP[family]``.

    Seeded and deterministic: attempt 0 draws from ``[5, FAMILY_ID, seed]``
    exactly as every tier-A family does, a redraw from ``[5, FAMILY_ID, seed,
    attempt]``. A draw is kept when ``_clutter_check`` passes (start and goal
    free, a route inside the flight band); ``params['attempts']`` and
    ``params['rejected']`` record what that took.
    """
    rejected = []
    for attempt in range(CLUTTER_ATTEMPTS):
        key = [5, FAMILY_ID[family], seed] + ([attempt] if attempt else [])
        spec = _draw_clutter(family, seed, split, np.random.default_rng(key))
        why = _clutter_check(spec)
        if not why:
            spec.params.update(attempts=attempt + 1, rejected=rejected)
            return spec
        rejected.append(why)
    raise ClutterUnbuildable(f"build_patch(5, {family!r}, {seed}, {split!r})", rejected)


def build_patch(test: int, family: str, seed: int, split: str = "train") -> PatchSpec:
    if test == 5:
        return _patch_t5(family, seed, split)
    rng = np.random.default_rng([test, FAMILY_ID[family], seed])
    prims: list = []
    z = FLIGHT_Z
    if family in ("wall_offset", "wall_sym"):
        w = float(rng.choice([1.0, 2.0, 4.0]))
        d = float(rng.uniform(3.0, 8.0))
        y_c = 0.0 if family == "wall_sym" else float(rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0]))
        t = 0.3
        _wall(prims, d + t / 2, w, y_c, family, t=t)
        params = {"width": w, "d": d, "y_c": y_c, "t": t, "height": TALL}
        face_x, far_x = d, d + t
    elif family == "float_box":
        d = float(rng.uniform(3.0, 8.0))
        half = (0.25, 1.0, 0.8)                      # 0.5 deep, 2.0 wide, 1.6 tall
        prims.append(A.box((d + half[0], 0.0, z), half, (0.0, 0.0, 0.0), family))
        params = {"d": d, "width": 2 * half[1], "height": 2 * half[2], "depth": 2 * half[0]}
        face_x, far_x = d, d + 2 * half[0]
    elif family in STAND_SPACING:
        s = STAND_SPACING[family]
        d, depth, half = 3.0, 30.0, 6.0
        for cx, cy in A._poisson_disc(rng, d, -half, d + depth, half, s, margin=0.3):
            prims.append(A.cyl_v(cx, cy, float(rng.uniform(0.2, 0.5)), 0.0, TALL, family))
        params = {"spacing": s, "d": d, "depth": depth, "half_width": half, "n_trees": len(prims)}
        face_x, far_x = d, d + depth
    else:
        raise ValueError(f"unknown family {family!r}")

    if test == 3:
        goal = [far_x + float(rng.uniform(4.0, 5.0)), float(rng.uniform(-1.0, 1.0)), z]
        v_goal = 0.0 if rng.random() < 0.7 else 2.0
        bounds = [-8.0, -8.0, max(goal[0] + 8.0, 20.0), 8.0]
    else:
        goal = [25.0, float(rng.uniform(-3.0, 3.0)), z]
        v_goal = CRUISE if rng.random() < 0.7 else 0.0
        bounds = [-8.0, -8.0, 36.0, 8.0]
    return PatchSpec(test=test, family=family, seed=seed, split=split, bounds=bounds, flight_z=z,
                     params=params, face_x=face_x, far_x=far_x, goal_pos=goal, goal_speed=v_goal,
                     goal_radius=GOAL_RADIUS, primitives=prims)



def patch_cloud(spec: PatchSpec, spacing: float = 0.1, floor_spacing: float = 0.25):
    """(points, region) in the arena ``.npz`` layout: the primitives' surfaces sampled one point per ``spacing``
    (region 0 and up) and the floor grid at ``floor_spacing`` (region -1)."""
    return PT.sample(spec.as_patch(), spacing=spacing, floor_spacing=floor_spacing)

class PatchGrid:
    """Occupancy, clearance and a time-to-go field from the goal, on the
    pair sampler's lattice; obstacles only (the floor is not an obstacle
    for clearance, the altitude band bounds z instead)."""

    def __init__(self, points, region, spec: PatchSpec, h: float = GRID_H, band=FLIGHT_BAND):
        obs = points[region >= 0]
        x0, y0, x1, y1 = spec.bounds
        self.h = h
        self.lo = np.array([x0, y0, max(0.3, band[0] - 1.0)])
        hi = np.array([x1, y1, band[1] + 1.0])
        self.dims = np.ceil((hi - self.lo) / h).astype(int) + 1
        occ = np.zeros(self.dims, bool)
        if len(obs):
            ijk = np.floor((obs - self.lo) / h).astype(int)
            ok = np.all((ijk >= 0) & (ijk < self.dims), axis=1)
            occ[tuple(ijk[ok].T)] = True
        self.occ = occ
        self.clearance = h * ndi.distance_transform_edt(~occ)
        self.free = self.clearance >= ROBOT_RADIUS
        zc = self.lo[2] + (np.arange(self.dims[2]) + 0.5) * h
        self.band = (zc >= band[0]) & (zc <= band[1])
        self.free_band = self.free & self.band[None, None, :]
        self.T = self.march(np.asarray(spec.goal_pos, float))

    def ijk(self, p):
        return np.clip(np.floor((np.asarray(p, float) - self.lo) / self.h).astype(int), 0, self.dims - 1)

    def centre(self, ijk):
        return self.lo + (np.asarray(ijk) + 0.5) * self.h

    def clear_at(self, p) -> float:
        return float(self.clearance[tuple(self.ijk(p))])

    def march(self, goal: np.ndarray) -> np.ndarray:
        g = self.ijk(goal)
        if not self.free[tuple(g)]:
            fi = np.argwhere(self.free)
            g = fi[int(np.argmin(np.sum((fi - g) ** 2, axis=1)))]
        phi = np.ones(self.dims); phi[tuple(g)] = -1.0
        T = skfmm.travel_time(np.ma.MaskedArray(phi, mask=~self.free), np.where(self.free, CRUISE, 0.0), dx=self.h)
        return np.ma.filled(T, np.inf).astype(float)

    def descent(self, ijk) -> np.ndarray | None:
        g = np.zeros(3); c = self.T[tuple(ijk)]
        for a in range(3):
            lo_, hi_ = ijk.copy(), ijk.copy()
            lo_[a] = max(lo_[a] - 1, 0); hi_[a] = min(hi_[a] + 1, self.dims[a] - 1)
            tl, th = self.T[tuple(lo_)], self.T[tuple(hi_)]
            if np.isfinite(tl) and np.isfinite(th):
                g[a] = (th - tl) / (2 * self.h)
            elif np.isfinite(th):
                g[a] = (th - c) / self.h
            elif np.isfinite(tl):
                g[a] = (c - tl) / self.h
        n = np.linalg.norm(g)
        return None if n < 1e-9 else -g / n

    def line_of_sight(self, a, b, radius=0.3) -> bool:
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(int(np.linalg.norm(b - a) / (self.h / 2)), 2)
        for t in np.linspace(0.0, 1.0, n):
            if self.clearance[tuple(self.ijk(a + t * (b - a)))] < radius:
                return False
        return True

def face_params(records: list[dict]) -> dict:
    """The four face keys of ``params``, projected from one list of records.

    A record is ``{face_x, far_x, y_c, half_width}`` plus whatever labels the
    tier can name (``family``, ``kind``); the projections are

    ``faces``        [face_x, ...]                     floats, near face first
    ``far_faces``    [far_x, ...]
    ``face_spans``   [[face_x, far_x, y_c, half_width], ...]  what ``faces()``
                     reads
    ``face_records`` the records themselves, for the QA that wants the labels

    Every tier writes them through here and nowhere else. Before this, tier A
    wrote ``faces`` as ``[float]`` and tiers B and C wrote ``[dict]`` under
    the same name: a shard holding both is unanalysable, and popping
    ``face_spans`` off a tier-B spec made ``faces()`` try ``float(dict)``.
    ``faces`` is a list of floats in every tier now, and the dicts have a key
    of their own -- the two can no longer be confused, rather than the
    confusion being caught.
    """
    out = []
    for r in records:
        fx, far = float(r["face_x"]), float(r["far_x"])
        out.append((fx, far, float(r["y_c"]), float(r["half_width"])))
    return {"faces": [t[0] for t in out], "far_faces": [t[1] for t in out],
            "face_spans": [list(t) for t in out], "face_records": list(records)}

class _BandGrid(PatchGrid):
    """``PatchGrid`` without the time-to-go march: occupancy, clearance and the
    flight band only, which is all a reachability-in-the-band test reads."""

    def march(self, goal):
        return np.zeros(self.dims)

@contextmanager
def primitive_clouds():
    """The tier builders validate against the primitives' own cloud. In the labeller this switched off its
    course-cloud hook; this package has no such hook, so it is a no-op kept for the builders' call sites."""
    yield
