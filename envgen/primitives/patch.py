"""Per-sample training patches: one state, one goal, the geometry between.

A *patch* is the small scene one label is drawn in. Where ``arena.py`` lays
nine obstacle families out on one map for inspection and closed-loop
evaluation, a patch places one or two families deliberately around a state
so that the goal regime (``docs/GOAL_REGIMES.md``) is chosen, not found:
the family sits a known distance ahead of the state, the goal is put where
the regime says, and every draw is its own seed.

Patch frame: the state is at the origin flying +x at ``speed``; the patch
runs ``X_BEHIND`` behind it and ``X_AHEAD`` ahead, ``Y_HALF`` to each side,
floor at z = 0, flight altitude ``FLIGHT_Z``. Far goals sit 12-22 m past the
obstruction, so the patch is long along the heading and narrow across it.

Families (each takes the distance ``d`` of its near face ahead of the state):

    wall_slot   wall across the patch with a floor-to-top slot (``width``)
    window      wall across the patch with a hole at flight height
                (``width`` x ``height``, sill at ``sill``); the wall runs the
                patch's full width so "around" is a long detour and
                "through" is the intended label
    slab        one box across the heading (blocked goals, stop goals)
    pillar      one vertical cylinder on the heading (inside goals)
    stand       cylinder trees in a band ahead, Poisson spacing ``spacing``
    chicane     two staggered walls with an S passage between them
    ring        cylindrical shell around a point with one gap (enclosed goals)
    room        four walls with a doorway under a lintel
    bars        thin horizontal cylinders across the heading at band height
    lintel      a beam across the heading, the route must pass under it
    open        nothing

Generic clutter (test 6, tier A; each returns one list of primitives per
*site*, so a caller can drop a whole object -- an overhang and its post --
rather than half of one):

    clutter_mixed   Poisson-disc sites, every site one of six objects of mixed
                    height: poles, half-sunk boulders, floating spheres,
                    crates (some below the band), overhanging slabs (some on a
                    post), horizontal bars at any yaw
    clutter_tilted  complete spatial randomness (a Poisson count, uniform
                    positions), every site a grounded object that leans:
                    tilted blocks, tilted panels, leaning struts, boulders

    low_line        (test 6 DAgger round 2) LOW obstacles on the start->goal
                    line: spheres with the centre anywhere from 0.4 m up,
                    crates 0.3-1.3 m tall, half-sunk boulders -- the tops sit
                    around and below the 1.0-2.5 m flight band, where the
                    other families never put anything on the line -- plus a
                    few of the same off the line

The Isaac evaluation fields (``superfly/sim/obstacle_field.py``: ``generate``
and ``generate_diffaero``) are the held-out out-of-distribution test, so these
two share their primitive *types* only. Sizes, heights, tilt, counts, placement
law and density are drawn from different distributions on purpose -- the
numbers side by side are in ``draw_states.CLUTTER_VS_ISAAC``.

The same JSON spec + ``.npz`` contract as the arena, so the pair sampler,
the C++ labeller and any renderer read it unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math

import numpy as np

from . import arena as A

X_BEHIND, X_AHEAD, Y_HALF = 8.0, 32.0, 10.0
FLIGHT_Z = 1.75
FLIGHT_BAND = (1.0, 2.5)
#: The labeller's altitude band around the state, a hard reject on samples.
ALT_BAND = 1.5
GOAL_RADIUS = 0.5
ROBOT_RADIUS = A.ROBOT_RADIUS


@dataclass
class Patch:
    seed: int
    regime: str
    family: str
    bounds: list[float]                      # x0, y0, x1, y1
    flight_z: float
    state_pos: list[float]
    state_vel: list[float]
    goal_pos: list[float]
    goal_speed: float
    goal_radius: float
    primitives: list[dict] = field(default_factory=list)
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    def regions(self) -> dict[str, list[float]]:
        return {self.family: list(self.bounds)}


# --- families ------------------------------------------------------------------

def wall_slot(prims, d, width=1.4, y=0.0, height=4.0, t=0.3, fam="wall_slot", half=6.0):
    s = y + half                             # distance of the slot along p0->p1
    prims.append(A.wall((d, -half), (d, half), 0.0, height, t, fam,
                        slots=[[s, width, 0.0, height]]))


def window(prims, d, width=1.2, height=0.9, sill=1.3, y=0.0, t=0.25, fam="window"):
    prims.append(A.wall((d, -Y_HALF), (d, Y_HALF), 0.0, 4.0, t, fam,
                        slots=[[Y_HALF + y, width, sill, sill + height]]))


def slab(prims, d, half=(0.3, 2.0, 2.0), y=0.0, fam="slab"):
    prims.append(A.box((d + half[0], y, half[2]), half, (0, 0, 0), fam))


def pillar(prims, d, r=0.6, y=0.0, fam="pillar"):
    prims.append(A.cyl_v(d + r, y, r, 0.0, A.TALL, fam))


def stand(prims, rng, d, depth=10.0, spacing=2.5, half=Y_HALF - 1.0, fam="stand"):
    for cx, cy in A._poisson_disc(rng, d, -half, d + depth, half, spacing):
        prims.append(A.cyl_v(cx, cy, rng.uniform(0.15, 0.35), 0.0, A.TALL, fam))


def chicane(prims, d, gap=1.8, depth=4.0, half=6.0, fam="chicane"):
    """Wall from the left leaving a gap on the right, then the mirror."""
    prims.append(A.wall((d, -half), (d, half - gap), 0.0, 3.5, 0.25, fam))
    prims.append(A.wall((d + depth, half), (d + depth, -half + gap), 0.0, 3.5, 0.25, fam))


def ring(prims, centre, r=2.5, gap_deg=70.0, gap_at_deg=180.0, fam="ring", n=48):
    """Cylindrical shell of thin pillars, one gap facing ``gap_at_deg``."""
    cx, cy = centre
    for k in range(n):
        ang = 360.0 * k / n
        if abs((ang - gap_at_deg + 180.0) % 360.0 - 180.0) < gap_deg / 2:
            continue
        a = math.radians(ang)
        prims.append(A.cyl_v(cx + r * math.cos(a), cy + r * math.sin(a), 0.18, 0.0, 3.5, fam))


def room(prims, d, w=6.0, depth=6.0, door_w=1.2, door_h=2.2, door_y=0.0, fam="room"):
    x0, x1, y0, y1 = d, d + depth, -w / 2, w / 2
    prims.append(A.wall((x0, y0), (x0, y1), 0.0, 3.0, 0.2, fam,
                        slots=[[door_y - y0, door_w, 0.0, door_h]]))
    prims.append(A.wall((x0, y1), (x1, y1), 0.0, 3.0, 0.2, fam))
    prims.append(A.wall((x1, y1), (x1, y0), 0.0, 3.0, 0.2, fam))
    prims.append(A.wall((x1, y0), (x0, y0), 0.0, 3.0, 0.2, fam))


def bars(prims, rng, d, n=4, depth=8.0, fam="bars"):
    for _ in range(n):
        x = rng.uniform(d, d + depth)
        z = rng.uniform(1.1, 2.4)
        prims.append(A.cyl_h((x, -Y_HALF, z), (x, Y_HALF, z), rng.uniform(0.05, 0.15), fam))


def lintel(prims, d, z_under=1.4, t=0.6, fam="lintel"):
    """A beam whose underside is at ``z_under``: duck or go round nothing."""
    prims.append(A.box((d + t / 2, 0.0, z_under + 1.3), (t / 2, Y_HALF, 1.3), (0, 0, 0), fam))


# --- generic clutter ---------------------------------------------------------------
# Both builders fill the rectangle [x0, x1] x [-half_y, half_y] with *sites* and
# put one object on each. Every draw comes from ``rng`` in a fixed order, so a
# seed is a scene. Object kinds are chosen per site from a fixed weight table.

def grounded_box(x, y, half, rpy, fam):
    """A box, rotated by ``rpy`` (intrinsic XYZ, as ``arena.box``), lowered
    until its lowest corner rests on the floor."""
    corners = np.array([[i, j, k] for i in (-1.0, 1.0) for j in (-1.0, 1.0) for k in (-1.0, 1.0)])
    z_low = float(((corners * np.asarray(half, float)) @ A._euler_xyz(*rpy).T)[:, 2].min())
    return A.box((x, y, -z_low), half, rpy, fam)


def _tilt(rng, max_deg):
    """(roll, pitch) of a lean of magnitude U(0, max_deg) in a uniform
    direction -- a lean, not two independent axis draws."""
    mag = math.radians(rng.uniform(0.0, max_deg))
    phi = rng.uniform(-math.pi, math.pi)
    return mag * math.cos(phi), mag * math.sin(phi)


#: clutter_mixed: (kind, weight). Heights are mixed on purpose: some crates and
#: poles stop below or inside the 1.0-2.5 m band, slabs float over it.
MIXED_KINDS = (("pole", 0.25), ("boulder", 0.15), ("floater", 0.10),
               ("crate", 0.25), ("overhang", 0.15), ("bar", 0.10))


def _mixed_object(rng, x, y, kind, fam):
    if kind == "pole":
        return [A.cyl_v(x, y, rng.uniform(0.08, 0.6), 0.0, rng.uniform(1.3, 6.5), fam)]
    if kind == "boulder":                       # grounded, partly sunk
        r = rng.uniform(0.3, 1.6)
        return [A.sphere((x, y, r * rng.uniform(0.4, 1.0)), r, fam)]
    if kind == "floater":
        return [A.sphere((x, y, rng.uniform(0.9, 3.2)), rng.uniform(0.2, 0.8), fam)]
    if kind == "crate":                         # 0.3-3.5 m tall: low ones are flown over
        half = (rng.uniform(0.25, 1.5), rng.uniform(0.25, 1.5), 0.5 * rng.uniform(0.3, 3.5))
        return [A.box((x, y, half[2]), half, (0.0, 0.0, rng.uniform(0.0, math.pi)), fam)]
    if kind == "overhang":                      # a slab to fly under, sometimes on a post
        half = (rng.uniform(0.3, 1.0), rng.uniform(0.6, 2.5), rng.uniform(0.08, 0.3))
        z_bot, yaw = rng.uniform(1.6, 3.0), rng.uniform(0.0, math.pi)
        out = [A.box((x, y, z_bot + half[2]), half, (0.0, 0.0, yaw), fam)]
        if rng.random() < 0.5:
            s = rng.uniform(-0.8, 0.8) * half[1]      # under the slab, along its long axis
            out.append(A.cyl_v(x - s * math.sin(yaw), y + s * math.cos(yaw),
                               rng.uniform(0.08, 0.2), 0.0, z_bot, fam))
        return out
    if kind == "bar":                           # any yaw, not only across the heading
        L, z, yaw = rng.uniform(1.5, 6.0), rng.uniform(0.5, 3.0), rng.uniform(0.0, math.pi)
        dx, dy = 0.5 * L * math.cos(yaw), 0.5 * L * math.sin(yaw)
        return [A.cyl_h((x - dx, y - dy, z), (x + dx, y + dy, z), rng.uniform(0.04, 0.2), fam)]
    raise ValueError(f"unknown clutter kind {kind!r}")


def clutter_mixed(rng, x0, x1, half_y, spacing, fam="clutter_mixed"):
    """Mixed-height clutter on Poisson-disc sites ``spacing`` apart.

    Returns ``[(kind, [primitive, ...]), ...]``, one entry per site, in
    placement order."""
    kinds = [k for k, _ in MIXED_KINDS]
    w = np.array([v for _, v in MIXED_KINDS]); w = w / w.sum()
    out = []
    for cx, cy in A._poisson_disc(rng, x0, -half_y, x1, half_y, spacing):
        kind = str(kinds[int(rng.choice(len(kinds), p=w))])
        out.append((kind, _mixed_object(rng, float(cx), float(cy), kind, fam)))
    return out


#: clutter_tilted: (kind, weight). Everything stands on the floor and leans.
TILTED_KINDS = (("block", 0.50), ("panel", 0.20), ("strut", 0.15), ("boulder", 0.15))


def _tilted_object(rng, x, y, kind, fam):
    if kind == "block":                         # 0.3-2.0 m footprint, 1.2-9 m tall, leaning 0-35 deg
        half = (rng.uniform(0.15, 1.0), rng.uniform(0.15, 1.0), 0.5 * rng.uniform(1.2, 9.0))
        roll, pitch = _tilt(rng, 35.0)
        return [grounded_box(x, y, half, (roll, pitch, rng.uniform(-math.pi, math.pi)), fam)]
    if kind == "panel":                         # a thin board, leaning 0-45 deg
        half = (rng.uniform(0.04, 0.12), rng.uniform(0.5, 2.0), rng.uniform(0.6, 2.5))
        roll, pitch = _tilt(rng, 45.0)
        return [grounded_box(x, y, half, (roll, pitch, rng.uniform(-math.pi, math.pi)), fam)]
    if kind == "strut":                         # a pole leaning 20-60 deg off vertical
        L, th, phi = rng.uniform(2.5, 7.0), math.radians(rng.uniform(20.0, 60.0)), rng.uniform(-math.pi, math.pi)
        top = (x + L * math.sin(th) * math.cos(phi), y + L * math.sin(th) * math.sin(phi), L * math.cos(th))
        return [A.cyl_h((x, y, 0.0), top, rng.uniform(0.06, 0.25), fam)]
    if kind == "boulder":
        r = rng.uniform(0.3, 1.2)
        return [A.sphere((x, y, r), r, fam)]
    raise ValueError(f"unknown clutter kind {kind!r}")


def clutter_tilted(rng, x0, x1, half_y, density, fam="clutter_tilted"):
    """Leaning clutter, complete spatial randomness: a Poisson number of sites
    (mean ``density`` per m^2 of the rectangle), each uniform in it.

    Returns ``[(kind, [primitive, ...]), ...]`` as ``clutter_mixed``."""
    kinds = [k for k, _ in TILTED_KINDS]
    w = np.array([v for _, v in TILTED_KINDS]); w = w / w.sum()
    n = int(rng.poisson(density * (x1 - x0) * 2.0 * half_y))
    out = []
    for _ in range(n):
        cx, cy = float(rng.uniform(x0, x1)), float(rng.uniform(-half_y, half_y))
        kind = str(kinds[int(rng.choice(len(kinds), p=w))])
        out.append((kind, _tilted_object(rng, cx, cy, kind, fam)))
    return out


#: low_line: (kind, weight). Everything is low: the tier-A spheres elsewhere
#: float at centre z 1.0-2.6, so a low object on the goal line never occurs.
LOW_KINDS = (("low_sphere", 0.45), ("low_crate", 0.35), ("sunk_boulder", 0.20))


def _low_object(rng, x, y, kind, fam):
    if kind == "low_sphere":                    # centre 0.4-2.6 m, r 0.3-0.8: tops 0.7-3.4 m
        return [A.sphere((x, y, rng.uniform(0.4, 2.6)), rng.uniform(0.3, 0.8), fam)]
    if kind == "low_crate":                     # 0.3-1.3 m tall, 0.5-2.0 m footprint, yawed
        half = (rng.uniform(0.25, 1.0), rng.uniform(0.25, 1.0), 0.5 * rng.uniform(0.3, 1.3))
        return [A.box((x, y, half[2]), half, (0.0, 0.0, rng.uniform(0.0, math.pi)), fam)]
    if kind == "sunk_boulder":                  # r 0.6-1.4, centre from 0.3 r below to 0.5 r above the floor
        r = rng.uniform(0.6, 1.4)
        return [A.sphere((x, y, r * rng.uniform(-0.3, 0.5)), r, fam)]
    raise ValueError(f"unknown low kind {kind!r}")


def low_line(rng, x0, x1, goal_xy, half_y, n_line, n_off, fam="low_line"):
    """``n_line`` low objects ON the start->goal line (x spread over [x0, x1],
    lateral offset U(-0.8, 0.8) m from the chord from the origin to
    ``goal_xy``) and ``n_off`` of the same kinds anywhere in |y| <= ``half_y``.

    Returns ``[(kind, [primitive, ...]), ...]`` as ``clutter_mixed``."""
    kinds = [k for k, _ in LOW_KINDS]
    w = np.array([v for _, v in LOW_KINDS]); w = w / w.sum()
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    xs = np.sort(rng.uniform(x0, x1, size=int(n_line)))
    out = []
    for cx in xs:
        cy = gy * float(cx) / max(gx, 1e-6) + float(rng.uniform(-0.8, 0.8))
        kind = str(kinds[int(rng.choice(len(kinds), p=w))])
        out.append((kind, _low_object(rng, float(cx), cy, kind, fam)))
    for _ in range(int(n_off)):
        cx, cy = float(rng.uniform(x0, x1)), float(rng.uniform(-half_y, half_y))
        kind = str(kinds[int(rng.choice(len(kinds), p=w))])
        out.append((kind, _low_object(rng, cx, cy, kind, fam)))
    return out


# --- the regimes, one patch each ------------------------------------------------

def _patch(seed, regime, family, prims, goal, v_goal, speed=3.0, note=""):
    return Patch(
        seed=seed, regime=regime, family=family,
        bounds=[-X_BEHIND, -Y_HALF, X_AHEAD, Y_HALF], flight_z=FLIGHT_Z,
        state_pos=[0.0, 0.0, FLIGHT_Z], state_vel=[speed, 0.0, 0.0],
        goal_pos=[float(v) for v in goal], goal_speed=float(v_goal),
        goal_radius=GOAL_RADIUS, primitives=prims, note=note,
    )


def catalogue(seed: int = 0) -> list[Patch]:
    """One example patch per regime and per uncovered case of GOAL_REGIMES.md."""
    rng = np.random.default_rng(seed)
    z = FLIGHT_Z
    out = []

    p = []; wall_slot(p, 5.0, width=1.5, y=rng.uniform(-3, 3))
    out.append(_patch(seed, "R1 beyond, reachable", "wall_slot", p, (9.0, 0.0, z), 2.0,
                      note="Detour through the slot or round the wall's end, arrive at 2 m/s."))

    p = []; window(p, 5.0, width=1.2, height=0.9, sill=1.3, y=rng.uniform(-1.5, 1.5))
    out.append(_patch(seed, "R1 beyond, reachable", "window", p, (9.5, 0.0, z), 2.0,
                      note="The wall spans the patch: through the window is the only short route."))

    p = []; stand(p, rng, 3.0, depth=12.0, spacing=3.0)
    out.append(_patch(seed, "R2 beyond, not reachable", "stand", p, (26.0, rng.uniform(-3, 3), z), 3.0,
                      note="Far goal past a stand: the label is the first second of the route at cruise."))

    p = []; slab(p, 7.0, half=(0.3, 2.5, 2.0))
    out.append(_patch(seed, "R3 stop before an obstacle", "slab", p, (5.5, 0.0, z), 0.0,
                      note="Goal 1.5 m short of the slab; brake and hold, never a state that cannot stop."))

    p = []; slab(p, 6.0, half=(0.3, 2.5, 2.0))
    out.append(_patch(seed, "R3 stop, too fast", "slab", p, (4.0, 0.0, z), 0.0, speed=6.0,
                      note="At 6 m/s the goal is inside the stopping distance: max brake, overshoot, return."))

    p = []; pillar(p, -6.0, r=0.5, y=-1.0)
    out.append(_patch(seed, "R3 stop, turn-around", "pillar", p, (-4.0, 1.5, z), 0.0,
                      note="Goal behind the drone: brake, turn, arrive; the pillar sits on the way back."))

    p = []; pillar(p, 6.0, r=0.6)
    out.append(_patch(seed, "R4 inside geometry", "pillar", p, (6.6, 0.0, z), 0.0,
                      note="Goal at the pillar's axis: project to the nearest free cell on the drone's side, stop."))

    p = []; ring(p, (8.0, 0.0), r=2.5, gap_deg=70.0, gap_at_deg=0.0)
    out.append(_patch(seed, "R1 beyond, long detour", "ring", p, (8.0, 0.0, z), 0.0,
                      note="Ring with its one gap on the far side: reachable, but only by going round to the gap. The nearest free cell is right in front of the drone, and it is the wrong answer."))

    p = []; ring(p, (8.0, 0.0), r=2.5, gap_deg=0.0)
    out.append(_patch(seed, "R4 enclosed", "ring", p, (8.0, 0.0, z), 0.0,
                      note="Closed ring, no route: the goal projects to the reachable free cell nearest it, on the drone's side; the label stops at the shell instead of being skipped."))

    out.append(_patch(seed, "open", "open", [], (7.0, rng.uniform(-2, 2), z), 0.0,
                      note="Nothing in the way: stop goal within reach. Budgeted so it does not swamp the rest."))

    p = []; wall_slot(p, 5.0, width=0.9, y=0.0)
    out.append(_patch(seed, "gap near body size", "wall_slot", p, (9.0, 0.0, z), 2.0,
                      note="0.9 m slot against a 0.85 m feasibility diameter: 'through' stops being an option."))

    p = []; chicane(p, 4.0, gap=1.8, depth=4.0); stand(p, rng, 11.0, depth=8.0, spacing=3.5, half=5.0)
    out.append(_patch(seed, "sequence", "chicane+stand", p, (22.0, 0.0, z), 3.0,
                      note="Several bends inside one route: the S, then a stand, goal beyond both."))

    p = []; lintel(p, 5.0, z_under=1.3); bars(p, rng, 8.0, n=3, depth=6.0)
    out.append(_patch(seed, "altitude", "lintel+bars", p, (17.0, 0.0, z), 2.0,
                      note="Under the beam, between the bars: the band decides, not the top-down map."))
    return out


# --- surfaces + the labeller's route ----------------------------------------------

def sample(patch: Patch, spacing=0.1, floor_spacing=0.25):
    names = sorted({p["region"] for p in patch.primitives})
    scene = A.Arena(seed=patch.seed, size=0.0, flight_band=FLIGHT_BAND, robot_radius=ROBOT_RADIUS,
                    primitives=patch.primitives, regions={n: patch.bounds for n in names})
    if patch.primitives:
        pts, ids = A.sample_surfaces(scene, spacing=spacing, floor_spacing=1e9, floor_margin=0.0)
        pts, ids = pts[ids >= 0], ids[ids >= 0]
    else:
        pts, ids = np.zeros((0, 3)), np.zeros(0, np.int16)
    x0, y0, x1, y1 = patch.bounds
    xs = np.arange(x0, x1 + floor_spacing / 2, floor_spacing)
    ys = np.arange(y0, y1 + floor_spacing / 2, floor_spacing)
    X, Y = np.meshgrid(xs, ys)
    floor = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    return (np.concatenate([pts, floor]),
            np.concatenate([ids, np.full(len(floor), -1, np.int16)]))


def route(patch: Patch, points, region, h=0.2):
    """Geodesic route from the state to the goal through the labeller's own
    construction: occupancy inflated by the robot radius, the altitude band
    as a hard limit, a Fast-Marching time field from the goal, steepest
    descent from the state. Goals inside geometry are projected to the
    nearest free cell reachable from the state (the R4 rule).

    Returns ``(path (N,3), projected_goal (3,), reachable: bool)``.
    """
    import scipy.ndimage as ndi
    import skfmm

    obs = points[region >= 0]
    x0, y0, x1, y1 = patch.bounds
    z0, z1 = max(0.5, patch.flight_z - ALT_BAND), patch.flight_z + ALT_BAND
    lo = np.array([x0, y0, z0])
    dims = np.ceil((np.array([x1, y1, z1]) - lo) / h).astype(int) + 1
    occ = np.zeros(dims, bool)
    if len(obs):
        ijk = np.floor((obs - lo) / h).astype(int)
        ok = np.all((ijk >= 0) & (ijk < dims), axis=1)
        occ[tuple(ijk[ok].T)] = True
    free = h * ndi.distance_transform_edt(~occ) > ROBOT_RADIUS + 0.05
    # a floor at z=0 is below the volume; keep the lowest slab as a clearance floor
    free[:, :, 0] = False

    def cell(p):
        return tuple(np.clip(np.floor((np.asarray(p) - lo) / h).astype(int), 0, dims - 1))

    s = cell(patch.state_pos)
    # reachable set from the state
    phi = np.ones(dims); phi[s] = -1
    T_s = skfmm.distance(np.ma.MaskedArray(phi, mask=~free), dx=h)
    reach = ~np.ma.getmaskarray(T_s)
    g = cell(patch.goal_pos)
    reachable = bool(reach[g])
    if not reachable:
        idx = np.argwhere(reach)
        centres = lo + (idx + 0.5) * h
        d = np.linalg.norm(centres - np.asarray(patch.goal_pos), axis=1)
        # nearest reachable cell to the goal; ties toward the state by T
        near = d <= d.min() + h
        g = tuple(idx[near][np.argmin(T_s[tuple(idx[near].T)])])
    goal = lo + (np.array(g) + 0.5) * h

    phi = np.ones(dims); phi[g] = -1
    T = skfmm.distance(np.ma.MaskedArray(phi, mask=~free), dx=h).filled(np.inf)
    path = [np.asarray(patch.state_pos, float)]
    c = np.array(s)
    offsets = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                        if (i, j, k) != (0, 0, 0)])
    for _ in range(4000):
        nb = c + offsets
        ok = np.all((nb >= 0) & (nb < dims), axis=1)
        nb = nb[ok]
        t = T[tuple(nb.T)]
        best = np.argmin(t)
        if not np.isfinite(t[best]) or t[best] >= T[tuple(c)]:
            break
        c = nb[best]
        path.append(lo + (c + 0.5) * h)
        if np.all(c == g):
            break
    return np.array(path), goal, reachable
