"""A mixed-difficulty training arena as analytic primitives.

One square map, flat floor, nine regions on a 3 x 3 tiling so a single state
sample "all over the map" meets every obstacle class at once: open ground,
a sparse and a dense tree stand, a wall with a slot, a chicane, a walled
room with a doorway, a box yard with floating crates, thin horizontal bars
with floating spheres, and a pillar-and-tilted-slab clutter cell.

The two shipped field generators (``obstacle_field.py``) are corridor-shaped
and sparse; they never build walls, so no detour longer than a few percent
of the chord exists anywhere in them. The arena exists to cross that
ceiling: the wall, chicane and room force real detours, the tree stands set
a clearance curriculum, the bars and spheres are the thin, non-vertical
features a depth policy has to notice.

Everything is sized against ``ROBOT_RADIUS`` (the labeller's coarse gate):
the narrowest deliberate gap (the room doorway) is 1.2 m, four radii.

The *spec* -- the list of primitives with their parameters -- is the scene.
``sample_surfaces`` turns it into the ``.npz`` cloud the experts read;
anything that renders it (Isaac, a ray-caster) reads the same spec. Every
primitive carries a ``region`` tag so states and pairs can later be
attributed to the class of geometry they were drawn in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Iterable

import numpy as np

#: What ``scripts/toa_field.py`` and the labeller's endpoint gate assume.
ROBOT_RADIUS = 0.30
#: Default side of the square arena, metres, centred on the origin.
ARENA = 40.0
#: The band the pair sampler draws states in; obstacles are built to matter here.
FLIGHT_BAND = (1.0, 2.5)
#: Trees, walls and pillars are this tall unless said otherwise: well above
#: the band, so "over the top" is not a route the labeller can take.
TALL = 5.0

Primitive = dict


@dataclass
class Arena:
    seed: int
    size: float
    flight_band: tuple[float, float]
    robot_radius: float
    primitives: list[Primitive] = field(default_factory=list)
    regions: dict[str, list[float]] = field(default_factory=dict)  # name -> [x0,y0,x1,y1]
    start: list[float] = field(default_factory=list)
    goal: list[float] = field(default_factory=list)
    families: dict[str, str] = field(default_factory=dict)         # name -> family kind

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Arena":
        raw = json.loads(text)
        raw["flight_band"] = tuple(raw["flight_band"])
        return cls(**raw)

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.primitives:
            out[p["kind"]] = out.get(p["kind"], 0) + 1
        return out


# --- primitive constructors -------------------------------------------------
# Kinds and their parameters (metres, radians, world ENU, z up):
#   sphere    c[3] r
#   box       c[3] half[3] rpy[3]         intrinsic XYZ euler, as obstacle_field
#   cyl_v     c[2] r z0 z1                vertical, from z0 to z1
#   cyl_h     a[3] b[3] r                 capsule-less cylinder between two points
#   wall      p0[2] p1[2] z0 z1 t         thick vertical slab along p0->p1
#             slots: [[s, w, z0, z1], ...] openings at distance s along p0->p1


def sphere(c, r, region):
    return {"kind": "sphere", "c": [float(v) for v in c], "r": float(r), "region": region}


def box(c, half, rpy, region):
    return {"kind": "box", "c": [float(v) for v in c], "half": [float(v) for v in half],
            "rpy": [float(v) for v in rpy], "region": region}


def cyl_v(cx, cy, r, z0, z1, region):
    return {"kind": "cyl_v", "c": [float(cx), float(cy)], "r": float(r),
            "z0": float(z0), "z1": float(z1), "region": region}


def cyl_h(a, b, r, region):
    return {"kind": "cyl_h", "a": [float(v) for v in a], "b": [float(v) for v in b],
            "r": float(r), "region": region}


def wall(p0, p1, z0, z1, t, region, slots=()):
    return {"kind": "wall", "p0": [float(v) for v in p0], "p1": [float(v) for v in p1],
            "z0": float(z0), "z1": float(z1), "t": float(t),
            "slots": [[float(v) for v in s] for s in slots], "region": region}


# --- placement helpers --------------------------------------------------------

def _poisson_disc(rng, x0, y0, x1, y1, spacing, margin=0.0, tries=4000):
    """Dart-throwing with a minimum centre spacing; good enough for a cell."""
    pts: list[np.ndarray] = []
    for _ in range(tries):
        p = rng.uniform([x0 + margin, y0 + margin], [x1 - margin, y1 - margin])
        if all(np.linalg.norm(p - q) >= spacing for q in pts):
            pts.append(p)
    return pts


# --- the regions -------------------------------------------------------------

def _forest(rng, cell, name, spacing, prims):
    x0, y0, x1, y1 = cell
    for cx, cy in _poisson_disc(rng, x0, y0, x1, y1, spacing, margin=0.8):
        prims.append(cyl_v(cx, cy, rng.uniform(0.15, 0.35), 0.0, TALL, name))


def _wall_slot(rng, cell, name, prims):
    """A wall across the cell's full width, one slot off-centre.

    The wall runs north-south through the cell centre, so any east-west route
    through the middle of the map goes through the slot or around the ends.
    """
    x0, y0, x1, y1 = cell
    xm = 0.5 * (x0 + x1)
    length = y1 - y0
    slot_w = rng.uniform(1.4, 1.8)
    slot_s = rng.uniform(0.25, 0.4) * length  # off-centre, closer to y0
    prims.append(wall((xm, y0), (xm, y1), 0.0, 4.0, 0.3, name,
                      slots=[[slot_s, slot_w, 0.0, 4.0]]))


def _chicane(rng, cell, name, prims):
    """Two staggered walls with overlapping ends: the route is an S."""
    x0, y0, x1, y1 = cell
    w = x1 - x0
    gap = rng.uniform(1.6, 2.2)          # passage width between the wall tips
    overlap = 0.55 * w
    ya = y0 + 0.36 * (y1 - y0)
    yb = y0 + 0.64 * (y1 - y0)
    prims.append(wall((x0, ya), (x0 + overlap, ya), 0.0, 3.5, 0.25, name))
    prims.append(wall((x1 - overlap, yb), (x1, yb), 0.0, 3.5, 0.25, name))
    # a pillar in each passage so the S is not a straight slide
    prims.append(cyl_v(x0 + overlap + gap, ya - 0.5, 0.2, 0.0, TALL, name))
    prims.append(cyl_v(x1 - overlap - gap, yb + 0.5, 0.2, 0.0, TALL, name))


def _room(rng, cell, name, prims):
    """A rectangular room with one doorway and one window at flight height."""
    x0, y0, x1, y1 = cell
    m = 2.5
    rx0, ry0, rx1, ry1 = x0 + m, y0 + m, x1 - m, y1 - m
    door_w = 1.2
    door_s = rng.uniform(0.3, 0.7) * (rx1 - rx0)
    win_s = rng.uniform(0.3, 0.7) * (ry1 - ry0)
    # south wall carries the doorway (floor to 2.2 m), east wall a window in the band
    prims.append(wall((rx0, ry0), (rx1, ry0), 0.0, 3.0, 0.2, name,
                      slots=[[door_s, door_w, 0.0, 2.2]]))
    prims.append(wall((rx1, ry0), (rx1, ry1), 0.0, 3.0, 0.2, name,
                      slots=[[win_s, 1.4, 1.0, 2.4]]))
    prims.append(wall((rx1, ry1), (rx0, ry1), 0.0, 3.0, 0.2, name))
    prims.append(wall((rx0, ry1), (rx0, ry0), 0.0, 3.0, 0.2, name))
    # something inside worth flying to
    cx, cy = 0.5 * (rx0 + rx1), 0.5 * (ry0 + ry1)
    prims.append(box((cx, cy, 0.5), (0.5, 0.5, 0.5), (0, 0, rng.uniform(0, math.pi)), name))


def _box_yard(rng, cell, name, prims):
    """Crates on the ground and floating ones in the band, yawed."""
    x0, y0, x1, y1 = cell
    for cx, cy in _poisson_disc(rng, x0, y0, x1, y1, 3.2, margin=1.0):
        half = rng.uniform([0.3, 0.3, 0.3], [1.0, 1.0, 0.9])
        if rng.random() < 0.6:
            cz = half[2]                         # resting on the floor
        else:
            cz = rng.uniform(1.2, 2.4)           # floating in the band
        prims.append(box((cx, cy, cz), half, (0.0, 0.0, rng.uniform(0, math.pi)), name))


def _bars_spheres(rng, cell, name, prims):
    """Thin horizontal bars spanning the cell at band height, plus spheres."""
    x0, y0, x1, y1 = cell
    for _ in range(5):
        z = rng.uniform(1.1, 2.4)
        r = rng.uniform(0.05, 0.15)
        if rng.random() < 0.5:
            y = rng.uniform(y0 + 1.0, y1 - 1.0)
            a, b = (x0, y, z), (x1, y, z)
        else:
            x = rng.uniform(x0 + 1.0, x1 - 1.0)
            a, b = (x, y0, z), (x, y1, z)
        prims.append(cyl_h(a, b, r, name))
    for cx, cy in _poisson_disc(rng, x0, y0, x1, y1, 3.0, margin=1.0)[:8]:
        prims.append(sphere((cx, cy, rng.uniform(1.0, 2.6)), rng.uniform(0.3, 0.7), name))


def _clutter(rng, cell, name, prims):
    """Pillars and tilted slabs, the diffaero flavour, denser than shipped."""
    x0, y0, x1, y1 = cell
    for cx, cy in _poisson_disc(rng, x0, y0, x1, y1, 2.6, margin=0.8):
        if rng.random() < 0.5:
            prims.append(cyl_v(cx, cy, rng.uniform(0.2, 0.5), 0.0, TALL, name))
        else:
            half = (rng.uniform(0.3, 0.6), rng.uniform(0.3, 0.6), rng.uniform(1.5, 2.5))
            rpy = (math.radians(rng.uniform(-15, 15)), math.radians(rng.uniform(-15, 15)),
                   rng.uniform(-math.pi, math.pi))
            prims.append(box((cx, cy, half[2]), half, rpy, name))


REGION_LAYOUT = {
    # column, row on the 3x3 tiling (0,0 is south-west)
    "open":        (0, 0),
    "forest_5m":   (1, 0),
    "forest_2p5m": (2, 0),
    "box_yard":    (0, 1),
    "wall_slot":   (1, 1),
    "chicane":     (2, 1),
    "room":        (0, 2),
    "bars":        (1, 2),
    "clutter":     (2, 2),
}


def generate(seed: int = 0, size: float = ARENA) -> Arena:
    rng = np.random.default_rng(seed)
    third = size / 3.0
    lo = -size / 2.0
    cells = {
        name: [lo + c * third, lo + r * third, lo + (c + 1) * third, lo + (r + 1) * third]
        for name, (c, r) in REGION_LAYOUT.items()
    }
    cells["perimeter"] = [lo, lo, -lo, -lo]
    prims: list[Primitive] = []
    _forest(rng, cells["forest_5m"], "forest_5m", 5.0, prims)
    _forest(rng, cells["forest_2p5m"], "forest_2p5m", 2.5, prims)
    _box_yard(rng, cells["box_yard"], "box_yard", prims)
    _wall_slot(rng, cells["wall_slot"], "wall_slot", prims)
    _chicane(rng, cells["chicane"], "chicane", prims)
    _room(rng, cells["room"], "room", prims)
    _bars_spheres(rng, cells["bars"], "bars", prims)
    _clutter(rng, cells["clutter"], "clutter", prims)
    _perimeter(size, prims)

    z = 0.5 * (FLIGHT_BAND[0] + FLIGHT_BAND[1])
    start = [lo + 2.0, lo + 2.0, z]
    scene = Arena(
        seed=seed, size=size, flight_band=FLIGHT_BAND, robot_radius=ROBOT_RADIUS,
        primitives=prims, regions=cells, start=start, goal=start,
    )
    scene.goal = _nominal_goal(scene, cells["clutter"], rng)
    return scene


def _perimeter(size, prims):
    """Closed arena: no detour and no time-to-arrival field can leave the map."""
    h = size / 2.0
    corners = [(-h, -h), (h, -h), (h, h), (-h, h)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        prims.append(wall(a, b, 0.0, TALL, 0.2, "perimeter"))


def _nominal_goal(scene, cell, rng, clearance=1.5, n=400):
    """A goal in the far cell that the endpoint gate accepts: draw candidates,
    keep those with ``clearance`` to every obstacle, take the one farthest
    from the start. Only the config's nominal pair; the pair sampler draws
    the real ones."""
    from scipy.spatial import cKDTree

    pts, ids = sample_surfaces(scene, spacing=0.2, floor_spacing=1.0)
    tree = cKDTree(pts[ids >= 0])
    x0, y0, x1, y1 = cell
    z = scene.start[2]
    cand = rng.uniform([x0 + 1, y0 + 1], [x1 - 1, y1 - 1], size=(n, 2))
    cand = np.column_stack([cand, np.full(n, z)])
    d, _ = tree.query(cand)
    ok = cand[d >= clearance]
    if len(ok) == 0:
        ok = cand[[int(np.argmax(d))]]
    far = np.linalg.norm(ok[:, :2] - np.asarray(scene.start[:2]), axis=1)
    return [float(v) for v in ok[int(np.argmax(far))]]


def _window_wall(rng, cell, name, prims, width=None, height=None, sill=None):
    """A wall across the whole cell with one opening at flight height."""
    x0, y0, x1, y1 = cell
    width = rng.uniform(1.0, 1.8) if width is None else width
    height = rng.uniform(0.8, 1.4) if height is None else height
    sill = rng.uniform(0.9, 1.5) if sill is None else sill
    length = y1 - y0
    xm = 0.5 * (x0 + x1)
    at = rng.uniform(0.25, 0.75) * length
    prims.append(wall((xm, y0), (xm, y1), 0.0, 4.0, 0.25, name,
                      slots=[[at, width, sill, sill + height]]))


def _ring_cell(rng, cell, name, prims, radius=None, gap_deg=None, n=40):
    """A cylindrical shell of thin pillars, one gap (or none) at a random bearing."""
    x0, y0, x1, y1 = cell
    radius = rng.uniform(2.0, 3.5) if radius is None else radius
    gap_deg = rng.choice([0.0, 50.0, 70.0, 90.0]) if gap_deg is None else gap_deg
    gap_at = rng.uniform(0.0, 360.0)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    for k in range(n):
        ang = 360.0 * k / n
        if gap_deg > 0 and abs((ang - gap_at + 180.0) % 360.0 - 180.0) < gap_deg / 2:
            continue
        a = math.radians(ang)
        prims.append(cyl_v(cx + radius * math.cos(a), cy + radius * math.sin(a), 0.18, 0.0, 3.5, name))


def _gate(rng, cell, name, prims):
    """A beam across the cell at head height, and posts: pass under, between."""
    x0, y0, x1, y1 = cell
    z_under = rng.uniform(1.2, 1.7)
    xm = 0.5 * (x0 + x1)
    prims.append(box((xm, 0.5 * (y0 + y1), z_under + 1.2), (0.3, 0.5 * (y1 - y0), 1.2), (0, 0, 0), name))
    for cy in np.linspace(y0 + 1.5, y1 - 1.5, int(rng.integers(2, 5))):
        prims.append(cyl_v(xm, cy, 0.15, 0.0, TALL, name))


#: Family -> (builder, draw weight). Builders take (rng, cell, name, prims).
FAMILIES = {
    "open":       (lambda rng, cell, name, prims: None, 1.5),
    "forest":     (lambda rng, cell, name, prims: _forest(rng, cell, name, rng.uniform(2.2, 6.0), prims), 2.2),
    "wall_slot":  (_wall_slot, 1.0),
    "window":     (_window_wall, 0.8),
    "chicane":    (_chicane, 0.8),
    "room":       (_room, 0.6),
    "box_yard":   (_box_yard, 1.0),
    "bars":       (_bars_spheres, 0.7),
    "clutter":    (_clutter, 1.0),
    "ring":       (_ring_cell, 0.5),
    "gate":       (_gate, 0.5),
}


def generate_random(seed: int, size: float | None = None, cell_target: float = 13.0,
                    nominal_goal: bool = True, families: list[str] | None = None) -> Arena:
    """A random training map: random side, random tiling, a family drawn per
    cell by weight, each family with its own random parameters. One seed, one
    map. ``families`` restricts the draw to a subset (weights renormalised)."""
    rng = np.random.default_rng(seed)
    size = float(rng.uniform(30.0, 60.0)) if size is None else float(size)
    n = max(2, int(round(size / cell_target)))
    cell = size / n
    lo = -size / 2.0
    names = list(FAMILIES) if families is None else list(families)
    unknown = set(names) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families {sorted(unknown)}; known: {sorted(FAMILIES)}")
    weights = np.array([FAMILIES[k][1] for k in names]); weights /= weights.sum()
    prims: list[Primitive] = []
    regions: dict[str, list[float]] = {}
    families: dict[str, str] = {}
    for c in range(n):
        for r in range(n):
            fam = names[rng.choice(len(names), p=weights)]
            name = f"{fam}_{c}{r}"
            box_ = [lo + c * cell, lo + r * cell, lo + (c + 1) * cell, lo + (r + 1) * cell]
            regions[name] = box_
            families[name] = fam
            FAMILIES[fam][0](rng, box_, name, prims)
    _perimeter(size, prims)
    regions["perimeter"] = [lo, lo, -lo, -lo]
    families["perimeter"] = "perimeter"

    z = 0.5 * (FLIGHT_BAND[0] + FLIGHT_BAND[1])
    start = [lo + 2.0, lo + 2.0, z]
    scene = Arena(seed=seed, size=size, flight_band=FLIGHT_BAND, robot_radius=ROBOT_RADIUS,
                  primitives=prims, regions=regions, start=start, goal=start, families=families)
    if nominal_goal:
        far = [v for k, v in regions.items() if k != "perimeter"]
        scene.goal = _nominal_goal(scene, far[-1], rng)
    return scene


# --- surface sampling ----------------------------------------------------------

def _euler_xyz(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rx @ ry @ rz


def _sphere_pts(c, r, h):
    n = max(int(4.0 * math.pi * r * r / (h * h)), 32)
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    unit = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], -1)
    return np.asarray(c) + r * unit


def _box_pts(c, half, rpy, h):
    half = np.asarray(half, float)
    faces = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        gu, gv = np.meshgrid(
            np.linspace(-half[u], half[u], max(int(2 * half[u] / h) + 1, 2)),
            np.linspace(-half[v], half[v], max(int(2 * half[v] / h) + 1, 2)),
        )
        for sign in (-1.0, 1.0):
            f = np.zeros((gu.size, 3))
            f[:, axis] = sign * half[axis]
            f[:, u] = gu.ravel()
            f[:, v] = gv.ravel()
            faces.append(f)
    pts = np.concatenate(faces)
    if any(rpy):
        pts = pts @ _euler_xyz(*rpy).T
    return np.asarray(c) + pts


def _cyl_pts(a, b, r, h, caps=True):
    """Cylinder surface between points a and b, with end caps."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    axis /= length
    helper = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(axis, helper); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    n_t = max(int(2 * math.pi * r / h), 8)
    n_l = max(int(length / h) + 1, 2)
    theta = np.linspace(0, 2 * math.pi, n_t, endpoint=False)
    ring = np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v)      # (n_t, 3)
    t = np.linspace(0, length, n_l)
    side = (a + np.outer(t, axis))[:, None, :] + r * ring[None, :, :]
    out = [side.reshape(-1, 3)]
    if caps:
        for centre in (a, b):
            for rr in np.arange(h, r, h):
                out.append(centre + rr * ring)
    return np.concatenate(out)


def _wall_pts(p, h):
    """Solid-grid slab (so coarse queries cannot tunnel), slots carved out."""
    p0, p1 = np.asarray(p["p0"], float), np.asarray(p["p1"], float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    d /= length
    n = np.array([-d[1], d[0]])
    s = np.arange(0.0, length + h / 2, h)
    tt = np.linspace(-p["t"] / 2, p["t"] / 2, max(int(p["t"] / h) + 1, 2))
    z = np.arange(p["z0"], p["z1"] + h / 2, h)
    S, T, Z = np.meshgrid(s, tt, z, indexing="ij")
    keep = np.ones(S.shape, bool)
    for s0, w, z0, z1 in p["slots"]:
        keep &= ~((S > s0 - w / 2) & (S < s0 + w / 2) & (Z >= z0) & (Z <= z1))
    S, T, Z = S[keep], T[keep], Z[keep]
    xy = p0[None, :] + S[:, None] * d[None, :] + T[:, None] * n[None, :]
    return np.column_stack([xy, Z])


def _floor_pts(size, margin, h):
    lo, hi = -size / 2 - margin, size / 2 + margin
    xs = np.arange(lo, hi + h / 2, h)
    X, Y = np.meshgrid(xs, xs)
    return np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])


def sample_surfaces(arena: Arena, spacing: float = 0.1, floor_spacing: float = 0.25,
                    floor_margin: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Surface points of every primitive plus the floor.

    Returns ``(points, region_id)``; region ids index ``sorted(arena.regions)``
    with ``-1`` for the floor.
    """
    names = sorted(arena.regions)
    clouds, ids = [], []
    for p in arena.primitives:
        k = p["kind"]
        if k == "sphere":
            pts = _sphere_pts(p["c"], p["r"], spacing)
        elif k == "box":
            pts = _box_pts(p["c"], p["half"], p["rpy"], spacing)
        elif k == "cyl_v":
            cx, cy = p["c"]
            pts = _cyl_pts((cx, cy, p["z0"]), (cx, cy, p["z1"]), p["r"], spacing)
        elif k == "cyl_h":
            pts = _cyl_pts(p["a"], p["b"], p["r"], spacing)
        elif k == "wall":
            pts = _wall_pts(p, spacing)
        else:
            raise ValueError(f"unknown primitive kind {k!r}")
        clouds.append(pts)
        ids.append(np.full(len(pts), names.index(p["region"]), np.int16))
    floor = _floor_pts(arena.size, floor_margin, floor_spacing)
    clouds.append(floor)
    ids.append(np.full(len(floor), -1, np.int16))
    return np.concatenate(clouds), np.concatenate(ids)
