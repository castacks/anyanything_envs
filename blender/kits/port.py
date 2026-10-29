"""Container terminal: stacked shipping containers in blocks and rows, the
patch's obstacles among them, on a stained concrete yard with painted lanes.

What makes it read as a port rather than boxes on a plane:

* containers are ISO sizes (20 ft 6.06 m, 40 ft 12.19 m, 2.44 m wide,
  2.59 m / 2.90 m high-cube), corrugated steel tinted to the livery colours
  a terminal actually shows, with a darker corner frame, and stacked in
  exact columns 1-3 high with the odd one set askew;
* the yard is organised: container blocks line both long sides of the patch
  (the canyon a drone flies down), straddle lanes between them are painted,
  and inside the patch the stacks keep to the lane grid;
* far scenery (render-only, beyond the cloud zone) repeats the blocks and
  adds 30 m light masts, so the skyline is a terminal, not the HDRI's town;
* forklifts, trucks and loaders (curated Objaverse models) stand on the
  lane grid, barrels, gas bottles, generators and skips round the yard.

Everything in the cloud zone collides and every container is closed to a
drone (``Yard.stack``), so the labeller never finds free space where the
render shows steel; the faces nobody can see or reach are left out, because
the cloud samples every face it is given and a yard's cloud is mostly
containers: the floor on the ground, the seam between two containers stacked
exactly (a frame where one is set askew), a wall row's back against the row
0.35 m behind it.

Levels: ``clutter`` scales the in-patch stacks (at most 12), vehicles and
yard clutter; ``hang`` > 0 sags reefer cables from 2.3-3 m between the
facing sides of two stacks.
"""

from __future__ import annotations

import math

import numpy as np

import build_course as BC
from kits import _street_util as U

NAME = "port"
INDOOR = False
DESCRIPTION = "container terminal: stacked ISO containers in blocks, painted lanes, light masts"

LIVERIES = [(0.05, 0.22, 0.45), (0.55, 0.08, 0.06), (0.10, 0.35, 0.18), (0.62, 0.62, 0.60), (0.85, 0.42, 0.06),
            (0.75, 0.72, 0.62), (0.30, 0.16, 0.10), (0.08, 0.30, 0.42), (0.55, 0.50, 0.12), (0.72, 0.72, 0.72)]
W_C, H_STD, H_HC = 2.438, 2.591, 2.896
L_20, L_40 = 6.058, 12.192


#: corrugated steel scans: grey/white ones take a livery tint, painted ones are their own livery
#: (a tint on a painted scan multiplies two colours: blue on green came out black)
STEEL_NEUTRAL = ["acg:CorrugatedSteel003", "acg:CorrugatedSteel005", "acg:CorrugatedSteel007B",
                 "acg:CorrugatedSteel009"]
STEEL_PAINTED = ["acg:CorrugatedSteel001", "acg:CorrugatedSteel002", "acg:CorrugatedSteel004",
                 "acg:CorrugatedSteel006A", "acg:CorrugatedSteel006B", "acg:CorrugatedSteel007A",
                 "acg:CorrugatedSteel007C", "acg:CorrugatedSteel008A", "acg:CorrugatedSteel008C"]
#: grey yard concrete (the library's 'concrete' also has beige, mossy and patterned wall panels)
YARD_CONCRETE = ["acg:Concrete006", "acg:Concrete010", "acg:Concrete014", "acg:Concrete015", "acg:Concrete016",
                 "acg:Concrete017", "acg:Concrete019", "acg:Concrete020", "acg:Concrete021", "acg:Concrete022",
                 "acg:Concrete023", "acg:Concrete024", "acg:Concrete027", "acg:Concrete030", "acg:Concrete035",
                 "acg:Concrete036", "acg:Concrete037", "acg:Concrete042A", "acg:Concrete044A", "acg:Concrete044C"]


def skin(ctx, rgb=None):
    """A container / cladding skin: a neutral scan in livery ``rgb`` (random if
    None) or, a third of the time, a painted scan as it is."""
    rng, lk = ctx.rng, ctx.looks
    neutral = [k for k in STEEL_NEUTRAL if k in ctx.lib.items]
    painted = [k for k in STEEL_PAINTED if k in ctx.lib.items]
    if painted and (rng.random() < 0.35 or not neutral):
        return lk.pbr(ctx.lib.pick(rng, painted), 1.2, bevel=0.02, dirt=0.5, jitter=0.08, vary=0.15)
    rgb = LIVERIES[int(rng.integers(len(LIVERIES)))] if rgb is None else rgb
    return lk.pbr(ctx.lib.pick(rng, neutral), 1.2, tint=[min(1.0, 1.3 * c + 0.05) for c in rgb], bevel=0.02,
                  dirt=0.5, jitter=0.08, vary=0.15)


#: a container set askew on the one below turns and slides by at most this (rad, m): twistlocks hold it
TILT_MAX, OFF_MAX = 0.03, 0.08
#: under a container set askew the roof below and its floor are frames this wide: wider than the most it
#: can hang over or be set in (TILT_MAX * 6.22 m + OFF_MAX * 2 ** 0.5 = 0.30 m), so the seam is closed
#: without a whole plane of cloud points inside the stack
SEAM_RING = 0.45


def body(hx, hy, h, floor=None, roof=math.inf, back=0):
    """A container body in local coordinates (x along its length, foot at z =
    0): corners 0 (-,-) 1 (+,-) 2 (-,+) 3 (+,+) at the foot, 4-7 the same at
    ``h``; the four sides, and a floor / roof that is None (none), a frame
    that wide, or whole (inf, or any width past the middle). ``back`` = +1/-1
    leaves out the long side at local +y/-y. Returns (verts, quads)."""
    V = [(sx * hx, sy * hy, z) for z in (0.0, h) for sy in (-1, 1) for sx in (-1, 1)]
    F = [(0, 1, 5, 4)] * (back >= 0) + [(1, 3, 7, 5)] + [(3, 2, 6, 7)] * (back <= 0) + [(2, 0, 4, 6)]
    for w, loop, z in ((floor, (0, 2, 3, 1), 0.0), (roof, (4, 5, 7, 6), h)):
        if w is None:
            continue
        if w >= min(hx, hy):
            F.append(loop)
            continue
        n = len(V)
        V += [(math.copysign(hx - w, V[i][0]), math.copysign(hy - w, V[i][1]), z) for i in loop]
        F += [(loop[k], loop[(k + 1) % 4], n + (k + 1) % 4, n + k) for k in range(4)]
    return V, F


class Yard:
    def __init__(self, ctx):
        self.ctx = ctx
        lk = ctx.looks
        self.skins = [skin(ctx, rgb) for rgb in LIVERIES]
        self.frame = lk.pbr(U.pick_key(ctx, U.PAINTED_METAL), 1.0, tint=(0.25, 0.25, 0.25), bevel=0.01, dirt=0.4)
        self.stacks = []           # colliding stacks in the zone (hang anchors)
        self.cable_z = []          # (lowest, highest) z of each cable
        self.n_askew = 0

    def container(self, pose, length, high, coll="structure", collide=None, floor=None, roof=math.inf, back=0):
        """One container at ``pose`` = (x, y, z0, yaw): its body (``body``)
        + four corner posts slightly proud of it (their two outer faces; the
        inner ones are inside the body). Returns its height."""
        ctx = self.ctx
        h = H_HC if high else H_STD
        skin = self.skins[int(ctx.rng.integers(len(self.skins)))]
        x, y, z0, yaw = pose
        cy, sy = math.cos(yaw), math.sin(yaw)
        R = np.array([[cy, -sy], [sy, cy]])
        Vl, F = body(length / 2, W_C / 2, h, floor=floor, roof=roof, back=back)
        V = [(*(np.array([x, y]) + R @ np.array(v[:2])), z0 + v[2]) for v in Vl]
        ctx.mesh(V, F, skin, coll=coll, collide=collide, name="container")
        for u in (-1, 1):
            for v in (-1, 1):
                px = x + cy * u * (length / 2 - 0.08) - sy * v * (W_C / 2 - 0.08)
                py = y + sy * u * (length / 2 - 0.08) + cy * v * (W_C / 2 - 0.08)
                P = [(*(np.array([px, py]) + R @ np.array(q[:2])), z0 + q[2])
                     for q in body(0.09, 0.09, h + 0.005)[0][:8]]
                outer = [(1, 3, 7, 5) if u > 0 else (2, 0, 4, 6), (3, 2, 6, 7) if v > 0 else (0, 1, 5, 4)]
                ctx.mesh(P, outer, self.frame, coll=coll, collide=collide, name="cpost")
        return h

    def stack(self, xy, yaw, length, levels, coll="structure", claim=True, collide=None, askew=0.35, back=0,
              back_below=0.0):
        """``levels`` containers (a count, or a list of high-cube flags) on one
        footprint, each above the first set askew on the one below with
        probability ``askew`` (turned / slid by at most TILT_MAX / OFF_MAX),
        else exactly on it. Closed: no floor on the ground; between two
        containers nothing where they meet exactly (their sides join), a
        SEAM_RING frame each side where one is askew. ``back`` leaves out
        the long side facing local +y/-y on each container whose top is at
        most ``back_below`` (a wall row's back against the row behind it)."""
        ctx = self.ctx
        highs = [bool(ctx.rng.random() < 0.35) for _ in range(levels)] if isinstance(levels, int) else list(levels)
        skew = [False] + [bool(ctx.rng.random() < askew) for _ in highs[1:]]
        pose = [float(xy[0]), float(xy[1]), 0.0, float(yaw)]
        for k, high in enumerate(highs):
            if skew[k]:
                d = np.clip(ctx.rng.normal(0, 0.04, 2), -OFF_MAX, OFF_MAX)
                pose = [pose[0] + d[0], pose[1] + d[1], pose[2],
                        pose[3] + float(np.clip(ctx.rng.normal(0, 0.02), -TILT_MAX, TILT_MAX))]
                self.n_askew += 1
            top = pose[2] + (H_HC if high else H_STD)
            h = self.container(pose, length, high, coll=coll, collide=collide,
                               floor=None if k == 0 else SEAM_RING if skew[k] else None,
                               roof=math.inf if k == len(highs) - 1 else SEAM_RING if skew[k + 1] else None,
                               back=back if top <= back_below + 1e-6 else 0)
            pose[2] += h
        z = pose[2]
        if claim:
            # (a container askew on the first one hangs over it by at most 0.30 m)
            ctx.claim_rect(xy, (length / 2 + 0.35, W_C / 2 + 0.35), yaw, kind="stack")
            if ctx.in_zone(xy) and collide is not False:
                self.stacks.append({"xy": np.asarray(xy, float), "yaw": yaw, "length": length, "h": z,
                                    "foot": ctx.foot[-1]})
        return z

    def anchor(self, st, toward):
        """The point on stack ``st``'s side nearest ``toward`` (xy)."""
        cy, sy = math.cos(st["yaw"]), math.sin(st["yaw"])
        R = np.array([[cy, -sy], [sy, cy]])
        q = R.T @ (np.asarray(toward, float) - st["xy"])
        h = np.array([st["length"] / 2, W_C / 2])
        q = np.clip(q, -h, h)
        i = int(np.argmax(np.abs(q) / h))        # push onto the nearest side
        q[i] = math.copysign(h[i], q[i] if q[i] else 1.0)
        return st["xy"] + R @ q


def build(ctx) -> dict:
    rng, lk = ctx.rng, ctx.looks
    x0, y0, x1, y1 = ctx.bounds
    sky = lk.sky(ctx.lib.pick(rng, ctx.lib.hdris(cats_all={"outdoor"}, cats_none={"night", "indoor"},
                                                 tags_any={"industrial", "harbour", "dock", "port", "parking",
                                                           "concrete", "asphalt", "road", "warehouse", "city"})))
    # ground: stained concrete slabs + asphalt patches, puddles after rain some days
    conc = [k for k in YARD_CONCRETE + U.CONCRETE_GROUND if k in ctx.lib.items]
    keys = [ctx.lib.pick(rng, conc), U.pick_key(ctx, U.ASPHALT), ctx.lib.pick(rng, conc)]
    wet = float(rng.choice([0.0, 0.0, 0.3, 0.6]))
    ctx.ground(lk.ground_material(keys, tile_m=3.5, puddles=wet, blend_scale=14.0))
    yellow = lk.paint((0.75, 0.55, 0.05), rough=0.6, name="lane_yellow")
    white = lk.paint((0.8, 0.8, 0.78), rough=0.6, name="lane_white")

    yard = Yard(ctx)
    wall = skin(ctx)
    concrete = lk.pbr(ctx.lib.pick(rng, conc), 2.0, bevel=0.02, dirt=0.5)
    # posts and bars: painted steel in safety yellow or grey (the library's 'metal' has brass and gold)
    steel = lk.pbr(U.pick_key(ctx, U.PAINTED_METAL), 1.0, bevel=0.01, dirt=0.3,
                   tint=[(0.9, 0.75, 0.15), (0.55, 0.56, 0.58), (0.85, 0.35, 0.08)][int(rng.integers(3))])
    ctx.dress_primitives({"wall": wall, "box": wall, "pole": steel, "bar": steel, "rock": concrete})

    # container blocks along both long sides, just outside the patch (in the cloud zone)
    gaps, reach, low = 0, 0, 0
    for side in (-1, 1):
        yb = (y0 - 1.6 - W_C / 2) if side < 0 else (y1 + 1.6 + W_C / 2)
        x = x0 - 6.0
        while x < x1 + 6.0:
            length = L_40 if rng.random() < 0.6 else L_20
            if rng.random() < 0.12:            # a gap in the wall now and then
                x += length + 0.4
                gaps += 1
                continue
            levels = int(rng.choice([1, 2, 2, 3, 3]))
            if not ctx.free_rect((x + length / 2, yb), (length / 2, W_C / 2), 0.0, margin=0.3):
                x += length + 0.4               # a patch obstacle reaches out here
                reach += 1
                continue
            # a second row 0.35 m behind it (in the cloud as far as it stands in the zone); both rows
            # stacked exactly, so the slot between them is 0.35 m everywhere and the front row needs no
            # back where the row behind covers it: nothing a drone can get into
            back_row = [bool(rng.random() < 0.35) for _ in range(int(rng.choice([2, 3, 3])))]
            yard.stack((x + length / 2, yb), 0.0, length, levels, askew=0.0, back=side,
                       back_below=sum(H_HC if hc else H_STD for hc in back_row))
            yard.stack((x + length / 2, yb + side * (W_C + 0.35)), 0.0, length, back_row, askew=0.0)
            low += levels == 1
            x += length + 0.35
        # lane markings along the block
        ctx.strip((x0 - 6, yb - side * (W_C / 2 + 0.6)), (x1 + 6, yb - side * (W_C / 2 + 0.6)), 0.15, yellow)

    # in-patch stacks on the lane grid, clear of the obstacle, start and goal (at most 12: an arena-sized
    # patch would otherwise take ~30 and its cloud 0.8 M points)
    n = min(ctx.count(7.0), 12)
    placed = 0
    for _ in range(n * 40):
        if placed >= n:
            break
        length = L_20 if rng.random() < 0.6 else L_40
        yaw = 0.0 if rng.random() < 0.75 else math.pi / 2
        xy = ctx.uniform_xy(pad=2.0)
        half = (length / 2 + 0.2, W_C / 2 + 0.2)
        if not ctx.free_rect(xy, half, yaw, margin=float(ctx.knobs.get("keepout", 2.5)) * 0.5 + 0.6):
            continue
        yard.stack(xy, yaw, length, int(rng.choice([1, 1, 2, 2, 3])))
        placed += 1

    # yard vehicles on the lane grid: forklifts, trucks and loaders (real models, Objaverse)
    vehicles = {"forklift": U.obj_pieces(ctx, U.OBJ_FORKLIFT), "truck": U.obj_pieces(ctx, U.OBJ_TRUCK),
                "loader": U.obj_pieces(ctx, U.OBJ_LOADER)}
    n_veh = {}
    for kind, per, cap in (("forklift", 1.5, 3), ("truck", 0.5, 1), ("loader", 0.3, 1)):
        ks, n_veh[kind], want = vehicles[kind], 0, U.draw_count(ctx, per, cap)
        for _ in range(want * 30 if ks else 0):
            if n_veh[kind] >= want:
                break
            k = ks[int(rng.integers(len(ks)))]
            y0_, L, Wd = U.long_yaw(ctx, k)
            yaw = (0.0 if rng.random() < 0.7 else math.pi / 2) + (math.pi if rng.random() < 0.5 else 0.0)
            xy = ctx.uniform_xy(pad=2.0)
            half = (L / 2 + 0.1, Wd / 2 + 0.1)
            if not ctx.free_rect(xy, half, yaw, margin=float(ctx.knobs.get("keepout", 2.5)) * 0.3 + 0.5):
                continue
            ctx.instance(k, xy, yaw=yaw + y0_ + float(rng.normal(0, 0.05)))
            ctx.claim_rect(xy, half, yaw, kind="vehicle")
            n_veh[kind] += 1

    # yard clutter: pallets, barrels, cones, bollards
    small = ctx.lib.models(ids=["Barrel_01", "Barrel_02", "barrel_03", "wooden_crate_02", "plastic_crate_01",
                                "industrial_pastic_container", "concrete_road_barrier", "concrete_road_barrier_02",
                                "metal_trash_can", "old_military_crate", "wooden_military_crate", "hand_truck",
                                "small_lpg_tank", "portable_generator", "utility_box_02", "old_tyre"]) + \
        U.obj_pieces(ctx, U.OBJ_BARREL + U.OBJ_GAS + U.OBJ_GENERATOR + U.OBJ_DUMPSTER)
    ctx.scatter(small, min(ctx.count(8.0), 16), margin=0.5)     # (caps: an arena is 4x a tier-A patch)
    bollard = lk.pbr(U.pick_key(ctx, U.PAINTED_METAL), 0.6,
                     tint=(0.85, 0.7, 0.1), bevel=0.02, dirt=0.4)
    for _ in range(min(ctx.count(4.0), 8)):
        xy = ctx.uniform_xy(pad=1.0)
        if ctx.free(xy, 0.3, 0.6):
            ctx.cylinder((xy[0], xy[1], 0.0), (xy[0], xy[1], 0.9), 0.16, bollard, coll="scatter", name="bollard",
                         claim=True)

    # reefer cables sagging between neighbouring stacks at hard levels, 2-3 m up
    hung = hang_cables(ctx, yard, ctx.count(float(ctx.knobs.get("hang", 0.0))))

    # scenery beyond the zone: more blocks, light masts
    zx0, zy0, zx1, zy1 = ctx.zone
    mast = lk.pbr(U.pick_key(ctx, U.GALV), 1.0, bevel=0.02)
    lamp = lk.paint((1.0, 0.95, 0.85), rough=0.3, name="mast_lamp", emission=(1.0, 0.9, 0.75, 3.0))
    for side in (-1, 1):
        yb = (zy0 - 8.0) if side < 0 else (zy1 + 8.0)
        x = zx0 - 30.0
        while x < zx1 + 30.0:
            length = L_40 if rng.random() < 0.7 else L_20
            for r in range(int(rng.integers(2, 5))):
                yard.stack((x + length / 2, yb + side * r * (W_C + 0.35)), 0.0, length, int(rng.integers(2, 5)),
                           coll="backdrop", claim=False)
            x += length + float(rng.choice([0.35, 0.35, 4.0]))
        for xm in np.arange(zx0 - 20, zx1 + 20, 45.0):
            ym = yb - side * 3.0
            ctx.cylinder((xm, ym, 0.0), (xm, ym, 30.0), 0.35, mast, coll="backdrop", name="mast", collide=False)
            ctx.box((xm, ym, 30.3), (1.6, 0.5, 0.3), lamp, coll="backdrop", collide=False, name="mast_head")
    for xe in (zx0 - 12.0, zx1 + 12.0):
        yy = zy0 - 20.0
        while yy < zy1 + 20.0:
            length = L_40
            yard.stack((xe, yy + length / 2), math.pi / 2, length, int(rng.integers(1, 4)), coll="backdrop",
                       claim=False)
            yy += length + 0.35
    return {"sky": sky, "ground": keys, "wet": wet, "stacks_in_patch": placed, "vehicles": n_veh, "hang_cables": hung,
            "wall_gaps": gaps, "wall_gaps_obstacle": reach, "wall_low_stacks": low,
            "semantics_changed": [f"container walls 1.6 m outside both long sides of the patch, 2.6-8.7 m tall: "
                                  f"out of the patch only through {gaps} gaps, {reach} more where a patch obstacle "
                                  f"runs out, and over {low} one-high stacks"] +
            ([f"{placed} container stacks inside the patch"] if placed else []) +
            ([f"{hung} reefer cables between stacks, {min(z for z, _ in yard.cable_z):.2f}-"
              f"{max(z for _, z in yard.cable_z):.2f} m up"] if hung else [])}


def hang_cables(ctx, yard, n: int) -> int:
    """``n`` cables sagging between two stacks 3-14 m apart (anchored on
    their facing sides, both stacks taller than the cable), clear of the
    patch obstacles, their openings and the start/goal by 0.4 m in plan."""
    rng, lk = ctx.rng, ctx.looks
    cable = lk.paint((0.05, 0.05, 0.05), rough=0.5, name="cable")
    sts = yard.stacks
    made = 0
    for _ in range(n * 30 if len(sts) >= 2 else 0):
        if made >= n:
            break
        i, j = (int(v) for v in rng.choice(len(sts), 2, replace=False))
        A, B = sts[i], sts[j]
        if not 3.0 <= float(np.linalg.norm(A["xy"] - B["xy"])) <= 14.0:
            continue
        a2, b2 = yard.anchor(A, B["xy"]), yard.anchor(B, A["xy"])
        z = float(rng.uniform(2.3, 3.0))
        if min(A["h"], B["h"]) < z + 0.3 or np.linalg.norm(b2 - a2) < 2.0:
            continue
        others = [f for f in ctx.foot if f is not A["foot"] and f is not B["foot"]]
        if not all(BC.footprint_distance(q, f) >= 0.4 for q in np.linspace(a2, b2, 12)[1:-1] for f in others):
            continue
        zb, sag = z + float(rng.normal(0, 0.1)), float(rng.uniform(0.15, 0.5))
        U.catenary(ctx, (*a2, z), (*b2, zb), sag, 0.02, cable, name="cable", n=6)
        yard.cable_z.append((min(z, zb) - sag - 0.02, max(z, zb) + 0.02))     # (a bound: the sag is at mid-span)
        made += 1
    return made
