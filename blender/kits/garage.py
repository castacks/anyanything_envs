"""Parking garage level: a concrete deck under a concrete ceiling, a grid of
columns, painted bays with parked cars, fluorescent battens, sprinkler and
duct runs under the soffit; open sides onto a street with daylight, or an
enclosed underground level lit only by its lamps.

What makes it read as a garage rather than boxes under a lid:

* the layout is a real parking module: a 6-7 m drive aisle, 90-degree bays
  (2.4-2.7 m x 4.8-5.3 m) nose-in on both sides, columns in the back-to-back
  bay line every 2-4 bays, repeated across the building;
* the ceiling is one of three structural systems -- flat plate, downstand
  girders, or precast double tees (stems every 1.2 m) -- in raw or painted
  concrete, with lamps, sprinkler mains/branches, conduit and ducts under it;
* cars are procedural but proportioned (hatch/sedan/wagon/SUV/van/pickup
  with wheel arches, glass house, clear-coat paint in the colours car parks
  actually show), plus the odd car under a cover;
* the floor is worn concrete with saw-cut joints, oil stains under the bays,
  tyre marks, worn paint: bay lines, arrows, zebra crossings, hatching;
* open sides have a parapet, spandrel and a street outside (sidewalk,
  buildings across, street lamps); underground levels have walls, doors and
  exit signs and a greener or warmer lamp tint.

The ceiling (and anything under it that does not hang on purpose) keeps its
underside at or above ``max(3.8, tallest primitive + 0.3)`` (<= 4.45 m); a
patch with primitives over ~4.15 m gets a ceiling at their top (>= 4.55 m)
so tall poles/walls read as columns and shear walls. Whether that caps the
cloud depends on the cloud top: a flat plate at <= 4.45 m does, girders and
double tees cap only where their slab is under it, and a ceiling at a 5 m
wall top leaves the cloud open over the patch (the walls cut at the top,
as in any outdoor kit). At L3 (``hang``) clearance bars, signs, sagging
conduit and pipe drops hang into the flight band.

The labeller's side of it (``_garage_util``): the family's own free space
(wall ends, round a pillar, a ring, a grove, ...) is claimed before any
column or car goes in; cars thin out evenly when the primitives and the
structure leave little of the cloud budget (an arena); and what the dressing
does to the tier-A family -- a ceiling inside the cloud caps the flight band
and closes the over-the-top route, wall tops are closed up to it, floating
primitives hang on bridles -- is measured on the built geometry
(``U.overhead``: capped only where the cloud really holds a ceiling over the
whole patch) and written to ``semantics_changed``. A few bays hold real
(Objaverse, CC-BY) cars, and two-wheelers stand by the column lines. The
cars and every Objaverse choice draw from their own generators and the
two-wheelers go in last, so the level itself (structure, materials, lamps,
which bays hold a car) does not depend on the Objaverse catalogue; small
props after the cars can land elsewhere where a real car's footprint differs.
"""

from __future__ import annotations

import math

import numpy as np

from kits import _garage_util as U

NAME = "garage"
INDOOR = True
DESCRIPTION = "parking garage level: columns, painted bays, parked cars, lamps and services under a concrete ceiling"

FLOORS = ["ph:garage_floor", "ph:concrete_floor_02", "ph:concrete_floor_worn_001", "ph:hangar_concrete_floor",
          "ph:concrete_floor_01", "ph:brushed_concrete", "ph:brushed_concrete_2", "ph:dirty_concrete",
          "acg:Concrete006", "acg:Concrete015", "acg:Concrete023", "acg:Concrete024", "acg:Concrete031",
          "acg:Concrete044A", "acg:Concrete043A"]
CEILINGS = ["ph:concrete_slab_wall", "ph:concrete_wall_001", "ph:concrete_wall_004", "ph:concrete_wall_008",
            "acg:Concrete002", "acg:Concrete004", "acg:Concrete013", "acg:Concrete016", "acg:Concrete034",
            "ph:concrete", "ph:concrete_panels", "ph:preconcrete_wall_001"]
WALLS = ["ph:concrete_brick_wall_001", "ph:concrete_slab_wall", "ph:concrete_slab_wall_02", "ph:concrete_wall_001",
         "ph:concrete_wall_004", "ph:concrete_wall_008", "acg:Concrete013", "acg:Concrete002", "acg:Concrete004",
         "ph:painted_concrete_02", "acg:PaintedBricks001", "acg:PaintedBricks002", "ph:factory_brick",
         "ph:brick_wall_08"]
STREET = ["ph:asphalt_02", "ph:asphalt_05", "ph:clean_asphalt", "ph:asphalt_floor", "acg:Asphalt012",
          "acg:Asphalt010", "acg:Asphalt026B"]
SIDEWALK = ["ph:concrete_pavement", "ph:concrete_tiles", "ph:concrete_pavers", "ph:floor_pavement",
            "ph:concrete_tiles_02", "ph:pavement_02"]
#: Objaverse cars for some bays (CC-BY, recorded per course in ``files``): contemporary everyday
#: models checked on the catalogue sheet for scale (length, width, height within ~15 % of the real
#: car) and look, <= 90k triangles, <= 16 textures; no classic/vintage, film, game, livery or show
#: cars (the 09-25 list had a Beetle, a DB5, a 1975 911, a PUBG coupe, ...), no scan scaled up
#: from a city car (Autolib' +17 %)
OBJ_CARS = ["obj:bec38f586d9349ee9c1f295b6f5569cb",      # Mitsubishi Pajero 4WD
            "obj:c592bf582a1b430daaef753418fab22f",      # Honda Integra Type-R DB8
            "obj:192011ef37994e8ca2f09cd1d73ca166",      # 2000 Jaguar XJ (X308), ~12 % short
            "obj:e0bb09118ab84fe0a0530bd0df2fd74a",      # Range Rover Evoque
            "obj:e398ad20e83d42e581ab43438d5b6e49",      # Nissan Silvia S13
            "obj:dade78dc96e34f1a8cbcf14dd47d84de"]      # Ford Ranger (double cab pickup)
#: at most this many of one Objaverse car model a scene (one model must not fill the level)
OBJ_REPEAT = 2
#: two-wheelers parked against walls / in odd bays
OBJ_BIKES = ["obj:5e3ba0954ffc4a8f853c86cb7fdaa91b", "obj:f623ab40922944f881caf38d057e5d9d",
             "obj:d8d0022e70d048b4a87f7248655f73bf", "obj:b0d453e026b24d8496b14a2fa046f457",
             "obj:197dd882a69846e0825e80566996cbba", "obj:3bde1ab67c6448169b604dedad6828d8",
             "obj:e559f6bccecf44a2b73e1da8ffce6b66", "obj:f256203558c24ea09385a861b917c75a"]
#: level colour codes (columns bands, signs)
LEVEL_COLOURS = [(0.05, 0.25, 0.65), (0.1, 0.45, 0.15), (0.75, 0.12, 0.08), (0.85, 0.6, 0.05), (0.45, 0.1, 0.5),
                 (0.05, 0.45, 0.55), (0.9, 0.35, 0.05)]


def _pick(ctx, keys, fallback_cats=None):
    ks = [k for k in keys if k in ctx.lib.items]
    if not ks and fallback_cats:
        ks = ctx.lib.materials(cats_any=fallback_cats)
    return ctx.lib.pick(ctx.rng, ks)


def build(ctx) -> dict:
    rng, lk, lib = ctx.rng, ctx.looks, ctx.lib
    X0, Y0, X1, Y1 = ctx.bounds
    ZX0, ZY0, ZX1, ZY1 = ctx.zone
    B = U.Batch(ctx)
    clutter = float(ctx.knobs.get("clutter", 1.0))

    # --- heights: the ceiling must clear every primitive --------------------------------------
    tops = U.prim_tops(ctx.spec)
    maxtop = max((t for _, t, _ in tops), default=0.0)
    touch = False
    if maxtop <= 4.15:
        lo = max(3.8, maxtop + 0.3)
        hb = float(rng.uniform(lo, max(lo, 4.45))) if lo < 4.45 else 4.45
    else:
        tall = {k for k, t, _ in tops if t > 4.15}
        if tall <= {"cyl_v", "wall", "box"}:
            hb, touch = max(maxtop, 4.55), True
        else:
            hb = max(maxtop + 0.3, 4.6)
    svc = min(max(3.8, maxtop + 0.3), 4.3)          # lowest underside for ceiling services (L1/L2)
    kinds = ["flat", "beams", "beams", "tees", "tees"]
    ceil_kind = "flat" if touch else kinds[int(rng.integers(len(kinds)))]
    if ceil_kind == "beams":
        depth = float(rng.uniform(0.35, 0.65))
    elif ceil_kind == "tees":
        depth = float(rng.uniform(0.6, 0.8))
    else:
        depth = 0.0
    H = hb + depth                                   # slab underside

    # --- style, sky, exposure -------------------------------------------------------------------
    style = "open" if rng.random() < 0.62 else "underground"
    night = style == "open" and rng.random() < 0.12
    if style == "underground":
        hd = lib.hdris(cats_all={"indoor"}, tags_any={"garage", "parking", "tunnel", "underground", "basement",
                                                       "underpass"})
        d_ev = 0.0
    elif night:
        hd = lib.hdris(cats_all={"night", "urban"}) or lib.hdris(cats_all={"night"})
        d_ev = float(rng.uniform(-3.5, -2.0))
    else:
        hd = lib.hdris(cats_all={"outdoor", "urban"}, cats_none={"night", "indoor"},
                       tags_any={"street", "city", "parking", "road", "asphalt", "building", "architecture",
                                 "concrete", "industrial", "suburb", "town", "sidewalk"})
        d_ev = float(rng.uniform(0.6, 2.0))
    sky = lk.sky(lib.pick(rng, hd))
    gain = U.exposure_gain(ctx, sky, d_ev)
    tint = [(1.0, 0.98, 0.95), (0.94, 1.0, 0.97), (0.96, 0.98, 1.0), (1.0, 1.0, 1.0)][int(rng.integers(4))]
    if style == "underground" and rng.random() < 0.35:
        tint = (0.88, 1.0, 0.9)                          # old fluorescent green cast
    # an enclosed level has no daylight: its lamps carry it (brighter, closer together)
    lamps = U.Lamps(ctx, gain * (3.0 if style == "underground" else 1.0), rgb=tint)

    # --- layout in (u, v): u along the drive aisles ------------------------------------------------
    ox = rng.random() < 0.7
    if ox:
        pu0, pv0, pu1, pv1 = X0, Y0, X1, Y1
        zu0, zv0, zu1, zv1 = ZX0, ZY0, ZX1, ZY1
    else:
        pu0, pv0, pu1, pv1 = Y0, X0, Y1, X1
        zu0, zv0, zu1, zv1 = ZY0, ZX0, ZY1, ZX1

    def W(u, v):
        return np.array([u, v], float) if ox else np.array([v, u], float)

    def W3(u, v, z):
        x, y = W(u, v)
        return np.array([x, y, z])

    yaw_u = 0.0 if ox else math.pi / 2               # world yaw of +u
    yaw_v = math.pi / 2 if ox else 0.0               # world yaw of +v

    def rect_w(u0, v0, u1, v1):
        """(centre, half) in world of a u-v rectangle."""
        c = W((u0 + u1) / 2, (v0 + v1) / 2)
        h = (abs(u1 - u0) / 2, abs(v1 - v0) / 2) if ox else (abs(v1 - v0) / 2, abs(u1 - u0) / 2)
        return c, h

    bw = float(rng.choice([2.4, 2.5, 2.5, 2.6, 2.7]))
    D = float(rng.uniform(4.8, 5.3))
    aw = float(rng.uniform(6.0, 7.2))
    Pm = 2 * D + aw
    if ox:
        a0 = float(rng.uniform(-1.0, 1.0))
    else:
        a0 = float(rng.uniform(pv0 + 4.0, pv0 + 12.0))
    # modules on each side of the central aisle: cover the patch in v
    # (the building's sides stand >= 1.6 m outside the patch, so a chase camera 1 m out is still inside)
    n_lo = max(0, int(math.ceil((a0 - aw / 2 - D - (pv0 - 1.3)) / Pm))) + int(rng.choice([0, 0, 1]))
    n_hi = max(0, int(math.ceil((pv1 + 1.3 - (a0 + aw / 2 + D)) / Pm))) + int(rng.choice([0, 0, 1]))
    v_lo = a0 - aw / 2 - D - n_lo * Pm - 0.3
    v_hi = a0 + aw / 2 + D + n_hi * Pm + 0.3
    aisles = [a0 + k * Pm for k in range(-n_lo, n_hi + 1)]
    col_v = [a0 + (k + 0.5) * Pm for k in range(-n_lo - 1, n_hi + 1)]
    e0, e1 = float(rng.uniform(-4.0, 30.0)), float(rng.uniform(-4.0, 30.0))
    u_lo = min(zu0 - e0, pu0 - 1.5)
    u_hi = max(zu1 + e1, pu1 + 1.5)
    nb = int(rng.choice([2, 3, 3, 3, 4]))
    S = nb * bw
    u_ph = u_lo + float(rng.uniform(0.5, S))
    col_u = list(np.arange(u_ph, u_hi - 0.5, S))
    bay_u = list(np.arange(u_ph - S * math.floor((u_ph - u_lo) / S), u_hi, bw))   # bay boundaries

    # --- materials ------------------------------------------------------------------------------
    fk = [_pick(ctx, FLOORS, {"concrete"})]
    if rng.random() < 0.35:                              # a second, similar scan patched in
        fk.append(_pick(ctx, FLOORS, {"concrete"}))
    wet = float(rng.choice([0.0, 0.0, 0.0, 0.15, 0.35])) if style == "open" else float(rng.choice([0.0, 0.0, 0.1]))
    floor = lk.ground_material(fk, tile_m=float(rng.uniform(3.0, 5.0)), puddles=wet, blend_scale=12.0,
                               name="garage_floor")
    epoxy = rng.random() < 0.25
    if epoxy:
        U.set_rough(floor, 0.55)
    painted_ceiling = rng.random() < 0.45
    ctint = (1.25, 1.25, 1.22) if painted_ceiling else None
    ceil_mat = lk.pbr(_pick(ctx, CEILINGS, {"concrete"}), 3.0, tint=ctint, bevel=0.02, dirt=0.25, name="ceiling")
    col_key = _pick(ctx, CEILINGS, {"concrete"})
    col_mat = lk.pbr(col_key, 2.0, tint=(1.3, 1.3, 1.28) if rng.random() < 0.5 else None, bevel=0.025, dirt=0.5,
                     name="column")
    wall_mat = lk.pbr(_pick(ctx, WALLS, {"concrete"}), float(rng.uniform(1.6, 2.6)), bevel=0.02, dirt=0.5,
                      name="garage_wall", tint=(1.2, 1.2, 1.18) if rng.random() < 0.3 else None)
    level_rgb = LEVEL_COLOURS[int(rng.integers(len(LEVEL_COLOURS)))]
    band = U.plain(level_rgb, rough=0.45, name="level_band", bevel=0.005)
    paint_w = U.decal((0.75, 0.75, 0.72), rough=0.5, wear=float(rng.uniform(0.1, 0.6)), seed=float(rng.uniform(0, 99)),
                      name="paint_white")
    paint_y = U.decal((0.8, 0.55, 0.04), rough=0.5, wear=float(rng.uniform(0.1, 0.6)), seed=float(rng.uniform(0, 99)),
                      name="paint_yellow")
    line_mat = paint_w if rng.random() < 0.65 else paint_y
    joint_mat = U.plain((0.05, 0.05, 0.05), rough=0.8, name="joint")
    oil = U.stain(name="oil")
    tyre_mark = U.stain(strength=0.35, rough=0.5, name="tyre_mark", tone=(0.02, 0.02, 0.02))
    steel = U.plain((0.55, 0.56, 0.58), rough=0.35, metal=1.0, name="galv")
    red = U.plain((0.55, 0.03, 0.02), rough=0.4, name="sprinkler_red", bevel=0.005)
    grey_pvc = U.plain((0.5, 0.5, 0.5), rough=0.5, name="pvc")
    housing = U.plain((0.85, 0.85, 0.85), rough=0.45, name="lamp_housing", bevel=0.005)
    yellow = lk.pbr(_pick(ctx, ["acg:PaintedMetal001", "acg:PaintedMetal008"], {"painted metal"}), 0.8,
                    tint=(1.0, 0.8, 0.2), bevel=0.01, dirt=0.3, name="safety_yellow")
    black = U.plain((0.03, 0.03, 0.03), rough=0.5, name="black_paint")

    # --- ground ---------------------------------------------------------------------------------
    c_b, h_b = rect_w(u_lo - 0.3, v_lo - 0.3, u_hi + 0.3, v_hi + 0.3)
    if style == "open":
        street = lk.ground_material([_pick(ctx, STREET, {"asphalt"}), _pick(ctx, STREET, {"asphalt"})], tile_m=4.0,
                                    puddles=wet, name="street")
        ctx.ground(street)
        ctx.flat([c_b + np.array(s) * h_b for s in ((-1, -1), (1, -1), (1, 1), (-1, 1))], floor, z=0.002,
                 name="deck")
    else:
        ctx.ground(floor)

    # --- the patch primitives -----------------------------------------------------------------
    conc_prim = lk.pbr(_pick(ctx, CEILINGS, {"concrete"}), 2.0, bevel=0.03, dirt=0.5, name="prim_concrete")
    pipe_col = [(0.55, 0.03, 0.02), (0.85, 0.65, 0.05), (0.55, 0.56, 0.58), (0.1, 0.3, 0.15)][int(rng.integers(4))]
    pipe = U.plain(pipe_col, rough=0.35, metal=0.6 if pipe_col[0] == 0.55 and pipe_col[2] > 0.5 else 0.0,
                   name="prim_pipe", bevel=0.01)
    ctx.dress_primitives({"wall": wall_mat, "box": conc_prim, "pole": col_mat, "bar": pipe, "rock": conc_prim})
    # thin poles read as steel bollards / pipe columns, fat ones as concrete columns;
    # floating boxes as ducts or downstand beams
    duct_mat = lk.pbr(_pick(ctx, ["acg:Metal009", "acg:Metal011", "acg:Metal041A", "acg:Metal052A"], {"metal"}),
                      1.5, bevel=0.01, dirt=0.2, name="duct_steel")
    for o in ctx.colls["obstacles"].objects:
        i = U.prim_index(o.name)                      # (o.name[4:6] took prim100.. for prim10)
        if i < 0 or i >= len(ctx.spec["primitives"]):
            continue
        p = ctx.spec["primitives"][i]
        if p["kind"] == "cyl_v" and p["r"] < 0.2:
            o.data.materials[0] = yellow
        elif p["kind"] == "box" and p["c"][2] - p["half"][2] > 0.4 and max(p["half"][:2]) > 0.5 \
                and abs(p.get("rpy", [0, 0, 0])[0]) + abs(p.get("rpy", [0, 0, 0])[1]) < 1e-3:
            o.data.materials[0] = duct_mat if rng.random() < 0.5 else conc_prim
    # the family's own free space (wall ends, round a pillar, a ring, a grove, ...) stays empty
    passages = U.claim_passages(ctx)
    # floating boxes / spheres hang from the slab on bridles (a hung sign or duct, not a levitating block)
    n_hung = U.hang_rods(ctx, B, H, U.plain((0.55, 0.56, 0.58), rough=0.35, metal=1.0, name="hanger_rod"))
    # close the gap between tall walls/lintels and the ceiling (above 3.9 m: never flyable)
    n_fill = _fillers(ctx, B, tops, hb, wall_mat, conc_prim)

    n_cores = _cores(ctx, B, lamps, col_v, (u_lo, u_hi), (pu0, pu1), W, ox, H, wall_mat, band, rng)

    # --- columns --------------------------------------------------------------------------------
    col_sec = rng.choice(["rect", "rect", "square", "round"])
    cw_u, cw_v = {"rect": (float(rng.uniform(0.35, 0.45)), float(rng.uniform(0.6, 0.8))),
                  "square": (0.5, 0.5), "round": (0.56, 0.56)}[col_sec]
    pole_cols = sum(1 for k, t, _ in tops if k == "cyl_v" and t >= hb - 0.05) >= 3
    band_z = float(rng.uniform(1.6, 2.2))
    guard = rng.random() < 0.5
    columns = []
    for vc in col_v:
        for uc in col_u:
            xy = W(uc, vc)
            inside = X0 <= xy[0] <= X1 and Y0 <= xy[1] <= Y1
            if pole_cols and inside:
                continue
            hx, hy = (cw_u / 2, cw_v / 2) if ox else (cw_v / 2, cw_u / 2)
            if not U.free_rect(ctx, xy, (hx, hy), 0.0, margin=0.45, step=0.25):
                continue
            ztop = hb if ceil_kind != "beams" else H
            if col_sec == "round":
                ctx.cylinder((xy[0], xy[1], 0.0), (xy[0], xy[1], ztop), cw_u / 2, col_mat, name="column")
                B.cyl((xy[0], xy[1], band_z), (xy[0], xy[1], band_z + 0.45), cw_u / 2 + 0.008, band, seg=24)
                if guard:
                    B.cyl((xy[0], xy[1], 0.0), (xy[0], xy[1], 1.0), cw_u / 2 + 0.006, yellow, seg=24)
            else:
                ctx.box((xy[0], xy[1], ztop / 2), (hx, hy, ztop / 2), col_mat, name="column")
                B.box((xy[0], xy[1], band_z + 0.22), (hx + 0.008, hy + 0.008, 0.225), band)
                if guard:
                    _stripes(B, xy, (hx + 0.006, hy + 0.006), 0.0, 1.0, yellow, black)
            ctx.claim_rect(xy, (hx + 0.1, hy + 0.1), 0.0, kind="column")
            columns.append((xy, hx, hy))

    # --- ceiling ----------------------------------------------------------------------------
    cz = H + 0.15
    c_ce, h_ce = rect_w(u_lo - 0.4, v_lo - 0.4, u_hi + 0.4, v_hi + 0.4)
    ctx.box((c_ce[0], c_ce[1], cz), (h_ce[0], h_ce[1], 0.15), ceil_mat, roof=True, name="ceiling")
    if ceil_kind == "beams":
        for uc in col_u:
            c, h = rect_w(uc - 0.22, v_lo - 0.3, uc + 0.22, v_hi + 0.3)
            B.box((c[0], c[1], hb + depth / 2), (h[0], h[1], depth / 2 + 0.01), ceil_mat, roof=True)
    elif ceil_kind == "tees":
        for vc in col_v:                              # inverted-tee girders on the column lines
            c, h = rect_w(u_lo - 0.3, vc - 0.3, u_hi + 0.3, vc + 0.3)
            B.box((c[0], c[1], hb + depth / 2), (h[0], h[1], depth / 2 + 0.01), ceil_mat, roof=True)
        for us in np.arange(u_lo + 0.6, u_hi, 1.2):   # double-tee stems spanning between them
            c, h = rect_w(us - 0.08, v_lo, us + 0.08, v_hi)
            B.box((c[0], c[1], hb + 0.1 + (depth - 0.1) / 2), (h[0], h[1], (depth - 0.1) / 2 + 0.01), ceil_mat,
                  roof=True)
    # --- lamps ------------------------------------------------------------------------------------
    lamp_top = H if ceil_kind != "tees" else hb + 0.1
    lamp_rows = []
    for c in aisles:
        lamp_rows += [c] if rng.random() < 0.5 else [c - aw / 4, c + aw / 4]
        lamp_rows += [c - aw / 2 - D / 2, c + aw / 2 + D / 2] if rng.random() < 0.6 else []
    Ls = S / 2 if rng.random() < 0.5 else S
    twin = rng.random() < 0.4
    dead = 0.06 if rng.random() < 0.5 else 0.0          # the odd dead tube
    for vr in lamp_rows:
        if not (v_lo < vr < v_hi):
            continue
        for ul in np.arange(u_ph + Ls / 2, u_hi, Ls):
            if ceil_kind == "tees":
                ul = u_lo + 0.6 + 1.2 * round((ul - u_lo - 0.6) / 1.2) + 0.6
            xy = W(ul, vr)
            far = not ctx.in_zone(xy, pad=10.0)
            on = rng.random() > dead
            if not on:
                lamps.tube(B, (xy[0], xy[1], lamp_top), yaw_u, housing, power=0.0, strength=0.0, light=False,
                           twin=twin)
                continue
            lamps.tube(B, (xy[0], xy[1], lamp_top), yaw_u, housing, power=90.0 if twin else 60.0,
                       strength=14.0, light=not far or rng.random() < 0.3, twin=twin)

    # --- services under the ceiling ------------------------------------------------------------------
    room = (hb if ceil_kind == "tees" else H) - svc
    zs_top = hb if ceil_kind == "tees" else H
    if room > 0.12:
        vm = col_v[int(rng.integers(len(col_v)))] + float(rng.choice([-1.2, 1.2]))
        zp = zs_top - 0.14
        a, b = W3(u_lo, vm, zp), W3(u_hi, vm, zp)
        B.cyl(a, b, 0.055, red, seg=12, roof=True)
        for ub in np.arange(u_lo + 1.8, u_hi, 3.6):          # branch lines + heads
            pa, pb = W3(ub, v_lo + 0.5, zs_top - 0.08), W3(ub, v_hi - 0.5, zs_top - 0.08)
            B.cyl(pa, pb, 0.022, red, seg=8, roof=True)
            for vh in np.arange(v_lo + 1.5, v_hi - 1.0, 3.2):
                h3 = W3(ub, vh, zs_top - 0.12)
                B.cyl(h3, h3 + np.array([0, 0, 0.04]), 0.012, steel, seg=6, roof=True)
        # conduit tray
        vt = aisles[int(rng.integers(len(aisles)))] + float(rng.uniform(-2.5, 2.5))
        c, h = rect_w(u_lo, vt - 0.15, u_hi, vt + 0.15)
        B.box((c[0], c[1], zs_top - 0.08), (h[0], h[1], 0.03), steel, roof=True)
    if room > 0.55 and rng.random() < 0.75:                  # ventilation duct run
        vd = col_v[int(rng.integers(len(col_v)))] + float(rng.choice([-1.8, 1.8]))
        dh = min(0.5, room - 0.05)
        c, h = rect_w(u_lo, vd - 0.4, u_hi, vd + 0.4)
        B.box((c[0], c[1], zs_top - dh / 2 - 0.02), (h[0], h[1], dh / 2), duct_mat, roof=True)
        for uj in np.arange(u_lo + 2.0, u_hi, 1.5):          # flange joints
            c, h = rect_w(uj - 0.02, vd - 0.43, uj + 0.02, vd + 0.43)
            B.box((c[0], c[1], zs_top - dh / 2 - 0.02), (h[0], h[1], dh / 2 + 0.03), duct_mat, roof=True)

    # --- perimeter: open sides, walls, ends ------------------------------------------------------
    sides = {}
    for side, vedge in (("lo", v_lo), ("hi", v_hi)):
        if style == "underground":
            sides[side] = "wall"
        else:
            sides[side] = "open" if rng.random() < 0.75 else "wall"
    if style == "open" and all(s == "wall" for s in sides.values()):
        sides["hi"] = "open"
    ends = {"lo": "wall" if style == "underground" or rng.random() < 0.5 else "open",
            "hi": "wall" if style == "underground" or rng.random() < 0.5 else "open"}
    parapet_h = float(rng.uniform(0.95, 1.1))
    cable_rail = rng.random() < 0.5
    edge_mat = lk.pbr(col_key, 2.0, bevel=0.02, dirt=0.6, name="spandrel",
                      tint=(1.25, 1.25, 1.22) if painted_ceiling else None)
    for side, vedge in (("lo", v_lo), ("hi", v_hi)):
        sgn = -1 if side == "lo" else 1
        vw = vedge + sgn * 0.12
        if sides[side] == "open":
            c, h = rect_w(u_lo - 0.4, vw - 0.12, u_hi + 0.4, vw + 0.12)
            zc = 0.3 if cable_rail else parapet_h
            ctx.box((c[0], c[1], zc / 2), (h[0], h[1], zc / 2), edge_mat, name="parapet")
            if cable_rail:
                for zc_ in np.arange(0.45, parapet_h + 0.01, 0.15):
                    a, b = W3(u_lo, vw, zc_), W3(u_hi, vw, zc_)
                    B.cyl(a, b, 0.008, steel, seg=6)
            sp = float(rng.uniform(0.4, 0.8))
            c, h = rect_w(u_lo - 0.4, vw - 0.12, u_hi + 0.4, vw + 0.12)
            zb_ = max(H - sp, svc)
            ctx.box((c[0], c[1], (zb_ + H + 0.3) / 2), (h[0], h[1], (H + 0.3 - zb_) / 2), edge_mat, roof=True,
                    name="spandrel")
        else:
            c, h = rect_w(u_lo - 0.4, vw - 0.15, u_hi + 0.4, vw + 0.15)
            ctx.box((c[0], c[1], H / 2), (h[0], h[1], H / 2), wall_mat, name="garage_wall",
                    roof=(side == "lo"))
            _wall_dressing(ctx, B, lamps, u_lo, u_hi, vw - sgn * 0.15, sgn, W, W3, yaw_u, band, H)
    for end, ue in (("lo", u_lo), ("hi", u_hi)):
        sgn = -1 if end == "lo" else 1
        uw = ue + sgn * 0.15
        if ends[end] == "wall":
            c, h = rect_w(uw - 0.15, v_lo - 0.4, uw + 0.15, v_hi + 0.4)
            ctx.box((c[0], c[1], H / 2), (h[0], h[1], H / 2), wall_mat, name="garage_wall")
            _end_dressing(ctx, B, lamps, uw - sgn * 0.15, sgn, v_lo, v_hi, W, W3, yaw_v, band, H, rng)
        else:
            c, h = rect_w(uw - 0.3, v_lo - 0.4, uw + 0.3, v_hi + 0.4)
            zb_ = max(H - 0.6, svc)
            ctx.box((c[0], c[1], (zb_ + H + 0.3) / 2), (h[0], h[1], (H + 0.3 - zb_) / 2), edge_mat, roof=True,
                    name="spandrel")
    if style == "open":
        _exterior(ctx, B, sides, ends, (u_lo, v_lo, u_hi, v_hi), W, ox, rng)

    # --- floor markings ------------------------------------------------------------------------------
    U.joints(ctx, (min(W(u_lo, v_lo)[0], W(u_hi, v_hi)[0]), min(W(u_lo, v_lo)[1], W(u_hi, v_hi)[1]),
                   max(W(u_lo, v_lo)[0], W(u_hi, v_hi)[0]), max(W(u_lo, v_lo)[1], W(u_hi, v_hi)[1])),
             float(rng.choice([6.0, 7.5, 9.0])), joint_mat)
    lw = float(rng.choice([0.1, 0.12, 0.15]))
    if rng.random() < 0.3:                                   # painted (epoxy) drive aisles, worn in the wheel tracks
        ep_rgb = [(0.2, 0.24, 0.21), (0.17, 0.18, 0.2), (0.24, 0.24, 0.24), (0.15, 0.19, 0.23)][int(rng.integers(4))]
        ep = U.decal(ep_rgb, rough=0.35, wear=float(rng.uniform(0.0, 0.12)), seed=float(rng.uniform(0, 99)),
                     name="epoxy_aisle", scale=0.35)
        for c in aisles:
            cc, hh = rect_w(u_lo + 0.3, c - aw / 2, u_hi - 0.3, c + aw / 2)
            ctx.flat([cc + np.array(s_) * hh for s_ in ((-1, -1), (1, -1), (1, 1), (-1, 1))], ep, z=0.0028,
                     name="epoxy")
    for c in aisles:
        for sgn in (-1, 1):
            vf, vb = c + sgn * aw / 2, c + sgn * (aw / 2 + D)
            for ub in bay_u:
                if u_lo + 0.3 < ub < u_hi - 0.3:
                    U.line(ctx, W(ub, vf), W(ub, vb - sgn * 0.15), lw, line_mat)
            if rng.random() < 0.5:
                U.line(ctx, W(u_lo + 0.5, vf), W(u_hi - 0.5, vf), lw, line_mat)
        two_way = rng.random() < 0.4
        if two_way:
            U.dashed(ctx, W(u_lo + 1, c), W(u_hi - 1, c), 0.12, paint_y, dash=3.0, gap=3.0)
        direction = 1 if rng.random() < 0.5 else -1
        for ua in np.arange(u_lo + float(rng.uniform(6, 14)), u_hi - 3, float(rng.uniform(14, 22))):
            du = np.array([1.0, 0.0]) if ox else np.array([0.0, 1.0])
            if two_way:
                U.arrow(ctx, W(ua, c - aw / 4), -du, paint_w)
                U.arrow(ctx, W(ua + 3, c + aw / 4), du, paint_w)
            else:
                U.arrow(ctx, W(ua, c), direction * du, paint_w)
        if rng.random() < 0.6:                                  # zebra crossing
            uz = float(rng.uniform(pu0 + 3, pu1 - 3))
            for vz in np.arange(c - aw / 2 + 0.3, c + aw / 2 - 0.2, 0.9):
                U.line(ctx, W(uz - 1.4, vz + 0.25), W(uz + 1.4, vz + 0.25), 0.5, paint_w)
        # tyre marks along the aisle
        for _ in range(int(rng.integers(2, 6))):
            vt = c + float(rng.normal(0, aw / 5))
            ua = float(rng.uniform(u_lo, u_hi - 10))
            ctx.strip(W(ua, vt), W(ua + float(rng.uniform(5, 20)), vt + float(rng.normal(0, 0.4))),
                      float(rng.uniform(0.18, 0.25)), tyre_mark, z=0.003, name="tyre")
    # hatched no-parking boxes next to some columns / ends
    for _ in range(int(rng.integers(0, 3))):
        c = aisles[int(rng.integers(len(aisles)))]
        sgn = int(rng.choice([-1, 1]))
        ub = bay_u[int(rng.integers(len(bay_u)))]
        cc = W(ub + bw / 2, c + sgn * (aw / 2 + D / 2))
        U.hatch(ctx, cc, (bw / 2 - 0.1, D / 2 - 0.1) if ox else (D / 2 - 0.1, bw / 2 - 0.1), 0.0, paint_y)

    # --- cars -----------------------------------------------------------------------------------------
    # procedural cars and the library's real ones draw from their own generators: whether a bay
    # holds a real car, or the Objaverse catalogue is there at all, does not shift the scene's
    # stream (only the claims of a different footprint can move a later prop)
    fleet = U.Fleet(ctx, rng=np.random.default_rng(int(rng.integers(1 << 31))))
    orng = np.random.default_rng(int(rng.integers(1 << 31)))
    occ = float(np.clip(0.3 + 0.45 * clutter, 0.3, 0.92)) * float(rng.uniform(0.75, 1.1))
    stops = rng.random() < 0.5
    stop_mat = lk.pbr(col_key, 1.0, tint=(1.1, 1.05, 0.7) if rng.random() < 0.4 else None, bevel=0.02, dirt=0.4,
                      name="wheel_stop")
    covered = lib.models(ids=["covered_car"])
    # a few real (modelled) cars among the procedural ones: <= 3 models a scene, only models that
    # fit this level's bays, <= OBJ_REPEAT of each
    obj_all = [k for k in lib.models(ids=OBJ_CARS, max_tris=90000)
               if min(lib.size(k)[:2]) <= bw - 0.35 and max(lib.size(k)[:2]) <= D + 0.4]
    obj_cars = [obj_all[int(i)] for i in orng.permutation(len(obj_all))[:3]]
    p_obj = float(orng.choice([0.0, 0.15, 0.3, 0.45]))
    obj_used: dict = {}
    n_cars = n_obj = 0
    extra = 0.0                                          # m^2 of in-zone instanced cars (not in B.area_in)
    special = {}
    # the cloud budget: cars take what the primitives and the structure leave (an arena's
    # primitives alone are ~450k points); occupancy thins evenly over the bays the cloud sees,
    # a hard stop catches the rest; bays outside it keep their occupancy
    budget = U.area_budget() - U.cloud_area(ctx, B)
    bays_in = sum(1 for c in aisles for sgn in (-1, 1) for ub in bay_u
                  if u_lo + 0.5 < ub + bw / 2 < u_hi - 0.5 and
                  ctx.in_zone(W(ub + bw / 2, c + sgn * (aw / 2 + D / 2)), pad=3.0))
    occ0 = occ
    if bays_in:
        occ = min(occ, max(0.0, budget) / (U.CAR_AREA * bays_in))
    area0 = B.area_in
    for c in aisles:
        for sgn in (-1, 1):
            vf = c + sgn * aw / 2
            for ub in bay_u:
                uc = ub + bw / 2
                if not (u_lo + 0.5 < uc < u_hi - 0.5):
                    continue
                vc = vf + sgn * D / 2
                bxy = W(uc, vc)
                stop = None
                if stops:
                    sxy = W(uc, vf + sgn * (D - 0.6))
                    sh = (0.9, 0.08) if ox else (0.08, 0.9)
                    if U.free_rect(ctx, sxy, sh, 0.0, margin=0.3, step=0.3):
                        stop = (sxy, sh)
                r_ = rng.random()
                if r_ < 0.06 and (ub, sgn) not in special:          # accessible / EV bay paint
                    col = (0.05, 0.2, 0.6) if rng.random() < 0.6 else (0.1, 0.45, 0.2)
                    m = U.decal(col, rough=0.5, wear=0.4, seed=float(rng.uniform(0, 99)), name="bay_fill")
                    cc, hh = rect_w(ub + 0.1, vf + sgn * 0.1, ub + bw - 0.1, vf + sgn * (D - 0.1))
                    ctx.flat([cc + np.array(s) * hh for s in ((-1, -1), (1, -1), (1, 1), (-1, 1))], m, z=0.0035,
                             name="bay_fill")
                if rng.random() < 0.5:
                    U.blob(ctx, bxy + rng.normal(0, 0.3, 2), float(rng.uniform(0.2, 0.6)), float(rng.uniform(0.2, 0.5)),
                           float(rng.uniform(0, 3)), oil)
                if stop is not None:
                    sxy, sh = stop
                    ctx.box((sxy[0], sxy[1], 0.06), (sh[0], sh[1], 0.06), stop_mat, name="wheel_stop",
                            coll="scatter")
                if rng.random() > (occ if ctx.in_zone(bxy, pad=3.0) else occ0):
                    if stop is not None:
                        ctx.claim_rect(stop[0], stop[1], 0.0, kind="stop")
                    continue
                kind = fleet.pick_kind()
                L_, W_, H_ = fleet.dims(kind)
                if L_ > D + 0.4 or W_ > bw - 0.35:
                    kind = "hatch"
                    L_, W_, H_ = fleet.dims(kind)
                nose = sgn if rng.random() < 0.75 else -sgn             # nose-in mostly
                depth_off = sgn * float(rng.uniform(-0.25, 0.15))
                cxy = W(uc + float(rng.normal(0, 0.12)), vf + sgn * (D / 2) + depth_off)
                yaw = (yaw_v if nose > 0 else yaw_v + math.pi) + float(rng.normal(0, 0.03))
                hL, hW = L_ / 2 + 0.1, W_ / 2 + 0.1
                half = (hW, hL) if ox else (hL, hW)
                o_pick, o_i = orng.random() < p_obj, int(orng.integers(3))
                okey = obj_cars[o_i % len(obj_cars)] if o_pick and obj_cars else None
                if okey is not None and obj_used.get(okey, 0) >= OBJ_REPEAT:
                    okey = None
                if okey is not None:                    # a real car: its own size decides the claim
                    sx, sy, _ = lib.size(okey)
                    oL, oW = max(sx, sy), min(sx, sy)
                    half = (oW / 2 + 0.1, oL / 2 + 0.1) if ox else (oL / 2 + 0.1, oW / 2 + 0.1)
                over = B.area_in - area0 + extra + U.CAR_AREA > budget and ctx.in_zone(cxy, pad=3.0)
                # a column at the bay's back corner: pull the car up to 0.4 m toward the aisle, as
                # drivers do, before leaving the bay empty (r6 corridor: 114 of 160 bays lost to it)
                fit = None if over else next(
                    (c for c in (cxy - W(0.0, sgn * pull) for pull in (0.0, 0.2, 0.4))
                     if U.free_rect(ctx, c, half, 0.0, margin=0.25, step=0.4)), None)
                if fit is None:
                    if stop is not None:
                        ctx.claim_rect(stop[0], stop[1], 0.0, kind="stop")
                    continue
                cxy = fit
                if okey is not None:
                    sx, sy, _ = lib.size(okey)
                    ctx.instance(okey, cxy, 0.0, yaw if sx >= sy else yaw - math.pi / 2, 1.0, coll="scatter")
                    extra += U.CAR_AREA if ctx.in_zone(cxy, pad=3.0) else 0.0
                    obj_used[okey] = obj_used.get(okey, 0) + 1
                    n_obj += 1
                elif orng.random() < 0.05 and covered:
                    ctx.instance(covered[0], cxy, 0.0, 0.0 if ox else math.pi / 2, 1.0, coll="scatter")
                    extra += U.CAR_AREA if ctx.in_zone(cxy, pad=3.0) else 0.0
                else:
                    fleet.car(B, cxy, yaw, kind=kind, dims=(L_, W_, H_))
                ctx.claim_rect(cxy, half, 0.0, kind="car")
                if stop is not None:
                    ctx.claim_rect(stop[0], stop[1], 0.0, kind="stop")
                n_cars += 1

    # --- props ----------------------------------------------------------------------------------------
    ext = lib.models(ids=["korean_fire_extinguisher_01"])
    for (xy, hx, hy) in columns:
        if ext and rng.random() < 0.25 and ctx.in_zone(xy):
            off = np.array([hx + 0.15, 0.0]) if rng.random() < 0.5 else np.array([0.0, hy + 0.15])
            off *= rng.choice([-1, 1])
            p = xy + off
            if ctx.free(p, 0.2, 0.2):
                ctx.instance(ext[0], p, 0.0, float(rng.uniform(0, 6.28)), 1.0, claim=True)
    small = lib.models(ids=["metal_trash_can", "concrete_road_barrier", "concrete_road_barrier_02", "old_tyre",
                            "utility_box_02", "plastic_container", "trashbag", "hand_truck", "plastic_broom"],
                       max_tris=65000)
    near_edge = lambda xy: any(abs((xy[1] if ox else xy[0]) - cv) < 1.4 for cv in col_v)  # noqa: E731
    ctx.scatter(small, ctx.count(3.0), margin=0.4, where=near_edge)
    cone_m = U.plain((0.9, 0.25, 0.02), rough=0.5, name="cone")
    cone_s = U.plain((0.85, 0.85, 0.85), rough=0.3, name="cone_stripe")
    for _ in range(ctx.count(2.5)):
        xy = ctx.uniform_xy(pad=1.0)
        if ctx.free(xy, 0.25, 0.5):
            U.cone(ctx, B, xy, cone_m, cone_s)
            ctx.claim_disc(xy, 0.25, kind="cone")

    # --- hanging obstacles (L3) ---------------------------------------------------------------------
    n_hang = ctx.count(float(ctx.knobs.get("hang", 0.0)))
    hung = _hang(ctx, B, n_hang, hb if ceil_kind != "flat" else H, W, W3, ox, (pu0, pv0, pu1, pv1), yellow, black,
                 steel, grey_pvc, duct_mat, lamps, housing) if n_hang > 0 else 0

    # two-wheelers by the column lines, last: whether the library has them changes nothing else
    bikes = lib.models(ids=OBJ_BIKES, max_tris=50000)
    n_bikes = len(ctx.scatter(bikes, ctx.count(0.8), margin=0.4, where=near_edge)) if bikes else 0

    car_area = (B.area_in - area0) / max(n_cars, 1)
    B.flush()
    # what the dressing does to the tier-A family, read off the geometry the cloud will sample
    sem, cover = U.ceiling_semantics(U.overhead(ctx), tops, H, rect=ctx.bounds)
    if n_fill:
        sem.append(f"{n_fill} wall/box tops closed up to the ceiling (above 3.9 m)")
    if n_hung:
        sem.append(f"{n_hung} floating primitives hung from the slab on a splayed bridle (two 12 mm cables "
                   f"leaning {U.BRIDLE_SPLAY:g} deg out from the ends of a box / the sides of a sphere; the prism "
                   f"over the primitive stays free)")
    miss = [k for k in OBJ_CARS + OBJ_BIKES if k not in lib.items]
    if miss:
        ctx.note(f"garage: {len(miss)} of {len(OBJ_CARS) + len(OBJ_BIKES)} Objaverse vehicles not in the library")
    ctx.note(f"garage {style} ceil={ceil_kind} hb={hb:.2f} H={H:.2f} maxtop={maxtop:.2f} cars={n_cars} "
             f"(occ {occ:.2f} of {occ0:.2f}, {bays_in} bays in zone, budget {budget:.0f} m2, ~{car_area:.0f} m2/car) "
             f"lamps={lamps.n} cloud_area~{U.cloud_area(ctx):.0f}m2")
    return {"sky": sky, "style": style, "night": night, "ceiling": ceil_kind, "ceiling_underside": round(hb, 3),
            "slab_underside": round(H, 3), "aisle_axis": "x" if ox else "y", "bay_w": bw, "bay_d": round(D, 2),
            "aisle_w": round(aw, 2), "cars": n_cars, "objaverse_cars": n_obj, "bikes": n_bikes,
            "occupancy": round(occ, 3), "lamps": lamps.n, "floor": fk, "wet": wet, "epoxy": epoxy, "sides": sides,
            "ends": ends, "cores": n_cores, "hanging": hung, "passages": passages, "semantics_changed": sem,
            "ceiling_cover": cover}


# --- pieces ---------------------------------------------------------------------------------------

def _stripes(B, xy, half, z0, z1, a, b, pitch=0.2):
    """Yellow/black hazard wrap on a column (horizontal bands)."""
    k = 0
    for z in np.arange(z0, z1, pitch):
        B.box((xy[0], xy[1], z + pitch / 2), (half[0], half[1], pitch / 2), a if k % 2 == 0 else b)
        k += 1


def _fillers(ctx, B, tops, hb, wall_mat, conc):
    """Solid wall/lintel tops within 0.6 m of the ceiling are closed up to it
    (above 3.9 m nothing can fly), except over full-height slots. Returns how many."""
    n = 0
    for k, t, p in tops:
        gap = hb - t
        if t < 3.9 or gap <= 0.01 or gap > 0.6:
            continue
        if k == "wall":
            p0, p1 = np.asarray(p["p0"], float), np.asarray(p["p1"], float)
            L = float(np.linalg.norm(p1 - p0))
            d = (p1 - p0) / L
            yaw = math.atan2(d[1], d[0])
            spans = [(0.0, L)]
            for s, w, sz0, sz1 in p.get("slots", []):
                if sz1 >= t - 1e-3:
                    a, b = s - w / 2, s + w / 2
                    spans = [piece for (u0, u1) in spans for piece in ((u0, min(u1, a)), (max(u0, b), u1))
                             if piece[1] - piece[0] > 0.05]
            for u0, u1 in spans:
                c = p0 + d * (u0 + u1) / 2
                B.box((c[0], c[1], t + gap / 2), ((u1 - u0) / 2, p["t"] / 2, gap / 2), wall_mat, yaw=yaw)
                n += 1
        elif k == "box":
            rpy = p.get("rpy", [0, 0, 0])
            if abs(rpy[0]) + abs(rpy[1]) > 1e-3:
                continue
            c = p["c"]
            B.box((c[0], c[1], t + gap / 2), (p["half"][0], p["half"][1], gap / 2), conc, yaw=rpy[2])
            n += 1
    return n


def _wall_dressing(ctx, B, lamps, u_lo, u_hi, vface, sgn, W, W3, yaw_u, band, H):
    """A long closed side: a painted dado band, doors with exit signs, pipes."""
    rng = ctx.rng
    dado = U.plain(tuple(0.4 + 0.5 * c for c in band.node_tree.nodes["Principled BSDF"].inputs["Base Color"]
                         .default_value[:3]), rough=0.6, name="dado")
    c = W((u_lo + u_hi) / 2, vface - sgn * 0.01)
    L = (u_hi - u_lo) / 2
    hz = float(rng.uniform(0.9, 1.3))
    if rng.random() < 0.6:
        B.box((c[0], c[1], hz / 2), ((L, 0.01) if abs(math.cos(yaw_u)) > 0.5 else (0.01, L)) + (hz / 2,), dado)
    door = U.plain((0.25, 0.27, 0.3), rough=0.4, metal=0.3, name="door", bevel=0.01)
    exit_m = U.plain((0.05, 0.6, 0.2), rough=0.3, name="exit_sign", emission=(0.1, 1.0, 0.3, 6.0 * lamps.gain))
    for ud in np.arange(u_lo + float(rng.uniform(5, 15)), u_hi - 3, float(rng.uniform(18, 35))):
        p = W(ud, vface - sgn * 0.03)
        hh = (0.5, 0.03) if abs(math.cos(yaw_u)) > 0.5 else (0.03, 0.5)
        B.box((p[0], p[1], 1.05), hh + (1.05,), door)
        B.box((p[0], p[1], 2.45), (hh[0] * 0.35 / 0.5 if hh[0] > 0.1 else 0.03,
                                   hh[1] * 0.35 / 0.5 if hh[1] > 0.1 else 0.03, 0.1), exit_m)


def _end_dressing(ctx, B, lamps, uface, sgn, v_lo, v_hi, W, W3, yaw_v, band, H, rng):
    """An end wall: a stair/lift core door, exit sign, painted level colour."""
    door = U.plain((0.2, 0.22, 0.25), rough=0.4, metal=0.3, name="door", bevel=0.01)
    exit_m = U.plain((0.05, 0.6, 0.2), rough=0.3, name="exit_sign", emission=(0.1, 1.0, 0.3, 6.0 * lamps.gain))
    vd = float(rng.uniform(v_lo + 2, v_hi - 2))
    p = W(uface - sgn * 0.03, vd)
    hh = (0.55, 0.03) if abs(math.cos(yaw_v)) > 0.5 else (0.03, 0.55)
    B.box((p[0], p[1], 1.05), hh + (1.05,), door)
    B.box((p[0], p[1], 2.45), (min(hh[0], 0.3), min(hh[1], 0.3), 0.1), exit_m)
    # a big painted level number panel (a coloured square)
    pv = W(uface - sgn * 0.02, vd + float(rng.choice([-3, 3])))
    B.box((pv[0], pv[1], 2.2), (0.02 if hh[0] < 0.1 else 0.6, 0.6 if hh[0] < 0.1 else 0.02, 0.6), band)


def _cores(ctx, B, lamps, col_v, urange, prange, W, ox, H, wall_mat, band, rng):
    """Stair / lift cores: concrete boxes on a column line with a steel door,
    a lift door, a glazed lobby panel and an exit sign on each aisle face."""
    u_lo, u_hi = urange
    pu0, pu1 = prange
    door = U.plain((0.22, 0.24, 0.27), rough=0.4, metal=0.3, name="core_door", bevel=0.01)
    lift = U.plain((0.6, 0.6, 0.62), rough=0.25, metal=1.0, name="lift_door", bevel=0.005)
    glass = ctx.looks.glass(name="lobby_glass")
    exit_m = U.plain((0.05, 0.6, 0.2), rough=0.3, name="exit_sign", emission=(0.1, 1.0, 0.3, 6.0 * lamps.gain))
    n = 0
    for _ in range(int(rng.choice([0, 1, 1, 2]))):
        for _t in range(20):
            vc = col_v[int(rng.integers(len(col_v)))]
            uc = float(rng.uniform(u_lo + 5, u_hi - 5))
            if pu0 - 4 < uc < pu1 + 4 and rng.random() < 0.7:
                continue                                # mostly beyond the patch ends
            lu, lv = float(rng.uniform(5.0, 7.5)), float(rng.uniform(4.0, 5.5))
            c = W(uc, vc)
            half = (lu / 2, lv / 2) if ox else (lv / 2, lu / 2)
            if not U.free_rect(ctx, c, half, 0.0, margin=0.6, step=0.5):
                continue
            ctx.box((c[0], c[1], H / 2), (half[0], half[1], H / 2), wall_mat, name="core")
            ctx.claim_rect(c, (half[0] + 0.2, half[1] + 0.2), 0.0, kind="core")
            for sgn in (-1, 1):                         # the two aisle faces (+-v)
                vf = vc + sgn * (lv / 2 + 0.02)
                for du, w, h, m in ((-lu / 4, 0.5, 1.05, door), (lu / 5, 0.55, 1.1, lift),
                                    (-lu / 4 + 1.1, 0.45, 1.2, glass)):
                    p = W(uc + du, vf)
                    hh = (w, 0.02) if ox else (0.02, w)
                    B.box((p[0], p[1], h), (hh[0], hh[1], h), m)
                p = W(uc - lu / 4, vf + sgn * 0.02)
                B.box((p[0], p[1], 2.4), (0.18 if ox else 0.02, 0.02 if ox else 0.18, 0.08), exit_m)
                p = W(uc + lu / 5, vf + sgn * 0.01)
                B.box((p[0], p[1], 2.6), (0.6 if ox else 0.015, 0.015 if ox else 0.6, 0.25), band)
            n += 1
            break
    return n


def _exterior(ctx, B, sides, ends, rect, W, ox, rng):
    """Outside the open sides: sidewalk, kerb, a street and buildings across it
    (render-only beyond the cloud zone), street lamps."""
    lk, lib = ctx.looks, ctx.lib
    u_lo, v_lo, u_hi, v_hi = rect
    ZX0, ZY0, ZX1, ZY1 = ctx.zone
    zv0, zv1 = (ZY0, ZY1) if ox else (ZX0, ZX1)
    walk = lk.pbr(_pick(ctx, SIDEWALK, {"paving stones"}), 2.0, proj="flat", name="sidewalk")
    facades = lib.materials(cats_any={"facade"}) or lib.materials(cats_any={"bricks"})
    kerb = U.plain((0.55, 0.55, 0.52), rough=0.7, name="kerb")
    lamps_k = lib.models(ids=["street_lamp_01"])
    for side, vedge in (("lo", v_lo), ("hi", v_hi)):
        if sides[side] != "open":
            continue
        sgn = -1 if side == "lo" else 1
        v_sw0 = vedge + sgn * 0.3
        v_sw1 = vedge + sgn * float(rng.uniform(2.5, 4.5))
        a, b = W(u_lo - 60, v_sw0), W(u_hi + 60, v_sw1)
        ctx.flat([(a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1])], walk, z=0.01, name="sidewalk")
        k0 = W(u_lo - 60, v_sw1)
        k1 = W(u_hi + 60, v_sw1)
        ctx.strip(k0, k1, 0.2, kerb, z=0.02, name="kerb")
        v_road1 = v_sw1 + sgn * float(rng.uniform(7, 12))
        zedge = zv1 if sgn > 0 else zv0
        vb = v_road1 + sgn * 3.5
        if sgn * vb < sgn * zedge + 8.0:
            vb = zedge + sgn * 8.0
        u = u_lo - 50
        while u < u_hi + 50:
            wdt = float(rng.uniform(10, 26))
            hgt = float(rng.choice([6, 9, 12, 15, 20, 28, 40]) * rng.uniform(0.9, 1.1))
            dep = float(rng.uniform(10, 18))
            fm = lk.pbr(ctx.lib.pick(rng, facades), float(rng.uniform(6, 12)), bevel=0.0, dirt=0.2,
                        name="facade")
            c = W(u + wdt / 2, vb + sgn * dep / 2)
            half = (wdt / 2, dep / 2) if ox else (dep / 2, wdt / 2)
            ctx.box((c[0], c[1], hgt / 2), (half[0], half[1], hgt / 2), fm, coll="backdrop", collide=False,
                    name="building")
            u += wdt + float(rng.choice([0.0, 0.0, 4.0, 8.0]))
        if lamps_k:
            for ul in np.arange(u_lo - 30, u_hi + 30, float(rng.uniform(18, 28))):
                p = W(ul, v_sw1 - sgn * 0.5)
                if not ctx.in_zone(p, pad=1.0):
                    dv = W(0, -sgn) - W(0, 0)
                    ctx.instance(lamps_k[0], p, 0.0, math.atan2(dv[1], dv[0]) - math.pi / 2, 1.0, coll="backdrop",
                                 collide=False)


def _hang(ctx, B, n, zceil, W, W3, ox, prect, yellow, black, steel, pvc, duct, lamps, housing):
    """Things that hang into the flight band: clearance bars on chains,
    signs, sagging conduit, pipe drops, a lowered duct."""
    rng = ctx.rng
    pu0, pv0, pu1, pv1 = prect
    sign_face = U.plain((0.08, 0.3, 0.7), rough=0.4, name="hang_sign", emission=(0.2, 0.5, 1.0, 1.5 * lamps.gain))
    done = 0
    for _ in range(n * 30):
        if done >= n:
            break
        kind = rng.choice(["bar", "bar", "sign", "conduit", "drop", "duct"])
        u = float(rng.uniform(pu0 + 1.0, pu1 - 1.0))
        v = float(rng.uniform(pv0 + 1.0, pv1 - 1.0))
        if kind == "bar":
            Lb = float(rng.uniform(2.5, 5.0))
            z = float(rng.uniform(2.1, 2.7))
            a, b = W3(u, v - Lb / 2, z), W3(u, v + Lb / 2, z)
            if not U.seg_clear(ctx, a, b, 0.5):
                continue
            k = 0
            d = (b - a) / Lb
            for t in np.arange(0, Lb, 0.3):
                B.cyl(a + d * t, a + d * min(t + 0.3, Lb), 0.05, yellow if k % 2 == 0 else black, seg=10)
                k += 1
            for p in (a + d * 0.2, b - d * 0.2):
                B.cyl(p, np.array([p[0], p[1], zceil]), 0.008, steel, seg=6)
        elif kind == "sign":
            z = float(rng.uniform(2.6, 3.2))
            p = W3(u, v, z)
            half = (0.7, 0.04) if not ox else (0.04, 0.7)
            a = p - np.array([half[0], half[1], 0]) * 0.9
            b = p + np.array([half[0], half[1], 0]) * 0.9
            if not U.seg_clear(ctx, a, b, 0.5):
                continue
            B.box(p, (half[0], half[1], 0.2), sign_face)
            for q in (a, b):
                B.cyl((q[0], q[1], z + 0.2), (q[0], q[1], zceil), 0.008, steel, seg=6)
        elif kind == "conduit":
            Lc = float(rng.uniform(4.0, 9.0))
            sag = float(rng.uniform(0.8, zceil - 2.6))
            a, b = W3(u - Lc / 2, v, zceil - 0.1), W3(u + Lc / 2, v, zceil - 0.1)
            if not U.seg_clear(ctx, a, b, 0.5):
                continue
            ts = np.linspace(0, 1, 9)
            pts = [a + (b - a) * t - np.array([0, 0, sag * 4 * t * (1 - t)]) for t in ts]
            for p, q in zip(pts[:-1], pts[1:]):
                B.cyl(p, q, 0.025, black, seg=8)
        elif kind == "drop":
            z = float(rng.uniform(2.3, 2.9))
            Lr = float(rng.uniform(2.0, 4.0))
            a = W3(u, v, zceil)
            m = W3(u, v, z)
            b = W3(u + Lr, v, z)
            if not U.seg_clear(ctx, m, b, 0.5):
                continue
            B.cyl(a, m, 0.06, pvc, seg=12)
            B.cyl(m, b, 0.06, pvc, seg=12)
            B.cyl(b, W3(u + Lr, v, zceil), 0.06, pvc, seg=12)
        else:
            z = float(rng.uniform(3.0, 3.4))
            Ld = float(rng.uniform(5.0, 9.0))
            a, b = W3(u - Ld / 2, v, z), W3(u + Ld / 2, v, z)
            if not U.seg_clear(ctx, a, b, 0.8) or zceil - z < 0.5:
                continue
            c = (a + b) / 2
            half = (Ld / 2, 0.4) if ox else (0.4, Ld / 2)
            B.box((c[0], c[1], z + 0.2), (half[0], half[1], 0.2), duct)
            for p in (a, b, c):
                B.cyl((p[0], p[1], z + 0.4), (p[0], p[1], zceil), 0.008, steel, seg=6)
        done += 1
    return done
