"""Distribution warehouse: pallet racking aisles along the patch, steel
columns, a sealed concrete floor with lane lines, high-bay lamps under a
steel roof with skylights, a dock wall with roller shutters (some open onto
trailers or the yard), staging lanes, forklifts and carts.

What makes it read as a real warehouse rather than shelves on a plane:

* selective pallet racking built the way it is: upright frames (orange,
  blue, grey, green) with bracing and base plates every 2.7 m bay, step
  beams at 1.5-2 m pitch, three euro pallets per bay per level, back-to-back
  rows with a 0.25 m flue, cross aisles every 20-35 m, and loads that are
  what a warehouse holds -- carton stacks, stretch-wrapped loads, drums,
  sacks, bins, IBC totes -- with empty positions;
* the aisles are real aisle widths (3.2-4.4 m) along the patch's long axis,
  so the drone flies down them; rack bays that would block a patch opening
  or the start/goal are simply left out (a cross aisle, a missing bay);
* the shell is 7-10 m high: insulated panel walls on a concrete plinth, a
  steel deck on girders and purlins, skylight strips the daylight falls
  through, rows of UFO or linear high-bays;
* floor: sealed/polished concrete with saw-cut joints, tyre marks, yellow
  lane lines, walkways and hatched keep-clear boxes.

Everything above the cloud top (roof, lamps, upper rack levels) is out of
the cloud, so the cloud is open up to its top and racks, patch walls and
poles taller than it are cut at it (``semantics_changed`` says so, measured
on the built geometry); the big walls stand >= 8 m beyond the cloud zone
unless the staging layout puts the dock wall just past the patch (then it
collides like the rest). L3 (``hang``) adds aisle signs, low lamps and cable
trays hanging into the flight band. Forklifts and the Objaverse choices draw
from their own generators and the Objaverse props go in last, so the hall
(structure, racks, loads, lamps) does not depend on the Objaverse catalogue.
"""

from __future__ import annotations

import math

import numpy as np

from kits import _garage_util as U

NAME = "warehouse"
INDOOR = True
DESCRIPTION = "distribution warehouse: pallet racking aisles, loads, dock doors, high-bay lamps, forklifts"

FLOORS = ["ph:hangar_concrete_floor", "ph:garage_floor", "ph:concrete_floor_01", "ph:concrete_floor_02",
          "ph:painted_concrete", "ph:brushed_concrete", "ph:brushed_concrete_2", "acg:Concrete006",
          "acg:Concrete015", "acg:Concrete023", "acg:Concrete024", "acg:Concrete034", "ph:concrete_floor_worn_001"]
PANELS = ["acg:CorrugatedSteel005", "acg:CorrugatedSteel001", "acg:CorrugatedSteel007A", "acg:CorrugatedSteel002",
          "ph:factory_wall", "ph:box_profile_metal_sheet", "ph:corrugated_iron", "ph:corrugated_iron_02",
          "acg:CorrugatedSteel006A", "acg:CorrugatedSteel004"]
PLINTH = ["ph:concrete_wall_001", "ph:concrete_wall_004", "ph:concrete_wall_008", "acg:Concrete013",
          "acg:Concrete002", "ph:concrete_slab_wall", "ph:concrete_brick_wall_001", "acg:PaintedBricks001",
          "acg:PaintedBricks002", "ph:factory_brick"]
SHUTTERS = ["ph:painted_metal_shutter", "ph:worn_shutter", "ph:rusty_metal_shutter", "ph:rusted_shutter",
            "acg:CorrugatedSteel009"]
OLD_FLOORS = ["ph:concrete_floor_worn_001", "ph:concrete_floor_02", "ph:dirty_concrete", "ph:garage_floor",
              "acg:Concrete031", "acg:Concrete044A", "ph:concrete_floor_damaged_01"]
BRICKS = ["ph:factory_brick", "ph:brick_wall_08", "ph:brick_wall_09", "ph:brick_wall_001", "ph:red_brick_03",
          "ph:large_red_bricks", "acg:PaintedBricks001", "acg:PaintedBricks003", "ph:brick_4"]
#: Objaverse forklifts (CC-BY; <= 60k triangles) for some of the trucks, and props -- steel drums,
#: gas bottles, generators, a tarped pallet, a step ladder -- checked on the catalogue sheets
OBJ_FORKLIFTS = ["obj:060f3f8bc7de4e6ca2f348d414702e9d", "obj:bb8b8dab8b4c4fc1ac6af310b70f1b6e",
                 "obj:d659c366e24643f0b72ad0fe1ef1a60b", "obj:9ad80a7bd90b483f8dbf0e293251cff7",
                 "obj:b7bea04856e1401898250f313131659f", "obj:ce8b2c714b2f472bbac58c6eab088ba2"]
OBJ_PROPS = ["obj:205a7012c31742afb2b81d2cbfe0f2b2", "obj:c9b66b0949f247c4ae78338053650ee2",
             "obj:e5fccac2cc97468f90048944188b1b7a", "obj:4da7e80c3b6a42328d48dd811642db6e",
             "obj:a68f90323fd44cf7a22c0afe30d6a2f8", "obj:f1075a710e554e21b9b678a053e1f9ea",
             "obj:83468fa4ed6742ddae1a9a79c764f883", "obj:e788acc021ac45a598178d24ce196737",
             "obj:423cfdbd612c486aada7bb1bc44695a9", "obj:a6c7477cc905448fa16310be52a989c1",
             "obj:f34ba6c159e04152ac1883caaca2ae18", "obj:f394f4f7f446474fa65e3217923bdaaf",
             "obj:ab550a0d20c147a08b9cf0ee97c50188", "obj:d9d7989ac61848a09eeb60708dc0d941"]
UPRIGHT_RGB = [(0.05, 0.2, 0.55), (0.05, 0.2, 0.55), (0.9, 0.35, 0.03), (0.35, 0.36, 0.38), (0.1, 0.4, 0.2),
               (0.85, 0.65, 0.05), (0.55, 0.56, 0.58)]
BEAM_RGB = [(0.9, 0.35, 0.03), (0.9, 0.35, 0.03), (0.9, 0.35, 0.03), (0.85, 0.65, 0.05), (0.05, 0.2, 0.55),
            (0.8, 0.1, 0.05)]


def _pick(ctx, keys, fallback_cats=None):
    ks = [k for k in keys if k in ctx.lib.items]
    if not ks and fallback_cats:
        ks = ctx.lib.materials(cats_any=fallback_cats)
    return ctx.lib.pick(ctx.rng, ks)


class Racking:
    """Selective pallet racking, one row at a time (along x)."""

    def __init__(self, ctx, B, loads, occ):
        rng = ctx.rng
        self.ctx, self.B, self.loads, self.occ = ctx, B, loads, occ
        ur = UPRIGHT_RGB[int(rng.integers(len(UPRIGHT_RGB)))]
        br = BEAM_RGB[int(rng.integers(len(BEAM_RGB)))]
        # powder-coated steel: clean saturated paint, satin, slightly rounded edges
        self.upright = U.plain(ur, rough=0.35, metal=0.2, name="rack_upright", bevel=0.006)
        self.beam = U.plain(br, rough=0.3, metal=0.2, name="rack_beam", bevel=0.006)
        self.steel = U.plain((0.5, 0.5, 0.52), rough=0.4, metal=1.0, name="rack_steel")
        self.label = U.plain((0.85, 0.85, 0.8) if rng.random() < 0.6 else (0.9, 0.75, 0.1), rough=0.4, name="label")
        self.guard = U.plain((0.9, 0.7, 0.05), rough=0.4, name="end_guard", bevel=0.01)
        self.bw = float(rng.choice([2.7, 2.7, 2.8, 3.3]))            # clear bay width
        self.dep = float(rng.choice([1.0, 1.1, 1.1]))
        self.z1 = float(rng.uniform(1.4, 2.0))
        self.pitch = float(rng.uniform(1.5, 2.0))
        self.n_pal = 3 if self.bw < 3.0 else 4 if self.bw > 3.2 else 3
        self.bays = 0
        self.budget = None                            # in-zone surface (m^2) past which loads thin out
        self.thinned = 0                              # loads left out for the budget

    def levels(self, top):
        return [0.0] + list(np.arange(self.z1, top - 0.8, self.pitch))

    def row(self, y_face, side, x_lo, x_hi, top, coll="structure", roof=False, x_ph=0.0, check=True):
        """A single-deep row: its aisle face at ``y_face``, extending ``side``
        (+1 / -1) away from the aisle. Bays that are not free are left out."""
        ctx, B = self.ctx, self.B
        rng = ctx.rng
        pitch = self.bw + 0.1
        yc = y_face + side * self.dep / 2
        xs = np.arange(x_lo + (x_ph % pitch), x_hi - pitch + 1e-6, pitch)
        kept = []
        for x in xs:
            c = (x + pitch / 2, yc)
            half = (pitch / 2, self.dep / 2)
            if check and ctx.in_zone(c, pad=2.0) and not U.free_rect(ctx, c, half, 0.0, margin=0.35, step=0.4):
                continue
            kept.append(x)
        frames = set()
        lv = self.levels(top)
        run = None                                    # claims: per bay near the zone, one per run elsewhere
        for x in kept:
            near = ctx.in_zone((x + pitch / 2, yc), pad=3.0)
            if near:
                ctx.claim_rect((x + pitch / 2, yc), (pitch / 2, self.dep / 2), 0.0, kind="rack")
            elif run is not None and abs(run[1] - x) < 1e-3:
                run[1] = x + pitch
            else:
                if run is not None:
                    ctx.claim_rect(((run[0] + run[1]) / 2, yc), ((run[1] - run[0]) / 2, self.dep / 2), 0.0,
                                   kind="rack_run")
                run = [x, x + pitch]
        if run is not None:
            ctx.claim_rect(((run[0] + run[1]) / 2, yc), ((run[1] - run[0]) / 2, self.dep / 2), 0.0, kind="rack_run")
        for x in kept:
            frames.add(round(x, 4))
            frames.add(round(x + pitch, 4))
            xc = x + pitch / 2
            for z in lv[1:]:
                for yb in (y_face + side * 0.05, y_face + side * (self.dep - 0.05)):
                    B.box((xc, yb, z - 0.065), (self.bw / 2 + 0.05, 0.025, 0.065), self.beam, coll=coll, roof=roof)
                for i in range(self.n_pal):                 # location labels on the aisle-side beam
                    lx = x + 0.05 + (i + 0.5) * self.bw / self.n_pal
                    B.box((lx, y_face - side * 0.003, z - 0.06), (0.06, 0.004, 0.035), self.label, coll=coll,
                          roof=roof)
            # loads
            for k, z in enumerate(lv):
                zb = z + (0.0 if k == 0 else 0.0)
                h_avail = (lv[k + 1] - 0.15 - zb) if k + 1 < len(lv) else min(1.6, top + 0.8 - zb)
                if h_avail < 0.4:
                    continue
                full = self.budget is not None and B.area_in > self.budget and ctx.in_zone((x, yc), pad=1.0)
                for i in range(self.n_pal):
                    if rng.random() > self.occ:
                        continue
                    if full and rng.random() < 0.85:
                        self.thinned += 1
                        continue
                    px = x + 0.05 + (i + 0.5) * self.bw / self.n_pal + float(rng.normal(0, 0.02))
                    py = yc + float(rng.normal(0, 0.02))
                    pw = self.bw / self.n_pal - 0.1
                    self.loads.put(B, px, py, zb, math.pi / 2 + float(rng.normal(0, 0.01)),
                                   h_avail - 0.05, coll=coll, pd=(self.dep + 0.1, min(pw, 1.0)))
            self.bays += 1
        # yellow end protectors where a run of bays ends (aisle ends, cross aisles, gaps)
        ks = set(round(x, 4) for x in kept)
        for x in kept:
            for xe, s_ in ((x, -1), (x + pitch, 1)):
                nxt = round(x + s_ * pitch, 4)
                if nxt not in ks and ctx.in_zone((xe, yc), pad=8.0):
                    B.box((xe + s_ * 0.12, yc, 0.2), (0.05, self.dep / 2 + 0.05, 0.2), self.guard, coll=coll,
                          roof=roof)
        for x in sorted(frames):
            for yu in (y_face + side * 0.04, y_face + side * (self.dep - 0.04)):
                B.box((x, yu, top / 2), (0.045, 0.04, top / 2), self.upright, coll=coll, roof=roof)
                B.box((x, yu, 0.005), (0.08, 0.08, 0.005), self.steel, coll=coll, roof=roof)
            # frame bracing: horizontals and diagonals in the frame plane
            ya, yb = y_face + side * 0.08, y_face + side * (self.dep - 0.08)
            zz = np.arange(0.15, top - 0.2, 1.2)
            for j, z in enumerate(zz):
                B.cyl((x, ya, z), (x, yb, z), 0.018, self.upright, seg=6, coll=coll, roof=roof)
                if j + 1 < len(zz):
                    a, b = ((x, ya, z), (x, yb, zz[j + 1])) if j % 2 == 0 else ((x, yb, z), (x, ya, zz[j + 1]))
                    B.cyl(a, b, 0.014, self.upright, seg=6, coll=coll, roof=roof)
        return kept


def build(ctx) -> dict:
    rng, lk, lib = ctx.rng, ctx.looks, ctx.lib
    X0, Y0, X1, Y1 = ctx.bounds
    ZX0, ZY0, ZX1, ZY1 = ctx.zone
    B = U.Batch(ctx)
    clutter = float(ctx.knobs.get("clutter", 1.0))
    layout = rng.choice(["aisles", "aisles", "aisles", "staging", "mixed"])

    # --- sky + exposure: lamps light the hall, the sky comes in through doors/skylights ---
    if rng.random() < 0.7:
        hd = lib.hdris(cats_all={"outdoor"}, cats_none={"night", "indoor"},
                       tags_any={"industrial", "warehouse", "parking", "asphalt", "concrete", "road", "harbour",
                                 "dock", "construction", "hangar", "field"})
    else:
        hd = lib.hdris(cats_all={"indoor"}, tags_any={"warehouse", "industrial", "hangar", "hall", "factory"})
    sky = lk.sky(lib.pick(rng, hd))
    gain = U.exposure_gain(ctx, sky, float(rng.uniform(1.5, 3.0)))
    old = rng.random() < 0.3                           # an older brick shed rather than a modern panel box
    tint = [(1.0, 0.97, 0.93), (1.0, 0.92, 0.8), (0.95, 0.98, 1.0), (1.0, 0.95, 0.88)][int(rng.integers(4))]
    if old and rng.random() < 0.6:
        tint = (1.0, 0.82, 0.6)                        # metal-halide / sodium warmth
    lamps = U.Lamps(ctx, gain, rgb=tint)

    # --- hall ------------------------------------------------------------------------------------
    Hr = float(rng.uniform(7.0, 10.0))
    hx0 = ZX0 - float(rng.uniform(8.0, 30.0))
    hx1 = ZX1 + float(rng.uniform(8.0, 30.0))
    a0 = float(rng.uniform(-0.8, 0.8))
    wa = float(rng.uniform(3.2, 4.4))                  # aisle width
    flue = float(rng.uniform(0.2, 0.3))
    loads = U.Loads(ctx)
    occ = float(np.clip(rng.uniform(0.55, 0.9), 0, 1))
    rack = Racking(ctx, B, loads, occ)
    top = float(min(Hr - 1.2, rng.uniform(5.5, 9.0)))
    dd = 2 * rack.dep + flue                           # back-to-back row depth
    period = wa + dd
    # the dock wall: far (>= 8 m past the zone) or, for staging, just past the patch
    dock_side = int(rng.choice([-1, 1]))
    if layout == "staging":
        y_dock = (Y1 + float(rng.uniform(3.0, 5.5))) if dock_side > 0 else (Y0 - float(rng.uniform(3.0, 5.5)))
    else:
        y_dock = (ZY1 + float(rng.uniform(8.0, 20.0))) if dock_side > 0 else (ZY0 - float(rng.uniform(8.0, 20.0)))
    y_back = (ZY0 - float(rng.uniform(8.0, 25.0))) if dock_side > 0 else (ZY1 + float(rng.uniform(8.0, 25.0)))
    hy0, hy1 = min(y_dock, y_back), max(y_dock, y_back)
    stage_d = float(rng.uniform(12.0, 20.0)) if layout != "staging" else abs(y_dock - a0) - wa / 2
    # rack rows stop short of the staging area in front of the docks
    y_rack_lim = y_dock - dock_side * stage_d if layout != "staging" else a0 - dock_side * wa / 2

    # --- materials -------------------------------------------------------------------------------
    fk = [_pick(ctx, FLOORS if not old else OLD_FLOORS, {"concrete"})]
    if old or rng.random() < 0.3:
        fk.append(_pick(ctx, FLOORS if not old else OLD_FLOORS, {"concrete"}))
    floor = lk.ground_material(fk, tile_m=float(rng.uniform(3.0, 5.0)), puddles=0.0, blend_scale=14.0,
                               name="wh_floor")
    polish = float(rng.uniform(0.35, 0.8)) if not old else float(rng.uniform(0.85, 1.0))
    U.set_rough(floor, polish)
    ctx.ground(floor)
    panel_tints = [(1.3, 1.3, 1.3), (1.2, 1.2, 1.15), (1.1, 1.1, 1.1), None, (1.15, 1.1, 1.0)]
    pk = _pick(ctx, PANELS, {"corrugated steel"})
    panel = lk.pbr(pk, 1.2, tint=panel_tints[int(rng.integers(len(panel_tints)))], bevel=0.0, dirt=0.3,
                   name="wall_panel")
    if pk.startswith("acg:CorrugatedSteel"):
        U._desaturate(panel)
    if old:
        panel = lk.pbr(_pick(ctx, BRICKS, {"bricks"}), float(rng.uniform(1.2, 2.0)), bevel=0.0, dirt=0.5,
                       name="wall_brick")
    plinth = lk.pbr(_pick(ctx, PLINTH, {"concrete"}), 2.0, bevel=0.02, dirt=0.4, name="plinth")
    deck = lk.pbr(_pick(ctx, ["ph:corrugated_iron", "acg:CorrugatedSteel005", "ph:box_profile_metal_sheet",
                              "acg:CorrugatedSteel009"], {"corrugated steel"}), 1.0,
                  tint=(1.2, 1.2, 1.2) if rng.random() < 0.5 else None, name="roof_deck")
    if old:
        deck = lk.pbr(_pick(ctx, ["acg:Planks012", "acg:Planks020", "ph:brown_planks_05", "ph:wood_planks_dirt",
                                  "acg:Wood026"], {"planks"}), 1.5, name="roof_boards")
    steel_rgb = [(0.35, 0.36, 0.38), (0.05, 0.2, 0.45), (0.75, 0.75, 0.73), (0.55, 0.1, 0.05), (0.15, 0.3, 0.2)]
    girder = U.plain(steel_rgb[int(rng.integers(len(steel_rgb)))], rough=0.5, metal=0.3, name="girder", bevel=0.01)
    col_mat = girder if rng.random() < 0.6 else U.plain(steel_rgb[int(rng.integers(len(steel_rgb)))], rough=0.5,
                                                        metal=0.3, name="column_steel", bevel=0.01)
    housing = U.plain((0.3, 0.3, 0.32), rough=0.4, metal=0.6, name="highbay_housing")
    yellow = lk.pbr(_pick(ctx, ["acg:PaintedMetal001", "acg:PaintedMetal008"], {"painted metal"}), 0.8,
                    tint=(1.0, 0.85, 0.3), bevel=0.02, dirt=0.4, name="safety_yellow")
    black = U.plain((0.03, 0.03, 0.03), rough=0.5, name="black_paint")
    lane_y = U.decal((0.85, 0.62, 0.03), rough=0.45, wear=float(rng.uniform(0.05, 0.5)),
                     seed=float(rng.uniform(0, 99)), name="lane_yellow")
    lane_w = U.decal((0.8, 0.8, 0.78), rough=0.45, wear=float(rng.uniform(0.05, 0.5)), seed=float(rng.uniform(0, 99)),
                     name="lane_white")
    walk_g = U.decal((0.1, 0.4, 0.15), rough=0.45, wear=0.3, seed=float(rng.uniform(0, 99)), name="walkway_green")
    joint = U.plain((0.06, 0.06, 0.06), rough=0.8, name="joint")
    tyre = U.stain(strength=0.3, rough=0.5, name="tyre_mark", tone=(0.02, 0.02, 0.02))
    spill = U.stain(strength=0.6, rough=0.1, name="spill", tone=(0.05, 0.05, 0.045))
    steel = U.plain((0.55, 0.56, 0.58), rough=0.35, metal=1.0, name="galv")

    # --- the patch primitives ------------------------------------------------------------------------
    pwall = lk.pbr(_pick(ctx, PLINTH, {"concrete"}), 1.8, bevel=0.02, dirt=0.4, name="partition",
                   tint=(1.15, 1.15, 1.12) if rng.random() < 0.5 else None)
    conveyor = U.plain((0.3, 0.32, 0.35), rough=0.4, metal=0.5, name="conveyor", bevel=0.01)
    wrapped_box = loads.film if rng.random() < 0.5 else loads.carton[0]
    ctx.dress_primitives({"wall": pwall, "box": wrapped_box, "pole": col_mat, "bar": steel if rng.random() < 0.5
                          else U.plain((0.55, 0.03, 0.02), rough=0.4, name="pipe_red", bevel=0.005),
                          "rock": loads.film})
    for o in ctx.colls["obstacles"].objects:
        i = U.prim_index(o.name)                      # (o.name[4:6] took prim100.. for prim10)
        if i < 0 or i >= len(ctx.spec["primitives"]):
            continue
        p = ctx.spec["primitives"][i]
        if p["kind"] == "cyl_v" and p["r"] < 0.2:
            o.data.materials[0] = yellow                 # bollards / guard posts
        elif p["kind"] == "box" and p["c"][2] - p["half"][2] > 0.3:
            o.data.materials[0] = conveyor               # floating boxes: conveyor sections, ducts
        elif p["kind"] == "wall" and rng.random() < 0.35:
            o.data.materials[0] = panel                  # insulated partition panels

    # tall patch poles are building columns: continue them to the roof girders (above 4.3 m nothing can fly)
    sem = []
    n_ext = 0
    for p in ctx.spec["primitives"]:
        if p["kind"] == "cyl_v" and p["z1"] >= 4.3 and p["r"] >= 0.15:
            cx, cy = p["c"]
            ctx.cylinder((cx, cy, p["z1"]), (cx, cy, Hr), p["r"], col_mat, name="column_ext")
            ctx.cylinder((cx, cy, Hr - 0.25), (cx, cy, Hr), p["r"] + 0.12, col_mat, name="column_cap")
            n_ext += p["z1"] < U.cloud_zmax()
    if n_ext:
        sem.append(f"{n_ext} poles topping out at 4.3-{U.cloud_zmax():g} m continued up to the roof (in the cloud)")
    # the family's own free space (wall ends, round a pillar, a ring, a grove, ...) stays empty
    passages = U.claim_passages(ctx)
    # floating boxes (conveyor / duct sections) and spheres hang from the roof on bridles
    n_hung = U.hang_rods(ctx, B, Hr, steel)

    # --- building columns (claimed after the racks: they stand in the flues) -----------------------------
    col_dx = float(rng.choice([12.0, 15.0, 18.0, 24.0]))
    col_x = list(np.arange(hx0 + float(rng.uniform(4, col_dx)), hx1 - 2, col_dx))
    row_lines = []                                     # flue centre lines of the back-to-back rows
    k_lo = int(math.floor((hy0 - a0) / period)) - 1
    k_hi = int(math.ceil((hy1 - a0) / period)) + 1
    for k in range(k_lo, k_hi + 1):
        yf = a0 + wa / 2 + dd / 2 + k * period     # flue line of the double row above aisle k
        row_lines.append(yf)
    col_pos = []
    pole_cols = sum(1 for p in ctx.spec["primitives"] if p["kind"] == "cyl_v" and p["z1"] >= 4.9) >= 3
    for yf in row_lines[::int(rng.choice([1, 2]))]:
        if not (hy0 + 2 < yf < hy1 - 2):
            continue
        for xc in col_x:
            inside = X0 <= xc <= X1 and Y0 <= yf <= Y1
            if pole_cols and inside:
                continue
            if ctx.free((xc, yf), 0.25, 0.4):
                col_pos.append((xc, yf))

    # --- racks -----------------------------------------------------------------------------------------
    cross_w = float(rng.uniform(3.5, 4.5))
    cross_dx = float(rng.uniform(20.0, 35.0))
    cross = list(np.arange(hx0 + float(rng.uniform(6, cross_dx)), hx1, cross_dx))
    x_segments = []
    xs = hx0 + 1.5
    for cx in cross + [hx1 - 1.5]:
        if cx - cross_w / 2 - xs > 3.0:
            x_segments.append((xs, cx - cross_w / 2))
        xs = cx + cross_w / 2
    x_ph = float(rng.uniform(0, 3))
    n_rows = 0
    lo_side_roof = True                             # rows on the -y side outside the zone hide in the oblique

    def rack_ok(yf0, yf1):
        """A double row between faces yf0 < yf1 fits in the hall and not in the dock staging area."""
        if yf0 < hy0 + 1.0 or yf1 > hy1 - 1.0:
            return False
        if dock_side > 0 and yf1 > y_rack_lim:
            return False
        if dock_side < 0 and yf0 < y_rack_lim:
            return False
        return True

    # the cloud budget: what the primitives leave; loads thin out in the zone past 80 % of it
    # (rows go nearest the patch first), staging and block stacks stop at 90 %
    budget = U.area_budget() - U.cloud_area(ctx, B)
    rack.budget = B.area_in + 0.8 * budget
    stage_cap = B.area_in + 0.9 * budget
    yc_patch = (Y0 + Y1) / 2
    for k in sorted(range(k_lo, k_hi + 1), key=lambda k: abs(a0 + k * period + wa / 2 + dd / 2 - yc_patch)):
        c_aisle = a0 + k * period
        yf0 = c_aisle + wa / 2                       # lower face of the double row above this aisle
        yf1 = yf0 + dd
        if layout == "mixed" and (k % 3 == 1):
            continue                                 # a block-stack floor area instead of a rack pair
        if not rack_ok(yf0, yf1):
            continue
        in_zone = ctx.rect_meets_zone((hx0, yf0), (hx1, yf1))
        for (xa, xb) in x_segments:
            roof = lo_side_roof and yf1 < Y0 - 1.0 and not in_zone
            rack.row(yf0, +1, xa, xb, top, roof=roof, x_ph=x_ph)
            rack.row(yf1, -1, xa, xb, top, roof=roof, x_ph=x_ph)
        n_rows += 2
    # single-deep rows against the long walls
    for yw, sd in ((hy0 + 0.3, +1), (hy1 - 0.3, -1)):
        if (sd > 0 and dock_side < 0) or (sd < 0 and dock_side > 0):
            continue                                 # not along the dock wall
        for (xa, xb) in x_segments:
            rack.row(yw + sd * rack.dep, -sd, xa, xb, top, roof=(yw < Y0), x_ph=x_ph)
    for (xc, yf) in col_pos:
        _column(B, (xc, yf), Hr, col_mat, yellow)
        ctx.claim_rect((xc, yf), (0.2, 0.2), 0.0, kind="column")

    # --- shell ----------------------------------------------------------------------------------------
    ph = float(rng.uniform(1.2, 3.0))                  # concrete plinth height
    band = None
    if rng.random() < 0.6:
        glow = U.plain((0.8, 0.82, 0.8), rough=0.5, name="daylight_panel",
                       emission=(0.95, 0.97, 1.0, float(rng.uniform(3.0, 8.0)) * gain))
        bz0 = Hr - float(rng.uniform(1.8, 2.6))
        band = (glow, bz0, bz0 + float(rng.uniform(0.8, 1.4)), 0)
    for yw, is_dock in ((hy0, dock_side < 0), (hy1, dock_side > 0)):
        roof = yw < 0                                  # the -y wall hides for the oblique/top previews
        if is_dock:
            _dock_wall(ctx, B, yw, -1 if yw == hy0 else 1, hx0, hx1, Hr, ph, panel, plinth, yellow, black, lamps,
                       roof, rng)
        else:
            bd = None if band is None else (band[0], band[1], band[2], max(1, int((hx1 - hx0) / 7.0)))
            _wall(ctx, B, (hx0, yw), (hx1, yw), Hr, ph, panel, plinth, roof=roof, band=bd)
    for xw in (hx0, hx1):
        bd = None if band is None else (band[0], band[1], band[2], max(1, int((hy1 - hy0) / 7.0)))
        _wall(ctx, B, (xw, hy0), (xw, hy1), Hr, ph, panel, plinth, roof=False, band=bd)
    # roof: deck strips with skylight gaps, girders on the column lines, purlins
    sky_w = float(rng.uniform(1.2, 2.2))
    sky_dy = float(rng.choice([6.0, 8.0, 10.0, 12.0]))
    skylights = rng.random() < 0.8
    ys = hy0 - 0.3
    sky_y = list(np.arange(hy0 + float(rng.uniform(3, sky_dy)), hy1 - 2, sky_dy)) if skylights else []
    for sy in sky_y + [hy1 + 0.3]:
        y_end = sy - sky_w / 2 if sy < hy1 else hy1 + 0.3
        if y_end - ys > 0.2:
            ctx.box(((hx0 + hx1) / 2, (ys + y_end) / 2, Hr + 0.08), ((hx1 - hx0) / 2 + 0.3, (y_end - ys) / 2, 0.08),
                    deck, roof=True, name="roof")
        if sy < hy1:
            for s in (-1, 1):                           # skylight curb
                B.box(((hx0 + hx1) / 2, sy + s * sky_w / 2, Hr + 0.25), ((hx1 - hx0) / 2, 0.04, 0.2), girder,
                      roof=True)
        ys = sy + sky_w / 2
    gd = float(rng.uniform(0.7, 1.2))
    for xc in col_x:
        B.box((xc, (hy0 + hy1) / 2, Hr - gd / 2), (0.12, (hy1 - hy0) / 2, gd / 2), girder, roof=True)
        B.box((xc, (hy0 + hy1) / 2, Hr - gd + 0.02), (0.15, (hy1 - hy0) / 2, 0.02), girder, roof=True)
    for yp in np.arange(hy0 + 0.9, hy1, 1.8):
        B.box(((hx0 + hx1) / 2, yp, Hr - 0.12), ((hx1 - hx0) / 2, 0.035, 0.1), girder, roof=True)

    # --- high-bay lamps over the aisles ---------------------------------------------------------------------
    disc = rng.random() < 0.55
    lamp_dx = float(rng.uniform(5.0, 8.0))
    drop = float(rng.uniform(0.5, 2.0))
    zl = Hr - drop
    for k in range(k_lo, k_hi + 1):
        yc = a0 + k * period
        if not (hy0 + 1 < yc < hy1 - 1):
            continue
        for xl in np.arange(hx0 + lamp_dx / 2, hx1, lamp_dx):
            far = not ctx.in_zone((xl, yc), pad=12.0)
            if far and rng.random() < 0.5:
                continue
            if disc:
                lamps.disc(B, (xl, yc, Hr), housing, r=0.24, power=210.0, strength=25.0, drop=drop, rod_mat=steel)
            else:
                lamps.tube(B, (xl, yc, zl), 0.0, housing, length=1.4, power=150.0, strength=20.0, twin=True)
                B.cyl((xl, yc, zl), (xl, yc, Hr), 0.006, steel, seg=4, roof=True)
    # staging area lamps (a grid)
    ys0, ys1 = sorted((y_dock, y_dock - dock_side * stage_d))
    for yl in np.arange(ys0 + 3, ys1, 6.0):
        for xl in np.arange(hx0 + 4, hx1, 7.0):
            if disc:
                lamps.disc(B, (xl, yl, Hr), housing, r=0.24, power=210.0, strength=25.0, drop=drop, rod_mat=steel)
            else:
                lamps.tube(B, (xl, yl, zl), 0.0, housing, length=1.4, power=150.0, strength=20.0, twin=True)

    # --- floor ---------------------------------------------------------------------------------------
    U.joints(ctx, (hx0, hy0, hx1, hy1), float(rng.choice([5.0, 6.0, 7.5])), joint)
    for k in range(k_lo, k_hi + 1):
        c_aisle = a0 + k * period
        if not (hy0 + 1 < c_aisle < hy1 - 1):
            continue
        if rng.random() < 0.75:
            for s in (-1, 1):
                U.line(ctx, (hx0 + 1, c_aisle + s * (wa / 2 - 0.25)), (hx1 - 1, c_aisle + s * (wa / 2 - 0.25)),
                       0.1, lane_y)
        for _ in range(int(rng.integers(2, 6))):
            xa = float(rng.uniform(hx0, hx1 - 10))
            yt = c_aisle + float(rng.normal(0, wa / 6))
            ctx.strip((xa, yt), (xa + float(rng.uniform(5, 25)), yt + float(rng.normal(0, 0.2))),
                      float(rng.uniform(0.15, 0.25)), tyre, z=0.003, name="tyre")
    for cx in cross:
        for s in (-1, 1):
            U.line(ctx, (cx + s * (cross_w / 2 - 0.2), hy0 + 1), (cx + s * (cross_w / 2 - 0.2), hy1 - 1), 0.1, lane_y)
    # walkway along the dock staging area
    yw = y_dock - dock_side * float(rng.uniform(1.5, 2.5))
    if rng.random() < 0.7:
        U.line(ctx, (hx0 + 1, yw), (hx1 - 1, yw), 1.2, walk_g if rng.random() < 0.5 else lane_w)
    for _ in range(int(rng.integers(1, 4))):
        U.blob(ctx, (float(rng.uniform(X0, X1)), float(rng.uniform(Y0, Y1))), float(rng.uniform(0.3, 1.0)),
               float(rng.uniform(0.2, 0.6)), float(rng.uniform(0, 3)), spill)

    # --- staging area: pallet lanes, block stacks -----------------------------------------------------------
    n_stage = 0
    lane_box = lane_y if rng.random() < 0.5 else lane_w
    ys0, ys1 = sorted((y_dock - dock_side * 1.0, y_dock - dock_side * (stage_d - 1.0)))
    for xl in np.arange(hx0 + 5, hx1 - 5, float(rng.uniform(3.2, 4.2))):
        if rng.random() < 0.25:
            continue
        # a painted lane box perpendicular to the dock wall, pallets in it
        ya = y_dock - dock_side * 4.0
        yb = ya - dock_side * float(rng.uniform(6.0, stage_d - 5.0)) if stage_d > 11 else ya - dock_side * 5.0
        U.hatch(ctx, (xl, (ya + yb) / 2), (0.7, abs(yb - ya) / 2), 0.0, lane_box, pitch=100.0, w=0.08)
        n = int(abs(yb - ya) // 1.3)
        for j in range(n):
            if rng.random() > 0.35 + 0.4 * clutter:
                continue
            y = ya - dock_side * (0.7 + 1.3 * j)
            if not ctx.free((xl, y), 0.75, 0.4) or (B.area_in > stage_cap and ctx.in_zone((xl, y), pad=1.0)):
                continue
            t = loads.put(B, xl, y, 0.0, 0.0, float(rng.uniform(1.0, 1.8)), coll="scatter")
            if rng.random() < 0.25 and t < 1.6:
                loads.put(B, xl, y, t, float(rng.normal(0, 0.03)), 1.3, coll="scatter")
            ctx.claim_rect((xl, y), (0.62, 0.52), 0.0, kind="pallet")
            n_stage += 1
    if layout == "mixed":                               # block-stacked floor storage between rack pairs
        for k in range(k_lo, k_hi + 1):
            if k % 3 != 1:
                continue
            yf0 = a0 + k * period + wa / 2
            for xb in np.arange(hx0 + 3, hx1 - 3, 1.35):
                for yb in (yf0 + 0.6, yf0 + dd - 0.6):
                    if rng.random() > 0.7 or not ctx.free((xb, yb), 0.7, 0.35) or \
                            (B.area_in > stage_cap and ctx.in_zone((xb, yb), pad=1.0)):
                        continue
                    z = 0.0
                    for _ in range(int(rng.integers(1, 4))):
                        z = loads.put(B, xb, yb, z, math.pi / 2 + float(rng.normal(0, 0.02)), 1.4, coll="scatter",
                                      kind="wrapped")
                    ctx.claim_rect((xb, yb), (0.62, 0.52), 0.0, kind="block")

    # --- vehicles and props -------------------------------------------------------------------------
    # procedural trucks and the library's real ones draw from their own generators: whether a truck
    # is a real one, or the Objaverse catalogue is there at all, does not shift the scene's stream
    # (only the claims of a different footprint can move a later prop)
    fork = U.Forklift(ctx, rng=np.random.default_rng(int(rng.integers(1 << 31))))
    orng = np.random.default_rng(int(rng.integers(1 << 31)))
    obj_fl = lib.models(ids=OBJ_FORKLIFTS, max_tris=60000, h=(1.5, 2.3))
    p_obj = float(orng.choice([0.0, 0.3, 0.6]))
    n_fl = n_obj = 0
    for _ in range(ctx.count(2.0) * 20):
        if n_fl >= max(1, ctx.count(2.0)):
            break
        xy = ctx.uniform_xy(rect=(hx0 + 3, max(hy0 + 3, Y0 - 6), hx1 - 3, min(hy1 - 3, Y1 + 6)), pad=1.5)
        yaw = float(rng.choice([0.0, math.pi, math.pi / 2, -math.pi / 2])) + float(rng.normal(0, 0.1))
        if not ctx.free(xy, 1.6, 0.5):
            continue
        o_pick, o_i = orng.random() < p_obj, int(orng.integers(len(OBJ_FORKLIFTS)))
        if o_pick and obj_fl:
            k = obj_fl[o_i % len(obj_fl)]
            sx, sy, _ = lib.size(k)
            ctx.instance(k, xy, 0.0, yaw if sx >= sy else yaw - math.pi / 2, 1.0, coll="scatter")
            n_obj += 1
        else:
            fork.build(B, xy, yaw)
        ctx.claim_disc(xy, 1.6, kind="forklift")
        n_fl += 1
    jack = U.plain((0.8, 0.1, 0.05), rough=0.45, name="pallet_jack", bevel=0.01)
    for _ in range(ctx.count(1.5)):
        xy = ctx.uniform_xy(pad=1.0)
        if ctx.free(xy, 0.9, 0.5):
            _pallet_jack(B, xy, float(rng.uniform(0, 6.28)), jack, black, rng, loads)
            ctx.claim_disc(xy, 0.9, kind="jack")
    props = lib.models(ids=["industrial_storage_cart", "tool_cart", "hand_truck", "plastic_crate_01",
                            "plastic_crate_02", "plastic_crate_03", "cardboard_box_01", "Barrel_01", "Barrel_02",
                            "metal_trash_can", "plastic_container", "worn_metal_rack", "steel_frame_shelves_02",
                            "portable_generator", "industrial_pastic_container", "wooden_crate_02", "plastic_broom",
                            "trashbag", "metal_toolbox"], max_tris=40000)
    ctx.scatter(props, ctx.count(5.0), margin=0.4)                     # + the Objaverse ones below
    ctx.scatter(props, ctx.count(5.0), rect=(hx0 + 2, hy0 + 2, hx1 - 2, hy1 - 2), margin=0.4)
    ext = lib.models(ids=["korean_fire_extinguisher_01"])
    for (xc, yf) in col_pos:
        if ext and rng.random() < 0.3:
            for s in (-1, 1):
                p = (xc + s * 0.35, yf)
                if ctx.free(p, 0.15, 0.1):
                    ctx.instance(ext[0], p, 0.0, float(rng.uniform(0, 6.28)), 1.0, claim=True)
                    break
    # rack end guards (yellow) at row ends inside the zone
    n_hang = ctx.count(float(ctx.knobs.get("hang", 0.0)))
    hung = _hang(ctx, B, n_hang, Hr, steel, lamps, housing, black) if n_hang > 0 else 0

    # Objaverse props (<= 4 models a scene: packed size), last: whether the library has them
    # changes nothing else in the scene
    op = lib.models(ids=OBJ_PROPS, max_tris=40000)
    op = [op[int(i)] for i in orng.permutation(len(op))[:4]]
    n_oprops = len(ctx.scatter(op, ctx.count(1.2), margin=0.4)) if op else 0
    n_oprops += len(ctx.scatter(op, ctx.count(1.2), rect=(hx0 + 2, hy0 + 2, hx1 - 2, hy1 - 2), margin=0.4)) \
        if op else 0
    miss = [k for k in OBJ_FORKLIFTS + OBJ_PROPS if k not in lib.items]
    if miss:
        ctx.note(f"warehouse: {len(miss)} of {len(OBJ_FORKLIFTS) + len(OBJ_PROPS)} Objaverse models not in the library")

    B.flush()
    # what the dressing does to the tier-A family, read off the geometry the cloud will sample
    roof, cover = U.ceiling_semantics(U.overhead(ctx), U.prim_tops(ctx.spec), Hr, rect=ctx.bounds, what="roof",
                                      parts="rack levels, lamps, trays")
    sem = roof + sem
    if n_hung:
        sem.append(f"{n_hung} floating primitives hung from the roof on a splayed bridle (two 12 mm cables "
                   f"leaning {U.BRIDLE_SPLAY:g} deg out from the ends of a box / the sides of a sphere; the prism "
                   f"over the primitive stays free)")
    ctx.note(f"warehouse {layout} Hr={Hr:.1f} rows={n_rows} bays={rack.bays} stage={n_stage} forklifts={n_fl} "
             f"lamps={lamps.n} budget {budget:.0f} m2 thinned={rack.thinned} cloud_area~{U.cloud_area(ctx):.0f}m2")
    return {"sky": sky, "layout": layout, "roof_h": round(Hr, 2), "rack_top": round(top, 2), "aisle_w": round(wa, 2),
            "bays": rack.bays, "occupancy": round(occ, 2), "dock_side": dock_side, "floor": fk,
            "polish": round(polish, 2), "old": old, "forklifts": n_fl, "objaverse_forklifts": n_obj,
            "objaverse_props": n_oprops, "staged_pallets": n_stage, "lamps": lamps.n, "skylights": skylights,
            "hanging": hung, "passages": passages, "semantics_changed": sem, "ceiling_cover": cover}


# --- pieces ---------------------------------------------------------------------------------------

def _column(B, xy, Hr, mat, guard):
    """A steel H-column (flanges along x) with a yellow guard at its base."""
    x, y = xy
    B.box((x, y, Hr / 2), (0.15, 0.012, Hr / 2), mat)
    for s in (-1, 1):
        B.box((x + s * 0.14, y, Hr / 2), (0.01, 0.12, Hr / 2), mat)
    B.box((x, y, 0.5), (0.2, 0.12, 0.5), guard)


def _wall(ctx, B, a, b, Hr, ph, panel, plinth, roof=False, t=0.25, band=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    d = (b - a) / L
    c = (a + b) / 2
    yaw = math.atan2(d[1], d[0])
    if band is not None:                  # translucent daylight panels high on the wall
        mat, z0, z1, n = band
        nrm = np.array([-d[1], d[0]])
        for k in range(n):
            f0, f1 = (k + 0.15) / n, (k + 0.85) / n
            p0, p1 = a + d * L * f0, a + d * L * f1
            for sgn in (-1, 1):
                pc = (p0 + p1) / 2 + nrm * sgn * (t / 2 + 0.01)
                B.box((pc[0], pc[1], (z0 + z1) / 2), (L * (f1 - f0) / 2, 0.01, (z1 - z0) / 2), mat, yaw=yaw,
                      roof=roof)
    ctx.box((c[0], c[1], ph / 2), (L / 2 + t, t / 2 + 0.03, ph / 2), plinth, yaw=yaw, roof=roof, name="plinth")
    ctx.box((c[0], c[1], (ph + Hr + 0.3) / 2), (L / 2 + t, t / 2, (Hr + 0.3 - ph) / 2), panel, yaw=yaw, roof=roof,
            name="wall")


def _dock_wall(ctx, B, yw, sgn, hx0, hx1, Hr, ph, panel, plinth, yellow, black, lamps, roof, rng):
    """The dock wall: doors every ~4 m with shutters (some open onto a
    trailer or the yard), bumpers, guard posts, levelers, dock lights."""
    lk = ctx.looks
    shut_k = _pick(ctx, SHUTTERS, {"metal"})
    shut = lk.pbr(shut_k, float(rng.uniform(1.5, 2.2)), bevel=0.0, dirt=0.3, name="shutter",
                  tint=[(1.2, 1.2, 1.2), (0.9, 0.95, 1.2), (1.2, 1.1, 0.8), None][int(rng.integers(4))])
    rubber = U.plain((0.02, 0.02, 0.02), rough=0.7, name="dock_rubber", bevel=0.02)
    plate = lk.pbr(_pick(ctx, ["acg:DiamondPlate001", "acg:DiamondPlate008A", "acg:DiamondPlate005C"],
                         {"diamond plate"}), 0.6, proj="flat", name="leveler")
    trailer_in = lk.pbr(_pick(ctx, ["ph:plywood", "acg:Plywood001", "acg:Wood026"], {"wood"}), 1.0, name="trailer")
    trailer_out = U.plain((0.8, 0.8, 0.8), rough=0.4, name="trailer_skin", bevel=0.02)
    tl = U.Loads(ctx)
    t = 0.25
    door_w = float(rng.uniform(2.7, 3.0))
    door_h = float(rng.uniform(2.9, 3.4))
    pitch = float(rng.uniform(3.8, 4.6))
    xs = list(np.arange(hx0 + 4, hx1 - 4, pitch))
    # the wall as pieces between the doors, and a header over them
    x_prev = hx0 - t
    yc = yw + sgn * t / 2
    for xd in xs + [hx1 + t + door_w / 2]:
        x_end = xd - door_w / 2
        if x_end > x_prev:
            ctx.box(((x_prev + x_end) / 2, yc, ph / 2), ((x_end - x_prev) / 2, t / 2 + 0.03, ph / 2), plinth,
                    roof=roof, name="plinth")
            ctx.box(((x_prev + x_end) / 2, yc, (ph + door_h) / 2), ((x_end - x_prev) / 2, t / 2,
                                                                    (door_h - ph) / 2), panel, roof=roof, name="wall")
        x_prev = xd + door_w / 2
    ctx.box(((hx0 + hx1) / 2, yc, (door_h + Hr + 0.3) / 2), ((hx1 - hx0) / 2, t / 2, (Hr + 0.3 - door_h) / 2),
            panel, roof=roof, name="wall")
    for xd in xs:
        state = rng.choice(["closed", "closed", "closed", "trailer", "trailer", "yard"])
        face = yw - sgn * 0.01
        if state == "closed":
            B.box((xd, yc, door_h / 2), (door_w / 2, 0.05, door_h / 2), shut, roof=roof)
        else:
            B.box((xd, yc, door_h - 0.35), (door_w / 2, 0.08, 0.35), shut, roof=roof)
            if state == "trailer":
                ty = yw + sgn * (t + 6.8)
                zf = 0.0
                # trailer box behind the opening: plywood interior, white skin
                for s in (-1, 1):
                    B.box((xd + s * 1.28, ty, 1.35), (0.03, 6.6, 1.35), trailer_in, roof=roof)
                B.box((xd, ty, 2.72), (1.3, 6.6, 0.03), trailer_in, roof=roof)
                B.box((xd, ty + sgn * 6.6, 1.35), (1.3, 0.03, 1.35), trailer_in, roof=roof)
                B.box((xd, ty, zf + 0.02), (1.3, 6.6, 0.02), trailer_in, roof=roof)
                B.box((xd, ty, 2.76), (1.32, 6.62, 0.02), trailer_out, roof=roof)
                if rng.random() < 0.6:
                    for j in range(int(rng.integers(1, 6))):
                        tl.put(B, xd + float(rng.choice([-0.62, 0.62])), ty + sgn * (5.5 - 1.3 * j), 0.0, 0.0, 1.8,
                                coll="scatter")
        # leveler plate, bumpers, guard posts, dock light
        ctx.flat([(xd - 1.0, face), (xd + 1.0, face), (xd + 1.0, face - sgn * 2.4), (xd - 1.0, face - sgn * 2.4)],
                 plate, z=0.006, name="leveler")
        for s in (-1, 1):
            B.box((xd + s * 0.9, face - sgn * 0.06, 0.35), (0.12, 0.06, 0.22), rubber, roof=roof)
            px, py = xd + s * (door_w / 2 + 0.25), face - sgn * 0.3
            if ctx.free((px, py), 0.12, 0.2):
                B.cyl((px, py, 0.0), (px, py, 1.2), 0.1, yellow, seg=12)
        if ctx.in_zone((xd, face), pad=-0.1):
            ctx.claim_rect((xd, face - sgn * 0.3), (door_w / 2 + 0.4, 0.35), 0.0, kind="dock")


def _pallet_jack(B, xy, yaw, mat, black, rng, loads):
    R = U._rotz(yaw)
    c = np.array([xy[0], xy[1], 0.0])
    P = lambda u, v, z: R @ np.array([u, v, z]) + c  # noqa: E731
    for v in (-0.28, 0.28):
        B.box(P(0.55, v, 0.05), (0.58, 0.08, 0.035), mat, R=R, coll="scatter")
    B.box(P(-0.1, 0, 0.3), (0.12, 0.28, 0.25), mat, R=R, coll="scatter")
    a = P(-0.15, 0, 0.5)
    b = P(-0.55, 0, 1.2)
    B.cyl(a, b, 0.02, mat, seg=6, coll="scatter")
    B.cyl(P(-0.55, -0.15, 1.2), P(-0.55, 0.15, 1.2), 0.02, black, seg=6, coll="scatter")
    if rng.random() < 0.5:
        loads.put(B, P(0.55, 0, 0)[0], P(0.55, 0, 0)[1], 0.09, yaw + math.pi / 2, 1.4, coll="scatter")


def _hang(ctx, B, n, Hr, steel, lamps, housing, black):
    """Hanging into the flight band: aisle signs on chains, lamps on long
    drops, a lowered cable tray, a banner."""
    rng = ctx.rng
    X0, Y0, X1, Y1 = ctx.bounds
    sign_cols = [(0.05, 0.2, 0.6), (0.85, 0.65, 0.05), (0.8, 0.8, 0.8), (0.1, 0.45, 0.2)]
    done = 0
    for _ in range(n * 30):
        if done >= n:
            break
        kind = rng.choice(["sign", "sign", "lamp", "tray", "banner"])
        x = float(rng.uniform(X0 + 1, X1 - 1))
        y = float(rng.uniform(Y0 + 1, Y1 - 1))
        if kind == "sign":
            z = float(rng.uniform(2.6, 3.4))
            a, b = (x, y - 0.7), (x, y + 0.7)
            if not U.seg_clear(ctx, a, b, 0.5):
                continue
            m = U.plain(sign_cols[int(rng.integers(len(sign_cols)))], rough=0.4, name="aisle_sign", bevel=0.01)
            B.box((x, y, z + 0.25), (0.03, 0.7, 0.25), m)
            for q in (a, b):
                B.cyl((q[0], q[1], z + 0.5), (q[0], q[1], Hr), 0.006, steel, seg=4)
        elif kind == "lamp":
            z = float(rng.uniform(3.0, 3.6))
            if not ctx.free((x, y), 0.3, 0.5):
                continue
            lamps.disc(B, (x, y, Hr), housing, r=0.24, power=200.0, strength=40.0, drop=Hr - z, rod_mat=steel,
                       roof=False)
        elif kind == "tray":
            z = float(rng.uniform(3.1, 3.6))
            L = float(rng.uniform(4.0, 10.0))
            a, b = (x, y - L / 2), (x, y + L / 2)
            if not U.seg_clear(ctx, a, b, 0.6):
                continue
            B.box((x, y, z), (0.2, L / 2, 0.04), steel)
            for q in (a, b, (x, y)):
                B.cyl((q[0], q[1], z), (q[0], q[1], Hr), 0.008, steel, seg=4)
            for _k in range(3):
                B.cyl((x - 0.1 + 0.1 * _k, y - L / 2, z + 0.06), (x - 0.1 + 0.1 * _k, y + L / 2, z + 0.06), 0.02,
                      black, seg=6)
        else:
            z = float(rng.uniform(2.6, 3.2))
            a, b = (x - 1.5, y), (x + 1.5, y)
            if not U.seg_clear(ctx, a, b, 0.5):
                continue
            m = U.plain(sign_cols[int(rng.integers(len(sign_cols)))], rough=0.8, name="banner")
            B.box((x, y, z + 0.4), (1.5, 0.01, 0.4), m)
            for q in (a, b):
                B.cyl((q[0], q[1], z + 0.8), (q[0], q[1], Hr), 0.005, steel, seg=4)
        done += 1
    return done
