"""Building site: the patch is the site yard in front of a half-built
concrete frame, among stacked materials, site cabins and fences.

What makes it read as a site rather than boxes on a plane:

* **the building** -- a cast-in-place concrete frame (columns on a grid,
  slabs, edge beams, a lift core) 1-6 storeys high, the top storey still
  in progress: starter bars sticking out of the columns, shoring props
  under the fresh slab, edge protection, formwork stacks; blockwork infill
  on the lower floors;
* **scaffolding** along its face: standards, ledgers and transoms of
  galvanised tube on sole boards, timber boards and toe boards on every
  lift, guard rails, diagonal bracing, sometimes debris netting;
* **the yard**: pallets of bricks and blocks, cement bags, rebar bundles on
  bearers, timber stacks, concrete and PVC pipes, formwork panels, sand /
  gravel / soil heaps, skips, water-filled and concrete barriers, cones,
  Heras fence panels on their feet, portable toilets, site cabins (stacked,
  with stairs), an excavator, mobile scaffold towers;
* the ground: mud, gravel and tyre-tracked dirt with puddles, soft-edged
  gravel haul roads and a poured slab; tower cranes and the neighbouring
  blocks on the skyline.

Each scene draws which side the building is on, its height and bays,
concrete / scaffold / fence / cabin colours, the ground family, weather
(dry / wet / snow) and HDRI. Levels: ``clutter`` scales the yard;
``hang`` > 0 adds scaffold tubes across the flight band (standards with
ledgers at 1.8-3.3 m), crane loads hanging on the hook at 2.4-3.4 m and
temporary power cables strung between posts.
"""

from __future__ import annotations

import math

import numpy as np

from kits import _industrial_util as U

NAME = "construction"
INDOOR = False
DESCRIPTION = "building site: concrete frame + scaffolding, stacked materials, cabins, fences, cranes, muddy ground"

HDRI_DRY = ["ph:construction_yard", "ph:future_parking", "ph:portland_landing_pad", "ph:factory_yard",
            "ph:reinforced_concrete_02", "ph:abandoned_parking", "ph:brick_factory_01", "ph:german_town_street",
            "ph:suburban_parking_area", "ph:modern_buildings_2", "ph:canary_wharf", "ph:mall_parking_lot",
            "ph:learner_park", "ph:quarry_02", "ph:derelict_airfield_01", "ph:driving_school", "ph:bloem_olive_house",
            "ph:red_church", "ph:irish_institute", "ph:roofless_ruins", "ph:sunset_jhbcentral",
            "ph:rooftop_day", "ph:modern_buildings"]
HDRI_WET = ["ph:overcast_industrial_courtyard", "ph:hanger_exterior_cloudy", "ph:dam_bridge", "ph:beach_parking",
            "ph:teufelsberg_roof", "ph:stadium_exterior", "ph:alte_veste_station", "ph:abandoned_slipway",
            "ph:killesberg_park"]
HDRI_SNOW = ["ph:snowy_field", "ph:rural_winter_roadside", "ph:snow_field", "ph:snowy_park_01",
             "ph:snowy_cemetery", "ph:abandoned_hopper_terminal_01"]

MUD = ["ph:muddy_tracks", "ph:brown_mud", "ph:brown_mud_02", "ph:brown_mud_03", "acg:Ground036", "acg:Ground043",
       "acg:Ground044", "acg:Ground045", "acg:Ground046", "acg:Ground017", "ph:dry_ground_rocks",
       "ph:brown_mud_rocks_01", "acg:Ground015", "acg:Ground024"]
TRACKS = ["ph:aerial_mud_1", "ph:dirt_aerial_02", "ph:muddy_tracks", "acg:Ground017", "ph:red_sand"]
GRAVEL = ["acg:Gravel006", "acg:Gravel017", "acg:Gravel020", "acg:Gravel024", "acg:Gravel040", "acg:Gravel043",
          "ph:gravel_stones", "ph:concrete_debris", "ph:rubble", "acg:Gravel001", "acg:Gravel004", "acg:Ground110",
          "acg:Ground108"]
SAND = ["acg:Ground033", "acg:Ground034", "acg:Ground049A", "ph:gravelly_sand", "ph:sandy_gravel", "acg:Ground062S",
        "acg:Ground079S"]
SNOW = ["acg:Snow011", "acg:Snow012", "acg:Snow013", "ph:snow_03", "ph:snow_04", "ph:snow_05", "acg:Snow009C"]
CONC_WALL = ["ph:concrete_wall_006", "ph:concrete_wall_004", "ph:concrete_slab_wall", "ph:concrete_slab_wall_02",
             "ph:concrete_layers", "ph:wood_textured_concrete", "acg:Concrete034", "ph:rough_concrete",
             "ph:concrete_wall_001", "acg:Concrete002", "acg:Concrete006"]
CONC_FLOOR = ["ph:concrete_floor_02", "ph:concrete_floor_worn_001", "ph:dirty_concrete", "acg:Concrete031",
              "acg:Concrete044A", "ph:brushed_concrete", "ph:hangar_concrete_floor"]
BLOCK = ["ph:concrete_block_wall", "ph:concrete_block_wall_02"]
BRICKS = ["acg:Bricks076A", "acg:Bricks090", "acg:Bricks094", "acg:Bricks088", "acg:Bricks051", "acg:Bricks059",
          "ph:red_brick_03", "ph:brick_wall_02", "ph:brick_4", "ph:brick_wall_09"]
WOOD = ["ph:wood_planks", "ph:weathered_planks", "acg:Planks012", "acg:Wood049", "ph:raw_plank_wall",
        "ph:plywood", "acg:WoodFloor041"]
OSB = ["ph:oriented_strand_board", "ph:plywood", "acg:Chipboard004", "acg:Chipboard005"]
ROCK = ["ph:rock_face", "ph:rock_boulder_dry", "ph:rocks_ground_01", "ph:aerial_rocks_02", "acg:Rock030",
        "acg:Rock035", "acg:Rock022"]
RUST = ["acg:Rust004", "acg:Rust006", "acg:Rust008", "acg:Rust003"]
FACADES = ["acg:Bricks076A", "acg:Bricks094", "ph:yellow_plaster_02", "acg:PaintedPlaster017", "ph:beige_wall_001",
           "ph:painted_plaster_wall", "ph:plastered_wall", "ph:concrete_tile_facade", "ph:white_plaster_02",
           "acg:Plaster003", "ph:red_brick_03", "acg:Bricks059"]

CABIN_RGB = [(0.85, 0.85, 0.82), (0.55, 0.57, 0.58), (0.20, 0.30, 0.50), (0.25, 0.38, 0.28), (0.80, 0.62, 0.12),
             (0.72, 0.70, 0.62)]
NET_RGB = [(0.25, 0.5, 0.25), (0.2, 0.35, 0.6), (0.85, 0.85, 0.85), (0.15, 0.15, 0.15), (0.3, 0.45, 0.3)]
TOILET_RGB = [(0.15, 0.3, 0.65), (0.2, 0.45, 0.25), (0.55, 0.55, 0.55), (0.15, 0.4, 0.6), (0.8, 0.8, 0.78)]
MACHINE_RGB = [(0.85, 0.6, 0.05), (0.85, 0.45, 0.05), (0.8, 0.12, 0.08), (0.15, 0.35, 0.65), (0.2, 0.2, 0.2)]
#: what the kit adds on the patch primitives (course.json ``chosen``): nothing that changes a route
SEMANTICS = {"semantics_changed": False,
             "semantics_notes": ["starter bars only on in-situ walls reaching the cloud top (bars above the cloud)",
                                 "family core, wall ends and slot approaches kept clear of dressing",
                                 "L3 hang (scaffold gates, crane loads, cables) outside the family core"]}


def _pieces(ctx, keys):
    out = []
    for k in keys:
        fk, i = k.split("#")
        if fk in ctx.lib.items and any(p["index"] == int(i) for p in ctx.lib.pieces(fk)):
            out.append(k)
    return out


class Site:
    """The scene's palette and site builders."""

    def __init__(self, ctx, B, weather):
        self.ctx, self.B, self.rng = ctx, B, ctx.rng
        rng, lk = ctx.rng, ctx.looks
        self.weather = weather
        self.concrete = U.scan_native(ctx, CONC_WALL, 2.2, bevel=0.02, dirt=0.45)
        self.concrete2 = U.scan_native(ctx, CONC_WALL, 2.2, bevel=0.02, dirt=0.35)
        self.column = lk.pbr(U.pick(ctx, ["acg:Concrete034", "ph:rough_concrete", "acg:Concrete002", "acg:Concrete006",
                                          "acg:Concrete020", "acg:Concrete015", "acg:Concrete023"]), 2.0, bevel=0.02,
                             dirt=0.4)
        self.slab = U.scan_native(ctx, CONC_FLOOR, 3.0, bevel=0.02, dirt=0.3)
        self.block = U.scan_native(ctx, BLOCK, 2.0, bevel=0.01, dirt=0.3)
        self.brick = U.scan_native(ctx, BRICKS, 1.4, bevel=0.01, dirt=0.2)
        self.galv = U.galv(ctx, tile=1.0, bevel=0.0, dirt=0.4)
        self.board = lk.pbr(U.pick(ctx, WOOD, ctx.lib.materials(cats_any={"planks"})), 2.0, bevel=0.004, dirt=0.4)
        self.timber = lk.pbr(U.pick(ctx, WOOD, ctx.lib.materials(cats_any={"planks"})), 1.5, bevel=0.004, dirt=0.3,
                             tint=(1.0, 0.95, 0.85))
        self.osb = lk.pbr(U.pick(ctx, OSB), 1.2, bevel=0.004, dirt=0.3)
        hoard_rgb = [(0.15, 0.3, 0.2), (0.12, 0.2, 0.4), (0.8, 0.8, 0.78), (0.1, 0.1, 0.1), (0.65, 0.5, 0.25)][
            int(rng.integers(5))]
        self.hoard_mat = lk.pbr(U.pick(ctx, ["ph:plywood", "acg:Plastic013A", "ph:painted_metal_shutter"]), 2.4,
                               tint=hoard_rgb, bevel=0.005, dirt=0.4)
        self.rebar = lk.pbr(U.pick(ctx, RUST, ctx.lib.materials(cats_any={"rust"})), 0.4, bevel=0.0)
        self.yellow = U.paint(ctx, (0.85, 0.62, 0.05), rusty=0.3, tile=1.0, bevel=0.005)
        self.red = U.paint(ctx, (0.7, 0.1, 0.06), rusty=0.2, tile=1.0, bevel=0.005)
        self.white = lk.paint((0.85, 0.85, 0.83), rough=0.45, name="white_plastic")
        self.orange = lk.paint((0.9, 0.3, 0.03), rough=0.4, name="cone_orange")
        self.black = lk.paint((0.03, 0.03, 0.03), rough=0.6, name="rubber")
        self.glass = lk.glass(tint=(0.5, 0.55, 0.58), rough=0.1)
        net_key = U.pick(ctx, ["acg:Net001A", "acg:Net002A", "acg:Net003A", "acg:Net004A"])
        self.net = lk.pbr(net_key, None, tint=NET_RGB[int(rng.integers(len(NET_RGB)))])
        self.net_tile = float(ctx.lib.items[net_key].get("tile_m") or 1.3) if net_key else 1.3
        hk = U.pick(ctx, ["acg:Fence007A", "acg:Fence008A", "acg:Fence007B"])
        self.heras_mesh = lk.pbr(hk, None)
        self.heras_tile = float(ctx.lib.items[hk].get("tile_m") or 1.0) if hk else 1.0
        self.foot = lk.pbr(U.pick(ctx, ["acg:Concrete020", "acg:Concrete006", "ph:rough_concrete"]), 1.0,
                           dirt=0.4) if rng.random() < 0.5 else lk.paint((0.08, 0.08, 0.08), rough=0.8, name="foot")
        self.cabin_skin = lk.pbr(U.pick(ctx, ["acg:CorrugatedSteel005", "acg:CorrugatedSteel009",
                                              "ph:corrugated_iron", "ph:container_side"]), 1.2,
                                 tint=CABIN_RGB[int(rng.integers(len(CABIN_RGB)))], bevel=0.02, dirt=0.4)
        self.machine = U.paint(ctx, MACHINE_RGB[int(rng.integers(len(MACHINE_RGB)))], rusty=0.2, tile=1.5,
                               bevel=0.02)
        self.steel = U.paint(ctx, (0.25, 0.26, 0.27), rusty=0.5, tile=1.5, bevel=0.01)
        self.lamp = lk.paint((1.0, 0.95, 0.85), rough=0.3, name="lamp", emission=(1.0, 0.9, 0.75, 2.0))

    # --- the building -------------------------------------------------------------------------

    def frame_building(self, T, L, D, n, fh, bay, roof=False, top_frac=0.6, infill=0.4):
        """Local frame: facade along x (0..L) at y = 0, building towards +y."""
        rng = self.rng
        kw = {"roof": roof}
        nbx = max(1, int(round(L / bay)))
        nby = max(1, int(round(D / bay)))
        xs = np.linspace(0, L, nbx + 1)
        ys = np.linspace(0, D, nby + 1)
        top = n * fh
        cs = 0.2
        x_top = L * top_frac
        for x in xs:
            for y in ys:
                h = top if x <= x_top + 1e-6 else (n - 1) * fh
                if h <= 0:
                    continue
                T.box(self.concrete, (x, y, h / 2), (cs, cs, h / 2), name="column", **kw)
                if x > x_top + 1e-6 or (n >= 1 and rng.random() < 0.3):   # starter bars above the last pour
                    for sx in (-0.13, 0.13):
                        for sy in (-0.13, 0.13):
                            T.cyl(self.rebar, (x + sx, y + sy, h), (x + sx, y + sy, h + float(rng.uniform(0.8, 1.2))),
                                  0.012, seg=5, caps=(False, True), name="rebar", **kw)
        for k in range(1, n + 1):
            z = k * fh
            xe = L if k < n else x_top
            if xe <= 0.5:
                continue
            T.box(self.slab, (xe / 2, D / 2, z - 0.125), (xe / 2 + 0.3, D / 2 + 0.3, 0.125), name="slab", **kw)
            T.box(self.concrete, (xe / 2, 0.0, z - 0.35), (xe / 2 + 0.3, 0.2, 0.2), name="edge_beam", **kw)
        # lift / stair core
        cx = float(rng.uniform(0.3, 0.6)) * L
        core_h = top + (2.8 if top_frac > 0.5 else 0.0)
        for (a, b) in (((cx, D - 5), (cx + 4.5, D - 5)), ((cx, D - 0.2), (cx + 4.5, D - 0.2)),
                       ((cx, D - 5), (cx, D - 0.2)), ((cx + 4.5, D - 5), (cx + 4.5, D - 0.2))):
            a, b = np.asarray(a), np.asarray(b)
            c = (a + b) / 2
            Lw = float(np.linalg.norm(b - a))
            yaw = math.atan2(b[1] - a[1], b[0] - a[0])
            T.box(self.concrete2, (c[0], c[1], core_h / 2), (Lw / 2 + 0.12, 0.12, core_h / 2), yaw=yaw, name="core",
                  **kw)
        # blockwork infill on the facade of lower floors: sill walls and full bays
        for k in range(0, max(0, n - 1)):
            z0 = k * fh
            for x_a, x_b in zip(xs[:-1], xs[1:]):
                if rng.random() > infill:
                    continue
                full = rng.random() < 0.3
                h = fh - 0.45 if full else 0.95
                T.box(self.block, ((x_a + x_b) / 2, 0.0, z0 + h / 2), ((x_b - x_a) / 2 - cs, 0.1, h / 2),
                      name="infill", **kw)
        # shoring props under the newest slab, edge protection on it
        z0, z1 = (n - 1) * fh, n * fh - 0.25
        if n >= 1:
            for x in np.arange(0.6, x_top, 1.2):
                for y in np.arange(0.6, D, 1.2):
                    if rng.random() < 0.15:
                        continue
                    T.cyl(self.galv, (x, y, z0), (x, y, z1), 0.024, seg=6, name="prop", **kw)
            zt = n * fh
            pts = [(0.0, -0.3), (x_top + 0.3, -0.3)]
            for p in np.linspace(0, x_top, max(2, int(x_top / 2.0) + 1)):
                T.box(self.yellow, (p, -0.25, zt + 0.55), (0.03, 0.03, 0.55), name="edge_post", **kw)
            for zz in (zt + 0.5, zt + 1.0):
                T.cyl(self.yellow, (*pts[0], zz), (*pts[1], zz), 0.02, seg=6, name="edge_rail", **kw)
            # formwork panel stacks and a skip on the top slab
            for _ in range(int(rng.integers(1, 4))):
                px, py = float(rng.uniform(1, max(1.1, x_top - 2))), float(rng.uniform(2, max(2.1, D - 2)))
                hh = float(rng.uniform(0.3, 1.0))
                T.box(self.yellow, (px, py, zt + hh / 2), (1.35, 0.65, hh / 2), name="formwork", **kw)
                T.box(self.osb, (px, py, zt + hh + 0.01), (1.3, 0.6, 0.01), name="formwork", **kw)
        return xs

    def scaffold(self, xa, xb, yf, side, H, net=False, roof=False, bay=None):
        """Along the facade at ``y = yf`` (building on the ``side`` of it): two
        lines of standards (0.3 and 1.35 m off the face) up to ``H``."""
        B, rng = self.B, self.rng
        kw = {"roof": roof}
        bay = bay or float(rng.uniform(2.0, 2.5))
        n = max(1, int(round((xb - xa) / bay)))
        xs = np.linspace(xa, xb, n + 1)
        yi, yo = yf - side * 0.3, yf - side * 1.35
        lifts = np.arange(2.0, H + 0.01, 2.0)
        r = 0.024
        for x in xs:
            for y in (yi, yo):
                B.cyl(self.galv, (x, y, 0.03), (x, y, H + 1.1), r, seg=8, name="scaffold", **kw)
                B.box(self.board, (x, y, 0.02), (0.12, 0.3, 0.02), name="soleboard", **kw)
        for z in [0.15] + list(lifts):
            for y in (yi, yo):
                B.cyl(self.galv, (xa - 0.2, y, z), (xb + 0.2, y, z), r, seg=8, name="scaffold", **kw)
            for x in xs:
                B.cyl(self.galv, (x, yi + side * 0.15, z + 0.05), (x, yo - side * 0.15, z + 0.05), r, seg=8,
                      name="scaffold", **kw)
        for z in lifts:
            for x_a, x_b in zip(xs[:-1], xs[1:]):
                if rng.random() < 0.08:
                    continue
                B.box(self.board, ((x_a + x_b) / 2, (yi + yo) / 2, z + 0.1), ((x_b - x_a) / 2 + 0.1, 0.52, 0.019),
                      name="board", **kw)
            B.box(self.board, ((xa + xb) / 2, yo, z + 0.2), ((xb - xa) / 2, 0.02, 0.075), name="toeboard", **kw)
            for dz in (0.55, 1.05):
                B.cyl(self.galv, (xa, yo - side * 0.05, z + dz), (xb, yo - side * 0.05, z + dz), r, seg=8,
                      name="scaffold", **kw)
        for i in range(0, n, 3):            # diagonal bracing on the outer face
            for j, z in enumerate([0.15] + list(lifts[:-1])):
                x_a, x_b = xs[i], xs[min(i + 1, n)]
                a_, b_ = ((x_a, z), (x_b, z + 2.0)) if j % 2 == 0 else ((x_b, z), (x_a, z + 2.0))
                B.cyl(self.galv, (a_[0], yo - side * 0.06, a_[1]), (b_[0], yo - side * 0.06, b_[1]), r, seg=8,
                      name="scaffold", **kw)
        if net:
            B.panel(self.net, (xa, yo - side * 0.1), (xb, yo - side * 0.1), 0.4 if rng.random() < 0.5 else 2.2,
                    H + 1.0, tile=self.net_tile, name="net", **kw)

    # --- site furniture -------------------------------------------------------------------------

    def heras(self, a, b, check=True):
        """Heras panels (3.45 x 2.0 m, weld mesh in a tube frame) on feet a -> b."""
        ctx, B, rng = self.ctx, self.B, self.rng
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        n = max(1, int(round(L / 3.5)))
        d = (b - a) / max(L, 1e-9)
        nrm = np.array([-d[1], d[0]])
        pts = [a + (b - a) * i / n + nrm * float(rng.normal(0, 0.05)) for i in range(n + 1)]
        ok = [(not check) or ctx.free(p, 0.3, 0.2) for p in pts]
        placed = 0
        for i in range(n):
            if not (ok[i] and ok[i + 1]):
                continue
            p, q = pts[i], pts[i + 1]
            e = (q - p) / np.linalg.norm(q - p)
            p2, q2 = p + e * 0.04, q - e * 0.04
            B.panel(self.heras_mesh, p2, q2, 0.12, 2.0, tile=self.heras_tile, name="heras")
            for c in (p2, q2):
                B.cyl(self.galv, (*c, 0.08), (*c, 2.02), 0.02, seg=6, name="heras")
            for z in (0.12, 2.0):
                B.cyl(self.galv, (*p2, z), (*q2, z), 0.02, seg=6, caps=(False, False), name="heras")
            if check:
                ctx.claim_rect((p + q) / 2, (np.linalg.norm(q - p) / 2, 0.2), math.atan2(e[1], e[0]), kind="heras")
            placed += 1
        for i, p in enumerate(pts):
            if ok[i]:
                yaw = math.atan2(d[1], d[0]) + math.pi / 2
                B.hexa(self.foot, U.tapered(p, yaw, (0.33, 0.1), (0.25, 0.08), 0.0, 0.13), name="heras_foot")
        return placed

    def hoarding(self, a, b, h=2.4, far=False):
        """Painted plywood hoarding with posts (site boundary)."""
        B = self.B
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        d = (b - a) / L
        yaw = math.atan2(d[1], d[0])
        c = (a + b) / 2
        B.box(self.hoard_mat, (c[0], c[1], h / 2), (L / 2, 0.02, h / 2), yaw=yaw, name="hoarding", far=far)
        nrm = np.array([-d[1], d[0]])
        for t in np.arange(0, L, 2.4):
            q = a + d * t - nrm * 0.08
            B.box(self.timber, (q[0], q[1], h / 2), (0.05, 0.05, h / 2), yaw=yaw, name="hoarding", far=far)

    def pallet(self, c, yaw, z=0.0):
        B = self.B
        R = U.rotz(yaw)
        for i in range(5):
            o = R @ np.array([-0.53 + 0.265 * i, 0, 0])
            B.box(self.timber, (c[0] + o[0], c[1] + o[1], z + 0.133), (0.05, 0.4, 0.011), yaw=yaw, name="pallet")
        for j in (-1, 0, 1):
            o = R @ np.array([0, 0.33 * j, 0])
            B.box(self.timber, (c[0] + o[0], c[1] + o[1], z + 0.07), (0.6, 0.05, 0.05), yaw=yaw, name="pallet")
        return z + 0.144

    def unit_load(self, c, yaw, kind):
        """A pallet of bricks / blocks / cement bags (stacked 1-2 high)."""
        B, rng = self.B, self.rng
        levels = 2 if (kind != "cement" and rng.random() < 0.35) else 1
        z = 0.0
        for _ in range(levels):
            z = self.pallet(c, yaw, z)
            if kind == "bricks":
                h = float(rng.uniform(0.6, 0.9))
                B.box(self.brick, (c[0], c[1], z + h / 2), (0.55, 0.4, h / 2), yaw=yaw, name="bricks")
                z += h
            elif kind == "blocks":
                h = float(rng.uniform(0.6, 1.0))
                B.box(self.block, (c[0], c[1], z + h / 2), (0.58, 0.42, h / 2), yaw=yaw, name="blocks")
                z += h
            else:
                bag = _pieces(self.ctx, ["ph:cement_bag#0"])
                R = U.rotz(yaw)
                for layer in range(int(rng.integers(3, 7))):
                    for u, v in ((-0.3, -0.2), (0.3, -0.2), (-0.3, 0.25), (0.3, 0.25)):
                        o = R @ np.array([u, v, 0])
                        by = yaw + (math.pi / 2 if layer % 2 else 0.0) + float(rng.normal(0, 0.05))
                        if bag:
                            self.ctx.instance(bag[0], (c[0] + o[0], c[1] + o[1]), z, by, 1.0, coll="scatter")
                        else:
                            B.box(self.white, (c[0] + o[0], c[1] + o[1], z + 0.08), (0.3, 0.22, 0.08), yaw=by,
                                  name="bag")
                    z += 0.16
        self.ctx.claim_rect(c, (0.65, 0.48), yaw, kind=kind)
        return z

    def rebar_bundles(self, c, yaw):
        B, rng = self.B, self.rng
        R = U.rotz(yaw)
        f, s = R @ np.array([1.0, 0, 0]), R @ np.array([0, 1.0, 0])
        L = float(rng.uniform(5.0, 9.0))
        c3 = np.array([c[0], c[1], 0.0])
        nb = int(rng.integers(2, 5))
        for u in (-L * 0.35, 0.0, L * 0.35):
            B.box(self.timber, c3 + f * u + U.UP * 0.05, (0.05, 0.25 * nb + 0.2, 0.05), yaw=yaw, name="bearer")
        for k in range(nb):
            cb = c3 + s * (k - (nb - 1) / 2) * 0.45 + U.UP * 0.2
            for i in range(14):
                a = 2 * math.pi * i / 14 * 2.3
                rr = 0.05 + 0.06 * ((i * 7) % 5) / 5
                o = s * rr * math.cos(a) + U.UP * rr * math.sin(a) * 0.8
                p = cb + o + f * float(rng.normal(0, 0.1))
                B.cyl(self.rebar, p - f * L / 2, p + f * L / 2, 0.012, seg=5, name="rebar")
        self.ctx.claim_rect(c, (L / 2 + 0.1, 0.25 * nb + 0.25), yaw, kind="rebar")

    def timber_stack(self, c, yaw):
        B, rng = self.B, self.rng
        R = U.rotz(yaw)
        f, s = R @ np.array([1.0, 0, 0]), R @ np.array([0, 1.0, 0])
        L = float(rng.uniform(3.0, 5.0))
        c3 = np.array([c[0], c[1], 0.0])
        w, t = float(rng.choice([0.1, 0.15, 0.2])), float(rng.choice([0.05, 0.075, 0.1]))
        nw = int(rng.integers(5, 10))
        z = 0.0
        for u in (-L * 0.35, L * 0.35):
            B.box(self.timber, c3 + f * u + U.UP * 0.05, (0.05, nw * w / 2 + 0.1, 0.05), yaw=yaw, name="bearer")
        z = 0.1
        for layer in range(int(rng.integers(3, 9))):
            for k in range(nw):
                p = c3 + s * (k - (nw - 1) / 2) * (w + 0.005) + U.UP * (z + t / 2) + f * float(rng.normal(0, 0.03))
                B.box(self.timber, p, (L / 2, w / 2, t / 2), yaw=yaw, name="timber")
            z += t
            if layer % 3 == 2:
                for u in (-L * 0.35, L * 0.35):
                    B.box(self.timber, c3 + f * u + U.UP * (z + 0.02), (0.02, nw * w / 2, 0.02), yaw=yaw, name="stick")
                z += 0.04
        self.ctx.claim_rect(c, (L / 2 + 0.1, nw * w / 2 + 0.15), yaw, kind="timber")

    def pipe_stack(self, c, yaw):
        B, rng = self.B, self.rng
        R = U.rotz(yaw)
        f, s = R @ np.array([1.0, 0, 0]), R @ np.array([0, 1.0, 0])
        c3 = np.array([c[0], c[1], 0.0])
        if rng.random() < 0.5:        # concrete drainage pipes
            r, L, mat, n0 = float(rng.uniform(0.3, 0.55)), 2.4, self.concrete2, int(rng.integers(2, 4))
        else:                         # PVC
            r, L, n0 = float(rng.choice([0.08, 0.11, 0.16, 0.2])), 6.0, int(rng.integers(3, 6))
            mat = self.ctx.looks.paint([(0.75, 0.35, 0.1), (0.35, 0.35, 0.37), (0.1, 0.1, 0.1), (0.2, 0.4, 0.2)][
                int(rng.integers(4))], rough=0.35, name="pvc")
        for row in range(n0):
            for k in range(n0 - row):
                v = (k - (n0 - row - 1) / 2) * 2 * r
                p = c3 + s * v + U.UP * (r + row * r * 1.73)
                B.cyl(mat, p - f * L / 2, p + f * L / 2, r, r, seg=20 if r > 0.25 else 12, caps=(True, True),
                      name="pipes")
                if r > 0.25:          # hollow look: a dark disc at each end
                    B.cyl(self.black, p - f * (L / 2 + 0.005), p - f * (L / 2 - 0.02), r * 0.8, seg=20, name="pipes")
                    B.cyl(self.black, p + f * (L / 2 - 0.02), p + f * (L / 2 + 0.005), r * 0.8, seg=20, name="pipes")
        self.ctx.claim_rect(c, (L / 2 + 0.1, n0 * r + 0.1), yaw, kind="pipes")

    def formwork_stack(self, c, yaw):
        """Framed formwork panels (2.7 x 1.2 m) stacked face down on bearers."""
        B, rng = self.B, self.rng
        n = int(rng.integers(4, 14))
        z = 0.05
        for i in range(n):
            o = rng.normal(0, 0.02, 2)
            p = (c[0] + o[0], c[1] + o[1])
            B.box(self.yellow, (p[0], p[1], z + 0.05), (1.35, 0.6, 0.05), yaw=yaw + float(rng.normal(0, 0.01)),
                  name="formwork")
            B.box(self.osb, (p[0], p[1], z + 0.105), (1.33, 0.58, 0.009), yaw=yaw, name="formwork")
            z += 0.118
        for u in (-0.9, 0.9):
            o = U.rotz(yaw) @ np.array([u, 0, 0])
            B.box(self.timber, (c[0] + o[0], c[1] + o[1], 0.025), (0.05, 0.6, 0.025), yaw=yaw, name="bearer")
        self.ctx.claim_rect(c, (1.45, 0.7), yaw, kind="formwork")

    def heap(self, c, R, kind=None):
        rng, lk = self.rng, self.ctx.looks
        kind = kind or str(rng.choice(["sand", "gravel", "soil", "soil"]))
        keys = {"sand": SAND, "gravel": GRAVEL, "soil": MUD}[kind]
        if self.weather == "snow" and rng.random() < 0.6:
            keys = SNOW
        mat = lk.pbr(U.pick(self.ctx, keys, self.ctx.lib.materials(cats_any={"ground"})), 1.5, vary=0.2, jitter=0.0,
                     bump=0.6)
        h = R * float(rng.uniform(0.35, 0.6))
        ry = R * float(rng.uniform(0.6, 1.0))
        self.B.mound(mat, c, R, h, ry=ry, yaw=float(rng.uniform(0, 3.14)), name="heap")
        self.ctx.claim_disc(c, max(R, ry) * 1.05, kind="heap")

    def skip(self, c, yaw):
        B, rng = self.B, self.rng
        col = U.paint(self.ctx, [(0.8, 0.55, 0.05), (0.2, 0.35, 0.2), (0.15, 0.25, 0.5), (0.7, 0.3, 0.05)][
            int(rng.integers(4))], rusty=0.7, tile=1.0, bevel=0.01)
        B.hexa(col, U.tapered(c, yaw, (1.2, 0.8), (1.8, 0.85), 0.05, 1.2), name="skip")
        fill = self.ctx.looks.pbr(U.pick(self.ctx, ["ph:rubble", "ph:concrete_debris", "acg:Gravel020"]), 1.0)
        B.mound(fill, (c[0], c[1], 1.0), 1.3, 0.35, ry=0.6, yaw=yaw, name="skip_fill")
        for s_ in (-1, 1):
            o = U.rotz(yaw) @ np.array([s_ * 0.6, 0, 0])
            B.box(col, (c[0] + o[0], c[1] + o[1], 0.03), (0.08, 0.85, 0.03), yaw=yaw, name="skip")
        self.ctx.claim_rect(c, (1.9, 0.95), yaw, kind="skip")

    def toilets(self, c, yaw, n):
        B, rng = self.B, self.rng
        R = U.rotz(yaw)
        col = self.ctx.looks.paint(TOILET_RGB[int(rng.integers(len(TOILET_RGB)))], rough=0.5, name="toilet")
        roof = self.ctx.looks.paint((0.85, 0.85, 0.82), rough=0.5, name="toilet_roof")
        for i in range(n):
            o = R @ np.array([(i - (n - 1) / 2) * 1.18, 0, 0])
            p = (c[0] + o[0], c[1] + o[1])
            B.box(col, (p[0], p[1], 1.1), (0.56, 0.56, 1.1), yaw=yaw, name="toilet")
            B.hexa(roof, U.tapered(p, yaw, (0.6, 0.6), (0.45, 0.45), 2.2, 2.38), name="toilet")
            d = R @ np.array([0, -0.565, 0])
            B.box(self.black, (p[0] + d[0], p[1] + d[1], 1.05), (0.44, 0.008, 0.95), yaw=yaw, name="toilet_door")
            B.box(col, (p[0] + d[0] * 1.01, p[1] + d[1] * 1.01, 1.05), (0.42, 0.008, 0.93), yaw=yaw, name="toilet")
        self.ctx.claim_rect(c, (0.6 * n + 0.05, 0.62), yaw, kind="toilets")

    def cabins(self, c, yaw, n, stack=False, roof=False):
        """Site cabins (6.0 x 2.45 x 2.6) in a row, optionally stacked with an outside stair."""
        B = self.B
        R = U.rotz(yaw)
        kw = {"roof": roof}
        levels = 2 if stack else 1
        for lev in range(levels):
            z0 = lev * 2.62
            for i in range(n - (1 if lev and n > 1 else 0)):
                o = R @ np.array([(i - (n - 1) / 2) * 6.1, 0, 0])
                p = np.array([c[0] + o[0], c[1] + o[1]])
                B.box(self.cabin_skin, (p[0], p[1], z0 + 1.35), (3.0, 1.22, 1.22), yaw=yaw, name="cabin", **kw)
                B.box(self.galv, (p[0], p[1], z0 + 2.6), (3.05, 1.27, 0.04), yaw=yaw, name="cabin_roof", **kw)
                B.box(self.steel, (p[0], p[1], z0 + 0.07), (3.05, 1.25, 0.07), yaw=yaw, name="cabin_base", **kw)
                for u in (-1.8, 0.0, 1.8):
                    w = R @ np.array([u, -1.235, 0])
                    if u == 0.0:
                        B.box(self.steel, (p[0] + w[0], p[1] + w[1], z0 + 1.15), (0.45, 0.02, 1.0), yaw=yaw,
                              name="cabin_door", **kw)
                    else:
                        B.box(self.glass, (p[0] + w[0], p[1] + w[1], z0 + 1.6), (0.6, 0.02, 0.45), yaw=yaw,
                              name="cabin_window", **kw)
        if stack:
            f = R @ np.array([1.0, 0, 0])
            s = R @ np.array([0, -1.0, 0])
            end = np.array([c[0], c[1]]) + f[:2] * ((n - 1) / 2 * 6.1 - 3.0) + s[:2] * 1.72
            U.stair(B, self.steel, self.galv, end - f[:2] * 3.1, yaw, 2.62, width=0.9, rail=self.yellow, **kw)
            B.box(self.galv, (end[0] + f[0] * 0.5, end[1] + f[1] * 0.5, 2.64), (0.5, 0.48, 0.03), yaw=yaw,
                  name="landing", **kw)
            self.ctx.claim_rect(end - f[:2] * 1.3, (1.9, 0.5), yaw, kind="stair")
        self.ctx.claim_rect(c, (3.1 * n + 0.1, 1.3), yaw, kind="cabins")

    def cone(self, xy):
        B = self.B
        x, y = xy
        B.box(self.black, (x, y, 0.015), (0.19, 0.19, 0.015), name="cone")
        B.cyl(self.orange, (x, y, 0.03), (x, y, 0.72), 0.15, 0.025, seg=14, name="cone")
        B.cyl(self.white, (x, y, 0.4), (x, y, 0.52), 0.093, 0.075, seg=14, caps=(False, False), name="cone")
        self.ctx.claim_disc(xy, 0.2, kind="cone")

    def water_barriers(self, a, b):
        """Red/white water-filled barriers end to end a -> b."""
        B, ctx = self.B, self.ctx
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        d = (b - a) / max(L, 1e-9)
        yaw = math.atan2(d[1], d[0])
        n = int(L / 1.2)
        placed = 0
        for i in range(n):
            c = a + d * (0.6 + 1.2 * i)
            if not ctx.free(c, 0.6, 0.15):
                continue
            mat = self.red if i % 2 == 0 else self.white
            B.hexa(mat, U.tapered(c, yaw, (0.6, 0.22), (0.5, 0.1), 0.0, 0.8), name="barrier")
            ctx.claim_rect(c, (0.62, 0.25), yaw, kind="barrier")
            placed += 1
        return placed

    def tower(self, c, yaw, H):
        """A mobile scaffold tower: 4 standards, ledgers every 2 m, top platform with rails."""
        B = self.B
        R = U.rotz(yaw)
        hx, hy = 1.25, 0.7
        pts = [np.array([c[0], c[1]]) + (R @ np.array([sx * hx, sy * hy, 0]))[:2]
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        for p in pts:
            B.cyl(self.galv, (*p, 0.15), (*p, H + 1.0), 0.024, seg=8, name="tower")
            B.cyl(self.black, (*p, 0.0), (*p, 0.15), 0.06, seg=10, name="castor")
        for z in list(np.arange(0.4, H, 2.0)) + [H, H + 0.5, H + 1.0]:
            for p, q in zip(pts, pts[1:] + pts[:1]):
                B.cyl(self.galv, (*p, z), (*q, z), 0.022, seg=6, name="tower")
        B.box(self.board, (c[0], c[1], H + 0.02), (hx, hy, 0.02), yaw=yaw, name="tower_deck")
        for p, q in list(zip(pts, pts[1:] + pts[:1]))[::2]:     # diagonal braces on two faces
            B.cyl(self.galv, (*p, 0.4), (*q, min(H, 2.4)), 0.02, seg=6, name="tower")
        self.ctx.claim_rect(c, (hx + 0.1, hy + 0.1), yaw, kind="tower")

    def excavator(self, c, yaw, arm=0.0, far=False, roof=False):
        """A tracked excavator (~7 m reach): tracks, house, cab, boom, stick, bucket."""
        B, rng = self.B, self.rng
        kw = {"far": far, "roof": roof}
        R = U.rotz(yaw)
        f, s = R @ np.array([1.0, 0, 0]), R @ np.array([0, 1.0, 0])
        c3 = np.array([c[0], c[1], 0.0])
        for sgn in (-1, 1):
            B.box(self.black, c3 + s * sgn * 1.15 + U.UP * 0.45, (1.9, 0.3, 0.42), yaw=yaw, name="track", **kw)
            for u in (-1.9, 1.9):
                B.cyl(self.black, c3 + s * sgn * 0.85 + f * u + U.UP * 0.45, c3 + s * sgn * 1.45 + f * u + U.UP * 0.45,
                      0.42, seg=16, name="track", **kw)
        B.box(self.steel, c3 + U.UP * 0.95, (1.0, 0.9, 0.12), yaw=yaw, name="slew", **kw)
        hy = yaw + arm
        Rh = U.rotz(hy)
        fh, sh = Rh @ np.array([1.0, 0, 0]), Rh @ np.array([0, 1.0, 0])
        B.box(self.machine, c3 - fh * 0.4 + U.UP * 1.55, (1.4, 1.2, 0.5), yaw=hy, name="house", **kw)
        B.box(self.machine, c3 - fh * 1.55 + U.UP * 1.45, (0.35, 1.2, 0.6), yaw=hy, name="counterweight", **kw)
        cab = c3 + fh * 0.5 + sh * 0.6 + U.UP * 2.45
        B.box(self.machine, cab, (0.5, 0.45, 0.45), yaw=hy, name="cab", **kw)
        B.box(self.glass, cab + fh * 0.505, (0.01, 0.4, 0.38), yaw=hy, name="cab_glass", **kw)
        B.box(self.glass, cab - sh * 0.455, (0.4, 0.01, 0.35), yaw=hy, name="cab_glass", **kw)
        root = c3 + fh * 0.8 - sh * 0.2 + U.UP * 1.7
        knee = root + fh * 3.2 + U.UP * float(rng.uniform(1.2, 2.5))
        tip = knee + fh * float(rng.uniform(0.8, 1.8)) - U.UP * (knee[2] - 0.6)
        tip[2] = max(0.6, tip[2])
        B.member(self.machine, root, knee, 0.35, 0.5, kind="box", name="boom", **kw)
        B.member(self.machine, knee, tip, 0.25, 0.35, kind="box", name="stick", **kw)
        B.cyl(self.steel, root + U.UP * -0.2 + fh * 0.2, knee - fh * 0.8 - U.UP * 0.1, 0.07, seg=8, name="ram", **kw)
        B.hexa(self.steel, U.tapered(tip - U.UP * 0.3, hy, (0.35, 0.5), (0.5, 0.55), tip[2] - 0.6, tip[2] + 0.1),
               name="bucket", **kw)
        if not far:
            self.ctx.claim_disc(c, 2.6, kind="excavator")
            self.ctx.claim_disc(np.array(tip[:2]), 0.9, kind="bucket")

    def crane(self, xy, H, jib, yaw, roof=False, hook_xy=None, hook_z=None):
        """A tower crane: lattice mast, slewing unit, cab, apex, jib, counter-jib,
        trolley and hook (over ``hook_xy`` at ``hook_z`` when given). Parts are
        routed by the zone like everything else: the mast stands outside it,
        the jib passes above the cloud band, a hook cable that comes down into
        the zone collides."""
        B, rng = self.B, self.rng
        kw = {"roof": roof}
        if hook_xy is not None:
            dv = np.asarray(hook_xy, float) - np.asarray(xy, float)
            yaw = math.atan2(dv[1], dv[0])
            jib = max(jib, float(np.linalg.norm(dv)) + 4.0)
        col = self.yellow if rng.random() < 0.6 else (self.red if rng.random() < 0.5 else
                                                       self.ctx.looks.paint((0.85, 0.85, 0.82), rough=0.4,
                                                                            name="crane_white"))
        x, y = xy
        a = 0.8
        corners = [np.array([x + sx * a, y + sy * a]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        for p in corners:
            B.cyl(col, (*p, 0.0), (*p, H), 0.07, seg=6, name="mast", **kw)
        B.box(self.concrete2, (x, y, 0.5), (2.5, 2.5, 0.5), name="crane_base", **kw)
        for z in np.arange(0.0, H, 2.0):
            for i, (p, q) in enumerate(zip(corners, corners[1:] + corners[:1])):
                B.cyl(col, (*p, z), (*q, z), 0.035, seg=4, name="mast", **kw)
                B.cyl(col, (*p, z), (*q, z + 2.0) if i % 2 == 0 else (*q, z + 2.0), 0.03, seg=4, name="mast", **kw)
        R = U.rotz(yaw)
        f = (R @ np.array([1.0, 0, 0]))[:2]
        s = (R @ np.array([0, 1.0, 0]))[:2]
        c0 = np.array([x, y])
        B.box(self.steel, (x, y, H + 0.4), (1.1, 1.1, 0.4), yaw=yaw, name="slew", **kw)
        cab = c0 + f * 1.4 + s * 1.2
        B.box(col, (cab[0], cab[1], H + 1.2), (0.8, 0.6, 0.9), yaw=yaw, name="crane_cab", **kw)
        B.box(self.glass, (cab[0] + f[0] * 0.81, cab[1] + f[1] * 0.81, H + 1.3), (0.01, 0.55, 0.6), yaw=yaw,
              name="crane_cab", **kw)
        zj = H + 1.0
        # jib: triangular lattice
        h_j = 1.6
        for sgn in (-1, 1):
            B.cyl(col, (*(c0 + s * sgn * 0.7), zj), (*(c0 + f * jib + s * sgn * 0.4), zj), 0.06, seg=5, name="jib",
                  **kw)
        B.cyl(col, (*c0, zj + h_j), (*(c0 + f * jib), zj + 0.6), 0.06, seg=5, name="jib", **kw)
        n = int(jib / 2.0)
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n
            for sgn in (-1, 1):
                w0, w1 = 0.7 - 0.3 * t0, 0.7 - 0.3 * t1
                p = c0 + f * jib * t0 + s * sgn * w0
                q = c0 + f * jib * t1
                B.cyl(col, (*p, zj), (*q, zj + h_j - (h_j - 0.6) * t1), 0.025, seg=4, name="jib", **kw)
        # counter-jib with ballast
        cj = jib * float(rng.uniform(0.25, 0.35))
        B.box(col, (*(c0 - f * cj / 2), zj), (cj / 2, 0.8, 0.2), yaw=yaw, name="counterjib", **kw)
        for k in range(int(rng.integers(2, 5))):
            p = c0 - f * (cj - 0.8 - 1.0 * k)
            B.box(self.concrete2, (p[0], p[1], zj - 0.8), (0.45, 0.9, 0.9), yaw=yaw, name="ballast", **kw)
        # apex and ties
        B.cyl(col, (x, y, zj), (x, y, zj + 6.0), 0.18, 0.1, seg=8, name="apex", **kw)
        B.cyl(self.galv, (x, y, zj + 6.0), (*(c0 + f * jib * 0.6), zj + h_j - 0.4), 0.02, seg=4, name="tie", **kw)
        B.cyl(self.galv, (x, y, zj + 6.0), (*(c0 - f * cj), zj + 0.2), 0.02, seg=4, name="tie", **kw)
        # trolley + hook
        tr = np.asarray(hook_xy, float) if hook_xy is not None else c0 + f * jib * float(rng.uniform(0.3, 0.85))
        B.box(self.steel, (tr[0], tr[1], zj - 0.2), (0.5, 0.5, 0.2), yaw=yaw, name="trolley", **kw)
        hz = hook_z if hook_z is not None else float(rng.uniform(6, H - 5))
        B.cyl(self.galv, (tr[0], tr[1], zj - 0.4), (tr[0], tr[1], hz + 0.5), 0.015, seg=4, name="hook_cable", **kw)
        B.box(self.yellow, (tr[0], tr[1], hz + 0.3), (0.2, 0.12, 0.3), name="hook", **kw)
        return tr

    def apartment(self, x0, x1, yf, side, h, depth, far=True, roof=False):
        """A finished neighbouring block: facade material + a window grid."""
        B, rng, lk = self.B, self.rng, self.ctx.looks
        kw = {"far": far, "roof": roof}
        mat = lk.pbr(U.pick(self.ctx, FACADES, self.ctx.lib.materials(cats_any={"plaster"})), 2.5, bevel=0.02,
                     dirt=0.3)
        yc = yf + side * depth / 2
        B.box(mat, ((x0 + x1) / 2, yc, h / 2), ((x1 - x0) / 2, depth / 2, h / 2), name="apartment", **kw)
        B.box(self.concrete2, ((x0 + x1) / 2, yc, h + 0.3), ((x1 - x0) / 2 + 0.2, depth / 2 + 0.2, 0.3),
              name="apartment_roof", **kw)
        fh = float(rng.uniform(2.8, 3.2))
        ww = float(rng.uniform(1.2, 1.8))
        sp = float(rng.uniform(2.6, 3.6))
        for z in np.arange(1.0 + fh * 0.3, h - 1.0, fh):
            for x in np.arange(x0 + 1.5, x1 - 1.0, sp):
                B.box(self.glass, (x, yf - side * 0.02, z + 0.7), (ww / 2, 0.03, 0.7), name="window", **kw)
                B.box(self.concrete2, (x, yf - side * 0.06, z - 0.02), (ww / 2 + 0.08, 0.06, 0.04), name="sill", **kw)


# --- scene ---------------------------------------------------------------------------------------

def _sky(ctx, weather):
    lib, rng = ctx.lib, ctx.rng
    pool = {"dry": HDRI_DRY, "wet": HDRI_WET, "snow": HDRI_SNOW}[weather]
    ks = U.keys_in(lib, pool) or lib.hdris(cats_all={"outdoor"}, cats_none={"night", "indoor"},
                                           tags_any={"construction", "urban", "parking"})
    return ctx.looks.sky(lib.pick(rng, ks))


def _ground(ctx, weather):
    rng, lk = ctx.rng, ctx.looks
    fam = str(rng.choice(["mud", "mud", "gravel", "tracks"]))
    if weather == "snow":
        keys = [U.pick(ctx, SNOW), U.pick(ctx, MUD), U.pick(ctx, SNOW)]
    elif fam == "mud":
        keys = [U.pick(ctx, MUD), U.pick(ctx, MUD), U.pick(ctx, GRAVEL)]
    elif fam == "gravel":
        keys = [U.pick(ctx, GRAVEL), U.pick(ctx, GRAVEL), U.pick(ctx, MUD)]
    else:
        keys = [U.pick(ctx, TRACKS), U.pick(ctx, MUD), U.pick(ctx, GRAVEL)]
    wet = {"dry": float(rng.choice([0.0, 0.0, 0.25])), "wet": float(rng.uniform(0.6, 1.0)), "snow": 0.0}[weather]
    ctx.ground(lk.ground_material(keys, tile_m=float(rng.uniform(2.5, 4.0)), puddles=wet,
                                  blend_scale=float(rng.uniform(6.0, 12.0)) * (2.0 if weather == "snow" else 1.0)))
    return fam, keys, wet


def _dress(ctx, S, B):
    rng, lk = ctx.rng, ctx.looks
    wk = str(rng.choice(["block", "block", "insitu", "insitu", "hoarding", "brick"]))
    wall = {"block": S.block, "insitu": S.concrete, "hoarding": S.hoard_mat, "brick": S.brick}[wk]
    rs = [p["r"] for p in ctx.spec["primitives"] if p["kind"] == "cyl_v"]
    rh = [p["r"] for p in ctx.spec["primitives"] if p["kind"] == "cyl_h"]
    pole = S.galv if (rs and float(np.median(rs)) < 0.12) else S.column
    bar = S.galv if (not rh or float(np.median(rh)) < 0.1) else (S.steel if rng.random() < 0.5 else S.timber)
    box = S.concrete2 if rng.random() < 0.5 else (S.block if rng.random() < 0.5 else S.osb)
    rock = lk.pbr(U.pick(ctx, ROCK, ctx.lib.materials(cats_any={"rock"})), 1.5, bump=0.8, jitter=0.1)
    ctx.dress_primitives({"wall": wall, "box": box, "pole": pole, "bar": bar, "rock": rock})
    # in-situ walls: starter bars out of the top -- only on walls that reach the cloud top, so the bars
    # stand wholly above it and the cloud (and every route) is the patch's own
    if wk == "insitu":
        for p in ctx.spec["primitives"]:
            if p["kind"] != "wall" or p["z1"] < U.ZSPLIT - 0.1:
                continue
            p0, p1 = np.asarray(p["p0"], float)[:2], np.asarray(p["p1"], float)[:2]
            L = float(np.linalg.norm(p1 - p0))
            d = (p1 - p0) / max(L, 1e-9)
            nrm = np.array([-d[1], d[0]])
            slots = [(s - w / 2, s + w / 2, z0) for s, w, z0, _ in p.get("slots", [])]
            for t in np.arange(0.2, L - 0.1, 0.3):
                if any(a - 0.05 < t < b + 0.05 and z0 < p["z1"] - 0.01 for a, b, z0 in slots):
                    continue
                for o in (-p["t"] / 4, p["t"] / 4):
                    q = p0 + d * t + nrm * o
                    B.cyl(S.rebar, (*q, p["z1"]), (*q, p["z1"] + 0.7), 0.01, seg=5, name="rebar")
    return {"wall": wk}


def _layout_building(ctx, S, B, side, info):
    """The frame + scaffold on one long side (just outside the patch)."""
    rng = ctx.rng
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    roof = side < 0
    lvl = str(ctx.level)
    n = int(rng.integers(1, 4)) if lvl == "1" else int(rng.integers(2, 7))
    fh = float(rng.uniform(3.0, 3.4))
    bay = float(rng.uniform(5.0, 7.0))
    L = float(rng.uniform(22, 50))
    D = float(rng.uniform(12, 20))
    xa = float(rng.uniform(zx0 - 10, max(zx0 - 9.9, zx1 - L + 4)))
    gap = float(rng.uniform(2.0, 3.2))
    yf = (y1 + gap) if side > 0 else (y0 - gap)
    if not ctx.free_rect((xa + L / 2, yf - side * 0.8), (L / 2, 0.8), margin=0.2):
        # a patch opening's approach reaches out here: build further back
        yf += side * 2.5
    T = U.Xf(B, (xa, yf), 0.0 if side > 0 else math.pi)
    if side < 0:
        T = U.Xf(B, (xa + L, yf), math.pi)
    S.frame_building(T, L, D, n, fh, bay, roof=roof, top_frac=float(rng.uniform(0.3, 1.0)),
                     infill=float(rng.uniform(0.1, 0.6)))
    ctx.claim_rect((xa + L / 2, yf + side * D / 2), (L / 2 + 0.3, D / 2 + 0.3), kind="building")
    # scaffold along (part of) the face
    sa = xa + float(rng.uniform(0, L * 0.3))
    sb = min(xa + L, sa + float(rng.uniform(L * 0.5, L)))
    Hs = min(n * fh + 1.0, float(rng.choice([n * fh + 1.0, 6.0, 8.0])))
    S.scaffold(sa, sb, yf, side, Hs, net=rng.random() < 0.4, roof=roof)
    ctx.claim_rect(((sa + sb) / 2, yf - side * 0.8), ((sb - sa) / 2 + 0.2, 0.75), kind="scaffold")
    # the poured slab in front of / under the building (flat overlay)
    slab = U.soften(ctx.looks.pbr(U.pick(ctx, CONC_FLOOR), 3.0, proj="flat", vary=0.171, jitter=0.0), "blob",
                    width=0.08, noise_scale=0.6, noise_amp=0.2)
    U.decal(ctx, slab, (xa + L / 2, yf + side * D / 2), L / 2 + 3.0, D / 2 + 2.0, 0.0, z=0.004, name="slab")
    info["building"] = {"side": side, "floors": n, "L": round(L, 1), "scaffold_net": bool(rng.random() < 0.4)}
    return yf, xa, L


def _layout_other_side(ctx, S, B, side, info):
    """Low things on the other side: cabins / toilets, a fence line, storage."""
    rng = ctx.rng
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    roof = side < 0
    kind = str(rng.choice(["cabins", "fence", "cabins_fence", "storage"]))
    yb = (y0 - 0.8) if side < 0 else (y1 + 0.8)
    if kind in ("cabins", "cabins_fence"):
        n = int(rng.integers(2, 5))
        stack = rng.random() < 0.5
        yc = yb + side * (1.4 + (1.0 if stack else 0.0))
        for _ in range(20):
            xc = float(rng.uniform(zx0 + 3.1 * n, max(zx0 + 3.1 * n + 0.1, zx1 - 3.1 * n)))
            if ctx.free_rect((xc, yc), (3.1 * n, 1.3), margin=0.3):
                S.cabins((xc, yc), 0.0 if side > 0 else math.pi, n, stack=stack, roof=roof or stack)
                t_xy = (xc + 3.1 * n + 2.0, yc)
                if ctx.free_rect(t_xy, (1.3, 0.7), margin=0.2):
                    S.toilets(t_xy, 0.0 if side > 0 else math.pi, int(rng.integers(1, 4)))
                break
        veh = U.obj_keys(ctx, "pickup", "van", "truck")
        for _ in range(int(rng.integers(0, 3)) if veh else 0):     # site vehicles parked by the cabins
            k = veh[int(rng.integers(len(veh)))]
            lh, sh = U.long_half(ctx, k)
            xy = (float(rng.uniform(zx0 + lh, max(zx0 + lh + 0.1, zx1 - lh))), yb + side * (sh + float(rng.uniform(0.3, 3.0))))
            U.place_obj(ctx, k, xy, U.along_yaw(ctx, k, 0.0) + float(rng.normal(0, 0.05)), margin=0.4)
    if kind in ("fence", "cabins_fence"):
        yf = yb + side * (3.3 if kind == "cabins_fence" else 0.4)
        S.heras((zx0 - 1, yf), (zx1 + 1, yf))
    if kind == "storage":
        yc = yb + side * 1.8
        x = zx0 + 1.0
        while x < zx1 - 2:
            what = str(rng.choice(["bricks", "blocks", "timber", "pipes", "formwork", "heap"]))
            if what == "heap":
                R = float(rng.uniform(1.8, 3.5))
                c = (x + R, yb + side * (R + 0.3))
                if ctx.free(c, R, 0.3):
                    S.heap(c, R)
                x += 2 * R + 1.0
                continue
            if what in ("bricks", "blocks"):
                c = (x + 0.6, yc)
                if ctx.free_rect(c, (0.65, 0.48), margin=0.2):
                    S.unit_load(c, 0.0, what)
                x += 1.5
            elif what == "timber":
                c = (x + 2.5, yc)
                if ctx.free_rect(c, (2.6, 0.8), margin=0.2):
                    S.timber_stack(c, 0.0)
                x += 5.5
            elif what == "pipes":
                c = (x + 3.1, yc)
                if ctx.free_rect(c, (3.2, 0.9), margin=0.2):
                    S.pipe_stack(c, 0.0)
                x += 6.5
            else:
                c = (x + 1.4, yc)
                if ctx.free_rect(c, (1.45, 0.7), margin=0.2):
                    S.formwork_stack(c, 0.0)
                x += 3.2
    info["other_side"] = kind


def _anchor(ctx, half, yaw):
    rng = ctx.rng
    x0, y0, x1, y1 = ctx.bounds
    u = rng.random()
    if u < 0.5:
        side = -1 if rng.random() < 0.5 else 1
        off = float(rng.uniform(0.3, 2.0))
        y = (y0 + half[1] + off) if side < 0 else (y1 - half[1] - off)
        return np.array([float(rng.uniform(x0 + half[0], x1 - half[0])), y]), float(rng.choice([0.0, math.pi]))
    walls = [p for p in ctx.spec["primitives"] if p["kind"] == "wall"]
    if u < 0.75 and walls:
        p = walls[int(rng.integers(len(walls)))]
        a, b = np.asarray(p["p0"], float)[:2], np.asarray(p["p1"], float)[:2]
        L = float(np.linalg.norm(b - a))
        d = (b - a) / max(L, 1e-9)
        nrm = np.array([-d[1], d[0]]) * (1 if rng.random() < 0.5 else -1)
        t = float(rng.uniform(half[0], max(half[0] + 0.01, L - half[0])))
        return a + d * t + nrm * (p["t"] / 2 + half[1] + float(rng.uniform(0.3, 1.0))), math.atan2(d[1], d[0])
    return ctx.uniform_xy(pad=1.0), yaw


def _yard(ctx, S, B, info):
    """Materials, heaps, barriers, cones, fence sections, towers in the patch."""
    rng = ctx.rng
    x0, y0, x1, y1 = ctx.bounds
    ko = float(ctx.knobs.get("keepout", 2.5))
    n = ctx.count(12.0)
    sizes = {"bricks": (0.65, 0.48), "blocks": (0.65, 0.48), "cement": (0.65, 0.48), "rebar": (4.6, 0.9),
             "timber": (2.6, 0.9), "pipes": (3.2, 0.9), "formwork": (1.45, 0.7), "heap": (2.5, 2.5),
             "skip": (1.9, 0.95), "toilets": (1.3, 0.62), "barriers": (3.0, 0.3), "cones": (1.5, 1.5),
             "heras": (5.3, 0.3), "tower": (1.35, 0.8), "excavator": (3.2, 3.2), "ibc": (0.7, 0.6)}
    weights = {"bricks": 3, "blocks": 3, "cement": 2, "rebar": 2, "timber": 2, "pipes": 2, "formwork": 2, "heap": 2,
               "skip": 1, "toilets": 1, "barriers": 2, "cones": 2, "heras": 2, "tower": 1 if ctx.knobs.get(
                   "clutter", 1) >= 1 else 0, "excavator": 0.5, "ibc": 1}
    machines = U.obj_keys(ctx, "excavator", "loader", "dozer")
    skips = U.obj_keys(ctx, "skip", "dumpster")
    cones = U.obj_keys(ctx, "cone")
    kinds = list(weights)
    p = np.array([weights[k] for k in kinds], float)
    p /= p.sum()
    made = {k: 0 for k in kinds}
    for _ in range(n * 40):
        if sum(made.values()) >= n:
            break
        k = kinds[int(rng.choice(len(kinds), p=p))]
        if k == "excavator" and made[k] >= (2 if ctx.knobs.get("clutter", 1) >= 1 and machines else 1):
            continue
        half = sizes[k]
        yaw = float(rng.choice([0.0, math.pi / 2])) + float(rng.normal(0, 0.1))
        if k in ("heap", "excavator", "cones"):
            xy, yaw = ctx.uniform_xy(pad=half[0]), float(rng.uniform(0, 6.28))
        else:
            xy, yaw = _anchor(ctx, half, yaw)
        if not ctx.free_rect(xy, half, yaw, margin=0.5 * ko + 0.3):
            continue
        if k == "excavator" and not ctx.free(xy, 5.5, 0.3):
            continue
        if k in ("bricks", "blocks", "cement"):
            S.unit_load(xy, yaw, k)
            d = np.array([math.cos(yaw), math.sin(yaw)])
            for m in range(int(rng.integers(0, 4))):          # a row of pallets
                q = xy + d * 1.35 * (m + 1)
                if not ctx.free_rect(q, half, yaw, margin=0.5 * ko + 0.2):
                    break
                S.unit_load(q, yaw, k)
        elif k == "rebar":
            S.rebar_bundles(xy, yaw)
        elif k == "timber":
            S.timber_stack(xy, yaw)
        elif k == "pipes":
            S.pipe_stack(xy, yaw)
        elif k == "formwork":
            S.formwork_stack(xy, yaw)
        elif k == "heap":
            S.heap(xy, float(rng.uniform(1.5, 2.6)))
        elif k == "skip":
            ks = skips[int(rng.integers(len(skips)))] if skips and rng.random() < 0.5 else None
            if ks is None or U.place_obj(ctx, ks, xy, U.along_yaw(ctx, ks, yaw), margin=0.5 * ko + 0.3) is None:
                S.skip(xy, yaw)
        elif k == "toilets":
            S.toilets(xy, yaw, int(rng.integers(1, 3)))
        elif k == "barriers":
            d = np.array([math.cos(yaw), math.sin(yaw)])
            S.water_barriers(xy - d * 3.0, xy + d * 3.0)
        elif k == "cones":
            for _c in range(int(rng.integers(2, 6))):
                q = xy + rng.normal(0, 0.8, 2)
                if ctx.free(q, 0.2, 0.3):
                    if cones and rng.random() < 0.6:        # a scanned cone
                        ctx.instance(cones[int(rng.integers(len(cones)))], q, 0.0, float(rng.uniform(0, 6.28)), 1.0,
                                     coll="scatter", claim=True)
                    else:
                        S.cone(q)
        elif k == "heras":
            d = np.array([math.cos(yaw), math.sin(yaw)])
            S.heras(xy - d * 5.2, xy + d * 5.2)
        elif k == "tower":
            S.tower(xy, yaw, float(rng.choice([2.4, 3.4, 4.4])))
        elif k == "excavator":     # a scanned excavator / wheel loader / dozer, else the procedural one
            km = machines[int(rng.integers(len(machines)))] if machines and rng.random() < 0.65 else None
            if km is None or U.place_obj(ctx, km, xy, yaw, margin=0.5 * ko + 0.3) is None:
                S.excavator(xy, yaw, arm=float(rng.uniform(-0.8, 0.8)))
        elif k == "ibc":
            zt = S.pallet(xy, yaw)
            B.box(S.white, (xy[0], xy[1], zt + 0.48), (0.56, 0.46, 0.46), yaw=yaw, name="ibc")
            ctx.claim_rect(xy, half, yaw, kind="ibc")
        made[k] += 1
    info["yard"] = {k: v for k, v in made.items() if v}
    small = _pieces(ctx, ["ph:concrete_road_barrier#0", "ph:concrete_road_barrier_02#0", "ph:portable_generator#0",
                          "ph:Barrel_01#0", "ph:barrel_03#0", "ph:wooden_crate_02#0", "ph:old_tyre#0",
                          "ph:ladder_sectioned_01#0", "ph:hand_truck#0", "ph:plastic_crate_03#0",
                          "ph:rusted_spade_01#0", "ph:sledgehammer_01#0", "ph:trashbag#0", "ph:wooden_ladder_02#0",
                          "ph:metal_jerrycan#0", "ph:compost_bags#1", "ph:worn_metal_rack#0",
                          "ph:portable_welding_cart#0"])
    small += U.obj_keys(ctx, "wheelbarrow", "generator", "barrier", "ladder", "gas")
    ns = ctx.count(7.0)
    ctx.scatter(small, int(ns * 0.7), margin=0.5, where=lambda q: min(q[1] - y0, y1 - q[1]) < 2.5)
    ctx.scatter(small, ns - int(ns * 0.7), margin=0.6)


def _hang(ctx, S, B, info, crane_xy):
    """L3: scaffold tubes across the flight band, crane loads on the hook, cables."""
    rng = ctx.rng
    n = ctx.count(float(ctx.knobs.get("hang", 0.0)))
    boxes = U.prim_boxes(ctx.spec)
    made = {"gate": 0, "load": 0, "cable": 0}
    for _ in range(n * 40):
        if sum(made.values()) >= n:
            break
        kind = str(rng.choice(["gate", "gate", "load", "cable"]))
        p = ctx.uniform_xy(pad=0.8)
        if min(np.linalg.norm(p - ctx.start), np.linalg.norm(p - ctx.goal)) < 2.0:
            continue
        if kind == "load":
            yaw = float(rng.uniform(0, 3.14))
            what = str(rng.choice(["bricks", "formwork", "rebar"]))
            half = {"bricks": (0.65, 0.5), "formwork": (1.4, 0.65), "rebar": (2.55, 0.25)}[what]
            if not ctx.free_rect(p, half, yaw, margin=0.5):
                continue
            ext = np.abs(U.rotz(yaw)[:2, :2]) @ np.asarray(half)
            z = float(rng.uniform(2.4, 3.4))
            if not U.box_clear(boxes, (p[0] - ext[0], p[1] - ext[1], z - 0.2), (p[0] + ext[0], p[1] + ext[1], 5.0),
                               pad=0.3):
                continue
            # the crane that holds it: mast outside the zone (the closest of 20 bearings), trolley right above;
            # no load without one
            base = None
            for _t in range(20):
                th = float(rng.uniform(0, 2 * math.pi))
                d_out = float(max(ctx.zone[2] - p[0], p[0] - ctx.zone[0], ctx.zone[3] - p[1], p[1] - ctx.zone[1]))
                dist = float(rng.uniform(18, 32)) if _t < 10 else d_out + float(rng.uniform(4, 10))
                b = p + dist * np.array([math.cos(th), math.sin(th)])
                if not ctx.rect_meets_zone(b - 3.0, b + 3.0):
                    base = b
                    break
            if base is None:
                continue
            if what == "bricks":
                zt = S.pallet(p, yaw, z)
                B.box(S.brick, (p[0], p[1], zt + 0.4), (0.55, 0.4, 0.4), yaw=yaw, name="load")
                top = zt + 0.8
            elif what == "formwork":
                B.box(S.yellow, (p[0], p[1], z + 0.3), (1.35, 0.6, 0.3), yaw=yaw, name="load")
                top = z + 0.6
            else:
                R = U.rotz(yaw)
                f = R @ np.array([1.0, 0, 0])
                for i in range(10):
                    q = np.array([p[0], p[1], z + 0.1]) + R @ np.array([0, (i - 4.5) * 0.03, (i % 3) * 0.03])
                    B.cyl(S.rebar, q - f * 2.5, q + f * 2.5, 0.012, seg=5, name="load")
                top = z + 0.2
            hook = np.array([p[0], p[1], top + 1.0])
            R = U.rotz(yaw)
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                o = R @ np.array([sx * 0.5, sy * 0.35, 0])
                B.cyl(S.galv, (p[0] + o[0], p[1] + o[1], top), hook, 0.008, seg=4, name="sling")
            S.crane(base, float(rng.uniform(30, 45)), dist + 6.0, 0.0, roof=base[1] < ctx.bounds[1], hook_xy=p,
                    hook_z=float(hook[2]))
            ctx.claim_rect(p, half, yaw, kind="load")
            made["load"] += 1
        else:
            ang = float(rng.uniform(0, math.pi))
            L = float(rng.uniform(2.5, 5.0)) if kind == "gate" else float(rng.uniform(5.0, 10.0))
            q = p + L * np.array([math.cos(ang), math.sin(ang)])
            if not ctx.in_zone(q) or not (ctx.free(p, 0.15, 0.6) and ctx.free(q, 0.15, 0.6)):
                continue
            seg = q - p
            if min(U.seg_dist(ctx.start, p, q), U.seg_dist(ctx.goal, p, q)) < 2.2:
                continue
            if not U.seg_clear(ctx, p, q, margin=0.6):     # not across a slot's approach or the family's core
                continue
            if kind == "gate":
                zs = sorted(float(v) for v in rng.uniform(1.8, 3.3, int(rng.integers(1, 3))))
                if not U.box_clear(boxes, (*np.minimum(p, q), zs[0] - 0.2), (*np.maximum(p, q), 3.8), pad=0.3):
                    continue
                top = zs[-1] + float(rng.uniform(0.3, 1.2))
                for c in (p, q):
                    B.cyl(S.galv, (*c, 0.03), (*c, top), 0.024, seg=8, name="gate")
                    B.box(S.board, (c[0], c[1], 0.02), (0.15, 0.25, 0.02), name="soleboard")
                    ctx.claim_disc(c, 0.3, kind="gate")
                for z in zs:
                    B.cyl(S.galv, (*(p - seg / L * 0.2), z), (*(q + seg / L * 0.2), z), 0.024, seg=8, name="gate")
                if rng.random() < 0.5:
                    B.cyl(S.galv, (*p, 0.3), (*q, zs[0]), 0.022, seg=8, name="gate")
                made["gate"] += 1
            else:
                zc = float(rng.uniform(2.6, 3.4))
                if not U.box_clear(boxes, (*np.minimum(p, q), zc - 0.8), (*np.maximum(p, q), zc + 0.3), pad=0.3):
                    continue
                for c in (p, q):
                    B.cyl(S.timber, (*c, 0.0), (*c, zc + 0.4), 0.06, seg=8, name="post")
                    ctx.claim_disc(c, 0.3, kind="post")
                U.catenary(B, S.black, (*p, zc), (*q, zc), sag=float(rng.uniform(0.2, 0.6)), r=0.012)
                made["cable"] += 1
    info["hang"] = made


def _backdrop(ctx, S, B, bside, info):
    rng = ctx.rng
    zx0, zy0, zx1, zy1 = ctx.zone
    cranes = []
    # tower cranes (1-2), mostly behind the building
    for i in range(int(rng.integers(1, 3))):
        side = bside if i == 0 else -bside
        H = float(rng.uniform(28, 50))
        x = float(rng.uniform(zx0, zx1))
        y = (zy1 + float(rng.uniform(10, 25))) if side > 0 else (zy0 - float(rng.uniform(10, 25)))
        jib = float(rng.uniform(35, 55))
        yaw = float(rng.uniform(0, 2 * math.pi))
        cranes.append(S.crane((x, y), H, jib, yaw, roof=side < 0))
    # neighbouring blocks / more frames on both sides, beyond the zone
    for side in (-1, 1):
        roof = side < 0
        y_near = (zy0 - 9.0) if side < 0 else (zy1 + 9.0)
        x = zx0 - 45.0
        while x < zx1 + 45.0:
            L = float(rng.uniform(20, 45))
            kind = str(rng.choice(["apartment", "apartment", "frame", "gap"]))
            if side == bside and kind == "apartment" and rng.random() < 0.3:
                kind = "frame"
            if kind == "apartment":
                S.apartment(x, x + L, y_near + side * float(rng.uniform(4, 20)), side, float(rng.uniform(9, 30)),
                            float(rng.uniform(10, 16)), far=True, roof=roof)
            elif kind == "frame":
                yb = y_near + side * float(rng.uniform(4, 15))
                T = U.Xf(B, (x, yb), 0.0) if side > 0 else U.Xf(B, (x + L, yb), math.pi)
                S.frame_building(T, L, float(rng.uniform(12, 18)), int(rng.integers(3, 9)), 3.2, 6.0, roof=roof,
                                 top_frac=float(rng.uniform(0.3, 1.0)))
            x += L + float(rng.uniform(6, 20))
    # site boundary beyond the ends: hoarding or Heras, and soil heaps / an excavator
    for end in (-1, 1):
        xe = (zx0 - 3.0) if end < 0 else (zx1 + 3.0)
        if rng.random() < 0.5:
            S.hoarding((xe + end * 6.0, zy0 - 10), (xe + end * 6.0, zy1 + 10), far=True)
        for _ in range(int(rng.integers(1, 4))):
            R = float(rng.uniform(3, 7))
            c = (xe + end * float(rng.uniform(8, 25)), float(rng.uniform(zy0, zy1)))
            B.mound(ctx.looks.pbr(U.pick(ctx, MUD + GRAVEL), 2.0, vary=0.2, jitter=0.0, bump=0.6), c, R,
                    R * float(rng.uniform(0.3, 0.5)), far=True, name="heap_far")
        if rng.random() < 0.6:
            xy = (xe + end * float(rng.uniform(12, 20)), float(rng.uniform(zy0, zy1)))
            machines = U.obj_keys(ctx, "excavator", "loader", "dozer", "truck")
            if machines and rng.random() < 0.6:
                U.place_obj(ctx, machines[int(rng.integers(len(machines)))], xy, float(rng.uniform(0, 6.28)),
                            check=False)
            else:
                S.excavator(xy, float(rng.uniform(0, 6.28)), arm=float(rng.uniform(-1, 1)), far=True)
        # a building across the end of the view
        if rng.random() < 0.75:
            xb = xe + end * float(rng.uniform(14, 30))
            ya, yz = zy0 - float(rng.uniform(0, 15)), zy1 + float(rng.uniform(0, 15))
            T = U.Xf(B, (xb, yz if end > 0 else ya), -math.pi / 2 if end > 0 else math.pi / 2)
            if rng.random() < 0.5:
                S.frame_building(T, yz - ya, float(rng.uniform(12, 18)), int(rng.integers(2, 8)), 3.2, 6.0,
                                 top_frac=float(rng.uniform(0.3, 1.0)))
            else:
                _apartment_local(S, T, yz - ya, float(rng.uniform(9, 30)), float(rng.uniform(10, 16)))
    info["cranes"] = len(cranes)
    return cranes


def _apartment_local(S, T, L, h, depth):
    """``Site.apartment`` in a local frame (facade on y = 0 along x, block towards +y)."""
    rng, lk = S.rng, S.ctx.looks
    mat = lk.pbr(U.pick(S.ctx, FACADES, S.ctx.lib.materials(cats_any={"plaster"})), 2.5, bevel=0.02, dirt=0.3)
    T.box(mat, (L / 2, depth / 2, h / 2), (L / 2, depth / 2, h / 2), name="apartment")
    T.box(S.concrete2, (L / 2, depth / 2, h + 0.3), (L / 2 + 0.2, depth / 2 + 0.2, 0.3), name="apartment_roof")
    fh = float(rng.uniform(2.8, 3.2))
    ww = float(rng.uniform(1.2, 1.8))
    sp = float(rng.uniform(2.6, 3.6))
    for z in np.arange(1.0 + fh * 0.3, h - 1.0, fh):
        for x in np.arange(1.5, L - 1.0, sp):
            T.box(S.glass, (x, -0.02, z + 0.7), (ww / 2, 0.03, 0.7), name="window")
            T.box(S.concrete2, (x, -0.06, z - 0.02), (ww / 2 + 0.08, 0.06, 0.04), name="sill")


def _ground_detail(ctx, S, weather, bside):
    rng, lk = ctx.rng, ctx.looks
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    # a gravel haul road along the patch, soft-edged, with tyre-track mud on it
    if weather != "snow" or rng.random() < 0.4:
        gk = U.pick(ctx, GRAVEL)
        road = U.soften(lk.pbr(gk, 2.5, proj="flat", vary=0.172, jitter=0.0), "strip", width=0.35, noise_scale=0.5,
                        noise_amp=0.6)
        y = float(rng.uniform(y0 + 2, y1 - 2))
        U.overlay_strip(ctx, road, (zx0 - 20, y), (zx1 + 20, y), float(rng.uniform(4, 7)), z=0.003, name="haul_road")
        if rng.random() < 0.7:
            tk = U.pick(ctx, TRACKS)
            tr = U.soften(lk.pbr(tk, 3.0, proj="flat", vary=0.173, jitter=0.0), "strip", width=0.4, noise_scale=0.7,
                          noise_amp=0.6)
            U.overlay_strip(ctx, tr, (zx0 - 20, y + rng.normal(0, 0.5)), (zx1 + 20, y + rng.normal(0, 0.5)), 3.0,
                            z=0.005, name="tracks")
    # mud / sand / water blotches
    for _ in range(int(rng.integers(4, 12))):
        k = U.pick(ctx, MUD + SAND if weather != "snow" else SNOW)
        m = U.soften(lk.pbr(k, 2.0, proj="flat", vary=0.174 + 0.0001 * int(rng.integers(100)), jitter=0.0), "blob",
                     width=0.5, noise_scale=0.9, noise_amp=0.7)
        c = ctx.uniform_xy(ctx.zone, pad=1.0)
        r = float(rng.uniform(1.0, 4.0))
        U.decal(ctx, m, c, r, r * float(rng.uniform(0.5, 1.0)), float(rng.uniform(0, 3.14)), z=0.006, name="blotch")
    dark = U.decal_material("wet_patch", rgb=(0.06, 0.055, 0.05), rough=0.05, opacity=0.4, scale=1.0)
    for _ in range(int(rng.integers(3, 9)) if weather == "wet" else int(rng.integers(0, 4))):
        c = ctx.uniform_xy(ctx.zone, pad=1.0)
        r = float(rng.uniform(0.5, 2.0))
        U.decal(ctx, dark, c, r, r * 0.6, float(rng.uniform(0, 3.14)), z=0.007, name="puddle")


def build(ctx) -> dict:
    rng = ctx.rng
    weather = str(rng.choice(["dry", "dry", "dry", "wet", "wet", "snow"]))
    sky = _sky(ctx, weather)
    fam, gkeys, wet = _ground(ctx, weather)
    B = U.Batch(ctx)
    S = Site(ctx, B, weather)
    info = {"sky": sky, "weather": weather, "ground": gkeys, "wet": wet}
    info["dress"] = _dress(ctx, S, B)
    info["core_reserved"] = U.reserve_core(ctx)
    info.update(SEMANTICS)
    bside = 1 if rng.random() < 0.7 else -1
    info["walled"] = U.walled(ctx)
    if not info["walled"]:          # an arena's perimeter walls: nothing to build between them and the zone edge
        _layout_building(ctx, S, B, bside, info)
        _layout_other_side(ctx, S, B, -bside, info)
    _yard(ctx, S, B, info)
    cranes = _backdrop(ctx, S, B, bside, info)
    if float(ctx.knobs.get("hang", 0.0)) > 0:
        _hang(ctx, S, B, info, cranes[0] if cranes else None)
    _ground_detail(ctx, S, weather, bside)
    B.flush()
    info["objaverse"] = U.obj_record(ctx)
    return info
