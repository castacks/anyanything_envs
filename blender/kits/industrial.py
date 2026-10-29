"""Process plant / factory yard: the patch runs down a plant road between
pipe racks, storage tanks, corrugated sheds and open steel process frames.

What makes it read as a plant rather than boxes on a plane:

* **pipe racks** -- steel I-section bents every 5-6.5 m carrying one or two
  tiers of pipes of mixed bore (painted to the service colours a plant uses:
  gas yellow, water green, fire red, grey, insulated aluminium jacketing),
  with flanges, cable trays and pipe drops; **pipe bridges** span the road
  on trusses, and at L2/L3 they come down into the upper flight band;
* **storage tanks** in concrete bunds with helical stairs and rim rails,
  bullet vessels on saddles, **process frames** (three-level open steel
  structures with grating decks, vessels and stairs), distillation columns
  and chimney stacks on the skyline;
* **sheds**: portal-frame buildings in corrugated cladding with roller
  doors, plinths, window bands, gutters, wall ducts and downpipes;
* the yard: stained concrete / asphalt / gravel with oil stains, worn paint
  lines, concrete joints, drains and manholes, and equipment on it: pump
  skids, electrical cabinets, gas-bottle cages, IBC totes, drums on pallets,
  cable reels, pipe spools, bollards.

Each scene draws a layout style (pipe alley, tank farm, factory yard,
process unit), a steel paint colour, pipe colours, cladding colour, plant
age (clean paint .. rust), ground family and weather (dry / wet / snow).
Levels: ``clutter`` scales the yard equipment; ``hang`` > 0 adds low pipe
runs on T-supports across the patch and pipe drops / hoses hanging from
the pipe bridges into the flight band.
"""

from __future__ import annotations

import math

import numpy as np

from kits import _industrial_util as U

NAME = "industrial"
INDOOR = False
DESCRIPTION = "process plant / factory yard: pipe racks and bridges, tanks, sheds, process frames, yard equipment"

HDRI_DRY = ["ph:abandoned_hopper_terminal_03", "ph:abandoned_hopper_terminal_04", "ph:abandoned_parking",
            "ph:abandoned_tank_farm_01", "ph:abandoned_tank_farm_02", "ph:abandoned_tank_farm_03",
            "ph:derelict_airfield_02", "ph:factory_yard", "ph:freight_station", "ph:german_town_street",
            "ph:overcast_industrial_courtyard", "ph:poolbeg", "ph:pump_station", "ph:reinforced_concrete_02",
            "ph:teufelsberg_ground_1", "ph:teufelsberg_ground_2", "ph:ticknock_01", "ph:brick_factory_01",
            "ph:portland_landing_pad", "ph:mall_parking_lot", "ph:driving_school", "ph:rooftop_day",
            "ph:construction_yard", "ph:future_parking", "ph:sunset_jhbcentral", "ph:derelict_airfield_01"]
HDRI_WET = ["ph:abandoned_slipway", "ph:dam_bridge", "ph:hanger_exterior_cloudy", "ph:overcast_industrial_courtyard",
            "ph:teufelsberg_roof", "ph:beach_parking", "ph:abandoned_waterworks", "ph:stadium_exterior",
            "ph:petit_port", "ph:alte_veste_station"]
HDRI_SNOW = ["ph:abandoned_hopper_terminal_01", "ph:abandoned_hopper_terminal_02", "ph:snowy_field",
             "ph:rural_winter_roadside", "ph:snowy_park_01", "ph:snow_field", "ph:snowy_cemetery"]

STEEL = {"yellow": (0.80, 0.58, 0.07), "grey": (0.42, 0.44, 0.45), "green": (0.16, 0.30, 0.20),
         "blue": (0.10, 0.22, 0.40), "oxide": (0.40, 0.12, 0.07), "orange": (0.78, 0.33, 0.06),
         "white": (0.78, 0.78, 0.75), "dark": (0.12, 0.13, 0.14)}
PIPES = [(0.80, 0.62, 0.08), (0.15, 0.38, 0.20), (0.62, 0.08, 0.06), (0.50, 0.50, 0.50), (0.18, 0.36, 0.60),
         (0.35, 0.20, 0.10), (0.75, 0.75, 0.72), (0.08, 0.08, 0.08)]
CLAD_RGB = [(0.55, 0.57, 0.58), (0.72, 0.72, 0.70), (0.30, 0.40, 0.50), (0.25, 0.35, 0.28), (0.55, 0.20, 0.14),
            (0.78, 0.74, 0.62), (0.35, 0.36, 0.38)]

CONC_GROUND = ["acg:Concrete041A", "acg:Concrete042A", "acg:Concrete043A", "acg:Concrete044A", "acg:Concrete044B",
               "acg:Concrete031", "acg:Concrete032", "ph:damaged_concrete_floor", "ph:hangar_concrete_floor",
               "ph:garage_floor", "ph:concrete_floor_worn_001", "ph:dirty_concrete", "ph:gravel_concrete",
               "ph:concrete_floor_02", "acg:Concrete020", "acg:Concrete006", "acg:Concrete002", "ph:cracked_concrete",
               "ph:concrete_floor_damaged_01", "ph:grooved_concrete_driveway"]
ASPH_GROUND = ["acg:Asphalt024A", "acg:Asphalt025A", "acg:Asphalt026A", "acg:Asphalt027A", "acg:Asphalt028A",
               "acg:Asphalt012", "acg:Asphalt013", "acg:Asphalt021", "ph:asphalt_02", "ph:tarred_gravel",
               "ph:asphalt_01", "acg:Asphalt031"]
GRAV_GROUND = ["acg:Gravel006", "acg:Gravel017", "acg:Gravel020", "acg:Gravel024", "acg:Gravel040", "acg:Gravel043",
               "ph:gravel_stones", "ph:tarred_gravel", "acg:Ground002", "acg:Ground018", "acg:Ground110"]
SNOW_GROUND = ["acg:Snow011", "acg:Snow012", "acg:Snow013", "ph:snow_03", "ph:snow_floor", "ph:asphalt_snow",
               "acg:Snow007C", "acg:Snow009C"]
CLAD = ["acg:CorrugatedSteel005", "acg:CorrugatedSteel009", "ph:corrugated_iron", "acg:CorrugatedSteel003"]
CLAD_OLD = ["ph:worn_corrugated_iron", "ph:rusty_corrugated_iron", "acg:CorrugatedSteel007B",
            "acg:CorrugatedSteel006B", "ph:corrugated_iron_02", "ph:factory_wall"]
ROOF = ["ph:box_profile_metal_sheet", "ph:corrugated_iron", "acg:CorrugatedSteel005", "ph:corrugated_iron_03",
        "ph:worn_corrugated_iron"]
PLAIN_CONC = ["acg:Concrete034", "ph:rough_concrete", "acg:Concrete002", "acg:Concrete006", "acg:Concrete020",
              "acg:Concrete015", "acg:Concrete023", "ph:brushed_concrete"]
BLOCK = ["ph:concrete_block_wall", "ph:concrete_block_wall_02"]
PRECAST = ["ph:concrete_slab_wall", "ph:concrete_slab_wall_02", "ph:concrete_panels", "ph:concrete_wall_006",
           "ph:concrete_wall_004", "ph:concrete_layers", "acg:Concrete034"]
BRICK_IND = ["ph:brick_wall_09", "ph:red_brick_03", "ph:brick_wall_02", "ph:brick_4", "acg:Bricks076A",
             "acg:Bricks090", "acg:Bricks094", "ph:dark_brick_wall"]
SHUTTER = ["ph:painted_metal_shutter", "ph:rusty_metal_shutter", "ph:worn_shutter", "ph:rusted_shutter"]
GRATING = ["acg:MetalWalkway001", "acg:MetalWalkway004", "acg:MetalWalkway006", "acg:MetalWalkway009",
           "acg:MetalWalkway010"]
WOOD = ["ph:wood_planks", "ph:weathered_planks", "acg:Planks012", "acg:WoodFloor041", "ph:raw_plank_wall",
        "acg:Wood049", "ph:plywood"]
PROPS = ["ph:Barrel_01#0", "ph:Barrel_02#0", "ph:barrel_03#0", "ph:metal_trash_can#0", "ph:metal_trash_can#2",
         "ph:utility_box_01#0", "ph:utility_box_02#0", "ph:portable_generator#0", "ph:portable_welding_cart#0",
         "ph:small_lpg_tank#0", "ph:propane_tank#0", "ph:korean_fire_extinguisher_01#0", "ph:hand_truck#0",
         "ph:tool_cart#0", "ph:industrial_storage_cart#0", "ph:worn_metal_rack#0", "ph:wooden_crate_02#0",
         "ph:plastic_crate_03#0", "ph:old_tyre#0", "ph:old_military_compressor#0", "ph:wooden_military_crate#0",
         "ph:fire_hydrant#0", "ph:concrete_road_barrier#0", "ph:concrete_road_barrier_02#0"]
BARRELS = ["ph:Barrel_01#0", "ph:Barrel_02#0", "ph:barrel_03#0"]
#: what the kit adds on the patch primitives (course.json ``chosen``): nothing that changes a route
SEMANTICS = {"semantics_changed": False,
             "semantics_notes": ["wall trim <= 0.12 m proud (clad posts, precast joints, signs)",
                                 "copings only on walls >= 3.9 m (no route over them)",
                                 "3 cm base plates round steel poles (below the 0.3 m floor band)",
                                 "pipe decks / vessel heads only above the cloud top on poles >= 4.8 m",
                                 "family core, wall ends and slot approaches kept clear of dressing"]}


def _pieces(ctx, keys):
    out = []
    for k in keys:
        fk, i = k.split("#")
        if fk in ctx.lib.items and any(p["index"] == int(i) for p in ctx.lib.pieces(fk)):
            out.append(k)
    return out


class Plant:
    """The scene's palette and the builders that use it."""

    def __init__(self, ctx, B, weather, age):
        self.ctx, self.B, self.rng = ctx, B, ctx.rng
        rng, lk = ctx.rng, ctx.looks
        self.weather, self.age = weather, age
        self.steel_name = str(rng.choice(list(STEEL)))
        self.steel = U.paint(ctx, STEEL[self.steel_name], rusty=age)
        self.steel2 = U.paint(ctx, STEEL[str(rng.choice(["grey", "dark", "white", "yellow", "green"]))], rusty=age)
        self.galv = U.galv(ctx)
        self.rail = U.paint(ctx, STEEL["yellow"] if rng.random() < 0.6 else STEEL[self.steel_name], rusty=age,
                            bevel=0.0)
        idx = rng.permutation(len(PIPES))[:int(rng.integers(3, 6))]
        self.pipes = [U.paint(ctx, PIPES[i], rusty=age * 0.8, tile=1.0, bevel=0.0) for i in idx]
        self.jacket = U.galv(ctx, tile=1.0, bevel=0.0)
        self.pipes.append(self.jacket)
        clad_rgb = CLAD_RGB[int(rng.integers(len(CLAD_RGB)))]
        ck = U.pick(ctx, CLAD_OLD if rng.random() < age else CLAD)
        self.clad = lk.pbr(ck, 1.4, tint=clad_rgb if "corrugated_iron" not in ck and "factory" not in ck else None,
                           bevel=0.02, dirt=0.5)
        clad2_rgb = CLAD_RGB[int(rng.integers(len(CLAD_RGB)))]
        self.clad2 = lk.pbr(U.pick(ctx, CLAD), 1.4, tint=clad2_rgb, bevel=0.02, dirt=0.5)
        self.roof = lk.pbr(U.pick(ctx, ROOF), 2.0, dirt=0.4)
        if weather == "snow":
            self.roof = lk.pbr(U.pick(ctx, ["acg:Snow005", "acg:Snow002", "ph:snow_02"]), 3.0)
        self.shutter = lk.pbr(U.pick(ctx, SHUTTER), 1.6, bevel=0.01, dirt=0.4)
        self.concrete = U.scan_native(ctx, PRECAST, 2.0, bevel=0.02, dirt=0.5)
        self.block = U.scan_native(ctx, BLOCK, 2.0, bevel=0.015, dirt=0.4)
        self.column = ctx.looks.pbr(U.pick(ctx, PLAIN_CONC), 2.0, bevel=0.02, dirt=0.4)
        self.grating = lk.pbr(U.pick(ctx, GRATING), 1.0, bevel=0.0)
        self.glass = lk.glass(tint=(0.55, 0.6, 0.62), rough=0.15)
        self.tank_shell = U.paint(ctx, (0.8, 0.8, 0.78) if rng.random() < 0.6 else STEEL["grey"], rusty=age,
                                  tile=2.5, bevel=0.0, plain=True, dirt=0.25)
        self.tank_shell2 = U.galv(ctx, tile=2.0, bevel=0.0) if rng.random() < 0.5 else \
            U.paint(ctx, STEEL[str(rng.choice(["green", "grey", "white", "dark"]))], rusty=age, tile=2.5, bevel=0.0,
                    plain=True, dirt=0.25)
        self.wood = lk.pbr(U.pick(ctx, WOOD, ctx.lib.materials(cats_any={"planks"})), 1.5, bevel=0.005, dirt=0.3)
        self.lamp = lk.paint((1.0, 0.95, 0.85), rough=0.3, name="lamp", emission=(1.0, 0.9, 0.75, 2.0))
        self.black = lk.paint((0.03, 0.03, 0.03), rough=0.5, name="rubber")
        self.plastic_w = lk.paint((0.82, 0.82, 0.78), rough=0.45, name="ibc_white")

    # --- pipe racks ---------------------------------------------------------------------------

    def rack(self, a, b, width=3.0, tiers=(4.2,), spacing=6.0, n_pipes=(4, 7), roof=False, check=True, drops=True,
             split=True):
        """A pipe rack along a -> b (xy): bents every ``spacing``, pipes per tier.
        With ``check`` it is cut into the stretches that pass no patch
        primitive (a wall running out past the bounds stops the rack; the
        pipes never go through it)."""
        ctx, B, rng = self.ctx, self.B, self.rng
        a, b = np.asarray(a, float), np.asarray(b, float)
        top = tiers[-1] + 0.4
        if check and split:
            return sum(self.rack(p, q, width, tiers, spacing, n_pipes, roof, check, drops, split=False)
                       for p, q in U.free_runs(ctx, a, b, width / 2 + 0.4, 0.0, top + 1.0))
        L = float(np.linalg.norm(b - a))
        if L < 3:
            return 0
        d = (b - a) / L
        nrm = np.array([-d[1], d[0]])
        n = max(1, int(round(L / spacing)))
        ts = [L * i / n for i in range(n + 1)]
        bents = []
        for t in ts:
            p = a + d * t
            cols = [p + nrm * width / 2, p - nrm * width / 2]
            if check and not all(ctx.free(c, 0.25, 0.3) for c in cols if ctx.in_zone(c)):
                continue
            bents.append(t)
        if len(bents) < 2:
            return 0
        kw = {"roof": roof}
        for t in bents:
            p = a + d * t
            for s in (-1, 1):
                c = p + nrm * s * width / 2
                B.member(self.steel, (*c, 0.0), (*c, top), 0.25, 0.25, roll=(*d, 0.0), name="rack", **kw)
                B.box(self.concrete, (c[0], c[1], 0.1), (0.35, 0.35, 0.1), yaw=math.atan2(d[1], d[0]), name="plinth",
                      **kw)
                if ctx.in_zone(c):
                    ctx.claim_disc(c, 0.4, kind="rack")
            for z in tiers:
                B.member(self.steel, (*(p - nrm * width / 2), z), (*(p + nrm * width / 2), z), 0.2, 0.25,
                         name="rack", **kw)
                for s in (-1, 1):   # knee braces
                    c = p + nrm * s * width / 2
                    B.member(self.steel, (*c, z - 0.9), (*(c - nrm * s * 0.7), z - 0.12), 0.08, 0.08, kind="L",
                             name="rack", **kw)
        t0, t1 = bents[0], bents[-1]
        for z in tiers:        # longitudinal struts
            for s in (-1, 1):
                q0 = a + d * t0 + nrm * s * width / 2
                q1 = a + d * t1 + nrm * s * width / 2
                B.member(self.steel, (*q0, z - 0.05), (*q1, z - 0.05), 0.12, 0.15, name="rack", **kw)
        for ti, z in enumerate(tiers):   # pipes
            k = int(rng.integers(*n_pipes))
            u = -width / 2 + 0.25
            for j in range(k):
                r = float(rng.choice([0.05, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2, 0.25, 0.3]))
                if u + 2 * r > width / 2 - 0.1:
                    break
                mat = self.pipes[int(rng.integers(len(self.pipes)))]
                if r > 0.15 and rng.random() < 0.35:
                    mat, r = self.jacket, r + 0.05
                off = u + r
                u += 2 * r + float(rng.uniform(0.08, 0.3))
                zc = z + 0.125 + r
                pa = a + d * (t0 - 0.4) + nrm * off
                pb = a + d * (t1 + 0.4) + nrm * off
                seg = 16 if r > 0.12 else 10
                B.cyl(mat, (*pa, zc), (*pb, zc), r, seg=seg, name="pipe", **kw)
                for t in bents[::2]:     # flanges
                    q = a + d * (t + 0.6) + nrm * off
                    B.cyl(mat, (*(q - d * 0.04), zc), (*(q + d * 0.04), zc), r * 1.3 + 0.01, seg=seg, name="pipe",
                          **kw)
                if drops and ti == 0 and rng.random() < 0.25:     # drop to grade at one end
                    end = pa if rng.random() < 0.5 else pb
                    if not ctx.in_zone(end) or ctx.free(end, r, 0.3):
                        B.cyl(mat, (*end, zc), (*end, 0.25), r, seg=seg, name="pipe", **kw)
                        B.box(self.concrete, (end[0], end[1], 0.12), (r + 0.2, r + 0.2, 0.12), name="plinth", **kw)
                        if ctx.in_zone(end):
                            ctx.claim_disc(end, r + 0.3, kind="drop")
        # cable tray on the top tier
        z = tiers[-1] + 0.35
        q0 = a + d * t0 + nrm * (width / 2 - 0.35)
        q1 = a + d * t1 + nrm * (width / 2 - 0.35)
        c = (q0 + q1) / 2
        B.box(self.galv, (c[0], c[1], z), (np.linalg.norm(q1 - q0) / 2, 0.3, 0.05), yaw=math.atan2(d[1], d[0]),
              name="tray", **kw)
        return len(bents)

    def bridge(self, xc, ya, yb, zb, width=2.4, depth=1.6, hang=0):
        """A truss pipe bridge across the road at x = ``xc`` from ya to yb, bottom chord at ``zb``."""
        ctx, B, rng = self.ctx, self.B, self.rng
        zt = zb + depth
        for y in (ya, yb):
            for s in (-1, 1):
                c = (xc + s * width / 2, y)
                B.member(self.steel, (*c, 0.0), (*c, zt + 0.2), 0.3, 0.3, roll=(0, 1, 0), name="bridge")
                B.box(self.concrete, (c[0], c[1], 0.1), (0.4, 0.4, 0.1), name="plinth")
                ctx.claim_disc(c, 0.45, kind="bridge")
            B.member(self.steel, (xc - width / 2, y, zt), (xc + width / 2, y, zt), 0.25, 0.3, name="bridge")
            B.member(self.steel, (xc - width / 2, y, zb - 0.15), (xc + width / 2, y, zb - 0.15), 0.25, 0.3,
                     name="bridge")
        L = yb - ya
        n = max(2, int(round(L / 2.5)))
        for s in (-1, 1):       # the two trusses
            x = xc + s * width / 2
            B.member(self.steel, (x, ya, zt), (x, yb, zt), 0.2, 0.25, name="bridge")
            B.member(self.steel, (x, ya, zb), (x, yb, zb), 0.2, 0.25, name="bridge")
            for i in range(n + 1):
                y = ya + L * i / n
                B.member(self.steel, (x, y, zb), (x, y, zt), 0.12, 0.12, kind="box", name="bridge")
                if i < n:
                    y2 = ya + L * (i + 1) / n
                    a_, b_ = ((x, y, zb), (x, y2, zt)) if i < n / 2 else ((x, y, zt), (x, y2, zb))
                    B.member(self.steel, a_, b_, 0.09, 0.09, kind="L", name="bridge")
        for i in range(n + 1):     # cross beams under the pipes
            y = ya + L * i / n
            B.member(self.steel, (xc - width / 2, y, zb), (xc + width / 2, y, zb), 0.15, 0.2, name="bridge")
        u = -width / 2 + 0.25
        pipes = []
        for j in range(int(rng.integers(3, 7))):
            r = float(rng.choice([0.06, 0.08, 0.1, 0.15, 0.2, 0.25]))
            if u + 2 * r > width / 2 - 0.1:
                break
            mat = self.pipes[int(rng.integers(len(self.pipes)))]
            x = xc + u + r
            u += 2 * r + float(rng.uniform(0.08, 0.25))
            zc = zb + 0.1 + r
            B.cyl(mat, (x, ya - 1.0, zc), (x, yb + 1.0, zc), r, seg=14, name="pipe")
            pipes.append((x, zc, r, mat))
        # hanging drops / hoses into the flight band
        y0, y1 = self.ctx.bounds[1], self.ctx.bounds[3]
        for _ in range(hang):
            if not pipes:
                break
            x, zc, r, mat = pipes[int(rng.integers(len(pipes)))]
            y = float(rng.uniform(y0 + 1.0, y1 - 1.0))
            if min(np.hypot(x - ctx.start[0], y - ctx.start[1]), np.hypot(x - ctx.goal[0], y - ctx.goal[1])) < 2.5:
                continue
            if U.clear_of(ctx, (x, y)) < 0.8 or U.clear_of(ctx, (x, y + 1.6)) < 0.8:
                continue        # not into a slot's approach or the family's core
            z_end = float(rng.uniform(2.0, 2.9))
            if rng.random() < 0.5:        # a pipe drop with a valve
                rr = min(0.08, r * 0.6)
                B.cyl(mat, (x, y, zc), (x, y, z_end), rr, seg=10, name="drop")
                B.cyl(mat, (x, y, z_end + 0.25), (x, y, z_end + 0.33), rr * 1.5 + 0.02, seg=10, name="drop")
                B.cyl(self.rail, (x - 0.2, y, z_end + 0.5), (x + 0.2, y, z_end + 0.5), 0.015, seg=6, name="drop")
            else:                         # a hose hanging in a loop
                h = self.black
                p0 = np.array([x, y, zb])
                p1 = np.array([x + float(rng.normal(0, 0.3)), y + float(rng.uniform(0.8, 1.6)), zb])
                pts = [p0 + (p1 - p0) * t - U.UP * (zb - z_end) * math.sin(math.pi * t) for t in np.linspace(0, 1, 9)]
                for q0, q1 in zip(pts[:-1], pts[1:]):
                    B.cyl(h, q0, q1, 0.035, seg=8, caps=(False, False), name="hose")

    # --- process frame --------------------------------------------------------------------------

    def process_frame(self, T, nx=2, ny=1, bay=6.0, levels=(5.0, 9.5, 14.0), roof=False, vessels=True):
        """An open steel frame (local frame ``T``: x along, y away from the road)."""
        rng = self.rng
        kw = {"roof": roof}
        X, Y = nx * bay, ny * bay
        top = levels[-1] + 1.2
        for i in range(nx + 1):
            for j in range(ny + 1):
                c = (i * bay, j * bay)
                T.member(self.steel, (*c, 0.0), (*c, top), 0.3, 0.3, name="frame", **kw)
                T.box(self.concrete, (c[0], c[1], 0.12), (0.45, 0.45, 0.12), name="plinth", **kw)
        for z in levels:
            for i in range(nx + 1):
                T.member(self.steel, (i * bay, 0, z), (i * bay, Y, z), 0.25, 0.35, name="frame", **kw)
            for j in range(ny + 1):
                T.member(self.steel, (0, j * bay, z), (X, j * bay, z), 0.25, 0.35, name="frame", **kw)
            T.box(self.grating, (X / 2, Y / 2, z + 0.2), (X / 2 + 0.3, Y / 2 + 0.3, 0.025), name="deck", **kw)
            pts = [(-0.3, -0.3), (X + 0.3, -0.3), (X + 0.3, Y + 0.3), (-0.3, Y + 0.3)]
            for p, q in zip(pts, pts[1:] + pts[:1]):
                for zz in (z + 1.25, z + 0.75):
                    T.cyl(self.rail, (*p, zz), (*q, zz), 0.024, seg=6, name="rail", **kw)
            for i in range(nx + 1):       # bracing on the far face
                T.member(self.steel, (i * bay, Y, z - (levels[0] if z == levels[0] else 4.5) + 0.3),
                         ((i + 0.5) * bay if i < nx else (i - 0.5) * bay, Y, z - 0.2), 0.1, 0.1, kind="L",
                         name="frame", **kw)
        if vessels:
            for _ in range(int(rng.integers(1, 4))):
                i, j = int(rng.integers(nx)), int(rng.integers(ny))
                cx, cy = (i + 0.5) * bay + rng.normal(0, 0.5), (j + 0.5) * bay + rng.normal(0, 0.5)
                lev = levels[int(rng.integers(len(levels) - 1))]
                r = float(rng.uniform(0.6, 1.4))
                h = float(rng.uniform(3.0, 7.0))
                mat = self.tank_shell if rng.random() < 0.5 else self.jacket
                T.cyl(mat, (cx, cy, lev + 0.3), (cx, cy, lev + 0.3 + h), r, seg=24, caps=(False, False), name="vessel",
                      **kw)
                T.dome(mat, (cx, cy, lev + 0.3 + h), r, 0.5 * r, seg=24, rings=4, name="vessel", **kw)
        # a stair at the road-side corner
        for z0, z1 in zip((0.0,) + tuple(levels[:-1]), levels):
            run = (z1 - z0) / math.tan(math.radians(40))
            x0 = -1.2
            for sgn in (-1, 1):
                p0 = np.array([x0 + sgn * 0.45, 0.2 if z0 == 0 else 0.2 + run, z0])
                p1 = np.array([x0 + sgn * 0.45, 0.2 + run if z0 == 0 else 0.2, z1])
                T.member(self.steel2, p0, p1, 0.03, 0.2, kind="box", name="stair", **kw)
            break
        return X, Y

    # --- small plant ---------------------------------------------------------------------------

    def pallet(self, c, yaw, z=0.0):
        B = self.B
        R = U.rotz(yaw)
        for i in range(5):
            o = R @ np.array([-0.53 + 0.265 * i, 0, 0])
            B.box(self.wood, (c[0] + o[0], c[1] + o[1], z + 0.133), (0.05, 0.4, 0.011), yaw=yaw, name="pallet")
        for j in (-1, 0, 1):
            o = R @ np.array([0, 0.33 * j, 0])
            B.box(self.wood, (c[0] + o[0], c[1] + o[1], z + 0.07), (0.6, 0.05, 0.05), yaw=yaw, name="pallet")
            B.box(self.wood, (c[0] + o[0], c[1] + o[1], z + 0.011), (0.6, 0.05, 0.011), yaw=yaw, name="pallet")
        return z + 0.144

    def ibc(self, c, yaw):
        B = self.B
        zt = self.pallet(c, yaw)
        R = U.rotz(yaw)
        B.box(self.plastic_w, (c[0], c[1], zt + 0.48), (0.56, 0.46, 0.46), yaw=yaw, name="ibc")
        B.cyl(self.black, (c[0], c[1], zt + 0.93), (c[0], c[1], zt + 1.0), 0.08, seg=10, name="ibc")
        for u in np.linspace(-0.6, 0.6, 5):
            for v in (-0.5, 0.5):
                o = R @ np.array([u, v, 0])
                B.cyl(self.galv, (c[0] + o[0], c[1] + o[1], zt), (c[0] + o[0], c[1] + o[1], zt + 0.97), 0.012,
                      seg=5, name="ibc")
        for z in (zt + 0.3, zt + 0.95):
            pts = [R @ np.array(p) for p in ((-0.6, -0.5, 0), (0.6, -0.5, 0), (0.6, 0.5, 0), (-0.6, 0.5, 0))]
            for p, q in zip(pts, pts[1:] + pts[:1]):
                B.cyl(self.galv, (c[0] + p[0], c[1] + p[1], z), (c[0] + q[0], c[1] + q[1], z), 0.012, seg=5,
                      name="ibc")

    def drums(self, c, yaw):
        zt = self.pallet(c, yaw)
        R = U.rotz(yaw)
        keys = _pieces(self.ctx, BARRELS)
        if not keys:
            return
        k = keys[int(self.rng.integers(len(keys)))]
        for u in (-0.3, 0.3):
            for v in (-0.2, 0.2):
                if self.rng.random() < 0.15:
                    continue
                o = R @ np.array([u, v, 0])
                self.ctx.instance(k, (c[0] + o[0], c[1] + o[1]), zt, float(self.rng.uniform(0, 6.28)), 0.95,
                                  coll="scatter")

    def gas_cage(self, c, yaw):
        B, rng = self.B, self.rng
        R = U.rotz(yaw)
        w, dpt, h = 2.0, 0.9, 2.0
        mesh = self.ctx.looks.pbr(U.pick(self.ctx, ["acg:Fence007A", "acg:Fence008A", "acg:Fence001"]), 0.95)
        corners = [R @ np.array(p) for p in ((-w / 2, -dpt / 2, 0), (w / 2, -dpt / 2, 0), (w / 2, dpt / 2, 0),
                                               (-w / 2, dpt / 2, 0))]
        corners = [np.array([c[0] + p[0], c[1] + p[1]]) for p in corners]
        for p in corners:
            B.box(self.galv, (p[0], p[1], h / 2), (0.03, 0.03, h / 2), yaw=yaw, name="cage")
        for p, q in list(zip(corners, corners[1:] + corners[:1]))[1:]:
            B.panel(mesh, p, q, 0.05, h, tile=0.95, name="cage_mesh")
        B.box(self.galv, (c[0], c[1], h + 0.02), (w / 2 + 0.05, dpt / 2 + 0.05, 0.02), yaw=yaw, name="cage")
        top_rgb = [(0.6, 0.1, 0.05), (0.1, 0.35, 0.15), (0.1, 0.1, 0.1), (0.8, 0.8, 0.8), (0.2, 0.3, 0.6)]
        body = U.paint(self.ctx, [(0.15, 0.2, 0.5), (0.1, 0.1, 0.1), (0.6, 0.55, 0.5), (0.4, 0.1, 0.05)][
            int(rng.integers(4))], rusty=0.2, tile=0.5, bevel=0.0)
        top = self.ctx.looks.paint(top_rgb[int(rng.integers(len(top_rgb)))], rough=0.4, name="gas_top")
        for u in np.arange(-w / 2 + 0.2, w / 2 - 0.1, 0.26):
            for v in (-0.2, 0.15):
                if rng.random() < 0.2:
                    continue
                o = R @ np.array([u, v, 0])
                x, y = c[0] + o[0], c[1] + o[1]
                B.cyl(body, (x, y, 0.0), (x, y, 1.35), 0.115, seg=12, caps=(True, False), name="gas")
                B.dome(top, (x, y, 1.35), 0.115, 0.12, seg=12, rings=3, name="gas")
                B.cyl(self.galv, (x, y, 1.45), (x, y, 1.55), 0.03, seg=6, name="gas")
        self.ctx.claim_rect(c, (w / 2 + 0.1, dpt / 2 + 0.1), yaw, kind="gas_cage")

    def cabinets(self, c, yaw, n):
        B = self.B
        R = U.rotz(yaw)
        col = U.paint(self.ctx, [(0.75, 0.73, 0.66), (0.5, 0.52, 0.52), (0.3, 0.4, 0.35), (0.72, 0.72, 0.7)][
            int(self.rng.integers(4))], rusty=0.0, tile=1.0, bevel=0.008)
        w = 0.8 * n
        B.box(self.concrete, (c[0], c[1], 0.06), (w / 2 + 0.15, 0.4, 0.06), yaw=yaw, name="plinth")
        for i in range(n):
            o = R @ np.array([-w / 2 + 0.4 + 0.8 * i, 0, 0])
            B.box(col, (c[0] + o[0], c[1] + o[1], 0.12 + 1.0), (0.39, 0.3, 1.0), yaw=yaw, name="cabinet")
            o2 = R @ np.array([-w / 2 + 0.7 + 0.8 * i, -0.31, 0])
            B.box(self.black, (c[0] + o2[0], c[1] + o2[1], 1.2), (0.015, 0.012, 0.08), yaw=yaw, name="cabinet")
        B.box(col, (c[0], c[1], 2.14), (w / 2 + 0.05, 0.35, 0.02), yaw=yaw, name="cabinet")
        self.ctx.claim_rect(c, (w / 2 + 0.2, 0.45), yaw, kind="cabinets")

    def pump_skid(self, c, yaw):
        B, rng = self.B, self.rng
        R = U.rotz(yaw)
        f = R @ np.array([1.0, 0, 0])
        c3 = np.array([c[0], c[1], 0.0])
        motor = U.paint(self.ctx, [(0.1, 0.25, 0.5), (0.15, 0.35, 0.2), (0.45, 0.45, 0.45), (0.7, 0.4, 0.1)][
            int(rng.integers(4))], rusty=0.2, tile=0.6, bevel=0.005)
        B.box(self.concrete, (c[0], c[1], 0.1), (1.1, 0.5, 0.1), yaw=yaw, name="plinth")
        B.box(self.steel2, (c[0], c[1], 0.27), (1.0, 0.4, 0.07), yaw=yaw, name="skid")
        B.cyl(motor, c3 + f * 0.1 + U.UP * 0.62, c3 + f * 0.85 + U.UP * 0.62, 0.28, seg=16, name="motor")
        B.cyl(self.steel2, c3 - f * 0.1 + U.UP * 0.62, c3 + f * 0.1 + U.UP * 0.62, 0.12, seg=10, name="motor")
        B.cyl(motor, c3 - f * 0.55 + U.UP * 0.62, c3 - f * 0.15 + U.UP * 0.62, 0.3, seg=16, name="pump")
        pm = self.pipes[int(rng.integers(len(self.pipes)))]
        p = c3 - f * 0.35
        top = float(rng.uniform(1.4, 2.6))
        B.cyl(pm, p + U.UP * 0.9, p + U.UP * top, 0.09, seg=10, name="pipe")
        B.cyl(pm, p + U.UP * 1.2, p + U.UP * 1.28, 0.14, seg=10, name="pipe")
        B.cyl(self.rail, p + U.UP * 1.5 - f * 0.2, p + U.UP * 1.5 + f * 0.2, 0.015, seg=6, name="valve")
        B.cyl(pm, c3 - f * 1.3 + U.UP * 0.62, c3 - f * 0.85 + U.UP * 0.62, 0.1, seg=10, name="pipe")
        self.ctx.claim_rect(c, (1.3, 0.55), yaw, kind="pump")

    def reel(self, c, yaw):
        B = self.B
        R = U.rotz(yaw)
        s = R @ np.array([0, 1.0, 0])
        r = float(self.rng.uniform(0.5, 0.9))
        w = float(self.rng.uniform(0.5, 0.8))
        c3 = np.array([c[0], c[1], r])
        B.cyl(self.wood, c3 - s * w / 2, c3 - s * (w / 2 - 0.05), r, seg=20, name="reel")
        B.cyl(self.wood, c3 + s * (w / 2 - 0.05), c3 + s * w / 2, r, seg=20, name="reel")
        B.cyl(self.black, c3 - s * (w / 2 - 0.05), c3 + s * (w / 2 - 0.05), r * 0.8, seg=20, name="reel")
        self.ctx.claim_rect(c, (r + 0.1, w / 2 + 0.1), yaw, kind="reel")

    def spools(self, c, yaw):
        B, rng = self.B, self.rng
        R = U.rotz(yaw)
        f, s = R @ np.array([1.0, 0, 0]), R @ np.array([0, 1.0, 0])
        L = float(rng.uniform(3.0, 5.5))
        r = float(rng.choice([0.08, 0.1, 0.15, 0.2]))
        mat = self.pipes[int(rng.integers(len(self.pipes)))] if rng.random() < 0.5 else self.galv
        n0 = int(rng.integers(2, 5))
        c3 = np.array([c[0], c[1], 0.0])
        for u in (-L * 0.3, L * 0.3):
            B.box(self.wood, c3 + f * u + U.UP * 0.05, (0.08, n0 * r + 0.2, 0.05), yaw=yaw, name="sleeper")
        for row in range(n0):
            for k in range(n0 - row):
                v = (k - (n0 - row - 1) / 2) * 2 * r
                z = 0.1 + r + row * r * 1.73
                p = c3 + s * v + U.UP * z
                B.cyl(mat, p - f * L / 2, p + f * L / 2, r, seg=12, name="spool")
        self.ctx.claim_rect(c, (L / 2 + 0.1, n0 * r + 0.25), yaw, kind="spools")

    def container(self, c, yaw):
        """A 20 ft ISO container (corrugated, livery-tinted, corner posts), sometimes two high."""
        B, rng, ctx = self.B, self.rng, self.ctx
        liv = [(0.05, 0.22, 0.45), (0.55, 0.08, 0.06), (0.10, 0.35, 0.18), (0.62, 0.62, 0.60), (0.85, 0.42, 0.06),
               (0.30, 0.16, 0.10), (0.08, 0.30, 0.42)]
        steel = ctx.lib.materials(cats_any={"corrugated steel"})
        n = 2 if rng.random() < 0.25 else 1
        z = 0.0
        for _ in range(n):
            skin = ctx.looks.pbr(ctx.lib.pick(rng, steel), 1.2, tint=[min(1.0, 1.6 * v + 0.05) for v in
                                                                      liv[int(rng.integers(len(liv)))]],
                                 bevel=0.02, dirt=0.5, jitter=0.08)
            B.box(skin, (c[0], c[1], z + 1.295), (3.029, 1.219, 1.295), yaw=yaw + float(rng.normal(0, 0.02)),
                  name="container")
            R = U.rotz(yaw)
            for u in (-1, 1):
                for v in (-1, 1):
                    o = R @ np.array([u * 2.95, v * 1.14, 0])
                    B.box(self.steel2, (c[0] + o[0], c[1] + o[1], z + 1.295), (0.09, 0.09, 1.3), yaw=yaw,
                          name="cpost")
            z += 2.59
        self.ctx.claim_rect(c, (3.2, 1.35), yaw, kind="container")

    def skip(self, c, yaw):
        """An open waste skip (tapered steel box) with debris inside."""
        B, rng = self.B, self.rng
        col = U.paint(self.ctx, [(0.8, 0.55, 0.05), (0.2, 0.35, 0.2), (0.15, 0.25, 0.5), (0.7, 0.3, 0.05)][
            int(rng.integers(4))], rusty=0.6, tile=1.0, bevel=0.01)
        B.hexa(col, U.tapered(c, yaw, (1.2, 0.8), (1.8, 0.85), 0.05, 1.2), name="skip")
        B.hexa(self.black, U.tapered(c, yaw, (1.5, 0.7), (1.6, 0.72), 1.2, 1.26), name="skip_fill")
        for s in (-1, 1):
            o = U.rotz(yaw) @ np.array([s * 0.6, 0, 0])
            B.box(col, (c[0] + o[0], c[1] + o[1], 0.03), (0.08, 0.85, 0.03), yaw=yaw, name="skip")
        self.ctx.claim_rect(c, (1.9, 0.95), yaw, kind="skip")

    def pallet_stack(self, c, yaw):
        z = 0.0
        for _ in range(int(self.rng.integers(4, 14))):
            z = self.pallet((c[0] + self.rng.normal(0, 0.02), c[1] + self.rng.normal(0, 0.02)),
                            yaw + float(self.rng.normal(0, 0.03)), z)
        self.ctx.claim_rect(c, (0.65, 0.45), yaw, kind="pallets")

    def bigbags(self, c, yaw):
        B, rng = self.B, self.rng
        R = U.rotz(yaw)
        bag = self.ctx.looks.paint((0.82, 0.8, 0.72) if rng.random() < 0.7 else (0.15, 0.3, 0.55), rough=0.9,
                                   name="bigbag")
        n = int(rng.integers(1, 4))
        for i in range(n):
            o = R @ np.array([(i - (n - 1) / 2) * 1.2, 0, 0])
            zt = self.pallet((c[0] + o[0], c[1] + o[1]), yaw)
            B.hexa(bag, U.tapered((c[0] + o[0], c[1] + o[1]), yaw, (0.45, 0.45), (0.5, 0.5), zt, zt + 0.95),
                   name="bigbag")
            B.hexa(bag, U.tapered((c[0] + o[0], c[1] + o[1]), yaw, (0.5, 0.5), (0.3, 0.3), zt + 0.95, zt + 1.08),
                   name="bigbag")
        self.ctx.claim_rect(c, (0.62 * n + 0.1, 0.5), yaw, kind="bigbags")

    def beams(self, c, yaw):
        B, rng = self.B, self.rng
        R = U.rotz(yaw)
        f, s = R @ np.array([1.0, 0, 0]), R @ np.array([0, 1.0, 0])
        L = float(rng.uniform(4, 7))
        h = float(rng.choice([0.2, 0.3, 0.4]))
        mat = self.steel if rng.random() < 0.5 else U.paint(self.ctx, (0.35, 0.12, 0.06), rusty=0.8, tile=1.0)
        c3 = np.array([c[0], c[1], 0.0])
        for u in (-L * 0.3, L * 0.3):
            B.box(self.wood, c3 + f * u + U.UP * 0.06, (0.08, 0.9, 0.06), yaw=yaw, name="sleeper")
        rows = int(rng.integers(1, 4))
        for row in range(rows):
            for k in range(3):
                p = c3 + s * (k - 1) * (h * 0.7 + 0.1) + U.UP * (0.12 + h / 2 + row * h)
                B.member(mat, p - f * L / 2, p + f * L / 2, h * 0.6, h, name="beam")
        self.ctx.claim_rect(c, (L / 2 + 0.1, 0.9), yaw, kind="beams")

    def transformer(self, c, yaw):
        B = self.B
        R = U.rotz(yaw)
        col = U.paint(self.ctx, (0.45, 0.5, 0.48), rusty=0.4, tile=1.0, bevel=0.01)
        B.box(self.concrete, (c[0], c[1], 0.1), (1.2, 0.9, 0.1), yaw=yaw, name="plinth")
        B.box(col, (c[0], c[1], 1.1), (0.8, 0.5, 0.9), yaw=yaw, name="trafo")
        for u in np.linspace(-0.7, 0.7, 12):
            for sgn in (-1, 1):
                o = R @ np.array([u, sgn * 0.65, 0])
                B.box(col, (c[0] + o[0], c[1] + o[1], 1.0), (0.012, 0.15, 0.7), yaw=yaw, name="trafo")
        ins = self.ctx.looks.paint((0.35, 0.18, 0.1), rough=0.3, name="insulator")
        for u in (-0.45, 0.0, 0.45):
            o = R @ np.array([u, 0, 0])
            B.cyl(ins, (c[0] + o[0], c[1] + o[1], 2.0), (c[0] + o[0], c[1] + o[1], 2.45), 0.07, 0.05, seg=10,
                  name="trafo")
        self.ctx.claim_rect(c, (1.2, 0.95), yaw, kind="trafo")

    def forklift(self, c, yaw):
        """A counterbalance forklift: body, counterweight, mast, forks, overhead guard, wheels."""
        B, rng = self.B, self.rng
        R = U.rotz(yaw)
        f, s = R @ np.array([1.0, 0, 0]), R @ np.array([0, 1.0, 0])
        col = U.paint(self.ctx, [(0.85, 0.55, 0.05), (0.8, 0.3, 0.05), (0.7, 0.1, 0.06), (0.15, 0.4, 0.2),
                                 (0.2, 0.3, 0.55)][int(rng.integers(5))], rusty=0.2, tile=1.0, bevel=0.02)
        c3 = np.array([c[0], c[1], 0.0])
        B.box(col, c3 + U.UP * 0.6, (0.9, 0.52, 0.35), yaw=yaw, name="forklift")
        B.box(self.steel2, c3 - f * 0.95 + U.UP * 0.75, (0.25, 0.55, 0.45), yaw=yaw, name="forklift")
        B.box(self.black, c3 - f * 0.2 + U.UP * 1.05, (0.3, 0.3, 0.12), yaw=yaw, name="seat")
        for u, v in ((-0.7, -1), (-0.7, 1), (0.55, -1), (0.55, 1)):
            p = c3 + f * u + s * v * 0.5 + U.UP * 0.3
            B.cyl(self.black, p - s * 0.12, p + s * 0.12, 0.3, seg=16, name="wheel")
        for u, v in ((-0.55, -0.48), (-0.55, 0.48), (0.55, -0.48), (0.55, 0.48)):
            p = c3 + f * u + s * v
            B.cyl(self.steel2, p + U.UP * 0.9, p + U.UP * 2.15, 0.03, seg=6, name="guard")
        B.box(self.steel2, c3 + U.UP * 2.17, (0.6, 0.5, 0.025), yaw=yaw, name="guard")
        mh = float(rng.uniform(2.0, 2.4))
        for v in (-0.35, 0.35):
            p = c3 + f * 1.05 + s * v
            B.box(self.steel2, p + U.UP * mh / 2, (0.05, 0.06, mh / 2), yaw=yaw, name="mast")
        fz = float(rng.uniform(0.05, 0.4))
        B.box(self.steel2, c3 + f * 1.13 + U.UP * (fz + 0.35), (0.03, 0.45, 0.35), yaw=yaw, name="carriage")
        for v in (-0.25, 0.25):
            B.box(self.steel2, c3 + f * 1.7 + s * v + U.UP * (fz + 0.025), (0.55, 0.05, 0.025), yaw=yaw, name="fork")
        self.ctx.claim_rect(c + (f[:2] * 0.35), (1.65, 0.65), yaw, kind="forklift")

    def sign(self, p, yaw, z, w=0.6, h=0.45):
        """A safety sign (danger / first aid / hazard stripes) as a UV-mapped plate."""
        ctx = self.ctx
        k = U.pick(ctx, ["acg:PaintedMetal017", "acg:PaintedMetal018", "acg:PaintedMetal016",
                         "acg:PaintedMetal015"])
        if not k:
            return
        m = ctx.looks.pbr(k, 1.0, proj="uv", antitile=False, vary=0.0, jitter=0.0, dirt=0.0)
        R = U.rotz(yaw)
        s = (R @ np.array([1.0, 0, 0]))[:2]
        a, b = np.asarray(p, float) - s * w / 2, np.asarray(p, float) + s * w / 2
        self.B.quad(m, (*a, z - h / 2), (*b, z - h / 2), (*b, z + h / 2), (*a, z + h / 2),
                    uv=[(0, 0), (1, 0), (1, 1), (0, 1)], name="sign")

    def bollard(self, xy):
        B = self.B
        col = self.rail
        B.cyl(col, (xy[0], xy[1], 0.0), (xy[0], xy[1], 1.0), 0.1, seg=12, name="bollard")
        B.dome(col, (xy[0], xy[1], 1.0), 0.1, 0.06, seg=12, rings=3, name="bollard")
        self.ctx.claim_disc(xy, 0.15, kind="bollard")

    def tee_pipe_run(self, p, q, z, n=2):
        """Pipes at height ``z`` on two T-supports from p to q (the hang obstacle)."""
        B, rng = self.B, self.rng
        p, q = np.asarray(p, float), np.asarray(q, float)
        d = (q - p) / np.linalg.norm(q - p)
        nrm = np.array([-d[1], d[0]])
        for c in (p, q):
            B.member(self.steel, (*c, 0.0), (*c, z - 0.12), 0.15, 0.15, roll=(*d, 0.0), name="tee")
            B.member(self.steel, (*(c - nrm * 0.45), z - 0.08), (*(c + nrm * 0.45), z - 0.08), 0.12, 0.12, name="tee")
            B.box(self.concrete, (c[0], c[1], 0.08), (0.25, 0.25, 0.08), name="plinth")
            self.ctx.claim_disc(c, 0.3, kind="tee")
        for i in range(n):
            r = float(rng.choice([0.05, 0.08, 0.1, 0.12]))
            o = nrm * (-0.3 + 0.6 * (i + 0.5) / n)
            mat = self.pipes[int(rng.integers(len(self.pipes)))]
            B.cyl(mat, (*(p + o - d * 0.3), z + r - 0.02), (*(q + o + d * 0.3), z + r - 0.02), r, seg=12, name="pipe")

    def sleeper_run(self, p, q, n=3):
        """Low pipes on concrete sleepers (0.3-0.7 m) along p -> q."""
        B, rng = self.B, self.rng
        p, q = np.asarray(p, float), np.asarray(q, float)
        L = float(np.linalg.norm(q - p))
        d = (q - p) / L
        nrm = np.array([-d[1], d[0]])
        zs = float(rng.uniform(0.25, 0.45))
        for t in np.arange(0.5, L, 4.0):
            c = p + d * t
            B.box(self.concrete, (c[0], c[1], zs / 2), (0.15, 0.5 + 0.12 * n, zs / 2), yaw=math.atan2(d[1], d[0]),
                  name="sleeper")
        u = -0.1 * n
        for i in range(n):
            r = float(rng.choice([0.05, 0.08, 0.1, 0.15]))
            mat = self.pipes[int(rng.integers(len(self.pipes)))]
            o = nrm * (u + r)
            u += 2 * r + 0.12
            B.cyl(mat, (*(p + o), zs + r), (*(q + o), zs + r), r, seg=12, name="pipe")
        c = (p + q) / 2
        self.ctx.claim_rect(c, (L / 2, 0.6 + 0.12 * n), math.atan2(d[1], d[0]), kind="sleepers")

    def duct_run(self, a, b, z, w=0.8, h=0.6, mat=None):
        """A rectangular duct with flanges every 1.5 m."""
        B = self.B
        mat = mat or self.galv
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        d = (b - a) / L
        yaw = math.atan2(d[1], d[0])
        c = (a + b) / 2
        B.box(mat, (c[0], c[1], z), (L / 2, w / 2, h / 2), yaw=yaw, name="duct")
        for t in np.arange(0.0, L + 1e-6, 1.5):
            q = a + d * t
            B.box(mat, (q[0], q[1], z), (0.025, w / 2 + 0.03, h / 2 + 0.03), yaw=yaw, name="duct")


# --- scene ----------------------------------------------------------------------------------------

def _anchor(ctx, half, yaw):
    """Where yard equipment goes: mostly against the patch's long edges or
    along a patch wall (aligned to it), sometimes anywhere."""
    rng = ctx.rng
    x0, y0, x1, y1 = ctx.bounds
    u = rng.random()
    if u < 0.55:
        side = -1 if rng.random() < 0.5 else 1
        off = float(rng.uniform(0.3, 1.8))
        yaw = float(rng.choice([0.0, math.pi]))
        y = (y0 + half[1] + off) if side < 0 else (y1 - half[1] - off)
        return np.array([float(rng.uniform(x0 + half[0], x1 - half[0])), y]), yaw
    walls = [p for p in ctx.spec["primitives"] if p["kind"] == "wall"]
    if u < 0.8 and walls:
        p = walls[int(rng.integers(len(walls)))]
        a, b = np.asarray(p["p0"], float)[:2], np.asarray(p["p1"], float)[:2]
        L = float(np.linalg.norm(b - a))
        d = (b - a) / max(L, 1e-9)
        nrm = np.array([-d[1], d[0]]) * (1 if rng.random() < 0.5 else -1)
        t = float(rng.uniform(half[0], max(half[0] + 0.01, L - half[0])))
        xy = a + d * t + nrm * (p["t"] / 2 + half[1] + float(rng.uniform(0.3, 0.9)))
        return xy, math.atan2(d[1], d[0])
    return ctx.uniform_xy(pad=1.0), yaw


#: Objaverse yard equipment by weight (``U.OBJ`` roles); gas bottles and drums come in groups
OBJ_YARD = [("forklift", 3), ("generator", 2), ("dumpster", 2), ("ibc", 1), ("gas", 1), ("barrel", 1),
            ("container", 1)]


def _obj_yard(ctx, ko) -> int:
    """Objaverse equipment against the patch edges and walls, like the
    procedural equipment: forklifts, generators, dumpsters, IBC totes, gas
    bottles, drums, 20 ft containers (real scans at their real size)."""
    rng = ctx.rng
    roles = [(r, w) for r, w in OBJ_YARD if U.obj_keys(ctx, r)]
    if not roles:
        return 0
    w = np.array([v for _, v in roles], float)
    n, placed = ctx.count(3.0), 0
    for _ in range(n * 20):
        if placed >= n:
            break
        role = roles[int(rng.choice(len(roles), p=w / w.sum()))][0]
        keys = U.obj_keys(ctx, role)
        k = keys[int(rng.integers(len(keys)))]
        xy, d = _anchor(ctx, U.long_half(ctx, k), float(rng.choice([0.0, math.pi / 2])))
        yaw = U.along_yaw(ctx, k, d) + float(rng.normal(0, 0.05))
        if U.place_obj(ctx, k, xy, yaw, margin=0.5 * ko + 0.3) is None:
            continue
        placed += 1
        if role in ("gas", "barrel"):          # a group of them, touching
            e = np.array([math.cos(d), math.sin(d)])
            step = 2 * U.long_half(ctx, k)[0] + 0.04
            first = len(ctx.foot) - 1           # the group ignores its own footprints
            for m in range(int(rng.integers(1, 5))):
                q = np.asarray(xy) + e * step * (m + 1)
                if U.place_obj(ctx, k, q, float(rng.uniform(0, 2 * math.pi)), margin=0.5 * ko + 0.1,
                               ignore_from=first) is None:
                    break
    return placed


def _near_prim(ctx, q, dist):
    for f in ctx.foot:
        if f.get("kind") == "prim" and f["type"] in ("rect", "disc"):
            from build_course import footprint_distance
            if 0.0 < footprint_distance(np.asarray(q, float), f) < dist:
                return True
    return False


def _sky(ctx, weather):
    lib, rng = ctx.lib, ctx.rng
    pool = {"dry": HDRI_DRY, "wet": HDRI_WET, "snow": HDRI_SNOW}[weather]
    ks = U.keys_in(lib, pool)
    if not ks:
        ks = lib.hdris(cats_all={"outdoor"}, cats_none={"night", "indoor"}, tags_any={"industrial", "parking"})
    return ctx.looks.sky(lib.pick(rng, ks))


def _ground(ctx, weather, age):
    lib, rng, lk = ctx.lib, ctx.rng, ctx.looks
    fam = str(rng.choice(["concrete", "concrete", "asphalt", "gravel"]))
    if weather == "snow":
        k0 = U.pick(ctx, SNOW_GROUND)
        keys = [k0, U.pick(ctx, SNOW_GROUND), k0]
    elif fam == "concrete":
        k0 = U.pick(ctx, CONC_GROUND)
        keys = [k0, U.pick(ctx, CONC_GROUND), k0 if age < 0.5 else U.pick(ctx, GRAV_GROUND)]
    elif fam == "asphalt":
        k0 = U.pick(ctx, ASPH_GROUND)
        keys = [k0, U.pick(ctx, ASPH_GROUND), k0 if age < 0.5 else U.pick(ctx, CONC_GROUND)]
    else:
        k0 = U.pick(ctx, GRAV_GROUND)
        keys = [k0, U.pick(ctx, GRAV_GROUND), k0]
    wet = {"dry": 0.0, "wet": float(rng.uniform(0.5, 1.0)), "snow": 0.0}[weather]
    if weather == "dry" and rng.random() < 0.3:
        wet = 0.25
    if weather == "snow":        # a ploughed lane down the road, soft-edged
        x0, y0, x1, y1 = ctx.bounds
        zx0, zy0, zx1, zy1 = ctx.zone
        lane = U.soften(lk.pbr(U.pick(ctx, ["ph:asphalt_snow", "ph:snow_floor", "acg:Snow013", "acg:Snow012"]), 2.5,
                               proj="flat", vary=0.175, jitter=0.0), "strip", width=0.35, noise_scale=0.6,
                        noise_amp=0.7)
        y = float(rng.uniform(y0 + 2, y1 - 2))
        U.overlay_strip(ctx, lane, (zx0 - 30, y), (zx1 + 30, y), float(rng.uniform(4, 8)), z=0.003, name="lane")
    ctx.ground(lk.ground_material(keys, tile_m=float(rng.uniform(3.0, 4.5)), puddles=wet,
                                  blend_scale=float(rng.uniform(8.0, 16.0)) * (2.0 if weather == "snow" else 1.0)))
    return fam, keys, wet


def _dress(ctx, P, B):
    rng, lk = ctx.rng, ctx.looks
    wk = str(rng.choice(["clad", "clad", "block", "precast", "brick"]))
    if wk == "clad":
        wall = P.clad
    elif wk == "block":
        wall = P.block
    elif wk == "precast":
        wall = P.concrete
    else:
        wall = U.scan_native(ctx, BRICK_IND, 1.6, ctx.lib.materials(cats_any={"bricks"}), bevel=0.015, dirt=0.5)
    bk = str(rng.choice(["steel", "concrete", "clad"]))
    box = {"steel": P.steel2, "concrete": P.concrete, "clad": P.clad2}[bk]
    pole = P.steel if rng.random() < 0.7 else P.column
    bar = P.pipes[int(rng.integers(len(P.pipes)))]
    rock = P.column
    ctx.dress_primitives({"wall": wall, "box": box, "pole": pole, "bar": bar, "rock": rock})
    _wall_trim(ctx, P, B, wk)
    for p in ctx.spec["primitives"]:
        if p["kind"] == "wall" and ctx.rng.random() < 0.6 and p["z1"] - p["z0"] > 2.0:
            p0, p1 = np.asarray(p["p0"], float)[:2], np.asarray(p["p1"], float)[:2]
            L = float(np.linalg.norm(p1 - p0))
            if L < 1.5:             # a short wall (wall_offset) has no room for a sign
                continue
            d = (p1 - p0) / L
            nrm = np.array([-d[1], d[0]]) * (1 if ctx.rng.random() < 0.5 else -1)
            slots = [(sc - w / 2 - 0.5, sc + w / 2 + 0.5) for sc, w, _, _ in p.get("slots", [])]
            t = float(ctx.rng.uniform(0.6, L - 0.6))
            if not any(a < t < b for a, b in slots):
                q = p0 + d * t + nrm * (p["t"] / 2 + 0.01)
                P.sign(q, math.atan2(d[1], d[0]), float(ctx.rng.uniform(1.4, 1.9)))
    cap = P.concrete if wk in ("block", "brick", "precast") else P.galv
    U.wall_caps(ctx, B, cap, ctx.spec, overhang=0.03 if wk != "clad" else 0.02, thick=0.06 if wk != "clad" else 0.03)
    for p in ctx.spec["primitives"]:            # base plates (round, 8 cm proud) under steel poles on the ground
        if p["kind"] == "cyl_v" and p["z0"] < 0.05 and pole is P.steel:
            B.cyl(P.steel, (p["c"][0], p["c"][1], 0.0), (p["c"][0], p["c"][1], 0.03), p["r"] + 0.08, seg=24,
                  name="baseplate")
    tops = _pole_tops(ctx, P, B, pole)
    return {"wall": wk, "box": bk, "pole_tops": tops}


def _wall_trim(ctx, P, B, wk):
    """Clad walls get steel posts on their faces, precast walls panel joints --
    only on solid stretches (>= 0.4 m from a slot edge), proud by <= 0.12 m."""
    for p in ctx.spec["primitives"]:
        if p["kind"] != "wall":
            continue
        p0, p1 = np.asarray(p["p0"], float)[:2], np.asarray(p["p1"], float)[:2]
        L = float(np.linalg.norm(p1 - p0))
        if L < 1.0:
            continue
        d = (p1 - p0) / L
        nrm = np.array([-d[1], d[0]])
        yaw = math.atan2(d[1], d[0])
        slots = [(sc - w / 2 - 0.4, sc + w / 2 + 0.4) for sc, w, _, _ in p.get("slots", [])]
        step = float(ctx.rng.uniform(3.0, 5.0)) if wk == "clad" else float(ctx.rng.uniform(2.4, 3.2))
        for t in np.arange(step / 2, L - 0.2, step):
            if any(a < t < b for a, b in slots):
                continue
            for sgn in (-1, 1):
                q = p0 + d * t + nrm * sgn * (p["t"] / 2 + (0.06 if wk == "clad" else 0.004))
                h = p["z1"] - p["z0"]
                if wk == "clad":
                    B.box(P.steel, (q[0], q[1], p["z0"] + h / 2), (0.08, 0.06, h / 2), yaw=yaw, name="wall_post")
                elif wk == "precast":
                    B.box(P.black, (q[0], q[1], p["z0"] + h / 2), (0.012, 0.004, h / 2), yaw=yaw, name="joint")


def _pole_tops(ctx, P, B, pole):
    """Above the cloud band only (z > 4.8): tall poles carry a rack of beams
    and pipes (a stand of columns reads as the underside of a pipe deck), fat
    poles get vessel heads. Nothing here reaches the flight band."""
    rng = ctx.rng
    tall = [p for p in ctx.spec["primitives"]
            if p["kind"] == "cyl_v" and p["z1"] >= max(4.8, U.ZSPLIT - 0.1) and p["z0"] < 0.1]
    if not tall:
        return None
    fat = [p for p in tall if p["r"] >= 0.55]
    for p in fat:
        c = p["c"]
        B.dome(P.tank_shell, (c[0], c[1], p["z1"]), p["r"], 0.45 * p["r"], seg=32, rings=5, name="vessel_head")
        B.cyl(P.rail, (c[0], c[1], p["z1"] + 0.45 * p["r"]), (c[0], c[1], p["z1"] + 0.45 * p["r"] + 0.5), 0.08,
              seg=10, name="vessel_head")
    thin = [p for p in tall if p["r"] < 0.55]
    if len(thin) < 3 or rng.random() < 0.25:
        return "vessels" if fat else None
    zt = max(p["z1"] for p in thin) + float(rng.uniform(0.6, 1.8))
    C = np.array([p["c"][:2] for p in thin], float)
    for p in thin:
        B.cyl(pole, (p["c"][0], p["c"][1], p["z1"]), (p["c"][0], p["c"][1], zt + 0.2), p["r"], seg=24,
              name="pole_top")
    edges = set()
    for i, c in enumerate(C):
        d = np.linalg.norm(C - c, axis=1)
        for j in np.argsort(d)[1:3]:
            if d[j] < 7.5:
                edges.add((min(i, j), max(i, j)))
    pm = [P.pipes[int(rng.integers(len(P.pipes)))] for _ in range(2)]
    for i, j in edges:
        a, b = C[i], C[j]
        B.member(P.steel, (*a, zt), (*b, zt), 0.2, 0.3, name="deck_beam")
        if rng.random() < 0.3:
            d = (b - a) / max(np.linalg.norm(b - a), 1e-6)
            nrm = np.array([-d[1], d[0]])
            u = -0.25
            for _ in range(int(rng.integers(1, 4))):
                r = float(rng.choice([0.06, 0.1, 0.15, 0.2]))
                mat = pm[int(rng.integers(2))]
                o = nrm * (u + r)
                u += 2 * r + 0.1
                B.cyl(mat, (*(a + o - d * 0.3), zt + 0.15 + r), (*(b + o + d * 0.3), zt + 0.15 + r), r, seg=12,
                      name="pipe")
    return "pipe_deck"


def _side_rack(ctx, P, side, x_a, x_b, roof):
    y0, y1 = ctx.bounds[1], ctx.bounds[3]
    rng = ctx.rng
    width = float(rng.uniform(2.4, 4.0))
    tiers = (float(rng.uniform(3.8, 4.6)),) if rng.random() < 0.4 else \
        (float(rng.uniform(3.8, 4.4)), float(rng.uniform(5.8, 6.6)))
    for off in (1.2, 2.2, 3.2):
        yc = (y0 - off - width / 2) if side < 0 else (y1 + off + width / 2)
        n = P.rack((x_a, yc), (x_b, yc), width=width, tiers=tiers, spacing=float(rng.uniform(5.0, 6.5)), roof=roof)
        if n:
            return yc, width
    return None, width


def _side_shed(ctx, P, B, side, x_a, x_b, roof, far_off=0.0):
    rng = ctx.rng
    y0, y1 = ctx.bounds[1], ctx.bounds[3]
    yf = (y0 - 2.5 - far_off) if side < 0 else (y1 + 2.5 + far_off)
    x = x_a
    built = []
    while x < x_b - 8:
        L = float(rng.uniform(18, 40))
        xe = min(x + L, x_b)
        h = float(rng.uniform(6.0, 10.0))
        depth = float(rng.uniform(14, 28))
        bay = float(rng.uniform(5.0, 7.5))
        clad = P.clad if rng.random() < 0.6 else P.clad2
        doors = []
        for xd in np.arange(x + bay * 1.5, xe - bay, bay * float(rng.choice([2, 3]))):
            if rng.random() < 0.7:
                doors.append((float(xd), float(rng.uniform(3.5, 5.0)), float(min(h - 1.2, rng.uniform(4.0, 5.2)))))
        if not ctx.free_rect(((x + xe) / 2, yf - side * 0.3), ((xe - x) / 2, 0.4), margin=0.2):
            x = xe + float(rng.uniform(3, 8))
            continue
        U.gable_shed(B, clad, P.roof, P.steel2, x, xe, yf, depth, side, h, pitch=float(rng.uniform(5, 15)),
                     plinth=P.block if rng.random() < 0.6 else P.concrete, door=P.shutter, glass=P.glass, bay=bay,
                     doors=doors, roof_flag=roof, gutter=P.galv)
        ctx.claim_rect(((x + xe) / 2, yf + side * depth / 2), ((xe - x) / 2 + 0.4, depth / 2 + 0.4), kind="shed")
        # wall furniture: downpipes, a duct along the wall, lamps, aircon units
        for xp in np.arange(x + 0.4, xe, bay * 2):
            B.cyl(P.galv, (xp, yf - side * 0.2, 0.0), (xp, yf - side * 0.2, h - 0.1), 0.05, seg=8, name="downpipe",
                  roof=roof)
        if rng.random() < 0.5:
            P.duct_run((x + 1.0, yf - side * 0.65), (xe - 1.0, yf - side * 0.65), float(rng.uniform(3.2, 4.2)),
                       w=float(rng.uniform(0.5, 0.9)), h=float(rng.uniform(0.4, 0.7)))
            for xb in np.arange(x + 2.0, xe - 1.0, 3.0):
                B.box(P.galv, (xb, yf - side * 0.4, 3.0), (0.03, 0.3, 0.03), name="bracket", roof=roof)
        for xl in np.arange(x + bay, xe - 1, bay * 2):
            B.box(P.steel2, (xl, yf - side * 0.3, h - 1.0), (0.2, 0.2, 0.08), name="lamp", roof=roof)
            B.box(P.lamp, (xl, yf - side * 0.35, h - 1.1), (0.15, 0.15, 0.02), name="lamp", roof=roof)
        ac = _pieces(ctx, ["ph:exterior_aircon_unit#0", "ph:exterior_aircon_unit#1", "ph:utility_box_02#0",
                           "ph:power_box_01#0"])
        if ac:
            for xa in np.arange(x + 3.0, xe - 2, float(rng.uniform(5, 12))):
                if any(abs(xa - xd) < wd / 2 + 1.0 for xd, wd, _ in doors):
                    continue
                k = ac[int(rng.integers(len(ac)))]
                sy = ctx.lib.size(k)
                yy = yf - side * (0.1 + sy[1] / 2 + 0.05)
                if ctx.free((xa, yy), 0.4, 0.1):
                    ctx.instance(k, (xa, yy), 0.0, 0.0 if side > 0 else math.pi, 1.0, coll="scatter", claim=True)
        built.append((x, xe, h, depth))
        x = xe + float(rng.uniform(3, 10))
    return built


def _side_tanks(ctx, P, B, side, x_a, x_b, roof):
    rng = ctx.rng
    y0, y1 = ctx.bounds[1], ctx.bounds[3]
    x = x_a
    yb = (y0 - 1.5) if side < 0 else (y1 + 1.5)
    placed = []
    while x < x_b:
        r = float(rng.uniform(2.5, 6.5))
        h = float(rng.uniform(6, 14)) if r > 3.5 else float(rng.uniform(5, 9))
        yc = yb + side * (1.8 + r)
        xc = x + r
        if xc > x_b + 4:
            break
        if not ctx.free((xc, yc), r, 0.5):
            x += 2 * r + 2
            continue
        kind = str(rng.choice(["cone", "dome", "cone", "flat"]))
        shell = P.tank_shell if rng.random() < 0.7 else P.tank_shell2
        U.tank(B, shell, shell if kind != "flat" else P.steel2, P.rail, (xc, yc), r, h, kind=kind,
               stair_mat=P.grating, roof_flag=roof)
        ctx.claim_disc((xc, yc), r + 0.9, kind="tank")
        placed.append((xc, yc, r))
        x += 2 * r + float(rng.uniform(2.5, 5.0))
    if placed:     # bund wall around the group
        xa, xb = placed[0][0] - placed[0][2] - 1.2, placed[-1][0] + placed[-1][2] + 1.2
        ya = yb
        yz = yb + side * (1.8 + 2 * max(p[2] for p in placed) + 1.8)
        hb = float(rng.uniform(0.8, 1.2))
        for (p, q) in (((xa, ya), (xb, ya)), ((xa, yz), (xb, yz)), ((xa, ya), (xa, yz)), ((xb, ya), (xb, yz))):
            p, q = np.asarray(p), np.asarray(q)
            c = (p + q) / 2
            L = float(np.linalg.norm(q - p))
            yaw = math.atan2(q[1] - p[1], q[0] - p[0])
            B.box(P.concrete, (c[0], c[1], hb / 2), (L / 2 + 0.12, 0.12, hb / 2), yaw=yaw, name="bund", roof=roof)
        ctx.claim_rect(((xa + xb) / 2, ya), ((xb - xa) / 2, 0.3), kind="bund")
        # a bullet vessel in front of the bund, if it fits
        if rng.random() < 0.6:
            L = float(rng.uniform(5, 9))
            rv = float(rng.uniform(0.9, 1.4))
            xv = float(rng.uniform(x_a + L / 2, max(x_a + L / 2 + 0.1, x_b - L / 2)))
            yv = yb - side * (rv + 0.7)
            if ctx.free_rect((xv, yv), (L / 2 + 0.5, rv + 0.2), margin=0.5 * float(ctx.knobs.get("keepout", 2.5))):
                U.vessel_h(B, P.tank_shell2, P.concrete, (xv, yv), 0.0, rv, L, roof_flag=roof)
                ctx.claim_rect((xv, yv), (L / 2 + 0.6, rv + 0.2), kind="vessel")
    return placed


def _backdrop(ctx, P, B, weather):
    """Render-only plant beyond the zone: sheds, tanks, columns, stacks, frames, masts, poles."""
    rng = ctx.rng
    zx0, zy0, zx1, zy1 = ctx.zone
    far = {"far": True}
    info = {"stacks": 0, "columns": 0}
    for side in (-1, 1):
        roof = side < 0
        y_near = (zy0 - 8.0) if side < 0 else (zy1 + 8.0)
        x = zx0 - 40.0
        while x < zx1 + 40.0:
            kind = str(rng.choice(["shed", "shed", "tanks", "frame", "columns", "gap"]))
            if kind == "shed":
                L = float(rng.uniform(25, 60))
                h = float(rng.uniform(8, 16))
                depth = float(rng.uniform(20, 40))
                yf = y_near + side * float(rng.uniform(0, 12))
                bay = float(rng.uniform(6, 8))
                doors = [(float(xd), 5.0, 5.5) for xd in np.arange(x + bay, x + L - bay, bay * 3) if rng.random() < 0.5]
                U.gable_shed(B, P.clad if rng.random() < 0.5 else P.clad2, P.roof, P.steel2, x, x + L, yf, depth, side,
                             h, pitch=float(rng.uniform(4, 15)), plinth=P.block, door=P.shutter, glass=P.glass,
                             bay=bay, doors=doors, roof_flag=roof, gutter=P.galv, far=True)
                x += L + float(rng.uniform(4, 15))
            elif kind == "tanks":
                for _ in range(int(rng.integers(2, 5))):
                    r = float(rng.uniform(4, 12))
                    h = float(rng.uniform(8, 18))
                    yc = y_near + side * (r + float(rng.uniform(0, 15)))
                    shell = P.tank_shell if rng.random() < 0.6 else P.tank_shell2
                    U.tank(B, shell, shell, P.rail, (x + r, yc), r, h, kind=str(rng.choice(["cone", "dome"])),
                           stair_mat=P.grating, far=True, roof_flag=roof, seg=48)
                    x += 2 * r + float(rng.uniform(2, 6))
            elif kind == "frame":
                T = U.Xf(B, (x, y_near + side * 2.0), 0.0 if side > 0 else 0.0)
                nx, ny = int(rng.integers(2, 4)), int(rng.integers(1, 3))
                if side < 0:
                    T = U.Xf(B, (x, y_near - ny * 6.0 - 2.0), 0.0)
                X, _ = P.process_frame(T, nx, ny, 6.0, levels=(5.0, 9.5, 14.0, 18.5)[:int(rng.integers(2, 5))],
                                       roof=roof)
                for i in range(int(rng.integers(1, 3))):   # columns rising from the frame
                    xc = x + float(rng.uniform(1, X - 1))
                    yc = y_near + side * float(rng.uniform(3, 9))
                    r = float(rng.uniform(0.8, 2.0))
                    h = float(rng.uniform(20, 40))
                    B.cyl(P.jacket, (xc, yc, 0.0), (xc, yc, h), r, seg=24, caps=(False, False), far=True, roof=roof,
                          name="column")
                    B.dome(P.jacket, (xc, yc, h), r, 0.5 * r, seg=24, far=True, roof=roof, name="column")
                    for z in np.arange(6.0, h, 6.0):
                        B.cyl(P.grating, (xc, yc, z), (xc, yc, z + 0.05), r + 0.9, seg=24, far=True, roof=roof,
                              name="platform")
                    info["columns"] += 1
                x += X + float(rng.uniform(4, 12))
            elif kind == "columns":
                for _ in range(int(rng.integers(1, 4))):
                    r = float(rng.uniform(0.7, 2.2))
                    h = float(rng.uniform(18, 45))
                    yc = y_near + side * float(rng.uniform(2, 20))
                    B.cyl(P.jacket if rng.random() < 0.6 else P.tank_shell, (x, yc, 0.0), (x, yc, h), r, seg=24,
                          caps=(False, False), far=True, roof=roof, name="column")
                    B.dome(P.jacket, (x, yc, h), r, 0.5 * r, seg=24, far=True, roof=roof, name="column")
                    for z in np.arange(5.0, h, float(rng.uniform(5, 8))):
                        B.cyl(P.grating, (x, yc, z), (x, yc, z + 0.05), r + 1.0, seg=24, far=True, roof=roof,
                              name="platform")
                        B.cyl(P.rail, (x, yc, z + 1.0), (x, yc, z + 1.03), r + 1.0, seg=24, caps=(False, False),
                              far=True, roof=roof, name="platform")
                    info["columns"] += 1
                    x += 2 * r + float(rng.uniform(3, 8))
            else:
                x += float(rng.uniform(8, 20))
        # a pipe rack along the plant road beyond the zone
        if rng.random() < 0.7:
            yc = y_near - side * 3.0
            P.rack((zx0 - 30, yc), (zx1 + 30, yc), width=float(rng.uniform(3, 5)),
                   tiers=(float(rng.uniform(4.5, 5.5)), float(rng.uniform(6.8, 8.0))), spacing=6.0, roof=roof,
                   check=False, drops=False)
        # light masts
        for xm in np.arange(zx0 - 20, zx1 + 20, float(rng.uniform(30, 50))):
            U.light_mast(B, P.galv, P.lamp, (xm, y_near - side * 1.0), float(rng.uniform(18, 28)), far=True,
                         roof=roof)
    # stacks on the skyline
    for _ in range(int(rng.integers(1, 4))):
        side = int(rng.choice([-1, 1]))
        x = float(rng.uniform(zx0 - 30, zx1 + 30))
        y = (zy1 + float(rng.uniform(30, 70))) if side > 0 else (zy0 - float(rng.uniform(30, 70)))
        r = float(rng.uniform(1.0, 2.8))
        h = float(rng.uniform(35, 70))
        B.cyl(P.concrete, (x, y, 0.0), (x, y, h), r * 1.3, r, seg=24, far=True, roof=side < 0, name="stack")
        red = ctx.looks.paint((0.55, 0.08, 0.05), rough=0.6, name="stack_red")
        white = ctx.looks.paint((0.8, 0.8, 0.78), rough=0.6, name="stack_white")
        for i in range(4):
            z = h - 2.5 * (i + 1)
            B.cyl(red if i % 2 == 0 else white, (x, y, z), (x, y, z + 2.5), r + 0.02, seg=24, caps=(False, False),
                  far=True, roof=side < 0, name="stack")
        info["stacks"] += 1
    # the ends of the road: a shed or a tank row across each end
    for end in (-1, 1):
        xe = (zx0 - 10.0) if end < 0 else (zx1 + 10.0)
        yc = (zy0 + zy1) / 2
        if rng.random() < 0.6:
            L = float(zy1 - zy0 + 40)
            T = U.Xf(B, (xe, yc + L / 2 * (1 if end > 0 else -1)), -math.pi / 2 if end > 0 else math.pi / 2)
            U.gable_shed(T, P.clad2, P.roof, P.steel2, 0.0, L, 0.0, float(rng.uniform(15, 30)), 1,
                         float(rng.uniform(7, 14)), pitch=10, plinth=P.block, door=P.shutter, glass=P.glass, bay=6.5,
                         doors=[(float(L / 2 + rng.uniform(-8, 8)), 5.0, 5.5)], gutter=P.galv, far=True)
        else:
            y = zy0 - 10
            while y < zy1 + 10:
                r = float(rng.uniform(4, 9))
                U.tank(B, P.tank_shell, P.tank_shell, P.rail, (xe + end * r, y + r), r, float(rng.uniform(8, 16)),
                       kind="cone", stair_mat=P.grating, far=True, seg=48)
                y += 2 * r + float(rng.uniform(2, 5))
    return info


def _decals(ctx, P, fam, weather):
    rng = ctx.rng
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    oil = U.decal_material("oil_stain", rgb=(0.015, 0.013, 0.012), rough=0.15, opacity=float(rng.uniform(0.5, 0.85)))
    n_oil = int(rng.integers(8, 25))
    for _ in range(n_oil):
        c = ctx.uniform_xy(ctx.zone, pad=1.0)
        r = float(rng.uniform(0.3, 1.6))
        U.decal(ctx, oil, c, r, r * float(rng.uniform(0.4, 1.0)), float(rng.uniform(0, math.pi)), z=0.005,
                name="oil")
    if weather != "snow":
        rust = U.decal_material("rust_stain", rgb=(0.12, 0.07, 0.04), rough=0.8, opacity=0.35, scale=2.0)
        for _ in range(int(rng.integers(0, 5))):
            c = ctx.uniform_xy(ctx.zone, pad=1.0)
            r = float(rng.uniform(0.4, 1.4))
            U.decal(ctx, rust, c, r, r * 0.7, float(rng.uniform(0, 3.1)), z=0.006, name="rust")
    # painted lane edges and walkways (worn)
    if weather != "snow":
        yellow = U.worn_paint_material("line_yellow", (0.75, 0.55, 0.05), wear=float(rng.uniform(0.1, 0.8)))
        white = U.worn_paint_material("line_white", (0.8, 0.8, 0.78), wear=float(rng.uniform(0.1, 0.8)))
        for y in (y0 + 0.3, y1 - 0.3):
            ctx.strip((zx0, y), (zx1, y), 0.12, yellow, z=0.007, name="line")
        if rng.random() < 0.5:
            ym = (y0 + y1) / 2 + float(rng.uniform(-2, 2))
            for xs in np.arange(zx0, zx1, 6.0):
                ctx.strip((xs, ym), (xs + 3.0, ym), 0.12, white, z=0.007, name="line")
        # hatched no-parking box somewhere
        if rng.random() < 0.6:
            c = ctx.uniform_xy(pad=2.0)
            for t in np.arange(-2.0, 2.0, 0.6):
                ctx.strip((c[0] + t - 0.5, c[1] - 0.8), (c[0] + t + 0.5, c[1] + 0.8), 0.15, yellow, z=0.008,
                          name="hatch")
    # concrete saw-cut joints
    if fam == "concrete" and weather != "snow":
        joint = ctx.looks.paint((0.05, 0.05, 0.05), rough=0.9, name="joint")
        sp = float(rng.uniform(4.0, 6.0))
        for xs in np.arange(math.floor(zx0 / sp) * sp, zx1, sp):
            ctx.strip((xs, zy0), (xs, zy1), 0.012, joint, z=0.002, name="joint")
        for ys in np.arange(math.floor(zy0 / sp) * sp, zy1, sp):
            ctx.strip((zx0, ys), (zx1, ys), 0.012, joint, z=0.002, name="joint")
    # manholes and a drain channel
    mh = _pieces(ctx, ["ph:water_manhole_cover#0"])
    if mh:
        for _ in range(int(rng.integers(1, 4))):
            c = ctx.uniform_xy(ctx.zone, pad=1.0)
            if ctx.free(c, 0.4, 0.1):
                ctx.instance(mh[0], c, -0.085, float(rng.uniform(0, 6.28)), 1.0, coll="ground", collide=False)
    if rng.random() < 0.6:
        grate = ctx.looks.pbr(U.pick(ctx, ["acg:Grate001", "acg:Grate002"]), 0.65, alpha=False)
        dark = ctx.looks.paint((0.02, 0.02, 0.02), rough=0.7, name="drain")
        y = float(rng.choice([y0 + 1.0, y1 - 1.0]))
        ctx.strip((zx0, y), (zx1, y), 0.3, dark, z=0.003, name="drain")
        ctx.strip((zx0, y), (zx1, y), 0.26, grate, z=0.009, name="drain")


def build(ctx) -> dict:
    rng = ctx.rng
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    weather = str(rng.choice(["dry", "dry", "dry", "dry", "wet", "wet", "snow"]))
    age = float(rng.uniform(0.0, 1.0))
    style = str(rng.choice(["pipe_alley", "tank_farm", "factory_yard", "process"]))
    sky = _sky(ctx, weather)
    fam, gkeys, wet = _ground(ctx, weather, age)
    B = U.Batch(ctx)
    P = Plant(ctx, B, weather, age)
    dressed = _dress(ctx, P, B)
    core = U.reserve_core(ctx)
    lvl_hang = float(ctx.knobs.get("hang", 0.0))
    info = {"sky": sky, "weather": weather, "age": round(age, 2), "style": style, "ground": gkeys, "wet": wet,
            "steel": P.steel_name, "dress": dressed, "core_reserved": core, **SEMANTICS}

    # --- the two long sides (the -y side stays low near the patch: the oblique preview looks over it)
    xa, xb = zx0 - 2.0, zx1 + 2.0
    sides = {"pipe_alley": ("rack", "rack"), "tank_farm": ("rack", "tanks"), "factory_yard": ("rack", "shed"),
             "process": ("rack", "frame")}[style]
    if rng.random() < 0.3:
        sides = (sides[0], "rack") if sides[1] != "rack" else ("fence", "rack")
    fronts = {}
    boxes0 = U.prim_boxes(ctx.spec)
    walled = U.walled(ctx)          # an arena's perimeter walls: nothing to build between them and the zone edge
    info["walled"] = walled
    for side, kind in ([] if walled else zip((-1, 1), sides)):
        roof = side < 0
        if kind == "rack":
            fronts[side] = ("rack", _side_rack(ctx, P, side, xa, xb, roof))
        elif kind == "tanks":
            fronts[side] = ("tanks", len(_side_tanks(ctx, P, B, side, xa, xb, roof)))
        elif kind == "shed":
            fronts[side] = ("shed", len(_side_shed(ctx, P, B, side, xa, xb, roof)))
        elif kind == "frame":
            yb = (y1 + 2.0) if side > 0 else (y0 - 2.0 - 6.0)
            nx = int(rng.integers(2, 4))
            xs = float(rng.uniform(x0, max(x0 + 0.1, x1 - nx * 6.0)))
            ok = all(ctx.free((xs + i * 6.0, yb + j * 6.0), 0.3, 0.3) for i in range(nx + 1) for j in range(2)
                     if ctx.in_zone((xs + i * 6.0, yb + j * 6.0)))
            ok = ok and U.box_clear(boxes0, (xs - 1.8, yb - 0.5, 0.0), (xs + nx * 6.0 + 0.5, yb + 6.5, 20.0), pad=0.3)
            if ok:
                T = U.Xf(B, (xs, yb), 0.0)
                P.process_frame(T, nx, 1, 6.0, levels=(5.0, 9.5, 14.0), roof=roof)
                for i in range(nx + 1):
                    for j in range(2):
                        ctx.claim_disc((xs + i * 6.0, yb + j * 6.0), 0.5, kind="frame")
            fronts[side] = ("frame", ok)
            _side_rack(ctx, P, side, xa, xb, roof)
        else:
            fm = ctx.looks.pbr(U.pick(ctx, ["acg:Fence001", "acg:Fence002", "acg:Fence005", "acg:Fence006"]), 1.0)
            yf = (y0 - 1.5) if side < 0 else (y1 + 1.5)
            U.fence_line(ctx, B, (xa, yf), (xb, yf), float(rng.uniform(2.0, 2.6)), fm, P.galv, barbed=rng.random() < 0.5)
            fronts[side] = ("fence", True)
    info["fronts"] = {str(k): v[0] for k, v in fronts.items()}

    # --- pipe bridges across the road
    boxes = U.prim_boxes(ctx.spec)
    n_br = int(rng.integers(1, 3)) if (x1 - x0) < 30 else int(rng.integers(1, 4))
    lvl = ctx.level
    # L1: wholly above the cloud top and the drone's reach (lowest member zb - 0.3); L2/L3 in the upper band
    zb_rng = {1: (U.ZSPLIT + 0.35, U.ZSPLIT + 1.0), 2: (3.7, 4.4), 3: (3.0, 3.6)}.get(
        int(lvl) if str(lvl).isdigit() else 2, (3.7, 4.4))
    bridges = 0
    for _ in range(n_br * 6):
        if bridges >= n_br:
            break
        xc = float(rng.uniform(x0 + 1.0, x1 - 1.0))
        if min(abs(xc - ctx.start[0]), abs(xc - ctx.goal[0])) < 3.0:
            continue
        width = float(rng.uniform(1.8, 2.8))
        zb = float(rng.uniform(*zb_rng))
        ya, yb = y0 - 1.6, y1 + 1.6
        if not U.box_clear(boxes, (xc - width / 2 - 0.4, ya, zb - 0.6), (xc + width / 2 + 0.4, yb, zb + 2.0), pad=0.3):
            continue
        if zb < U.ZSPLIT and not U.rect_clear(ctx, (xc, (y0 + y1) / 2), (width / 2, (y1 - y0) / 2), margin=0.3,
                                              kinds=("core",)):
            continue            # an overhead bridge never crosses the family's own space
        if not all(ctx.free((xc + s * width / 2, y), 0.3, 0.3) for s in (-1, 1) for y in (ya, yb)):
            continue
        P.bridge(xc, ya, yb, zb, width=width, hang=ctx.count(lvl_hang * 0.6) if lvl_hang > 0 else 0)
        bridges += 1
    info["bridges"] = bridges

    # --- the yard: equipment and props in the patch
    ko = float(ctx.knobs.get("keepout", 2.5))
    n_eq = ctx.count(9.0)
    made = {"ibc": 0, "drums": 0, "gas": 0, "cabinets": 0, "pump": 0, "reel": 0, "spools": 0, "container": 0,
            "skip": 0, "pallets": 0, "bigbags": 0, "beams": 0, "trafo": 0, "forklift": 0}
    kinds = list(made)
    for _ in range(n_eq * 30):
        if sum(made.values()) >= n_eq:
            break
        k = kinds[int(rng.integers(len(kinds)))]
        yaw = float(rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2])) + float(rng.normal(0, 0.08))
        half = {"ibc": (0.65, 0.55), "drums": (0.65, 0.45), "gas": (1.1, 0.55), "cabinets": (1.4, 0.5),
                "pump": (1.3, 0.6), "reel": (1.0, 0.5), "spools": (2.8, 0.8), "container": (3.2, 1.35),
                "skip": (1.9, 0.95), "pallets": (0.65, 0.45), "bigbags": (1.9, 0.55), "beams": (3.6, 0.95),
                "trafo": (1.25, 1.0), "forklift": (2.0, 0.7)}[k]
        xy, yaw = _anchor(ctx, half, yaw)
        if not ctx.free_rect(xy, half, yaw, margin=0.5 * ko + 0.3):
            continue
        {"ibc": P.ibc, "drums": P.drums, "gas": P.gas_cage, "cabinets": lambda c, y: P.cabinets(c, y, int(rng.integers(2, 5))),
         "pump": P.pump_skid, "reel": P.reel, "spools": P.spools, "container": P.container, "skip": P.skip,
         "pallets": P.pallet_stack, "bigbags": P.bigbags, "beams": P.beams, "trafo": P.transformer,
         "forklift": P.forklift}[k](xy, yaw)
        if k in ("ibc", "drums"):
            ctx.claim_rect(xy, half, yaw, kind=k)
        made[k] += 1
        if k in ("ibc", "drums", "pallets", "bigbags", "gas", "skip") and rng.random() < 0.7:
            # a row of the same thing next to it
            d = np.array([math.cos(yaw), math.sin(yaw)])
            step = 2 * half[0] + float(rng.uniform(0.1, 0.4))
            for m in range(int(rng.integers(1, 4))):
                q = xy + d * step * (m + 1) * (1 if rng.random() < 0.8 else -1)
                if not ctx.free_rect(q, half, yaw, margin=0.5 * ko + 0.2):
                    break
                {"ibc": P.ibc, "drums": P.drums, "pallets": P.pallet_stack, "bigbags": P.bigbags,
                 "gas": P.gas_cage, "skip": P.skip}[k](q, yaw + float(rng.normal(0, 0.03)))
                if k in ("ibc", "drums"):
                    ctx.claim_rect(q, half, yaw, kind=k)
    info["equipment"] = made
    _obj_yard(ctx, ko)
    props = _pieces(ctx, PROPS)
    ne = ctx.count(10.0)
    edge = 2.2
    ctx.scatter(props, int(ne * 0.7), margin=0.5,
                where=lambda q: min(q[1] - y0, y1 - q[1]) < edge or _near_prim(ctx, q, 1.6))
    ctx.scatter(props, ne - int(ne * 0.7), margin=0.6)
    for _ in range(ctx.count(3.0)):
        xy = ctx.uniform_xy(pad=0.8)
        if ctx.free(xy, 0.15, 0.6):
            P.bollard(xy)
    # low pipe runs on sleepers (L2+)
    if ctx.knobs.get("clutter", 1.0) >= 1.0 and rng.random() < 0.6:
        for _ in range(20):
            L = float(rng.uniform(6, 14))
            yaw = 0.0 if rng.random() < 0.7 else math.pi / 2
            c = ctx.uniform_xy(pad=2.0)
            d = np.array([math.cos(yaw), math.sin(yaw)])
            if ctx.free_rect(c, (L / 2, 0.9), yaw, margin=0.5 * ko + 0.4):
                P.sleeper_run(c - d * L / 2, c + d * L / 2, n=int(rng.integers(2, 4)))
                info["sleepers"] = True
                break
    # overhead pipe runs on T-supports across the flight band (L3)
    n_hang = ctx.count(lvl_hang) if lvl_hang > 0 else 0
    tees = 0
    for _ in range(n_hang * 30):
        if tees >= n_hang:
            break
        p = ctx.uniform_xy(pad=0.8)
        ang = float(rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2])) + float(rng.normal(0, 0.03))
        L = float(rng.uniform(4.0, 9.0))
        q = p + L * np.array([math.cos(ang), math.sin(ang)])
        if not ctx.in_zone(q) or not (ctx.free(p, 0.2, 0.6) and ctx.free(q, 0.2, 0.6)):
            continue
        z = float(rng.uniform(2.2, 3.3))
        lo = np.array([min(p[0], q[0]), min(p[1], q[1]), z - 0.3])
        hi = np.array([max(p[0], q[0]), max(p[1], q[1]), z + 0.4])
        if not U.box_clear(boxes, lo, hi, pad=0.4):
            continue
        if min(U.seg_dist(ctx.start, p, q), U.seg_dist(ctx.goal, p, q)) < 2.2:
            continue
        if not U.seg_clear(ctx, p, q, margin=0.6):     # not across a slot's approach or the family's core
            continue
        P.tee_pipe_run(p, q, z, n=int(rng.integers(1, 4)))
        tees += 1
    info["hang_runs"] = tees

    lamps = _pieces(ctx, ["ph:street_lamp_01#0"])
    if lamps and not walled and rng.random() < 0.6:          # yard lamps just outside the patch edge
        side = -1 if rng.random() < 0.5 else 1
        y = (y0 - 0.7) if side < 0 else (y1 + 0.7)
        for x in np.arange(x0 + float(rng.uniform(2, 8)), x1, float(rng.uniform(10, 16))):
            if ctx.free((x, y), 0.5, 0.2):
                ctx.instance(lamps[0], (x, y), 0.0, math.pi / 2 if side > 0 else -math.pi / 2, 1.0, coll="scatter",
                             claim=True)
    cars = _pieces(ctx, ["ph:covered_car#0"])
    if cars and not walled and rng.random() < 0.6:
        zx0_, zy0_, zx1_, zy1_ = ctx.zone
        for _ in range(int(rng.integers(1, 4))):
            side = -1 if rng.random() < 0.5 else 1
            xy = np.array([float(rng.uniform(zx0_ + 3, zx1_ - 3)),
                           (y0 - float(rng.uniform(1.8, 3.5))) if side < 0 else (y1 + float(rng.uniform(1.8, 3.5)))])
            if ctx.free(xy, 2.3, 0.3):
                ctx.instance(cars[0], xy, 0.0, math.pi / 2 + float(rng.normal(0, 0.1)), 1.0, coll="scatter", claim=True)
    veh = U.obj_keys(ctx, "truck", "van", "pickup")
    if veh and not walled and rng.random() < 0.75:     # vehicles parked along the plant road, outside the patch
        for _ in range(int(rng.integers(1, 4))):
            k = veh[int(rng.integers(len(veh)))]
            lh, sh = U.long_half(ctx, k)
            side = -1 if rng.random() < 0.5 else 1
            gap = float(rng.uniform(0.6, 2.2))
            xy = (float(rng.uniform(zx0 + lh + 1.0, max(zx0 + lh + 1.1, zx1 - lh - 1.0))),
                  (y0 - sh - gap) if side < 0 else (y1 + sh + gap))
            U.place_obj(ctx, k, xy, U.along_yaw(ctx, k, 0.0) + float(rng.normal(0, 0.04)), margin=0.4)
    info.update(_backdrop(ctx, P, B, weather))
    _decals(ctx, P, fam, weather)
    B.flush()
    info["objaverse"] = U.obj_record(ctx)
    return info
