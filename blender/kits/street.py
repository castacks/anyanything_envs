"""City street: a canyon of building fronts on both long sides of the patch
(a square with buildings all round on arena-shaped patches), pavements with
kerbs, an asphalt carriageway with worn markings, parked cars, street lamps,
trees in pits, bins, benches, bollards, signs, hydrants -- the patch's
obstacles are things on the street (hoardings, posts, scaffold tubes,
concrete ball bollards).

What makes it read as a real street rather than boxes on a plane:

* buildings are closed volumes whose fronts have real window openings
  (reveals, frames, glass, sills, lintels), shopfronts with fascia signs
  carrying real lettering, roller shutters, awnings, balconies, cornices,
  downpipes; a third of them are Poly Haven's modular brick factory /
  apartment facades, recoloured per building;
* a street has a *character* -- downtown, residential, old town,
  industrial or pedestrian -- that sets the building heights, the ground
  floors, the palette (brick / plaster / stone / concrete), the road
  surface (asphalt, patched asphalt, cobbles, paving) and the furniture;
* the street runs 80 m past the patch both ways with render-only buildings,
  taller blocks behind the fronts (skyscraper facade scans), side streets
  opening off it, so every view ends in city, not in the HDRI's ground;
* weather: dry, wet (overcast sky + puddles), dusk (low sun, lit lamps and
  signs) or snow;
* real things on the street (curated Objaverse models): a fleet of parked
  cars and vans, bikes and motorbikes, phone booths, mailboxes, parking
  meters, vending machines, a bus shelter, hydrants, bins, traffic signals
  at the zebra crossing, plane trees.

Render budget: street trees and a stand's crowns share triangle budgets
(``TREE_TRIS``, ``CANOPY_TRIS``) -- every instanced copy is rendered.

Levels: ``clutter`` scales the pavement furniture and parked cars;
``hang`` > 0 strings cables, festoon lights and span wires across the
street at 2.6-4.5 m, and puts up utility poles with service drops.
"""

from __future__ import annotations

import math

import numpy as np

import build_course as BC
from kits import _street_util as U

NAME = "street"
INDOOR = False
DESCRIPTION = "city street: building canyon with shopfronts, pavements, parked cars, lamps, trees, markings"

#: render triangles all street-tree copies may add (Poly Haven trees are 0.3-2 M each)
TREE_TRIS = 4.5e6
#: ... and the crowns on a stand's trunks (searsia is 0.36 M a copy; a stand has 30-80 trunks)
CANOPY_TRIS = 3.0e6

CHARACTERS = ["downtown", "residential", "oldtown", "industrial", "pedestrian"]
URBAN_TAGS = {"street", "city", "sidewalk", "building", "town", "square", "alley", "apartments", "cobblestone",
              "old town", "intersection", "metropolis", "district", "suburb", "shop"}


def pick_sky(ctx, weather, tags=URBAN_TAGS, target_ev=0.0):
    lib, rng = ctx.lib, ctx.rng
    none = {"night", "indoor"}
    if weather == "wet":
        keys = lib.hdris(cats_all={"outdoor", "overcast"}, cats_none=none, tags_any=tags)
    elif weather == "dusk":
        keys = lib.hdris(cats_all={"outdoor", "sunrise-sunset"}, cats_none=none, tags_any=tags)
    elif weather == "snow":        # a white winter day: urban snow skies or any urban overcast one
        keys = lib.hdris(cats_all={"outdoor", "urban"}, cats_none=none, tags_any={"snow", "winter"}) + \
            lib.hdris(cats_all={"outdoor", "urban", "overcast"}, cats_none=none, tags_any=tags)
    else:
        keys = lib.hdris(cats_all={"outdoor", "urban"}, cats_none=none | {"overcast"}, tags_any=tags)
    if not keys:
        keys = lib.hdris(cats_all={"outdoor", "urban"}, cats_none=none)
    return ctx.looks.sky(lib.pick(rng, keys), target_ev=target_ev)


class Street:
    def __init__(self, ctx):
        self.ctx = ctx
        rng = ctx.rng
        x0, y0, x1, y1 = ctx.bounds
        self.W, self.D = x1 - x0, y1 - y0
        self.plaza = self.D > 0.75 * self.W
        self.char = "pedestrian" if self.plaza else str(rng.choice(CHARACTERS, p=[0.26, 0.26, 0.2, 0.14, 0.14]))
        self.weather = str(rng.choice(["dry", "wet", "dusk", "snow"], p=[0.6, 0.2, 0.13, 0.07]))
        self.lit = self.weather == "dusk" or (self.weather == "wet" and rng.random() < 0.4)
        ctx._lit_shops = self.lit
        look = {"downtown": ["mixed", "modern", "brick", "stone"], "residential": ["brick", "plaster", "mixed"],
                "oldtown": ["plaster", "stone", "worn"], "industrial": ["brick", "modern", "worn"],
                "pedestrian": ["mixed", "plaster", "stone", "brick"]}[self.char]
        self.pal = U.Palette(ctx, look=str(rng.choice(look)), grime=float(rng.uniform(0.25, 0.6)))
        zx0, zy0, zx1, zy1 = ctx.zone
        self.XA, self.XB = zx0 - 80.0, zx1 + 80.0
        self.kh = float(rng.uniform(0.11, 0.15))           # kerb height
        self.floors = {"downtown": (3, 8), "residential": (1, 4), "oldtown": (2, 5), "industrial": (1, 4),
                       "pedestrian": (2, 6)}[self.char]
        self.info = {"character": self.char, "weather": self.weather, "plaza": self.plaza,
                     "look": self.pal.look}

    # --- ground --------------------------------------------------------------------------------
    def ground(self):
        ctx, rng, lk = self.ctx, self.ctx.rng, self.ctx.looks
        puddles = {"dry": 0.0, "wet": float(rng.uniform(0.5, 1.0)), "dusk": 0.0, "snow": 0.2}[self.weather]
        # one paving pattern (two noise-mixed patterns read as a patchwork quilt, not a square)
        if self.char == "pedestrian" or self.plaza:
            keys = [U.pick_key(ctx, U.PLAZA)]
            tile = 2.0
        elif self.char == "oldtown" and rng.random() < 0.6:
            keys = [U.pick_key(ctx, U.COBBLE)]
            tile = 1.6
        else:
            keys = [U.pick_key(ctx, U.ASPHALT), U.pick_key(ctx, U.ASPHALT_PATCHED if rng.random() < 0.5 else U.ASPHALT)]
            tile = 4.0
        if self.weather == "snow":
            keys = [keys[0], "ph:asphalt_snow", "ph:snow_02"]
        m = lk.ground_material(keys, tile_m=tile, puddles=puddles, blend_scale=float(rng.uniform(8, 16)))
        ctx.ground(m)
        self.info["ground"] = keys
        self.info["puddles"] = puddles

    # --- the obstacles as street things -----------------------------------------------------------
    def dress(self):
        ctx, rng, lk, pal = self.ctx, self.ctx.rng, self.ctx.looks, self.pal
        wall_kind = str(rng.choice(["hoarding", "hoarding", "concrete", "brick", "corrugated"]))
        if wall_kind == "hoarding":
            wall = lk.pbr(U.pick_key(ctx, ["ph:plywood", "ph:blue_painted_planks", "ph:distressed_painted_planks",
                                           "acg:PaintedWood005", "acg:PaintedWood006A", "acg:PaintedWood007A",
                                           "acg:PaintedWood009A", "acg:Plywood001"]), 2.2, bevel=0.01, dirt=0.35,
                          tint=U.PASTELS[int(rng.integers(len(U.PASTELS)))] if rng.random() < 0.4 else None)
        elif wall_kind == "concrete":
            wall = lk.pbr(U.pick_key(ctx, U.CONCRETE_WALL), 3.0, bevel=0.02, dirt=0.5)
        elif wall_kind == "brick":
            wall = lk.pbr(U.pick_key(ctx, U.BRICK_RED + U.BRICK_DARK), 1.5, bevel=0.015, dirt=0.4)
        else:
            wall = lk.pbr(U.pick_key(ctx, U.CORRUGATED), 1.5, bevel=0.01, dirt=0.4,
                          tint=[(1.0, 1.0, 1.0), (0.75, 0.95, 0.8), (0.75, 0.85, 1.0), (1.0, 0.8, 0.7)][
                              int(rng.integers(4))])
        box = lk.pbr(U.pick_key(ctx, U.CONCRETE_WALL if rng.random() < 0.5 else U.PAINTED_METAL), 2.0,
                     bevel=0.02, dirt=0.4)
        pole = lk.pbr(U.pick_key(ctx, U.PAINTED_METAL), 1.0, bevel=0.01, dirt=0.3,
                      tint=[(0.12, 0.12, 0.12), (0.08, 0.18, 0.12), (0.35, 0.35, 0.36), (0.1, 0.12, 0.2)][
                          int(rng.integers(4))])
        bar = lk.pbr(U.pick_key(ctx, U.GALV), 1.0, bevel=0.005, dirt=0.25)
        rock = lk.pbr(U.pick_key(ctx, U.TRIM + U.KERB), 1.5, bevel=0.01, dirt=0.35)
        ctx.dress_primitives({"wall": wall, "box": box, "pole": pole, "bar": bar, "rock": rock})
        # thick poles are masonry columns, not lamp posts
        column = lk.pbr(U.pick_key(ctx, U.STONE + U.CONCRETE_WALL), 2.0, bevel=0.02, dirt=0.4)
        n_heads = 0
        for o in ctx.colls["obstacles"].objects:
            if "_cyl" in o.name and max(o.dimensions[0], o.dimensions[1]) > 1.1:
                o.data.materials[0] = column
        # the patch's posts become street lamps, bollard posts or pollarded plane trees
        poles = [p for p in ctx.spec["primitives"] if p["kind"] == "cyl_v" and p["r"] <= 0.6]
        many = len(poles) >= 4
        styles, pr = (["trunk", "post", "lamp"], [0.45, 0.35, 0.2]) if many else (["lamp", "post", "trunk"],
                                                                                 [0.5, 0.3, 0.2])
        pole_style = str(rng.choice(styles, p=pr))
        top = BC.CLOUD_ZMAX + 0.05                  # crowns stand wholly above the cloud (render-only)
        tall = [p for p in poles if p["z1"] >= top]
        if pole_style == "trunk" and not tall:
            pole_style = "post"
        canopies, crown_s = ctx.lib.models(ids=["searsia_burchellii"]), (0.8, 1.1)
        c_tris = max((q.get("tris", 0) for q in ctx.lib.pieces("ph:searsia_burchellii")), default=0)
        if len(tall) * c_tris > CANOPY_TRIS:        # a stand of 30-80 trunks: a light pollarded crown instead
            canopies, crown_s = U.obj_pieces(ctx, U.OBJ_CROWN), (2.0, 2.8)
        if pole_style == "trunk" and not canopies:
            pole_style = "post"
        if pole_style == "trunk":
            bark = lk.pbr(U.pick_key(ctx, ["ph:bark_platanus", "ph:japanese_sycamore", "ph:japanese_zelkova_bark",
                                           "ph:tree_bark_03", "ph:bark_brown_01", "acg:Bark007", "acg:Bark012"]),
                          1.2, dirt=0.2, jitter=0.1)
            pit = lk.pbr(U.pick_key(ctx, ["acg:Grate001", "ph:metal_grate_rusty"]), 1.2)
            names = {f"prim{i:02d}_cyl" for i, p in enumerate(ctx.spec["primitives"])
                     if p["kind"] == "cyl_v" and p["r"] <= 0.6}
            for o in ctx.colls["obstacles"].objects:
                if o.name in names:
                    o.data.materials[0] = bark
            for p in poles:
                cx, cy = p["c"]
                if p["z1"] >= top:
                    zc = max(p["z1"] - 0.2, top)
                    ctx.instance(ctx.lib.pick(rng, canopies), (cx, cy), z=zc, yaw=float(rng.uniform(0, 6.28)),
                                 s=float(rng.uniform(*crown_s)), collide=False, coll="scatter")
                h = p["r"] + 0.5
                ctx.flat([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)], pit, z=0.004,
                         name="treepit")
        for p in ctx.spec["primitives"]:
            if p["kind"] != "cyl_v" or p["r"] > 0.6 or pole_style == "trunk":
                continue
            cx, cy = p["c"]
            if pole_style == "lamp" and p["r"] <= 0.45:
                z = p["z1"]
                ctx.box((cx, cy, z + 0.06), (p["r"] + 0.25, p["r"] + 0.08, 0.06), pal.metal, name="pole_head")
                lens = lk.paint((1.0, 0.95, 0.85), rough=0.3, name="lamp_lens",
                                emission=(1.0, 0.85, 0.6, 8.0) if self.lit else None)
                ctx.box((cx, cy, z - 0.005), (p["r"] + 0.18, p["r"] + 0.04, 0.006), lens, name="pole_head")
            else:
                band = lk.paint((0.85, 0.85, 0.8), rough=0.3, name="band")
                ctx.cylinder((cx, cy, 1.6), (cx, cy, 1.75), p["r"] + 0.004, band, name="pole_band")
        self.info["poles_as"] = pole_style
        self.cap_walls(wall, wall_kind)
        if wall_kind in ("hoarding", "corrugated", "concrete") and rng.random() < 0.75:
            self.info["posters"] = U.posters(ctx, self.pal)
        self.info["walls_as"] = wall_kind

    def cap_walls(self, wall, kind):
        ctx, lk = self.ctx, self.ctx.looks
        cap = lk.pbr(U.pick_key(ctx, U.TRIM), 1.5, bevel=0.01, dirt=0.3) if kind in ("brick", "concrete") else \
            self.pal.metal
        for p in ctx.spec["primitives"]:
            if p["kind"] != "wall":
                continue
            p0, p1 = np.asarray(p["p0"], float), np.asarray(p["p1"], float)
            L = float(np.linalg.norm(p1 - p0))
            d = (p1 - p0) / L
            yaw = math.atan2(d[1], d[0])
            t = float(p["t"])
            # the wall's top runs between openings that reach the top
            cuts = sorted([(s - w / 2, s + w / 2) for s, w, zb, zt in p.get("slots", []) if zt >= p["z1"] - 1e-3])
            segs, a = [], 0.0
            for ca, cb in cuts:
                if ca > a:
                    segs.append((a, ca))
                a = max(a, cb)
            if a < L:
                segs.append((a, L))
            for sa, sb in segs:
                c = p0 + d * (sa + sb) / 2
                ctx.box((c[0], c[1], p["z1"] + 0.03), ((sb - sa) / 2, t / 2 + 0.03, 0.03), cap, yaw=yaw,
                        name="coping")
            # extend a wall that ends at the patch edge to the building line (it closes the street)
            x0_, y0_, x1_, y1_ = ctx.bounds
            for end in (p0, p1):
                yfac = self.facade_y(end[1])
                if yfac is None or abs(d[0]) > 0.3:
                    continue
                edge = y1_ if end[1] > 0 else y0_
                if abs(end[1] - edge) > 0.6 and not (end[1] > y1_ or end[1] < y0_):
                    continue
                if (end[1] > 0 and end[1] < yfac) or (end[1] < 0 and end[1] > yfac):
                    e2 = np.array([end[0], yfac + (0.3 if yfac > 0 else -0.3)])
                    ctx.panel(end, e2, p["z0"], p["z1"], wall, thickness=t, name="hoarding_ext")
                    self.n_ext = getattr(self, "n_ext", 0) + 1

    def facade_y(self, y):
        if self.plaza:
            return None
        return self.y_fn if y > 0 else self.y_fs

    # --- layout ------------------------------------------------------------------------------------
    def layout(self):
        ctx, rng = self.ctx, self.ctx.rng
        x0, y0, x1, y1 = ctx.bounds
        self.y_fn = y1 + float(rng.uniform(0.3, 2.2))
        self.y_fs = y0 - float(rng.uniform(0.3, 2.2))
        span = self.y_fn - self.y_fs
        self.kerbs = self.char != "pedestrian" and not self.plaza
        if self.kerbs:
            lo, hi = (2.0, 3.5) if self.char == "oldtown" else (2.4, 4.8)
            sw_n, sw_s = float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi))
            if span - sw_n - sw_s < 6.5:
                f = (span - 6.5) / (sw_n + sw_s)
                sw_n, sw_s = sw_n * f, sw_s * f
            self.y_kn, self.y_ks = self.y_fn - sw_n, self.y_fs + sw_s
            road = self.y_kn - self.y_ks
            self.park_n = road >= 8.5 and rng.random() < 0.8
            self.park_s = road >= 11.0 and rng.random() < 0.8
        else:
            self.y_kn, self.y_ks = self.y_fn, self.y_fs
            self.park_n = self.park_s = False
        # side streets opening off the street inside the zone (at most one per side)
        zx0, _, zx1, _ = ctx.zone
        self.gaps = {1: [], -1: []}
        for side in (1, -1):
            if rng.random() < 0.35 and not self.plaza:
                w = float(rng.uniform(8.0, 12.0))
                xc = float(rng.uniform(zx0 + w, zx1 - w)) if zx1 - zx0 > 3 * w else (zx0 + zx1) / 2
                self.gaps[side].append((xc - w / 2, xc + w / 2))
        self.info.update({"y_facades": [round(self.y_fs, 2), round(self.y_fn, 2)],
                          "y_kerbs": [round(self.y_ks, 2), round(self.y_kn, 2)] if self.kerbs else None,
                          "side_streets": {k: v for k, v in self.gaps.items() if v}})

    # --- buildings ------------------------------------------------------------------------------
    def frontage(self, fr, u_lo, u_hi, gaps_u=(), depth=(9.0, 16.0), roof=False):
        """Buildings along frame ``fr`` from u_lo to u_hi, skipping gaps."""
        ctx, rng, pal = self.ctx, self.ctx.rng, self.pal
        zx0, zy0, zx1, zy1 = ctx.zone
        u = u_lo
        n = 0
        mod_p = {"downtown": 0.2, "residential": 0.3, "oldtown": 0.15, "industrial": 0.55, "pedestrian": 0.25}[
            self.char]
        while u < u_hi - 2.0:
            g = next((g for g in gaps_u if g[0] - 0.5 <= u < g[1]), None)
            if g is not None:
                u = g[1]
                continue
            nxt = min([g[0] for g in gaps_u if g[0] > u] + [u_hi])
            w = float(rng.uniform(6.0, 16.0)) if self.char != "industrial" else float(rng.uniform(10.0, 28.0))
            w = min(w, nxt - u)
            if nxt - u - w < 4.0:
                w = nxt - u
            if w < 3.0:
                u = nxt
                continue
            mid = fr.xy(u + w / 2)
            near = ctx.in_zone(mid, pad=25.0)
            far = not ctx.in_zone(mid, pad=55.0)
            dep = float(rng.uniform(*depth))
            fl = int(rng.integers(self.floors[0], self.floors[1] + 1))
            v0 = float(rng.uniform(-0.3, 0.05))
            if rng.random() < mod_p and w >= 6.0 and not far:
                kind = "factory" if (self.char == "industrial" or rng.random() < 0.5) else "apts"
                nm = int(w // 3)
                off = (w - 3 * nm) / 2
                U.modular_building(ctx, pal, fr, u + off, nm, max(1, min(fl + 1, 6)), kind=kind, depth=dep, v0=v0,
                                   roof=roof)
                if off > 0.05:                  # fill the slack with plain party walls
                    for a, b in ((u, u + off), (u + w - off, u + w)):
                        acc = U.Acc(ctx)
                        fr.box(acc, pal.pick(pal.walls), a, b, v0 - dep, v0 - 0.1, 0.0, 3.0 * min(fl + 1, 6) + 0.7)
                        acc.emit(name="infill", roof_all=roof)
            else:
                ground = {"downtown": "shops", "residential": "res" if rng.random() < 0.7 else "shops",
                          "oldtown": "shops" if rng.random() < 0.6 else "res",
                          "industrial": "garage" if rng.random() < 0.7 else "blank",
                          "pedestrian": "shops"}[self.char]
                pitched = (self.char in ("oldtown", "residential") and rng.random() < 0.55) or rng.random() < 0.1
                b = U.Building(ctx, pal, fr, u, u + w, dep, ground=ground, floors=fl, v0=v0, lit=self.lit,
                               pitched=pitched, far=far, roof=roof)
                if not near:
                    b.balconies = 0.0
                b.check = self._clear_overhang
                b.build()
            n += 1
            u += w
        return n

    def _clear_overhang(self, pts):
        return all(self.ctx.free(p, 0.0, 0.25) for p in pts)

    def buildings(self):
        ctx, rng, pal = self.ctx, self.ctx.rng, self.pal
        XA, XB = self.XA, self.XB
        x0, y0, x1, y1 = ctx.bounds
        if not self.plaza:
            fn = U.Frame((XA, self.y_fn), (0.0, -1.0))
            gaps_n = [(a - XA, b - XA) for a, b in self.gaps[1]]
            self.frontage(fn, 0.0, XB - XA, gaps_n)
            fs = U.Frame((XB, self.y_fs), (0.0, 1.0))
            gaps_s = [(XB - b, XB - a) for a, b in self.gaps[-1]]
            self.frontage(fs, 0.0, XB - XA, gaps_s, roof=True)
            # buildings along the side streets (facing into them), past the main rows' depth
            for side, gl in self.gaps.items():
                for a, b in gl:
                    yf = self.y_fn if side > 0 else self.y_fs
                    if side > 0:
                        frs = [U.Frame((a, yf + 16.5), (1.0, 0.0)), U.Frame((b, yf + 46.5), (-1.0, 0.0))]
                    else:
                        frs = [U.Frame((a, yf - 46.5), (1.0, 0.0)), U.Frame((b, yf - 16.5), (-1.0, 0.0))]
                    for fr in frs:
                        self.frontage(fr, 0.0, 30.0, depth=(8.0, 12.0), roof=side < 0)
            # ends of the street: a building across it now and then (a T-junction)
            for xe, nrm in ((XA - 4.0, (1.0, 0.0)), (XB + 4.0, (-1.0, 0.0))):
                if rng.random() < 0.5:
                    o = (xe, self.y_fs - 20.0) if nrm[0] > 0 else (xe, self.y_fn + 20.0)
                    self.frontage(U.Frame(o, nrm), 0.0, (self.y_fn - self.y_fs) + 40.0)
            self.backdrop()
        else:
            s = [float(rng.uniform(0.4, 2.5)) for _ in range(4)]
            s[0] = float(rng.uniform(3.4, 5.0))           # the chase preview stands 3 m west of the patch
            X0, X1, Y0, Y1 = x0 - s[0], x1 + s[1], y0 - s[2], y1 + s[3]
            self.y_fn, self.y_fs = Y1, Y0
            self.plaza_rect, self.plaza_gaps = (X0, Y0, X1, Y1), 0

            def gap(L, fr):
                if rng.random() > 0.6:
                    return []
                g = float(rng.uniform(4.0, max(4.1, L - 16.0)))
                gg = (g, g + float(rng.uniform(8.0, 12.0)))
                self.plaza_gaps += int(ctx.in_zone(fr.xy((gg[0] + gg[1]) / 2)))     # (a way out of the square)
                return [gg]
            for fr, L, roof in ((U.Frame((X0 - 30.0, Y1), (0.0, -1.0)), X1 - X0 + 60.0, False),
                                (U.Frame((X1 + 30.0, Y0), (0.0, 1.0)), X1 - X0 + 60.0, True),
                                (U.Frame((X1, Y1), (-1.0, 0.0)), Y1 - Y0, False),
                                (U.Frame((X0, Y0), (1.0, 0.0)), Y1 - Y0, False)):
                self.frontage(fr, 0.0, L, gap(L, fr), roof=roof)
            self.backdrop(plaza=(X0, Y0, X1, Y1))

    def backdrop(self, plaza=None):
        """Taller blocks behind the fronts (render-only): skyline."""
        ctx, rng, lk = self.ctx, self.ctx.rng, self.ctx.looks
        skins = [lk.pbr(k, 13.0, jitter=0.1, vary=0.1) for k in
                 [U.pick_key(ctx, ["acg:Facade018A", "acg:Facade018B", "acg:Facade019A", "acg:Facade020A",
                                   "acg:Facade020B", "acg:Facade017", "acg:Facade019B"]) for _ in range(3)]]
        skins += [self.pal.pick(self.pal.walls)]
        zx0, zy0, zx1, zy1 = ctx.zone
        if plaza is None:
            rows = [(self.y_fn + 22.0, 1), (self.y_fs - 22.0, -1)]
            x_lo, x_hi = self.XA, self.XB
        else:
            X0, Y0, X1, Y1 = plaza
            rows = [(Y1 + 22.0, 1), (Y0 - 22.0, -1)]
            x_lo, x_hi = X0 - 60, X1 + 60
        tall = {"downtown": (14, 60), "residential": (8, 20), "oldtown": (9, 18), "industrial": (8, 25),
                "pedestrian": (10, 40)}[self.char]
        for yb, side in rows:
            x = x_lo
            while x < x_hi:
                w = float(rng.uniform(12, 30))
                h = float(rng.uniform(*tall))
                dep = float(rng.uniform(12, 25))
                yc = yb + side * (dep / 2 + float(rng.uniform(0, 12)))
                ctx.box((x + w / 2, yc, h / 2), (w / 2, dep / 2, h / 2), skins[int(rng.integers(len(skins)))],
                        coll="backdrop", collide=False, name="block", roof=side < 0)
                x += w + float(rng.uniform(0, 6))

    # --- pavements, kerbs, markings -------------------------------------------------------------
    def pavements(self):
        ctx, rng, lk = self.ctx, self.ctx.rng, self.ctx.looks
        if not self.kerbs:
            return
        walk = lk.pbr(U.pick_key(ctx, U.SIDEWALK), float(rng.uniform(1.6, 2.6)), bevel=0.01, dirt=0.3,
                      vary=0.15, jitter=0.0)
        kerb = lk.pbr(U.pick_key(ctx, U.KERB), 1.2, bevel=0.02, dirt=0.4, jitter=0.02)
        gutter = lk.pbr(U.pick_key(ctx, U.CONCRETE_GROUND), 1.5, dirt=0.4, jitter=0.0)
        snow = lk.pbr("ph:snow_02", 3.0) if self.weather == "snow" and "ph:snow_02" in ctx.lib.items else None
        kh = self.kh
        for side, yk, yf in ((1, self.y_kn, self.y_fn), (-1, self.y_ks, self.y_fs)):
            xs = [self.XA - 10]
            for a, b in sorted(self.gaps[side]):
                xs += [a, b]
            xs.append(self.XB + 10)
            for i in range(0, len(xs), 2):
                a, b = xs[i], xs[i + 1]
                ya, yb_ = sorted((yk + side * 0.16, yf + side * 0.4))
                ctx.box(((a + b) / 2, (ya + yb_) / 2, (kh - 0.01) / 2), ((b - a) / 2, (yb_ - ya) / 2, (kh - 0.01) / 2),
                        walk, name="pavement")
                ka, kb = sorted((yk, yk + side * 0.16))
                ctx.box(((a + b) / 2, (ka + kb) / 2, kh / 2), ((b - a) / 2, (kb - ka) / 2, kh / 2), kerb,
                        name="kerb")
                ga, gb = sorted((yk, yk - side * 0.35))
                ctx.flat([(a, ga), (b, ga), (b, gb), (a, gb)], gutter, z=0.002, name="gutter")
                if snow is not None:       # snow banked against the kerb
                    sa, sb = sorted((yk - side * 0.05, yk - side * 0.6))
                    ctx.flat([(a, sa), (b, sa), (b, sb), (a, sb)], snow, z=0.01, name="snowbank")
                # side-street kerb returns
                for xe in (a, b):
                    if self.XA - 5 < xe < self.XB + 5:
                        ctx.box((xe, (yk + yf) / 2 + side * 0.2, kh / 2), (0.08, abs(yf - yk) / 2 + 0.2, kh / 2),
                                kerb, name="kerb")

    def markings(self):
        ctx, rng = self.ctx, self.ctx.rng
        if not self.kerbs:
            return
        us = rng.random() < 0.3
        white = U.worn_paint(ctx, (0.85, 0.85, 0.82), wear=float(rng.uniform(0.1, 0.6)))
        yellow = U.worn_paint(ctx, (0.8, 0.6, 0.08), wear=float(rng.uniform(0.1, 0.5)), name="line_yellow")
        XA, XB = self.XA, self.XB
        yc = (self.y_kn + self.y_ks) / 2
        if self.park_n:
            yc -= 1.1
        if self.park_s:
            yc += 1.1
        road = self.y_kn - self.y_ks
        if road > 6.0:
            if us:
                for dy in (-0.09, 0.09):
                    ctx.strip((XA, yc + dy), (XB, yc + dy), 0.1, yellow, z=0.004, name="line")
            else:
                x = XA
                dash, gap = (3.0, 6.0) if rng.random() < 0.6 else (1.5, 1.5)
                while x < XB:
                    ctx.strip((x, yc), (x + dash, yc), 0.12, white, z=0.004, name="line")
                    x += dash + gap
        # parking lane lines / bays
        for on, yk, s in ((self.park_n, self.y_kn, -1), (self.park_s, self.y_ks, 1)):
            if not on:
                continue
            yl = yk + s * 2.1
            ctx.strip((XA, yl), (XB, yl), 0.1, white, z=0.004, name="line")
            if rng.random() < 0.5:
                x = XA
                while x < XB:
                    ctx.strip((x, yk), (x, yl), 0.1, white, z=0.004, name="line")
                    x += 5.8
        # a zebra crossing with stop lines, somewhere near the patch
        x0, _, x1, _ = ctx.bounds
        if rng.random() < 0.6:
            xz = float(rng.uniform(x0 - 4, x1 + 4))
            y = self.y_ks + 0.2
            while y < self.y_kn - 0.3:
                ctx.strip((xz - 1.5, y + 0.25), (xz + 1.5, y + 0.25), 0.5, white, z=0.005, name="zebra")
                y += 1.0
            for xs in (xz - 3.0, xz + 3.0):
                ctx.strip((xs, self.y_ks), (xs, self.y_kn), 0.3, white, z=0.004, name="stopline")
            self.zebra = xz
            tl = U.obj_pieces(ctx, U.OBJ_TRAFFIC_LIGHT)
            if tl and rng.random() < 0.7:                   # signals at the stop lines, facing the traffic
                k = tl[int(rng.integers(len(tl)))]
                for side, dx in ((1, 3.4), (-1, -3.4)):
                    yk = self.y_kn if side > 0 else self.y_ks
                    self.put([k], np.array([xz + dx, yk + side * 0.45]), math.copysign(math.pi / 2, dx))
        else:
            self.zebra = None
        # manholes and drain grates
        for _ in range(int(rng.integers(2, 6))):
            x = float(rng.uniform(x0 - 10, x1 + 10))
            y = float(rng.uniform(self.y_ks + 1, self.y_kn - 1))
            if "ph:water_manhole_cover" in ctx.lib.items:
                ctx.instance("ph:water_manhole_cover#0", (x, y), z=-0.075, yaw=float(rng.uniform(0, 6.28)),
                             collide=False)

    # --- furniture ------------------------------------------------------------------------------
    def _walk_spot(self, side, zone_kind, x):
        """A point on the pavement: 'kerb' furniture zone or 'front' zone."""
        yk = self.y_kn if side > 0 else self.y_ks
        yf = self.y_fn if side > 0 else self.y_fs
        if zone_kind == "kerb":
            return np.array([x, yk + side * float(self.ctx.rng.uniform(0.45, 0.8))])
        return np.array([x, yf - side * float(self.ctx.rng.uniform(0.45, 0.9))])

    def _on_walk(self, xy):
        if not self.kerbs:
            return 0.0
        y = xy[1]
        in_n = min(self.y_kn, self.y_fn) <= y <= max(self.y_kn, self.y_fn) + 0.5
        in_s = min(self.y_ks, self.y_fs) - 0.5 <= y <= max(self.y_ks, self.y_fs)
        if (in_n or in_s) and not any(a <= xy[0] <= b for s in self.gaps.values() for a, b in s):
            return self.kh - 0.01
        return 0.0

    def put(self, pkeys, xy, yaw, r=None, margin=0.3, s=1.0, collide=None):
        ctx = self.ctx
        k = ctx.lib.pick(ctx.rng, pkeys) if isinstance(pkeys, list) else pkeys
        if k is None:
            return None
        sx, sy, _ = np.asarray(ctx.lib.size(k)) * s
        rr = r if r is not None else 0.5 * math.hypot(sx, sy)
        if not ctx.free(xy, rr, margin):
            return None
        e = ctx.instance(k, xy, z=self._on_walk(xy), yaw=yaw, s=s, collide=collide)
        ctx.claim_disc(xy, rr, kind="prop")
        return e

    def furniture(self):
        ctx, rng, lk, pal = self.ctx, self.ctx.rng, self.ctx.looks, self.pal
        zx0, zy0, zx1, zy1 = ctx.zone
        x0, y0, x1, y1 = ctx.bounds
        info = {}
        # street trees (before lamps: their canopies claim space)
        # (tree_small_02 / island_tree_01, 1.6-2.1 M triangles, take ~27 s to import and fit twice in the budget)
        trees = ctx.lib.models(ids=["island_tree_02", "searsia_burchellii"], h=(3.0, 8.0)) + \
            U.obj_pieces(ctx, U.OBJ_STREET_TREE)
        tree_p = {"downtown": 0.5, "residential": 0.75, "oldtown": 0.25, "industrial": 0.1, "pedestrian": 0.7}[
            self.char]
        nt = 0
        pit = lk.pbr(U.pick_key(ctx, ["acg:Grate001", "ph:metal_grate_rusty"]), 1.2) \
            if rng.random() < 0.5 else lk.pbr(U.pick_key(ctx, ["ph:dirt", "acg:Ground048", "ph:brown_mud"] +
                                                         ctx.lib.materials(cats_any={"ground"})[:5]), 1.5)
        if trees and rng.random() < tree_p:
            tk = ctx.lib.pick(rng, trees)                      # one species per street
            fk, idx = tk.split("#")
            tris = next((p.get("tris", 0) for p in ctx.lib.pieces(fk) if p["index"] == int(idx)), 0)
            cap = 3 if tris > 1.2e6 else (4 if tris > 0.6e6 else 8)   # colliding copies (the cloud realises each)
            n_all = max(2, int(TREE_TRIS // max(tris, 1)))    # every copy is rendered: the nearest spots first
            n_in = 0
            sp = float(rng.uniform(7.0, 11.0))
            sides = [1, -1] if rng.random() < 0.6 else [int(rng.choice([1, -1]))]
            xm = 0.5 * (x0 + x1)
            spots = sorted(((side, x) for side in sides for x in np.arange(zx0 - 30 + rng.uniform(0, sp), zx1 + 30, sp)),
                           key=lambda q: abs(q[1] - xm))
            for side, x in spots:
                if nt >= n_all:
                    break
                xy = self._walk_spot(side, "kerb", x) if self.kerbs else \
                    np.array([x, (self.y_fn - 3.0) if side > 0 else (self.y_fs + 3.0)])
                if self.kerbs:
                    xy[1] = (self.y_kn + side * 1.0) if side > 0 else (self.y_ks - 1.0)
                sx, sy, _ = ctx.lib.size(tk)
                s = float(rng.uniform(0.85, 1.15)) * (0.72 if tk.startswith("obj:") else 1.0)   # 10.6 m -> 7.6 m
                rc = 0.45 * max(sx, sy) * s
                inside = ctx.rect_meets_zone(xy - rc, xy + rc)
                if inside and (n_in >= cap or not ctx.free(xy, rc, 0.3)):
                    continue
                n_in += int(inside)
                if any(a - 1 <= x <= b + 1 for a, b in self.gaps[side]):
                    continue
                if self.kerbs:
                    self._pit(xy, pit)
                else:
                    ctx.flat([(xy[0] - 0.7, xy[1] - 0.7), (xy[0] + 0.7, xy[1] - 0.7), (xy[0] + 0.7, xy[1] + 0.7),
                              (xy[0] - 0.7, xy[1] + 0.7)], pit, z=0.004, name="treepit")
                ctx.instance(tk, xy, z=self._on_walk(xy) - 0.05, yaw=float(rng.uniform(0, 6.28)), s=s)
                ctx.claim_disc(xy, rc, kind="tree")
                nt += 1
        info["trees"] = nt
        # street lamps
        heritage = self.char in ("oldtown", "pedestrian") and rng.random() < 0.6
        sp = float(rng.uniform(16.0, 28.0))
        stagger = rng.random() < 0.5
        nl = 0
        for side in (1, -1):
            off = sp / 2 if (stagger and side < 0) else 0.0
            for x in np.arange(self.XA + off + rng.uniform(0, 3), self.XB, sp):
                xy = self._walk_spot(side, "kerb", x) if self.kerbs else \
                    np.array([x, (self.y_fn - 1.2) if side > 0 else (self.y_fs + 1.2)])
                if any(a - 1 <= x <= b + 1 for a, b in self.gaps[side]):
                    continue
                if ctx.in_zone(xy) and not ctx.free(xy, 0.3, 0.4):
                    continue
                z = self._on_walk(xy)
                if heritage and "ph:street_lamp_01" in ctx.lib.items:
                    ctx.instance("ph:street_lamp_01#0", xy, z=z, yaw=float(rng.uniform(0, 6.28)))
                    if self.lit:
                        ctx.point_light((xy[0], xy[1], z + 3.5), 60.0, 0.15)
                else:
                    U.lamp_post(ctx, pal, xy, (0.0, -float(side)), z0=z, lit=self.lit)
                ctx.claim_disc(xy, 0.3, kind="lamp")
                nl += 1
        info["lamps"] = nl
        # parked cars: a street's own fleet of real models (Objaverse) and tarp-covered cars
        ncar = 0
        fleet = U.obj_pieces(ctx, U.OBJ_CARS) + U.obj_pieces(ctx, U.OBJ_VANS)
        if fleet:
            fleet = [fleet[int(i)] for i in rng.choice(len(fleet), min(len(fleet), 5), replace=False)]
        covered = "ph:covered_car#0" if "ph:covered_car" in ctx.lib.items else None
        variants = []
        if covered:
            tarps = [(0.55, 0.57, 0.6), (0.2, 0.25, 0.4), (0.6, 0.55, 0.45), (0.12, 0.12, 0.12), (0.3, 0.38, 0.3),
                     (0.5, 0.15, 0.12)]
            variants = [U.piece_variant(ctx, covered, f"tarp{i}",
                                        lambda m, c=c: U.tint_material(m, c, sat=0.2, val=1.6)
                                        if m.name.startswith("covered_car") else None)
                        for i, c in enumerate(tarps)]
        fill = {"downtown": 0.45, "residential": 0.75, "oldtown": 0.4, "industrial": 0.4, "pedestrian": 0}[
            self.char] * min(1.4, float(ctx.knobs.get("clutter", 1.0)) + 0.3)
        p_cov = 0.2 if fleet else 1.0
        for on, yk, sgn in ((self.park_n, self.y_kn, -1), (self.park_s, self.y_ks, 1)):
            if not on or not (fleet or covered):
                continue
            x = self.XA + float(rng.uniform(0, 4))
            while x < self.XB:
                use_cov = covered is not None and rng.random() < p_cov
                k = covered if use_cov else fleet[int(rng.integers(len(fleet)))]
                y0_, L, Wd = U.long_yaw(ctx, k)
                if rng.random() < fill:
                    xy = np.array([x + L / 2 + float(rng.normal(0, 0.2)),
                                   yk + sgn * (Wd / 2 + float(rng.uniform(0.15, 0.35)))])
                    yaw = y0_ + float(rng.normal(0, 0.03)) + (math.pi if rng.random() < 0.5 else 0.0)
                    half = (L / 2 + 0.05, Wd / 2 + 0.05)
                    if not ctx.in_zone(xy, pad=1.0) or ctx.free_rect(xy, half, 0.0, margin=0.3):
                        U.place_variant(ctx, k, variants[int(rng.integers(len(variants)))] if use_cov else None,
                                        xy, yaw=yaw)
                        ctx.claim_rect(xy, half, 0.0, kind="car")
                        ncar += int(ctx.in_zone(xy, pad=1.0))
                x += L + float(rng.uniform(0.7, 2.2))
        info["cars"] = ncar
        # pavement furniture inside the zone (scaled by clutter)
        bin_mat = lk.pbr(U.pick_key(ctx, U.PAINTED_METAL), 0.8, bevel=0.01, dirt=0.3,
                         tint=[(0.1, 0.1, 0.1), (0.08, 0.2, 0.1), (0.3, 0.3, 0.3)][int(rng.integers(3))])
        bol_mat = lk.pbr(U.pick_key(ctx, U.PAINTED_METAL + U.GALV), 0.8, bevel=0.01, dirt=0.3,
                         tint=[(0.08, 0.08, 0.08), (0.9, 0.9, 0.9), (0.35, 0.35, 0.35)][int(rng.integers(3))])
        band = lk.paint((0.9, 0.9, 0.9), rough=0.3, name="band")
        wood = lk.pbr(U.pick_key(ctx, U.WOOD), 1.0, bevel=0.005, dirt=0.2)
        props = {
            "bin": 2.0, "bollards": 1.5, "bikestand": 1.2, "bench": 1.0, "sign": 1.5, "hydrant": 0.8,
            "utility": 0.8, "planter": 0.8 if self.char in ("pedestrian", "oldtown", "downtown") else 0.2,
            "cafe": 1.4 if self.char in ("pedestrian", "oldtown", "downtown") else 0.0,
            "wheelie": 1.5 if self.char in ("residential", "industrial") else 0.3,
            "cones": 0.6, "barrier": 0.6 if self.char in ("industrial", "downtown") else 0.2,
            "pallet": 0.8 if self.char == "industrial" else 0.0,
            "phone": {"downtown": 0.3, "oldtown": 0.4, "pedestrian": 0.4}.get(self.char, 0.1),
            "mailbox": 0.4, "meter": 0.8 if (self.park_n or self.park_s) else 0.0,
            "vending": 0.4 if self.char in ("downtown", "pedestrian", "industrial") else 0.1,
            "busstop": 0.25 if self.kerbs else 0.0, "twowheel": 1.0 if self.char != "industrial" else 0.4,
        }
        self.obj = {k: U.obj_pieces(ctx, v) for k, v in (
            ("phone", U.OBJ_PHONE), ("mailbox", U.OBJ_MAIL), ("meter", U.OBJ_METER), ("vending", U.OBJ_VENDING),
            ("busstop", U.OBJ_BUS_STOP[1:]), ("twowheel", U.OBJ_TWO_WHEEL), ("hydrant", U.OBJ_HYDRANT),
            ("bin", U.OBJ_BIN))}
        counts = {}
        lo_x, hi_x = zx0 + 1, zx1 - 1
        for kind, per in props.items():
            n = U.draw_count(ctx, per)
            placed = 0
            for _ in range(n * 6):
                if placed >= n:
                    break
                side = int(rng.choice([1, -1]))
                x = float(rng.uniform(lo_x, hi_x))
                ok = self._prop(kind, side, x, bin_mat, bol_mat, band, wood)
                placed += int(bool(ok))
            counts[kind] = placed
        info["props"] = counts
        self.info.update(info)

    def _pit(self, xy, pit):
        """A tree pit cut into the pavement: a dark square flush with the paving."""
        ctx = self.ctx
        z = self._on_walk(xy)
        ctx.box((xy[0], xy[1], z + 0.002), (0.65, 0.65, 0.004), pit, name="treepit", collide=False)
        kerb = self.ctx.looks.pbr(U.pick_key(ctx, U.KERB), 1.0, bevel=0.01)
        for dx, dy, hx, hy in ((0.7, 0, 0.05, 0.75), (-0.7, 0, 0.05, 0.75), (0, 0.7, 0.75, 0.05),
                               (0, -0.7, 0.75, 0.05)):
            ctx.box((xy[0] + dx, xy[1] + dy, z + 0.01), (hx, hy, 0.012), kerb, name="treepit", collide=False)

    def _prop(self, kind, side, x, bin_mat, bol_mat, band, wood):
        ctx, rng, pal = self.ctx, self.ctx.rng, self.pal
        if any(a - 1 <= x <= b + 1 for a, b in self.gaps[side]):
            return False
        kerbside = self._walk_spot(side, "kerb", x)
        front = self._walk_spot(side, "front", x)
        if not self.kerbs:          # pedestrian / plaza: anywhere in the patch or along the fronts
            kerbside = ctx.uniform_xy(pad=1.0) if rng.random() < 0.5 else kerbside
        face_road = -math.pi / 2 if side > 0 else math.pi / 2       # yaw turning a piece's -y to the road
        obj = getattr(self, "obj", {})
        if kind == "bin":
            if obj.get("bin") and rng.random() < 0.5:
                return self.put(obj["bin"], kerbside, float(rng.uniform(0, 6.28))) is not None
            if not ctx.free(kerbside, 0.3, 0.4):
                return False
            U.litter_bin(ctx, bin_mat, kerbside, z0=self._on_walk(kerbside))
            return True
        if kind in ("phone", "mailbox", "vending", "meter"):         # real models, their fronts to the road
            if not obj.get(kind):
                return False
            p = kerbside if kind == "meter" or (kind == "mailbox" and rng.random() < 0.5) else front
            return self.put(obj[kind], p, face_road + float(rng.normal(0, 0.05))) is not None
        if kind == "twowheel":                  # bikes and motorbikes parked along the kerb
            if not obj.get("twowheel"):
                return False
            ok = False
            for i in range(int(rng.integers(1, 4))):
                k = obj["twowheel"][int(rng.integers(len(obj["twowheel"])))]
                y0_, L, Wd = U.long_yaw(ctx, k)
                p = kerbside + np.array([1.0 * i, 0.0])
                if ctx.free_rect(p, (L / 2, Wd / 2), 0.0, margin=0.3):
                    ctx.instance(k, p, z=self._on_walk(p), yaw=y0_ + (math.pi if rng.random() < 0.5 else 0.0))
                    ctx.claim_rect(p, (L / 2, Wd / 2), 0.0, kind="prop")
                    ok = True
            return ok
        if kind == "busstop":                   # a shelter on the pavement, its back to the road
            if not obj.get("busstop"):
                return False
            k = obj["busstop"][0]
            y0_, L, Wd = U.long_yaw(ctx, k)
            yk = self.y_kn if side > 0 else self.y_ks
            yf = self.y_fn if side > 0 else self.y_fs
            if abs(yf - yk) < Wd + 0.9:
                return False
            p = np.array([x, yk + side * (Wd / 2 + 0.4)])
            if not ctx.free_rect(p, (L / 2, Wd / 2), 0.0, margin=0.3):
                return False
            ctx.instance(k, p, z=self._on_walk(p), yaw=y0_ + (0.0 if side > 0 else math.pi))
            ctx.claim_rect(p, (L / 2, Wd / 2), 0.0, kind="prop")
            return True
        if kind == "bollards":
            if not self.kerbs:
                return False
            n = int(rng.integers(2, 6))
            yb = (self.y_kn + side * 0.3) if side > 0 else (self.y_ks - 0.3)
            pts = [np.array([x + 1.4 * i, yb]) for i in range(n)]
            if not all(ctx.free(p, 0.15, 0.4) for p in pts):
                return False
            sq = rng.random() < 0.3
            for p in pts:
                U.bollard(ctx, bol_mat, p, z0=self._on_walk(p), band=band if rng.random() < 0.5 else None, square=sq)
            return True
        if kind == "bikestand":
            n = int(rng.integers(1, 4))
            pts = [kerbside + np.array([0.9 * i, 0.0]) for i in range(n)]
            if not all(ctx.free(p, 0.45, 0.4) for p in pts):
                return False
            for p in pts:
                U.bike_stand(ctx, pal.galv, p, math.pi / 2, z0=self._on_walk(p))
            return True
        if kind == "bench":
            p = kerbside if rng.random() < 0.5 else front
            if not ctx.free(p, 1.0, 0.4):
                return False
            U.bench(ctx, wood, pal.metal, p, 0.0 if side > 0 else math.pi, z0=self._on_walk(p))
            return True
        if kind == "sign":
            if not ctx.free(kerbside, 0.2, 0.5):
                return False
            U.road_sign(ctx, pal, kerbside, 0.0 if rng.random() < 0.5 else math.pi, z0=self._on_walk(kerbside))
            return True
        if kind == "hydrant":
            keys = ["ph:fire_hydrant#0", "ph:fire_hydrant#1"] + obj.get("hydrant", [])
            return self.put(keys, kerbside, float(rng.uniform(0, 6.28))) is not None
        if kind == "utility":
            return self.put(["ph:utility_box_01#0", "ph:utility_box_02#0"], front, face_road, r=0.55) is not None
        if kind == "planter":
            p = front if rng.random() < 0.5 else kerbside
            e = self.put(["ph:planter_box_01#0", "ph:planter_box_02#0", "ph:planter_box_03#0",
                          "ph:potted_plant_01#0", "ph:potted_plant_02#0"], p, face_road, r=0.7)
            return e is not None
        if kind == "cafe":
            n = int(rng.integers(1, 4))
            ok = False
            for i in range(n):
                p = front + np.array([1.9 * i, 0.0])
                p[1] += -side * 0.5
                ok |= self.put(["ph:outdoor_table_chair_set_01#0"], p, 0.0, r=0.95) is not None
            if ok and rng.random() < 0.6:
                self.put(["ph:standing_chalkboard_01#0"], front + np.array([-1.5, -side * 0.2]), face_road, r=0.5)
            return ok
        if kind == "wheelie":
            n = int(rng.integers(1, 4))
            cols = [(0.1, 0.35, 0.12), (0.1, 0.15, 0.45), (0.12, 0.12, 0.12), (0.5, 0.5, 0.52), (0.45, 0.3, 0.1)]
            body = ctx.looks.paint(cols[int(rng.integers(len(cols)))], rough=0.55, name="wheelie")
            lid = ctx.looks.paint(cols[int(rng.integers(len(cols)))], rough=0.5, name="wheelie_lid")
            ok = False
            for i in range(n):
                p = front + np.array([0.7 * i, 0.0])
                if ctx.free(p, 0.5, 0.3):
                    U.wheelie_bin(ctx, body, lid, p, face_road + math.pi, z0=self._on_walk(p))
                    ok = True
            return ok
        if kind == "cones":
            n = int(rng.integers(2, 6))
            p0 = kerbside + np.array([0.0, -side * float(rng.uniform(0.8, 2.5))])
            pts = [p0 + np.array([0.9 * i, float(rng.normal(0, 0.1))]) for i in range(n)]
            if not all(ctx.free(p, 0.25, 0.4) for p in pts):
                return False
            for p in pts:
                U.cone(ctx, p, z0=self._on_walk(p))
            return True
        if kind == "barrier":
            p = kerbside + np.array([0.0, -side * float(rng.uniform(0.6, 2.0))])
            return self.put(["ph:concrete_road_barrier#0", "ph:concrete_road_barrier_02#0"], p,
                            float(rng.normal(0, 0.1)), r=0.85) is not None
        if kind == "pallet":
            if not ctx.free(front, 0.8, 0.3):
                return False
            U.pallet(ctx, ctx.looks.pbr(U.pick_key(ctx, U.WOOD), 1.0, dirt=0.4), front, float(rng.uniform(0, 3.14)),
                     z0=self._on_walk(front), stack=int(rng.integers(1, 5)))
            return True
        return False

    def semantics(self) -> list[str]:
        """How this dressing departs from the bare tier-A/B/C patch (course.json
        ``chosen.semantics_changed``): what the labeller's cloud has that the
        primitives do not, beyond ordinary clutter."""
        i, out = self.info, []
        x0, y0, x1, y1 = self.ctx.bounds
        if self.plaza:
            X0, Y0, X1, Y1 = self.plaza_rect
            n = self.plaza_gaps
            out.append(f"building fronts on all four sides, {x0 - X0:.1f} / {X1 - x1:.1f} / {y0 - Y0:.1f} / "
                       f"{Y1 - y1:.1f} m outside the patch (W/E/S/N); out of the square only through "
                       f"{n} gap(s) 8-12 m wide" if n else
                       f"building fronts on all four sides, {x0 - X0:.1f} / {X1 - x1:.1f} / {y0 - Y0:.1f} / "
                       f"{Y1 - y1:.1f} m outside the patch (W/E/S/N): no way round beyond the bounds")
        else:
            n = sum(len(v) for v in self.gaps.values())
            out.append(f"building fronts {y0 - self.y_fs:.1f} / {self.y_fn - y1:.1f} m outside the long sides of the "
                       f"patch (S/N): no way round beyond the bounds" + (f" but {n} side street(s) 8-12 m wide"
                                                                          if n else "") +
                       (f"; pavements {self.kh:.2f} m high behind kerbs" if self.kerbs else ""))
        if getattr(self, "n_ext", 0):
            out.append(f"{self.n_ext} wall end(s) at the patch edge extended to the building line")
        if any(p["kind"] == "wall" for p in self.ctx.spec["primitives"]):
            out.append("wall tops capped (+6 cm)")
        if i.get("poles_as") == "lamp":
            out.append("posts r <= 0.45 m carry a lamp head at the top (+12 cm, r + 0.25 m)")
        if i.get("overhead"):
            out.append(f"{i['overhead']} cable/festoon/span-sign runs across the street, lowest at "
                       f"{min(self.overhead_z):.2f} m")
        return out

    # --- overhead (L3) --------------------------------------------------------------------------
    def overhead(self):
        ctx, rng, pal, lk = self.ctx, self.ctx.rng, self.pal, self.ctx.looks
        n = ctx.count(float(ctx.knobs.get("hang", 0.0)))
        if n <= 0:
            return 0
        x0, _, x1, _ = ctx.bounds
        cable = lk.paint((0.03, 0.03, 0.03), rough=0.5, name="cable")
        bulb = lk.paint((1.0, 0.9, 0.7), rough=0.3, name="bulb",
                        emission=(1.0, 0.85, 0.55, 6.0) if self.lit else (1.0, 0.85, 0.55, 0.5))
        made = 0
        self.overhead_z = []               # the lowest z of each run (a bound where the sag is at mid-span)
        kinds = ["cable", "festoon", "span_sign", "service"]
        for _ in range(n * 10):
            if made >= n:
                break
            kind = str(rng.choice(kinds))
            xa = float(rng.uniform(x0, x1))
            xb = xa + float(rng.normal(0, 4.0))
            za = float(rng.uniform(3.4, 5.0))
            zb = float(rng.uniform(3.4, 5.0))
            sag = float(rng.uniform(0.3, 1.0))
            a = np.array([xa, self.y_fs - 0.3, za])
            b = np.array([xb, self.y_fn + 0.3, zb])
            if not U.seg_clear(ctx, a, b, clear=0.4, kinds={"prim", None, "start", "goal", "tree", "lamp", "upole"}):
                continue
            # the lowest point must leave the start and goal (1.75 m up) alone
            if min(za, zb) - sag < 2.6:
                sag = min(za, zb) - 2.6
            if kind in ("cable", "service"):
                U.catenary(ctx, a, b, sag, 0.012, cable, name="cable")
                self.overhead_z.append(min(za, zb) - sag - 0.012)
                if rng.random() < 0.5:
                    U.catenary(ctx, a + [0, 0, 0.25], b + [0, 0, 0.3], sag, 0.01, cable, name="cable")
            elif kind == "festoon":
                pts = U.catenary(ctx, a, b, sag, 0.008, cable, name="festoon", n=12)
                for i in range(1, 24):
                    t = i / 24
                    p = a + (b - a) * t
                    p[2] -= sag * 4 * t * (1 - t) + 0.08
                    U.ball(ctx, p, 0.045, bulb, name="bulb", collide=True, n=8)
                    self.overhead_z.append(float(p[2]) - 0.045)
                if self.lit:
                    ctx.point_light(((a + b) / 2)[:2].tolist() + [float((za + zb) / 2 - sag)], 40.0, 0.3)
            else:   # a sign hung from a span wire over the road
                U.catenary(ctx, a, b, sag * 0.4, 0.012, cable, name="span")
                mid = (a + b) / 2
                mid[2] -= sag * 0.4
                yaw = math.atan2(b[1] - a[1], b[0] - a[0])
                m = pal.pick(pal.signs)
                ctx.box((mid[0], mid[1], mid[2] - 0.45), (0.9, 0.04, 0.35), m, yaw=yaw, name="span_sign")
                self.overhead_z.append(float(mid[2]) - 0.8)
                ctx.cylinder(mid, mid - np.array([0, 0, 0.1]), 0.01, cable, name="span")
                for sgn in (1, -1):
                    nrm = np.array([math.cos(yaw - math.pi / 2), math.sin(yaw - math.pi / 2)]) * sgn
                    txt = str(rng.choice(["CITY CENTRE", "FESTIVAL", "MARKET DAY", "PARKING ->", "WELCOME",
                                          "SALE", "HOTEL", "MUSEUM"]))
                    U.text_obj(ctx, txt, (mid[0] + nrm[0] * 0.045, mid[1] + nrm[1] * 0.045, mid[2] - 0.5),
                               math.atan2(nrm[1], nrm[0]) + math.pi / 2, 0.22, pal.letters[0], coll="scatter")
            made += 1
        # utility poles with service drops (wooden, US/residential feel)
        if self.char in ("residential", "industrial") and self.kerbs and rng.random() < 0.7:
            wood = lk.pbr(U.pick_key(ctx, ["acg:Wood049", "acg:Wood051", "ph:bark_brown_01"]), 1.5, dirt=0.4,
                          tint=(0.55, 0.45, 0.35))
            side = int(rng.choice([1, -1]))
            yk = (self.y_kn + 0.4) if side > 0 else (self.y_ks - 0.4)
            prev = None
            for x in np.arange(self.XA, self.XB, float(rng.uniform(30, 40))):
                xy = np.array([x, yk])
                if ctx.in_zone(xy) and not ctx.free(xy, 0.25, 0.4):
                    prev = None
                    continue
                h = float(rng.uniform(9.0, 11.0))
                ctx.cylinder((x, yk, 0.0), (x, yk, h), 0.14, wood, name="upole", segments=12)
                ctx.box((x, yk, h - 0.5), (0.06, 1.1, 0.06), wood, name="upole")
                ctx.claim_disc(xy, 0.25, kind="upole")
                if prev is not None:
                    for dy in (-0.9, 0.0, 0.9):
                        U.catenary(ctx, (prev[0], prev[1] + dy, h - 0.45), (x, yk + dy, h - 0.45), 0.6, 0.01, cable,
                                   name="wire", n=6)
                prev = (x, yk)
        return made


def build(ctx) -> dict:
    st = Street(ctx)
    lk = ctx.looks
    tev = {"dry": 0.35, "wet": 0.5, "dusk": 0.5, "snow": 0.0}[st.weather]
    sky = pick_sky(ctx, st.weather, target_ev=tev)
    st.layout()
    st.ground()
    st.dress()
    st.buildings()
    st.pavements()
    st.markings()
    st.furniture()
    st.info["overhead"] = st.overhead()
    st.info["semantics_changed"] = st.semantics()
    return {"sky": sky, **st.info}
