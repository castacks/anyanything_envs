"""Back alley: a narrow lane between the backs of buildings -- brick and
plaster walls close on both sides (colliding, some inside the patch), fire
escapes, AC units, pipes, gutters, vents, security lights and cameras on the
walls; dumpsters, wheelie bins, trash cans and bags, pallets, crates,
tyres and roller-shuttered back doors on the ground; chain-link fences
across the gaps; puddles on cracked asphalt or cobbles; cables, laundry
lines and hanging lamps strung across at hard levels.

How it keeps the labeller's contract with walls *inside* the patch: the
building lines are chosen per building (5-12 m apart, jittered), and a
building's footprint must clear every patch obstacle (and every opening's
approach, the start and the goal) -- except the ones that span the whole
lane (walls, bars, lintels that reach both building lines), which simply
run into the walls. A building that does not fit steps back, so the lane
jogs round what the patch put there.

A wall the drone must go round (no opening in the flight band) is never
swallowed: each of its ends keeps a ``BYPASS`` disc the buildings, fences
and clutter all stay off, so the lane cannot seal the patch.

Arena-shaped patches become a back court (a yard enclosed by building
backs on all four sides) instead of a lane.
"""

from __future__ import annotations

import math

import numpy as np

import build_course as BC
from kits import _street_util as U
from kits.street import pick_sky

NAME = "alley"
INDOOR = False
DESCRIPTION = "back alley: narrow lane between building backs, fire escapes, dumpsters, pipes, cables overhead"

#: metres a building keeps clear of each end of a solid wall / grounded box (the way round it)
BYPASS = 1.8

ALLEY_TAGS = {"alley", "brick", "courtyard", "street", "industrial", "backyard", "garage", "cobblestone",
              "narrow", "warehouse", "apartments", "graffiti", "old town", "sidewalk"}


class Alley:
    def __init__(self, ctx):
        self.ctx = ctx
        rng = ctx.rng
        x0, y0, x1, y1 = ctx.bounds
        self.court = (y1 - y0) > 0.75 * (x1 - x0)
        self.weather = str(rng.choice(["dry", "wet", "dusk", "snow"], p=[0.5, 0.33, 0.12, 0.05]))
        self.lit = self.weather == "dusk" or rng.random() < 0.15
        # the alley's flavour: a US brick back alley, an old European lane, or a modern service lane
        self.flavour = str(rng.choice(["us_brick", "euro_old", "service"], p=[0.45, 0.3, 0.25]))
        F = {"us_brick": dict(look=["brick", "brick", "worn"], ground=[0.6, 0.3, 0.1], modular=0.4, fire=0.6,
                              pitched=0.05, grounds=["back", "back", "back", "garage", "blank"],
                              hang=["cable", "cable", "laundry", "lamp"]),
             "euro_old": dict(look=["plaster", "worn", "stone"], ground=[0.15, 0.2, 0.65], modular=0.1, fire=0.05,
                              pitched=0.5, grounds=["back", "res", "back", "blank"],
                              hang=["laundry", "laundry", "cable", "lamp"]),
             "service": dict(look=["modern", "mixed", "brick"], ground=[0.35, 0.6, 0.05], modular=0.3, fire=0.2,
                             pitched=0.0, grounds=["garage", "garage", "blank", "back"],
                             hang=["cable", "cable", "lamp"])}[self.flavour]
        self.F = F
        look = str(rng.choice(F["look"]))
        self.pal = U.Palette(ctx, look=look, grime=float(rng.uniform(0.45, 0.8)))
        zx0, zy0, zx1, zy1 = ctx.zone
        self.XA, self.XB = zx0 - 40.0, zx1 + 40.0
        self.fronts = []
        self.info = {"weather": self.weather, "court": self.court, "look": look, "flavour": self.flavour}
        # the lane: centred between start and goal, 5-12 m between building lines
        g = ctx.goal
        self.yc = float(np.clip(g[1] / 2, -1.0, 1.0))
        width = float(rng.choice([rng.uniform(5.0, 7.0), rng.uniform(6.5, 9.0), rng.uniform(8.0, 12.0)]))
        a = width / 2
        skew = float(rng.uniform(-0.6, 0.6))
        self.a_n = max(2.4, a + skew)
        self.a_s = max(2.4, a - skew)
        self.info["lane_m"] = round(self.a_n + self.a_s, 2)
        if not self.court:           # the building lines (buildings stand outside, stepped back where needed)
            self.info["lane_y"] = [round(self.yc - self.a_s, 3), round(self.yc + self.a_n, 3)]

    # --- obstacles that buildings may swallow / must avoid ---------------------------------------
    def blockers(self):
        """Footprints a building must clear: every patch solid except those
        spanning the lane that leave a way through (a wall with an opening,
        a floating box, a bar), every opening's approach, the start and the
        goal. A solid wall or grounded box the drone must go round keeps a
        ``BYPASS`` disc at each end, so a building never seals the way round
        it (the lane jogs out instead)."""
        ctx = self.ctx
        ko = float(ctx.knobs.get("keepout", 2.5))
        y_lo, y_hi = self.yc - self.a_s, self.yc + self.a_n
        out = [f for f in ctx.foot if f.get("kind") in ("start", "goal")]
        spanning = {}                      # what reaches both building lines and may run into the buildings

        def ends(c, half, R):
            """Bypass discs at both ends of a footprint's long side."""
            ax = R[:, 0] if half[0] >= half[1] else R[:, 1]
            L = max(half[0], half[1])
            return [{"type": "disc", "c": np.asarray(c, float) + sgn * ax * L, "r": BYPASS} for sgn in (1, -1)]
        for p in ctx.spec["primitives"]:
            k = p["kind"]
            if k == "wall":
                p0, p1 = np.asarray(p["p0"], float), np.asarray(p["p1"], float)
                opening = any(zb < 2.5 and zt > 1.0 for _, _, zb, zt in p.get("slots", []))
                spans = opening and min(p0[1], p1[1]) <= y_lo + 0.2 and max(p0[1], p1[1]) >= y_hi - 0.2
                d = (p1 - p0) / np.linalg.norm(p1 - p0)
                Rw = np.array([[d[0], -d[1]], [d[1], d[0]]])
                for s, w, _, _ in p.get("slots", []):          # openings' approaches: always kept
                    c = p0 + d * s
                    out.append({"type": "rect", "c": c, "half": np.array([w / 2 + 0.4 * ko + 0.4, ko + 0.3]),
                                "R": Rw})
                if spans:
                    spanning["wall with an opening"] = spanning.get("wall with an opening", 0) + 1
                else:
                    for c, h, R in BC.wall_pieces(p):
                        out.append({"type": "rect", "c": np.asarray(c[:2]), "half": np.asarray(h[:2]),
                                    "R": R[:2, :2]})
                    if not opening:
                        out += [{"type": "disc", "c": q, "r": BYPASS} for q in (p0[:2], p1[:2])]
            elif k == "box":
                c, h = np.asarray(p["c"], float), np.asarray(p["half"], float)
                R = BC.rpy_matrix(p["rpy"])
                ext = np.abs(R[:2, :2]) @ h[:2]
                grounded = c[2] - abs(R[2]) @ h < 0.6          # nothing to fly under
                if not grounded and c[1] - ext[1] <= y_lo + 0.2 and c[1] + ext[1] >= y_hi - 0.2:
                    spanning["floating box"] = spanning.get("floating box", 0) + 1
                    continue
                out.append({"type": "rect", "c": c[:2], "half": h[:2], "R": R[:2, :2]})
                if grounded:
                    out += ends(c[:2], h[:2], R[:2, :2])
            elif k == "cyl_h":
                a, b = np.asarray(p["a"][:2], float), np.asarray(p["b"][:2], float)
                if min(a[1], b[1]) <= y_lo + 0.2 and max(a[1], b[1]) >= y_hi - 0.2:
                    spanning["bar"] = spanning.get("bar", 0) + 1
                    continue
                L = float(np.linalg.norm(b - a))
                d = (b - a) / max(L, 1e-9)
                out.append({"type": "rect", "c": (a + b) / 2, "half": np.array([L / 2, p["r"] + 0.3]),
                            "R": np.array([[d[0], -d[1]], [d[1], d[0]]])})
            elif k == "cyl_v":
                out.append({"type": "disc", "c": np.asarray(p["c"], float), "r": p["r"]})
            elif k == "sphere":
                out.append({"type": "disc", "c": np.asarray(p["c"][:2], float), "r": p["r"]})
        for f in out:                      # clutter, fences and wall kit keep off the ways round too
            if f.get("r") == BYPASS:
                ctx.claim_disc(f["c"], BYPASS, kind="bypass")
        self._blk = out
        self.info["lane_spanning"] = spanning
        return out

    def rect_free(self, xa, xb, ya, yb, margin=0.3):
        """The axis-aligned rectangle clears every blocker by ``margin``."""
        c = np.array([(xa + xb) / 2, (ya + yb) / 2])
        half = np.array([abs(xb - xa) / 2, abs(yb - ya) / 2])
        for f in self._blk:
            if f["type"] == "disc":
                q = np.abs(f["c"] - c) - half
                if np.linalg.norm(np.maximum(q, 0)) < f["r"] + margin:
                    return False
            else:
                # sample the blocker rect's outline + centre against our rect (both are small in number)
                R, h = f["R"], f["half"]
                pts = [f["c"] + R @ np.array([sx * h[0], sy * h[1]]) for sx in np.linspace(-1, 1, 9)
                       for sy in np.linspace(-1, 1, 5)]
                for q in pts:
                    d = np.abs(q - c) - half
                    if np.linalg.norm(np.maximum(d, 0)) < margin:
                        return False
                # our rect's corners/centre inside the blocker
                for qx in np.linspace(xa, xb, max(2, int(abs(xb - xa) / 0.5) + 1)):
                    for qy in np.linspace(ya, yb, max(2, int(abs(yb - ya) / 0.5) + 1)):
                        if BC.footprint_distance(np.array([qx, qy]), f) < margin:
                            return False
        return True

    # --- ground ------------------------------------------------------------------------------------
    def ground(self):
        ctx, rng, lk = self.ctx, self.ctx.rng, self.ctx.looks
        puddles = {"dry": float(rng.choice([0.0, 0.2])), "wet": float(rng.uniform(0.6, 1.0)), "dusk": 0.3,
                   "snow": 0.2}[self.weather]
        g = self.F["ground"]
        r = rng.random()
        if r < g[0]:
            keys = [U.pick_key(ctx, U.ASPHALT), U.pick_key(ctx, U.ASPHALT_PATCHED), U.pick_key(ctx, U.CONCRETE_GROUND)]
            tile = 3.5
        elif r < g[0] + g[1]:
            keys = [U.pick_key(ctx, U.CONCRETE_GROUND), U.pick_key(ctx, U.CONCRETE_GROUND),
                    U.pick_key(ctx, U.ASPHALT)]
            tile = 3.0
        else:
            keys = [U.pick_key(ctx, U.COBBLE), U.pick_key(ctx, U.CONCRETE_GROUND)]      # one cobble pattern
            tile = 1.6
        if self.weather == "snow":
            keys = [keys[0], "ph:asphalt_snow", "ph:snow_02"]
        ctx.ground(lk.ground_material(keys, tile_m=tile, puddles=puddles, blend_scale=float(rng.uniform(5, 11))))
        # a drainage channel down the middle, now and then
        if not self.court and rng.random() < 0.5:
            ch = lk.pbr(U.pick_key(ctx, ["acg:Concrete017", "acg:Concrete019", "ph:concrete", "acg:Granite002A"]),
                        1.0, dirt=0.6)
            ctx.strip((self.XA, self.yc), (self.XB, self.yc), float(rng.uniform(0.3, 0.6)), ch, z=0.003,
                      name="channel")
        self.info.update({"ground": keys, "puddles": puddles})

    # --- the obstacles ------------------------------------------------------------------------------
    def dress(self):
        ctx, rng, lk, pal = self.ctx, self.ctx.rng, self.ctx.looks, self.pal
        wk = str(rng.choice(["brick", "block", "corrugated", "plywood", "brick"]))
        if wk == "brick":
            wall = lk.pbr(U.pick_key(ctx, U.BRICK_RED + U.BRICK_DARK), 1.5, bevel=0.015, dirt=0.6)
        elif wk == "block":
            wall = lk.pbr(U.pick_key(ctx, ["ph:concrete_block_wall", "ph:concrete_block_wall_02",
                                           "acg:Bricks064", "acg:Bricks066"]), 2.0, bevel=0.015, dirt=0.6)
        elif wk == "corrugated":
            wall = lk.pbr(U.pick_key(ctx, U.CORRUGATED), 1.5, bevel=0.01, dirt=0.5)
        else:
            wall = lk.pbr(U.pick_key(ctx, ["ph:plywood", "ph:distressed_painted_planks", "acg:PaintedWood005"]),
                          2.2, bevel=0.01, dirt=0.5)
        box = lk.pbr(U.pick_key(ctx, U.RUSTY_PAINT + U.CONCRETE_WALL + U.CORRUGATED[:3]), 1.5, bevel=0.02, dirt=0.5)
        pole = lk.pbr(U.pick_key(ctx, ["ph:rusty_metal", "ph:rusty_metal_03", "acg:Rust002"] + U.GALV), 1.0,
                      bevel=0.01, dirt=0.4)
        bar = lk.pbr(U.pick_key(ctx, U.GALV + ["ph:rusty_metal_04", "ph:green_metal_rust"]), 1.0, bevel=0.005,
                     dirt=0.4)
        rock = lk.pbr(U.pick_key(ctx, U.CONCRETE_WALL + ["acg:Concrete031"]), 1.5, bevel=0.02, dirt=0.5)
        ctx.dress_primitives({"wall": wall, "box": box, "pole": pole, "bar": bar, "rock": rock})
        # coping on masonry walls
        n_cap = 0
        if wk in ("brick", "block"):
            cap = lk.pbr(U.pick_key(ctx, U.TRIM), 1.5, bevel=0.01, dirt=0.4)
            for p in ctx.spec["primitives"]:
                if p["kind"] != "wall":
                    continue
                p0, p1 = np.asarray(p["p0"], float), np.asarray(p["p1"], float)
                L = float(np.linalg.norm(p1 - p0))
                d = (p1 - p0) / L
                cuts = sorted([(s - w / 2, s + w / 2) for s, w, zb, zt in p.get("slots", [])
                               if zt >= p["z1"] - 1e-3])
                a = 0.0
                segs = []
                for ca, cb in cuts:
                    if ca > a:
                        segs.append((a, ca))
                    a = max(a, cb)
                if a < L:
                    segs.append((a, L))
                for sa, sb in segs:
                    c = p0 + d * (sa + sb) / 2
                    ctx.box((c[0], c[1], p["z1"] + 0.04), ((sb - sa) / 2, p["t"] / 2 + 0.03, 0.04), cap,
                            yaw=math.atan2(d[1], d[0]), name="coping")
                    n_cap += 1
        if wk in ("plywood", "block", "corrugated") and rng.random() < 0.6:
            self.info["posters"] = U.posters(ctx, pal, max_groups=4)
        self.info["walls_as"] = wk
        self.info["copings"] = n_cap

    # --- buildings -----------------------------------------------------------------------------------
    def rows(self):
        ctx, rng, pal = self.ctx, self.ctx.rng, self.pal
        self.blockers()
        x0, y0, x1, y1 = ctx.bounds
        for side in (1, -1):
            a = self.a_n if side > 0 else self.a_s
            x = self.XA
            while x < self.XB - 2.0:
                w = float(rng.uniform(5.0, 15.0))
                xa, xb = x, min(x + w, self.XB)
                y_line, built = None, False
                inside = ctx.rect_meets_zone((xa, -1e3), (xb, 1e3))
                for tries in range(40):
                    yl = self.yc + side * (a + 0.35 * tries)
                    dep = float(rng.uniform(8.0, 14.0))
                    ya, yb = sorted((yl, yl + side * dep))
                    if not inside or self.rect_free(xa, xb, ya, yb, margin=0.25):
                        y_line = yl
                        break
                if y_line is None and xb - xa > 4.0:           # try the halves
                    x += (xb - xa) / 2 if xb - xa > 6 else 1.0
                    continue
                if y_line is None:
                    x = xb
                    continue
                kind = self._building(side, xa, xb, y_line, dep)
                self.fronts.append({"side": side, "xa": xa, "xb": xb, "y": y_line, "kind": kind})
                x = xb
        self.info["buildings"] = len(self.fronts)

    def court_rows(self):
        """Arena patches: a back court -- building backs on all four sides,
        just outside the patch, stepping back where the patch reaches out."""
        ctx, rng = self.ctx, self.ctx.rng
        self.a_n = self.a_s = 1e4                 # nothing spans a court
        self.blockers()
        x0, y0, x1, y1 = ctx.bounds
        s = [float(rng.uniform(0.3, 1.5)) for _ in range(4)]
        s[0] = float(rng.uniform(3.4, 4.5))               # the chase preview stands 3 m west of the patch
        X0, X1, Y0, Y1 = x0 - s[0], x1 + s[1], y0 - s[2], y1 + s[3]
        sides = [(U.Frame((X0 - 25.0, Y1), (0.0, -1.0)), X1 - X0 + 50.0, False),
                 (U.Frame((X1 + 25.0, Y0), (0.0, 1.0)), X1 - X0 + 50.0, True),
                 (U.Frame((X1, Y1), (-1.0, 0.0)), Y1 - Y0, False),
                 (U.Frame((X0, Y0), (1.0, 0.0)), Y1 - Y0, False)]
        for fr, L, roof in sides:
            u = 0.0
            while u < L - 2.0:
                w = min(float(rng.uniform(6.0, 14.0)), L - u)
                if L - u - w < 3.0:
                    w = L - u
                dep = float(rng.uniform(8.0, 12.0))
                v0 = None
                for k in range(30):
                    v = -0.35 * k
                    ps = [fr.xy(u, v), fr.xy(u + w, v), fr.xy(u, v - dep), fr.xy(u + w, v - dep)]
                    xs, ys = [q[0] for q in ps], [q[1] for q in ps]
                    if not ctx.rect_meets_zone((min(xs), min(ys)), (max(xs), max(ys))) or \
                            self.rect_free(min(xs), max(xs), min(ys), max(ys), margin=0.25):
                        v0 = v
                        break
                if v0 is not None:
                    self._building(None, u, u + w, None, dep, fr=fr, v0=v0, roof=roof)
                u += w
        self.info["buildings"] = len(getattr(self, "fronts_rec", []))
        # a gate in one side and far blocks
        for side in (1, -1):
            x = X0 - 40
            while x < X1 + 40:
                w, h = float(rng.uniform(10, 25)), float(rng.uniform(12, 35))
                y = (Y1 + 22 + float(rng.uniform(0, 15))) if side > 0 else (Y0 - 22 - float(rng.uniform(0, 15)))
                ctx.box((x + w / 2, y, h / 2), (w / 2, 8.0, h / 2), self.pal.pick(self.pal.walls), coll="backdrop",
                        collide=False, name="block", roof=side < 0)
                x += w + float(rng.uniform(0, 5))

    def _building(self, side, xa, xb, y_line, dep, fr=None, v0=0.0, roof=None):
        ctx, rng, pal = self.ctx, self.ctx.rng, self.pal
        if fr is None:
            roof = side < 0
            if side > 0:
                fr = U.Frame((xa, y_line), (0.0, -1.0))
            else:
                fr = U.Frame((xb, y_line), (0.0, 1.0))
            u_off = 0.0
        else:
            u_off = xa                     # court mode: xa/xb are u along the given frame
        w = xb - xa
        far = not ctx.in_zone(fr.xy(u_off + w / 2), pad=30.0)
        if rng.random() < self.F["modular"] and w >= 6.0 and not far:
            nm = int(w // 3)
            off = (w - 3 * nm) / 2
            fl = int(rng.integers(2, 5))
            H = U.modular_building(ctx, pal, fr, u_off + off, nm, fl, kind="factory" if rng.random() < 0.7 else "apts",
                                   depth=dep, roof=roof, v0=v0)
            if off > 0.05:
                for a_, b_ in ((0.0, off), (w - off, w)):
                    acc = U.Acc(ctx)
                    fr.box(acc, pal.pick(pal.walls), u_off + a_, u_off + b_, v0 - dep, v0 - 0.1, 0.0, 3.0 * fl + 0.7)
                    acc.emit(name="infill", roof_all=roof)
            rec = {"fr": fr, "u0": u_off, "u1": u_off + w, "g": 3.0, "f": 3.0, "n": fl - 1, "H": H, "v0": v0,
                   "mod": True}
        else:
            ground = str(rng.choice(self.F["grounds"]))
            b = U.Building(ctx, pal, fr, u_off, u_off + w, dep, ground=ground, floors=int(rng.integers(1, 5)),
                           v0=v0, lit=self.lit, pitched=rng.random() < self.F["pitched"], far=far,
                           roof=roof)
            b.cornice = b.cornice and rng.random() < 0.5
            b.balconies = 0.0
            b.shutters = b.shutters and self.flavour == "euro_old"
            b.french = False
            b.check = lambda pts: all(self.ctx.free(p, 0.0, 0.2) for p in pts)
            b.build()
            rec = {"fr": fr, "u0": u_off, "u1": u_off + w, "g": b.g, "f": b.f, "n": b.n, "H": b.H, "v0": v0,
                   "mod": False,
                   "holes": [(h[0], h[1], h[2], h[3]) for h in b.holes]}
        rec["roof"] = roof
        rec["far"] = far
        self.fronts_rec = getattr(self, "fronts_rec", [])
        self.fronts_rec.append(rec)
        return "modular" if rec["mod"] else ground

    def ends(self):
        """What the lane runs into at both ends: a cross street's far side or a
        dead-end wall with a gate."""
        ctx, rng, lk, pal = self.ctx, self.ctx.rng, self.ctx.looks, self.pal
        y_lo, y_hi = self.yc - self.a_s - 20, self.yc + self.a_n + 20
        for xe, nrm in ((self.XA - 2.0, (1.0, 0.0)), (self.XB + 2.0, (-1.0, 0.0))):
            if rng.random() < 0.5:           # a cross street: the building fronts across it
                x_far = xe - 14.0 if nrm[0] > 0 else xe + 14.0
                o = (x_far, y_lo) if nrm[0] > 0 else (x_far, y_hi)
                fr = U.Frame(o, nrm)
                u = 0.0
                while u < y_hi - y_lo:
                    w = float(rng.uniform(7, 14))
                    U.Building(ctx, pal, fr, u, u + w, 10.0, ground="shops", floors=int(rng.integers(2, 6)),
                               far=True).build()
                    u += w
            else:                            # dead end: a brick wall with a gate
                wall = pal.pick(pal.walls)
                ctx.box((xe, self.yc, 2.5), (0.2, (self.a_n + self.a_s) / 2 + 3.0, 2.5), wall, coll="backdrop",
                        name="deadend")
        # far skyline blocks
        for side in (1, -1):
            x = self.XA - 20
            while x < self.XB + 20:
                w, h = float(rng.uniform(10, 25)), float(rng.uniform(12, 35))
                y = self.yc + side * (max(self.a_n, self.a_s) + 22 + float(rng.uniform(0, 15)))
                ctx.box((x + w / 2, y, h / 2), (w / 2, 8.0, h / 2), pal.pick(pal.walls), coll="backdrop",
                        collide=False, name="block", roof=side < 0)
                x += w + float(rng.uniform(0, 5))

    # --- wall-mounted stuff --------------------------------------------------------------------------
    def wall_stuff(self):
        ctx, rng, pal, lk = self.ctx, self.ctx.rng, self.pal, self.ctx.looks
        n_fe = n_ac = n_pipe = n_light = n_shut = 0
        has = lambda k: k.split("#")[0] in ctx.lib.items  # noqa: E731
        for rec in getattr(self, "fronts_rec", []):
            if rec["far"]:
                continue
            fr, u0, u1 = rec["fr"], rec["u0"], rec["u1"]
            inzone = ctx.in_zone(fr.xy((u0 + u1) / 2), pad=2.0)
            holes = rec.get("holes", [])

            def clear_of_holes(ua, ub, za, zb, pad=0.15):
                return not any(ua < h[1] + pad and ub > h[0] - pad and za < h[3] + pad and zb > h[2] - pad
                               for h in holes)

            def mount(pkey, u, z, depth_off=None, s=1.0):
                if not has(pkey):
                    return None
                sx, sy, sz = ctx.lib.size(pkey)
                dv = sy * s / 2 + 0.01 if depth_off is None else depth_off
                xy = fr.xy(u, rec["v0"] + dv)
                # a wall unit's whole depth keeps off the claims (the building may stand 0.25 m from one)
                if inzone and not ctx.free(xy, sy * s / 2 if depth_off is None else 0.0, 0.15):
                    return None
                e = ctx.instance(pkey, xy, z=z, yaw=fr.yaw, s=s, coll="structure")
                if rec["roof"]:
                    e["roof"] = 1
                return e

            # fire escape on multi-storey backs
            if rec["n"] >= 2 and u1 - u0 >= 6.0 and rng.random() < self.F["fire"] and has("ph:modular_fire_escape#1"):
                if self._fire_escape(rec, mount):
                    n_fe += 1
            if rec["mod"]:
                holes = [(u0, u1, 0.0, rec["H"])]      # module windows are unknown: keep the walls clear
            # AC units on brackets, vents
            for _ in range(int(rng.integers(0, 4))):
                u = float(rng.uniform(u0 + 0.8, u1 - 0.8))
                z = float(rng.uniform(1.6, min(rec["H"] - 1.2, 5.5)))
                if clear_of_holes(u - 0.5, u + 0.5, z - 0.1, z + 1.0):
                    if mount("ph:exterior_aircon_unit#0" if rng.random() < 0.5 else "ph:exterior_aircon_unit#1",
                             u, z):
                        n_ac += 1
            if rng.random() < 0.5:
                u = float(rng.uniform(u0 + 0.8, u1 - 0.8))
                z = float(rng.uniform(2.2, 4.0))
                if clear_of_holes(u - 0.4, u + 0.4, z, z + 0.6):
                    mount(str(rng.choice(["ph:modular_airduct_circular_01#14", "ph:modular_airduct_circular_01#6"])),
                          u, z)
            # pipes: a downpipe and a conduit run
            if rng.random() < 0.7:
                u = float(rng.uniform(u0 + 0.3, u1 - 0.3))
                if clear_of_holes(u - 0.12, u + 0.12, 0.0, rec["H"]):
                    r = float(rng.uniform(0.045, 0.08))
                    vv = rec["v0"] + r + 0.03
                    ob = ctx.cylinder((*fr.xy(u, vv), 0.0), (*fr.xy(u, vv), rec["H"] - 0.3), r,
                                      pal.galv if rng.random() < 0.6 else pal.metal, coll="structure",
                                      name="pipe", segments=12)
                    if rec["roof"]:
                        ob["roof"] = 1
                    n_pipe += 1
            if rng.random() < 0.5:
                z = float(rng.uniform(2.4, 3.6))
                ua, ub = u0 + 0.2, u1 - 0.2
                if clear_of_holes(ua, ub, z - 0.06, z + 0.06):
                    ob = ctx.cylinder((*fr.xy(ua, rec["v0"] + 0.05), z), (*fr.xy(ub, rec["v0"] + 0.05), z), 0.025,
                                      pal.metal,
                                      coll="structure", name="conduit", segments=8)
                    if rec["roof"]:
                        ob["roof"] = 1
            # lights over doors / security lights and cameras
            for _ in range(int(rng.integers(0, 3))):
                u = float(rng.uniform(u0 + 0.5, u1 - 0.5))
                z = float(rng.uniform(2.6, 3.4))
                key = str(rng.choice(["ph:security_light#0", "ph:street_lamp_02#0", "ph:security_camera_01#0",
                                      "ph:industrial_wall_lamp#0", "ph:industrial_caged_sconce#0"]))
                if clear_of_holes(u - 0.3, u + 0.3, z - 0.2, z + 0.6) and mount(key, u, z):
                    n_light += 1
                    if self.lit and "camera" not in key:
                        ctx.point_light((*fr.xy(u, rec["v0"] + 0.6), z - 0.1), 80.0, 0.1, rgb=(1.0, 0.8, 0.55))
            # graffiti roller shutters on the ground floor
            if rng.random() < 0.5:
                key = str(rng.choice(["ph:rollershutter_door#1", "ph:rollershutter_window_01#1",
                                      "ph:rollershutter_window_02#1", "ph:rollershutter_window_03#1",
                                      "ph:rollershutter_door#0", "ph:rollershutter_window_01#0"]))
                if has(key):
                    sx, sy, sz = ctx.lib.size(key)
                    u = float(rng.uniform(u0 + sx / 2 + 0.3, max(u0 + sx / 2 + 0.31, u1 - sx / 2 - 0.3)))
                    z = 0.0 if "door" in key else float(rng.uniform(0.6, 1.0))
                    if clear_of_holes(u - sx / 2, u + sx / 2, z, z + sz) and mount(key, u, z):
                        n_shut += 1
        self.info.update({"fire_escapes": n_fe, "ac_units": n_ac, "pipes": n_pipe, "wall_lights": n_light,
                          "shutters": n_shut})

    def _fire_escape(self, rec, mount):
        """Platforms + railings at every upper floor, stairs between them and
        a drop ladder from the lowest; the whole stack checked clear."""
        ctx, rng = self.ctx, self.ctx.rng
        fr = rec["fr"]
        L = 4.8
        u = float(rng.uniform(rec["u0"] + L / 2 + 0.3, max(rec["u0"] + L / 2 + 0.31, rec["u1"] - L / 2 - 0.3)))
        dep = 1.4
        v0 = rec["v0"]
        corners = [fr.xy(u + du, v0 + dv) for du in (-L / 2, 0, L / 2) for dv in (0.1, dep / 2, dep)]
        if ctx.in_zone(corners[0], pad=2.0) and not all(ctx.free(q, 0.0, 0.3) for q in corners):
            return False
        g, f, n = rec["g"], rec["f"], rec["n"]
        levels = [g + k * f for k in range(n)]
        if not levels or levels[0] < 2.8:
            return False
        for k, z in enumerate(levels):
            mount("ph:modular_fire_escape#1", u, z - 0.77, depth_off=dep / 2 + 0.02)
            mount("ph:modular_fire_escape#3", u, z, depth_off=dep / 2 + 0.02)
            if k + 1 < len(levels):
                s = f / 4.0
                mount("ph:modular_fire_escape#5", u - 0.6, z, depth_off=dep / 2 + 0.02, s=s)
        mount("ph:modular_fire_escape#0", u + L / 2 - 0.5, levels[0] - 0.05 - 2.2, depth_off=dep - 0.15,
              s=0.5 * f / 3.0 + 0.2)
        # the space under it is not free for yard clutter (the ladder comes down)
        c = fr.xy(u, v0 + dep / 2)
        ctx.claim_rect(c, (L / 2, dep / 2), fr.yaw, kind="fire_escape")
        return True

    # --- ground clutter --------------------------------------------------------------------------------
    def clutter(self):
        ctx, rng, lk, pal = self.ctx, self.ctx.rng, self.ctx.looks, self.pal
        counts = {}
        dump_cols = [(0.08, 0.25, 0.12), (0.1, 0.18, 0.4), (0.35, 0.35, 0.35), (0.55, 0.25, 0.05), (0.4, 0.05, 0.04)]
        dmat = [lk.pbr(U.pick_key(ctx, U.PAINTED_METAL), 1.2, tint=[min(1, 1.8 * c + 0.05) for c in col],
                       bevel=0.015, dirt=0.7, jitter=0.1) for col in dump_cols[:2]]
        dmat += [lk.pbr(U.pick_key(ctx, U.RUSTY_PAINT), 1.2, bevel=0.015, dirt=0.6, jitter=0.1) for _ in range(3)]
        wood = lk.pbr(U.pick_key(ctx, U.WOOD + ["acg:Wood026", "acg:Wood027"]), 1.0, bevel=0.004, dirt=0.5)
        wcols = [(0.1, 0.35, 0.12), (0.1, 0.15, 0.45), (0.12, 0.12, 0.12), (0.5, 0.5, 0.52), (0.45, 0.3, 0.1)]
        spec = {"us_brick": {"dumpster": 2.5, "wheelie": 1.5, "cans": 2.0, "bags": 2.5, "pallets": 2.0, "boxes": 1.5,
                             "crates": 1.0, "barrels": 1.0, "tyres": 0.8, "barrier": 0.4, "car": 0.6, "misc": 1.2},
                "euro_old": {"dumpster": 0.6, "wheelie": 2.5, "cans": 0.8, "bags": 1.2, "pallets": 0.4, "boxes": 0.8,
                             "crates": 1.2, "barrels": 0.4, "plants": 3.0, "bikestand": 1.0, "misc": 1.5,
                             "car": 0.3},
                "service": {"dumpster": 2.0, "wheelie": 1.0, "cans": 0.6, "bags": 1.0, "pallets": 3.0, "boxes": 2.0,
                            "crates": 1.5, "barrels": 1.5, "tyres": 0.5, "barrier": 1.5, "cones": 1.5, "car": 0.8,
                            "misc": 1.5}}[self.flavour]
        spec["bikes"] = {"us_brick": 0.6, "euro_old": 1.5, "service": 0.4}[self.flavour]
        self.obj = {"dumpster": U.obj_pieces(ctx, U.OBJ_DUMPSTER), "bin": U.obj_pieces(ctx, U.OBJ_BIN),
                    "twowheel": U.obj_pieces(ctx, U.OBJ_TWO_WHEEL),
                    "car": U.obj_pieces(ctx, U.OBJ_CARS) + U.obj_pieces(ctx, U.OBJ_VANS),
                    "misc": U.obj_pieces(ctx, U.OBJ_GAS + U.OBJ_GENERATOR + U.OBJ_VENDING)}
        x0, y0, x1, y1 = ctx.bounds
        for kind, per in spec.items():
            n = U.draw_count(ctx, per)
            placed = 0
            for _ in range(n * 8):
                if placed >= n:
                    break
                xy, yaw = self._wall_spot()
                if xy is None:
                    continue
                ok = self._clutter_one(kind, xy, yaw, dmat, wood, wcols)
                placed += int(bool(ok))
            counts[kind] = placed
        self.info["clutter"] = counts

    def _wall_spot(self):
        """A point against a building front (clutter lives along the walls)."""
        ctx, rng = self.ctx, self.ctx.rng
        recs = [r for r in getattr(self, "fronts_rec", []) if not r["far"]]
        self._spot = None                  # (wall normal, distance from the wall) of a wall spot
        if not recs:
            return ctx.uniform_xy(pad=1.0), float(rng.uniform(0, 6.28))
        if self.court and rng.random() < 0.4:
            return ctx.uniform_xy(pad=1.5), float(rng.uniform(0, 6.28))
        r = recs[int(rng.integers(len(recs)))]
        u = float(rng.uniform(r["u0"] + 0.6, r["u1"] - 0.6))
        dv = float(rng.uniform(0.5, 1.1))
        xy = r["fr"].xy(u, r["v0"] + dv)
        if not ctx.in_zone(xy):
            return None, 0.0
        self._spot = (r["fr"].n, dv)
        return xy, r["fr"].yaw

    def _clutter_one(self, kind, xy, yaw, dmat, wood, wcols):
        ctx, rng, pal, lk = self.ctx, self.ctx.rng, self.pal, self.ctx.looks
        spot = getattr(self, "_spot", None)

        def push(depth):
            """``xy`` moved out from the wall so a thing ``depth`` deep (half, towards the
            wall) stands clear of it -- the buildings claim no footprint to test against."""
            if spot is None:
                return xy
            n, dv = spot
            return xy + n * max(0.0, depth + 0.05 - dv)

        def piece(keys, r=None, jitter_yaw=0.3, s=1.0):
            keys = [k for k in keys if k.split("#")[0] in ctx.lib.items]
            if not keys:
                return False
            k = keys[int(rng.integers(len(keys)))]
            sx, sy, _ = ctx.lib.size(k)
            rr = r if r is not None else 0.5 * math.hypot(sx, sy) * s
            p = push(rr)
            if not ctx.free(p, rr, 0.3):
                return False
            ctx.instance(k, p, yaw=yaw + float(rng.normal(0, jitter_yaw)), s=s)
            ctx.claim_disc(p, rr, kind="clutter")
            return True

        if kind == "dumpster":
            if self.obj.get("dumpster") and rng.random() < 0.4:
                return piece(self.obj["dumpster"], jitter_yaw=0.08)
            p = push(0.75)
            if not ctx.free(p, 1.1, 0.3):
                return False
            U.dumpster(ctx, dmat[int(rng.integers(len(dmat)))], p, yaw + float(rng.normal(0, 0.08)),
                       open_lid=rng.random() < 0.25)
            return True
        if kind == "wheelie":
            n = int(rng.integers(1, 4))
            body = lk.paint(wcols[int(rng.integers(len(wcols)))], rough=0.55, name="wheelie")
            lid = lk.paint(wcols[int(rng.integers(len(wcols)))], rough=0.5, name="wheelie_lid")
            ok = False
            t = np.array([math.cos(yaw), math.sin(yaw)])
            base = push(0.65)
            for i in range(n):
                p = base + t * 0.72 * i
                if ctx.free(p, 0.5, 0.25):
                    U.wheelie_bin(ctx, body, lid, p, yaw + math.pi + float(rng.normal(0, 0.15)),
                                  big=rng.random() < 0.2)
                    ok = True
            return ok
        if kind == "cans":
            return piece(["ph:metal_trash_can#0", "ph:metal_trash_can#2", "ph:barrel_stove#0"] + self.obj.get("bin", []),
                         jitter_yaw=3.0)
        if kind == "bikes":                    # bicycles and motorbikes against the wall
            ks = self.obj.get("twowheel", [])
            if not ks:
                return False
            k = ks[int(rng.integers(len(ks)))]
            y0_, L, Wd = U.long_yaw(ctx, k)
            p = push(Wd / 2)
            if not ctx.free_rect(p, (L / 2, Wd / 2), yaw, margin=0.3):
                return False
            ctx.instance(k, p, yaw=yaw + y0_ + (math.pi if rng.random() < 0.5 else 0.0))
            ctx.claim_rect(p, (L / 2, Wd / 2), yaw, kind="clutter")
            return True
        if kind == "bags":
            ok = False
            for i in range(int(rng.integers(1, 5))):
                p = xy + rng.normal(0, 0.35, 2)
                if "ph:trashbag" in ctx.lib.items and ctx.free(p, 0.3, 0.1):
                    ctx.instance("ph:trashbag#0", p, yaw=float(rng.uniform(0, 6.28)), s=float(rng.uniform(0.8, 1.2)))
                    ctx.claim_disc(p, 0.3, kind="clutter")
                    ok = True
            return ok
        if kind == "pallets":
            p = push(0.5)
            if not ctx.free(p, 0.8, 0.3):
                return False
            if rng.random() < 0.3:     # leaning against the wall
                U.pallet(ctx, wood, p, yaw, stack=1)
            else:
                U.pallet(ctx, wood, p, yaw + float(rng.normal(0, 0.2)), stack=int(rng.integers(1, 7)))
            return True
        if kind == "boxes":
            return piece(["ph:cardboard_box_01#0", "ph:compost_bags#1", "ph:plastic_crate_02#0",
                          "ph:plastic_crate_03#0"], jitter_yaw=1.0)
        if kind == "crates":
            return piece(["ph:wooden_crate_01#0", "ph:wooden_crate_02#0", "ph:old_military_crate#0",
                          "ph:wooden_military_crate#0", "ph:plastic_container#0"], jitter_yaw=0.5)
        if kind == "barrels":
            return piece(["ph:Barrel_01#0", "ph:Barrel_02#0", "ph:barrel_03#0", "ph:barrel_stove#0"],
                         jitter_yaw=3.0)
        if kind == "tyres":
            return piece(["ph:old_tyre#0", "ph:rusted_wheel_rim_01#0"], jitter_yaw=3.0)
        if kind == "barrier":
            return piece(["ph:concrete_road_barrier#0", "ph:concrete_road_barrier_02#0"], jitter_yaw=0.15)
        if kind == "car":                      # parked along the wall: a covered car or a real one
            ks = (["ph:covered_car#0"] if "ph:covered_car" in ctx.lib.items else []) + self.obj.get("car", [])
            if not ks:
                return False
            k = ks[int(rng.integers(len(ks)))]
            y0_, L, Wd = U.long_yaw(ctx, k)
            c = push(Wd / 2 + 0.2)
            if not ctx.free_rect(c, (L / 2 + 0.05, Wd / 2 + 0.05), yaw, margin=0.3):
                return False
            ctx.instance(k, c, yaw=yaw + y0_ + float(rng.normal(0, 0.04)) + (math.pi if rng.random() < 0.5 else 0.0))
            ctx.claim_rect(c, (L / 2 + 0.05, Wd / 2 + 0.05), yaw, kind="car")
            return True
        if kind == "plants":
            return piece(["ph:potted_plant_01#0", "ph:potted_plant_01#1", "ph:potted_plant_02#0",
                          "ph:planter_box_01#0", "ph:planter_box_03#0", "ph:pachira_aquatica_01#3",
                          "ph:anthurium_botany_01#0"], jitter_yaw=3.0)
        if kind == "bikestand":
            if not ctx.free(xy, 0.5, 0.3):
                return False
            U.bike_stand(ctx, pal.galv, xy, yaw)
            return True
        if kind == "cones":
            if not ctx.free(xy, 0.3, 0.3):
                return False
            U.cone(ctx, xy)
            return True
        if kind == "misc":
            return piece(["ph:hand_truck#0", "ph:utility_box_01#0", "ph:utility_box_02#0", "ph:wooden_ladder_02#0",
                          "ph:plastic_monobloc_chair_01#0", "ph:Sofa_01#0",
                          "ph:portable_generator#0", "ph:small_lpg_tank#0", "ph:worn_metal_rack#0"] +
                         self.obj.get("misc", []), jitter_yaw=0.6)
        return False

    # --- gaps: chain-link fences between buildings ----------------------------------------------------
    def fences(self):
        ctx, rng, lk, pal = self.ctx, self.ctx.rng, self.ctx.looks, self.pal
        mesh = lk.pbr(U.pick_key(ctx, ["acg:Fence001", "acg:Fence002", "acg:Fence003", "acg:Fence006"]), 1.0)
        n = 0
        for side in (1, -1):
            recs = sorted([r for r in self.fronts if r["side"] == side], key=lambda r: r["xa"])
            for r0, r1 in zip(recs, recs[1:]):
                if r1["xa"] - r0["xb"] < 0.8:
                    continue
                y = self.yc + side * (self.a_n if side > 0 else self.a_s)
                a, b = np.array([r0["xb"], y]), np.array([r1["xa"], y])
                if ctx.rect_meets_zone(np.minimum(a, b) - 0.5, np.maximum(a, b) + 0.5) and not (
                        U.seg_clear(ctx, a, b, clear=0.3, kinds=None, step=0.25) and
                        all(BC.footprint_distance(q, f) >= 0.3 for q in np.linspace(a, b, 24) for f in self._blk)):
                    continue
                h = float(rng.uniform(1.8, 2.6))
                ctx.panel(a, b, 0.0, h, mesh, tile_m=1.0, name="chainlink")
                for q in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 2.5) + 1)):
                    p = a + (b - a) * q
                    ctx.cylinder((p[0], p[1], 0.0), (p[0], p[1], h + 0.05), 0.03, pal.galv, name="fencepost",
                                 segments=8)
                ctx.cylinder((a[0], a[1], h), (b[0], b[1], h), 0.02, pal.galv, name="fencerail", segments=8)
                ctx.claim_rect((a + b) / 2, (np.linalg.norm(b - a) / 2, 0.1), 0.0, kind="fence")
                n += 1
        self.info["fences"] = n

    # --- overhead ----------------------------------------------------------------------------------
    def overhead(self):
        """Cables across the lane above the flight band at every level; at
        hard levels (``hang``) cables, laundry lines and hanging lamps come
        down into it (2.6-4.5 m)."""
        ctx, rng, lk, pal = self.ctx, self.ctx.rng, self.ctx.looks, self.pal
        cable = lk.paint((0.03, 0.03, 0.03), rough=0.5, name="cable")
        cloth = [lk.pbr(k, 0.5, tint=c) for k, c in
                 zip([U.pick_key(ctx, ["acg:Fabric004", "acg:Fabric026", "ph:denim_fabric"],
                                 fallback_cats=("fabric",)) for _ in range(3)],
                     [(0.9, 0.9, 0.9), (0.3, 0.4, 0.8), (0.8, 0.3, 0.3)]) if k] or \
            [lk.paint(c, rough=0.9, name="cloth") for c in [(0.9, 0.9, 0.88), (0.3, 0.4, 0.7), (0.7, 0.2, 0.2),
                                                           (0.9, 0.8, 0.3)]]
        recs_n = [r for r in self.fronts_rec if r["fr"].n[1] < 0 and not r["far"]] if hasattr(self, "fronts_rec") \
            else []
        recs_s = [r for r in self.fronts_rec if r["fr"].n[1] > 0 and not r["far"]] if hasattr(self, "fronts_rec") \
            else []
        x0, _, x1, _ = ctx.bounds

        def wall_y(recs, x, default):
            for r in recs:
                us = sorted((r["fr"].xy(r["u0"])[0], r["fr"].xy(r["u1"])[0]))
                if us[0] <= x <= us[1]:
                    return float(r["fr"].xy(r["u0"], r["v0"])[1]), r["H"]
            return default, 0.0

        made_high = 0
        for _ in range(int(rng.integers(2, 7))):            # high cables at every level
            xa = float(rng.uniform(x0 - 5, x1 + 5))
            xb = xa + float(rng.normal(0, 3))
            ya, ha = wall_y(recs_s, xa, None)
            yb, hb = wall_y(recs_n, xb, None)
            if ya is None or yb is None or min(ha, hb) < 6.0:
                continue
            za, zb = float(rng.uniform(5.2, min(ha, 8.0))), float(rng.uniform(5.2, min(hb, 8.0)))
            U.catenary(ctx, (xa, ya - 0.2, za), (xb, yb + 0.2, zb), float(rng.uniform(0.2, 0.5)), 0.012, cable,
                       name="cable_hi")
            made_high += 1
        n = ctx.count(float(ctx.knobs.get("hang", 0.0)))
        made, z_lo = 0, []
        for _ in range(n * 10):
            if made >= n:
                break
            kind = str(rng.choice(self.F["hang"]))
            xa = float(rng.uniform(x0, x1))
            xb = xa + float(rng.normal(0, 2.0))
            ya, ha = wall_y(recs_s, xa, None)
            yb, hb = wall_y(recs_n, xb, None)
            if ya is None or yb is None:
                continue
            za = float(rng.uniform(3.0, min(max(ha, 3.1), 4.6)))
            zb = float(rng.uniform(3.0, min(max(hb, 3.1), 4.6)))
            sag = float(rng.uniform(0.2, 0.6))
            a = np.array([xa, ya - 0.2, za])
            b = np.array([xb, yb + 0.2, zb])
            if not U.seg_clear(ctx, a, b, clear=0.4, kinds={"prim", "start", "goal", "fire_escape"}):
                continue
            if min(za, zb) - sag < 2.6:
                sag = max(0.05, min(za, zb) - 2.6)
            if kind == "cable":
                U.catenary(ctx, a, b, sag, 0.012, cable, name="cable")
                z_lo.append(min(za, zb) - sag - 0.012)          # (a bound: the sag is at mid-span)
            elif kind == "laundry":
                z_lo.append(min(za, zb) - sag - 0.006)
                U.catenary(ctx, a, b, sag, 0.006, cable, name="laundry", n=10)
                for i in range(2, 18, 2):
                    if rng.random() < 0.35:
                        continue
                    t = i / 20
                    p = a + (b - a) * t
                    p[2] -= sag * 4 * t * (1 - t)
                    w, h = float(rng.uniform(0.3, 0.8)), float(rng.uniform(0.35, 0.9))
                    yaw = math.atan2(b[1] - a[1], b[0] - a[0])
                    ctx.box((p[0], p[1], p[2] - h / 2), (w / 2, 0.01, h / 2), cloth[int(rng.integers(len(cloth)))],
                            yaw=yaw + math.pi / 2 * 0, name="laundry")
                    z_lo.append(p[2] - h)
            else:     # a lamp hanging from a span wire
                U.catenary(ctx, a, b, sag * 0.3, 0.01, cable, name="span")
                mid = (a + b) / 2
                mid[2] -= sag * 0.3
                ctx.cylinder(mid, mid - np.array([0, 0, 0.5]), 0.008, cable, name="span")
                shade = pal.metal
                ctx.cylinder(mid - np.array([0, 0, 0.5]), mid - np.array([0, 0, 0.72]), 0.22, shade, name="lamp",
                             segments=16)
                bulb = lk.paint((1.0, 0.9, 0.7), rough=0.3, name="bulb",
                                emission=(1.0, 0.85, 0.55, 8.0) if self.lit else None)
                U.ball(ctx, mid - np.array([0, 0, 0.74]), 0.06, bulb, name="lamp", n=8)
                z_lo.append(mid[2] - 0.80)
                if self.lit:
                    ctx.point_light((mid - np.array([0, 0, 0.85])).tolist(), 60.0, 0.1)
            made += 1
        self.info["overhead"] = {"high": made_high, "in_band": made}
        if z_lo:
            self.info["overhead"]["z_lowest"] = round(float(min(z_lo)), 2)


def build(ctx) -> dict:
    al = Alley(ctx)
    tev = {"dry": 0.35, "wet": 0.5, "dusk": 0.6, "snow": 0.1}[al.weather]
    sky = pick_sky(ctx, al.weather, tags=ALLEY_TAGS, target_ev=tev)
    al.ground()
    al.dress()
    if al.court:
        al.court_rows()
    else:
        al.rows()
        al.ends()
    al.wall_stuff()
    al.fences()
    al.clutter()
    al.overhead()
    sem = ["building backs on all four sides just outside the patch (no way round beyond the bounds)" if al.court
           else f"lane {al.info['lane_m']} m wide between building backs *inside* the patch; buildings step back "
                "round every obstacle, its openings' approaches and a 1.8 m way round each solid wall/slab end"]
    spanning = al.info.get("lane_spanning") or {}
    if spanning:
        sem.append("lane-spanning obstacles reach both building lines, so the buildings may close the way round "
                   "them: " + ", ".join(f"{n} {k}" for k, n in sorted(spanning.items())))
    if al.info.get("copings"):
        sem.append(f"wall tops capped (+8 cm, {al.info['copings']} copings)")
    ov = al.info.get("overhead", {})
    if ov.get("in_band"):
        sem.append(f"{ov['in_band']} cables/laundry/lamps across the lane, lowest at {ov['z_lowest']:.2f} m")
    return {"sky": sky, **al.info, "semantics_changed": sem}
