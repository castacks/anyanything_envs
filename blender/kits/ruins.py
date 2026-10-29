"""Stone ruins: an abandoned courtyard, cloister, castle bailey or temple
site built round the patch -- broken masonry walls with stepped, weathered
tops, arcades and colonnades, rubble and fallen drums, grass and shrubs
taking the paving back, dead trees, rolling hills far out.

What makes it read as a ruin rather than blocks on a plane:

* every wall is masonry scanned at its real size (1.5-3.5 m repeats) with
  bevelled edges, grime low down, and a *broken* top: courses lost
  unevenly, highest near the corners, loose stones askew on top; the
  patch's own walls get the same broken crown added above them where
  flying over them is out of the drone's reach anyway (top >= 3.9 m), so
  a 3 m room wall keeps its over-the-top route;
* coherent architecture per style: a cloister (arcade of round arches on
  piers round a paved garth), a castle bailey (Poly Haven's modular fort
  curtain walls, round tower and gate), an abbey (a tall nave wall
  pierced by arched windows, low foundations, a graveyard), or a
  sun-baked temple site (sandstone colonnade, fallen drums, succulents);
* the ground is zoned: paving that frays into grass/dirt at its edges,
  worn paths, snow some days; rubble where walls fell;
* the patch's poles become columns (base + capital when there is room),
  bars become timber beams propped at their ends, boxes masonry blocks,
  spheres boulders.

Levels: ``clutter`` scales the in-patch fragments, rubble and plants;
``hang`` > 0 adds timber shoring frames and props leaning on column stubs
across the flight band.

Contract: beams, props and dead-tree limbs are checked for free space along
their whole length (not at their ends), plants are shrunk to fit the free
space round them, and nothing dressing a primitive (plinths, capitals, crown
blocks, snow caps) creeps into the start / goal keep-out. The openings the
family implies beyond its slots -- a ring's interior and gap, the passages
past a short wall's ends, the approach under a floating box -- are claimed at
the level's keep-out before anything else is placed (``U.claim_openings``;
course.json ``chosen.openings``). What the dressing adds to the patch's own
collision geometry (crowns, plinths, capitals, snow caps, posts under box /
bar ends outside the bounds) and the structure round the patch are listed
in the result's ``semantics_changed`` (course.json ``chosen``; ``{prim,
kind, change, detail, z_top}``, ``prim`` None for the place as a whole).
"""

from __future__ import annotations

import math

import numpy as np

from kits import _ruins_util as U

NAME = "ruins"
INDOOR = False
DESCRIPTION = "stone ruins: broken walls, arcades, columns, rubble and overgrowth; cloister, castle, abbey or temple"

FAMILIES = {
    "grey": dict(
        wall=["castle_wall_slates", "castle_wall_varriation", "stone_wall", "stone_wall_02", "old_stone_wall",
              "old_stone_wall_02", "medieval_wall_01", "rustic_stone_wall", "rustic_stone_wall_02",
              "stacked_stone_wall", "mossy_stone_wall", "broken_wall", "stone_block_wall", "rock_wall_08",
              "rock_wall_13"],
        column=["acg:Travertine004", "acg:Travertine009", "marble_01", "worn_rock_natural_01", "medieval_blocks_03"],
        floor=["cobblestone_floor_01", "cobblestone_floor_03", "cobblestone_floor_04", "cobblestone_floor_05",
               "cobblestone_floor_07", "mossy_cobblestone", "grassy_cobblestone", "stone_floor",
               "monastery_stone_floor", "slate_floor_02", "cobblestone_large_01", "stone_pathway_02"]),
    "sand": dict(
        wall=["medieval_blocks_02", "medieval_blocks_05", "medieval_blocks_06", "large_sandstone_blocks",
              "large_sandstone_blocks_01", "sandstone_blocks_04", "sandstone_blocks_05", "sandstone_blocks_08",
              "sandstone_brick_wall_01", "old_sandstone_02", "mossy_sandstone", "yellow_stone_wall",
              "white_sandstone_blocks_02", "seaworn_sandstone_brick"],
        column=["old_sandstone_02", "acg:Travertine003", "acg:Travertine013", "marble_01", "sandstone_cracks"],
        floor=["red_sandstone_pavement", "cobblestone_floor_13", "stone_floor", "cobblestone_large_01",
               "patterned_cobblestone", "rock_tile_floor", "mixed_rock_tiles", "large_sandstone_blocks"]),
    "brick": dict(
        wall=["medieval_red_brick", "castle_brick_01", "castle_brick_02_red", "castle_brick_broken_06",
              "random_bricks_thick", "broken_brick_wall", "mixed_brick_wall", "worn_brick_wall", "church_bricks_03",
              "slumped_mortar_brick"],
        column=["acg:Travertine004", "old_sandstone_02", "marble_01", "castle_brick_02_white"],
        floor=["brick_pavement", "herringbone_brick", "worn_brick_floor", "cobblestone_floor_06", "mossy_brick_floor",
               "cobblestone_floor_04"]),
    "coral": dict(
        wall=["coral_fort_wall_01", "coral_fort_wall_02", "coral_fort_wall_03", "coral_stone_wall",
              "plastered_stone_wall", "plaster_stone_wall_01"],
        column=["marble_01", "acg:Travertine013", "white_sandstone_bricks", "old_sandstone_02"],
        floor=["stone_floor", "slate_floor_02", "rock_tile_floor_02", "cobblestone_floor_13", "stone_pathway_02"]),
    "dark": dict(
        wall=["rock_wall_09", "rock_wall_10", "rock_wall_14", "rock_wall_15", "rabdentse_ruins_wall",
              "japanese_stone_wall", "rock_wall_16"],
        column=["acg:Granite002A", "acg:Travertine009", "worn_rock_natural_01", "rock_boulder_dry"],
        floor=["cobblestone_floor_06", "cobblestone_floor_02", "slate_floor_02", "stone_pathway_02",
               "cobblestone_floor_08"]),
}

GROUND = {
    "temperate": dict(
        grass=["grass_ground", "forrest_ground_01", "grass_path_2", "grass_path_3", "aerial_grass_rock",
               "acg:Ground037", "acg:Ground003", "brown_mud_leaves_01", "forest_ground_04"],
        dirt=["park_dirt", "rocky_trail", "dry_ground_01", "brown_mud_dry", "rock_ground", "rocks_ground_02",
              "gravel_ground_01", "acg:Ground023", "forest_leaves_02"]),
    "arid": dict(
        grass=["dry_ground_01", "rocky_trail", "coral_ground_02", "sandy_gravel", "dirt_aerial_02", "rock_2",
               "dry_ground_rocks", "acg:Ground025", "red_laterite_soil_stones"],
        dirt=["sand_02", "rocky_trail_02", "coral_gravel", "gravelly_sand", "rock_ground_02", "acg:Ground049C"]),
    "snow": dict(
        grass=["snow_02", "snow_03", "snow_01", "snow_field_aerial"],
        dirt=["snow_05", "snow_03", "rocky_trail", "rock_ground"]),
}

# (not moss_wood: box-projected on a 0.2 m beam it reads as a lime-green stick)
WOOD = ["weathered_planks", "old_planks_02", "medieval_wood", "weathered_brown_planks", "wood_planks_grey",
        "acg:Wood027", "acg:Wood051"]
ROCK = ["rock_face", "mossy_rock", "rock_boulder_dry", "rock_boulder_cracked", "lichen_rock", "rock_surface",
        "acg:Rock030", "acg:Rock036", "acg:Rock049", "gray_rocks"]

STYLES = ("cloister", "castle", "abbey", "temple")
#: far tree clumps per climate: Objaverse oaks / broadleaves / cypress (7-140k
#: triangles, 9-14 m) beside the lighter Poly Haven plants
FAR_TREES = {
    "temperate": ["obj:3dc59560f2d24345bdbe65c44636453b", "obj:d841c3bcc5324daebee50f45619e05fc",
                  "obj:b5041d5648984ac2b833e2bde96fae85", "obj:ef48485b7c534acc905c0d9964c5fe7c",
                  "obj:249d8cb553a8469e9d645713f8e96ed1"],
    "arid": ["obj:249d8cb553a8469e9d645713f8e96ed1", "quiver_tree_01", "searsia_lucida", "quiver_tree_02",
             "obj:3dc59560f2d24345bdbe65c44636453b"],
    "snow": ["obj:249d8cb553a8469e9d645713f8e96ed1", "fir_sapling_medium"],
}
FAR_TREE_TRIS = 2_000_000           # instanced triangles of the far clumps, all trees together
#: the structure each style stands round the patch (outside its bounds, inside the cloud zone)
ENCLOSURE = {
    "cloister": "arcades 1.2-2.5 m outside the long sides (each 85 %), broken walls behind them and 1.5-4 m "
                "beyond both short ends",
    "castle": "curtain walls 2-4.5 m outside both long sides, a gatehouse wall 2-4.5 m beyond the far end, a "
              "broken wall 2-4 m behind the start",
    "abbey": "a 6-10 m nave wall 1-3 m outside one long side, a low broken wall 1-3.5 m outside the other",
    "temple": "colonnades 1.2-3.5 m outside both long sides, a podium wall 2-4.5 m beyond the far end",
}
#: patch walls get a broken crown only when flying over them is already out of
#: reach (top + body radius above the v7 ceiling 1.75 + 2.5 m)
CROWN_MIN_TOP = U.V7_REACH_TOP - 2 * U.V7_BODY_R


def _keys(lib, names):
    out = []
    for n in names:
        k = n if ":" in n else f"ph:{n}"
        if k in lib.items and "diff" in lib.items[k].get("maps", {}):
            out.append(k)
    return out


class Ruin:
    """Materials + builders shared by the styles."""

    def __init__(self, ctx, family, climate, weather):
        self.ctx, self.rng, self.lk, self.lib = ctx, ctx.rng, ctx.looks, ctx.lib
        rng, lk, lib = self.rng, self.lk, self.lib
        fam = FAMILIES[family]
        walls = _keys(lib, fam["wall"]) or lib.materials(cats_any={"brick", "wall"})
        self.wall_key = U.pick(rng, walls)
        self.wall2_key = U.pick(rng, walls)
        dirt = float(rng.uniform(0.25, 0.55))
        self.wall = lk.pbr(self.wall_key, None, bevel=0.03, dirt=dirt, vary=0.25, jitter=0.07)
        self.wall2 = lk.pbr(self.wall2_key, None, bevel=0.03, dirt=dirt, vary=0.25, jitter=0.07)
        # age: damp foot, rain streaks, lichen (greener in a temperate climate)
        moss_rgb = (0.14, 0.17, 0.06) if climate == "temperate" else (0.32, 0.29, 0.22)
        lich = float(rng.uniform(0.15, 0.55)) if climate == "temperate" else float(rng.uniform(0.0, 0.2))
        self.age = dict(moss=moss_rgb, damp=float(rng.uniform(0.25, 0.6)), streaks=float(rng.uniform(0.3, 0.9)),
                        lichen=lich)
        for m in dict.fromkeys([self.wall, self.wall2]):   # ordered: a set's order is per run
            U.weather(m, rng, **self.age)
        self.col_key = U.pick(rng, _keys(lib, fam["column"]) or walls)
        self.colmat = lk.pbr(self.col_key, 1.6, bevel=0.02, dirt=dirt * 0.8, vary=0.15, jitter=0.05)
        U.weather(self.colmat, rng, **{**self.age, "damp": self.age["damp"] * 0.6})
        self.floor_keys = _keys(lib, fam["floor"]) or lib.materials(cats_any={"cobblestone"})
        self.wood = lk.pbr(U.pick(rng, _keys(lib, WOOD)), None, bevel=0.01, dirt=0.4, jitter=0.1)
        self.rock = lk.pbr(U.pick(rng, _keys(lib, ROCK)), 2.5, bevel=0.0, dirt=0.25)
        hk = _keys(lib, ["rocks_ground_02", "rocky_trail", "rock_ground", "gravel_ground_01", "brown_mud_rocks_01",
                          "rocks_ground_01", "dry_ground_rocks", "rocks_ground_08"])
        if climate == "arid":
            hk = _keys(lib, ["rocky_trail_02", "dry_ground_rocks", "sandy_gravel", "rock_2"]) or hk
        self.heap = lk.pbr(U.pick(rng, hk), 1.2, vary=0.2, jitter=0.1)
        # the patch's poles: monolithic shafts or built masonry piers
        self.pole_masonry = rng.random() < 0.4
        self.bark = lk.pbr(U.pick(rng, lib.materials(cats_any={"bark"})), 1.2, dirt=0.3, jitter=0.1)
        self.snow = None
        if weather == "snow":
            sk = _keys(lib, ["snow_02", "snow_03", "acg:Snow010A"])
            self.snow = lk.pbr(U.pick(rng, sk), 2.0, vary=0.05, jitter=0.0) if sk else None
            if self.snow is not None:
                self.heap = self.snow
        moss = _keys(lib, ["acg:Moss001", "acg:Moss002", "acg:Moss003", "acg:Moss004"])
        self.moss = lk.pbr(U.pick(rng, moss), 1.2, vary=0.25, jitter=0.1) if moss and climate == "temperate" else None
        # leaf-card ivy was tried: the leaf scans render as near-black blotches on a wall, so none
        self.ivy = None
        self.climate, self.weather, self.family = climate, weather, family
        self.n_walls = 0
        self.sem: list = []             # semantics_changed entries (``_dress``)
        self.openings: list = []        # the family's implied openings, claimed (``U.claim_openings``)

    # --- masonry ---------------------------------------------------------------------------

    def wall_run(self, p0, p1, thick, h_hi, h_lo, mat=None, coll="structure", claim=True, check=True,
                 margin=0.35, ends_high=True, ivy=True):
        """A broken wall p0 -> p1, built only where it is free (split round
        whatever is in the way). Returns the number of pieces built."""
        ctx = self.ctx
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = float(np.linalg.norm(p1 - p0))
        if L < 0.8:
            return 0
        d = (p1 - p0) / L
        yaw = math.atan2(d[1], d[0])
        # walk along, collecting free stretches
        spans, cur = [], None
        step = 0.4
        for u in np.arange(0, L + 1e-6, step):
            xy = p0 + d * u
            ok = (not check) or ctx.free(xy, thick / 2, margin)
            if ok and cur is None:
                cur = [u, u]
            elif ok:
                cur[1] = u
            elif cur is not None:
                spans.append(cur)
                cur = None
        if cur is not None:
            spans.append(cur)
        built = 0
        for a, b in spans:
            if b - a < 0.8:
                continue
            q0, q1 = p0 + d * a, p0 + d * b
            parts = U.Parts()
            ws, hs = U.broken_wall(ctx, parts, q0, q1, thick, h_hi, h_lo, rng=self.rng, ends_high=ends_high)
            parts.emit(ctx, mat or (self.wall if self.rng.random() < 0.8 else self.wall2), name="ruin_wall",
                       coll=coll)
            c = (q0 + q1) / 2
            if claim:
                ctx.claim_rect(c, ((b - a) / 2 + 0.1, thick / 2 + 0.1), yaw, kind="ruin_wall")
            self._toppings(q0, q1, thick, ws, hs, coll)
            if ivy:
                self.ivy_on(q0, q1, thick, float(np.max(hs)), coll)
            built += 1
            self.n_walls += 1
        return built

    def _toppings(self, q0, q1, thick, ws, hs, coll):
        mat = self.snow
        if mat is None:
            return
        ctx = self.ctx
        L = float(np.linalg.norm(q1 - q0))
        d = (q1 - q0) / max(L, 1e-9)
        yaw = math.atan2(d[1], d[0])
        parts = U.Parts()
        u = 0.0
        t = 0.05 if mat is self.snow else 0.03
        for w, h in zip(ws, hs):
            c = q0 + d * (u + w / 2)
            if self.snow is not None or self.rng.random() < 0.6:
                parts.box((c[0], c[1], h + t / 2), (w / 2 * 0.98, thick / 2 + 0.015, t / 2), yaw=yaw)
            u += w
        parts.emit(ctx, mat, name="wall_cap", coll=coll)

    def ivy_on(self, q0, q1, thick, h, coll, p=0.55):
        """Leaf cards hanging down a wall face from its top."""
        if self.ivy is None or self.rng.random() > p or h < 1.2:
            return
        ctx, rng = self.ctx, self.rng
        L = float(np.linalg.norm(q1 - q0))
        d = (q1 - q0) / max(L, 1e-9)
        nrm = np.array([-d[1], d[0]])
        for _ in range(int(rng.integers(1, 3))):
            w = float(min(L * 0.8, rng.uniform(1.0, 2.8)))
            u0 = float(rng.uniform(0, max(0.01, L - w)))
            side = 1 if rng.random() < 0.5 else -1
            off = nrm * side * (thick / 2 + 0.025)
            a = q0 + d * u0 + off
            b = q0 + d * (u0 + w) + off
            top = h * float(rng.uniform(0.55, 0.9))
            bot = max(0.0, top - float(rng.uniform(1.0, 2.6)))
            ctx.panel(a, b, bot, top, self.ivy, tile_m=1.0, coll=coll, name="ivy")

    def column(self, xy, r, h, capital=True, base=True, coll="structure", claim=True, broken=False):
        ctx = self.ctx
        x, y = float(xy[0]), float(xy[1])
        z0 = 0.0
        parts = U.Parts()
        if base:
            parts.box((x, y, 0.14), (r + 0.14, r + 0.14, 0.14))
            parts.frustum((x, y, 0.28), (x, y, 0.4), r + 0.08, r + 0.02, seg=20)
            z0 = 0.4
        top = h - (0.32 if capital and not broken else 0.0)
        if broken:
            # a broken shaft: slanted top
            self._drums(parts, x, y, z0, top, r)
            ang = self.rng.uniform(0, 6.3)
            parts.box((x, y, top + 0.02), (r * 0.8, r * 0.55, 0.06), R=U.rot(ang, 0.25, 0))
        else:
            self._drums(parts, x, y, z0, top, r)
        if capital and not broken:
            parts.frustum((x, y, top), (x, y, top + 0.14), r * 0.9, r + 0.1, seg=20)
            parts.box((x, y, top + 0.23), (r + 0.16, r + 0.16, 0.09))
        parts.emit(ctx, self.colmat, name="column", coll=coll)
        if claim:
            ctx.claim_disc((x, y), r + (0.16 if base or capital else 0.02), kind="column")

    def _drums(self, parts, x, y, z0, z1, r):
        """A shaft of stacked drums, each a touch off-round and off-axis."""
        rng = self.rng
        z = z0
        while z < z1 - 1e-3:
            h = min(float(rng.uniform(0.55, 1.1)), z1 - z)
            if z1 - z - h < 0.25:
                h = z1 - z
            rr = r * (1 - 0.1 * (z - z0) / max(z1 - z0, 1e-3)) + float(rng.normal(0, 0.006))
            o = rng.normal(0, 0.008, 2)
            parts.frustum((x + o[0], y + o[1], z), (x + o[0], y + o[1], z + h), rr, rr * 0.995, seg=20,
                          twist=float(rng.uniform(0, 1)))
            z += h

    def arcade(self, a, d, n_bays, ri, pw, zs, depth, coll="structure", decay=0.3, upper=0.0):
        """A row of round arches on square piers from ``a`` along ``d``."""
        ctx, rng = self.ctx, self.rng
        a, d = np.asarray(a, float), np.asarray(d, float)
        yaw = math.atan2(d[1], d[0])
        bay = 2 * ri + pw
        built = 0
        parts = U.Parts()
        piers = []
        for i in range(n_bays + 1):
            pc = a + d * (i * bay)
            piers.append(pc)
        ok = [ctx.free(pc, pw * 0.75, 0.3) for pc in piers]
        free_bay = [ok[i] and ok[i + 1] and ctx.free_rect(a + d * (i * bay + bay / 2), (bay / 2, depth / 2), yaw,
                                                          margin=0.25) for i in range(n_bays)]
        pier_h = [0.0] * (n_bays + 1)
        for i in range(n_bays):
            if not free_bay[i]:
                continue
            mid = a + d * (i * bay + bay / 2)
            state = rng.random()
            if state < decay * 0.5:          # collapsed: piers only, rubble below
                hpier = zs * float(rng.uniform(0.35, 0.9))
                pier_h[i], pier_h[i + 1] = max(pier_h[i], hpier), max(pier_h[i + 1], hpier)
                heap = U.Parts()
                U.rubble(ctx, parts, mid, 1.2, int(rng.integers(6, 12)), rng=rng, height=0.45, base=heap)
                heap.emit(ctx, self.heap, name="heap", coll=coll, smooth=True)
            else:
                pier_h[i] = pier_h[i + 1] = zs
                jag = None if state > decay else list(rng.uniform(0.0, 0.6, 24))
                U.arch_block(parts, mid, yaw, ri, ri + pw / 2, zs, depth, top=float(rng.uniform(0.3, 0.6)) + upper,
                             jag=jag)
            ctx.claim_rect(mid, (bay / 2 + 0.1, depth / 2 + 0.1), yaw, kind="arcade")
            built += 1
        # each pier once, as tall as the tallest bay beside it needs: two bays sharing a pier used
        # to build it twice, and the coincident faces rendered black under the bevel shader
        for pc, h in zip(piers, pier_h):
            if h > 0:
                parts.box((pc[0], pc[1], h / 2), (pw / 2, depth / 2, h / 2), yaw=yaw)
        parts.emit(ctx, self.wall, name="arcade", coll=coll)
        return built

    def window_wall(self, a, d, L, H, thick, coll="structure", sill=1.2):
        """A tall nave/hall wall pierced by arched windows, top broken."""
        ctx, rng = self.ctx, self.rng
        a, d = np.asarray(a, float), np.asarray(d, float)
        yaw = math.atan2(d[1], d[0])
        c = a + d * L / 2
        if not ctx.free_rect(c, (L / 2, thick / 2), yaw, margin=0.3):
            return 0
        ri = float(rng.uniform(0.45, 0.8))
        pw = float(rng.uniform(1.2, 2.2))
        jamb = float(rng.uniform(1.4, 2.4))
        zs = sill + jamb
        bay = 2 * ri + pw
        n = max(1, int((L - pw) // bay))
        L2 = n * bay + pw
        a2 = a + d * (L - L2) / 2
        parts = U.Parts()
        top_arch = zs + ri + pw / 2 + 0.35
        for i in range(n + 1):
            pc = a2 + d * (pw / 2 + i * bay)
            parts.box((pc[0], pc[1], zs / 2), (pw / 2, thick / 2, zs / 2), yaw=yaw)
        for i in range(n):
            mid = a2 + d * (pw + ri + i * bay)
            parts.box((mid[0], mid[1], sill / 2), (ri, thick / 2, sill / 2), yaw=yaw)
            U.arch_block(parts, mid, yaw, ri, ri + pw / 2, zs, thick, top=0.35)
        # the wall above the windows, broken
        if H > top_arch + 0.4:
            ws, hs = U.broken_profile(rng, L2, H - top_arch, 0.3, course=0.35)
            u = 0.0
            for w, h in zip(ws, hs):
                cc = a2 + d * (u + w / 2)
                parts.box((cc[0], cc[1], top_arch + h / 2), (w / 2, thick / 2, h / 2), yaw=yaw)
                u += w
        # ends of the wall: extra buttresses
        for s in (0, 1):
            pc = a2 + d * (s * L2) + np.array([-d[1], d[0]]) * (thick / 2 + 0.35)
            hb = H * float(rng.uniform(0.4, 0.8))
            parts.box((pc[0], pc[1], hb / 2), (0.45, 0.35, hb / 2), yaw=yaw)
        parts.emit(ctx, self.wall, name="nave_wall", coll=coll)
        ctx.claim_rect(c, (L / 2 + 0.2, thick / 2 + 0.8), yaw, kind="nave_wall")
        return n

    def window_wall_spans(self, a, d, L, H, thick):
        """``window_wall`` on every free stretch (>= 5 m) of the line."""
        ctx = self.ctx
        a, d = np.asarray(a, float), np.asarray(d, float)
        spans, cur = [], None
        for u in np.arange(0, L + 1e-6, 0.5):
            ok = ctx.free(a + d * u, thick / 2 + 0.8, 0.3)
            if ok and cur is None:
                cur = [u, u]
            elif ok:
                cur[1] = u
            elif cur is not None:
                spans.append(cur)
                cur = None
        if cur is not None:
            spans.append(cur)
        n = 0
        for u0, u1 in spans:
            if u1 - u0 >= 5.0:
                n += self.window_wall(a + d * u0, d, u1 - u0, H, thick)
        return n

    def pile(self, xy, R, n, coll="scatter", claim=True):
        parts, base = U.Parts(), U.Parts()
        U.rubble(self.ctx, parts, xy, R, int(n * 1.6), rng=self.rng, height=float(self.rng.uniform(0.25, 0.6)),
                 base=base)
        parts.emit(self.ctx, self.wall if self.rng.random() < 0.6 else self.wall2, name="rubble", coll=coll)
        base.emit(self.ctx, self.heap, name="heap", coll=coll, smooth=True)
        if claim:
            self.ctx.claim_disc(xy, R + 0.1, kind="rubble")


# --- the build ---------------------------------------------------------------------------------

def _sky(ctx, style, climate, weather):
    lib, rng, lk = ctx.lib, ctx.rng, ctx.looks
    none = {"night", "indoor"}
    if weather == "snow":
        keys = lib.hdris(cats_none=none, tags_any={"snow"})
    elif climate == "arid":
        keys = lib.hdris(cats_all={"outdoor"}, cats_none=none | {"urban"},
                         tags_any={"desert", "arid", "veld", "dry grass", "dry", "rocky", "collection: namaqualand"})
    else:
        keys = lib.hdris(cats_all={"outdoor", "nature"}, cats_none=none | {"urban"},
                         tags_any={"hill", "mountain", "field", "valley", "countryside", "meadow", "hilltop", "rock",
                                   "ruin", "castle", "grass"})
        named = [k for k in ("ph:zavelstein", "ph:roofless_ruins", "ph:smelting_tower_01", "ph:smelting_tower_02",
                             "ph:abandoned_church", "ph:teutonic_castle_moat", "ph:colosseum", "ph:cannon",
                             "ph:gothic_manor_01") if k in lib.items]
        if named and rng.random() < 0.25:
            keys = named
        if weather == "overcast":
            oc = [k for k in keys if "overcast" in lib.items[k].get("categories", [])]
            keys = oc or keys
        elif weather == "sunset":
            ss = [k for k in keys if "sunrise-sunset" in lib.items[k].get("categories", [])]
            keys = ss or keys
        else:
            day = [k for k in keys if "sunrise-sunset" not in lib.items[k].get("categories", [])]
            keys = day or keys
    return lk.sky(lib.pick(rng, keys))


def _ground(ctx, R, style, climate, weather, paved_rects, path):
    """Zoned ground: grass/dirt base, paving in ``paved_rects`` fraying at
    the edges, a worn path along ``path`` (a strip rect)."""
    lib, rng, lk = ctx.lib, ctx.rng, ctx.looks
    G = GROUND["snow" if weather == "snow" else climate]
    grass = _keys(lib, G["grass"])
    dirt = _keys(lib, G["dirt"])
    base = [(U.pick(rng, grass), None), (U.pick(rng, dirt), None), (U.pick(rng, grass), None)]
    floor_key = U.pick(rng, R.floor_keys)
    layers = []
    fray = float(rng.uniform(0.6, 1.6))
    off = rng.uniform(-80, 80, 3)
    for (c, half) in paved_rects:
        def mask(nk, c=c, half=half):
            dist = nk.rect_dist(c, half)
            nz = nk.noise(0.45, detail=6, rough=0.65, offset=off)
            nz2 = nk.noise(2.5, detail=3, offset=off + 3)
            dd = nk.math("ADD", dist, nk.math("MULTIPLY", nk.math("SUBTRACT", nz, 0.5), 7.0 * fray))
            dd = nk.math("ADD", dd, nk.math("MULTIPLY", nk.math("SUBTRACT", nz2, 0.5), 1.2))
            inside = nk.mrange(dd, 0.12, -0.12)
            # grass reclaiming the paving inside too
            holes = nk.mrange(nk.noise(0.18, detail=6, offset=off + 17), 0.62, 0.7)
            return nk.math("MULTIPLY", inside, nk.math("SUBTRACT", 1.0, holes), clamp=True)
        layers.append({"key": floor_key, "tile": None, "mask": mask})
    if path is not None and weather != "snow":
        pk = U.pick(rng, dirt)
        (c, half) = path

        def pmask(nk, c=c, half=half):
            dist = nk.rect_dist(c, half)
            nz = nk.noise(0.6, detail=4, offset=off + 5)
            dd = nk.math("ADD", dist, nk.math("MULTIPLY", nk.math("SUBTRACT", nz, 0.5), 1.2))
            return nk.mrange(dd, 0.3, -0.3)
        layers.append({"key": pk, "tile": None, "mask": pmask})
    wet = float(rng.choice([0.0, 0.0, 0.0, 0.3, 0.6])) if weather in ("overcast",) else 0.0
    m = U.zoned_ground(lk, base, layers, puddles=wet, blend_scale=float(rng.uniform(6, 12)), rng=rng)
    ctx.ground(m)
    return {"base": [b[0] for b in base], "floor": floor_key, "wet": wet}


def _dress(ctx, R, climate):
    """Dress the patch primitives and add the trims that make them ruins."""
    rng = ctx.rng
    spec = ctx.spec
    import build_course as BC  # noqa: PLC0415 (bpy side)
    bar_mat = R.wood if rng.random() < 0.8 else R.colmat
    ctx.dress_primitives({"wall": R.wall, "box": R.wall if rng.random() < 0.6 else R.wall2,
                          "pole": R.wall2 if R.pole_masonry else R.colmat,
                          "bar": bar_mat, "rock": R.rock})
    R.openings = U.claim_openings(ctx)
    x0, y0, x1, y1 = ctx.bounds
    poles = [p for p in spec["primitives"] if p["kind"] == "cyl_v"]
    pc = np.array([p["c"] for p in poles]) if poles else np.zeros((0, 2))
    trims = U.Parts()
    snow = U.Parts()
    n_trim = 0
    sem = {}                        # what the dressing adds to the patch's own collision geometry

    def note(i, kind, change, detail, z=None):
        e = sem.setdefault((i, change), {"prim": i, "kind": kind, "change": change, "detail": detail})
        if z is not None:
            e["z_top"] = round(max(e.get("z_top", -1e9), float(z)), 3)

    for i, p in enumerate(spec["primitives"]):
        k = p["kind"]
        if k == "cyl_v":
            r = float(p["r"])
            c = np.asarray(p["c"], float)
            others = pc[np.linalg.norm(pc - c, axis=1) > 1e-6] if len(pc) else pc
            gap = float(np.min(np.linalg.norm(others - c, axis=1) - r)) if len(others) else 9.0
            if not U.clear_of_ends(ctx, c, r + 0.4):
                continue                # a plinth/capital must not creep into the start/goal keep-out
            if r >= 0.14 and gap > r + 0.55:
                # base plinth and capital (the shaft is the primitive)
                trims.box((c[0], c[1], 0.12), (r + 0.13, r + 0.13, 0.12))
                trims.frustum((c[0], c[1], 0.24), (c[0], c[1], 0.34), r + 0.08, r + 0.005, seg=20)
                note(i, k, "plinth", "square plinth r + 0.13 m, z 0-0.34 m", 0.34)
                z1 = float(p["z1"])
                if z1 < 6.0:
                    trims.frustum((c[0], c[1], z1 - 0.12), (c[0], c[1], z1), r + 0.005, r + 0.09, seg=20)
                    trims.box((c[0], c[1], z1 + 0.08), (r + 0.15, r + 0.15, 0.08))
                    note(i, k, "capital", f"capital r + 0.15 m from z {z1 - 0.12:.2f} m", z1 + 0.16)
                    if R.snow is not None:
                        snow.box((c[0], c[1], z1 + 0.19), (r + 0.14, r + 0.14, 0.03))
                n_trim += 1
            elif r >= 0.1:
                z1 = float(p["z1"])
                trims.frustum((c[0], c[1], z1), (c[0], c[1], z1 + 0.06), r + 0.03, r + 0.03, seg=16)
                note(i, k, "cap_ring", "cap ring r + 0.03 m on the top", z1 + 0.06)
        elif k == "wall":
            # a broken crown only where flying over the wall is already out of reach (top + body
            # radius above the v7 ceiling): a crown on a low wall (room, 3 m) would close its
            # over-the-top route
            crown = float(p["z1"]) >= CROWN_MIN_TOP
            for (cc, h, Rm) in BC.wall_pieces(p):
                top = cc[2] + h[2]
                if abs(top - float(p["z1"])) > 1e-3 or h[0] < 0.4:
                    continue
                d = Rm[:2, 0]
                yaw = math.atan2(d[1], d[0])
                L = 2 * h[0]
                thick = 2 * h[1]
                # (crown blocks and snow caps stay out of the start / goal keep-out, as the pole trims
                # do: a wall standing in it gains nothing there)
                if crown:
                    ws, hs = U.broken_profile(rng, L, float(rng.uniform(0.3, 1.0)), 0.0, course=0.3, ends_high=False)
                    u = -L / 2
                    for w, hh in zip(ws, hs):
                        if hh >= 0.29 and rng.random() < 0.75:
                            q = np.asarray(cc[:2]) + d * (u + w / 2)
                            if U.rect_clear_of_ends(ctx, q, (w / 2, thick / 2), yaw):
                                trims.box((q[0], q[1], top + hh / 2), (w / 2, thick / 2, hh / 2), yaw=yaw)
                                note(i, k, "crown", "broken crown above the top (over the wall already out of "
                                     "v7 reach)", top + hh)
                        u += w
                if R.snow is not None and U.rect_clear_of_ends(ctx, cc, (h[0], h[1] + 0.02), yaw):
                    snow.box((cc[0], cc[1], top + 0.03), (h[0], h[1] + 0.02, 0.03), yaw=yaw)
                    note(i, k, "snow_cap", "snow +6 cm on the top", top + 0.06)
                n_trim += 1
        elif k == "box":
            c = np.asarray(p["c"], float)
            h = np.asarray(p["half"], float)
            Rm = BC.rpy_matrix(p["rpy"])
            bot = c[2] - h[2]
            ext = np.abs(Rm) @ h
            if bot > 0.4 and abs(p["rpy"][0]) < 1e-3 and abs(p["rpy"][1]) < 1e-3:
                # floating block: prop it where its ends lie outside the patch
                ax = 0 if ext[0] > ext[1] else 1
                for s in (-1, 1):
                    e = c[:2].copy()
                    e[ax] += s * (ext[ax] - 0.35)
                    if (e[1] < y0 - 0.2 or e[1] > y1 + 0.2 or e[0] < x0 - 0.2 or e[0] > x1 + 0.2):
                        w = min(0.35, ext[1 - ax])
                        hp = (0.35, w) if ax == 0 else (w, 0.35)
                        trims.box((e[0], e[1], bot / 2), (hp[0], hp[1], bot / 2))
                        note(i, k, "posts_under_ends", f"posts under the ends outside the bounds, z 0-{bot:.2f} m")
                        n_trim += 1
            # snow on the box's own top face (its footprint, tilt and yaw -- not the rotated box's
            # axis-aligned extents, which overhang a yawed box by up to 2x its area)
            if R.snow is not None and U.rect_clear_of_ends(ctx, c[:2], ext[:2], 0.0):
                sc, hs = U.top_cap(c, h, Rm, 0.03)
                snow.box(sc, hs, R=Rm)
                note(i, k, "snow_cap", "snow 6 cm on the top face (its own footprint)",
                     sc[2] + float(np.abs(Rm[2]) @ hs))
        elif k == "cyl_h":
            a, b = np.asarray(p["a"], float), np.asarray(p["b"], float)
            r = float(p["r"])
            for e, o in ((a, b), (b, a)):
                inside = x0 - 0.3 <= e[0] <= x1 + 0.3 and y0 - 0.3 <= e[1] <= y1 + 0.3
                if inside or e[2] < 0.5:
                    continue
                dxy = (e - o)[:2]
                dxy = dxy / max(np.linalg.norm(dxy), 1e-9)
                q = e[:2] - dxy * 0.15
                if min(np.linalg.norm(q - ctx.start), np.linalg.norm(q - ctx.goal)) < 2.0:
                    continue
                w = max(0.12, r * 1.4)
                trims.box((q[0], q[1], (e[2] - r) / 2), (w, w, (e[2] - r) / 2),
                          yaw=math.atan2(dxy[1], dxy[0]))
                ctx.claim_disc(q, w * 1.5, kind="bar_post")
                note(i, k, "posts_under_ends", f"posts under the ends outside the bounds, z 0-{e[2] - r:.2f} m")
                n_trim += 1
    trims.emit(ctx, R.wall, name="prim_trim", coll="structure")
    if R.snow is not None:
        snow.emit(ctx, R.snow, name="prim_snow", coll="structure")
    R.sem = list(sem.values())
    return n_trim


def _vegetation(ctx, R, climate, weather, n_mul=1.0):
    """Grass tufts, shrubs and flowers in and round the patch (in the zone
    they collide -- they are small)."""
    lib, rng = ctx.lib, ctx.rng
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    if climate == "temperate" and weather != "snow":
        tufts = lib.models(ids=["grass_medium_01", "grass_medium_02", "fern_02", "celandine_01", "dandelion_01",
                                "periwinkle_plant", "nettle_plant", "weed_plant_02"], h=(0.08, 0.6), max_tris=6000)
        shrubs = lib.models(ids=["shrub_02", "searsia_lucida", "othonna_cerarioides", "pachira_aquatica_01",
                                 "fir_sapling", "pine_sapling_small"], h=(0.7, 2.6), max_tris=160000)
    elif climate == "arid":
        tufts = lib.models(ids=["grass_medium_02", "cheiridopsis_succulent", "crystalline_iceplant", "flower_gazania",
                                "flower_ursinia", "wild_rooibos_bush", "dead_quiver_branch_01",
                                "dead_quiver_branch_02"], h=(0.05, 0.6), max_tris=30000)
        shrubs = lib.models(ids=["othonna_cerarioides", "searsia_lucida", "quiver_tree_02", "shrub_02",
                                 "didelta_spinosa"], h=(0.7, 2.6), max_tris=160000)
    else:  # snow: sparse dry grass and dead shrubs
        tufts = lib.models(ids=["grass_medium_02", "dry_branches_medium_01"], h=(0.05, 0.6), max_tris=30000)
        shrubs = lib.models(ids=["shrub_02", "fir_sapling", "pine_sapling_small"], h=(0.7, 2.6), max_tris=160000)
    out = {}
    out["tufts"] = sow(ctx, tufts, ctx.count(220 * n_mul), s=(1.0, 2.2))
    ns = int(ctx.count(7) * n_mul)
    out["shrubs"] = len(ctx.scatter(shrubs, ns, rect=(zx0 + 1, zy0 + 1, zx1 - 1, zy1 - 1), margin=0.6, s=(0.8, 1.2)))
    return out


def sow(ctx, keys, n, s=(0.9, 1.6), rect=None, near=1.0, p_far=0.3):
    """Small plants (no footprint claims): biased towards wall feet and
    obstacle edges (clearance < ``near``), kept off everything else -- a
    plant is shrunk (to half size at most) to stay out of every claim, the
    start/goal discs and the opening approaches included, else skipped."""
    rng = ctx.rng
    if not keys:
        return 0
    zx0, zy0, zx1, zy1 = rect or ctx.zone
    placed = 0
    for _ in range(n * 6):
        if placed >= n:
            break
        xy = np.array([rng.uniform(zx0, zx1), rng.uniform(zy0, zy1)])
        cl = ctx.clearance(xy)
        if cl < 0.08:
            continue
        if cl > near and rng.random() > p_far:
            continue
        k = ctx.lib.pick(rng, keys)
        sz = ctx.lib.size(k)
        sc = U.fit_scale(0.5 * math.hypot(sz[0], sz[1]), float(rng.uniform(*s)), cl, pad=0.03)
        if sc is None:
            continue
        ctx.instance(k, xy, 0.0, float(rng.uniform(0, 6.3)), sc, coll="scatter")
        placed += 1
    return placed


def _far(ctx, R, style, climate, weather):
    """Scenery beyond the zone: hills, tree clumps, more ruin."""
    lib, rng, lk = ctx.lib, ctx.rng, ctx.looks
    zx0, zy0, zx1, zy1 = ctx.zone
    cx, cy = (zx0 + zx1) / 2, (zy0 + zy1) / 2
    hk = U.pick(rng, _keys(lib, {"temperate": ["aerial_grass_rock", "aerial_rocks_04", "aerial_rocks_02",
                                               "rocky_terrain_02", "grass_ground"],
                                 "arid": ["dirt_aerial_02", "dirt_aerial_03", "aerial_rocks_01", "aerial_ground_rock",
                                          "rocky_terrain_03"],
                                 "snow": ["snow_field_aerial", "snow_03"]}["snow" if weather == "snow" else climate]))
    hill_m = lk.pbr(hk, float(rng.uniform(35, 60)), proj="box", vary=0.35, jitter=0.0, antitile=True)
    rad = max(zx1 - zx0, zy1 - zy0)
    U.heightfield_ring(ctx, (cx, cy), 260 + rad, 800 + rad, float(rng.uniform(12, 45)), hill_m, rng,
                       n_th=140, n_r=14)
    # tree clumps between the zone and the hills: full-size low-poly trees (Objaverse,
    # 7-140k triangles) so twenty of them cost what one 2M-triangle Poly Haven scan does
    trees = lib.models(ids=FAR_TREES["snow" if weather == "snow" else climate], h=(1.4, 20.0))
    use = [trees[int(i)] for i in rng.choice(len(trees), size=min(2, len(trees)), replace=False)] if trees else []
    n_tree, far_tris = 0, 0
    for _ in range(int(rng.integers(14, 30))):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.55, 1.6) * rad
        xy = np.array([cx + r * math.cos(a), cy + r * math.sin(a)])
        if ctx.in_zone(xy, pad=8.0) or not use:
            continue
        k = U.pick(rng, use)
        t = U.piece_tris(lib, k)
        if far_tris + t > FAR_TREE_TRIS:
            continue
        ctx.instance(k, xy, 0.0, rng.uniform(0, 6.3), float(rng.uniform(0.8, 1.3)), coll="backdrop")
        far_tris += t
        n_tree += 1
    # rock outcrops
    rocks = lib.models(ids=["namaqualand_cliff_01", "namaqualand_cliff_02", "rock_face_01", "mountainside",
                            "rock_moss_set_01"], max_tris=200_000)
    for _ in range(int(rng.integers(2, 6))):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(1.0, 2.5) * rad + 20
        xy = np.array([cx + r * math.cos(a), cy + r * math.sin(a)])
        if ctx.in_zone(xy, pad=15.0) or not rocks:
            continue
        ctx.instance(U.pick(rng, rocks), xy, -0.3, rng.uniform(0, 6.3), float(rng.uniform(1.5, 3.5)),
                     coll="backdrop")
    return {"trees": n_tree, "tree_tris": far_tris, "tree_kinds": use, "hill_scan": hk}


def _style_cloister(ctx, R):
    """Arcades along the long sides of the patch, a broken back wall behind
    each, a doorway wall at each end."""
    rng = ctx.rng
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    ri = float(rng.uniform(0.9, 1.5))
    pw = float(rng.uniform(0.6, 1.0))
    zs = float(rng.uniform(2.0, 3.0))
    depth = float(rng.uniform(0.6, 0.9))
    n = 0
    for side in (-1, 1):
        if rng.random() < 0.15:
            continue
        yl = (y1 + float(rng.uniform(1.2, 2.5))) if side > 0 else (y0 - float(rng.uniform(1.2, 2.5)))
        bay = 2 * ri + pw
        n_b = int((zx1 - zx0 - 4) // bay)
        a = np.array([zx0 + 2 + rng.uniform(0, bay / 2), yl])
        n += R.arcade(a, np.array([1.0, 0.0]), n_b, ri, pw, zs, depth, decay=float(rng.uniform(0.2, 0.5)))
        # back wall of the walk
        yb = yl + side * float(rng.uniform(2.6, 3.6))
        if abs(yb) < max(abs(zy0), abs(zy1)) + 3:
            R.wall_run((zx0 + 1, yb), (zx1 - 1, yb), 0.7, float(rng.uniform(3.5, 5.5)), 1.0)
    for xe in (x0 - float(rng.uniform(1.5, 4.0)), x1 + float(rng.uniform(1.5, 4.0))):
        R.wall_run((xe, zy0 + 1), (xe, zy1 - 1), 0.8, float(rng.uniform(3.0, 6.0)), 0.6)
    return {"bays": n}


def _style_castle(ctx, R):
    """Curtain walls from the modular fort kit round the bailey, a round
    tower at a corner, a gate at the far end; ruined buildings inside."""
    lib, rng = ctx.lib, ctx.rng
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    straight = [f"ph:modular_fort_01#{i}" for i in (6, 7, 13, 14, 18)]
    straight = [k for k in straight if k.split("#")[0] in lib.items]
    placed = 0
    if straight:
        for side in (-1, 1):
            k = U.pick(rng, straight)
            sx, sy, sz = lib.size(k)
            L, T = max(sx, sy), min(sx, sy)
            yl = (y1 + float(rng.uniform(2.0, 4.5)) + T / 2) if side > 0 else (y0 - float(rng.uniform(2.0, 4.5)) - T / 2)
            x = zx0 - 10.0
            while x < zx1 + 10.0:
                c = np.array([x + L / 2, yl])
                if ctx.free_rect(c, (L / 2 - 0.2, T / 2), 0.0, margin=0.3):
                    ctx.instance(k, c, 0.0, math.pi / 2 if sy > sx else 0.0, 1.0, coll="structure", claim=False)
                    # claim short of the ends so the next module can butt against this one
                    ctx.claim_rect(c, (L / 2 - 0.6, T / 2 + 0.2), 0.0, kind="fort")
                    placed += 1
                else:
                    # a gap: broken procedural stub instead
                    R.wall_run((x, yl), (x + L, yl), T * 0.7, float(rng.uniform(2.5, 5.0)), 0.4)
                x += L
        tower = "ph:modular_fort_01#0"
        if tower.split("#")[0] in lib.items:
            corner = np.array([zx1 + 6.0 if rng.random() < 0.6 else zx0 - 6.0,
                               zy1 + 6.0 if rng.random() < 0.5 else zy0 - 6.0])
            ctx.instance(tower, corner, 0.0, rng.uniform(0, 6.3), 1.0, coll="backdrop")
    # the far end: a gatehouse wall with a door or gate, broken walls beyond
    xg = x1 + float(rng.uniform(2.0, 4.5))
    gate_y = float(np.clip(ctx.goal[1] + rng.normal(0, 2.0), y0 + 2, y1 - 2))
    gate = U.pick(rng, ["ph:large_castle_door#0", "ph:large_iron_gate#0"])
    gw = 3.2
    if ctx.free_rect((xg, gate_y), (0.6, gw / 2), 0.0, margin=0.3):
        R.wall_run((xg, zy0 - 2), (xg, gate_y - gw / 2), 1.2, float(rng.uniform(4.5, 7.0)), 2.5, ends_high=True)
        R.wall_run((xg, gate_y + gw / 2), (xg, zy1 + 2), 1.2, float(rng.uniform(4.5, 7.0)), 2.5, ends_high=True)
        parts = U.Parts()
        U.arch_block(parts, (xg, gate_y), math.pi / 2, gw / 2 - 0.1, gw / 2 + 0.6, 2.4, 1.2, top=0.8)
        for s in (-1, 1):
            parts.box((xg, gate_y + s * (gw / 2 + 0.25), 1.2), (0.6, 0.35, 1.2))
        parts.emit(ctx, R.wall, name="gatehouse")
        if gate and gate.split("#")[0] in lib.items:
            ctx.instance(gate, (xg + 0.45, gate_y), 0.0, math.pi / 2, 0.95, coll="structure")
    else:
        R.wall_run((xg, zy0 - 2), (xg, zy1 + 2), 1.2, float(rng.uniform(4.0, 6.5)), 1.5)
    # behind the start
    R.wall_run((x0 - float(rng.uniform(2.0, 4.0)), zy0 + 1), (x0 - 3.0, zy1 - 1), 1.0,
               float(rng.uniform(3.0, 6.0)), 0.8)
    # castle props
    props = lib.models(ids=["cannon_01", "wooden_barrels_01", "wooden_crate_01", "wooden_crate_02",
                            "wooden_bucket_02", "wooden_military_crate", "stone_fire_pit"], h=(0.3, 1.2))
    ctx.scatter(props, ctx.count(4), margin=0.5)
    return {"fort_pieces": placed}


def _style_abbey(ctx, R):
    """A tall nave wall with windows along one side, low foundations of
    rooms inside, a gable end far along and a graveyard off to the side."""
    rng = ctx.rng
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    side = 1 if rng.random() < 0.5 else -1
    yl = (y1 + float(rng.uniform(1.0, 3.0))) if side > 0 else (y0 - float(rng.uniform(1.0, 3.0)))
    H = float(rng.uniform(6.0, 10.0))
    thick = float(rng.uniform(0.9, 1.3))
    n = R.window_wall_spans((zx0 + 2, yl), (1.0, 0.0), zx1 - zx0 - 4, H, thick)
    if not n:                       # blocked on that side: try the other
        side = -side
        yl = (y1 + float(rng.uniform(1.0, 3.0))) if side > 0 else (y0 - float(rng.uniform(1.0, 3.0)))
        n = R.window_wall_spans((zx0 + 2, yl), (1.0, 0.0), zx1 - zx0 - 4, H, thick)
    # opposite side: low foundations / a lower wall with gaps
    yo = (y0 - float(rng.uniform(1.0, 3.5))) if side > 0 else (y1 + float(rng.uniform(1.0, 3.5)))
    R.wall_run((zx0 + 1, yo), (zx1 - 1, yo), 0.9, float(rng.uniform(1.2, 3.0)), 0.3, ends_high=False)
    # gable end beyond the goal (partly outside the zone: a picture)
    xg = zx1 + float(rng.uniform(4, 14))
    parts = U.Parts()
    wg = abs(yo - yl) + 2
    yc = (yo + yl) / 2
    Hg = H + float(rng.uniform(2, 5))
    for i in range(10):
        f = i / 10
        hh = 2.5 + (Hg - 2.5) * (1 - abs(2 * f + 0.05 - 1))
        yy = yc - wg / 2 + wg * (f + 0.05)
        parts.box((xg, yy, hh / 2), (0.6, wg / 20, hh / 2))
    parts.emit(ctx, R.wall, name="gable", coll="backdrop")
    # graveyard beyond the low wall
    gs = U.Parts()
    ygy = yo - side * float(rng.uniform(2.0, 4.0))
    for i in range(int(rng.integers(8, 20))):
        xy = np.array([rng.uniform(zx0 + 2, zx1 - 2), ygy - side * rng.uniform(0, 5)])
        if not ctx.free(xy, 0.4, 0.3):
            continue
        h = float(rng.uniform(0.5, 1.1))
        gs.box((xy[0], xy[1], h / 2), (0.08, float(rng.uniform(0.22, 0.35)), h / 2),
               R=U.rot(float(rng.normal(0, 0.08)), float(rng.normal(0, 0.1)), float(rng.normal(0, 0.1))))
        ctx.claim_disc(xy, 0.4, kind="grave")
    gs.emit(ctx, R.colmat, name="headstones", coll="scatter")
    # statues on the site
    lib = ctx.lib
    st = lib.models(ids=["gothic_statue"])
    ctx.scatter(st, int(rng.integers(0, 2)), margin=0.8)
    return {"windows": n, "nave_h": round(H, 2)}


def _style_temple(ctx, R):
    """A colonnade (columns with an architrave in places) along the sides,
    a podium edge and fallen drums."""
    rng = ctx.rng
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    r = float(rng.uniform(0.3, 0.45))
    h = float(rng.uniform(4.0, 6.5))
    sp = float(rng.uniform(2.6, 3.6))
    n = 0
    arch = U.Parts()
    for side in (-1, 1):
        yl = (y1 + float(rng.uniform(1.2, 3.5))) if side > 0 else (y0 - float(rng.uniform(1.2, 3.5)))
        prev = None
        for x in np.arange(zx0 + 1.5, zx1 - 1.0, sp):
            xy = (x, yl)
            if not ctx.free(xy, r + 0.2, 0.3):
                prev = None
                continue
            s = rng.random()
            if s < 0.2:
                prev = None
                R.pile(xy, 1.0, int(rng.integers(4, 9)))
                continue
            broken = s < 0.45
            hh = h if not broken else h * float(rng.uniform(0.25, 0.8))
            R.column(xy, r, hh, capital=not broken, broken=broken)
            n += 1
            if not broken and prev is not None and rng.random() < 0.6:
                cxy = ((prev[0] + x) / 2, yl)
                # beams meet end to end (overlapping coplanar faces shadow each other black), and sit
                # clear of the capitals' abaci (the top of which is at h)
                zb = h + 0.303 + abs(float(rng.normal(0, 0.006)))
                arch.box((cxy[0], cxy[1], zb), (sp / 2 - 0.004, r + 0.12, 0.3))
            prev = xy if not broken else None
    arch.emit(ctx, R.colmat, name="architrave")
    # podium edge: a low step band along the far end
    for xe in (x1 + float(rng.uniform(2.0, 4.5)),):
        R.wall_run((xe, zy0 + 1), (xe, zy1 - 1), 1.1, float(rng.uniform(1.5, 4.0)), 0.5)
    return {"columns": n}


def _interior(ctx, R, style):
    """In-patch ruins: wall fragments, column stubs, drums, rubble, blocks."""
    rng = ctx.rng
    x0, y0, x1, y1 = ctx.bounds
    ko = float(ctx.knobs.get("keepout", 2.5))
    out = {"frag": 0, "stubs": 0, "piles": 0, "drums": 0, "blocks": 0, "dead_trees": 0}
    # wall fragments (L-shaped sometimes: the corner of a lost room)
    for _ in range(ctx.count(3.0)):
        for _t in range(40):
            c = ctx.uniform_xy(pad=1.5)
            L = float(rng.uniform(1.5, 4.5))
            yaw = float(rng.choice([0.0, math.pi / 2])) + float(rng.normal(0, 0.05))
            if not ctx.free_rect(c, (L / 2, 0.45), yaw, margin=0.5 * ko + 0.4):
                continue
            d = np.array([math.cos(yaw), math.sin(yaw)])
            hi = float(rng.uniform(0.8, 2.8))
            R.wall_run(c - d * L / 2, c + d * L / 2, float(rng.uniform(0.5, 0.8)), hi, 0.3, check=False)
            out["frag"] += 1
            break
    # column stubs
    for _ in range(ctx.count(3.0)):
        for _t in range(40):
            c = ctx.uniform_xy(pad=1.0)
            r = float(rng.uniform(0.25, 0.4))
            if not ctx.free(c, r + 0.2, 0.5 * ko + 0.3):
                continue
            R.column(c, r, float(rng.uniform(0.7, 3.2)), capital=False, broken=True)
            out["stubs"] += 1
            break
    # fallen drums
    for _ in range(ctx.count(2.0)):
        for _t in range(40):
            c = ctx.uniform_xy(pad=1.0)
            r = float(rng.uniform(0.25, 0.4))
            L = float(rng.uniform(0.5, 1.4))
            yaw = float(rng.uniform(0, math.pi))
            if not ctx.free_rect(c, (L / 2 + 0.1, r + 0.1), yaw, margin=0.4):
                continue
            d = np.array([math.cos(yaw), math.sin(yaw)])
            a = c - d * L / 2
            b = c + d * L / 2
            ctx.cylinder((a[0], a[1], r), (b[0], b[1], r), r, R.colmat, coll="scatter", name="drum", segments=20)
            ctx.claim_rect(c, (L / 2 + 0.1, r + 0.1), yaw, kind="drum")
            out["drums"] += 1
            break
    # rubble piles
    for _ in range(ctx.count(4.0)):
        for _t in range(40):
            c = ctx.uniform_xy(pad=0.8)
            Rr = float(rng.uniform(0.6, 1.4))
            if not ctx.free(c, Rr, 0.3):
                continue
            R.pile(c, Rr, int(rng.integers(6, 16)))
            out["piles"] += 1
            break
    # loose blocks
    blocks = U.Parts()
    for _ in range(ctx.count(8.0)):
        c = ctx.uniform_xy(pad=0.5)
        s = float(rng.uniform(0.3, 0.8))
        if not ctx.free(c, s * 0.7, 0.3):
            continue
        blocks.box((c[0], c[1], s * 0.3), (s / 2, s * 0.35, s * 0.3),
                   R=U.rot(float(rng.uniform(0, 6.3)), float(rng.normal(0, 0.12)), float(rng.normal(0, 0.12))))
        ctx.claim_disc(c, s * 0.7, kind="block")
        out["blocks"] += 1
    blocks.emit(ctx, R.wall2, name="blocks", coll="scatter")
    # a dead tree or two: the limbs spread well past the trunk, so every limb
    # vertex (not just the base) must be clear of the claims
    if R.climate != "arid" or rng.random() < 0.5:
        for _ in range(int(rng.integers(0, 3))):
            for _t in range(6):
                c = ctx.uniform_xy(ctx.zone, pad=2.0)
                if not ctx.free(c, 1.5, 0.5):
                    continue
                parts = U.Parts()
                U.dead_tree(parts, (c[0], c[1], 0.0), float(rng.uniform(4.0, 8.0)), float(rng.uniform(0.12, 0.25)),
                            rng)
                if not U.parts_free(ctx, parts, margin=0.3):
                    continue
                parts.emit(ctx, R.bark, name="dead_tree", coll="scatter")
                V = np.asarray(parts.V)
                ctx.claim_disc(c, max(1.2, float(np.hypot(V[:, 0] - c[0], V[:, 1] - c[1]).max()) * 0.6),
                               kind="dead_tree")
                out["dead_trees"] += 1
                break
    # boulders and statues
    rocks = ctx.lib.models(ids=["rock_moss_set_02", "namaqualand_boulder_05", "namaqualand_boulder_06",
                                "boulder_01"], max_tris=120000, h=(0.3, 1.4))
    ctx.scatter(rocks, ctx.count(2.0), margin=0.5)
    return out


def _hang(ctx, R):
    """L3: timber shoring frames and props leaning on column stubs, in the band."""
    rng = ctx.rng
    n = ctx.count(float(ctx.knobs.get("hang", 0.0)))
    built = 0
    for _ in range(n):
        for _t in range(60):
            a = ctx.uniform_xy(pad=1.0)
            ang = float(rng.uniform(0, math.pi))
            L = float(rng.uniform(2.5, 4.0))
            d = np.array([math.cos(ang), math.sin(ang)])
            b = a + d * L
            mid = (a + b) / 2
            if not (ctx.free(a, 0.25, 0.5) and ctx.free(b, 0.25, 0.5) and ctx.free(mid, 0.1, 0.6)):
                continue
            # the beam / prop spans a -> b in the band: clear along its length (start, goal, openings)
            if not U.seg_free(ctx, a - d * 0.2, b + d * 0.2, 0.12, 0.4):
                continue
            z = float(rng.uniform(2.0, 3.2))
            if rng.random() < 0.55:
                # shoring frame: two posts and a beam
                for p in (a, b):
                    ctx.box((p[0], p[1], z / 2), (0.1, 0.1, z / 2), R.wood, coll="scatter", name="shore_post",
                            claim=True)
                ctx.box((mid[0], mid[1], z + 0.1), (L / 2 + 0.2, 0.1, 0.1), R.wood, yaw=ang, coll="scatter",
                        name="shore_beam")
                ctx.claim_rect(mid, (L / 2 + 0.2, 0.25), ang, kind="shore")
            else:
                # a column stub with a timber leaning on it
                R.column(b, 0.28, z + 0.15, capital=False, broken=True, coll="scatter")
                ctx.cylinder((a[0], a[1], 0.08), (b[0] - d[0] * 0.3, b[1] - d[1] * 0.3, z), 0.08, R.wood,
                             coll="scatter", name="prop")
                ctx.claim_rect(mid, (L / 2, 0.3), ang, kind="prop")
            built += 1
            break
    return built


def build(ctx) -> dict:
    rng = ctx.rng
    x0, y0, x1, y1 = ctx.bounds
    style = str(STYLES[int(rng.integers(len(STYLES)))])
    if style == "temple":
        climate = "arid" if rng.random() < 0.7 else "temperate"
        family = str(rng.choice(["sand", "coral", "sand", "grey"]))
    elif style == "castle":
        climate = "temperate" if rng.random() < 0.85 else "arid"
        family = str(rng.choice(["grey", "grey", "sand", "dark", "brick", "coral"]))
    elif style == "abbey":
        climate = "temperate"
        family = str(rng.choice(["grey", "sand", "brick", "dark"]))
    else:
        climate = "temperate" if rng.random() < 0.8 else "arid"
        family = str(rng.choice(["grey", "sand", "brick", "coral", "dark"]))
    wr = rng.random()
    if climate == "temperate" and wr < 0.14:
        weather = "snow"
    elif wr < 0.4:
        weather = "overcast"
    elif wr < 0.55:
        weather = "sunset"
    else:
        weather = "clear"
    sky = _sky(ctx, style, climate, weather)
    R = Ruin(ctx, family, climate, weather)
    # the ground: paving over the patch (cloister/temple: most of it), a path
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
    paved = []
    if style in ("cloister", "temple"):
        paved.append(((xm, ym), (hw + rng.uniform(-1, 3), hh + rng.uniform(-1, 3))))
    elif style == "castle":
        if rng.random() < 0.6:
            paved.append(((xm, ym), (hw * rng.uniform(0.4, 0.9), hh * rng.uniform(0.3, 0.8))))
    else:
        for _ in range(int(rng.integers(1, 3))):
            c = ctx.uniform_xy(pad=2)
            paved.append(((c[0], c[1]), (rng.uniform(2, 6), rng.uniform(2, 5))))
    gy = float(ctx.goal[1])
    path = (((x0 + x1) / 2, gy / 2), ((x1 - x0) / 2 + 6, float(rng.uniform(0.6, 1.4)))) if rng.random() < 0.6 else None
    ginfo = _ground(ctx, R, style, climate, weather, paved, path)
    n_trim = _dress(ctx, R, climate)
    sinfo = {"cloister": _style_cloister, "castle": _style_castle, "abbey": _style_abbey,
             "temple": _style_temple}[style](ctx, R)
    iinfo = _interior(ctx, R, style)
    hang = _hang(ctx, R) if ctx.knobs.get("hang", 0) > 0 else 0
    vinfo = _vegetation(ctx, R, climate, weather)
    finfo = _far(ctx, R, style, climate, weather)
    guard = U.cloud_guard(ctx, 600_000)
    sem = R.sem + [{"prim": None, "kind": "place", "change": "enclosure", "detail": ENCLOSURE[style]}]
    return {"cloud_guard": guard, "sky": sky, "style": style, "family": family, "climate": climate, "weather": weather,
            "wall": R.wall_key, "column": R.col_key, "ground": ginfo, "trims": n_trim, "style_info": sinfo,
            "interior": iinfo, "hang": hang, "vegetation": vinfo, "far": finfo, "walls": R.n_walls,
            "openings": R.openings, "semantics_changed": sem}
