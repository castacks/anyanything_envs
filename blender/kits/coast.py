"""Coast: a shore round the patch -- sandy beach, shingle fishing beach,
rocky smugglers' cove or small harbour -- with the sea along one side
(sometimes across part of the patch), a timber pier on pilings, boats
pulled up or moored, fishermen's huts, nets drying on frames, groynes,
driftwood and rocks, a lighthouse out on the water.

What makes it read as a real shore:

* the waterline is a wavy curve, and the ground follows it: dry sand /
  pebbles / rock inland, a dark wet band at the water's edge, dune grass
  and shingle further back; the sea is a glossy Fresnel surface with
  swell and chop normals, turquoise-to-deep colour with distance from the
  shore and broken foam along the waterline;
* everything is built for the place: pilings with a dark weed line at the
  water, clinker-style boat hulls (beached ones heeled over, moored ones
  afloat), plank huts with corrugated/tiled/tarred roofs, nets on
  frames, groynes of timber posts running into the sea;
* far out: a lighthouse on a rock, a headland or cliffs behind, the odd
  ship at anchor; the horizon is the sea itself.

The patch's poles become pilings/mooring posts, walls sea walls or plank
walls, bars timber rails, boxes crates or blocks, spheres boulders.
Levels: ``clutter`` scales props/boats/rocks; ``hang`` > 0 strings
mooring lines, net lines and fish-drying lines between posts across the
flight band.

Contract: lines and fences are checked for free space along their whole
length (a fence rope may touch only the two posts it hangs from), dune grass
is shrunk to fit the free space round it, the openings the family implies
beyond its slots (a ring's interior and gap, the passages past a short wall's
ends, the approach under a floating box) are claimed at the level's keep-out
before anything else is placed (``U.claim_openings``; course.json
``chosen.openings``), a cap beam joins two tall poles only above the drone's
reach (``V7_REACH_TOP``), and what the dressing adds to the patch's own
collision geometry is listed in the result's ``semantics_changed``
(course.json ``chosen``; ``{prim, kind, change, detail, z_top}``). The sea
is floor (a flat overlay 1 cm up): the drone may fly over it; piers stay
outside the patch bounds, a hut in the patch keeps the level's keep-out
from everything (as ruins' fragments).
"""

from __future__ import annotations

import math

import numpy as np

from kits import _ruins_util as U

NAME = "coast"
INDOOR = False
DESCRIPTION = "rocky harbour shore: beach/shingle/cove/harbour, sea with foam, pier on pilings, boats, huts, nets"

STYLES = ("beach", "shingle", "cove", "harbour")

SAND = ["coast_sand_01", "coast_sand_02", "coast_sand_04", "sand_01", "sand_02", "dense_sand", "sand_03",
        "gravelly_sand", "acg:Ground049A", "acg:Ground049B", "acg:Ground033", "forrest_sand_01"]
WET = ["damp_sand", "damp_beach_sand_02", "coast_sand_05", "coast_sand_03", "mud_cracked_dry_riverbed_002"]
PEBBLE = ["clean_pebbles", "dry_river_pebbles", "ganges_river_pebbles", "floor_pebbles_01", "sandy_gravel",
          "pebble_ground_01", "river_small_rocks"]
ROCKY = ["coast_sand_rocks_02", "low_tide_rocks", "coast_land_rocks_01", "rock_ground", "rocks_ground_05",
         "dry_riverbed_rock", "rocky_trail"]
DUNE = ["grass_path_3", "grass_ground", "dry_ground_01", "acg:Ground037", "rocky_terrain_02", "aerial_grass_rock",
        "forrest_sand_01", "park_dirt"]
QUAY = ["cobblestone_floor_01", "cobblestone_floor_04", "cobblestone_large_01", "stone_floor", "cobblestone_floor_07",
        "patterned_cobblestone", "concrete_pavement_02", "square_cobblestone"]
TIMBER = ["weathered_planks", "old_planks_02", "weathered_brown_planks", "wood_planks_grey", "worn_planks",
          "old_wood_floor", "wood_floor_deck", "acg:Wood027", "acg:Planks021"]
POLE_WOOD = ["weathered_brown_planks", "medieval_wood", "acg:Wood051", "old_planks_02", "wood_planks_grey",
             "weathered_planks", "acg:Wood027"]
SIDING = ["weathered_plank_siding", "wood_plank_wall", "brown_planks_04", "brown_planks_05", "dark_planks",
          "raw_plank_wall", "wood_planks_grey", "green_rough_planks", "blue_painted_planks"]
ROOF = ["corrugated_iron", "corrugated_iron_02", "corrugated_iron_03", "clay_roof_tiles", "clay_roof_tiles_02",
        "roof_slates_02", "grey_roof_tiles", "bitumen", "red_slate_roof_tiles_01", "reed_roof_03"]
SEAWALL = ["seaworn_sandstone_brick", "coral_fort_wall_01", "rustic_stone_wall", "stone_wall", "old_stone_wall",
           "large_sandstone_blocks", "concrete_wall_004", "coral_stone_wall", "stone_block_wall", "defense_wall_02"]
ROCK = ["rock_face", "rock_boulder_cracked", "lichen_rock", "gray_rocks", "rock_surface", "acg:Rock030",
        "acg:Rock036", "seaside_rock", "marble_cliff_05"]
#: coast-tagged HDRIs that do not fit a shore seen from the ground (city skylines, balconies)
SKY_EXCLUDE = ("balcony", "rooftop", "docklands", "limehouse", "venice")
PAINT = [(0.72, 0.72, 0.68), (0.12, 0.25, 0.45), (0.55, 0.12, 0.08), (0.15, 0.35, 0.22), (0.8, 0.6, 0.15),
         (0.3, 0.45, 0.55), (0.6, 0.3, 0.15), (0.08, 0.08, 0.08)]


def _keys(lib, names):
    out = []
    for n in names:
        k = n if ":" in n else f"ph:{n}"
        if k in lib.items and "diff" in lib.items[k].get("maps", {}):
            out.append(k)
    return out


class Shore:
    def __init__(self, ctx, style, shore):
        self.ctx, self.rng, self.lk, self.lib = ctx, ctx.rng, ctx.looks, ctx.lib
        rng, lk, lib = self.rng, self.lk, self.lib
        self.style, self.shore = style, shore
        self.timber = lk.pbr(U.pick(rng, _keys(lib, TIMBER)), None, bevel=0.01, dirt=0.45, jitter=0.12)
        pk = U.pick(rng, _keys(lib, POLE_WOOD))
        self.pile = lk.pbr(pk, 1.2, bevel=0.01, dirt=0.3, jitter=0.12)
        U.weather(self.pile, rng, moss=(0.06, 0.08, 0.03), damp=0.85, streaks=0.4, lichen=0.35, base_h=0.9)
        self.siding_keys = _keys(lib, SIDING)
        self.roof_keys = _keys(lib, ROOF)
        sw = U.pick(rng, _keys(lib, SEAWALL))
        self.seawall = lk.pbr(sw, None, bevel=0.03, dirt=0.4, vary=0.25)
        U.weather(self.seawall, rng, moss=(0.1, 0.12, 0.05), damp=0.8, streaks=0.6, lichen=0.4, base_h=1.0)
        self.rock = lk.pbr(U.pick(rng, _keys(lib, ROCK)), 2.5, dirt=0.2)
        self.metal = lk.pbr(U.pick(rng, lib.materials(cats_any={"painted metal"})), 0.8, bevel=0.01, dirt=0.4,
                            tint=(0.18, 0.18, 0.18))
        self.rope = lk.paint((0.16, 0.13, 0.09), rough=0.85, name="rope")
        self.dark = lk.paint((0.03, 0.03, 0.03), rough=0.6, name="tar")
        nets = _keys(lib, ["acg:Net001A", "acg:Net002A", "acg:Net003B", "acg:Net004A", "acg:Net001B"])
        self.net = lk.pbr(U.pick(rng, nets), None) if nets else None
        self.boat_skins = []
        pw = _keys(lib, ["acg:PaintedWood006A", "acg:PaintedWood007A", "acg:PaintedWood008A", "acg:PaintedWood005",
                         "acg:PaintedWood004", "distressed_painted_planks", "wood_peeling_paint_weathered"])
        for _ in range(4):
            col = PAINT[int(rng.integers(len(PAINT)))]
            key = U.pick(rng, pw) if pw else U.pick(rng, _keys(lib, TIMBER))
            self.boat_skins.append(lk.pbr(key, 1.2, tint=[min(1.0, 1.5 * c + 0.1) for c in col], bevel=0.005,
                                          dirt=0.35, jitter=0.1))
        self.boat_skins.append(self.timber)
        self.counts = {}
        self.sem: list = []             # semantics_changed entries (``_dress``)
        self.openings: list = []        # the family's implied openings, claimed (``U.claim_openings``)

    def d(self, xy):
        return U.shore_signed(self.shore, xy)

    def to_patch(self, xy) -> float:
        """Distance from ``xy`` to the patch bounds (0 inside)."""
        x0, y0, x1, y1 = self.ctx.bounds
        return float(math.hypot(max(x0 - xy[0], 0.0, xy[0] - x1), max(y0 - xy[1], 0.0, xy[1] - y1)))

    def _n(self, k):
        self.counts[k] = self.counts.get(k, 0) + 1

    # --- structures -------------------------------------------------------------------------

    def pier(self, a, dirv, L, W, deck_z, spacing=2.4, rail=True, ruined=0.0, check=True):
        """Timber pier from ``a`` (on the beach) along ``dirv``: pile bents,
        cap beams, stringers, planked deck with gaps, handrail posts."""
        ctx, rng = self.ctx, self.rng
        a, dv = np.asarray(a, float), np.asarray(dirv, float)
        dv = dv / np.linalg.norm(dv)
        nv = np.array([-dv[1], dv[0]])
        yaw = math.atan2(dv[1], dv[0])
        # trim the pier where it is not free, or where it would run into the patch (structure
        # stays round the patch: a 40 m deck across it would cut the family's layout in two)
        good = 0.0
        for u in np.arange(0.0, L + 1e-6, 0.8):
            if check and (not ctx.free_rect(a + dv * u, (0.45, W / 2 + 0.2), yaw, margin=0.3, step=0.6)
                          or self.to_patch(a + dv * u) < W / 2 + 0.5):
                break
            good = u
        L = good
        if L < 5.0:
            return 0.0
        piles, deck, frame = U.Parts(), U.Parts(), U.Parts()
        r = float(rng.uniform(0.12, 0.17))
        for u in np.arange(0.4, L, spacing):
            for s in (-1, 1):
                p = a + dv * u + nv * s * (W / 2 - 0.1)
                top = deck_z + (1.05 if rail and rng.random() < 0.9 else -0.05)
                if ruined and rng.random() < ruined * 0.4:
                    top = deck_z * float(rng.uniform(0.3, 0.9))
                piles.frustum((p[0], p[1], 0.0), (p[0], p[1], top), r, r * 0.9, seg=10)
            c = a + dv * u
            frame.box((c[0], c[1], deck_z - 0.2), (0.09, W / 2 + 0.12, 0.1), yaw=yaw)
            # cross bracing on some bents
            if rng.random() < 0.5 and deck_z > 1.0:
                p0 = a + dv * u + nv * (W / 2 - 0.1)
                p1 = a + dv * u - nv * (W / 2 - 0.1)
                frame.frustum((p0[0], p0[1], 0.25), (p1[0], p1[1], deck_z - 0.35), 0.05, 0.05, seg=6)
        for s in (-0.35, 0.35):
            c = a + dv * (L / 2) + nv * s * W
            frame.box((c[0], c[1], deck_z - 0.08), (L / 2, 0.07, 0.07), yaw=yaw)
        pw = float(rng.uniform(0.16, 0.24))
        for u in np.arange(pw / 2, L, pw + 0.015):
            if ruined and rng.random() < ruined * 0.25:
                continue
            c = a + dv * u
            jz = float(rng.normal(0, 0.006))
            deck.box((c[0], c[1], deck_z + 0.025 + jz), (pw / 2, W / 2 + float(rng.uniform(0.0, 0.08)), 0.025),
                     R=U.rot(yaw + float(rng.normal(0, 0.01)), 0.0, float(rng.normal(0, 0.008))))
        if rail:
            for s in (-1, 1):
                c = a + dv * (L / 2) + nv * s * (W / 2 - 0.1)
                frame.box((c[0], c[1], deck_z + 1.0), (L / 2 - 0.3, 0.05, 0.05), yaw=yaw)
        piles.emit(ctx, self.pile, name="pier_piles")
        frame.emit(ctx, self.timber, name="pier_frame")
        deck.emit(ctx, self.timber, name="pier_deck")
        ctx.claim_rect(a + dv * L / 2, (L / 2 + 0.2, W / 2 + 0.3), yaw, kind="pier")
        self._n("piers")
        return L

    def hut(self, c, yaw, w, l, h, pitch=None, colour=None, coll="structure", claim=True, check=True):
        """Plank hut: walls, gabled roof with overhang, door, window."""
        ctx, rng, lk = self.ctx, self.rng, self.lk
        c = np.asarray(c, float)
        # in the patch a hut is clutter: kept back from everything by the level's keep-out, as the
        # ruins' wall fragments are
        inside = self.to_patch(c) < math.hypot(l / 2 + 0.3, w / 2 + 0.3) + 0.5
        margin = 0.5 * float(ctx.knobs.get("keepout", 2.5)) + 0.4 if inside else 0.3
        if check and not ctx.free_rect(c, (l / 2 + 0.4, w / 2 + 0.4), yaw, margin=margin):
            return False
        sk = U.pick(rng, self.siding_keys)
        tint = None if colour is None else [min(1.0, 1.4 * x + 0.12) for x in colour]
        wall = lk.pbr(sk, None, bevel=0.01, dirt=0.45, tint=tint, jitter=0.08)
        U.weather(wall, rng, moss=(0.1, 0.1, 0.07), damp=0.5, streaks=0.6, lichen=0.1, base_h=0.4)
        roof = lk.pbr(U.pick(rng, self.roof_keys), None, bevel=0.005, dirt=0.4)
        pitch = float(rng.uniform(0.35, 0.65)) if pitch is None else pitch
        Rz = U.rot(yaw)
        ex, ey = Rz[:, 0], Rz[:, 1]

        def P(u, v, z):
            q = c + ex[:2] * u + ey[:2] * v
            return (q[0], q[1], z)

        walls, roofp, trim = U.Parts(), U.Parts(), U.Parts()
        t = 0.1
        for s in (-1, 1):
            q = c + ey[:2] * s * (w / 2 - t / 2)
            walls.box((q[0], q[1], h / 2), (l / 2, t / 2, h / 2), yaw=yaw)
            q = c + ex[:2] * s * (l / 2 - t / 2)
            walls.box((q[0], q[1], h / 2), (t / 2, w / 2 - t, h / 2), yaw=yaw)
        zr = h + (w / 2) * math.tan(pitch)
        for s in (-1, 1):                      # gables (triangles on the short walls)
            u = s * (l / 2 - t / 2)
            walls.quads([P(u - t / 2, -w / 2, h), P(u - t / 2, w / 2, h), P(u - t / 2, 0, zr)], [(0, 1, 2)])
            walls.quads([P(u + t / 2, -w / 2, h), P(u + t / 2, 0, zr), P(u + t / 2, w / 2, h)], [(0, 1, 2)])
        ov = 0.3
        span = (w / 2 + ov) / math.cos(pitch)
        for s in (-1, 1):
            vc = s * (w / 2 + ov) / 2
            zc = zr - ((w / 2 + ov) / 2) * math.tan(pitch) + 0.04
            q = c + ey[:2] * vc
            roofp.box((q[0], q[1], zc), (l / 2 + ov, span / 2, 0.035), R=U.rot(yaw, 0.0, -s * pitch))
        # door on a long side, window on the other
        ds = 1 if rng.random() < 0.5 else -1
        q = c + ey[:2] * ds * (w / 2 + 0.02) + ex[:2] * float(rng.uniform(-l / 4, l / 4))
        trim.box((q[0], q[1], 0.95), (0.45, 0.03, 0.95), yaw=yaw)
        walls.emit(ctx, wall, name="hut_wall", coll=coll)
        roofp.emit(ctx, roof, name="hut_roof", coll=coll)
        trim.emit(ctx, self.timber if rng.random() < 0.5 else self.dark, name="hut_door", coll=coll)
        q = c - ey[:2] * ds * (w / 2 + 0.02)
        ctx.box((q[0], q[1], 1.5), (0.4, 0.03, 0.35), lk.glass(), yaw=yaw, coll=coll, name="hut_window")
        if claim:
            ctx.claim_rect(c, (l / 2 + ov, w / 2 + ov), yaw, kind="hut")
        self._n("huts")
        return True

    def boat(self, c, yaw, L=None, afloat=False, upturned=False, heel=None, coll="scatter", check=True):
        ctx, rng = self.ctx, self.rng
        L = float(rng.uniform(3.4, 5.2)) if L is None else L
        B = L * float(rng.uniform(0.3, 0.36))
        D = B * float(rng.uniform(0.38, 0.48))
        c = np.asarray(c, float)
        if check and not ctx.free_rect(c, (L / 2 + 0.2, B / 2 + 0.2), yaw, margin=0.3):
            return False
        V, F = U.boat_hull(L, B, D, rng)
        heel = (float(rng.uniform(0.12, 0.35)) * (1 if rng.random() < 0.5 else -1)) if heel is None else heel
        if afloat:
            heel = float(rng.normal(0, 0.02))
        if upturned:
            V = V * np.array([1, 1, -1]) + np.array([0, 0, D + 0.55])
        R = U.rot(yaw, float(rng.normal(0, 0.03)), heel)
        W = V @ R.T + np.array([c[0], c[1], 0.0])
        if afloat:
            W[:, 2] -= D * 0.35
        W[:, 2] = np.maximum(W[:, 2], 0.006 if not afloat else 0.02)
        skin = self.boat_skins[int(rng.integers(len(self.boat_skins)))]
        hull = U.Parts()
        hull.quads(W, F)
        hull.emit(ctx, skin, name="boat", coll=coll)
        if not upturned:
            seats = U.Parts()
            for xs in (-0.22 * L, 0.05 * L, 0.28 * L):
                p = R @ np.array([xs, 0.0, D * 0.62]) + np.array([c[0], c[1], 0.0])
                if afloat:
                    p[2] -= D * 0.35
                seats.box(p, (0.1, B * 0.4, 0.02), R=R)
            seats.emit(ctx, self.timber, name="thwarts", coll=coll)
        else:
            # trestles under an upturned hull
            for xs in (-0.3 * L, 0.3 * L):
                p = c + R[:2, 0] * xs
                ctx.box((p[0], p[1], 0.27), (0.05, B * 0.45, 0.27), self.timber, yaw=yaw, coll=coll, name="trestle")
        ctx.claim_rect(c, (L / 2 + 0.2, B / 2 + 0.2), yaw, kind="boat")
        self._n("boats")
        return True

    def post_row(self, a, b, spacing, r, h, coll="structure", check=True, jitter=0.15, lean=0.05):
        """A row of timber posts (a groyne / fence line)."""
        ctx, rng = self.ctx, self.rng
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        dv = (b - a) / max(L, 1e-9)
        parts = U.Parts()
        n = 0
        for u in np.arange(0, L, spacing):
            p = a + dv * u
            if check and not ctx.free(p, r, 0.3):
                continue
            hh = h * float(rng.uniform(1 - jitter, 1 + jitter * 0.5))
            top = p + rng.normal(0, lean, 2)
            parts.frustum((p[0], p[1], 0.0), (top[0], top[1], hh), r, r * 0.85, seg=9)
            ctx.claim_disc(p, r + 0.05, kind="post")
            n += 1
        parts.emit(ctx, self.pile, name="posts", coll=coll)
        return n

    def net_frame(self, c, yaw, L, h, coll="scatter", check=True):
        ctx, rng = self.ctx, self.rng
        if self.net is None:
            return False
        c = np.asarray(c, float)
        dv = np.array([math.cos(yaw), math.sin(yaw)])
        if check and not ctx.free_rect(c, (L / 2 + 0.2, 0.3), yaw, margin=0.3):
            return False
        posts = U.Parts()
        a, b = c - dv * L / 2, c + dv * L / 2
        for p in (a, b):
            posts.frustum((p[0], p[1], 0.0), (p[0], p[1], h + 0.1), 0.06, 0.055, seg=8)
        posts.frustum((a[0], a[1], h), (b[0], b[1], h), 0.035, 0.035, seg=6)
        posts.emit(ctx, self.timber, name="net_frame", coll=coll)
        lo = float(rng.uniform(0.2, 0.7))
        ctx.panel(a + dv * 0.08, b - dv * 0.08, lo, h - 0.03, self.net, tile_m=1.2, coll=coll, name="net")
        ctx.claim_rect(c, (L / 2 + 0.2, 0.3), yaw, kind="net")
        self._n("nets")
        return True

    def lifebuoy(self, c, yaw, coll="scatter", check=True):
        ctx = self.ctx
        c = np.asarray(c, float)
        if check and not ctx.free(c, 0.25, 0.4):
            return False
        post = U.Parts()
        post.box((c[0], c[1], 0.9), (0.06, 0.06, 0.9), yaw=yaw)
        post.emit(ctx, self.timber, name="buoy_post", coll=coll)
        red = self.lk.paint((0.75, 0.12, 0.05), rough=0.5, name="buoy_red")
        white = self.lk.paint((0.85, 0.85, 0.82), rough=0.5, name="buoy_white")
        R0, r0 = 0.3, 0.065
        dv = np.array([math.cos(yaw), math.sin(yaw)])
        nv = np.array([-dv[1], dv[0]])
        cen = np.array([c[0], c[1], 1.25]) + np.array([nv[0], nv[1], 0]) * 0.1
        segs = 24
        for part, mat in ((0, red), (1, white)):
            P = U.Parts()
            for k in range(segs):
                if (k // 3) % 2 != part:
                    continue
                t0, t1 = 2 * math.pi * k / segs, 2 * math.pi * (k + 1) / segs
                a = cen + R0 * (math.cos(t0) * np.array([dv[0], dv[1], 0]) + math.sin(t0) * np.array([0, 0, 1]))
                b = cen + R0 * (math.cos(t1) * np.array([dv[0], dv[1], 0]) + math.sin(t1) * np.array([0, 0, 1]))
                P.frustum(a, b, r0, r0, seg=8)
            P.emit(ctx, mat, name="lifebuoy", coll=coll)
        ctx.claim_disc(c, 0.4, kind="lifebuoy")
        self._n("lifebuoys")
        return True

    def line(self, a, b, za, zb, sag=0.25, r=0.012, coll="scatter"):
        """A sagging rope from (a, za) to (b, zb)."""
        P = U.Parts()
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = 8
        pts = []
        for i in range(n + 1):
            t = i / n
            xy = a + (b - a) * t
            z = za + (zb - za) * t - sag * 4 * t * (1 - t)
            pts.append((xy[0], xy[1], z))
        for p, q in zip(pts[:-1], pts[1:]):
            P.frustum(p, q, r, r, seg=5, caps=False)
        return P.emit(self.ctx, self.rope, name="rope", coll=coll)


# --- the build ---------------------------------------------------------------------------------

def _sky(ctx, style, weather):
    lib, rng = ctx.lib, ctx.rng
    sea = lib.hdris(cats_all={"outdoor"}, cats_none={"night", "indoor"},
                    tags_any={"ocean", "beach", "coastal", "harbour", "harbor", "pier", "dock", "sea view", "boat"})
    # not a skyline of glass towers over a fishing beach, nor a view from a balcony
    sea = [k for k in sea if not any(w in k for w in SKY_EXCLUDE)]
    if style != "harbour":
        sea = [k for k in sea if "urban" not in lib.items[k].get("categories", [])] or sea
    if style in ("beach", "cove"):
        nat = [k for k in sea if "nature" in lib.items[k].get("categories", [])]
        sea = nat or sea
    if weather == "overcast":
        k = [x for x in sea if "overcast" in lib.items[x].get("categories", [])]
    elif weather == "sunset":
        k = [x for x in sea if "sunrise-sunset" in lib.items[x].get("categories", [])]
    else:
        k = [x for x in sea if not {"overcast", "sunrise-sunset"} & set(lib.items[x].get("categories", []))]
    return ctx.looks.sky(lib.pick(rng, k or sea))


def _shoreline(ctx):
    """Choose where the sea is: which side, and how far from the patch."""
    rng = ctx.rng
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    r = rng.random()
    w, h = x1 - x0, y1 - y0
    if r < 0.7:
        s = 1 if rng.random() < 0.5 else -1
        n = (0.0, float(s))
        edge = y1 if s > 0 else -y0          # the patch edge, measured along n
        lo = -y0 if s > 0 else y1
        q = rng.random() * (0.8 if h > 0.7 * w else 1.0)   # square patches: cross it more often
        if q < 0.4:                          # the waterline crosses the patch
            off = float(rng.uniform(edge - 0.45 * (edge + lo), edge - 2.0))
        elif q < 0.85:                       # just outside the patch, in the zone
            off = float(rng.uniform(edge + 0.5, edge + 5.0))
        else:                                # beyond the zone: the patch is all shore
            off = float(rng.uniform(edge + 7.0, edge + 12.0))
    else:                                    # the sea beyond the goal
        n = (1.0, 0.0)
        off = float(rng.uniform(x1 - 2.0, x1 + 12.0))
    L = float(rng.uniform(25, 70))
    return {"n": n, "off": off, "A": float(rng.uniform(0.5, 2.5)), "k": 2 * math.pi / L,
            "phi": float(rng.uniform(0, 2 * math.pi))}


def _ground(ctx, style, shore, weather):
    lib, rng, lk = ctx.lib, ctx.rng, ctx.looks
    sand, wet, peb, rocky, dune, quay = (_keys(lib, x) for x in (SAND, WET, PEBBLE, ROCKY, DUNE, QUAY))
    if style == "beach":
        base = [(U.pick(rng, sand), None), (U.pick(rng, sand), None), (U.pick(rng, peb), None)]
        back = U.pick(rng, dune)
    elif style == "shingle":
        base = [(U.pick(rng, peb), None), (U.pick(rng, peb), None), (U.pick(rng, sand), None)]
        back = U.pick(rng, dune + peb)
    elif style == "cove":
        base = [(U.pick(rng, rocky), None), (U.pick(rng, sand), None), (U.pick(rng, rocky), None)]
        back = U.pick(rng, rocky + dune)
    else:
        base = [(U.pick(rng, sand + peb), None), (U.pick(rng, rocky), None), (U.pick(rng, sand), None)]
        back = U.pick(rng, quay)
    wk = U.pick(rng, wet)
    wband = float(rng.uniform(2.5, 6.0))
    bdist = float(rng.uniform(8.0, 16.0))
    off3 = rng.uniform(-50, 50, 3)

    def wet_mask(nk):
        d = nk.shore_dist(shore)
        nz = nk.noise(0.7, detail=5, offset=off3)
        dd = nk.math("ADD", d, nk.math("MULTIPLY", nk.math("SUBTRACT", nz, 0.5), 2.5))
        return nk.mrange(dd, -wband, -wband + 1.2)

    def back_mask(nk):
        d = nk.shore_dist(shore)
        nz = nk.noise(0.25, detail=6, offset=off3 + 9)
        dd = nk.math("ADD", d, nk.math("MULTIPLY", nk.math("SUBTRACT", nz, 0.5), 8.0))
        return nk.mrange(dd, -bdist + 1.5, -bdist - 1.5)

    wet_tint = (0.62, 0.58, 0.52)
    layers = [{"key": back, "mask": back_mask},
              {"key": base[0][0] if rng.random() < 0.6 else wk, "mask": wet_mask, "tint": wet_tint}]
    m = U.zoned_ground(lk, base, layers, puddles=0.0, blend_scale=float(rng.uniform(5, 11)), rng=rng)
    ctx.ground(m)
    return {"base": [b[0] for b in base], "wet": wk, "back": back, "wet_band": round(wband, 2),
            "back_dist": round(bdist, 2)}


def _dress(ctx, S, style):
    rng = ctx.rng
    wall = S.seawall if style in ("harbour", "cove") or rng.random() < 0.5 else \
        ctx.looks.pbr(U.pick(rng, S.siding_keys), None, bevel=0.01, dirt=0.45)
    crates = ctx.looks.pbr(U.pick(rng, _keys(ctx.lib, TIMBER)), 1.0, bevel=0.01, dirt=0.4)
    poles = [float(p["r"]) for p in ctx.spec["primitives"] if p["kind"] == "cyl_v"]
    pole = S.seawall if poles and min(poles) > 0.55 else S.pile      # fat posts: stone dolphins/bollards
    ctx.dress_primitives({"wall": wall, "box": S.seawall if rng.random() < 0.4 else crates, "pole": pole,
                          "bar": S.timber if rng.random() < 0.7 else S.metal, "rock": S.rock})
    S.openings = U.claim_openings(ctx)
    sem = S.sem                     # what the dressing adds to the patch's own collision geometry
    # rope bands round the poles, a cap on tall ones
    ropes = U.Parts()
    for i, p in enumerate(ctx.spec["primitives"]):
        if p["kind"] == "cyl_v" and rng.random() < 0.25 and U.clear_of_ends(ctx, p["c"], float(p["r"]) + 0.1):
            c = p["c"]
            r = float(p["r"])
            z = float(rng.uniform(0.8, 1.6))
            ropes.frustum((c[0], c[1], z), (c[0], c[1], z + 0.08), r + 0.02, r + 0.02, seg=14)
            sem.append({"prim": i, "kind": "cyl_v", "change": "rope_band",
                        "detail": f"rope band r + 0.02 m at z {z:.2f}-{z + 0.08:.2f} m", "z_top": round(z + 0.08, 3)})
    ropes.emit(ctx, S.rope, name="pole_rope", coll="structure")
    # tall poles in pairs get a cap beam across their tops (above the flight band): an old pier's bents
    tall = [(i, p) for i, p in enumerate(ctx.spec["primitives"]) if p["kind"] == "cyl_v" and float(p["z1"]) >= 4.4]
    beams = U.Parts()
    used = set()
    for i, (pi, p) in enumerate(tall):
        if i in used:
            continue
        ci = np.asarray(p["c"], float)
        best, bd = None, 1e9
        for j, (_, q) in enumerate(tall):
            if j == i or j in used:
                continue
            dd = float(np.linalg.norm(np.asarray(q["c"], float) - ci))
            if 1.2 < dd < 4.0 and dd < bd:
                best, bd = j, dd
        if best is None or rng.random() < 0.3:
            continue
        qi, q = tall[best]
        cj = np.asarray(q["c"], float)
        z = min(float(p["z1"]), float(q["z1"])) - 0.12
        if z - 0.12 < U.V7_REACH_TOP:           # a beam in reach would close the gap between the poles
            continue
        d = (cj - ci) / bd
        a, b = ci - d * 0.25, cj + d * 0.25
        yaw = math.atan2(d[1], d[0])
        mid = (a + b) / 2
        beams.box((mid[0], mid[1], z), (bd / 2 + 0.25, 0.1, 0.12), yaw=yaw)
        for k in (pi, qi):
            sem.append({"prim": k, "kind": "cyl_v", "change": "cap_beam",
                        "detail": f"beam to prim {qi if k == pi else pi} across the tops from z {z - 0.12:.2f} m "
                                  "(above v7 reach)", "z_top": round(z + 0.12, 3)})
        used |= {i, best}
    beams.emit(ctx, S.timber, name="pole_beams", coll="structure")


def _style_features(ctx, S, style):
    """The shore furniture: pier, huts, boats, groynes, nets, sea wall."""
    rng, lib = ctx.rng, ctx.lib
    shore = S.shore
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    n = np.array(shore["n"])
    t = np.array([-n[1], n[0]])
    cz = np.array([(zx0 + zx1) / 2, (zy0 + zy1) / 2])
    half_t = 0.5 * abs(np.dot(np.array([zx1 - zx0, zy1 - zy0]), np.abs(t)))

    def at(tt, dd):
        """Point at tangent coordinate tt and signed distance dd from the waterline."""
        return t * tt + n * (U.shore_y(shore, tt) + dd)

    tc = float(cz @ t)
    info = {}
    # pier: harbour always, others sometimes
    if style == "harbour" or rng.random() < 0.45:
        for _ in range(6):
            tt = tc + float(rng.uniform(-half_t - 10, half_t + 10))
            a = at(tt, -float(rng.uniform(2.0, 5.0)))
            L = float(rng.uniform(18, 45))
            W = float(rng.uniform(1.8, 3.2))
            deck = float(rng.uniform(1.1, 1.8))
            got = S.pier(a, n, L, W, deck, spacing=float(rng.uniform(2.0, 3.2)), rail=rng.random() < 0.6,
                         ruined=float(rng.choice([0.0, 0.0, 0.5, 1.0])))
            if got:
                info["pier_len"] = round(got, 1)
                # boats moored alongside
                for _k in range(int(rng.integers(0, 3))):
                    side = 1 if rng.random() < 0.5 else -1
                    u = float(rng.uniform(4, got - 2))
                    bl = float(rng.uniform(4.0, 7.5))
                    p = a + n * u + t * side * (W / 2 + bl * 0.17 + 0.6)
                    S.boat(p, math.atan2(n[1], n[0]) + float(rng.normal(0, 0.1)), L=bl, afloat=True)
                break
    # huts along the back of the beach
    n_huts = int(rng.integers(2, 6)) if style in ("shingle", "beach", "harbour") else int(rng.integers(0, 3))
    colourful = style == "beach" and rng.random() < 0.6
    back = float(rng.uniform(9.0, 16.0))
    tt = tc - half_t - float(rng.uniform(0, 6))
    for _ in range(n_huts * 3):
        if S.counts.get("huts", 0) >= n_huts or tt > tc + half_t + 10:
            break
        w = float(rng.uniform(2.2, 3.4)) if colourful else float(rng.uniform(3.0, 5.0))
        l = float(rng.uniform(2.2, 3.0)) if colourful else float(rng.uniform(4.0, 7.5))
        h = float(rng.uniform(2.2, 2.6)) if colourful else float(rng.uniform(2.4, 3.4))
        p = at(tt, -back - w / 2 - float(rng.uniform(0, 3)))
        yaw = math.atan2(t[1], t[0]) + float(rng.normal(0, 0.05))
        col = PAINT[int(rng.integers(len(PAINT) - 1))] if colourful else (None if rng.random() < 0.7 else
                                                                          PAINT[int(rng.integers(len(PAINT)))])
        if S.hut(p, yaw, w, l, h, colour=col):
            # nets / buoys / crates by the hut
            if rng.random() < 0.6:
                S.net_frame(p + n[:2] * (w / 2 + 1.5), yaw, float(rng.uniform(2.5, 4.5)), float(rng.uniform(1.8, 2.4)))
            if rng.random() < 0.4:
                S.lifebuoy(p + n[:2] * (w / 2 + 0.6) + t * (l / 2 + 0.5), yaw)
            tt += l + float(rng.uniform(1.0, 6.0))
        else:
            tt += 2.0
    # boats pulled up on the shore
    nb = ctx.count(2.5 if style in ("beach", "shingle") else 1.2)
    for _ in range(nb * 8):
        if S.counts.get("boats_shore", 0) >= nb:
            break
        tt = tc + float(rng.uniform(-half_t, half_t))
        p = at(tt, -float(rng.uniform(1.5, back - 2)))
        yaw = math.atan2(n[1], n[0]) + float(rng.normal(0, 0.5)) + (math.pi if rng.random() < 0.5 else 0)
        if S.boat(p, yaw, upturned=rng.random() < 0.2):
            S.counts["boats_shore"] = S.counts.get("boats_shore", 0) + 1
    # groynes
    if style in ("beach", "shingle") and rng.random() < 0.7:
        gap = float(rng.uniform(18, 35))
        tt = tc - half_t + float(rng.uniform(0, gap))
        while tt < tc + half_t:
            a = at(tt, -float(rng.uniform(3, 7)))
            b = at(tt, float(rng.uniform(10, 25)))
            S.post_row(a, b, float(rng.uniform(0.32, 0.5)), float(rng.uniform(0.1, 0.15)), float(rng.uniform(0.9, 1.7)))
            S._n("groynes")
            tt += gap
    # a dune fence (posts and a sagging rope) along the back of a beach
    if style in ("beach", "shingle") and rng.random() < 0.6:
        dd = -back + float(rng.uniform(-2, 2))
        tt = tc - half_t
        prev = prev_post = None
        while tt < tc + half_t:
            q = at(tt, dd + float(rng.normal(0, 0.1)))
            if ctx.free(q, 0.1, 0.4):
                n0 = len(ctx.foot)
                S.post_row(q, q + t * 0.01, 1.0, 0.05, float(rng.uniform(0.9, 1.1)), check=False, coll="scatter")
                post = ctx.foot[n0]                 # this post's claim (post_row: one post at q)
                # the rope may touch the two posts it is strung between (claimed already), nothing else
                if prev is not None and U.seg_free(ctx, prev, q, 0.02, 0.2, ignore=(prev_post, post)):
                    S.line(prev, q, 0.85, 0.85, sag=0.08, r=0.01)
                    sp = q - prev
                    ctx.claim_rect((prev + q) / 2, (float(np.linalg.norm(sp)) / 2, 0.1), math.atan2(sp[1], sp[0]),
                                   kind="rope")
                    S._n("fence_spans")
                prev, prev_post = q, post
            else:
                prev = None
            tt += float(rng.uniform(1.8, 2.4))
        S._n("dune_fence")
    # sea wall along the back (harbour / cove), if it does not cross the patch
    if style in ("harbour",) and rng.random() < 0.8:
        dd = -back - float(rng.uniform(4, 7))
        p0, p1 = at(tc - half_t - 5, dd), at(tc + half_t + 5, dd)
        mid = (p0 + p1) / 2
        inside = x0 - 1 < mid[0] < x1 + 1 and y0 - 1 < mid[1] < y1 + 1
        if not inside:
            parts = U.Parts()
            h = float(rng.uniform(0.9, 1.6))
            ok = 0
            for tt in np.arange(tc - half_t - 5, tc + half_t + 5, 1.0):
                q = at(tt, dd)
                if ctx.free(q, 0.5, 0.3):
                    parts.box((q[0], q[1], h / 2), (0.52, 0.45, h / 2), yaw=math.atan2(t[1], t[0]))
                    ctx.claim_disc(q, 0.55, kind="seawall")
                    ok += 1
            parts.emit(ctx, S.seawall, name="seawall")
            info["seawall"] = ok
    return info


def _scatter(ctx, S, style):
    """Driftwood, rocks, crates, barrels, buoys; plants at the back."""
    rng, lib = ctx.rng, ctx.lib
    zx0, zy0, zx1, zy1 = ctx.zone
    land = lambda xy: S.d(xy) < -0.5                       # noqa: E731
    near_water = lambda xy: -6.0 < S.d(xy) < 4.0            # noqa: E731
    back = lambda xy: S.d(xy) < -7.0                       # noqa: E731
    # the sea well beyond the zone (a wide beach): the strand-line debris goes where the zone is
    if max(S.d((x, y)) for x in (zx0, zx1) for y in (zy0, zy1)) < -3.0:
        near_water = land
    drift = lib.models(ids=["dead_tree_trunk", "dead_tree_trunk_02", "dry_branches_medium_01", "bark_debris_01",
                            "single_root"], max_tris=120000)
    ctx.scatter(drift, ctx.count(4), rect=ctx.zone, margin=0.3, where=near_water)
    rocks = lib.models(ids=["rock_moss_set_01", "rock_moss_set_02", "namaqualand_boulder_02", "namaqualand_boulder_03",
                            "namaqualand_boulder_04", "boulder_01", "rock_face_02"], max_tris=120000)
    nr = ctx.count(8 if style == "cove" else 3)
    ctx.scatter(rocks, nr, rect=ctx.zone, margin=0.4, s=(0.7, 1.4),
                where=(None if style == "cove" else near_water))
    # heavy scanned shore rocks: a couple, shared instances
    big = lib.models(ids=["coast_rocks_05", "sand_rocks_small_01"])
    if big and (style in ("cove", "harbour") or rng.random() < 0.3):
        ctx.scatter(big, int(rng.integers(1, 3)), rect=ctx.zone, margin=0.5, s=(0.7, 1.1), where=near_water)
    props = lib.models(ids=["wooden_barrels_01", "Barrel_01", "Barrel_02", "barrel_03", "wooden_crate_01",
                            "wooden_crate_02", "plastic_crate_01", "plastic_crate_02", "plastic_crate_03",
                            "wooden_bucket_02", "old_tyre", "metal_jerrycan", "plastic_container",
                            "wooden_lantern_01", "life_jacket"], h=(0.2, 1.2))
    if style == "cove":
        props += lib.models(ids=["cannon_01", "treasure_chest", "wooden_barrels_01"], h=(0.3, 1.2))
    ctx.scatter(props, ctx.count(10), rect=ctx.zone, margin=0.4, where=land)
    ctx.scatter(drift, ctx.count(3), rect=ctx.zone, margin=0.3, where=land)
    # buoys out on the water (in and beyond the zone)
    buoy = lib.models(ids=["ocean_buoy"])
    ctx.scatter(buoy, int(rng.integers(0, 3)), rect=(zx0 - 30, zy0 - 30, zx1 + 30, zy1 + 30), margin=1.0,
                where=lambda xy: S.d(xy) > 8.0)
    # plants at the back: dune grass, shrubs
    tufts = lib.models(ids=["grass_medium_02", "grass_medium_01"], h=(0.1, 0.6), max_tris=6000) * 4 + \
        lib.models(ids=["crystalline_iceplant", "flower_gazania", "cheiridopsis_succulent"], h=(0.05, 0.6),
                   max_tris=12000)
    shrubs = lib.models(ids=["shrub_02", "searsia_lucida", "othonna_cerarioides", "didelta_spinosa"], h=(0.5, 2.4),
                        max_tris=160000)
    n = 0
    for _ in range(ctx.count(200) * 5):
        if n >= ctx.count(200):
            break
        xy = np.array([rng.uniform(zx0, zx1), rng.uniform(zy0, zy1)])
        dd = S.d(xy)
        if dd > -3.0 or (dd > -8.0 and rng.random() < 0.7):
            continue
        cl = ctx.clearance(xy)
        if cl < 0.3:
            continue
        k = lib.pick(rng, tufts)
        sz = lib.size(k)
        sc = U.fit_scale(0.5 * math.hypot(sz[0], sz[1]), float(rng.uniform(1.2, 2.6)), cl, s_min=0.8)
        if sc is None:
            continue
        ctx.instance(k, xy, 0.0, float(rng.uniform(0, 6.3)), sc, coll="scatter")
        n += 1
    ctx.scatter(shrubs, ctx.count(4), rect=ctx.zone, margin=0.5, where=back)
    return {"tufts": n}


def _hang(ctx, S):
    """L3: lines between posts across the flight band (mooring / drying lines)."""
    rng = ctx.rng
    n = ctx.count(float(ctx.knobs.get("hang", 0.0)))
    built = 0
    for _ in range(n):
        for _t in range(60):
            a = ctx.uniform_xy(pad=1.0)
            ang = float(rng.uniform(0, math.pi))
            L = float(rng.uniform(3.0, 6.0))
            dv = np.array([math.cos(ang), math.sin(ang)])
            b = a + dv * L
            mid = (a + b) / 2
            if not (ctx.free(a, 0.2, 0.5) and ctx.free(b, 0.2, 0.5) and ctx.free(mid, 0.1, 0.6)):
                continue
            # the line (and a net hung on it) spans a -> b in the band: clear along its length
            if not U.seg_free(ctx, a, b, 0.1, 0.4):
                continue
            za, zb = float(rng.uniform(2.0, 3.3)), float(rng.uniform(2.0, 3.3))
            posts = U.Parts()
            for p, z in ((a, za), (b, zb)):
                posts.frustum((p[0], p[1], 0.0), (p[0], p[1], z + 0.15), 0.08, 0.07, seg=8)
                ctx.claim_disc(p, 0.15, kind="line_post")
            posts.emit(ctx, S.pile, name="line_posts", coll="scatter")
            S.line(a, b, za, zb, sag=float(rng.uniform(0.1, 0.5)))
            # things hung on it: floats / a net strip
            if S.net is not None and rng.random() < 0.4:
                zl = min(za, zb) - 0.35
                ctx.panel(a + dv * 0.4, b - dv * 0.4, max(0.8, zl - 1.2), zl, S.net, tile_m=1.2, coll="scatter",
                          name="hung_net")
            ctx.claim_rect(mid, (L / 2, 0.3), ang, kind="line")
            built += 1
            break
    return built


def _far(ctx, S, style, weather):
    """Lighthouse on a rock, a headland/cliffs behind the shore, a ship at anchor."""
    rng, lib, lk = ctx.rng, ctx.lib, ctx.looks
    shore = S.shore
    zx0, zy0, zx1, zy1 = ctx.zone
    n = np.array(shore["n"])
    t = np.array([-n[1], n[0]])
    cz = np.array([(zx0 + zx1) / 2, (zy0 + zy1) / 2])
    tc = float(cz @ t)
    info = {}

    def at(tt, dd):
        return t * tt + n * (U.shore_y(shore, tt) + dd)

    # hills / cliffs inland
    back_ang = math.atan2(-n[1], -n[0])
    grass = lk.pbr(U.pick(rng, _keys(lib, ["aerial_grass_rock", "rocky_terrain_02", "aerial_rocks_04",
                                           "aerial_rocks_02", "dirt_aerial_03"])), float(rng.uniform(35, 60)), vary=0.35)
    rad = max(zx1 - zx0, zy1 - zy0)
    U.heightfield_ring(ctx, cz - n * 20, 160 + rad, 650 + rad, float(rng.uniform(15, 60)), grass, rng,
                       arc=(back_ang - 1.35, back_ang + 1.35), n_th=100)
    # (the coastal_cliff_* scans are thin cliff faces: from the land they read as floating overhangs)
    cliffs = lib.models(ids=["namaqualand_cliff_02", "mountainside", "namaqualand_cliff_01"])
    if cliffs and (style in ("cove", "shingle", "harbour") or rng.random() < 0.4):
        for s in (-1, 1):
            if rng.random() < 0.3:
                continue
            k = U.pick(rng, cliffs)
            sc = float(rng.uniform(1.6, 3.0))
            sz = lib.size(k)
            half = 0.5 * math.hypot(sz[0], sz[1]) * sc
            tt = tc + s * (half_len(ctx, t) + half + float(rng.uniform(10, 50)))
            p = at(tt, -float(rng.uniform(2, 10)))
            yaw = math.atan2(n[1], n[0]) + math.pi / 2 + (0.0 if sz[0] >= sz[1] else math.pi / 2)
            ctx.instance(k, p, 0.0, yaw + float(rng.normal(0, 0.2)) + (math.pi if rng.random() < 0.5 else 0.0), sc,
                         coll="backdrop")
    # lighthouse on a rock out at sea (or on the point)
    if rng.random() < 0.75:
        tt = tc + float(rng.uniform(-250, 250))
        dd = float(rng.uniform(80, 320))
        p = at(tt, dd)
        rocks = lib.models(ids=["coast_rocks_05", "coast_land_rocks_03"])
        if rocks:
            ctx.instance(U.pick(rng, rocks), p, 0.0, float(rng.uniform(0, 6.3)), float(rng.uniform(3, 5)),
                         coll="backdrop")
        H = float(rng.uniform(18, 32))
        r0 = float(rng.uniform(2.4, 3.6))
        body = U.Parts()
        base_z = 3.0
        body.frustum((p[0], p[1], 0.0), (p[0], p[1], base_z), r0 + 1.5, r0 + 1.3, seg=24)
        body.frustum((p[0], p[1], base_z), (p[0], p[1], base_z + H), r0, r0 * 0.72, seg=32)
        plaster = lk.pbr(U.pick(rng, _keys(lib, ["white_plaster_rough_01", "white_rough_plaster", "white_stucco",
                                                 "grey_plaster", "plastered_wall"]) or S.siding_keys), 3.0,
                         dirt=0.3, vary=0.15)
        body.emit(ctx, plaster, name="lighthouse", coll="backdrop")
        band_c = PAINT[2] if rng.random() < 0.6 else PAINT[0 if rng.random() < 0.5 else 7]
        bands = U.Parts()
        for f in (0.33, 0.66):
            z = base_z + H * f
            rr = r0 + (r0 * 0.72 - r0) * f + 0.03
            bands.frustum((p[0], p[1], z), (p[0], p[1], z + H * 0.14), rr, rr - 0.28 * r0 * 0.14, seg=32)
        bands.emit(ctx, lk.paint(band_c, rough=0.5, name="lh_band"), name="lh_bands", coll="backdrop")
        top = base_z + H
        rt = r0 * 0.72
        cap = U.Parts()
        cap.frustum((p[0], p[1], top), (p[0], p[1], top + 0.35), rt + 0.9, rt + 0.9, seg=32)
        cap.frustum((p[0], p[1], top + 3.1), (p[0], p[1], top + 4.6), rt * 0.75, 0.15, seg=24)
        cap.emit(ctx, S.dark, name="lh_cap", coll="backdrop")
        glow = 30.0 if weather == "sunset" else 4.0
        lamp = lk.paint((0.9, 0.85, 0.6), rough=0.1, name="lh_lamp", emission=(1.0, 0.9, 0.6, glow))
        ctx.cylinder((p[0], p[1], top + 0.35), (p[0], p[1], top + 3.1), rt * 0.65, lamp, coll="backdrop",
                     collide=False, name="lh_lantern", segments=24)
        info["lighthouse"] = [round(float(p[0]), 1), round(float(p[1]), 1), round(H, 1)]
    # a ship at anchor
    ships = lib.models(ids=["dutch_ship_medium", "dutch_ship_large_01", "dutch_ship_large_02", "ship_pinnace"])
    if ships and (style == "cove" or rng.random() < 0.2):
        p = at(tc + float(rng.uniform(-200, 200)), float(rng.uniform(90, 300)))
        ctx.instance(U.pick(rng, ships), p, -1.2, float(rng.uniform(0, 6.3)), 1.0, coll="backdrop")
        info["ship"] = True
    # distant boats (procedural) at sea
    for _ in range(int(rng.integers(0, 4))):
        p = at(tc + float(rng.uniform(-150, 150)), float(rng.uniform(30, 160)))
        # (a big arena's zone can reach 30 m out: there the boat is checked like any other)
        S.boat(p, float(rng.uniform(0, 6.3)), L=float(rng.uniform(5, 9)), afloat=True, coll="backdrop",
               check=ctx.in_zone(p, pad=6.0))
    return info


def half_len(ctx, t):
    zx0, zy0, zx1, zy1 = ctx.zone
    return 0.5 * abs(np.dot(np.array([zx1 - zx0, zy1 - zy0]), np.abs(t)))


def build(ctx) -> dict:
    rng = ctx.rng
    style = str(STYLES[int(rng.integers(len(STYLES)))])
    wr = rng.random()
    weather = "overcast" if wr < 0.3 else ("sunset" if wr < 0.45 else "clear")
    if style == "shingle" and rng.random() < 0.4:
        weather = "overcast"
    sky = _sky(ctx, style, weather)
    shore = _shoreline(ctx)
    S = Shore(ctx, style, shore)
    ginfo = _ground(ctx, style, shore, weather)
    murky = float(rng.uniform(0.0, 0.5)) if weather == "overcast" else float(rng.uniform(0.0, 0.25))
    if style == "cove" and rng.random() < 0.5:
        shallow = (0.05, 0.2, 0.18)                      # clear turquoise over sand
    else:
        shallow = (0.06, 0.15, 0.13)
    sea = U.water("sea", shore=shore, shallow=shallow, rough=float(rng.uniform(0.02, 0.07)),
                  wave=float(rng.uniform(0.7, 1.4)), foam=float(rng.uniform(0.4, 0.9)), rng=rng, murky=murky)
    U.sea_mesh(ctx, shore, sea)
    _dress(ctx, S, style)
    finfo = _style_features(ctx, S, style)
    sinfo = _scatter(ctx, S, style)
    hang = _hang(ctx, S) if ctx.knobs.get("hang", 0) > 0 else 0
    far = _far(ctx, S, style, weather)
    guard = U.cloud_guard(ctx, 600_000)
    sem = S.sem + [{"prim": None, "kind": "place", "change": "sea_floor",
                    "detail": "the sea is floor (a flat overlay 1 cm up); piers stay outside the bounds"}]
    return {"cloud_guard": guard, "sky": sky, "style": style, "weather": weather,
            "shore": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in shore.items()},
            "ground": ginfo, "features": finfo, "scatter": sinfo, "hang": hang, "far": far, "counts": S.counts,
            "openings": S.openings, "semantics_changed": sem}
