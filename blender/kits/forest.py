"""Woodland: a stand of real-looking trees round the patch's obstacles, with
undergrowth, deadwood and mossy rock on a forest floor, under a forest sky.

What makes it read as a forest rather than poles on a plane:

* **trunks** are procedural (``_forest_util.trunk``): curved, tapering,
  root-flared, bark-scanned, a different radius, lean and bow each; every
  trunk carries a real Poly Haven crown ("crown on a pole": fir saplings for
  conifers, ``tree_small_02`` for broadleaf) set above the cloud ceiling,
  so the canopy is photographic and casts dappled shade, while the cloud (up
  to ``cloud_top``, 4.5-4.75 m) sees only the trunk -- which is exactly what
  a drone flying under a canopy has to avoid;
* the **stand continues** past the cloud zone for ~55 m (render-only), so
  every view ends in trees, not a horizon of flat ground;
* the **floor** mixes three litter/moss/needle scans, with leaf-litter
  decals at tree feet, an optional trail (routed through the obstacle's
  opening), ferns, grass tufts, roots, dry branches, stumps, fallen logs and
  mossy boulders;
* **forest types**: conifer, broadleaf (summer or autumn), birch, mixed,
  burnt (black-charred snags, no canopy, ash floor) and snowy (snow floor,
  snow on every up-facing surface), each with a fitting HDRI;
* the patch's **obstacles** join in: walls become rock bands, old dry-stone
  walls or plank fences (moss or snow on top); tall poles become trees
  (crown added above 5 m), short thin ones weathered posts; bars become
  fallen trunks lodged between trees; spheres become mossy boulders or,
  floating, a nest hung from a vine.

Levels: ``clutter`` scales tree density in the patch, undergrowth and
deadwood; ``keepout`` keeps trees back from openings; ``hang`` (L3) adds
leaning dead trees, low dead branches and hanging vines in the flight band.

Contract notes: everything that stands in the cloud zone keeps its whole
footprint off the start/goal discs and the primitives' claims (ground cover,
roots and sticks included, ``_forest_util.clear_of``); a crown that can
reach the zone starts above the cloud ceiling (stem and all: nothing of it is
rendered below the ceiling outside the cloud), else it collides
(``_forest_util.crown``). What the dressing adds to the family's own
geometry below the ceiling (stilts under a raised box, trunks at a bar's
ends, a vine above a floating sphere) is listed in the result's
``semantics_changed`` ({prim, kind, change, detail}), so family statistics
can tell a dressed patch from the bare one. ``FOREST_KIND=<kind>[:autumn]``
in the environment forces the forest type (a test hook; recorded as
``env_override``, never set by the pipeline).
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from kits import _forest_util as U

NAME = "forest"
INDOOR = False
DESCRIPTION = "woodland: bark trunks under real crowns, undergrowth, logs, mossy rock; conifer/broadleaf/birch/burnt/snow"

FIR = ["ph:fir_sapling_medium#0", "ph:fir_sapling_medium#1", "ph:fir_sapling_medium#2"]
BROAD = ["ph:tree_small_02#0"]

BARK = {
    "conifer": ["ph:pine_bark", "ph:knotted_pine_bark", "ph:japanese_cedar_bark", "ph:chinese_cedar_bark",
                "ph:metasequoia_bark", "acg:Bark001", "acg:Bark004", "acg:Bark014", "ph:bark_brown_02"],
    "broadleaf": ["ph:bark_brown_01", "ph:jolcham_oak_bark_01", "ph:trident_maple_bark", "ph:japanese_hackberry_bark",
                  "ph:chinese_hackberry_bark", "acg:Bark002", "acg:Bark003", "acg:Bark007", "acg:Bark008",
                  "acg:Bark012", "ph:bark_willow", "ph:tree_bark_03", "ph:japanese_zelkova_bark"],
    "birch": ["acg:Bark009", "acg:Bark010", "acg:Bark011", "ph:japanese_sycamore", "ph:tree_bark_03"],
}
GROUND = {
    "conifer": ["ph:forest_leaves_04", "ph:forest_ground_04", "acg:Ground037", "acg:Ground048", "ph:forrest_ground_03",
                "ph:forest_leaves_03", "ph:mud_forest", "ph:roots", "acg:Ground001", "ph:rocky_trail"],
    "summer": ["ph:brown_mud_leaves_01", "acg:Ground037", "ph:dirt_floor", "acg:Ground024", "ph:forrest_ground_01",
               "ph:forest_ground_06", "ph:forest_leaves_03", "acg:Ground023", "ph:grass_ground"],
    "autumn": ["ph:dry_decay_leaves", "ph:forest_floor", "ph:leaves_forest_ground", "acg:Ground041", "acg:Ground042",
               "ph:forest_leaves_02", "acg:Ground023", "ph:brown_mud_leaves_01"],
    "birch": ["ph:forrest_ground_01", "acg:Ground003", "acg:Ground037", "ph:grass_ground", "ph:sparse_grass",
              "ph:withered_grass", "acg:Ground013", "ph:forest_leaves_04"],
    "burnt": ["ph:burned_ground_01", "ph:forest_ground_06", "ph:brown_mud_rocks_01", "acg:Ground043", "acg:Ground045",
              "ph:mud_forest", "ph:burned_ground_01"],
    "snowy": ["ph:snow_02", "ph:snow_03", "acg:Snow014", "ph:snow_01", "acg:Snow011", "acg:Snow006"],
}
TRAIL = ["ph:dirt_floor", "ph:forest_ground_05", "ph:muddy_tracks", "ph:brown_mud_02", "ph:brown_mud_dry",
         "ph:forest_ground_06", "ph:dirt_floor"]
ROCK = ["ph:mossy_rock", "ph:lichen_rock", "ph:rock_face_03", "acg:Rock051", "acg:Rock036", "ph:rock_wall_02",
        "ph:cliff_side", "ph:quarry_wall", "acg:Rock063", "ph:rock_boulder_dry", "acg:Rock030"]
STONEWALL = ["ph:mossy_stone_wall", "ph:old_stone_wall_02", "ph:castle_wall_slates", "ph:medieval_wall_02",
             "ph:broken_wall", "acg:Rock064", "ph:japanese_stone_wall"]
PLANKS = ["ph:brown_planks_03", "ph:brown_planks_04", "ph:brown_planks_05", "ph:brown_planks_07", "acg:Planks012",
          "acg:Planks021", "acg:Planks009", "ph:weathered_brown_planks", "ph:old_planks_02"]
MOSS = ["acg:Moss002", "acg:Moss003", "acg:Moss004"]
SNOW = ["ph:snow_02", "acg:Snow004", "acg:Snow002"]

#: forest type -> (weight, HDRI preference/query)
KINDS = {"conifer": 0.24, "broadleaf": 0.22, "mixed": 0.14, "birch": 0.12, "burnt": 0.1, "snowy": 0.18}


def _sky(ctx, kind, autumn):
    lib = ctx.lib
    base = dict(cats_all={"outdoor", "nature"}, cats_none={"night", "indoor", "artificial light", "urban"})
    manmade = {"building", "city", "road", "parking", "park", "architecture", "car", "bridge", "tunnel", "garden",
               "street", "asphalt", "wall"}
    if kind == "snowy":
        keys = lib.hdris(tags_any={"snow"}, **base)
        keys = [k for k in keys if set(lib.items[k].get("tags", [])) & {"forest", "tree"}
                and not set(lib.items[k].get("tags", [])) & manmade] or keys
    elif kind == "burnt":
        keys = [k for k in lib.hdris(tags_any={"forest", "tree", "field"}, **base)
                if set(lib.items[k].get("categories", [])) & {"overcast", "partly cloudy"}
                and "nature" in lib.items[k].get("categories", []) and "urban" not in lib.items[k].get("categories", [])
                and not set(lib.items[k].get("tags", [])) & {"snow", "winter", "city", "building", "garden", "lake",
                                                             "architecture", "road"}]
    else:
        keys = lib.hdris(tags_any={"forest"}, **base)
        keys = [k for k in keys if not set(lib.items[k].get("tags", [])) & ({"snow", "winter"} | manmade)
                and not k.startswith("ph:missile")]
        if autumn:
            keys = [k for k in keys if "autumn" in lib.items[k].get("tags", [])] or keys
        elif kind in ("conifer", "birch"):
            keys = [k for k in keys if "autumn" not in lib.items[k].get("tags", [])] or keys
    key = lib.pick(ctx.rng, keys)
    # a closed canopy shades the floor 2-3 stops below the open-sky meter reading; a camera under it
    # exposes for the floor (r3 previews at the old +0.4..1.0: mean luminance 0.06-0.11 at the drone's eye;
    # the open burnt stand was fine at +0.1: 0.23-0.34)
    ev = {"broadleaf": 3.2, "mixed": 3.0, "conifer": 2.4, "birch": 3.2, "snowy": 1.4, "burnt": 0.2}[kind]
    return ctx.looks.sky(key, target_ev=ev)


class Stand:
    """The trees of one forest: species mix, bark, crowns, and the builder."""

    def __init__(self, ctx, kind, autumn):
        self.ctx, self.kind, self.autumn = ctx, kind, autumn
        rng, lk = ctx.rng, ctx.looks
        fam = {"conifer": "conifer", "snowy": "conifer", "burnt": "conifer", "broadleaf": "broadleaf",
               "birch": "birch", "mixed": "broadleaf"}[kind]
        barks = U.mats_existing(ctx, BARK[fam])
        self.barks = []
        for k in rng.choice(barks, size=min(2, len(barks)), replace=False):
            tile = float(ctx.lib.items[k].get("tile_m") or 1.6) * float(rng.uniform(0.6, 0.85))
            m = lk.pbr(str(k), tile, vary=0.2, jitter=0.12, bump=0.5)
            if kind == "burnt":
                m = U.coat(lk, lk.pbr(str(k), tile, vary=0.3, jitter=0.25, tint=(0.5, 0.46, 0.43), bump=0.6),
                           (0.02, 0.018, 0.016, 0.97), mode="low", lo=float(rng.uniform(1.0, 2.5)),
                           hi=float(rng.uniform(3.5, 7.0)), patchy=0.75, patch_scale=0.45)
            elif kind == "snowy":
                m = U.coat(lk, m, ctx.lib.pick(rng, U.mats_existing(ctx, SNOW)), 1.5, mode="up", lo=0.3, hi=0.7,
                           patchy=0.2)
            elif rng.random() < 0.7:                 # moss creeping up the foot of the trunk
                m = U.coat(lk, m, ctx.lib.pick(rng, U.mats_existing(ctx, MOSS)), 0.6, mode="low",
                           lo=float(rng.uniform(0.2, 0.6)), hi=float(rng.uniform(0.9, 2.2)), patchy=0.6,
                           patch_scale=0.5)
            self.barks.append(m)
        if kind == "mixed":                           # a conifer bark for the firs among the broadleaves
            k = ctx.lib.pick(rng, U.mats_existing(ctx, BARK["conifer"]))
            self.fir_bark = lk.pbr(k, 1.2, vary=0.2, jitter=0.12, bump=0.5)
        else:
            self.fir_bark = self.barks[0]
        # species: (crown pieces, weight, trunk radius range, crown scale range, crown base range)
        if kind in ("conifer", "snowy"):
            self.species = [(FIR, 1.0, (0.13, 0.3), (1.15, 2.0), (4.8, 7.5))]
        if autumn:                                    # the broadleaf crowns turn
            U.autumn_piece(ctx, BROAD[0], hue=float(rng.uniform(0.33, 0.41)), sat=1.3)
        if kind in ("conifer", "snowy"):
            pass
        elif kind == "broadleaf":
            self.species = [(BROAD, 1.0, (0.16, 0.36), (1.9, 2.8), (3.6, 5.2))]
        elif kind == "birch":
            self.species = [(BROAD, 1.0, (0.08, 0.16), (1.5, 2.2), (3.8, 5.5))]
        elif kind == "mixed":
            self.species = [(FIR, 0.5, (0.14, 0.3), (1.15, 1.9), (4.8, 7.0)),
                            (BROAD, 0.5, (0.16, 0.34), (1.9, 2.6), (3.6, 5.0))]
        else:                                         # burnt: bare snags, no canopy
            self.species = [([], 1.0, (0.12, 0.34), (1.0, 1.0), (4.0, 14.0))]
        w = np.array([s[1] for s in self.species])
        self.w = w / w.sum()
        if any(sp[0] is BROAD for sp in self.species):  # broadleaf crown stems in their trunks' bark (barks[0])
            U.rebark_piece(ctx, BROAD[0], self.barks[0])
        self.dead = lk.pbr(ctx.lib.pick(rng, U.mats_existing(ctx, BARK["conifer"] + BARK["broadleaf"])), 1.0,
                           tint=(0.55, 0.5, 0.45) if kind != "burnt" else (0.12, 0.11, 0.1), vary=0.25, bump=0.6)
        self.trees: list[dict] = []

    def pick(self):
        i = int(self.ctx.rng.choice(len(self.species), p=self.w))
        return self.species[i]

    def tree(self, xy, coll="structure", sp=None, r0=None, low_crown=False):
        """One tree at xy: trunk + crown (or a burnt snag). Returns its record."""
        ctx, rng = self.ctx, self.ctx.rng
        crowns, _, rr, sr, zr = sp or self.pick()
        r0 = float(rng.uniform(*rr)) if r0 is None else r0
        lean = rng.normal(0, 0.012, 2)
        bend = float(rng.uniform(0.0, 0.25))
        fir = crowns is FIR
        mat = self.fir_bark if (fir and self.kind == "mixed") else self.barks[int(rng.integers(len(self.barks)))]
        if crowns is BROAD:                           # the crown's stem carries barks[0]: its trunk does too
            mat = self.barks[0]
        if not crowns:                                # burnt snag: broken top, a few charred limbs
            h = float(rng.uniform(*zr))
            U.trunk(ctx, xy, h, r0, mat, r_top=r0 * float(rng.uniform(0.35, 0.7)), lean=lean, bend=bend,
                    flare=0.3, coll=coll)
            for _ in range(int(rng.integers(0, 4))):
                z = float(rng.uniform(max(3.0, h * 0.4), h * 0.95))
                a = float(rng.uniform(0, 2 * math.pi))
                L = float(rng.uniform(0.6, 2.2))
                p = np.array([xy[0], xy[1], z]) + np.array([lean[0] * z, lean[1] * z, 0])
                q = p + np.array([L * math.cos(a), L * math.sin(a), L * float(rng.uniform(-0.3, 0.6))])
                if min(z, q[2]) > U.cloud_top(ctx) + 0.05 or not ctx.in_zone(q[:2]):   # none in the flight band
                    U.limb(ctx, p, q, 0.05, 0.015, mat, coll=coll)
            rec = {"xy": np.asarray(xy, float), "r": r0, "h": h}
            self.trees.append(rec)
            return rec
        s = float(rng.uniform(*sr))
        pkey = crowns[int(rng.integers(len(crowns)))]
        ceil = U.cloud_top(ctx)
        R = U.piece_r(ctx, pkey, s) + float(np.linalg.norm(U.base_offset(ctx, pkey))) * s + 0.3  # any yaw
        if ctx.rect_meets_zone(np.asarray(xy) - R, np.asarray(xy) + R):
            # a crown that can reach the zone starts above the cloud ceiling, so none of it (its stem
            # included: 0.16-0.25 m thick at these scales, wider than the trunk top) is rendered
            # below the ceiling outside the cloud; the procedural trunk is all the labeller sees
            zc = float(rng.uniform(max(zr[0], ceil + 0.05), max(zr[1], ceil + 0.6)))
        elif low_crown:                               # scenery only: its crown cannot reach the zone
            zc = float(rng.uniform(2.2, zr[0]))
        else:
            zc = float(rng.uniform(*zr))
        h = zc + (0.9 if fir else 1.6)
        bdir = rng.normal(0, 1, 2)
        bdir /= np.linalg.norm(bdir) + 1e-9
        U.trunk(ctx, xy, h, r0, mat, r_top=max(0.05, (0.045 if fir else 0.06) * s), lean=lean, bend=bend,
                flare=float(rng.uniform(0.2, 0.5)), coll=coll, bdir=bdir)
        # on the trunk's centre line where the crown starts (the bow puts it up to 0.2 m off the top)
        U.crown(ctx, pkey, U.trunk_axis(xy, h, lean, bend, bdir, zc), zc, s,
                coll="backdrop" if coll == "backdrop" else "scatter")
        # dead twigs on conifer trunks below the living crown
        if fir and ctx.in_zone(xy, 3.0) and rng.random() < 0.6:
            for _ in range(int(rng.integers(2, 6))):
                z = float(rng.uniform(1.8, zc))
                a = float(rng.uniform(0, 2 * math.pi))
                L = float(rng.uniform(0.25, 0.55))
                r_here = r0 * (1 - 0.45 * z / h)
                p = np.array([xy[0] + lean[0] * z + r_here * 0.8 * math.cos(a),
                              xy[1] + lean[1] * z + r_here * 0.8 * math.sin(a), z])
                q = p + np.array([L * math.cos(a), L * math.sin(a), -0.25 * L])
                U.limb(ctx, p, q, 0.025, 0.008, self.dead, segs=5, coll=coll)
        rec = {"xy": np.asarray(xy, float), "r": r0, "h": h, "zc": zc}
        self.trees.append(rec)
        return rec


def _poisson(rng, rect, dmin, n_try, existing=()):
    """Dart-throwing blue noise in ``rect`` with spacing ``dmin``."""
    x0, y0, x1, y1 = rect
    cell = dmin / math.sqrt(2)
    grid: dict = {}
    pts = []
    for p in existing:
        grid.setdefault((int(p[0] // cell), int(p[1] // cell)), []).append(np.asarray(p, float))
    P = np.column_stack([rng.uniform(x0, x1, n_try), rng.uniform(y0, y1, n_try)])
    for p in P:
        gx, gy = int(p[0] // cell), int(p[1] // cell)
        ok = True
        for i in range(gx - 2, gx + 3):
            for j in range(gy - 2, gy + 3):
                for q in grid.get((i, j), ()):
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < dmin * dmin:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault((gx, gy), []).append(p)
            pts.append(p)
    return pts


def _dress(ctx, kind, stand, rng, lk):
    """Materials for the patch primitives, then per-object touches."""
    lib = ctx.lib
    snowy = kind == "snowy"
    topcoat = (lib.pick(rng, U.mats_existing(ctx, SNOW)), 1.5) if snowy else \
        (lib.pick(rng, U.mats_existing(ctx, MOSS)), 0.7)
    style = str(rng.choice(["rock", "stone", "planks"], p=[0.45, 0.4, 0.15]))
    if kind == "burnt":
        style = str(rng.choice(["rock", "stone"]))
    if style == "rock":
        k = lib.pick(rng, U.mats_existing(ctx, ROCK))
        wall = lk.pbr(k, float(rng.uniform(2.5, 4.0)), bevel=0.04, dirt=0.3, vary=0.25, bump=0.6)
    elif style == "stone":
        k = lib.pick(rng, U.mats_existing(ctx, STONEWALL))
        wall = lk.pbr(k, float(lib.items[k].get("tile_m") or 2.5) * float(rng.uniform(0.8, 1.1)), bevel=0.03,
                      dirt=0.4, vary=0.2, bump=0.5)
    else:
        k = lib.pick(rng, U.mats_existing(ctx, PLANKS))
        wall = lk.pbr(k, 2.0, bevel=0.01, dirt=0.5, vary=0.2, tint=(0.8, 0.75, 0.7))
    if kind != "burnt":
        wall = U.coat(lk, wall, topcoat[0], topcoat[1], mode="up", lo=0.5, hi=0.85, patchy=0.3 if snowy else 0.5)
    if not snowy and kind != "burnt" and rng.random() < 0.6:     # damp green at the foot of the wall
        wall = U.coat(lk, wall, lib.pick(rng, U.mats_existing(ctx, MOSS)), 0.7, mode="low", lo=0.1,
                      hi=float(rng.uniform(0.5, 1.2)), patchy=0.6, patch_scale=0.6)
    rk = lib.pick(rng, U.mats_existing(ctx, ROCK))
    rock = lk.pbr(rk, 1.5, bevel=0.02, dirt=0.2, vary=0.25, bump=0.8)
    if kind != "burnt":
        rock = U.coat(lk, rock, topcoat[0], topcoat[1], mode="up", lo=0.2, hi=0.6, patchy=0.5)
    pole = stand.barks[0]
    log = U.coat(lk, stand.dead, topcoat[0], topcoat[1], mode="up", lo=0.5, hi=0.9, patchy=0.5) \
        if kind != "burnt" else stand.dead
    ctx.dress_primitives({"wall": wall, "box": wall if style != "planks" else rock, "pole": pole, "bar": log,
                          "rock": rock})
    regions = U.claim_family_regions(ctx, keepout=float(ctx.knobs.get("keepout", 2.5)))
    prims = ctx.spec["primitives"]
    objs = {o.name: o for o in ctx.colls["obstacles"].objects}
    post = lk.pbr(lib.pick(rng, U.mats_existing(ctx, PLANKS)), 1.5, bevel=0.01, dirt=0.5, tint=(0.7, 0.65, 0.6))
    nest = lk.pbr(lib.pick(rng, U.mats_existing(ctx, MOSS)), 0.5, vary=0.3, bump=1.0) if not snowy else rock
    vine = lk.pbr(lib.pick(rng, U.mats_existing(ctx, BARK["broadleaf"])), 0.5, tint=(0.5, 0.5, 0.4))
    n_v = sum(1 for p in prims if p["kind"] == "cyl_v")
    ceil = U.cloud_top(ctx)
    info = {"wall_style": style, "trees_on_poles": 0}
    sem = []                                       # what the dressing adds to the family's own geometry
    flared = []
    for i, p in enumerate(prims):
        k = p["kind"]
        if k == "cyl_v":
            o = objs.get(f"prim{i:02d}_cyl")
            region = str(p.get("region", ""))
            standlike = region.startswith(("stand", "forest"))  # a stand's poles are trees whatever their radius
            thin_ring = region.startswith("ring") or (p["r"] < 0.22 and n_v > 12 and not standlike)
            if p["z1"] >= 4.4 and p["r"] >= 0.12 and not thin_ring:
                if U.root_flare(ctx, p["c"], p["r"], pole) is not None:
                    flared.append(i)
            if p["z1"] >= 4.4 and p["r"] >= 0.12 and not thin_ring and kind == "burnt":
                # a charred snag: the pole goes on up and ends in a broken top
                U.trunk(ctx, p["c"], float(rng.uniform(1.5, 6.0)), p["r"] * 0.97, pole,
                        r_top=p["r"] * float(rng.uniform(0.35, 0.7)), flare=0.0, rough=0.03, z0=p["z1"] - 0.02,
                        cap=True, coll="structure")
                if p["z1"] < ceil:
                    sem.append({"prim": i, "kind": k, "change": "trunk_above",
                                "detail": f"snag continued up from z1 {p['z1']:.2f} m"})
            elif p["z1"] >= 4.4 and p["r"] >= 0.12 and not thin_ring:
                # the pole is a trunk: continue it tapering above the cloud ceiling, crown on top
                sp = stand.pick()
                crowns, _, _, sr, _ = sp
                if crowns:
                    s = float(rng.uniform(*sr))
                    fir = crowns is FIR
                    z1 = p["z1"]
                    h_up = 2.2 + 4.0 * p["r"]           # a thick pole is an old tree: its trunk runs on up
                    U.trunk(ctx, p["c"], h_up, p["r"] * 0.97, pole,
                            r_top=max(0.05, (0.045 if fir else 0.06) * s, 0.3 * p["r"]),
                            flare=0.0, rough=0.02, z0=z1 - 0.02, coll="structure")
                    U.crown(ctx, crowns[int(rng.integers(len(crowns)))], p["c"],
                            max(z1 + (0.9 if fir else 0.4), ceil + 0.05), s)
                    info["trees_on_poles"] += 1
                    if z1 < ceil:
                        sem.append({"prim": i, "kind": k, "change": "trunk_above",
                                    "detail": f"trunk continued up from z1 {z1:.2f} m"})
            elif thin_ring or (p["r"] < 0.2 and not standlike):
                if o is not None:
                    o.data.materials[0] = post
        elif k == "sphere":
            o = objs.get(f"prim{i:02d}_sph")
            if p["c"][2] - p["r"] > 0.4:          # floating: a moss ball / nest hung from a vine
                if o is not None:
                    o.data.materials[0] = nest
                c = p["c"]
                ctx.cylinder((c[0], c[1], c[2] + p["r"] * 0.9), (c[0], c[1], 9.0), 0.012, vine, coll="scatter",
                             name="vine")
                sem.append({"prim": i, "kind": k, "change": "vine_above",
                            "detail": f"vine r 0.012 m from the sphere top ({c[2] + p['r']:.2f} m) up"})
        elif k == "box":
            lo_z = p["c"][2] - p["half"][2]
            if lo_z > 0.35 and max(p["half"][0], p["half"][1]) > 0.8 and abs(p["rpy"][0]) + abs(p["rpy"][1]) < 1e-3:
                # a raised timber platform / hunting stand: posts under its ends
                o = objs.get(f"prim{i:02d}_box")
                if o is not None:
                    o.data.materials[0] = post
                yaw = p["rpy"][2]
                cy, sy = math.cos(yaw), math.sin(yaw)
                hx, hy = p["half"][0], p["half"][1]
                for u in (-1, 1):
                    for v in (-1, 1):
                        du, dv = u * max(hx - 0.1, 0.0), v * max(hy - 0.1, 0.0)
                        px = p["c"][0] + cy * du - sy * dv
                        py = p["c"][1] + sy * du + cy * dv
                        ctx.cylinder((px, py, 0.0), (px, py, lo_z + 0.01), 0.07, stand.dead, coll="structure",
                                     name="stilt")
                sem.append({"prim": i, "kind": k, "change": "posts_under",
                            "detail": f"4 stilts r 0.07 m under the corners, 0-{lo_z:.2f} m"})
        elif k == "cyl_h":
            a, b = np.asarray(p["a"], float), np.asarray(p["b"], float)
            d = (b - a)[:2] / max(np.linalg.norm((b - a)[:2]), 1e-9)
            for end, sgn in ((a, -1), (b, 1)):        # a trunk at each end holds the fallen tree up
                r = float(rng.uniform(0.18, 0.3))
                xy = end[:2] + sgn * d * (r + 0.03)
                if abs(end[2]) < 0.5 or not ctx.free(xy, r, 0.02):
                    continue
                stand.tree(xy, r0=r)
                ctx.claim_disc(xy, r * 1.3, kind="tree")
                sem.append({"prim": i, "kind": k, "change": "tree_at_end",
                            "detail": f"a trunk r {r:.2f} m at ({xy[0]:.2f}, {xy[1]:.2f}) holding the bar up"})
    if flared:
        sem.append({"prims": flared, "kind": "cyl_v", "change": "root_flare",
                    "detail": "root collar round the pole foot: <= 1.45 r at the ground, 1.16 r at 0.3 m, "
                              "1.07 r at 0.55 m, flush at 0.8 m"})
    info["semantics_changed"] = sem
    info["family_regions"] = regions
    return info


def _decal(ctx, mat, c, size, yaw, z=0.006):
    ca, sa = math.cos(yaw), math.sin(yaw)
    h = size / 2
    V = [(c[0] + ca * u - sa * v, c[1] + sa * u + ca * v, z) for u, v in ((-h, -h), (h, -h), (h, h), (-h, h))]
    return ctx.mesh(V, [(0, 1, 2, 3)], mat, uvs=[[(0, 0), (1, 0), (1, 1), (0, 1)]], coll="ground", collide=False,
                    name="decal")


def build(ctx) -> dict:
    t0 = time.time()
    tick = lambda what: print(f"[forest] {what} {time.time() - t0:.1f} s", flush=True)  # noqa: E731
    rng, lk, lib = ctx.rng, ctx.looks, ctx.lib
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    names = list(KINDS)
    kind = str(rng.choice(names, p=np.array([KINDS[k] for k in names]) / sum(KINDS.values())))
    autumn = kind in ("broadleaf", "mixed") and rng.random() < 0.5
    force = os.environ.get("FOREST_KIND", "")          # test hook: FOREST_KIND=conifer[:autumn]
    if force:
        kind, autumn = force.split(":")[0], force.endswith(":autumn")
    sky = _sky(ctx, kind, autumn)
    clutter = float(ctx.knobs.get("clutter", 1.0))
    keepout = float(ctx.knobs.get("keepout", 2.5))

    # --- the floor -----------------------------------------------------------------------------
    gset = {"conifer": "conifer", "broadleaf": "autumn" if autumn else "summer", "mixed": "autumn" if autumn else
            "conifer", "birch": "birch", "burnt": "burnt", "snowy": "snowy"}[kind]
    gk = U.mats_existing(ctx, GROUND[gset])
    gkeys = [str(k) for k in rng.choice(gk, size=3, replace=len(gk) < 3)]
    wet = 0.0                                          # puddles read as black holes in leaf litter
    ctx.ground(lk.ground_material(gkeys, tile_m=float(rng.uniform(2.0, 3.0)), puddles=wet,
                                  blend_scale=float(rng.uniform(5.0, 10.0))))

    tick("ground+sky")
    stand = Stand(ctx, kind, autumn)
    dress = _dress(ctx, kind, stand, rng, lk)

    tick("dressed")
    # --- a trail, through the obstacle's opening when there is one ------------------------------
    trail = None
    if rng.random() < 0.55 and kind != "snowy" or (kind == "snowy" and rng.random() < 0.3):
        g = ctx.goal
        via = []
        for p in ctx.spec["primitives"]:
            if p["kind"] == "wall":
                p0, p1 = np.asarray(p["p0"], float), np.asarray(p["p1"], float)
                d = (p1 - p0) / np.linalg.norm(p1 - p0)
                for s_, w, sz0, sz1 in p.get("slots", []):
                    if sz0 < 0.05 and w > 0.9:
                        via.append(p0 + d * s_)
        if rng.random() < 0.6:
            anchors = [np.array([zx0 - 20, float(rng.uniform(-3, 3))]), np.zeros(2), *via, g,
                       np.array([zx1 + 20, g[1] + float(rng.uniform(-6, 6))])]
        else:                                        # across the patch at an angle
            xm = float(rng.uniform(x0 + 4, x1 - 4))
            anchors = [np.array([xm - 12, zy0 - 20]), np.array([xm + float(rng.normal(0, 3)), 0.0]),
                       np.array([xm + 12, zy1 + 20])]
        anchors = sorted(anchors, key=lambda a: a[0]) if len(anchors) > 3 else anchors
        # thin anchors closer than 4 m (start, slot and goal can crowd), then a smooth curve
        keep = [anchors[0]]
        for a in anchors[1:]:
            if np.linalg.norm(a - keep[-1]) > 4.0:
                keep.append(a)
        pts = U.spline(keep, step=0.7, rng=rng, wobble=0.5)
        width = float(rng.uniform(0.9, 1.6))
        tk = lib.pick(rng, U.mats_existing(ctx, TRAIL if kind != "snowy" else ["ph:snow_03", "ph:snow_05",
                                                                             "ph:snow_04"]))
        U.ribbon(ctx, pts, width, lk.pbr(tk, 2.0, proj="flat", vary=0.25, jitter=0.0, tint=(0.7, 0.66, 0.62)),
                 z=U.flat_z(ctx), jitter=0.25)
        U.claim_polyline(ctx, pts, width / 2 + 0.3, seg=3.0, kind="trail")
        trail = {"material": tk, "width": round(width, 2)}

    # --- the stand: patch, margin, scenery -----------------------------------------------------
    span = max(x1 - x0, y1 - y0)                      # keep the preview cameras out of the crowns
    cams = [np.array([(x0 + x1) / 2, y0 - 0.5 * span]), np.array([x0 - 3.0, (y0 + y1) / 2])]
    dens = float(rng.uniform(0.75, 1.3))              # this forest's density
    sp_patch = {"birch": 3.0, "burnt": 3.6}.get(kind, 3.8) / math.sqrt(dens * max(clutter, 0.3))
    sp_near = {"birch": 2.6, "burnt": 3.0}.get(kind, 3.3) / math.sqrt(dens)
    trees_in = 0
    tree_margin = max(0.7, 0.45 * keepout)
    pts = _poisson(rng, (x0 + 0.3, y0 + 0.3, x1 - 0.3, y1 - 0.3), sp_patch, 2500)
    for p in pts:
        sp = stand.pick()
        r = float(rng.uniform(*sp[2]))
        if ctx.clearance(p) < r * 1.35 + tree_margin:
            continue
        stand.tree(p, sp=sp, r0=r)
        ctx.claim_disc(p, r * 1.35, kind="tree")
        trees_in += 1
    # the rest of the cloud zone
    ring = [q for q in _poisson(rng, (zx0, zy0, zx1, zy1), sp_near, 4000,
                                existing=[t["xy"] for t in stand.trees])
            if not (x0 <= q[0] <= x1 and y0 <= q[1] <= y1)]
    for p in ring:
        if np.linalg.norm(p - cams[1]) < 1.5 or np.linalg.norm(p - cams[0]) < 8.0:
            continue
        sp = stand.pick()
        r = float(rng.uniform(*sp[2]))
        if ctx.clearance(p) < r * 1.35 + 0.3:
            continue
        stand.tree(p, sp=sp, r0=r)
        ctx.claim_disc(p, r * 1.35, kind="tree")
    # scenery out to ~55 m: denser near, sparser far (render only)
    far = (zx0 - 55, zy0 - 55, zx1 + 55, zy1 + 55)
    ex = [t["xy"] for t in stand.trees]
    n_scen = 0
    for p in _poisson(rng, far, sp_near * 1.15, 16000, existing=ex):
        if ctx.in_zone(p, 0.5) or np.linalg.norm(p - cams[0]) < 8.0:
            continue
        dz = max(zx0 - p[0], p[0] - zx1, zy0 - p[1], p[1] - zy1)
        if dz > 25 and rng.random() < 0.45:
            continue
        stand.tree(p, coll="backdrop", low_crown=dz > 8.0 and rng.random() < 0.2)   # a low crown, clear of the zone
        n_scen += 1

    tick("stand")
    # --- the understory ---------------------------------------------------------------------------
    under = {"conifer": ["ph:fir_sapling_medium#1", "ph:fir_sapling_medium#2"],
             "snowy": ["ph:fir_sapling_medium#1", "ph:fir_sapling_medium#2"],
             "mixed": ["ph:fir_sapling_medium#2", "ph:shrub_02#0", "ph:shrub_02#2", "ph:searsia_lucida#0"],
             "broadleaf": ["ph:shrub_02#0", "ph:shrub_02#1", "ph:shrub_02#2", "ph:searsia_lucida#0",
                           "ph:searsia_lucida#1", "ph:searsia_lucida#2"],
             "birch": ["ph:shrub_02#0", "ph:shrub_02#2", "ph:searsia_lucida#2", "ph:searsia_lucida#3"],
             "burnt": []}[kind]
    young_s = (0.35, 0.65)
    # in the zone a young tree collides and is realised for the cloud: a 0.4-0.7 M-triangle fir sapling
    # there costs more than the whole patch, so young conifers in the zone are the light saplings scaled up
    light = [k for k in ("ph:fir_sapling#0", "ph:fir_sapling#1", "ph:pine_sapling_small#0", "ph:pine_sapling_small#1")
             if k.split("#")[0] in lib.items]
    if under:
        # young trees / bushes: margin + scenery generously, a few in the patch
        for p in U.free_spots(ctx, int(26 * dens), (zx0 - 25, zy0 - 25, zx1 + 25, zy1 + 25), 1.5, 0.3, rng,
                              where=lambda q: not ctx.in_zone(q, -0.5) or not (x0 - 1 <= q[0] <= x1 + 1 and
                                                                               y0 - 1 <= q[1] <= y1 + 1)):
            k = under[int(rng.integers(len(under)))]
            s = float(rng.uniform(*young_s)) if "fir" in k else float(rng.uniform(0.8, 1.2))
            inz = ctx.in_zone(p, U.piece_r(ctx, k, s))
            if inz and "fir" in k and light:
                k, s = light[int(rng.integers(len(light)))], float(rng.uniform(1.4, 2.2))
            if inz and not ctx.free(p, U.piece_r(ctx, k, s), 0.2):
                continue
            ctx.instance(k, p, yaw=float(rng.uniform(0, 6.283)), s=s,
                         coll="scatter" if inz else "backdrop", claim=inz)
        n_young = ctx.count(1.2 if kind != "birch" else 2.0)
        firs, rest = [k for k in under if "fir" in k], [k for k in under if "fir" not in k]
        n_fir = int(round(n_young * len(firs) / len(under)))
        U.scatter(ctx, light or firs, n_fir, margin=max(0.6, 0.35 * keepout), s=(1.4, 2.2) if light else (0.35, 0.6))
        U.scatter(ctx, rest, n_young - n_fir, margin=max(0.6, 0.35 * keepout), s=(0.7, 1.0))

    # small saplings / ferns / grass / flowers / roots: no claims, just clear of footprints
    cover = {
        "conifer": ["ph:fern_02#0", "ph:fern_02#1", "ph:fern_02#2", "ph:fern_02#3", "ph:grass_medium_01#0",
                    "ph:grass_medium_01#1", "ph:grass_medium_01#2", "ph:grass_medium_02#3"],
        "mixed": ["ph:fern_02#0", "ph:fern_02#1", "ph:fern_02#2", "ph:grass_medium_01#0", "ph:grass_medium_01#1",
                  "ph:grass_medium_02#3", "ph:grass_medium_02#4"],
        "broadleaf": ["ph:fern_02#0", "ph:fern_02#1", "ph:fern_02#2", "ph:grass_medium_02#3", "ph:grass_medium_02#4",
                      "ph:shrub_03#0", "ph:shrub_03#1", "ph:nettle_plant#4", "ph:nettle_plant#5",
                      "ph:celandine_01#0", "ph:periwinkle_plant#0", "ph:grass_medium_01#0"],
        "birch": ["ph:grass_medium_02#3", "ph:grass_medium_02#4", "ph:grass_medium_02#2", "ph:grass_medium_01#0",
                  "ph:grass_medium_01#1", "ph:grass_medium_01#2", "ph:fern_02#1", "ph:dandelion_01#2",
                  "ph:celandine_01#0", "ph:shrub_03#0"],
        "burnt": ["ph:grass_medium_01#11", "ph:grass_medium_01#0", "ph:grass_medium_02#0", "ph:grass_medium_02#1",
                  "ph:grass_medium_02#3", "ph:weed_plant_02#0", "ph:fern_02#1", "ph:fern_02#3"],
        "snowy": ["ph:grass_medium_02#1", "ph:grass_medium_02#2", "ph:grass_medium_02#0"],
    }[kind]
    cover = [k for k in cover if k.split("#")[0] in lib.items]
    ferns = [k for k in cover if "fern" in k or "sapling" in k or "shrub_03" in k or "nettle" in k]
    grasses = [k for k in cover if k not in ferns] or cover
    # plants grow in clumps: cluster centres, then 3-14 plants of one or two kinds round each
    dens_c = {"snowy": 0.35, "burnt": 0.8}.get(kind, 1.0) * dens
    n_clumps = int((ctx.count(55) + 45) * dens_c)
    cov_rect = (zx0 - 12, zy0 - 12, zx1 + 12, zy1 + 12)
    n_cov = 0
    for c in U.free_spots(ctx, n_clumps, cov_rect, 0.3, 0.2, rng, skip=U.LOW):
        pool = ferns if (ferns and rng.random() < 0.45) else grasses
        kinds = [pool[int(rng.integers(len(pool)))] for _ in range(2)]
        m = int(rng.integers(3, 15))
        rad = float(rng.uniform(0.6, 2.0))
        P = c + rng.normal(0, rad / 2, (m, 2))
        ks = [kinds[int(rng.integers(2))] for _ in range(m)]
        scs = [float(rng.uniform(0.8, 1.5)) if "fern" in k else float(rng.uniform(0.7, 1.3)) for k in ks]
        rad_p = np.array([U.piece_r(ctx, k, sc) for k, sc in zip(ks, scs)])
        ok = (U.clear_many(ctx, P, U.LOW) >= 0.25) & (U.clear_of(ctx, P) >= rad_p + 0.05)
        for p, k, sc in zip(P[ok], np.asarray(ks)[ok], np.asarray(scs)[ok]):
            ctx.instance(str(k), p, yaw=float(rng.uniform(0, 6.283)), s=float(sc), coll="scatter")
            n_cov += 1
    # a few seedlings (150 k triangles each: singles, not clumps)
    seed = [k for k in ("ph:fir_sapling#0", "ph:fir_sapling#1", "ph:pine_sapling_small#1")
            if k.split("#")[0] in lib.items] if kind in ("conifer", "mixed", "snowy") else []
    for p in U.free_spots(ctx, 5 if seed else 0, (zx0, zy0, zx1, zy1), 0.5, 0.3, rng):
        k, sc = seed[int(rng.integers(len(seed)))], float(rng.uniform(0.8, 1.4))
        if U.clear_of(ctx, p)[0] >= U.piece_r(ctx, k, sc) + 0.1:
            ctx.instance(k, p, yaw=float(rng.uniform(0, 6.283)), s=sc, coll="scatter")
    # the forest floor litter: leaf / needle decals everywhere near the flight path
    # every card fades to nothing toward its rim (U.soft_decal): a drift, never a square
    litter = U.litter_materials(lib)               # cut-out scans only (an opaque carpet renders as a disc)
    if kind in ("snowy", "burnt"):                   # snow covers it; a fire has burnt it
        litter = []
    if litter:
        ltint = {"burnt": (0.35, 0.33, 0.3), "conifer": (0.75, 0.6, 0.5), "birch": (0.85, 0.9, 0.7)}.get(kind)
        lmats = [U.soft_decal(lk.pbr(k, None, vary=0.25, jitter=0.0, tint=ltint))
                 for k in rng.choice(litter, size=min(2, len(litter)), replace=False)]
        for p in U.free_spots(ctx, ctx.count(40) + 30, (zx0 - 6, zy0 - 6, zx1 + 6, zy1 + 6), 0.0, -3.0, rng,
                              skip=U.LOW):
            _decal(ctx, lmats[int(rng.integers(len(lmats)))], p, float(rng.uniform(1.5, 4.5)),
                   float(rng.uniform(0, 6.283)), z=U.flat_z(ctx))

    # roots at tree feet, leaf litter
    roots = [k for k in ["ph:pine_roots#0", "ph:root_cluster_02#1", "ph:root_cluster_02#3", "ph:root_cluster_02#4",
                         "ph:root_cluster_02#5", "ph:root_cluster_02#6"] if k.split("#")[0] in lib.items]
    for t in stand.trees:
        if not ctx.in_zone(t["xy"], 6.0) or kind == "snowy":
            continue
        if roots and rng.random() < 0.35 and t["r"] > 0.15:
            k = roots[int(rng.integers(len(roots)))]
            a = float(rng.uniform(0, 6.283))
            off = t["xy"] + (t["r"] + 0.3) * np.array([math.cos(a), math.sin(a)])
            sc = float(rng.uniform(0.8, 1.2))
            # a root clump may run into its own tree's claim, never into the start/goal or a primitive
            if U.clear_of(ctx, off[None])[0] >= U.piece_r(ctx, k, sc) + 0.05:
                ctx.instance(k, off, yaw=a + math.pi / 2, s=sc, coll="scatter")

    tick("cover")
    # --- deadwood and rock -------------------------------------------------------------------------
    rocks = lib.models(ids=["rock_moss_set_01", "rock_moss_set_02", "boulder_01"])
    stumps = lib.models(ids=["tree_stump_01", "tree_stump_02"])
    logs = lib.models(ids=["dead_tree_trunk", "dead_tree_trunk_02"])
    sticks = lib.models(ids=["dry_branches_medium_01"])
    if kind == "snowy":
        for k in rocks + stumps + logs:
            U.coat_piece(ctx, k, lib.pick(rng, U.mats_existing(ctx, SNOW)), 1.5, mode="up", lo=0.25, hi=0.6,
                         patchy=0.2)
    if kind == "burnt":
        for k in stumps + logs:
            U.tint_piece(ctx, k, (0.15, 0.14, 0.13))
    U.scatter(ctx, rocks, ctx.count(3.0 * dens) + 1, margin=max(0.6, 0.4 * keepout), s=(0.5, 1.1))
    U.scatter(ctx, stumps, ctx.count(2.5), margin=0.5, s=(0.6, 1.0))
    U.scatter(ctx, logs, ctx.count(1.5), margin=max(0.5, 0.3 * keepout), s=(0.8, 1.3))
    for p in U.free_spots(ctx, ctx.count(14) + 6, cov_rect, 0.6, 0.1, rng, skip=U.LOW):
        k, sc = sticks[int(rng.integers(len(sticks)))], float(rng.uniform(0.8, 1.5))
        if U.clear_of(ctx, p)[0] >= U.piece_r(ctx, k, sc) + 0.05:
            ctx.instance(k, p, yaw=float(rng.uniform(0, 6.283)), s=sc, coll="scatter")
    # scenery rocks, stumps and logs (render only)
    for p in U.free_spots(ctx, 40, (zx0 - 30, zy0 - 30, zx1 + 30, zy1 + 30), 1.0, 0.2, rng,
                          where=lambda q: not ctx.in_zone(q, 1.0)):
        pool = rocks if rng.random() < 0.5 else stumps + logs
        ctx.instance(pool[int(rng.integers(len(pool)))], p, yaw=float(rng.uniform(0, 6.283)),
                     s=float(rng.uniform(0.6, 1.4)), coll="backdrop")
    # procedural fallen trunks (the long ones a forest is full of)
    logmat = U.coat(lk, stand.dead, *( (lib.pick(rng, U.mats_existing(ctx, SNOW)), 1.5) if kind == "snowy" else
                                       (lib.pick(rng, U.mats_existing(ctx, MOSS)), 0.6)),
                    mode="up", lo=0.3, hi=0.8, patchy=0.5) if kind != "burnt" else stand.dead
    n_fall = ctx.count(2.0 * dens) + 4
    placed_logs = 0
    for _ in range(n_fall * 25):
        if placed_logs >= n_fall:
            break
        c = np.array([rng.uniform(zx0 - 10, zx1 + 10), rng.uniform(zy0 - 10, zy1 + 10)])
        L = float(rng.uniform(3.0, 9.0))
        r = float(rng.uniform(0.12, 0.3))
        a = float(rng.uniform(0, math.pi))
        d = np.array([math.cos(a), math.sin(a)])
        pa, pb = c - d * L / 2, c + d * L / 2
        inzone = ctx.rect_meets_zone(np.minimum(pa, pb) - r, np.maximum(pa, pb) + r)
        if inzone and not U.segment_clear(ctx, pa, pb, r + max(0.4, 0.3 * keepout)):
            continue
        U.limb(ctx, (*pa, r * 0.9), (*pb, r * 0.55), r, r * 0.6, logmat, segs=9, coll="scatter")
        if inzone:
            ctx.claim_rect(c, (L / 2 + 0.1, r + 0.1), a, kind="log")
        placed_logs += 1

    tick("deadwood")
    # --- hanging and leaning obstacles in the flight band (L3) ---------------------------------------
    n_hang = ctx.count(float(ctx.knobs.get("hang", 0.0)))
    hung = 0
    hosts = [t for t in stand.trees if x0 - 2 <= t["xy"][0] <= x1 + 2 and y0 - 2 <= t["xy"][1] <= y1 + 2]
    tries = 0
    while hung < n_hang and hosts and tries < 200:
        tries += 1
        t = hosts[int(rng.integers(len(hosts)))]
        what = rng.choice(["lean", "branch", "branch", "vine"])
        if what == "branch":                         # a low dead limb sticking out of a trunk
            z = float(rng.uniform(1.0, 3.2))
            a = float(rng.uniform(0, 2 * math.pi))
            L = float(rng.uniform(1.2, 2.8))
            dv = np.array([math.cos(a), math.sin(a)])
            p = t["xy"] + dv * t["r"] * 0.8
            q = p + dv * L
            if not U.segment_clear(ctx, p + dv * (t["r"] * 0.6 + 0.35), q, 0.25):
                continue
            U.limb(ctx, (*p, z), (*q, z + float(rng.uniform(-0.4, 0.5))), 0.06, 0.015, stand.dead, droop=0.1)
            U.claim_segment(ctx, p + dv * t["r"], q, 0.1)
        elif what == "lean":                         # a dead tree fallen against a living one
            a = float(rng.uniform(0, 2 * math.pi))
            L = float(rng.uniform(3.5, 6.0))
            zt = float(rng.uniform(2.0, 3.6))
            run = math.sqrt(max(L * L - zt * zt, 1.0))
            dv = np.array([math.cos(a), math.sin(a)])
            top = t["xy"] + dv * (t["r"] + 0.12)
            foot = top + dv * run
            if not U.segment_clear(ctx, top + dv * 0.4, foot, 0.4):
                continue
            U.limb(ctx, (*foot, 0.1), (*top, zt), 0.14, 0.08, logmat, segs=8)
            U.claim_segment(ctx, top, foot, 0.2)
        else:                                        # a vine / liana hanging from the canopy
            a = float(rng.uniform(0, 2 * math.pi))
            p = t["xy"] + np.array([math.cos(a), math.sin(a)]) * float(rng.uniform(0.8, 2.0))
            if U.clear_many(ctx, p[None])[0] < 0.5:
                continue
            zb = float(rng.uniform(1.0, 2.2))
            U.limb(ctx, (*p, zb), (*p, 7.0), 0.02, 0.025, vine_mat(ctx, stand), segs=5)
            ctx.claim_disc(p, 0.1, kind="hang")
        hung += 1

    tick("done")
    return {"sky": sky, "forest": kind, "autumn": autumn, "env_override": force or None, "ground": gkeys, "wet": wet,
            "density": round(dens, 2),
            "trees_in_patch": trees_in, "trees_zone": sum(1 for t in stand.trees if ctx.in_zone(t["xy"])),
            "trees_scenery": n_scen, "trail": trail, "logs": placed_logs, "hanging": hung, **dress}


def vine_mat(ctx, stand):
    if not hasattr(stand, "_vine"):
        stand._vine = ctx.looks.pbr(ctx.lib.pick(ctx.rng, U.mats_existing(ctx, BARK["broadleaf"])), 0.4,
                                    tint=(0.45, 0.5, 0.35))
    return stand._vine
