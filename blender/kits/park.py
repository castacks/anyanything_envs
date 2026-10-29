"""City park: lawns crossed by gravel or paved paths, specimen trees and
avenues, hedges and flower beds, benches, bins, lamps and bollards, a
fountain or a statue on a plaza, picnic tables, a playground -- and the city
standing round it beyond the railings.

What makes it read as a park rather than props on grass:

* **layout styles**: an English *landscape* park (curving paths, open
  lawns, tree clumps, maybe a pond), a *formal* garden (straight axes, a
  round plaza with a fountain, clipped hedges and parterre beds, an
  avenue), a paved *plaza* park (trees in grates, planters, lawn islands)
  and a *playground* park (a fenced play area with swings, slide tower,
  climbing frame and seesaw on sand or rubber);
* **furniture where people put it**: benches set back from the path edge
  every 8-15 m with a bin beside some, lamps in a rhythm along one side,
  bollards at path mouths; all procedural (one merged mesh each) in cast
  iron / timber / painted steel, or Poly Haven pieces (Victorian lamp,
  picnic table, statues);
* **seasons**: summer, late summer (dry lawn), autumn (leaf-litter,
  crowns turned yellow-orange) and winter (snow lawn, cleared paths, bare
  trees and firs, snow on every up-facing surface), each under a matching
  park HDRI; lamps light up at dusk;
* **the city beyond**: past the zone the park ends at a railing, then a
  street with kerbs, markings, parked cars and street trees, then blocks
  of buildings 10-40 m tall on some sides (render-only scenery), a denser
  tree belt on the others;
* the patch's **obstacles** join in: walls become clipped hedge walls,
  brick/stone garden walls with copings or rendered walls; tall poles
  become park trees (crown above 5 m), thin ones lamp posts, a thick
  pillar a monument column; bars become pergola rails on posts; spheres
  become topiary balls (a floating one on its clipped stem).

Levels: ``clutter`` scales furniture, trees in the patch and beds;
``keepout`` keeps everything back from openings; ``hang`` (L3) strings
festoon lights across paths, puts up a badminton net and grows low limbs
off the patch's trees (planting one when there is none).

Contract notes: a full library tree in the zone (olive, shrub tree, fir)
keeps its whole canopy -- it hangs in the flight band -- clear of every
claim but a path's and claims it, a bare tree its limbs' reach; a crown on a trunk
starts above the cloud ceiling (``_forest_util.cloud_top``), stem and all;
scenery trees keep their canopies out of the zone; festoon cables cross
only paths, beds and benches (``UNDER_CABLE``), never the start and goal
discs, an opening, a family's region, a tree or a lamp. What the dressing
adds to the family's own geometry below the cloud ceiling (wall copings, posts under raised boxes
and bar ends, a topiary stem under a floating ball, capitals, lamp heads) is
listed in the result's ``semantics_changed`` ({prim, kind, change, detail}).
``PARK_STYLE=<style>[:<season>]`` forces the layout (a test hook; recorded
as ``env_override``, never set by the pipeline).
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from kits import _forest_util as U

NAME = "park"
INDOOR = False
DESCRIPTION = "city park: lawns, paths, trees, hedges, benches, lamps, fountain/statue, playground; city beyond"

GRASS = ["acg:Grass001", "acg:Grass002", "acg:Grass004", "acg:Grass005", "acg:Grass006", "acg:Grass007",
         "acg:Grass008", "acg:Grass003"]
WORN = ["ph:park_dirt", "acg:Ground003", "ph:grass_path_2", "ph:grass_path_3", "ph:sparse_grass",
        "ph:forrest_ground_01", "ph:grass_ground", "acg:Ground037"]
DRY = ["ph:withered_grass", "acg:Grass004", "ph:sparse_grass", "acg:Grass003", "ph:grass_path_3"]
GRAVEL = ["ph:gravel_stones", "ph:park_dirt", "ph:sandy_gravel", "acg:Gravel017", "acg:Gravel019", "acg:Gravel031",
          "acg:Gravel022", "ph:gravel_floor", "ph:flower_scattered_gravel", "acg:Gravel033", "ph:bicolour_gravel",
          "acg:Gravel037", "ph:gravel_floor_02"]
PAVE = ["acg:PavingStones070", "acg:PavingStones081", "acg:PavingStones098", "acg:PavingStones107",
        "acg:PavingStones114", "acg:PavingStones118", "acg:PavingStones119", "acg:PavingStones122",
        "acg:PavingStones131", "acg:PavingStones132", "acg:PavingStones035", "acg:PavingStones037",
        "acg:PavingStones046", "ph:cobblestone_floor_04", "ph:stone_pathway_02", "ph:patterned_cobblestone",
        "acg:PavingStones138", "acg:PavingStones108", "acg:PavingStones128"]
ASPH = ["acg:Asphalt010", "acg:Asphalt012", "acg:Asphalt015", "acg:Asphalt004", "ph:asphalt_02", "acg:Asphalt021"]
SOIL = ["ph:farm_soil", "acg:Ground048", "ph:brown_mud", "ph:forest_ground_06", "ph:wood_chips", "ph:brown_mud_dry"]
PLAY = ["ph:playground_sand", "ph:park_sand", "acg:Rubber001", "acg:Rubber002", "acg:Rubber004", "ph:wood_chips",
        "ph:wood_chip_path"]
HEDGE = ["acg:Moss003", "acg:Moss004", "acg:Moss002", "acg:Grass002", "acg:Grass001"]
BRICK = ["acg:Bricks019", "acg:Bricks023", "acg:Bricks051", "acg:Bricks052", "acg:Bricks059", "acg:Bricks066",
         "acg:Bricks075A", "acg:Bricks076A", "acg:Bricks085", "acg:Bricks090", "acg:Bricks097", "acg:Bricks101"]
STONE = ["ph:castle_wall_slates", "ph:old_stone_wall_02", "ph:mossy_stone_wall", "ph:medieval_wall_02",
         "ph:japanese_stone_wall", "acg:Rock064", "ph:defense_wall"]
RENDER = ["ph:white_stucco", "ph:plastered_wall", "ph:beige_wall_001", "acg:PaintedPlaster003",
          "acg:PaintedPlaster008", "acg:PaintedPlaster012", "ph:grey_plaster", "acg:Plaster003"]
DRESSED = ["acg:Travertine004", "acg:Travertine009", "acg:Granite002A", "acg:Granite005A", "acg:Marble012",
           "acg:Concrete015", "acg:Concrete034"]
WOOD = ["acg:Wood049", "acg:Wood050", "acg:Wood051", "acg:Wood059", "acg:Wood060", "acg:Wood066", "acg:Wood067",
        "ph:brown_planks_05", "acg:Planks021", "acg:Planks012"]
FACADE = ["acg:Facade018A", "acg:Facade018B", "acg:Facade019A", "acg:Facade019B", "acg:Facade020A",
          "acg:Facade020B", "acg:Facade018C", "acg:Facade019C", "acg:Facade020C"]
BARK = ["ph:bark_platanus", "ph:japanese_sycamore", "acg:Bark009", "ph:bark_brown_01", "acg:Bark002", "acg:Bark003",
        "ph:tree_bark_03", "ph:japanese_zelkova_bark", "ph:trident_maple_bark", "acg:Bark007"]
SNOW = ["ph:snow_02", "acg:Snow004", "acg:Snow002"]

TREES = {"olive": ["ph:island_tree_01#0"], "shrubtree": ["ph:island_tree_02#0"], "airy": ["ph:tree_small_02#0"],
         "fir": ["ph:fir_sapling_medium#0", "ph:fir_sapling_medium#1", "ph:fir_sapling_medium#2"]}
FLOWERS = ["ph:flower_gazania#4", "ph:flower_gazania#5", "ph:flower_gazania#6", "ph:flower_gazania#7",
           "ph:periwinkle_plant#0", "ph:periwinkle_plant#1", "ph:celandine_01#0", "ph:celandine_01#2",
           "ph:flower_ursinia#1", "ph:flower_stinkkruid#2", "ph:flower_stinkkruid#3", "ph:dandelion_01#2"]
SHRUBS = ["ph:othonna_cerarioides#0", "ph:othonna_cerarioides#1", "ph:othonna_cerarioides#2",
          "ph:searsia_lucida#1", "ph:searsia_lucida#2", "ph:searsia_lucida#3", "ph:pachira_aquatica_01#3"]

#: claims a festoon cable may cross: flat or knee-high things (a cable hangs >= 1.8 m)
UNDER_CABLE = ("path", "bed", "bench", "bin", "bollard", "plaza")
#: claims a full tree's canopy may overhang: the flat ones (a park tree shades its path)
UNDER_CANOPY = ("path", "plaza")

STYLES = {"landscape": 0.32, "formal": 0.24, "plaza": 0.16, "playground": 0.28}
SEASONS = {"summer": 0.5, "dry": 0.12, "autumn": 0.2, "winter": 0.18}
PLAY_COLOURS = [(0.72, 0.08, 0.05), (0.05, 0.25, 0.6), (0.1, 0.45, 0.15), (0.85, 0.6, 0.05), (0.9, 0.35, 0.05),
                (0.35, 0.12, 0.45)]


def _ex(ctx, keys):
    return U.mats_existing(ctx, keys)


def _pick(ctx, keys):
    return ctx.lib.pick(ctx.rng, _ex(ctx, keys))


def _sky(ctx, season):
    lib, rng = ctx.lib, ctx.rng
    base = dict(cats_all={"outdoor"}, cats_none={"night", "indoor"})
    tags = lambda k: set(lib.items[k].get("tags", []))  # noqa: E731
    cats = lambda k: set(lib.items[k].get("categories", []))  # noqa: E731
    if season == "winter":
        keys = [k for k in lib.hdris(tags_any={"snow", "winter"}, **base) if tags(k) & {"park", "tree", "field", "city"}]
    else:
        keys = [k for k in lib.hdris(tags_any={"park", "garden"}, **base)
                if not tags(k) & {"snow", "winter"} and "artificial light" not in cats(k)]
        if season == "autumn":
            keys = [k for k in keys if "autumn" in tags(k)] or keys
        else:
            keys = [k for k in keys if "autumn" not in tags(k)] or keys
    key = lib.pick(rng, keys)
    dusk = "sunrise-sunset" in cats(key)
    return ctx.looks.sky(key, target_ev=0.15), dusk


# --- small builders ------------------------------------------------------------------------------

class Kit:
    """The park's palette and prop builders."""

    def __init__(self, ctx, season, dusk):
        self.ctx, self.season, self.dusk = ctx, season, dusk
        rng, lk = ctx.rng, ctx.looks
        self.snow = season == "winter"
        self.snowkey = _pick(ctx, SNOW)
        iron_rgb = [(0.03, 0.035, 0.03), (0.02, 0.06, 0.035), (0.05, 0.05, 0.055), (0.12, 0.03, 0.02)]
        rgb = iron_rgb[int(rng.integers(len(iron_rgb)))]
        pm = ctx.lib.materials(cats_any={"painted metal"})
        self.iron = self._snow(lk.pbr(ctx.lib.pick(rng, pm), 0.8, tint=[min(1, 6 * c + 0.02) for c in rgb],
                                      bevel=0.005, dirt=0.3, vary=0.1)) if pm else lk.paint(rgb, 0.4, 0.6)
        wk = _pick(ctx, WOOD)
        wood_tint = [(1.0, 1.0, 1.0), (0.75, 0.55, 0.4), (0.45, 0.62, 0.45), (0.9, 0.9, 0.85)][int(rng.integers(4))]
        self.wood = self._snow(lk.pbr(wk, 1.2, bevel=0.008, dirt=0.3, vary=0.15, tint=wood_tint))
        self.stone = self._snow(lk.pbr(_pick(ctx, DRESSED), 1.5, bevel=0.01, dirt=0.4, vary=0.15))
        self.concrete = self._snow(lk.pbr(ctx.lib.pick(rng, ctx.lib.materials(cats_any={"concrete"},
                                                                              source="ambientcg")), 2.0,
                                          bevel=0.015, dirt=0.5))
        tone = [((0.014, 0.04, 0.012), (0.045, 0.11, 0.03), (0.12, 0.19, 0.05)),       # privet
                ((0.012, 0.03, 0.014), (0.03, 0.075, 0.03), (0.07, 0.13, 0.05)),        # yew
                ((0.02, 0.045, 0.012), (0.06, 0.12, 0.025), (0.17, 0.22, 0.05)),        # box
                ((0.03, 0.04, 0.012), (0.08, 0.1, 0.03), (0.2, 0.17, 0.05))][int(rng.integers(4))]  # beech
        if season == "autumn":
            tone = ((0.04, 0.03, 0.01), (0.12, 0.08, 0.02), (0.3, 0.16, 0.04))
        # x1.8: foliage() darkens the leaf gaps and patches to ~0.57 of the tone on average, and at these
        # tones a clipped hedge read as a charcoal slab under an overcast sky (r5 preview wall_slot_24504)
        tone = tuple(tuple(min(1.0, 1.8 * c) for c in t) for t in tone)
        hedge = U.foliage("hedge", rng, *tone, leaf=float(rng.uniform(0.03, 0.06)))
        self.hedge = self._snow(hedge, lo=0.4)
        self.lamp_on = lk.paint((1.0, 0.85, 0.6), 0.3, emission=(1.0, 0.8, 0.55, 12.0 if dusk else 0.0),
                                name="lamp_glass") if dusk else lk.paint((0.85, 0.85, 0.8), 0.15, name="lamp_glass")
        self.bark = lk.pbr(_pick(ctx, BARK), 1.2, vary=0.2, jitter=0.12, bump=0.5)

    def _snow(self, m, lo=0.55):
        if not self.snow:
            return m
        return U.coat(self.ctx.looks, m, self.snowkey, 1.2, mode="up", lo=lo, hi=lo + 0.3, patchy=0.15)

    # furniture ---------------------------------------------------------------------------------
    def bench(self, xy, yaw, coll="scatter"):
        """A park bench facing ``yaw`` + 90 deg (its front is +y local)."""
        rng = self.ctx.rng
        style = int(rng.integers(3))
        L = float(rng.uniform(1.5, 2.0))
        b = U.Buf()
        cy, sy = math.cos(yaw), math.sin(yaw)
        ex, ey = np.array([cy, sy, 0.0]), np.array([-sy, cy, 0.0])
        o = np.array([xy[0], xy[1], 0.0])
        P = lambda u, v, z: o + ex * u + ey * v + np.array([0, 0, z])  # noqa: E731
        if style == 2:                               # concrete block bench with timber top
            b.box(P(0, 0, 0.2), (L / 2 - 0.05, 0.22, 0.2), yaw=yaw, m=1)
            for k in range(5):
                b.box(P(0, -0.2 + k * 0.1, 0.43), (L / 2, 0.04, 0.02), yaw=yaw, m=0)
            return b.flush(self.ctx, [self.wood, self.concrete], coll=coll, name="bench")
        for u in (-L / 2 + 0.12, L / 2 - 0.12):     # end frames
            if style == 0:                           # cast iron: front leg, back leg rising into the back support
                b.beam(P(u, 0.2, 0.0), P(u, 0.18, 0.44), 0.025, 0.02, up=ex, m=1)
                b.beam(P(u, -0.2, 0.0), P(u, -0.28, 0.86), 0.025, 0.02, up=ex, m=1)
                b.beam(P(u, 0.24, 0.42), P(u, -0.24, 0.42), 0.02, 0.02, up=ex, m=1)
                b.beam(P(u, 0.22, 0.64), P(u, -0.2, 0.64), 0.015, 0.02, up=ex, m=1)       # arm
                b.beam(P(u, 0.22, 0.44), P(u, 0.22, 0.66), 0.015, 0.015, up=ex, m=1)
            else:                                    # steel tube legs
                b.tube(P(u, 0.2, 0.0), P(u, 0.2, 0.42), 0.02, seg=8, m=1)
                b.tube(P(u, -0.2, 0.0), P(u, -0.26, 0.85), 0.02, seg=8, m=1)
                b.tube(P(u, 0.2, 0.42), P(u, -0.22, 0.42), 0.018, seg=8, m=1)
        for k in range(4):                           # seat slats
            v = -0.18 + k * 0.125
            b.box(P(0, v, 0.445), (L / 2, 0.045, 0.018), yaw=yaw, m=0)
        for k in range(3):                           # back slats, leaning back
            t = 0.52 + k * 0.13
            v = -0.2 - (t - 0.44) * 0.18
            b.beam(P(-L / 2, v, t), P(L / 2, v, t), 0.05, 0.016, up=tuple(ey * 0.98 + np.array([0, 0, 0.18])), m=0)
        return b.flush(self.ctx, [self.wood, self.iron], coll=coll, name="bench")

    def bin(self, xy, coll="scatter"):
        rng = self.ctx.rng
        b = U.Buf()
        x, y = xy
        if rng.random() < 0.6:                       # round metal bin on its post
            b.tube((x, y, 0.0), (x, y, 0.3), 0.03, seg=8, m=0)
            b.tube((x, y, 0.25), (x, y, 0.95), 0.24, seg=20, m=0)
            b.tube((x, y, 0.95), (x, y, 1.0), 0.26, seg=20, m=0)
            return b.flush(self.ctx, [self.iron], coll=coll, name="bin")
        for k in range(4):                           # timber slat bin
            a = k * math.pi / 2
            b.box((x + 0.24 * math.cos(a), y + 0.24 * math.sin(a), 0.45), (0.04, 0.26, 0.45), yaw=a, m=0)
        b.box((x, y, 0.92), (0.3, 0.3, 0.03), m=1)
        return b.flush(self.ctx, [self.wood, self.iron], coll=coll, name="bin")

    def bollard(self, xy, h=0.9, coll="scatter"):
        b = U.Buf()
        x, y = xy
        b.tube((x, y, 0.0), (x, y, h), 0.09, seg=14, m=0)
        b.tube((x, y, h), (x, y, h + 0.05), 0.1, seg=14, m=0)
        b.tube((x, y, h - 0.2), (x, y, h - 0.16), 0.095, seg=14, m=1)
        return b.flush(self.ctx, [self.iron, self.stone], coll=coll, name="bollard")

    def lamp(self, xy, yaw, coll="scatter", modern=False):
        ctx = self.ctx
        if not modern and "ph:street_lamp_01" in ctx.lib.items:
            e = ctx.instance("ph:street_lamp_01#0", xy, yaw=yaw, s=float(ctx.rng.uniform(1.0, 1.12)), coll=coll)
            if self.dusk:
                ctx.point_light((xy[0], xy[1], 3.7), 60.0, radius=0.12)
            return e
        b = U.Buf()
        x, y = xy
        h = float(ctx.rng.uniform(4.0, 5.0))
        b.tube((x, y, 0.0), (x, y, h), 0.055, seg=10, r1=0.04, m=0)
        b.tube((x, y, h), (x, y, h + 0.1), 0.26, seg=18, m=0)
        b.disc((x, y), 0.22, h - 0.005, seg=18, m=1)
        ob = b.flush(ctx, [self.iron, self.lamp_on], coll=coll, name="lamp")
        if self.dusk:
            ctx.point_light((x, y, h - 0.15), 80.0, radius=0.2)
        return ob

    def hoops(self, pts, h=0.45, coll="scatter"):
        """A low hoop fence along a polyline (lawn edges)."""
        b = U.Buf()
        P = np.asarray(pts, float)
        for a, c in zip(P[:-1], P[1:]):
            L = float(np.linalg.norm(c - a))
            n = max(1, int(L / 1.2))
            for k in range(n + 1):
                q = a + (c - a) * k / n
                b.tube((*q, 0.0), (*q, h), 0.012, seg=6, m=0)
            b.tube((*a, h), (*c, h), 0.01, seg=6, m=0)
        return b.flush(self.ctx, [self.iron], coll=coll, name="hoops")

    def railing(self, a, c, h=1.4, coll="backdrop"):
        """Park railings: vertical bars (render-only scenery)."""
        b = U.Buf()
        a, c = np.asarray(a, float), np.asarray(c, float)
        L = float(np.linalg.norm(c - a))
        for k in range(int(L / 0.14)):
            q = a + (c - a) * k * 0.14 / L
            b.box((q[0], q[1], h / 2), (0.01, 0.01, h / 2), m=0)
        for z in (0.12, h - 0.1):
            b.beam((*a, z), (*c, z), 0.015, 0.025, m=0)
        return b.flush(self.ctx, [self.iron], coll=coll, name="railing")


def _hedge_row(ctx, K, a, c, h, w, gaps=True):
    """A clipped hedge along a -> c, with gaps; each block claims its footprint."""
    a, c = np.asarray(a, float), np.asarray(c, float)
    L = float(np.linalg.norm(c - a))
    if L < 0.5:
        return 0
    d = (c - a) / L
    yaw = math.atan2(d[1], d[0])
    n = 0
    t = 0.0
    while t < L - 0.5:
        seg = float(ctx.rng.uniform(4.0, 9.0)) if gaps else L
        seg = min(seg, L - t)
        mid = a + d * (t + seg / 2)
        half = (seg / 2, w / 2)
        if ctx.in_zone(mid, 1.0):
            if U.free_rect(ctx, mid, half, yaw, margin=0.3):
                ctx.box((mid[0], mid[1], h / 2), (seg / 2, w / 2, h / 2), K.hedge, yaw=yaw, coll="structure",
                        name="hedge", claim=True)
                n += 1
        else:
            ctx.box((mid[0], mid[1], h / 2), (seg / 2, w / 2, h / 2), K.hedge, yaw=yaw, coll="backdrop",
                    name="hedge")
        t += seg + (float(ctx.rng.uniform(1.2, 2.5)) if gaps else 0.0)
    return n


def _fountain(ctx, K, c, R, water):
    b = U.Buf()
    b.ring(c, R, R - 0.32, 0.0, 0.48, seg=56, m=0)
    b.disc(c, R - 0.31, 0.36, seg=56, m=1)
    b.tube((c[0], c[1], 0.0), (c[0], c[1], 1.1), 0.32, seg=20, r1=0.22, m=0)
    b.tube((c[0], c[1], 1.05), (c[0], c[1], 1.25), 0.95, seg=32, r1=1.05, m=0)
    b.disc(c, 0.98, 1.2, seg=32, m=1)
    b.tube((c[0], c[1], 1.25), (c[0], c[1], 1.9), 0.16, seg=16, r1=0.12, m=0)
    b.tube((c[0], c[1], 1.85), (c[0], c[1], 1.98), 0.5, seg=24, r1=0.55, m=0)
    b.tube((c[0], c[1], 1.98), (c[0], c[1], 2.35), 0.08, seg=12, r1=0.03, m=0)
    ctx.claim_disc(c, R + 0.2, kind="fountain")
    return b.flush(ctx, [K.stone, water], coll="structure", name="fountain")


def _statue(ctx, K, c):
    rng = ctx.rng
    pieces = [k for k in ("ph:gothic_statue#0", "ph:bronze_whale_statue#0") if k.split("#")[0] in ctx.lib.items]
    k = pieces[int(rng.integers(len(pieces)))] if pieces else None
    ph = float(rng.uniform(1.0, 1.6))
    ctx.box((c[0], c[1], ph / 2), (0.65, 0.65, ph / 2), K.stone, coll="structure", name="plinth")
    ctx.box((c[0], c[1], ph + 0.05), (0.72, 0.72, 0.05), K.stone, coll="structure", name="plinth_cap")
    ctx.box((c[0], c[1], 0.1), (0.8, 0.8, 0.1), K.stone, coll="structure", name="plinth_base")
    if k:
        ctx.instance(k, c, z=ph + 0.1, yaw=float(rng.uniform(0, 6.283)), s=1.0 if "gothic" in k else 1.3,
                     coll="structure")
    ctx.claim_disc(c, 1.2, kind="statue")


def _playground(ctx, K, c, yaw, W, D, lk):
    """A fenced play area W x D centred at c: swings, slide tower, climbing frame, seesaw."""
    rng = ctx.rng
    cy, sy = math.cos(yaw), math.sin(yaw)
    ex, ey = np.array([cy, sy]), np.array([-sy, cy])
    P2 = lambda u, v: np.asarray(c, float) + ex * u + ey * v  # noqa: E731
    P3 = lambda u, v, z: np.array([*P2(u, v), z])  # noqa: E731
    corners = [P2(-W / 2, -D / 2), P2(W / 2, -D / 2), P2(W / 2, D / 2), P2(-W / 2, D / 2)]
    surf = _pick(ctx, PLAY)
    tint = PLAY_COLOURS[int(rng.integers(len(PLAY_COLOURS)))] if "Rubber" in surf else None
    ctx.flat(corners, lk.pbr(surf, 2.0, proj="flat", vary=0.2, jitter=0.0,
                             tint=[min(1.0, 2.2 * t + 0.1) for t in tint] if tint else None), z=U.flat_z(ctx),
             name="play")
    cols = [PLAY_COLOURS[int(i)] for i in rng.choice(len(PLAY_COLOURS), 3, replace=False)]
    paint = [lk.paint(col, 0.35, 0.3, name="play_paint") for col in cols]
    steel = K.iron
    post_wood = K.wood
    # swings along the far side
    b = U.Buf()
    sw_L = float(rng.uniform(3.0, 3.8))
    top = float(rng.uniform(2.2, 2.5))
    su, sv = -W / 2 + 0.6 + sw_L / 2, D / 2 - 1.2
    for u in (su - sw_L / 2, su + sw_L / 2):
        for dv in (-0.8, 0.8):
            b.tube(P3(u, sv + dv, 0.0), P3(u, sv, top), 0.05, seg=10, m=0)
    b.tube(P3(su - sw_L / 2 - 0.05, sv, top), P3(su + sw_L / 2 + 0.05, sv, top), 0.055, seg=12, m=0)
    for u in (su - sw_L / 4, su + sw_L / 4):
        for du in (-0.2, 0.2):
            b.tube(P3(u + du, sv, top), P3(u + du, sv, 0.45), 0.006, seg=4, m=1, cap=False)
        b.box(P3(u, sv, 0.44), (0.24, 0.1, 0.02), yaw=yaw, m=2)
    b.flush(ctx, [paint[0], steel, lk.paint((0.08, 0.08, 0.08), 0.6, name="rubber_seat")], coll="structure",
            name="swings")
    # slide tower near a corner
    b = U.Buf()
    tu, tv = W / 2 - 1.4, D / 2 - 1.4
    ph = float(rng.uniform(1.2, 1.5))
    for du in (-0.55, 0.55):
        for dv in (-0.55, 0.55):
            b.box(P3(tu + du, tv + dv, (ph + 1.1) / 2), (0.05, 0.05, (ph + 1.1) / 2), yaw=yaw, m=0)
    b.box(P3(tu, tv, ph), (0.6, 0.6, 0.04), yaw=yaw, m=0)
    for dv in (-0.55, 0.55):                        # side rails
        b.beam(P3(tu - 0.55, tv + dv, ph + 0.7), P3(tu + 0.55, tv + dv, ph + 0.7), 0.03, 0.03, m=1)
    b.beam(P3(tu - 0.7, tv - 0.7, ph + 1.1), P3(tu, tv, ph + 1.6), 0.8, 0.02,
           up=(ey[0] * 0.0 + ex[0] * -0.5, ex[1] * -0.5, 1.0), m=2)                           # roof half
    b.beam(P3(tu + 0.7, tv + 0.7, ph + 1.1), P3(tu, tv, ph + 1.6), 0.8, 0.02,
           up=(ex[0] * 0.5, ex[1] * 0.5, 1.0), m=2)
    run = ph * 1.9                                  # the slide chute, down toward -v
    a_ = P3(tu, tv - 0.6, ph)
    c_ = P3(tu, tv - 0.6 - run, 0.25)
    b.beam(a_, c_, 0.25, 0.02, m=3)
    for du in (-0.26, 0.26):
        b.beam(a_ + np.array([*(ex * du), 0.12]), c_ + np.array([*(ex * du), 0.12]), 0.015, 0.1, up=(0, 0, 1), m=3)
    b.beam(c_, c_ + np.array([*(-ey * 0.5), -0.2]), 0.25, 0.02, m=3)
    for du in (-0.25, 0.25):                        # ladder on the other side
        b.tube(P3(tu + du, tv + 0.6, 0.0), P3(tu + du, tv + 1.3, 0.0) + np.array([0, 0, 0]), 0.001, seg=3, m=1)
        b.beam(P3(tu + du, tv + 1.2, 0.0), P3(tu + du, tv + 0.6, ph + 0.8), 0.025, 0.025, m=1)
    for k in range(1, int(ph / 0.28) + 1):
        f = k * 0.28 / (ph + 0.8)
        z = k * 0.28
        v = tv + 1.2 - 0.6 * f / 1.0 * (ph + 0.8) / (ph + 0.8) - (0.6 * z / (ph + 0.8))
        b.tube(P3(tu - 0.25, v, z), P3(tu + 0.25, v, z), 0.015, seg=6, m=1)
    b.flush(ctx, [post_wood if rng.random() < 0.5 else paint[1], steel, paint[1], paint[2]], coll="structure",
            name="slide")
    # climbing frame
    b = U.Buf()
    n = int(rng.integers(2, 4))
    cell = float(rng.uniform(0.8, 1.0))
    fu, fv = -W / 2 + 1.0, -D / 2 + 1.0
    for i in range(n + 1):
        for j in range(n + 1):
            b.tube(P3(fu + i * cell, fv + j * cell, 0.0), P3(fu + i * cell, fv + j * cell, n * cell), 0.025, seg=8)
            for k in range(1, n + 1):
                z = k * cell
                if i < n:
                    b.tube(P3(fu + i * cell, fv + j * cell, z), P3(fu + (i + 1) * cell, fv + j * cell, z), 0.022, seg=8)
                if j < n:
                    b.tube(P3(fu + i * cell, fv + j * cell, z), P3(fu + i * cell, fv + (j + 1) * cell, z), 0.022, seg=8)
    b.flush(ctx, [paint[2]], coll="structure", name="climbing_frame")
    # seesaw
    b = U.Buf()
    ssu, ssv = 0.8, -D / 2 + 1.0
    b.box(P3(ssu, ssv, 0.25), (0.12, 0.2, 0.25), yaw=yaw, m=0)
    tilt = float(rng.uniform(-0.25, 0.25))
    b.beam(P3(ssu - 1.6, ssv, 0.5 - 1.6 * tilt), P3(ssu + 1.6, ssv, 0.5 + 1.6 * tilt), 0.12, 0.03, m=1)
    b.flush(ctx, [steel, post_wood], coll="structure", name="seesaw")
    # low fence round it, a gate gap in the near side
    pts = corners + [corners[0]]
    fence = []
    for a, q in zip(pts[:-1], pts[1:]):
        fence.append((a, q))
    b = U.Buf()
    for k, (a, q) in enumerate(fence):
        L = float(np.linalg.norm(q - a))
        d = (q - a) / L
        spans = [(0.0, L)]
        if k == 0:                                   # gate gap
            g = L / 2
            spans = [(0.0, g - 0.7), (g + 0.7, L)]
        for s0, s1 in spans:
            p0, p1 = a + d * s0, a + d * s1
            m = max(1, int((s1 - s0) / 1.5))
            for i in range(m + 1):
                q2 = p0 + (p1 - p0) * i / m
                b.box((q2[0], q2[1], 0.45), (0.04, 0.04, 0.45), m=0)
            for z in (0.3, 0.8):
                b.beam((*p0, z), (*p1, z), 0.02, 0.04, m=0)
    b.flush(ctx, [post_wood], coll="structure", name="play_fence")
    ctx.claim_rect(c, (W / 2 + 0.2, D / 2 + 0.2), yaw, kind="playground")
    return surf


def _city(ctx, K, lk, sides):
    """Streets and blocks beyond the park edge (all render-only)."""
    rng = ctx.rng
    zx0, zy0, zx1, zy1 = ctx.zone
    fac = [k for k in _ex(ctx, FACADE)]
    fac = list(rng.choice(fac, size=min(4, len(fac)), replace=False)) if fac else []
    asph = lk.pbr(_pick(ctx, ASPH), 4.0, proj="flat", vary=0.2, jitter=0.0)
    walk = lk.pbr(_pick(ctx, PAVE), 1.5, proj="flat", vary=0.15, jitter=0.0)
    kerb = lk.pbr(_pick(ctx, DRESSED), 1.0, bevel=0.01, vary=0.1)
    line = lk.paint((0.8, 0.8, 0.75), 0.6, name="road_line")
    roof = lk.pbr(ctx.lib.pick(rng, ctx.lib.materials(cats_any={"concrete"}, source="ambientcg")), 4.0)
    shop = lk.glass("shopfront", tint=(0.25, 0.27, 0.3), rough=0.08)
    built = 0
    for side, edge in sides:
        # side: 0=-x, 1=+x, 2=-y, 3=+y ; distance from the zone edge to the railing
        if side in (0, 1):
            sgn = -1 if side == 0 else 1
            xr = (zx0 if side == 0 else zx1) + sgn * edge
            lo, hi = zy0 - 70, zy1 + 70
            to = lambda along, off: np.array([xr + sgn * off, along])  # noqa: E731
        else:
            sgn = -1 if side == 2 else 1
            yr = (zy0 if side == 2 else zy1) + sgn * edge
            lo, hi = zx0 - 70, zx1 + 70
            to = lambda along, off: np.array([along, yr + sgn * off])  # noqa: E731
        K.railing(to(lo, 0.0), to(hi, 0.0))
        # sidewalk, kerb, road, far sidewalk
        for (o0, o1, m) in ((0.3, 3.3, walk), (3.3, 13.3, asph), (13.3, 17.0, walk)):
            ctx.flat([to(lo, o0), to(hi, o0), to(hi, o1), to(lo, o1)], m, z=U.flat_z(ctx))   # streets cross
        for o in (3.3, 13.3):
            a, c = to(lo, o), to(hi, o)
            d = (c - a) / np.linalg.norm(c - a)
            mid = (a + c) / 2
            ctx.box((mid[0], mid[1], 0.06), (np.linalg.norm(c - a) / 2, 0.12, 0.06), kerb,
                    yaw=math.atan2(d[1], d[0]), coll="backdrop", name="kerb", collide=False)
        for s in np.arange(lo, hi, 6.0):             # dashed centre line
            ctx.strip(to(s, 8.3), to(s + 3.0, 8.3), 0.12, line, z=U.flat_z(ctx))
        # parked cars and street trees
        for s in np.arange(lo + rng.uniform(0, 8), hi, float(rng.uniform(6.5, 11.0))):
            if rng.random() < 0.55 and "ph:covered_car" in ctx.lib.items:
                p = to(s, 4.5 if rng.random() < 0.5 else 12.1)
                yaw = math.atan2(*(to(s + 1, 4.5) - to(s, 4.5))[::-1]) + math.pi / 2
                ctx.instance("ph:covered_car#0", p, yaw=yaw, coll="backdrop")
        for s in np.arange(lo + 3, hi, float(rng.uniform(8.0, 12.0))):
            p = to(s, 15.5)
            if K.snow:
                U.bare_tree(ctx, p, float(rng.uniform(6, 9)), 0.14, K.bark, coll="backdrop", depth=2)
                continue
            U.trunk(ctx, p, 4.2, 0.14, K.bark, r_top=0.08, flare=0.2, coll="backdrop")
            ctx.instance(TREES["airy"][0], p, z=3.4, yaw=float(rng.uniform(0, 6.3)), s=float(rng.uniform(1.4, 1.9)),
                         coll="backdrop")
        # the blocks
        s = lo
        while s < hi and fac:
            w = float(rng.uniform(10.0, 28.0))
            dep = float(rng.uniform(12.0, 20.0))
            h = float(rng.choice([rng.uniform(9, 16), rng.uniform(16, 28), rng.uniform(28, 42)], p=[0.4, 0.45, 0.15]))
            c = to(s + w / 2, 18.0 + dep / 2)
            yaw = 0.0 if side in (2, 3) else math.pi / 2
            fk = str(fac[int(rng.integers(len(fac)))])
            m = lk.pbr(fk, float(ctx.lib.items[fk].get("tile_m") or 13.0), vary=0.12, jitter=0.1, dirt=0.3)
            half = (w / 2, dep / 2, h / 2) if side in (2, 3) else (dep / 2, w / 2, h / 2)
            ctx.box((c[0], c[1], h / 2), half, m, coll="backdrop", name="building", collide=False)
            ctx.box((c[0], c[1], h + 0.4), (half[0] + 0.15, half[1] + 0.15, 0.4), roof, coll="backdrop",
                    name="roof", collide=False)
            ctx.box((c[0], c[1], 1.9), (half[0] + 0.12, half[1] + 0.12, 1.9), shop, coll="backdrop",
                    name="shopfront", collide=False)
            built += 1
            s += w + float(rng.choice([0.0, 0.0, 0.0, rng.uniform(3, 8)]))
    return built


# --- the park ------------------------------------------------------------------------------------

def build(ctx) -> dict:
    t0 = time.time()
    tick = lambda what: print(f"[park] {what} {time.time() - t0:.1f} s", flush=True)  # noqa: E731
    rng, lk, lib = ctx.rng, ctx.looks, ctx.lib
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    keepout = float(ctx.knobs.get("keepout", 2.5))
    st = list(STYLES)
    style = str(rng.choice(st, p=np.array([STYLES[k] for k in st]) / sum(STYLES.values())))
    se = list(SEASONS)
    season = str(rng.choice(se, p=np.array([SEASONS[k] for k in se]) / sum(SEASONS.values())))
    force = os.environ.get("PARK_STYLE", "")          # test hook: PARK_STYLE=formal[:winter]
    if force:
        style = force.split(":")[0]
        season = force.split(":")[1] if ":" in force else season
    sky, dusk = _sky(ctx, season)
    K = Kit(ctx, season, dusk)
    winter = season == "winter"

    # --- ground ---------------------------------------------------------------------------------
    if winter:
        gkeys = [_pick(ctx, SNOW), _pick(ctx, ["ph:snow_03", "acg:Snow014", "ph:snow_01"]), _pick(ctx, SNOW)]
        tile = 2.5
    elif style == "plaza":
        gkeys = [_pick(ctx, PAVE)]
        tile = float(lib.items[gkeys[0]].get("tile_m") or 2.0)
    else:
        grass = DRY if season == "dry" else GRASS
        gkeys = [_pick(ctx, grass), _pick(ctx, grass), _pick(ctx, WORN)]
        tile = float(rng.uniform(1.4, 2.2))
    wet = float(rng.choice([0.0, 0.0, 0.0, 0.3])) if not winter else 0.0
    gm = lk.ground_material(gkeys, tile_m=tile, puddles=wet if style == "plaza" else 0.0,
                            blend_scale=float(rng.uniform(5.0, 12.0)))
    ctx.ground(U.matte(gm, 0.8) if style != "plaza" and not winter else gm)
    lawn_mat = None
    if style == "plaza":
        lawn_mat = lk.ground_material([_pick(ctx, GRASS), _pick(ctx, WORN)] if not winter else [_pick(ctx, SNOW)],
                                      tile_m=1.6, blend_scale=6.0, name="lawn")

    tick("ground+sky")
    info = {"style": style, "season": season, "dusk": dusk, "env_override": force or None}
    # two tree species per park (each Poly Haven tree file costs 7-17 s to import)
    if winter:
        tree_sets = ["bare", "fir"]
    else:
        tree_sets = ["airy", str(rng.choice(["olive", "shrubtree", "fir"], p=[0.45, 0.3, 0.25]))]
    info["trees_species"] = tree_sets
    _dress(ctx, K, lk, info, crowns=not winter)

    tick("dressed")
    # --- paths ----------------------------------------------------------------------------------
    pkind = str(rng.choice(["gravel", "pave", "asphalt"], p=[0.45, 0.4, 0.15]))
    pkey = _pick(ctx, {"gravel": GRAVEL, "pave": PAVE, "asphalt": ASPH}[pkind]) if not winter else \
        _pick(ctx, ["ph:snow_05", "ph:asphalt_snow", "ph:snow_03"] + PAVE[:4])
    ptile = float(lib.items[pkey].get("tile_m") or 2.0) if pkind == "pave" else 2.5
    pmat = lk.pbr(pkey, ptile, proj="flat", vary=0.2, jitter=0.0)
    edge = lk.pbr(_pick(ctx, DRESSED), 1.0, proj="flat", vary=0.1, jitter=0.0)
    paths = []                                      # (polyline, width)
    via = []
    for p in ctx.spec["primitives"]:
        if p["kind"] == "wall":
            p0, p1 = np.asarray(p["p0"], float), np.asarray(p["p1"], float)
            d = (p1 - p0) / np.linalg.norm(p1 - p0)
            for s_, w, sz0, sz1 in p.get("slots", []):
                if sz0 < 0.05 and w > 1.0:
                    via.append(p0 + d * s_)
    ym = float(rng.uniform(y0 + 2.5, y1 - 2.5))
    if style == "formal":
        main_w = float(rng.uniform(3.0, 5.0))
        paths.append((np.array([[zx0 - 40, ym], [zx1 + 40, ym]]), main_w))
        xm = float(rng.uniform(x0 + 5, x1 - 5))
        for _ in range(40):                          # the plaza (and its fountain) clear of the obstacles
            xt, yt = float(rng.uniform(x0 + 4, x1 - 4)), float(rng.uniform(y0 + 3, y1 - 3))
            if ctx.free((xt, yt), 3.5, max(1.0, 0.5 * keepout)):
                xm, ym = xt, yt
                break
        paths[0] = (np.array([[zx0 - 40, ym], [zx1 + 40, ym]]), main_w) if paths else None
        paths.append((np.array([[xm, zy0 - 40], [xm, zy1 + 40]]), float(rng.uniform(2.5, 4.0))))
        plaza_c = np.array([xm, ym])
    else:
        main_w = float(rng.uniform(2.2, 3.6))
        anchors = [np.array([zx0 - 40, float(rng.uniform(y0, y1))]), np.array([x0 + 2.0, ym]), *via,
                   np.array([x1 - 2.0, float(rng.uniform(y0 + 2, y1 - 2))]),
                   np.array([zx1 + 40, float(rng.uniform(y0, y1))])]
        anchors = sorted(anchors, key=lambda a: a[0])
        keep = [anchors[0]]
        for a in anchors[1:]:
            if np.linalg.norm(a - keep[-1]) > 5.0:
                keep.append(a)
        pts = list(U.spline(keep, step=1.0))
        paths.append((np.array(pts), main_w))
        # a branch path crossing toward a side
        k = int(rng.integers(len(pts) // 3, 2 * len(pts) // 3))
        start = np.asarray(pts[k])
        end = np.array([start[0] + float(rng.uniform(-15, 15)), (zy1 + 40) if rng.random() < 0.5 else (zy0 - 40)])
        paths.append((U.wander(start, end, rng, amp=2.5, step=1.0, waves=0.7), float(rng.uniform(1.6, 2.6))))
        plaza_c = start
    for pl, w in paths:
        U.ribbon(ctx, pl, w + 0.3, edge, z=U.flat_z(ctx), jitter=0.0, name="path_edge")
        U.ribbon(ctx, pl, w, pmat, z=U.flat_z(ctx), jitter=0.0, name="path")
        U.claim_polyline(ctx, pl, w / 2 + 0.15, seg=3.0)
    info["paths"] = {"kind": pkind, "material": pkey, "n": len(paths)}

    # --- plaza with a fountain or statue ---------------------------------------------------------
    centre = None
    if style in ("formal", "plaza") or rng.random() < 0.45:
        R = float(rng.uniform(4.5, 7.0))
        pl_mat = lk.pbr(_pick(ctx, PAVE), 1.5, proj="flat", vary=0.15, jitter=0.0) if pkind != "pave" or \
            rng.random() < 0.5 else pmat
        ctx.flat(U.poly_blob(plaza_c, R + 0.25, n=64, rough=0.0, rng=rng), edge, z=U.flat_z(ctx), name="plaza_edge")
        ctx.flat(U.poly_blob(plaza_c, R, n=64, rough=0.0, rng=rng), pl_mat, z=U.flat_z(ctx), name="plaza")
        water = _water(lk)
        feat = str(rng.choice(["fountain", "statue", "none"], p=[0.55, 0.3, 0.15]))
        FR = min(R - 1.5, float(rng.uniform(2.0, 3.4)))
        if feat == "fountain" and ctx.free(plaza_c, FR, max(1.0, 0.5 * keepout)):
            _fountain(ctx, K, plaza_c, FR, water)
            centre = "fountain"
        elif feat != "none" and ctx.free(plaza_c, 1.0, max(1.0, 0.5 * keepout)):
            _statue(ctx, K, plaza_c)
            centre = "statue"
        # benches round the plaza, facing in
        for a in np.linspace(0, 2 * math.pi, int(rng.integers(3, 7)), endpoint=False) + rng.uniform(0, 1):
            p = plaza_c + (R - 0.6) * np.array([math.cos(a), math.sin(a)])
            if ctx.free(p, 1.0, 0.2) or not ctx.in_zone(p):
                K.bench(p, a + math.pi / 2, coll="scatter" if ctx.in_zone(p) else "backdrop")
                if ctx.in_zone(p):
                    ctx.claim_disc(p, 1.0, kind="bench")
        ctx.claim_disc(plaza_c, 0.0, kind="plaza")
    info["centre"] = centre

    # --- playground ------------------------------------------------------------------------------
    play = None
    if style == "playground" or (style == "landscape" and rng.random() < 0.3):
        W, D = float(rng.uniform(8.0, 11.0)), float(rng.uniform(6.5, 8.5))
        cands = []
        for _ in range(250):
            c = np.array([rng.uniform(zx0 - 8, zx1 + 8), rng.uniform(zy0 - 8, zy1 + 8)])
            yaw = float(rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2])) + float(rng.normal(0, 0.1))
            inside = x0 <= c[0] <= x1 and y0 <= c[1] <= y1
            if U.free_rect(ctx, c, (W / 2 + 0.3, D / 2 + 0.3), yaw, margin=max(0.5, 0.5 * keepout), step=0.8):
                cands.append((0 if inside else 1, float(np.linalg.norm(c - (ctx.goal / 2))), c, yaw))
        if cands:
            cands.sort(key=lambda t: (t[0], t[1]))
            _, _, c, yaw = cands[int(rng.integers(min(4, len(cands))))]
            play = {"surface": _playground(ctx, K, c, yaw, W, D, lk), "centre": [round(float(v), 2) for v in c]}
    info["playground"] = play

    # --- hedges and beds ---------------------------------------------------------------------------
    n_hedge = 0
    if style == "formal":
        h = float(rng.uniform(0.5, 1.3))
        for pl, w in paths:
            a, b = pl[0], pl[-1]
            d = (b - a) / np.linalg.norm(b - a)
            nrm = np.array([-d[1], d[0]])
            hw = float(rng.uniform(0.5, 0.8))
            for sgn in (-1, 1):
                off = nrm * sgn * (w / 2 + 0.15 + 0.35 + hw / 2 + float(rng.uniform(0.3, 1.2)))
                n_hedge += _hedge_row(ctx, K, a + off, b + off, h, hw)
    elif style != "plaza":
        for _ in range(ctx.count(2.0) + 2):
            a = np.array([rng.uniform(zx0 - 5, zx1 + 5), rng.uniform(zy0 - 5, zy1 + 5)])
            ang = float(rng.choice([0, math.pi / 2])) + float(rng.normal(0, 0.3))
            L = float(rng.uniform(4.0, 12.0))
            b = a + L * np.array([math.cos(ang), math.sin(ang)])
            n_hedge += _hedge_row(ctx, K, a, b, float(rng.uniform(0.8, 1.9)), float(rng.uniform(0.7, 1.4)),
                                  gaps=False)
    info["hedges"] = n_hedge
    soil = lk.pbr(_pick(ctx, SOIL), 1.5, proj="flat", vary=0.2, jitter=0.0)
    flowers = [k for k in FLOWERS if k.split("#")[0] in lib.items]
    shrubs = [k for k in SHRUBS if k.split("#")[0] in lib.items]
    n_beds = 0
    if not winter:
        for p in U.free_spots(ctx, ctx.count(3.0) + 3, (zx0 - 6, zy0 - 6, zx1 + 6, zy1 + 6), 2.2, 0.5, rng):
            r = float(rng.uniform(1.2, 2.3))
            if style == "formal":
                poly = [p + np.array(q) for q in ((-r, -r * 0.6), (r, -r * 0.6), (r, r * 0.6), (-r, r * 0.6))]
            else:
                poly = U.poly_blob(p, r, n=18, rough=0.2, rng=rng)
            ctx.flat(poly, soil, z=U.flat_z(ctx), name="bed")
            if ctx.in_zone(p):
                ctx.claim_disc(p, r * 0.9, kind="bed")
            fl = [flowers[int(i)] for i in rng.choice(len(flowers), size=min(2, len(flowers)), replace=False)]
            for _ in range(int(r * r * 10)):
                q = p + rng.uniform(-0.85, 0.85, 2) * np.array([r, r * (0.6 if style == "formal" else 1.0)])
                if style != "formal" and np.linalg.norm(q - p) > r * 0.85:
                    continue
                ctx.instance(fl[int(rng.integers(len(fl)))], q, yaw=float(rng.uniform(0, 6.3)),
                             s=float(rng.uniform(0.9, 1.5)), coll="scatter")
            if shrubs and rng.random() < 0.6:
                ctx.instance(shrubs[int(rng.integers(len(shrubs)))], p, yaw=float(rng.uniform(0, 6.3)),
                             s=float(rng.uniform(0.8, 1.2)), coll="scatter")
            n_beds += 1
    info["beds"] = n_beds

    tick("layout")
    # --- trees -------------------------------------------------------------------------------------
    if season == "autumn":
        hue = float(rng.uniform(0.36, 0.44))
        for t in tree_sets:
            for k in TREES.get(t, []):
                if t != "fir":
                    U.autumn_piece(ctx, k, hue=hue, sat=1.15)
    scale = {"olive": (1.5, 2.3), "airy": (1.7, 2.5), "shrubtree": (1.4, 2.0), "fir": (0.8, 1.3)}
    n_trees = {"in": 0, "zone": 0, "scenery": 0}
    litter = None
    if season == "autumn":
        lk_ = U.litter_materials(lib)               # cut-out scans only (an opaque carpet renders as a disc)
        litter = U.soft_decal(lk.pbr(lib.pick(rng, lk_), None, vary=0.2, jitter=0.0)) if lk_ else None

    heavy = {"n": 0}
    trunks = []                                     # (xy, r, claim) of the trunks planted in the zone (L3 limbs)
    ceil = U.cloud_top(ctx)
    if "airy" in tree_sets:
        U.rebark_piece(ctx, TREES["airy"][0], K.bark)   # crown stems in the trunks' bark: one tree, not two

    def plant(p, coll, where, margin=0.3, force=None):
        """One tree at p (of kind ``force``, else drawn); False if it does
        not fit. In the zone a full tree's whole canopy (it hangs in the
        flight band) must be ``margin`` clear of every claim but a path's
        (``UNDER_CANOPY``) and claims its disc; a scenery tree keeps its canopy out of the zone (else it would
        collide and be realised for the cloud). At most 4 full trees in the
        zone (each is realised for the cloud: 1.6 M triangles); past that a
        crown on a trunk, or in winter a bare tree."""
        t = force or tree_sets[int(rng.integers(len(tree_sets)))]
        if where != "scenery" and t in ("olive", "shrubtree", "fir") and heavy["n"] >= 4:
            t = "airy_pole" if "airy" in tree_sets else "bare" if "bare" in tree_sets else t
        if where != "scenery" and t == "airy":
            t = "airy_pole"
        claim = 0.9
        full = t in ("olive", "shrubtree", "fir", "airy")
        if full:
            k = TREES[t][int(rng.integers(len(TREES[t])))]
            sc = float(rng.uniform(*scale[t]))
            r_can = U.piece_r(ctx, k, sc)
            fits = not ctx.rect_meets_zone(p - r_can, p + r_can) if where == "scenery" else \
                U.clear_many(ctx, np.asarray(p, float)[None], skip=UNDER_CANOPY)[0] >= r_can + margin
            if not fits and "airy" in tree_sets:      # a crown on a trunk instead (its canopy is above the cloud)
                t, full = "airy_pole", False
            elif not fits:
                return False
            claim = r_can
        host = None
        if t == "airy_pole":
            s_ = float(rng.uniform(1.8, 2.5))
            # the crown -- its stem too, thicker than the 0.11 m trunk top -- starts above the cloud
            # ceiling: all the cloud sees, and all that is rendered below the ceiling, is the trunk
            zc = ceil + float(rng.uniform(0.05, 0.6))
            ht = zc + 1.4
            r0 = float(rng.uniform(0.15, 0.3))
            U.trunk(ctx, p, ht, r0, K.bark, r_top=0.11, flare=0.3, coll=coll)
            U.crown(ctx, TREES["airy"][0], p, zc, s_, coll=coll)
            claim = 0.9
            host = r0
        elif t == "bare":
            h = float(rng.uniform(7.0, 13.0))
            r0 = float(rng.uniform(0.15, 0.3))
            ok = (lambda R: True) if where == "scenery" else (lambda R: ctx.free(p, R, margin))  # noqa: E731
            rec = U.bare_tree(ctx, p, h, r0, K.bark, coll=coll, clear_z=float(rng.uniform(2.6, 3.4)), reach_ok=ok)
            if rec is None:
                return False
            claim = rec["reach"]
            host = r0
        else:
            ctx.instance(k, p, yaw=float(rng.uniform(0, 6.283)), s=sc, coll=coll)
        if full and where != "scenery" and t in ("olive", "shrubtree", "fir"):
            heavy["n"] += 1                           # counted once placed (a tree that did not fit is free)
        if where != "scenery":
            ctx.claim_disc(p, claim, kind="tree")
            if host is not None:
                trunks.append((np.asarray(p, float), host, ctx.foot[-1]))
        if litter is not None and where != "scenery" and rng.random() < 0.8:
            q = p + rng.normal(0, 0.6, 2)
            zl = U.flat_z(ctx)
            V = [(q[0] + dx, q[1] + dy, zl) for dx, dy in ((-2, -2), (2, -2), (2, 2), (-2, 2))]
            ctx.mesh(V, [(0, 1, 2, 3)], litter, uvs=[[(0, 0), (1, 0), (1, 1), (0, 1)]], coll="ground",
                     collide=False, name="leaves")
        n_trees[where] += 1
        return True

    tm = max(1.2, 0.6 * keepout)
    for p in U.free_spots(ctx, ctx.count(3.0) + 1, (x0 + 1, y0 + 1, x1 - 1, y1 - 1), 1.5, tm, rng):
        if ctx.free(p, 1.5, tm):
            plant(p, "scatter", "in", margin=tm)
    for p in _blue(rng, (zx0, zy0, zx1, zy1), float(rng.uniform(6.0, 9.0)), 600):
        if (x0 <= p[0] <= x1 and y0 <= p[1] <= y1) or not ctx.free(p, 1.5, 0.5):
            continue
        plant(p, "scatter", "zone", margin=0.5)
    # an avenue of crowns on trunks along the main path (formal and some landscape parks)
    if "airy" in tree_sets and (style == "formal" or rng.random() < 0.35):
        pl, w = paths[0]
        seglen = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pl, axis=0), axis=1))]
        gap = float(rng.uniform(7.0, 10.0))
        for s in np.arange(float(rng.uniform(0, gap)), seglen[-1], gap):
            i = int(np.searchsorted(seglen, s)) - 1
            i = max(0, min(i, len(pl) - 2))
            d = pl[i + 1] - pl[i]
            d /= np.linalg.norm(d) + 1e-9
            q0 = pl[i] + d * (s - seglen[i])
            for sgn in (-1, 1):
                q = q0 + sgn * np.array([-d[1], d[0]]) * (w / 2 + 1.4)
                inz = ctx.in_zone(q)
                if inz and not ctx.free(q, 0.4, max(0.8, 0.4 * keepout)):
                    continue
                if not inz and ctx.in_zone(q, 12.0) is False and abs(q[0]) > 1e4:
                    continue
                s_ = float(rng.uniform(1.8, 2.3))
                zc = ceil + float(rng.uniform(0.05, 0.5))  # crown, stem and all, above the cloud ceiling
                ht = zc + 1.4
                U.trunk(ctx, q, ht, float(rng.uniform(0.15, 0.24)), K.bark, r_top=0.11, flare=0.25,
                        coll="structure" if inz else "backdrop")
                U.crown(ctx, TREES["airy"][0], q, zc, s_, coll="scatter" if inz else "backdrop")
                if inz:
                    ctx.claim_disc(q, 0.5, kind="tree")
    # scenery trees: clumps in the lawns beyond the zone
    far_rect = (zx0 - 40, zy0 - 40, zx1 + 40, zy1 + 40)
    for p in _blue(rng, far_rect, float(rng.uniform(7.0, 11.0)), 900):
        if ctx.in_zone(p, 1.0) or U.clear_many(ctx, p[None])[0] < 1.0:
            continue
        plant(p, "backdrop", "scenery")
    info["trees"] = n_trees

    tick("trees")
    # --- furniture along the paths --------------------------------------------------------------------
    counts = {"bench": 0, "bin": 0, "lamp": 0, "bollard": 0, "picnic": 0}
    clutter = float(ctx.knobs.get("clutter", 1.0))
    lamp_gap = float(rng.uniform(12.0, 20.0))
    modern = rng.random() < 0.3
    for pi, (pl, w) in enumerate(paths):
        seglen = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pl, axis=0), axis=1))]
        lamp_side = float(rng.choice([-1, 1]))

        def at(s):
            i = max(0, min(int(np.searchsorted(seglen, s)) - 1, len(pl) - 2))
            d = pl[i + 1] - pl[i]
            d /= np.linalg.norm(d) + 1e-9
            return pl[i] + d * (s - seglen[i]), d

        s = float(rng.uniform(2, 8))
        while s < seglen[-1]:
            q0, d = at(s)
            nrm = np.array([-d[1], d[0]])
            sgn = float(rng.choice([-1, 1]))
            p = q0 + nrm * sgn * (w / 2 + 0.65)
            inz = ctx.in_zone(p, 10.0)
            if inz and (not ctx.in_zone(p) or ctx.free(p, 1.0, 0.15)):
                K.bench(p, math.atan2(d[1], d[0]) + (0 if sgn < 0 else math.pi),
                        coll="scatter" if ctx.in_zone(p) else "backdrop")
                if ctx.in_zone(p):
                    ctx.claim_disc(p, 1.0, kind="bench")
                counts["bench"] += 1
                if rng.random() < 0.45:
                    pb = p + d * (1.4 if rng.random() < 0.5 else -1.4)
                    if not ctx.in_zone(pb) or ctx.free(pb, 0.3, 0.15):
                        K.bin(pb, coll="scatter" if ctx.in_zone(pb) else "backdrop")
                        if ctx.in_zone(pb):
                            ctx.claim_disc(pb, 0.35, kind="bin")
                        counts["bin"] += 1
            s += float(rng.uniform(8.0, 15.0)) / max(0.5, clutter)
        s = float(rng.uniform(0, lamp_gap))
        while s < seglen[-1]:
            q0, d = at(s)
            p = q0 + np.array([-d[1], d[0]]) * lamp_side * (w / 2 + 0.45)
            inz = ctx.in_zone(p)
            if ctx.in_zone(p, 25.0) and (not inz or ctx.free(p, 0.3, 0.3)):
                K.lamp(p, math.atan2(d[1], d[0]), coll="scatter" if inz else "backdrop", modern=modern)
                if inz:
                    ctx.claim_disc(p, 0.35, kind="lamp")
                counts["lamp"] += 1
            s += lamp_gap
        # bollards where the path meets the patch edge
        for s in (0.0, seglen[-1]):
            pass
    # picnic tables on the lawn
    if style in ("landscape", "playground") and not winter and "ph:wooden_picnic_table" in lib.items:
        for p in U.free_spots(ctx, ctx.count(1.5) + 1, (zx0, zy0, zx1, zy1), 1.9, max(0.6, 0.4 * keepout), rng):
            if ctx.free(p, 1.9, max(0.6, 0.4 * keepout)):
                ctx.instance("ph:wooden_picnic_table#0", p, yaw=float(rng.uniform(0, 6.3)), coll="scatter",
                             claim=True)
                counts["picnic"] += 1
    # bollards on plaza parks, planters
    if style == "plaza":
        for p in U.free_spots(ctx, ctx.count(8.0), (x0, y0, x1, y1), 0.15, max(0.8, 0.5 * keepout), rng):
            if ctx.free(p, 0.15, max(0.8, 0.5 * keepout)):
                K.bollard(p)
                ctx.claim_disc(p, 0.15, kind="bollard")
                counts["bollard"] += 1
        for p in U.free_spots(ctx, ctx.count(3.0) + 1, (zx0, zy0, zx1, zy1), 2.0, max(0.7, 0.5 * keepout), rng):
            r = float(rng.uniform(0.9, 1.4))
            if not ctx.free(p, r * 1.42, max(0.7, 0.5 * keepout)):
                continue
            ctx.box((p[0], p[1], 0.3), (r, r, 0.3), K.concrete, coll="structure", name="planter", claim=True)
            ctx.box((p[0], p[1], 0.605), (r - 0.1, r - 0.1, 0.006), soil, coll="structure", name="planter_soil")
            if shrubs:
                ctx.instance(shrubs[int(rng.integers(len(shrubs)))], p, z=0.6, yaw=float(rng.uniform(0, 6.3)),
                             coll="scatter")
        if lawn_mat is not None:
            U.matte(lawn_mat, 0.8)
            for p in U.free_spots(ctx, 4, (zx0, zy0, zx1, zy1), 2.5, 0.3, rng):
                ctx.flat(U.poly_blob(p, float(rng.uniform(2.0, 3.5)), n=24, rough=0.1, rng=rng), lawn_mat,
                         z=U.flat_z(ctx))
    info["furniture"] = counts

    # --- small lawn life --------------------------------------------------------------------------------
    if not winter and style != "plaza":
        # the light pieces only (dandelion #0/#1 are 23 k triangles a tuft, and every one in the zone collides)
        weeds = [k for k in ["ph:dandelion_01#2", "ph:dandelion_01#3", "ph:celandine_01#1", "ph:grass_medium_01#3",
                             "ph:grass_medium_01#4", "ph:grass_medium_02#1", "ph:grass_medium_01#5"]
                 if k.split("#")[0] in lib.items]
        for p in U.free_spots(ctx, ctx.count(60) + 40, (zx0 - 6, zy0 - 6, zx1 + 6, zy1 + 6), 0.1, 0.1, rng,
                              skip=U.LOW):
            ctx.instance(weeds[int(rng.integers(len(weeds)))], p, yaw=float(rng.uniform(0, 6.3)),
                         s=float(rng.uniform(0.7, 1.2)), coll="scatter")

    # --- hanging things in the flight band (L3) ----------------------------------------------------------
    n_hang = ctx.count(float(ctx.knobs.get("hang", 0.0)))
    hung = 0
    cable = lk.paint((0.03, 0.03, 0.03), 0.5, name="cable")
    bulb = lk.paint((1.0, 0.9, 0.7), 0.3, name="bulb", emission=(1.0, 0.85, 0.6, 6.0 if dusk else 1.5))
    net_keys = lib.materials(cats_any={"net"})
    planted = 0
    inner = (x0 + 1, y0 + 1, x1 - 1, y1 - 1)
    under = UNDER_CABLE + (("hedge",) if style == "formal" else ())   # formal hedges are <= 1.3 m
    for _ in range(400):
        if hung >= n_hang:
            break
        what = str(rng.choice(["lights", "lights", "net", "branch"]))
        if what == "branch":                          # a low limb off a tree planted in the patch
            hosts = [t for t in trunks if x0 - 2 <= t[0][0] <= x1 + 2 and y0 - 2 <= t[0][1] <= y1 + 2]
            if not hosts and planted < n_hang:        # no tree to grow it from: plant one (a crown on a trunk,
                sp = U.free_spots(ctx, 1, inner, 0.9, 0.5, rng)                        # bare in winter)
                if sp and plant(sp[0], "scatter", "in", margin=0.5, force="airy_pole" if not winter else "bare"):
                    planted += 1
                continue
            if not hosts:
                continue
            h, rh, hfoot = hosts[int(rng.integers(len(hosts)))]
            ang = float(rng.uniform(0, 2 * math.pi))
            dv = np.array([math.cos(ang), math.sin(ang)])
            p0 = h + dv * rh * 0.8
            q = p0 + dv * float(rng.uniform(1.2, 2.8))
            # clear of every claim but the host's own (the limb starts inside it: the r4 check measured
            # from within that 0.9 m disc and never passed, review of cd304aa)
            if not U.segment_clear(ctx, p0, q, 0.3, skip_foot=(hfoot,)):
                continue
            z = float(rng.uniform(1.4, 3.0))
            U.limb(ctx, (*p0, z), (*q, z + float(rng.uniform(-0.4, 0.5))), 0.07, 0.02, K.bark, droop=0.15)
            U.claim_segment(ctx, p0 + dv * rh, q, 0.15, kind="hang")
            hung += 1
            continue
        sp = U.free_spots(ctx, 1, inner, 0.2, 0.5, rng)   # the first post on free ground (a tier-C arena is
        if not sp:                                         # half claimed: uniform draws rarely were)
            continue
        a = sp[0]
        ang = float(rng.uniform(0, 2 * math.pi))
        L = float(rng.uniform(4.0, 8.0))
        b = a + L * np.array([math.cos(ang), math.sin(ang)])
        if not (ctx.in_zone(b) and ctx.free(b, 0.2, 0.5)):
            continue
        if what == "lights":                          # a festoon between two posts
            # the cable sags into the flight band: never over the start/goal, an opening, a family's region,
            # a tree or a lamp -- but across paths, beds and benches (it hangs >= 1.8 m, above them all)
            if not U.segment_clear(ctx, a, b, 0.4, skip=under):
                continue
            za, zb = float(rng.uniform(2.6, 3.4)), float(rng.uniform(2.6, 3.4))
            for q, zq in ((a, za), (b, zb)):
                ctx.cylinder((q[0], q[1], 0.0), (q[0], q[1], zq + 0.2), 0.05, K.iron, coll="scatter", name="post",
                             claim=True)
            sag = float(rng.uniform(0.3, 0.8))
            U.limb(ctx, (*a, za), (*b, zb), 0.008, 0.008, cable, segs=5, droop=sag)
            for t in np.linspace(0.08, 0.92, int(L / 0.6)):
                q = a + (b - a) * t
                zq = za + (zb - za) * t - sag * 4 * t * (1 - t) - 0.06
                ctx.sphere((q[0], q[1], zq), 0.035, bulb, name="bulb")
            U.claim_segment(ctx, a, b, 0.15, kind="hang")
        elif what == "net" and net_keys and U.segment_clear(ctx, a, b, 0.4):
            nm = lk.pbr(lib.pick(rng, net_keys), 0.5)
            for q in (a, b):
                ctx.cylinder((q[0], q[1], 0.0), (q[0], q[1], 2.45), 0.03, K.iron, coll="scatter", name="netpost",
                             claim=True)
            ctx.panel(a, b, 1.55, 2.4, nm, tile_m=0.5, coll="scatter", name="net")
            U.claim_segment(ctx, a, b, 0.1, kind="hang")
        else:
            continue
        hung += 1
    info["hanging"] = hung

    tick("furniture")
    # --- the city round the park -------------------------------------------------------------------------
    sides = []
    for sd in range(4):
        if rng.random() < 0.7:
            sides.append((sd, float(rng.uniform(6.0, 22.0))))
    if not sides:
        sides.append((int(rng.integers(4)), 10.0))
    info["city_blocks"] = _city(ctx, K, lk, sides)
    info["city_sides"] = [s for s, _ in sides]
    tick("done")
    return {"sky": sky, **info}


def _blue(rng, rect, dmin, n_try):
    x0, y0, x1, y1 = rect
    pts = []
    for p in np.column_stack([rng.uniform(x0, x1, n_try), rng.uniform(y0, y1, n_try)]):
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= dmin * dmin for q in pts):
            pts.append(p)
    return pts


def _water(lk):
    import bpy  # noqa: PLC0415
    m = bpy.data.materials.new("water")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.02, 0.035, 0.03, 1)
    b.inputs["Roughness"].default_value = 0.03
    b.inputs["IOR"].default_value = 1.33
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 6.0
    bp = nt.nodes.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = 0.08
    nt.links.new(nz.outputs["Fac"], bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    return m


def _dress(ctx, K, lk, info, crowns=True):
    """The patch primitives in park materials, plus copings, stems, crowns, posts."""
    rng, lib = ctx.rng, ctx.lib
    style = str(rng.choice(["hedge", "brick", "stone", "render"], p=[0.35, 0.3, 0.2, 0.15]))
    if style == "hedge":
        wall = K.hedge
    elif style == "brick":
        k = _pick(ctx, BRICK)
        wall = K._snow(lk.pbr(k, float(rng.uniform(1.2, 1.8)), bevel=0.01, dirt=0.4, vary=0.15))
    elif style == "stone":
        k = _pick(ctx, STONE)
        wall = K._snow(lk.pbr(k, float(lib.items[k].get("tile_m") or 2.5), bevel=0.02, dirt=0.4, vary=0.2))
    else:
        k = _pick(ctx, RENDER)
        wall = K._snow(lk.pbr(k, 2.5, bevel=0.01, dirt=0.6, vary=0.2))
    box = K.hedge if style == "hedge" and rng.random() < 0.5 else K.concrete
    pole = K.bark
    bar = K.iron
    rock = K.hedge
    ctx.dress_primitives({"wall": wall, "box": box, "pole": pole, "bar": bar, "rock": rock})
    info["wall_style"] = style
    info["family_regions"] = U.claim_family_regions(ctx, keepout=float(ctx.knobs.get("keepout", 2.5)))
    prims = ctx.spec["primitives"]
    objs = {o.name: o for o in ctx.colls["obstacles"].objects}
    n_v = sum(1 for p in prims if p["kind"] == "cyl_v")
    import build_course as BC  # noqa: PLC0415
    sem = []                                        # what the dressing adds to the family's own geometry
    flared = []
    top = U.cloud_top(ctx)
    for i, p in enumerate(prims):
        k = p["kind"]
        if k == "wall" and style != "hedge":         # a coping on every top piece
            for c, h, R in BC.wall_pieces(p):
                if abs(c[2] + h[2] - p["z1"]) < 1e-3:
                    yaw = math.atan2(R[1, 0], R[0, 0])
                    ctx.box((c[0], c[1], p["z1"] + 0.04), (h[0] + 0.03, h[1] + 0.04, 0.04), K.stone, yaw=yaw,
                            coll="structure", name="coping")
            if p["z1"] < top:
                sem.append({"prim": i, "kind": k, "change": "coping",
                            "detail": f"coping 0.08 m on the wall top ({p['z1']:.2f} m), 3-4 cm overhang"})
        elif k == "cyl_v":
            o = objs.get(f"prim{i:02d}_cyl")
            region = str(p.get("region", ""))
            # a stand's (or a tier-C forest cell's) poles are trees whatever their radius; a ring's are posts
            standlike = region.startswith(("stand", "forest"))
            ring = region.startswith("ring") or (p["r"] < 0.22 and n_v > 12 and not standlike)
            if p["r"] >= 0.35 and not standlike:     # a monument column, statue on top (no park tree is that thick)
                if o is not None:
                    o.data.materials[0] = K.stone
                ctx.cylinder((p["c"][0], p["c"][1], p["z1"]), (p["c"][0], p["c"][1], p["z1"] + 0.3), p["r"] * 1.12,
                             K.stone, name="capital")
                if "ph:gothic_statue" in lib.items and p["z1"] > 4.0:
                    ctx.instance("ph:gothic_statue#0", p["c"], z=p["z1"] + 0.3, s=1.4, coll="structure")
                if p["z1"] < top:
                    sem.append({"prim": i, "kind": k, "change": "capital",
                                "detail": f"capital r x1.12 (+ statue) above z1 {p['z1']:.2f} m"})
            elif ring or (p["r"] < 0.15 and not standlike):   # pergola / colonnade posts, painted
                if o is not None:
                    o.data.materials[0] = K.wood if ring else K.iron
                if not ring and p["z1"] >= 3.0:
                    ctx.cylinder((p["c"][0], p["c"][1], p["z1"]), (p["c"][0], p["c"][1], p["z1"] + 0.12), 0.25,
                                 K.iron, name="lamphead")
                    if p["z1"] < top:
                        sem.append({"prim": i, "kind": k, "change": "lamphead",
                                    "detail": f"lamp head r 0.25 m on the post top ({p['z1']:.2f} m)"})
            elif p["z1"] >= 4.4 and crowns:          # a park tree: the trunk tapers on up into its crown
                s_ = float(rng.uniform(1.8, 2.4))
                h_up = 1.8 + 3.0 * p["r"]
                if U.root_flare(ctx, p["c"], p["r"], K.bark) is not None:
                    flared.append(i)
                U.trunk(ctx, p["c"], h_up, p["r"] * 0.97, K.bark, r_top=max(0.07 * s_, 0.35 * p["r"]), flare=0.0,
                        rough=0.02, z0=p["z1"] - 0.02)
                U.crown(ctx, TREES["airy"][0], p["c"], max(p["z1"] + 0.4, top + 0.05), s_)
            elif p["z1"] >= 4.4:                     # winter: bare limbs above the trunk
                if U.root_flare(ctx, p["c"], p["r"], K.bark) is not None:
                    flared.append(i)
                U.bare_top(ctx, p["c"], p["z1"], p["r"], K.bark)
        elif k == "sphere":
            c = p["c"]
            if c[2] - p["r"] > 0.25:                 # standard topiary: the clipped ball on its stem
                ctx.cylinder((c[0], c[1], 0.0), (c[0], c[1], c[2] - p["r"] * 0.8), 0.035, K.bark, name="stem")
                sem.append({"prim": i, "kind": k, "change": "post_under",
                            "detail": f"stem r 0.035 m from the ground to the ball ({c[2] - p['r']:.2f} m)"})
        elif k == "box":
            lo_z = p["c"][2] - p["half"][2]
            if lo_z > 0.35 and max(p["half"][0], p["half"][1]) > 0.8 and abs(p["rpy"][0]) + abs(p["rpy"][1]) < 1e-3:
                o = objs.get(f"prim{i:02d}_box")
                if o is not None:
                    o.data.materials[0] = K.wood        # a pergola beam / sign board on posts
                yaw = p["rpy"][2]
                cy, sy = math.cos(yaw), math.sin(yaw)
                hx, hy = p["half"][0], p["half"][1]
                for u in (-1, 1):
                    for v in (-1, 1):
                        du, dv = u * max(hx - 0.1, 0.0), v * max(hy - 0.1, 0.0)
                        ctx.cylinder((p["c"][0] + cy * du - sy * dv, p["c"][1] + sy * du + cy * dv, 0.0),
                                     (p["c"][0] + cy * du - sy * dv, p["c"][1] + sy * du + cy * dv, lo_z + 0.01),
                                     0.06, K.iron, name="post")
                sem.append({"prim": i, "kind": k, "change": "posts_under",
                            "detail": f"4 posts r 0.06 m under the corners, 0-{lo_z:.2f} m"})
        elif k == "cyl_h":
            a, b = np.asarray(p["a"], float), np.asarray(p["b"], float)
            d = (b - a)[:2] / max(np.linalg.norm((b - a)[:2]), 1e-9)
            for end, sgn in ((a, -1), (b, 1)):
                xy = end[:2] + sgn * d * 0.1
                if end[2] > 0.5 and ctx.free(xy, 0.06, 0.02):
                    ctx.cylinder((xy[0], xy[1], 0.0), (xy[0], xy[1], end[2] + p["r"]), 0.07, K.iron, name="post",
                                 claim=True)
                    sem.append({"prim": i, "kind": k, "change": "post_at_end",
                                "detail": f"post r 0.07 m under the bar end ({xy[0]:.2f}, {xy[1]:.2f})"})
    if flared:
        sem.append({"prims": flared, "kind": "cyl_v", "change": "root_flare",
                    "detail": "root collar round the pole foot: <= 1.45 r at the ground, 1.16 r at 0.3 m, "
                              "1.07 r at 0.55 m, flush at 0.8 m"})
    info["semantics_changed"] = sem
