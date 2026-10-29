"""Suburban backyard: the rear of a house (siding, brick or render; two
storeys, a bungalow or a flat-roofed modern box) with windows, a patio door
and gutters along one side of the patch, a paved patio with garden
furniture and a parasol, a lawn with beds along a board / picket / hedge /
chain-link boundary, a shed, a trampoline, a swing set, a sandpit or a
paddling pool, bins, planters and trees; neighbours' houses, sheds and
trees beyond the fences as scenery.

What makes it read as a garden rather than props on a plane:

* the lawn is a real lawn -- two grass scans mixed with worn and bare
  patches, mulched beds with shrubs along the fences, a paved patio and a
  stepping-stone path, autumn leaves or snow in season;
* everything is to scale and where it belongs: the patio and its furniture
  against the house, bins by the side fence, the shed in a back corner,
  the trampoline and swings out on the lawn;
* the palette is chosen per house (siding colour, brick, render, roof
  tiles, fence stain) so two gardens are never the same;
* the patch obstacles become garden structure: walls as brick / rendered /
  boarded garden walls with a coping, boxes as clipped hedges, poles as
  trunks or pergola posts, bars as pergola beams on posts, spheres as
  topiary balls or globe lanterns on cords.

Levels: ``clutter`` scales furniture, toys and planters; ``hang`` > 0 adds
washing lines (1.8-2.0 m, with laundry), festoon lights across the lawn
(2.2-3 m) and a rope swing from a tree branch.

Garden furniture, play sets, gazebos and bicycles come partly from Objaverse
(``OBJ_*``, CC-BY / CC-BY-SA, scale-checked by tests/test_kits_farm.py). The
patch obstacles get the farm kit's frames plus hedge shoots on standing
boxes; both are listed in ``semantics_changed`` (course.json ``chosen``).
"""

from __future__ import annotations

import math

import numpy as np

from kits import _farm_util as U
from kits.farm import TREE_SETS, _frames, _half

NAME = "backyard"
INDOOR = False
DESCRIPTION = "suburban backyard: house rear with patio, lawn, board fence, shed, trampoline, swings, washing line"

SEASONS = {"summer": 0.42, "spring": 0.15, "autumn": 0.15, "grey": 0.14, "snow": 0.14}
HDRIS = {
    "summer": ["ph:stuttgart_suburbs", "ph:suburban_garden", "ph:suburban_parking_area", "ph:charolettenbrunn_park",
               "ph:killesberg_park", "ph:symmetrical_garden", "ph:symmetrical_garden_02", "ph:quadrangle_sunny",
               "ph:suburban_soccer_park", "ph:pretoria_gardens", "ph:resting_place_2", "ph:eilenriede_park",
               "ph:park_parking", "ph:meadow_2", "ph:bismarckturm_hillside", "ph:garden_nook", "ph:twilight_sunset",
               "ph:bryanston_park_sunrise"],
    "spring": ["ph:stuttgart_suburbs", "ph:symmetrical_garden_02", "ph:charolettenbrunn_park", "ph:wildflower_field",
               "ph:meadow_2", "ph:pretoria_gardens", "ph:quadrangle_sunny", "ph:etzwihl"],
    "autumn": ["ph:killesberg_park", "ph:tiergarten", "ph:eilenriede_park", "ph:autumn_field", "ph:rosendal_park_sunset",
               "ph:hausdorf_meadow"],
    "grey": ["ph:quadrangle_cloudy", "ph:kloetzle_blei", "ph:greenwich_park_03", "ph:approaching_storm", "ph:suburban_soccer_park",
             "ph:farmland_overcast", "ph:overcast_soil_2"],
    "snow": ["ph:snowy_park_01", "ph:snowy_hillside", "ph:winter_lake_01", "ph:passendorf_snow", "ph:snow_field_2"],
}
LAWN = {    # three lawn scans (mostly green); worn and bare spots are added as patches where people walk and play
    "summer": (["acg:Grass001", "acg:Grass004", "acg:Grass002", "ph:grass_ground"],
               ["acg:Grass007", "acg:Grass001", "ph:leafy_grass", "acg:Grass004"],
               ["acg:Grass002", "acg:Ground037", "acg:Grass006", "ph:grass_ground"]),
    "spring": (["acg:Grass005", "acg:Grass006", "acg:Grass008", "acg:Grass001"],
               ["ph:leafy_grass", "acg:Grass007", "acg:Grass004"],
               ["acg:Ground037", "acg:Grass005", "acg:Grass002"]),
    "autumn": (["acg:Grass002", "acg:Grass003", "acg:Grass001"],
               ["acg:Grass001", "acg:Grass002", "ph:leafy_grass"],
               ["acg:Ground040", "acg:Ground019", "ph:dry_decay_leaves", "acg:Ground041"]),
    "grey": (["acg:Grass002", "acg:Grass001", "acg:Grass003"],
             ["acg:Grass004", "acg:Grass001", "ph:grass_ground"],
             ["acg:Ground001", "acg:Ground013", "ph:grass_ground"]),
    "snow": (["ph:snow_02", "ph:snow_03", "acg:Snow004"], ["acg:Snow010A", "ph:snow_01", "acg:Snow006"],
             ["ph:snow_05", "acg:Snow007A"]),
}
WORN = ["ph:sparse_grass", "acg:Ground003", "ph:grass_path_2", "acg:Ground013", "ph:park_dirt", "acg:Ground037"]
PAVING = ["acg:PavingStones029", "acg:PavingStones037", "acg:PavingStones046", "acg:PavingStones070", "acg:PavingStones071",
          "acg:PavingStones077", "acg:PavingStones108", "acg:PavingStones112", "acg:PavingStones114", "acg:PavingStones118",
          "acg:PavingStones119", "acg:PavingStones122", "acg:PavingStones126A", "acg:PavingStones131", "acg:PavingStones132",
          "acg:PavingStones133", "acg:PavingStones136", "ph:brick_pavement", "ph:stone_tiles", "ph:grey_stone_path",
          "ph:herringbone_brick", "ph:worn_brick_path", "ph:stone_pavers", "ph:precast_stone_paving"]
SIDING_TINTS = [(1.0, 1.0, 1.0), (0.78, 0.86, 0.95), (0.8, 0.9, 0.78), (0.95, 0.88, 0.7), (0.7, 0.72, 0.75),
                (0.62, 0.72, 0.8), (1.0, 0.85, 0.78), (0.55, 0.6, 0.55)]
BRICK = ["ph:red_brick", "ph:red_bricks_02", "ph:brick_wall_001", "ph:brick_wall_02", "ph:brick_wall_04",
         "ph:yellow_brick", "ph:yellow_bricks", "ph:dark_brick_wall", "acg:Bricks052", "acg:Bricks019", "acg:Bricks074",
         "acg:Bricks077", "acg:Bricks085", "ph:whitewashed_brick", "ph:painted_brick", "ph:mixed_brick_wall",
         "ph:white_bricks"]
RENDER = ["ph:white_stucco", "ph:plastered_wall", "ph:plastered_wall_02", "ph:beige_wall_001", "ph:beige_wall_002",
          "ph:white_plaster_rough_01", "ph:white_rough_plaster", "acg:PaintedPlaster017", "ph:yellow_plaster",
          "ph:grey_plaster", "ph:painted_plaster_wall", "acg:Plaster001"]
ROOF_TILE = ["ph:clay_roof_tiles", "ph:clay_roof_tiles_02", "ph:clay_roof_tiles_03", "ph:roof_tiles_14", "ph:grey_roof_tiles",
             "ph:grey_roof_tiles_02", "ph:roof_slates_02", "ph:roof_slates_03", "ph:red_slate_roof_tiles_01", "ph:roof_07",
             "ph:roof_09", "ph:ceramic_roof_01", "acg:RoofingTiles003", "acg:RoofingTiles006", "acg:RoofingTiles011A",
             "acg:RoofingTiles012A", "acg:RoofingTiles013A", "acg:RoofingTiles014A", "ph:grey_roof_01"]
FENCE_BOARDS = ["acg:Planks008", "acg:Planks009", "ph:brown_planks_03", "ph:weathered_planks", "ph:wood_planks_grey",
                "ph:wood_plank_wall", "acg:Planks001", "ph:weathered_brown_planks"]
FENCE_STAINS = [None, (0.55, 0.4, 0.3), (0.35, 0.28, 0.22), (0.7, 0.62, 0.5), (0.45, 0.5, 0.42), (0.9, 0.9, 0.9)]
HEDGE = ["acg:Moss003", "acg:Moss004", "acg:Moss002", "acg:Grass001"]
HEDGE_SPRIGS = ["searsia_lucida", "othonna_cerarioides", "pachira_aquatica_01"]
BARK = ["acg:Bark001", "acg:Bark002", "acg:Bark007", "ph:bark_brown_01", "ph:bark_brown_02", "ph:bark_platanus",
        "ph:tree_bark_03", "ph:bark_willow"]
MULCH = ["ph:wood_chip_path", "acg:WoodChips001", "acg:WoodChips002", "ph:brown_mud_02", "ph:forest_ground_05",
         "ph:yellow_flower_mulch", "acg:Ground048", "ph:raked_dirt"]
FURNITURE = ["outdoor_table_chair_set_01", "wooden_picnic_table", "painted_wooden_bench", "round_wooden_table_02",
             "gallinera_table", "small_wooden_table_01"]
CHAIRS = ["plastic_monobloc_chair_01", "painted_wooden_chair_01", "painted_wooden_chair_02", "gallinera_chair",
          "folding_wooden_stool", "Rockingchair_01", "wooden_stool_01", "metal_stool_02"]
GARDEN_BITS = ["planter_box_01", "planter_box_02", "planter_box_03", "potted_plant_01", "potted_plant_02", "garden_gnome",
               "watering_can_metal_01", "garden_sprinkler_01", "football", "american_football", "rubber_duck_toy",
               "wicker_basket_02", "compost_bags", "compost_bag_02", "wooden_bucket_02", "plastic_crate_02",
               "stone_fire_pit", "wooden_ladder_02", "wheelchair_01"]
SHRUBS = ["searsia_lucida", "othonna_cerarioides", "pachira_aquatica_01", "shrub_02", "anthurium_botany_01",
          "calathea_orbifolia_01", "fern_02", "periwinkle_plant", "flower_gazania", "flower_ursinia", "dandelion_01",
          "celandine_01", "potted_plant_02"]
#: Objaverse models (CC-BY / CC-BY-SA; licences land in course.json "files"), checked for metre scale and look
OBJ_FURNITURE = ["obj:76d6011abc8848c89f080df54ed4d0f8", "obj:884d585e35eb4f389b098581246ea265",
                 "obj:ddb967e02f3c43be9082930d7caae023", "obj:887796ef013c42c9837320edaa316d07",
                 "obj:629fd23c112a4e79ab8470b3e0d2e2b7"]
OBJ_CHAIRS = ["obj:31d46740ef6a41b18f67808225cef468", "obj:3e6b91194c6348c0bc0c88d8d2f83cfd",
              "obj:261d81ec28d845eba3bb7373768c60c5", "obj:d15d2c598774442c91b03478a92cc7b5",
              "obj:bf4c7bcb6d184667822986614a7c5e97"]
OBJ_BITS = ["obj:197dd882a69846e0825e80566996cbba", "obj:3bde1ab67c6448169b604dedad6828d8",
            "obj:f256203558c24ea09385a861b917c75a", "obj:99d6dce2e49e4ba185ac108ab044524d",
            "obj:b43c4f717e904d938dcacfe58c345631", "obj:e559f6bccecf44a2b73e1da8ffce6b66",
            "obj:0fd3dd1da381468c86d3bbe50a4db27b", "obj:68c8d8c7715947d6a393feecc83c0553",
            "obj:8e8fd14f7f1546b9bcf2f5cfee5c116c", "obj:25b14611faa14bae992a7e26f1cafc4b"]
OBJ_PLAY = ["obj:083c55b093ea42d6a6ccb261aff2d6b0", "obj:8712f53da0e04736802f1bc84ca7df97",
            "obj:9ea308ef1062419183ec4315449a3967", "obj:8e116d30fcdd4b81984cf1b8c7c88a28",
            "obj:e65fec66a8d248dcb1086f47df0f41de", "obj:207a8a10d98e4607a7fb1812ee132b4a"]
OBJ_GAZEBO = ["obj:341f8e563e84442984f7286f2266fc68", "obj:100de2d5f0b14cb0b371295d07308e1c",
              "obj:8fd94526f61f41a9b2f5bccbcfb35b9f", "obj:ab95dfb3816e4160989f26972c4c7c39"]
YARD_TREE_FILES = ("ph:island_tree_01", "ph:tree_small_02", "ph:island_tree_02", "ph:searsia_burchellii")
PALETTE = [(0.1, 0.25, 0.55), (0.6, 0.1, 0.08), (0.1, 0.4, 0.2), (0.85, 0.55, 0.08), (0.1, 0.1, 0.1),
           (0.85, 0.85, 0.82), (0.45, 0.2, 0.5), (0.2, 0.5, 0.6)]


def _pick_season(rng):
    ks = list(SEASONS)
    p = np.array([SEASONS[k] for k in ks])
    return ks[int(rng.choice(len(ks), p=p / p.sum()))]


class Yard:
    def __init__(self, ctx, season):
        self.ctx, self.season, self.rng, self.lk = ctx, season, ctx.rng, ctx.looks
        lk, rng = self.lk, self.rng
        self.snow = season == "snow"
        self.white = lk.paint((0.8, 0.8, 0.78), rough=0.45, name="trim_white")
        self.glass = lk.glass(tint=(0.3, 0.34, 0.37))
        self.black = lk.paint((0.03, 0.03, 0.03), rough=0.35, name="enamel_black")
        self.steel = U.mat(ctx, ["acg:Metal032", "acg:Metal009", "ph:metal_plate", "acg:Metal035"], 1.0, bevel=0.005)
        self.bark = U.mat(ctx, BARK, 1.2, dirt=0.2)
        self.hedge = U.mat(ctx, HEDGE, 0.55, tint=(0.8, 0.92, 0.7), bump=1.0, vary=0.3, jitter=0.08)
        self.sprigs = ctx.lib.models(ids=HEDGE_SPRIGS, h=(0.4, 1.3), max_tris=60000)

        self.timber = U.mat(ctx, ["ph:wood_planks", "acg:Planks009", "acg:Planks021", "ph:weathered_brown_planks",
                                  "ph:brown_planks_03", "acg:Planks029L", "ph:wood_floor_deck"], 1.4, bevel=0.01, dirt=0.25)
        stain = FENCE_STAINS[int(rng.integers(len(FENCE_STAINS)))]
        self.fence_wood = U.mat(ctx, FENCE_BOARDS, 1.8, tint=stain, bevel=0.005, dirt=0.3, jitter=0.1)
        self.post = U.mat(ctx, ["ph:rough_wood", "ph:wood_planks_grey", "acg:Wood062", "ph:weathered_planks"], 1.0,
                          tint=stain, bevel=0.01, dirt=0.3)
        self.plastic = [lk.paint(c, rough=0.4, name="plastic") for c in PALETTE]
        self.rubber = U.mat(ctx, ["acg:Rubber001", "acg:Rubber002"], 0.5, tint=(0.3, 0.3, 0.3))
        self.roof_snow = U.mat(ctx, ["acg:Snow004", "ph:snow_02"], 3.0) if self.snow else None

    def sprig_top(self, c, half, yaw, top):
        """Loose shoots along the top of a clipped hedge box: breaks the
        ruler-straight silhouette. None within the start / goal keep-out
        column (a box near the goal keeps its flat top there)."""
        ctx, rng = self.ctx, self.rng
        if not self.sprigs or self.snow:
            return
        R = U.yaw_R(yaw)
        keep = [(f["c"], f["r"]) for f in ctx.foot if f.get("kind") in ("start", "goal")]
        for t in np.arange(-half[0] + 0.3, half[0] - 0.2, float(rng.uniform(0.45, 0.8))):
            off = R @ np.array([t, rng.uniform(-half[1] * 0.7, half[1] * 0.7), 0.0])
            k = ctx.lib.pick(rng, self.sprigs)
            sz = ctx.lib.size(k)[2]
            s_ = float(rng.uniform(0.35, 0.6)) / max(sz, 0.3)
            q = np.array([c[0] + off[0], c[1] + off[1]])
            if any(np.linalg.norm(q - kc) < kr + _half(ctx, k, s_) + 0.1 for kc, kr in keep):
                continue
            ctx.instance(k, q, z=top - 0.25, yaw=float(rng.uniform(0, 6.28)), s=s_, coll="scatter")

    # --- houses -------------------------------------------------------------------------------
    def house_mats(self):
        rng, ctx, lk = self.rng, self.ctx, self.lk
        fam = str(rng.choice(["siding", "brick", "render", "stone"], p=[0.35, 0.3, 0.3, 0.05]))
        if fam == "siding":
            k = U.pick(ctx, ["acg:WoodSiding009", "acg:WoodSiding009", "acg:WoodSiding010", "acg:WoodSiding003",
                             "ph:weathered_plank_siding"])
            tint = SIDING_TINTS[int(rng.integers(len(SIDING_TINTS)))] if k == "acg:WoodSiding009" else None
            wall = lk.pbr(k, 1.8, tint=tint, bevel=0.015, dirt=0.25, jitter=0.03)
        elif fam == "brick":
            wall = lk.pbr(U.pick(ctx, BRICK), 1.5, bevel=0.015, dirt=0.25, jitter=0.03)
        elif fam == "render":
            tint = [None, (1.0, 0.95, 0.85), (0.95, 0.9, 0.8), (0.85, 0.88, 0.9)][int(rng.integers(4))]
            wall = lk.pbr(U.pick(ctx, RENDER), 2.5, tint=tint, bevel=0.015, dirt=0.3, jitter=0.03)
        else:
            wall = lk.pbr(U.pick(ctx, ["ph:stone_wall", "ph:rustic_stone_wall", "ph:yellow_stone_wall",
                                       "ph:stone_brick_wall_001"]), 2.0, bevel=0.02, dirt=0.2)
        if self.roof_snow is not None:
            roof, rt = self.roof_snow, 3.0
        else:
            k = U.pick(ctx, ROOF_TILE)
            roof, rt = lk.pbr(k, None, proj="uv", dirt=0.25), float(ctx.lib.items[k].get("tile_m") or 2.0)
        trim = self.white if rng.random() < 0.7 else lk.paint([(0.12, 0.12, 0.12), (0.25, 0.2, 0.15),
                                                               (0.3, 0.35, 0.4)][int(rng.integers(3))], rough=0.4,
                                                              name="trim_dark")
        return fam, wall, roof, rt, trim

    def house(self, o, d, L, W, coll="structure", rear=True, style=None):
        """A house with its rear (v = -W/2 face) toward the garden."""
        ctx, rng = self.ctx, self.rng
        fam, wall, roof, rt, trim = self.house_mats()
        style = style or str(rng.choice(["two", "two", "bungalow", "modern"]))
        if style == "modern":
            h, kind, pitch = float(rng.uniform(5.8, 6.6)), "flat", 0.0
            roof = self.roof_snow or U.mat(ctx, ["ph:bitumen", "ph:concrete_floor_02"], 3.0)
            rt = 3.0
        elif style == "bungalow":
            h, kind, pitch = float(rng.uniform(2.8, 3.2)), "gable", float(rng.uniform(25, 38))
        else:
            h, kind, pitch = float(rng.uniform(5.4, 6.0)), "gable", float(rng.uniform(30, 45))
        b = U.Building(ctx, o, d, L, W, h, wall, roof, kind=kind, pitch=pitch, overhang=float(rng.uniform(0.3, 0.55)),
                       roof_tile=rt, coll=coll, trim=trim if kind == "gable" else None)
        if style == "modern":                           # parapet
            items = []
            for sv in (-1, 1):
                items.append((b.W3(0, sv * (W / 2 - 0.1), h + 0.25), (L / 2, 0.1, 0.25), b.yaw))
            U.boxes(ctx, items, wall, coll=coll, name="parapet")
        floors = [0.95] if style == "bungalow" else [0.95, 3.7]
        # the garden face: patio door, back door, windows; the far face gets windows too (seen from the neighbours)
        n_bays = max(3, int(L / 2.9))
        bays = np.linspace(-L / 2 + L / (2 * n_bays), L / 2 - L / (2 * n_bays), n_bays)
        door_bay = int(rng.integers(n_bays))
        patio_bay = (door_bay + int(rng.integers(1, n_bays))) % n_bays
        for i, u in enumerate(bays):
            for fz in floors:
                if fz < 1.0 and i == door_bay:
                    self.back_door(b, u, trim)
                elif fz < 1.0 and i == patio_bay:
                    w = 2.6 if style == "modern" else 2.2
                    b.window(self.glass, trim, side=-1, u=u, w=w, z0=0.08, z1=2.2, panes=(2, 1))
                else:
                    big = style == "modern" and rng.random() < 0.6
                    w = float(rng.uniform(1.6, 2.4)) if big else float(rng.uniform(0.9, 1.3))
                    z0 = fz if not big else fz - 0.4
                    b.window(self.glass, trim, side=-1, u=u, w=w, z0=z0, z1=z0 + (1.3 if not big else 1.8),
                             panes=(2, 2) if not big else (1, 1), sill=style != "modern")
            if rear:
                for fz in floors:
                    b.window(self.glass, trim, side=1, u=u, w=1.1, z0=fz, z1=fz + 1.3, panes=(2, 2), sill=True)
        # gutter + downpipe on the garden face
        if kind == "gable":
            gm = [self.white, self.black, self.steel][int(rng.integers(3))]
            vg, zg = b.profile[0][0] + 0.07, b.profile[0][1] - 0.07
            U.boxes(ctx, [(b.W3(0, vg, zg), (L / 2 + 0.3, 0.06, 0.06), b.yaw)], gm, coll=coll, name="gutter")
            ue = L / 2 - 0.3 if rng.random() < 0.5 else -L / 2 + 0.3
            U.tube(ctx, [b.W3(ue, vg, zg), b.W3(ue, -(W / 2 + 0.08), zg - 0.4),
                         b.W3(ue, -(W / 2 + 0.08), 0.15), b.W3(ue, -(W / 2 + 0.25), 0.02)], 0.04, gm, sides=10,
                   coll=coll, name="downpipe")
        return b, style

    def back_door(self, b, u, trim):
        rng, lk = self.rng, self.lk
        col = [(0.08, 0.2, 0.12), (0.5, 0.08, 0.06), (0.1, 0.12, 0.3), (0.85, 0.85, 0.82), (0.15, 0.15, 0.15),
               (0.35, 0.25, 0.15)][int(rng.integers(6))]
        panel = lk.pbr(U.pick(self.ctx, ["acg:PaintedWood009C", "acg:PaintedWood006C", "acg:PaintedWood005"]), 1.0,
                       tint=[c * 1.6 for c in col], bevel=0.01)
        b.door(panel, trim, side=-1, u=u, w=0.95, h=2.1)
        b.window(self.glass, trim, side=-1, u=u, w=0.5, z0=1.3, z1=1.9, panes=(1, 1)) if rng.random() < 0.5 else None
        # step and a wall light
        U.boxes(self.ctx, [b.on_face(-1, u, 1.4, 0.0, 0.16, 0.5)], U.mat(self.ctx, ["ph:concrete_floor_02",
                                                                                  "acg:PavingStones029"], 1.0,
                                                                         bevel=0.01), coll=b.coll, name="step")
        lamp = U.have(self.ctx, ["ph:industrial_wall_lamp", "ph:security_light", "ph:industrial_wall_sconce"])
        if lamp and rng.random() < 0.7:
            p = b.W3(u + 0.85, -(b.W / 2 + 0.08), 2.1)
            self.ctx.instance(lamp[int(rng.integers(len(lamp)))] + "#0", p[:2], z=p[2],
                              yaw=b.yaw + math.pi, coll=b.coll)

    def shed(self, o, d, L, W):
        ctx, rng, lk = self.ctx, self.rng, self.lk
        k = U.pick(ctx, ["acg:WoodSiding003", "acg:WoodSiding009", "acg:WoodSiding010", "ph:weathered_plank_siding",
                         "acg:Planks029L", "ph:wood_plank_wall", "acg:CorrugatedSteel006A", "acg:CorrugatedSteel005"])
        tint = SIDING_TINTS[int(rng.integers(len(SIDING_TINTS)))] if k == "acg:WoodSiding009" else None
        wall = lk.pbr(k, 1.4, tint=tint, bevel=0.01, dirt=0.35)
        roof = self.roof_snow or U.mat(ctx, ["ph:bitumen", "ph:roof_slates_02", "ph:corrugated_iron", "ph:grey_roof_01"],
                                       None, proj="uv")
        kind = str(rng.choice(["gable", "skillion"]))
        b = U.Building(ctx, o, d, L, W, float(rng.uniform(1.9, 2.2)), wall, roof, kind=kind,
                       pitch=float(rng.uniform(20, 30)) if kind == "gable" else 8.0, overhang=0.15, roof_tile=1.5,
                       trim=self.white if rng.random() < 0.5 else self.post)
        b.door(self.timber, self.post, side=-1, u=float(rng.uniform(-L / 4, L / 4)), w=0.8, h=1.8, brace=self.post)
        if rng.random() < 0.7:
            b.window(self.glass, self.white, end=int(rng.choice([-1, 1])), v=0.0, w=0.6, z0=1.1, z1=1.6, panes=(2, 1))
        return b

    # --- garden objects -------------------------------------------------------------------------
    def trampoline(self, c, R, net):
        ctx, rng = self.ctx, self.rng
        pad = self.plastic[int(rng.choice([0, 2, 7, 3]))]
        zf = 0.88
        ring = [np.array([c[0] + R * math.cos(a), c[1] + R * math.sin(a), zf]) for a in np.linspace(0, 2 * math.pi, 40,
                                                                                                    endpoint=False)]
        U.tube(ctx, ring, 0.024, self.steel, sides=8, closed=True, coll="scatter", name="tramp_frame")
        padr = [np.array([c[0] + (R - 0.15) * math.cos(a), c[1] + (R - 0.15) * math.sin(a), zf + 0.02])
                for a in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
        U.tube(ctx, padr, 0.14, pad, sides=6, closed=True, coll="scatter", name="tramp_pad")
        U.cyl(ctx, (c[0], c[1], zf - 0.02), (c[0], c[1], zf), R - 0.28, self.black, coll="scatter", name="tramp_mat",
                     segments=40)
        legs = []
        for a in np.linspace(0, 2 * math.pi, 6, endpoint=False) + rng.uniform(0, 1):
            p = np.array([c[0] + R * math.cos(a), c[1] + R * math.sin(a)])
            legs.append([np.r_[p, zf], np.r_[p, 0.0]])
        for L_ in legs:
            U.tube(ctx, L_, 0.022, self.steel, sides=8, coll="scatter", name="tramp_leg")
        if net:
            nk = U.pick(ctx, ["acg:Net001A", "acg:Net002A", "acg:Net003A", "acg:Net004A"])
            nm = ctx.looks.pbr(nk, None, tint=(0.25, 0.25, 0.25)) if nk else self.black
            n = 8
            angs = np.linspace(0, 2 * math.pi, n, endpoint=False) + rng.uniform(0, 1)
            top = 2.65
            for i, a in enumerate(angs):
                p = np.array([c[0] + (R - 0.05) * math.cos(a), c[1] + (R - 0.05) * math.sin(a)])
                U.tube(ctx, [np.r_[p, zf], np.r_[p, top]], 0.02, pad, sides=8, coll="scatter", name="tramp_pole")
                a2 = angs[(i + 1) % n]
                q = np.array([c[0] + (R - 0.05) * math.cos(a2), c[1] + (R - 0.05) * math.sin(a2)])
                ctx.panel(p, q, zf + 0.1, top - 0.05, nm, tile_m=0.4, coll="scatter", name="tramp_net")
            topr = [np.array([c[0] + (R - 0.05) * math.cos(a), c[1] + (R - 0.05) * math.sin(a), top])
                    for a in np.linspace(0, 2 * math.pi, 32, endpoint=False)]
            U.tube(ctx, topr, 0.018, self.steel, sides=6, closed=True, coll="scatter", name="tramp_top")
        ctx.claim_disc(c, R + 0.2, kind="trampoline")

    def swing_set(self, c, yaw, L):
        ctx, rng = self.ctx, self.rng
        wooden = rng.random() < 0.45
        mat_ = self.timber if wooden else self.plastic[int(rng.choice([0, 1, 2, 7]))]
        H = float(rng.uniform(2.0, 2.3))
        R = U.yaw_R(yaw)
        d, n = R[:2, 0], R[:2, 1]
        c = np.asarray(c, float)
        spread = 0.95
        segs = []
        for s in (-1, 1):
            top = np.r_[c + d * s * L / 2, H]
            for t in (-1, 1):
                segs.append((np.r_[c + d * s * (L / 2 + 0.15) + n * t * spread, 0.0], top))
        top_a, top_b = np.r_[c - d * (L / 2 + 0.1), H], np.r_[c + d * (L / 2 + 0.1), H]
        if wooden:
            U.beams(ctx, segs + [(top_a, top_b)], 0.09, 0.09, mat_, coll="scatter", name="swing_frame")
        else:
            for a_, b_ in segs + [(top_a, top_b)]:
                U.tube(ctx, [a_, b_], 0.03, mat_, sides=10, coll="scatter", name="swing_frame")
        seat = self.black if rng.random() < 0.6 else self.plastic[int(rng.integers(len(self.plastic)))]
        chain = self.steel
        ns = int(rng.integers(1, 3)) if L < 2.4 else int(rng.integers(2, 4))
        for k in range(ns):
            u = (k - (ns - 1) / 2) * L / max(ns, 1) * 0.9
            zs = float(rng.uniform(0.42, 0.55))
            sway = rng.normal(0, 0.06)
            for t in (-0.22, 0.22):
                p = c + d * (u + t)
                U.tube(ctx, [np.r_[p, H - 0.03], np.r_[p + n * sway, zs + 0.02]], 0.007, chain, sides=5, coll="scatter",
                       name="swing_chain")
            ps = c + d * u + n * sway
            U.boxes(ctx, [((ps[0], ps[1], zs), (0.24, 0.1, 0.015), yaw)], seat, coll="scatter", name="swing_seat")
        ctx.claim_rect(c, (L / 2 + 0.3, spread + 0.2), yaw, kind="swings")

    def parasol(self, c, r):
        ctx, rng = self.ctx, self.rng
        cloth = [self.white, *self.plastic][int(rng.integers(len(self.plastic) + 1))]
        h = float(rng.uniform(2.1, 2.4))
        U.cyl(ctx, (c[0], c[1], 0), (c[0], c[1], h + 0.25), 0.025, self.steel, coll="scatter", name="parasol_pole",
                     segments=10)
        U.cyl(ctx, (c[0], c[1], 0), (c[0], c[1], 0.08), 0.25, self.black, coll="scatter", name="parasol_base",
                     segments=16)
        k = 8
        V = [(c[0] + r * math.cos(2 * math.pi * i / k), c[1] + r * math.sin(2 * math.pi * i / k), h) for i in range(k)]
        V.append((c[0], c[1], h + 0.45))
        F = [(i, (i + 1) % k, k) for i in range(k)]
        ctx.mesh(V, F, cloth, coll="scatter", name="parasol")

    def kettle_bbq(self, c):
        ctx = self.ctx
        ctx.sphere((c[0], c[1], 0.82), 0.29, self.black, coll="scatter", name="bbq")
        for a in (0.0, 2.1, 4.2):
            p = np.array([c[0] + 0.28 * math.cos(a), c[1] + 0.28 * math.sin(a)])
            U.tube(ctx, [np.r_[p, 0.0], np.array([c[0], c[1], 0.7]) + np.r_[(p - c[:2]) * 0.4, 0]], 0.012, self.steel,
                   sides=6, coll="scatter", name="bbq_leg")
        ctx.claim_disc(c, 0.45, kind="bbq")

    def gas_bbq(self, c, yaw):
        ctx = self.ctx
        U.boxes(ctx, [((c[0], c[1], 0.45), (0.45, 0.28, 0.42), yaw)], self.black, coll="scatter", name="bbq_cart")
        U.boxes(ctx, [((c[0], c[1], 1.02), (0.42, 0.26, 0.16), yaw)], self.steel, coll="scatter", name="bbq_lid")
        for s in (-1, 1):
            p = np.array(c) + U.yaw_R(yaw)[:2, 0] * s * 0.62
            U.boxes(ctx, [((p[0], p[1], 0.88), (0.18, 0.26, 0.015), yaw)], self.steel, coll="scatter", name="bbq_shelf")
        ctx.claim_rect(c, (0.85, 0.35), yaw, kind="bbq")

    def wheelie_bins(self, c, yaw, n):
        ctx, rng = self.ctx, self.rng
        R = U.yaw_R(yaw)
        for i in range(n):
            col = [(0.08, 0.28, 0.1), (0.08, 0.15, 0.4), (0.05, 0.05, 0.05), (0.35, 0.35, 0.35), (0.35, 0.2, 0.08),
                   (0.6, 0.45, 0.05)][int(rng.integers(6))]
            m = self.lk.paint(col, rough=0.5, name="bin")
            p = np.asarray(c, float) + R[:2, 0] * (i - (n - 1) / 2) * 0.66
            y = yaw + rng.normal(0, 0.05)
            U.boxes(ctx, [((p[0], p[1], 0.5), (0.29, 0.36, 0.48), y), ((p[0], p[1], 1.0), (0.31, 0.39, 0.025), y)], m,
                    coll="scatter", name="wheelie_bin")
        ctx.claim_rect(c, (n * 0.33 + 0.1, 0.45), yaw, kind="bins")

    def raised_bed(self, c, yaw, L, W, soil, plants):
        ctx, rng = self.ctx, self.rng
        R = U.yaw_R(yaw)
        d, n = R[:2, 0], R[:2, 1]
        h = float(rng.uniform(0.25, 0.45))
        items = []
        for s in (-1, 1):
            items.append((np.r_[c + n * s * (W / 2 - 0.02), h / 2], (L / 2, 0.02, h / 2), yaw))
            items.append((np.r_[c + d * s * (L / 2 - 0.02), h / 2], (0.02, W / 2, h / 2), yaw))
        U.boxes(ctx, items, self.timber, coll="scatter", name="bed_frame")
        cc = np.asarray(c)
        V = [np.r_[cc + d * su * (L / 2 - 0.04) + n * sv * (W / 2 - 0.04), h - 0.05]
             for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        ctx.mesh(V, [(0, 1, 2, 3)], soil, coll="scatter", name="bed_soil")
        if plants:
            for _ in range(int(L * W * 4)):
                p = cc + d * rng.uniform(-L / 2 + 0.15, L / 2 - 0.15) + n * rng.uniform(-W / 2 + 0.15, W / 2 - 0.15)
                ctx.instance(ctx.lib.pick(rng, plants), p, z=h - 0.05, yaw=float(rng.uniform(0, 6.28)),
                             s=float(rng.uniform(0.8, 1.4)), coll="scatter")
        ctx.claim_rect(c, (L / 2 + 0.1, W / 2 + 0.1), yaw, kind="bed")

    def paddling_pool(self, c, r):
        ctx = self.ctx
        col = self.plastic[int(self.rng.choice([0, 7, 3]))]
        U.cyl(ctx, (c[0], c[1], 0), (c[0], c[1], 0.3), r, col, coll="scatter", name="pool", segments=36)
        water = self.lk.paint((0.08, 0.18, 0.2), rough=0.03, name="pool_water")
        U.cyl(ctx, (c[0], c[1], 0.3), (c[0], c[1], 0.303), r - 0.1, water, coll="scatter", name="pool_water",
                     segments=36)
        ctx.claim_disc(c, r + 0.1, kind="pool")

    def sandpit(self, c, yaw, s):
        ctx = self.ctx
        items = []
        R = U.yaw_R(yaw)
        for k in range(2):
            ax = R[:2, k]
            for sg in (-1, 1):
                p = np.asarray(c) + ax * sg * (s / 2 - 0.05)
                items.append((np.r_[p, 0.12], (0.05, s / 2, 0.12) if k == 0 else (s / 2, 0.05, 0.12), yaw))
        U.boxes(ctx, items, self.timber, coll="scatter", name="sandpit")
        sand = U.mat(ctx, ["ph:playground_sand", "ph:park_sand", "ph:sand_01"], 1.0, proj="flat")
        cc = np.asarray(c)
        V = [np.r_[cc + R[:2, 0] * a * (s / 2 - 0.1) + R[:2, 1] * b * (s / 2 - 0.1), 0.14]
             for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        ctx.mesh(V, [(0, 1, 2, 3)], sand, coll="scatter", name="sand")
        ctx.claim_rect(c, (s / 2 + 0.1, s / 2 + 0.1), yaw, kind="sandpit")

    def goal(self, c, yaw):
        ctx = self.ctx
        R = U.yaw_R(yaw)
        d, n = R[:2, 0], R[:2, 1]
        c = np.asarray(c, float)
        w, h, dp = 1.8, 1.2, 0.8
        pts = [np.r_[c - d * w / 2, 0], np.r_[c - d * w / 2, h], np.r_[c + d * w / 2, h], np.r_[c + d * w / 2, 0]]
        U.tube(ctx, pts, 0.025, self.white, sides=8, coll="scatter", name="goal")
        for s in (-1, 1):
            U.tube(ctx, [np.r_[c + d * s * w / 2, h], np.r_[c + d * s * w / 2 + n * dp, 0]], 0.015, self.white, sides=6,
                   coll="scatter", name="goal_back")
        nk = U.pick(ctx, ["acg:Net001A", "acg:Net002A", "acg:Net004A"])
        if nk:
            nm = ctx.looks.pbr(nk, None)
            ctx.panel(c - d * w / 2 + n * dp * 0.5, c + d * w / 2 + n * dp * 0.5, 0.0, h * 0.55, nm, tile_m=0.3,
                      coll="scatter", name="goal_net")
        ctx.claim_rect(c + n * dp / 2, (w / 2 + 0.1, dp / 2 + 0.1), yaw, kind="goal")

    def washing(self, a, b, z, clothes=True):
        """A washing line a -> b at height z with pegged laundry."""
        ctx, rng = self.ctx, self.rng
        line = self.lk.paint((0.85, 0.85, 0.8), rough=0.5, name="line")
        P = U.catenary(np.r_[a, z], np.r_[b, z], float(rng.uniform(0.08, 0.2)), 12)
        U.tube(ctx, P, 0.006, line, sides=5, coll="scatter", name="washing_line")
        if not clothes:
            return
        L = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
        dvec = (np.asarray(b) - np.asarray(a)) / max(L, 1e-9)
        yaw = math.atan2(dvec[1], dvec[0])
        t = float(rng.uniform(0.3, 0.8))
        while t < L - 0.5:
            w, h = float(rng.uniform(0.35, 0.9)), float(rng.uniform(0.4, 0.8))
            if rng.random() < 0.8:
                k = int(np.clip((t + w / 2) / L * 11, 0, 11))
                zc = P[k][2] - h / 2 - 0.01
                p = np.asarray(a) + dvec * (t + w / 2)
                col = (self.plastic + [self.white])[int(rng.integers(len(self.plastic) + 1))]
                U.boxes(ctx, [((p[0], p[1], zc), (w / 2, 0.006, h / 2), yaw)], col, coll="scatter", name="laundry")
            t += w + float(rng.uniform(0.1, 0.6))


def _hedge_row(ctx, yard, a, b, h, w):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    d = (b - a) / L
    yaw = math.atan2(d[1], d[0])
    items = []
    t = 0.0
    while t < L:
        seg = min(float(ctx.rng.uniform(2.0, 4.0)), L - t)
        c = a + d * (t + seg / 2)
        if not ctx.in_zone(c, 1.0) or ctx.free_rect(c, (seg / 2, w / 2), yaw, margin=0.2):
            hh = h + ctx.rng.normal(0, 0.06)
            items.append(((c[0], c[1], hh / 2), (seg / 2 + 0.02, w / 2 + ctx.rng.normal(0, 0.03), hh / 2), yaw))
            ctx.claim_rect(c, (seg / 2, w / 2), yaw, kind="hedge")
            if ctx.in_zone(c, 4.0):
                yard.sprig_top(c, (seg / 2, w / 2), yaw, hh)
        t += seg
    U.boxes(ctx, items, yard.hedge, name="hedge")


def build(ctx) -> dict:
    rng, lk = ctx.rng, ctx.looks
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    season = _pick_season(rng)
    sky = lk.sky(U.pick_hdri(ctx, HDRIS[season], ("garden", "park")), target_ev=float(rng.uniform(0.2, 0.55)))
    yard = Yard(ctx, season)
    info = {"season": season, "sky": sky}

    # --- ground: lawn --------------------------------------------------------------------------
    a_, b_, c_ = LAWN[season]
    keys = [U.pick(ctx, a_), U.pick(ctx, b_)] + ([U.pick(ctx, c_)] if season in ("autumn", "snow") else [])
    if len(keys) == 2 and rng.random() < 0.65:          # a used lawn: thin, worn and bare stretches
        keys.append(U.pick(ctx, WORN))
    puddles = float(rng.choice([0.0, 0.3])) if season == "grey" else 0.0
    ctx.ground(lk.ground_material(keys, tile_m=float(rng.uniform(1.6, 2.4)), puddles=puddles,
                                  blend_scale=float(rng.uniform(5.0, 10.0))))
    info["ground"] = keys

    # --- the patch obstacles as garden structure ----------------------------------------------------
    tall = max([p.get("z1", 0.0) for p in ctx.spec["primitives"] if p["kind"] == "wall"] or [0.0]) > 2.6
    wfam = str(rng.choice(["brick", "render", "board", "stone"], p=[0.4, 0.35, 0.0, 0.25] if tall else [0.35, 0.3, 0.2, 0.15]))
    if wfam == "brick":
        wall = U.mat(ctx, BRICK, 1.5, bevel=0.02, dirt=0.25)
    elif wfam == "render":
        wall = U.mat(ctx, RENDER, 2.5, tint=[None, (1.0, 0.93, 0.82), (0.9, 0.92, 0.95)][int(rng.integers(3))],
                     bevel=0.02, dirt=0.3)
    elif wfam == "board":
        wall = yard.fence_wood
    else:
        wall = U.mat(ctx, ["ph:stone_wall", "ph:rustic_stone_wall", "ph:yellow_stone_wall", "ph:stacked_stone_wall"],
                     2.0, bevel=0.02, dirt=0.2)
    pole_kind = "trunk" if rng.random() < 0.55 else "post"
    pole = yard.bark if pole_kind == "trunk" else U.mat(ctx, ["acg:PaintedWood009C", "acg:PaintedWood006C",
                                                              "ph:wood_planks_grey"], 1.0, bevel=0.01)
    bar = yard.timber if rng.random() < 0.6 else yard.plastic[int(rng.choice([0, 1, 2, 7]))]
    ctx.dress_primitives({"wall": wall, "box": yard.hedge, "pole": pole, "bar": bar, "rock": yard.hedge})
    U.claim_openings(ctx, float(ctx.knobs.get("keepout", 2.5)))
    lamp = lk.paint((0.95, 0.92, 0.85), rough=0.15, name="globe", emission=(1.0, 0.9, 0.7, 1.5))
    cap = U.mat(ctx, ["ph:concrete_floor_02", "acg:PavingStones029", "ph:stone_tiles"], 1.0, bevel=0.01) \
        if wfam != "board" else yard.post
    # floating boxes are timber (a hung planter / a sign board), floating spheres globe lanterns
    sprigs = False
    for i, p, objs in U.prim_list(ctx):
        if p["kind"] == "box" and p["c"][2] - p["half"][2] > 0.3:
            for o in objs:
                o.data.materials[0] = yard.timber
        elif p["kind"] == "box" and max(abs(v) for v in p.get("rpy", [0, 0, 0])[:2]) < 0.05:
            if yard.sprigs and not yard.snow:
                sprigs = True
            h_ = p["half"]
            yaw_ = float(p.get("rpy", [0, 0, 0])[2])
            if h_[0] >= h_[1]:
                yard.sprig_top(p["c"], (h_[0], h_[1]), yaw_, p["c"][2] + h_[2])
            else:
                yard.sprig_top(p["c"], (h_[1], h_[0]), yaw_ + math.pi / 2, p["c"][2] + h_[2])
        if p["kind"] == "sphere" and p["c"][2] - p["r"] > 0.3:
            for o in objs:
                o.data.materials[0] = lamp
    column = wall if wfam != "board" else U.mat(ctx, BRICK, 1.5, bevel=0.02, dirt=0.25)
    changed = _frames(ctx, yard, cap, yard.post, lk.paint((0.05, 0.05, 0.05), rough=0.5, name="cord"), pole_kind,
                      lamp=None, thick=(column, None))
    if sprigs:
        changed.append("hedge shoots on standing boxes (up to +0.35 m, ~0.2 m past the sides)")
    info.update({"walls": wfam, "poles": pole_kind, "semantics_changed": changed})

    # --- the house along one side, its patio ----------------------------------------------------------
    side = int(rng.choice([-1, 1]))
    yb = y1 if side > 0 else y0
    yo = y0 if side > 0 else y1
    d = np.array([float(side), 0.0])
    patio_d = float(rng.uniform(2.5, 4.5))
    house_style = None
    houses = []
    x = zx0 - float(rng.uniform(2, 10))
    first = True
    while x < zx1 + 4:
        L = float(rng.uniform(9, 16))
        W = float(rng.uniform(8, 11))
        for g in (patio_d, patio_d + 1.0, patio_d + 2.0):
            yc = yb + side * (g + W / 2)
            c = np.array([x + L / 2, yc])
            if ctx.free_rect(c, (L / 2, W / 2), 0.0, margin=0.3, step=0.6):
                b, style = yard.house(c, d, L, W, style=house_style)
                house_style = house_style or style
                ctx.claim_rect(c, (L / 2 + 0.3, W / 2 + 0.3), 0.0, kind="house")
                b.foot = ctx.foot[-1]                       # a festoon from its eave skips its own footprint
                houses.append((b, g))
                break
        # the next house along is the neighbour's (semi-detached or a gap and a side fence)
        x += L + (0.0 if rng.random() < 0.4 else float(rng.uniform(2.0, 4.0)))
        first = False
    info["house_style"] = house_style
    # patio in front of the house(s) facing the patch
    pave_k = U.pick(ctx, PAVING)
    pave = lk.pbr(pave_k, float(rng.uniform(1.6, 2.6)), proj="flat", vary=0.15, jitter=0.0)
    px0 = float(rng.uniform(x0 - 4, x0 + 6))
    px1 = px0 + float(rng.uniform(6, 14))
    ye = yb + side * (houses[0][1] if houses else patio_d)
    ctx.flat([(px0, ye), (px1, ye), (px1, ye - side * patio_d), (px0, ye - side * patio_d)], pave, z=0.012, name="patio")
    info["paving"] = pave_k

    # --- boundary: back fence / hedge, end fences ------------------------------------------------------
    btype = str(rng.choice(["board", "picket", "hedge", "chain", "board"]))
    y_f = yo - side * float(rng.uniform(0.6, 1.8))
    fm = {"post": yard.post, "rail": yard.post, "board": yard.fence_wood, "board_w": float(rng.uniform(0.09, 0.16)),
          "board_gap": 0.012}
    fh = float(rng.uniform(1.7, 2.0))
    ex0, ex1 = x0 - float(rng.uniform(1.0, 3.0)), x1 + float(rng.uniform(1.0, 3.0))
    y_house = yb + side * (houses[0][1] if houses else patio_d)

    def boundary(a, b, kind):
        if kind == "hedge":
            _hedge_row(ctx, yard, a, b, float(rng.uniform(1.8, 2.4)), float(rng.uniform(0.8, 1.2)))
        elif kind == "picket":
            pm = dict(fm, board=yard.white, post=yard.white, rail=yard.white, board_w=0.08, board_gap=0.07)
            U.fence_line(ctx, a, b, 2.2, "picket", pm, h=float(rng.uniform(1.0, 1.2)))
        elif kind == "chain":
            ck = U.pick(ctx, ["acg:Fence001", "acg:Fence002", "acg:Fence005", "acg:Fence006"])
            posts = U.fence_line(ctx, a, b, 2.8, "wire", {"post": yard.steel, "wire": yard.steel, "strands": 2,
                                                          "post_w": 0.06}, h=1.5)
            if ck:
                cm = lk.pbr(ck, None)
                a2, b2 = np.asarray(a, float), np.asarray(b, float)
                L_ = float(np.linalg.norm(b2 - a2))
                n_ = max(1, int(L_ / 6))
                for i in range(n_):
                    p0 = a2 + (b2 - a2) * i / n_
                    p1 = a2 + (b2 - a2) * (i + 1) / n_
                    if not ctx.in_zone((p0 + p1) / 2, 3) or ctx.free_rect((p0 + p1) / 2, (L_ / n_ / 2, 0.05),
                                                                         math.atan2(*(b2 - a2)[::-1]), margin=0.1):
                        ctx.panel(p0, p1, 0.02, 1.5, cm, tile_m=1.0, name="chainlink")
        else:
            U.fence_line(ctx, a, b, 2.4, "board", fm, h=fh)

    boundary((zx0 - 40, y_f), (zx1 + 40, y_f), btype)
    etype = btype if rng.random() < 0.7 else "board"
    for xe in (ex0, ex1):
        boundary((xe, y_f), (xe, y_house), etype)
    info["boundary"] = btype
    # the next gardens past the end fences: a shed in their back corner (in the zone: checked and claimed
    # before our own planting), their far fence line
    for xe, dirn in ((ex0, -1), (ex1, 1)):
        xs = xe + dirn * float(rng.uniform(4, 8))
        if rng.random() < 0.6:
            Ls, Ws = float(rng.uniform(2.4, 3.4)), float(rng.uniform(2.0, 2.6))
            cs = (xs, y_f + side * 2.5)
            if ctx.free_rect(cs, (Ls / 2 + 0.2, Ws / 2 + 0.2), 0.0, margin=0.3):
                yard.shed(cs, d, Ls, Ws)
                ctx.claim_rect(cs, (Ls / 2 + 0.2, Ws / 2 + 0.2), 0.0, kind="shed")
        U.fence_line(ctx, (xe + dirn * 14, y_f), (xe + dirn * 14, y_house), 2.4, "board", fm, h=fh, check=False,
                     coll="backdrop", claim=False)

    # beds along the back fence with shrubs
    mulch = lk.pbr(U.pick(ctx, MULCH), 1.5, proj="flat", vary=0.2, jitter=0.0)
    bw = float(rng.uniform(0.9, 1.6))
    if btype != "hedge":
        U.jagged_strip(ctx, (zx0 - 40, y_f + side * (bw / 2 + 0.1)), (zx1 + 40, y_f + side * (bw / 2 + 0.1)), bw, mulch,
                       z=0.008, jit=0.06, name="bed")
    shrubs = ctx.lib.models(ids=SHRUBS, h=(0.15, 2.4), max_tris=150000)
    if season != "snow":
        for xs in np.arange(zx0 + 0.5, zx1 - 0.5, float(rng.uniform(0.9, 1.6))):
            xy = np.array([xs + rng.normal(0, 0.3), y_f + side * (bw / 2 + 0.1 + rng.normal(0, bw / 4))])
            k = ctx.lib.pick(rng, shrubs)
            s = float(rng.uniform(0.8, 1.2))
            if k and ctx.free(xy, _half(ctx, k, s), 0.2):
                ctx.instance(k, xy, 0.0, float(rng.uniform(0, 6.28)), s, coll="scatter", claim=True)

    # --- neighbours: the houses behind the back fence (scenery; claimed so the tree rows keep off them) -----
    ny = y_f - side * float(rng.uniform(14, 22))
    x = zx0 - 50 + float(rng.uniform(0, 10))
    while x < zx1 + 50:
        L = float(rng.uniform(9, 15))
        W = float(rng.uniform(8, 10))
        ch = (x + L / 2, ny - side * W / 2)
        yard.house(ch, -d, L, W, coll="backdrop", rear=False)
        ctx.claim_rect(ch, (L / 2 + 0.7, W / 2 + 0.7), 0.0, kind="scenery")
        x += L + float(rng.uniform(3, 8))

    # --- yard trees (in the cloud: few), neighbour trees beyond the fences ---------------------------------
    files = [f for f in TREE_SETS if f in ctx.lib.items]
    rng.shuffle(files)
    files = files[:2]
    fk = [k for f in files for k in TREE_SETS[f][0]]
    tk = [k for k in fk if k.split("#")[0] in YARD_TREE_FILES]
    chase = np.array([x0 - 3.0, (y0 + y1) / 2])                   # keep the preview chase camera out of the leaves
    placed_trees = []                                             # (xy, height, crown r, footprint, placed entry)
    for _ in range(60):
        if len(placed_trees) >= int(rng.integers(1, 3)) or not tk:
            break
        k = tk[int(rng.integers(len(tk)))]
        s = float(rng.uniform(0.9, 1.25))
        sx, sy, sz = np.asarray(ctx.lib.size(k)) * s
        rr = 0.5 * max(sx, sy)
        xy = np.array([rng.uniform(zx0 + 1, zx1 - 1), y_f + side * float(rng.uniform(1.0, 3.5))])
        if ctx.free(xy, rr, 0.2) and np.linalg.norm(xy - chase) > rr + 3.0:
            ctx.instance(k, xy, 0.0, float(rng.uniform(0, 6.28)), s, coll="structure")
            ctx.claim_disc(xy, rr, kind="tree")
            placed_trees.append((xy, sz, rr, ctx.foot[-1], ctx.placed[-1]))
    avoid = ("house", "scenery", "shed")                          # no trunk inside a neighbour's house
    U.tree_row(ctx, fk, (zx0 - 80, y_f - side * 14.5), (zx1 + 80, y_f - side * 14.5), float(rng.uniform(7, 12)),
               s=(0.8, 1.3), skip=0.35, offset_j=3.0, avoid=avoid)
    U.tree_row(ctx, fk, (zx0 - 100, y_f - side * 30), (zx1 + 100, y_f - side * 30), float(rng.uniform(6, 10)),
               s=(0.9, 1.4), skip=0.2, offset_j=6.0, avoid=avoid)

    # --- shed in a back corner ---------------------------------------------------------------------
    for _ in range(12):
        Ls, Ws = float(rng.uniform(2.4, 3.6)), float(rng.uniform(1.9, 2.6))
        xs = float(rng.choice([rng.uniform(ex0 + Ls / 2 + 0.4, ex0 + 5), rng.uniform(ex1 - 5, ex1 - Ls / 2 - 0.4)]))
        ys = y_f + side * (Ws / 2 + float(rng.uniform(0.3, 1.2)))
        if ctx.free_rect((xs, ys), (Ls / 2, Ws / 2), 0.0, margin=0.3):
            yard.shed((xs, ys), -d, Ls, Ws)
            ctx.claim_rect((xs, ys), (Ls / 2 + 0.2, Ws / 2 + 0.2), 0.0, kind="shed")
            # a slab / stepping stones toward the patio
            stone = lk.pbr(U.pick(ctx, ["acg:PavingStones029", "ph:stone_tiles", "ph:grey_stone_path"]), 1.0,
                           proj="flat", jitter=0.0)
            p0 = np.array([xs, ys - side * (Ws / 2 + 0.5)])
            p1 = np.array([(px0 + px1) / 2, ye - side * patio_d])
            n_st = int(np.linalg.norm(p1 - p0) / 0.75)
            for i in range(n_st):
                q = p0 + (p1 - p0) * (i + 0.5) / n_st + rng.normal(0, 0.05, 2)
                a = rng.uniform(0, 1)
                if U.foot_clearance(ctx, q) < 0.35:     # the path goes round the patch's walls, not under them
                    continue
                ctx.flat([q + 0.24 * np.array([math.cos(a + k * math.pi / 2), math.sin(a + k * math.pi / 2)])
                          for k in range(4)], stone, z=0.01, name="stepstone")
            info["shed"] = True
            break

    # --- patio furniture ----------------------------------------------------------------------------
    patio_rect = (px0, min(ye, ye - side * patio_d), px1, max(ye, ye - side * patio_d))
    furn = ctx.lib.models(ids=FURNITURE + OBJ_FURNITURE, h=(0.3, 2.6))
    chairs = ctx.lib.models(ids=CHAIRS + OBJ_CHAIRS, h=(0.3, 1.4))
    placed = ctx.scatter(furn, max(1, ctx.count(1.2)), rect=patio_rect, margin=0.4, yaw=float(rng.choice([0, math.pi / 2])))
    ctx.scatter(chairs, ctx.count(float(rng.uniform(1.5, 3.5))), rect=patio_rect, margin=0.25)
    if rng.random() < 0.65:
        for _t in range(30):
            c = np.array([rng.uniform(px0 + 1, px1 - 1), rng.uniform(patio_rect[1] + 1, patio_rect[3] - 0.5)])
            r = float(rng.uniform(1.1, 1.5))
            if ctx.free(c, 0.4, 0.3) and all(ctx.clearance(c + r * np.array([math.cos(a), math.sin(a)])) > -0.5
                                             for a in np.linspace(0, 6.28, 8)):
                yard.parasol(c, r)
                ctx.claim_disc(c, r, kind="parasol")        # the canopy: no festoon through it, no BBQ under it
                break
    for _t in range(30):
        c = np.array([rng.uniform(px0 + 0.5, px1 - 0.5), rng.uniform(*sorted((ye - side * 0.6, ye - side * 1.2)))])
        if ctx.free(c, 0.6, 0.3):
            if rng.random() < 0.5:
                yard.kettle_bbq(c)
            else:
                yard.gas_bbq(c, 0.0)
            break

    # bins by the house end
    for _t in range(20):
        c = np.array([float(rng.choice([ex0 + 1.0, ex1 - 1.0])) + rng.normal(0, 0.4), ye - side * float(rng.uniform(0.5, 1.0))])
        nb = int(rng.integers(1, 4))
        if ctx.free_rect(c, (nb * 0.33, 0.4), 0.0, margin=0.2):
            yard.wheelie_bins(c, 0.0, nb)
            break

    # --- lawn: trampoline, swings, pool/sandpit, goal, beds, bits ------------------------------------------
    ko = float(ctx.knobs.get("keepout", 2.5))
    clut = float(ctx.knobs.get("clutter", 1.0))
    worn = lk.pbr(U.pick(ctx, WORN), 1.5, proj="flat", vary=0.2, jitter=0.0)
    play, gazebos = U.obj_models(ctx, OBJ_PLAY), U.obj_models(ctx, OBJ_GAZEBO)
    lawn_items = ["trampoline", "swings", "pool", "sandpit", "goal", "bed", "bed", "rotary"] + \
        ["play", "play"] * bool(play) + ["gazebo"] * bool(gazebos)       # (no Objaverse catalog: the kit's own)
    rng.shuffle(lawn_items)
    n_items = int(round(rng.uniform(2, 4) * clut * ctx.area_f ** 0.5))
    done = []
    for what in lawn_items[:max(1, n_items)]:
        for _t in range(60):
            xy = ctx.uniform_xy(pad=1.5)
            if what == "trampoline":
                R = float(rng.uniform(1.6, 2.2))
                if ctx.free(xy, R + 0.2, 0.4 + 0.3 * ko):
                    yard.trampoline(xy, R, net=rng.random() < 0.65)
                    U.blob(ctx, xy, R * 0.8, R * 0.8, worn, z=0.004, name="worn")
                    done.append(what)
                    break
            elif what == "swings":
                L = float(rng.uniform(2.0, 3.4))
                yaw = float(rng.choice([0.0, math.pi / 2]))
                if ctx.free_rect(xy, (L / 2 + 0.3, 1.2), yaw, margin=0.4 + 0.3 * ko):
                    yard.swing_set(xy, yaw, L)
                    for k_ in range(int(L / 0.75)):     # scuffed bare earth under the seats
                        U.blob(ctx, xy + U.yaw_R(yaw)[:2, 0] * (k_ - (int(L / 0.75) - 1) / 2) * 0.75, 0.45, 0.7,
                               worn, z=0.004, name="worn")
                    done.append(what)
                    break
            elif what == "pool":
                r = float(rng.uniform(0.8, 1.4))
                if ctx.free(xy, r, 0.4):
                    yard.paddling_pool(xy, r)
                    done.append(what)
                    break
            elif what == "sandpit":
                s = float(rng.uniform(1.4, 2.2))
                if ctx.free_rect(xy, (s / 2, s / 2), 0.0, margin=0.4):
                    yard.sandpit(xy, 0.0, s)
                    done.append(what)
                    break
            elif what == "goal":
                yaw = float(rng.uniform(0, 2 * math.pi))
                if ctx.free_rect(xy, (1.1, 0.6), yaw, margin=0.4):
                    yard.goal(xy, yaw)
                    U.blob(ctx, xy - U.yaw_R(yaw)[:2, 1] * 0.8, 1.0, 0.7, worn, z=0.004, name="worn")
                    done.append(what)
                    break
            elif what == "bed":
                L, W = float(rng.uniform(1.8, 3.0)), float(rng.uniform(0.9, 1.3))
                yaw = float(rng.choice([0.0, math.pi / 2]))
                if ctx.free_rect(xy, (L / 2, W / 2), yaw, margin=0.4):
                    soil = lk.pbr(U.pick(ctx, ["ph:farm_soil", "ph:brown_mud_02", "ph:forest_ground_05"]), 1.2, proj="flat")
                    veg = ctx.lib.models(ids=["nettle_plant", "weed_plant_02", "celandine_01", "calathea_orbifolia_01",
                                              "anthurium_botany_01", "dandelion_01"], h=(0.05, 0.6))
                    yard.raised_bed(xy, yaw, L, W, soil, veg if season != "snow" else [])
                    done.append(what)
                    break
            elif what in ("play", "gazebo"):
                keys = play if what == "play" else gazebos
                if U.place_model(ctx, ctx.lib.pick(rng, keys), xy, float(rng.choice([0.0, math.pi / 2])),
                                 margin=0.4 + 0.3 * ko):
                    done.append(what)
                    break
            elif what == "rotary":
                if ctx.free(xy, 1.4, 0.4):
                    # rotary washing line, folded or open
                    U.cyl(ctx, (xy[0], xy[1], 0), (xy[0], xy[1], 1.95), 0.025, yard.steel, coll="scatter",
                                 name="rotary_pole", segments=10)
                    a0 = float(rng.uniform(0, 1))
                    for a in np.linspace(0, 2 * math.pi, 4, endpoint=False) + a0:
                        e = xy + 1.3 * np.array([math.cos(a), math.sin(a)])
                        U.tube(ctx, [np.r_[xy, 1.9], np.r_[e, 1.75]], 0.012, yard.steel, sides=6, coll="scatter",
                               name="rotary_arm")
                    for rr in (0.45, 0.85, 1.25):
                        ring = [np.r_[xy + rr * np.array([math.cos(a), math.sin(a)]), 1.9 - 0.15 * rr / 1.3]
                                for a in np.linspace(0, 2 * math.pi, 4, endpoint=False) + a0]
                        U.tube(ctx, ring, 0.004, yard.white, sides=4, closed=True, coll="scatter", name="rotary_line")
                    ctx.claim_disc(xy, 1.35, kind="rotary")
                    done.append(what)
                    break
    info["lawn"] = done
    bits = ctx.lib.models(ids=GARDEN_BITS + OBJ_BITS, h=(0.05, 1.8))
    ctx.scatter(bits, ctx.count(float(rng.uniform(3, 7))), margin=0.35)

    # autumn leaves in piles / snow: nothing extra; spring: flowers along the patio edge
    if season in ("spring", "summer"):
        fl = ctx.lib.models(ids=["flower_gazania", "flower_ursinia", "periwinkle_plant", "celandine_01", "dandelion_01"],
                            h=(0.05, 0.5))
        for xs in np.arange(px0, px1, 0.5):
            xy = np.array([xs + rng.normal(0, 0.1), ye - side * (patio_d + 0.25)])
            k = ctx.lib.pick(rng, fl)
            s = float(rng.uniform(1.0, 1.6))
            if k and ctx.clearance(xy) > 0.05 + _half(ctx, k, s):
                ctx.instance(k, xy, 0.0, float(rng.uniform(0, 6.28)), s)

    # --- hanging obstacles in the flight band ----------------------------------------------------------
    hang = float(ctx.knobs.get("hang", 0.0))
    n_hang = ctx.count(hang) if hang > 0 else 0
    made = 0
    swing_trees = list(placed_trees)
    bulb = lk.paint((1.0, 0.9, 0.7), rough=0.2, name="bulb", emission=(1.0, 0.8, 0.5, 4.0))
    cord = lk.paint((0.03, 0.03, 0.03), rough=0.5, name="cord")
    for _ in range(n_hang * 40):
        if made >= n_hang:
            break
        what = rng.random()
        if what < 0.45:
            # washing line between two T-posts
            a = ctx.uniform_xy(pad=0.5)
            b = a + rng.normal(0, 1, 2) * 4
            L = float(np.linalg.norm(b - a))
            if not (3.0 < L < 9.0 and ctx.in_zone(b) and ctx.free(a, 0.1, 0.4) and ctx.free(b, 0.1, 0.4)
                    and U.seg_clear(ctx, a, b, 2.0) and U.seg_free(ctx, a, b, 0.7, kinds=None)):
                continue
            z = float(rng.uniform(1.8, 2.0))
            dv = (b - a) / L
            nv = np.array([-dv[1], dv[0]])
            for p in (a, b):
                U.cyl(ctx, (p[0], p[1], 0), (p[0], p[1], z + 0.1), 0.035, yard.steel, coll="scatter", name="line_post",
                             segments=10)
                U.tube(ctx, [np.r_[p - nv * 0.5, z + 0.05], np.r_[p + nv * 0.5, z + 0.05]], 0.02, yard.steel, sides=6,
                       coll="scatter", name="line_t")
            for off in (-0.35, 0.0, 0.35):
                yard.washing(a + nv * off, b + nv * off, z, clothes=off == 0.0 or rng.random() < 0.4)
            U.claim_line(ctx, a, b, 0.55)
            made += 1
        elif what < 0.8:
            # festoon lights, 3-12 m: from the eave of the nearest house (leaving its face) or a pole, to a
            # pole, sagging to 2.2-3 m, through nothing that stands that tall
            a = ctx.uniform_xy(pad=0.3)
            b = a + rng.normal(0, 1, 2) * 5
            skip = ()
            if houses and rng.random() < 0.5:
                def along(h):
                    return float((b - h.o) @ h.d)
                hb = min((h for h, _ in houses), key=lambda h: abs(np.clip(along(h), -h.L / 2 + 1, h.L / 2 - 1) - along(h)))
                u = float(np.clip(along(hb) + rng.normal(0, 2.5), -hb.L / 2 + 1, hb.L / 2 - 1))
                ve, ze = hb.profile[0]                          # the roof's garden-side edge (a gutter under it)
                if ze < 3.6:                                    # a low eave: hung from the gutter's outer face
                    pa = hb.W3(u, ve - 0.03, ze - 0.1)
                else:                                           # on the wall, well under the eave
                    pa = hb.W3(u, -(hb.W / 2 + 0.05), min(hb.h - 0.2, 3.2))
                if (b - pa[:2]) @ -hb.n < 0.5 * np.linalg.norm(b - pa[:2]):
                    continue                                    # not along the facade: out into the garden
                skip = (getattr(hb, "foot", None),)
            else:
                if not (ctx.free(a, 0.1, 0.4)):
                    continue
                pa = np.r_[a, 3.2]
            span = float(np.linalg.norm(pa[:2] - b))
            P = U.catenary(pa, np.r_[b, 3.15], float(rng.uniform(0.4, 0.8)), 16)
            if not (3.0 <= span <= 12.0 and ctx.in_zone(b) and ctx.free(b, 0.1, 0.4) and U.seg_clear(ctx, pa, b, 2.0)
                    and U.line_free(ctx, P, 0.4, skip=skip, trim=(0.3, 0.1))):
                continue
            for p in ([] if skip else [pa]) + [np.r_[b, 3.2]]:          # a pole at each free end
                U.cyl(ctx, (p[0], p[1], 0), (p[0], p[1], 3.3), 0.05, yard.post, coll="scatter", name="light_pole",
                             segments=10)
            U.tube(ctx, P, 0.006, cord, sides=5, name="festoon")
            L = float(np.linalg.norm(P[-1] - P[0]))
            nb = int(L / 0.6)
            for i in range(1, nb):
                k = int(round(i / nb * 15))
                ctx.sphere((P[k][0], P[k][1], P[k][2] - 0.1), 0.035, bulb, name="bulb")
            U.claim_line(ctx, pa, b)
            made += 1
        else:
            # a rope swing: a branch from the trunk at crown height out to the rope, which hangs under the
            # crown clear of everything but its own tree
            if not swing_trees:
                continue
            xy, sz, rr, tfoot, tplaced = swing_trees[int(rng.integers(len(swing_trees)))]
            ang = float(rng.uniform(0, 2 * math.pi))
            p = xy + float(rng.uniform(0.45, 0.8)) * rr * np.array([math.cos(ang), math.sin(ang)])
            zt = min(4.2, sz * 0.7)
            branch = [np.r_[xy - (p - xy) / np.linalg.norm(p - xy) * 0.3, zt + 0.35], np.r_[p + (p - xy) * 0.15, zt - 0.1]]
            skip = (tfoot, tplaced)
            if not (U.seg_clear(ctx, xy, p, 2.0) and U.line_free(ctx, branch, 0.3, skip=skip)
                    and U.line_free(ctx, [np.r_[p, zt], np.r_[p, 0.4]], 0.6, kinds=None, skip=skip)):
                continue
            U.tube(ctx, [np.r_[p, zt], np.r_[p, 0.5]], 0.015, lk.paint((0.6, 0.5, 0.35), rough=0.8, name="rope"),
                   sides=6, name="rope_swing")
            U.boxes(ctx, [((p[0], p[1], 0.48), (0.25, 0.12, 0.02), ang)], yard.timber, name="swing_board")
            U.tube(ctx, branch, 0.075, yard.bark, sides=8, name="branch")
            ctx.claim_disc(p, 0.5, kind="rope_swing")
            swing_trees = [t for t in swing_trees if t[3] is not tfoot]      # one swing per tree
            made += 1
    info["hanging"] = made
    return info
