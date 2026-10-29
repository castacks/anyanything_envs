"""Farmyard: a barn / machine shed / stone byre row along one side of the
patch, a paddock fence with a gate along the other, a rutted farm track,
hay (round, wrapped and square bales), a wagon or a tractor, troughs,
barrels, log piles and pallets in the yard, and fields, tree lines, grain
bins, a farmhouse and a pole line as scenery.

What makes it read as a farm rather than props on a plane:

* the ground is a farmyard -- packed dirt, trodden grass and gravel mixed
  by noise, a two-rut track with a grass crown, a concrete apron in front
  of the barn doors, mud and puddles on wet days, snow in winter;
* the buildings are built to size: a 16-26 m timber or sheet-metal barn
  (gable or gambrel roof) with braced sliding doors (some open onto a dark
  interior), windows and a hay-loft door; an open-fronted pole shed full of
  bales and machinery; a stone byre with a tiled roof;
* the season picks the whole palette: HDRI, ground scans, roof snow, hay
  colour (summer / dry / wet / autumn / snow);
* the patch obstacles are dressed as farm structure: walls as boarded,
  stone or block walls with a cap, poles as posts or pollarded trunks,
  bars and floating boxes as rails/beams on posts, spheres as boulders or
  hanging lamps.

Levels: ``clutter`` scales yard props and bales; ``hang`` > 0 adds a sagging
service cable from the pole line to the barn, electric-fence tape between
light posts at 0.8-1.2 m, and lamp ropes, all in the flight band.

Machines and props come partly from Objaverse (``OBJ_*``: real tractors,
loaders, wagons, pickups, barrels, generators; CC-BY, scale-checked by
tests/test_kits_farm.py) and fall back to the built ones without that catalog.

Contract notes: the entrance of every post ring is kept clear like a wall
slot (``_farm_util.claim_openings``); what ``_frames`` adds to the patch's own
obstacles (wall caps, posts under floating boxes/bars, ropes above floating
spheres, ...) changes the tier-A family's geometry, and is returned as
``semantics_changed`` (course.json ``chosen``). Scenery past the ends of the
yard stays wholly outside the cloud zone and off the building row
(``_end_scenery``); the pole shed's contents stay inside its end walls;
hanging lines are strung in the zone through nothing that stands that tall
(``_farm_util.line_free``).
"""

from __future__ import annotations

import math

import numpy as np

import build_course as BC
from kits import _farm_util as U

NAME = "farm"
INDOOR = False
DESCRIPTION = "farmyard: barn and pole shed, post-and-rail paddock fence, hay bales, wagon/tractor, fields and tree lines"

SEASONS = {"summer": 0.34, "dry": 0.18, "wet": 0.18, "autumn": 0.14, "snow": 0.16}

HDRIS = {
    "summer": ["ph:farm_field", "ph:etzwihl", "ph:dry_orchard_meadow", "ph:hausdorf_meadow", "ph:lilienstein",
               "ph:ludwikowice_farmland", "ph:straw_rolls_field_01", "ph:sunset_meadow_path", "ph:ticknock_04",
               "ph:waterkloof_farm", "ph:wildflower_field", "ph:steinbach_field", "ph:evening_meadow",
               "ph:citrus_orchard", "ph:rosendal_plains_2", "ph:dry_hay_field", "ph:meadow_2",
               "ph:bismarckturm_hillside", "ph:resting_place_2"],
    "dry": ["ph:dry_hay_field", "ph:autumn_field", "ph:rosendal_plains_2", "ph:hay_bales", "ph:straw_rolls_field_01",
            "ph:evening_field", "ph:dry_meadow", "ph:farm_sunset", "ph:steinbach_field", "ph:ticknock_04"],
    "wet": ["ph:farmland_overcast", "ph:overcast_soil", "ph:overcast_soil_2", "ph:mud_road", "ph:farm_road",
            "ph:misty_farm_road", "ph:approaching_storm", "ph:greenwich_park_03", "ph:kloetzle_blei"],
    "autumn": ["ph:autumn_field", "ph:autumn_hilly_field", "ph:hausdorf_meadow", "ph:evening_meadow",
               "ph:overcast_soil_2", "ph:farm_sunset", "ph:tiergarten"],
    "snow": ["ph:passendorf_snow", "ph:horn-koppe_snow", "ph:snow_field", "ph:snow_field_2", "ph:snowy_field",
             "ph:snowy_hillside_02", "ph:winter_lake_01", "ph:stierberg_sunrise", "ph:snowy_park_01"],
}

GROUND = {   # (packed yard dirt, grass, third) candidates per season
    "summer": (["ph:dirt_floor", "ph:park_dirt", "ph:dry_ground_01", "acg:Ground048", "acg:Ground067", "ph:brown_mud_dry",
                "ph:forest_ground_04", "acg:Ground023"],
               ["acg:Grass001", "acg:Grass004", "acg:Grass007", "ph:grass_ground", "ph:sparse_grass", "acg:Ground037",
                "acg:Ground003", "ph:leafy_grass", "ph:forrest_ground_01"],
               ["acg:Gravel022", "acg:Gravel026", "ph:gravel_road", "ph:rocky_trail", "ph:gravel_ground_01", "acg:Ground072",
                "ph:grass_path_2"]),
    "dry": (["ph:dry_ground_01", "ph:brown_mud_dry", "acg:Ground072", "acg:Ground073", "ph:dirt_floor", "acg:Ground062S"],
            ["acg:Ground102", "acg:Ground103", "acg:Ground063", "acg:Grass004", "ph:grass_path_3", "acg:Ground074"],
            ["ph:gravel_road", "acg:Gravel026", "ph:sandy_gravel", "ph:rocky_trail", "acg:Ground078"]),
    "wet": (["ph:brown_mud", "ph:brown_mud_02", "ph:brown_mud_03", "ph:muddy_tracks", "acg:Ground044", "acg:Ground045"],
            ["acg:Grass002", "acg:Grass003", "acg:Ground047", "acg:Ground001", "ph:grass_ground"],
            ["ph:brown_mud_rocks_01", "acg:Ground043", "ph:gravel_road", "acg:Ground046"]),
    "autumn": (["ph:brown_mud_leaves_01", "acg:Ground040", "acg:Ground041", "ph:park_dirt", "ph:forest_ground_04"],
               ["acg:Grass003", "acg:Grass002", "ph:forrest_ground_01", "acg:Ground013", "ph:dry_decay_leaves"],
               ["acg:Ground042", "ph:forest_leaves_02", "ph:gravel_road", "ph:raked_dirt"]),
    "snow": (["ph:snow_02", "ph:snow_03", "acg:Snow004", "acg:Snow006", "ph:snow_01"],
             ["ph:snow_03", "acg:Snow010A", "acg:Snow007A", "ph:snow_field_aerial"],
             ["acg:Snow012", "acg:Snow002", "ph:snow_01", "acg:Snow005"]),
}

TRACK = {"summer": ["ph:muddy_tracks", "ph:gravel_road", "acg:Ground067", "ph:dirt_floor", "ph:rocky_trail"],
         "dry": ["ph:gravel_road", "ph:dry_ground_01", "acg:Ground073", "ph:sandy_gravel"],
         "wet": ["ph:muddy_tracks", "ph:brown_mud_02", "ph:brown_mud_rocks_01"],
         "autumn": ["ph:muddy_tracks", "ph:brown_mud_leaves_01", "ph:gravel_road"],
         "snow": ["ph:snow_04", "ph:snow_05", "ph:asphalt_snow"]}

BARN_WALLS = {   # family -> (keys, tile, tint choices)
    "red": (["acg:WoodSiding002", "acg:WoodSiding005"], 2.2, [None, (1.1, 0.9, 0.9), (0.9, 0.85, 0.85)]),
    "weathered": (["acg:WoodSiding010", "acg:WoodSiding011", "acg:WoodSiding012", "acg:WoodSiding013",
                   "ph:weathered_plank_siding", "ph:weathered_planks", "ph:brown_planks_03", "ph:weathered_brown_planks"],
                  2.2, [None, (0.95, 0.95, 1.0)]),
    "dark": (["acg:WoodSiding001", "acg:WoodSiding003", "acg:WoodSiding004", "acg:WoodSiding006", "acg:WoodSiding007",
              "acg:WoodSiding008", "ph:wood_plank_wall"], 2.2, [None]),
    "sheet": (["acg:CorrugatedSteel002", "acg:CorrugatedSteel005", "acg:CorrugatedSteel006A", "acg:CorrugatedSteel006B",
               "acg:CorrugatedSteel007A", "ph:corrugated_iron", "ph:worn_corrugated_iron", "ph:rusty_corrugated_iron"],
              1.3, [None, (0.8, 0.8, 0.8)]),
    "painted": (["acg:WoodSiding009", "acg:PaintedWood006A", "acg:PaintedWood008A", "acg:PaintedWood009A"], 2.0,
                [None, (0.9, 0.9, 0.9)]),
}
ROOFS_METAL = ["ph:corrugated_iron", "ph:rusty_corrugated_iron", "ph:worn_corrugated_iron", "ph:box_profile_metal_sheet",
               "acg:CorrugatedSteel005", "acg:CorrugatedSteel002", "acg:CorrugatedSteel006A"]
ROOFS_TILE = ["ph:clay_roof_tiles", "ph:clay_roof_tiles_02", "ph:roof_tiles_14", "ph:grey_roof_tiles", "ph:roof_slates_02",
              "ph:red_slate_roof_tiles_01", "ph:ceramic_roof_01", "ph:roof_07", "ph:roof_09", "acg:RoofingTiles006",
              "acg:RoofingTiles013A"]
STONE = ["ph:stone_wall", "ph:old_stone_wall", "ph:rustic_stone_wall", "ph:mossy_stone_wall", "ph:stone_wall_02",
         "ph:stacked_stone_wall", "ph:plastered_stone_wall", "ph:yellow_stone_wall", "ph:stone_block_wall",
         "ph:rustic_stone_wall_02", "ph:old_stone_wall_02"]
BRICK = ["ph:red_brick", "ph:red_bricks_02", "ph:brick_wall_001", "ph:worn_brick_wall", "ph:mixed_brick_wall",
         "acg:Bricks052", "acg:Bricks019", "ph:medieval_red_brick", "ph:slumped_mortar_brick"]
PLASTER = ["ph:white_stucco", "ph:plastered_wall", "ph:plastered_wall_02", "ph:worn_plaster_wall", "ph:yellow_plaster",
           "ph:beige_wall_001", "ph:white_plaster_rough_01", "acg:PaintedPlaster017", "ph:clay_plaster"]
BLOCK = ["ph:concrete_block_wall", "ph:concrete_block_wall_02", "ph:concrete_block_wall_03", "ph:rough_block_wall"]
PLANKS = ["ph:weathered_brown_planks", "ph:wood_planks", "ph:weathered_planks", "ph:brown_planks_03", "acg:Planks023A",
          "acg:Planks027A", "acg:Planks029L", "ph:old_planks_02", "ph:rough_wood", "ph:wood_planks_grey"]
BARK = ["acg:Bark001", "acg:Bark002", "acg:Bark007", "acg:Bark012", "ph:bark_brown_01", "ph:bark_brown_02",
        "ph:pine_bark", "ph:bark_willow", "ph:tree_bark_03"]
ROCK = ["ph:rock_boulder_dry", "ph:mossy_rock", "ph:lichen_rock", "ph:rock_face", "ph:rock_05", "ph:rock_boulder_cracked",
        "ph:rock_wall_08"]
HAY = ["ph:reed_roof_04", "ph:reed_roof_04", "ph:reed_roof_03", "acg:ThatchedRoof001A"]
#: tree files cheap enough to import (a file imports whole: fir_tree_01 ~100 s, jacaranda ~50 s on 4 threads);
#: (piece keys, scale range for scenery rows). A build uses at most two of these files.
TREE_SETS = {"ph:island_tree_01": (["ph:island_tree_01#0"], (1.0, 1.7)),
             "ph:island_tree_02": (["ph:island_tree_02#0"], (1.3, 2.0)),
             "ph:fir_sapling_medium": (["ph:fir_sapling_medium#0", "ph:fir_sapling_medium#1", "ph:fir_sapling_medium#2"],
                                       (1.2, 1.8)),
             "ph:tree_small_02": (["ph:tree_small_02#0"], (1.3, 2.0)),
             "ph:searsia_burchellii": (["ph:searsia_burchellii#0", "ph:searsia_burchellii#1"], (1.0, 1.5))}
YARD_PROPS = ["Barrel_01", "Barrel_02", "barrel_03", "wooden_barrels_01", "wine_barrel_01", "wooden_crate_01",
              "wooden_crate_02", "plastic_crate_02", "metal_jerrycan", "metal_jerrycan_green", "plastic_jerrycan",
              "wooden_bucket_01", "wooden_bucket_02", "old_tyre", "compost_bags", "compost_bag_02", "cement_bag",
              "propane_tank", "small_lpg_tank", "rusted_wheel_rim_01", "metal_trash_can", "wooden_military_crate",
              "plastic_container", "hand_truck", "wooden_ladder", "tree_stump_01", "tree_stump_02", "portable_generator"]
#: Objaverse models (CC-BY; licences land in course.json "files"), checked for metre scale and look:
#: farm tractors (modern and vintage), loaders / telehandlers, farm wagons, pickups; small yard props.
OBJ_TRACTORS = ["068f17e7bcf74d04bacf5524c4a2079d", "16d799890d6c499a8762677449b9e179", "1813d718b3b84a9bbe1a0993e18f84d3",
                "2b7c6999b4f84f408752dd0ac038eb86", "45d5c17faee745f3889074435bede0b2", "4ff450c68b7143ea9df2e005147c0618",
                "56682eb214214102a00a5e62b391670b", "6080b7f9e1264520bc70478cfa952307", "c333e8395bef42c5b5305c4620ad698c",
                "d891c103cddd44b69d1357c832d9ad56", "e929f07c86f1435f9ff51808892ce9fc", "e9e844f2f9364044a181ce8f38a83dc3",
                "ae2d1d85b1a34466bf64810b85050a33"]
OBJ_LOADERS = ["d23c7e5565104a1db2e1c4641d4ed59b", "df9c3cfa98bb40598cfa7b96027db2a5"]
OBJ_WAGONS = ["91fa7ce2ea9e43a69d723cc3bd08f306", "b18a7d6210ca466f9dd9ceb8e1675a58", "f47e3429f5de4a2f95ee3f0a82a3edd0"]
OBJ_PICKUPS = ["14116da06a65420db32b585466061d31", "1986239266ee44c788fd6effe4055704", "3b9c6eaf305342b48ebcb8295b2d2632",
               "3edae53030ea45269fc63817312af4eb", "40604d4eb4f3454685e66a77e447f23e", "7cf3226824a94b0d909a7ba0949bb5fe",
               "aa1ff58f50ba4286b4e0c987824ed571", "dade78dc96e34f1a8cbcf14dd47d84de"]
OBJ_PROPS = ["obj:205a7012c31742afb2b81d2cbfe0f2b2", "obj:7628a1b3f3924562b1e45301fb5398c4", "obj:c9b66b0949f247c4ae78338053650ee2",
             "obj:e5fccac2cc97468f90048944188b1b7a", "obj:4da7e80c3b6a42328d48dd811642db6e",
             "obj:9800ee3d88004696b4a80b0115fee693", "obj:05cb0187751541f195d8b7cf3c4736ef",
             "obj:4b50a06663fe44079503ca53f7eb399f", "obj:f394f4f7f446474fa65e3217923bdaaf",
             "obj:0fd3dd1da381468c86d3bbe50a4db27b", "obj:c0dcebcb0d4b47bd98df534334f4caf6",
             "obj:68c8d8c7715947d6a393feecc83c0553", "obj:ab550a0d20c147a08b9cf0ee97c50188",
             "obj:1148b82e71ba4d6e93f0ea5fc23dc13f", "obj:c9699558a44247b88f73b7690ed11b8b"]
GROUND_GREEN = ["grass_medium_01", "grass_medium_02", "nettle_plant", "weed_plant_02", "dandelion_01", "celandine_01",
                "fern_02"]


def _pick_season(rng):
    ks = list(SEASONS)
    p = np.array([SEASONS[k] for k in ks])
    return ks[int(rng.choice(len(ks), p=p / p.sum()))]


def _half(ctx, pkey, s=1.0) -> float:
    """Half the footprint diagonal of a library piece at scale ``s``: what an
    unclaimed placement (weeds, tufts) must keep from every footprint."""
    sx, sy, _ = np.asarray(ctx.lib.size(pkey)) * s
    return 0.5 * math.hypot(sx, sy)


def _cone(ctx, c, r, h, mat_, z0, segs=28, coll="structure", name="cone"):
    V = [(c[0] + r * math.cos(2 * math.pi * i / segs), c[1] + r * math.sin(2 * math.pi * i / segs), z0)
         for i in range(segs)]
    V.append((c[0], c[1], z0 + h))
    F = [(i, (i + 1) % segs, segs) for i in range(segs)] + [tuple(range(segs))[::-1]]
    return ctx.mesh(V, F, mat_, coll=coll, name=name, smooth=True)


class Farm:
    def __init__(self, ctx, season):
        self.ctx, self.season = ctx, season
        rng, lk = ctx.rng, ctx.looks
        self.rng, self.lk = rng, lk
        snow = season == "snow"
        self.white = lk.paint((0.78, 0.77, 0.73), rough=0.55, name="trim_white")
        self.dark = lk.paint((0.035, 0.03, 0.028), rough=0.9, name="interior_dark")
        self.glass = lk.glass(tint=(0.35, 0.38, 0.4))
        self.tyre = U.mat(ctx, ["acg:Rubber001", "acg:Rubber002", "acg:Rubber004"], 0.5, tint=(0.35, 0.35, 0.35),
                          bevel=0.02)
        self.hub = lk.paint((0.55, 0.5, 0.42), rough=0.5, metal=0.3, name="hub")
        self.metal = U.mat(ctx, ["acg:Metal032", "acg:Metal009", "ph:metal_plate", "ph:rusty_metal_02", "acg:Metal035"],
                           1.0, bevel=0.01, dirt=0.4)
        self.galv = U.mat(ctx, ["ph:corrugated_iron", "ph:corrugated_iron_02", "ph:worn_corrugated_iron"], 1.2,
                          bevel=0.01, dirt=0.3)
        self.water = lk.paint((0.02, 0.025, 0.02), rough=0.04, name="water")
        hay_tint = {"summer": (1.15, 1.0, 0.7), "dry": (1.2, 1.05, 0.72), "wet": (0.95, 0.85, 0.65),
                    "autumn": (1.0, 0.88, 0.65), "snow": (1.0, 0.9, 0.7)}[season]
        hk = U.pick(ctx, HAY)
        self.hay = lk.pbr(hk, 1.1, tint=hay_tint, bevel=0.04, dirt=0.2, jitter=0.12)
        self.hay_end = lk.pbr(U.pick(ctx, HAY), 0.5, tint=[c * 0.85 for c in hay_tint], jitter=0.1)
        wrap = [(0.7, 0.7, 0.68), (0.04, 0.04, 0.04), (0.1, 0.24, 0.1), (0.5, 0.55, 0.5)][int(rng.integers(4))]
        self.wrap = U.mat(ctx, ["acg:Plastic006", "acg:Plastic008", "acg:Plastic010"], 0.8, tint=[c * 1.2 for c in wrap],
                          jitter=0.05)
        self.timber = U.mat(ctx, PLANKS, 1.6, bevel=0.015, dirt=0.4)
        self.post = U.mat(ctx, ["ph:rough_wood", "ph:weathered_planks", "ph:wood_planks_grey", "ph:weathered_brown_planks",
                                "acg:Wood062", "acg:Wood049"], 1.0, bevel=0.012, dirt=0.5, jitter=0.12)
        self.bark = U.mat(ctx, BARK, 1.2, dirt=0.3)
        self.tree_end = U.mat(ctx, ["acg:TreeEnd001", "acg:TreeEnd002", "acg:TreeEnd003", "acg:TreeEnd004"], 0.35)
        self.concrete = U.mat(ctx, ["ph:concrete_floor_02", "ph:concrete_floor_worn_001", "acg:Concrete034",
                                    "ph:dirty_concrete", "ph:brushed_concrete"], 3.0, proj="flat")
        self.roof_snow = U.mat(ctx, ["acg:Snow004", "acg:Snow006", "ph:snow_02"], 3.0) if snow else None
        self.buildings = []

    # --- buildings ------------------------------------------------------------------------------
    def wall_family(self):
        rng = self.rng
        fam = str(rng.choice(["red", "weathered", "dark", "sheet", "painted"], p=[0.26, 0.26, 0.16, 0.22, 0.10]))
        keys, tile, tints = BARN_WALLS[fam]
        tint = tints[int(rng.integers(len(tints)))]
        return fam, self.lk.pbr(U.pick(self.ctx, keys), tile, tint=tint, bevel=0.02, dirt=0.5, jitter=0.05)

    def roof_mat(self, tiles=False):
        if self.roof_snow is not None:
            return self.roof_snow, 3.0
        if tiles:
            k = U.pick(self.ctx, ROOFS_TILE)
            return self.lk.pbr(k, None, proj="uv", dirt=0.3), float(self.ctx.lib.items[k].get("tile_m") or 2.0)
        k = U.pick(self.ctx, ROOFS_METAL)
        tint = [None, (0.85, 0.85, 0.85), (0.7, 0.35, 0.3), (0.3, 0.42, 0.3)][int(self.rng.integers(4))]
        return self.lk.pbr(k, None, proj="uv", tint=tint, dirt=0.3), float(self.ctx.lib.items[k].get("tile_m") or 1.6)

    def barn(self, o, d, L, W):
        ctx, rng = self.ctx, self.rng
        fam, wall = self.wall_family()
        kind = "gambrel" if (fam in ("red", "painted", "weathered") and rng.random() < 0.45) else "gable"
        roof, rt = self.roof_mat(tiles=(fam == "painted" and rng.random() < 0.5))
        trim = self.white if fam in ("red", "painted") or rng.random() < 0.3 else self.post
        h = float(rng.uniform(4.2, 5.4))
        b = U.Building(ctx, o, d, L, W, h, wall, roof, kind=kind, pitch=float(rng.uniform(28, 42)),
                       overhang=float(rng.uniform(0.25, 0.5)), roof_tile=rt, trim=trim)
        b.corner_trim(trim, 0.16)
        # sliding doors on the patch face (side -1), some open onto the dark interior
        nd = 1 if L < 20 else int(rng.integers(1, 3))
        us = np.linspace(-L / 2, L / 2, nd + 2)[1:-1] + rng.normal(0, 1.0, nd)
        dmat = wall if rng.random() < 0.5 else self.timber
        for u in us:
            w, hd = float(rng.uniform(3.0, 4.2)), float(min(h - 0.4, rng.uniform(3.2, 4.0)))
            if rng.random() < 0.45:                 # open: dark opening, the leaf slid aside
                b.door(self.dark, trim, side=-1, u=u, w=w, h=hd, depth=0.02)
                shift = w * (1 if rng.random() < 0.5 else -1)
                if abs(u + shift) + w / 2 < L / 2 - 0.2:
                    b.door(dmat, trim, side=-1, u=u + shift, w=w, h=hd, brace=trim, depth=0.09)
            else:
                b.door(dmat, trim, side=-1, u=u, w=w, h=hd, brace=trim, depth=0.06)
            # the track rail above the door
            U.boxes(ctx, [b.on_face(-1, u, 2 * w + 0.4, hd + 0.12, hd + 0.24, 0.16)], self.metal, name="rail")
        # man door + windows
        um = float(rng.uniform(-L / 2 + 1.5, L / 2 - 1.5))
        if all(abs(um - u) > 3.0 for u in us):
            b.door(self.timber, trim, side=-1, u=um, w=0.95, h=2.1)
        for u in np.arange(-L / 2 + 2.0, L / 2 - 1.5, float(rng.uniform(3.5, 5.5))):
            if all(abs(u - uu) > 2.8 for uu in us) and abs(u - um) > 1.4 and rng.random() < 0.7:
                b.window(self.glass, trim, side=-1, u=u, w=0.9, z0=h - 1.5, z1=h - 0.6, panes=(2, 2))
        # the gable ends: loft door, a hoist beam, a big end door
        for end in (-1, 1):
            if kind in ("gable", "gambrel") and rng.random() < 0.8:
                zl = h + 0.3
                U.boxes(ctx, [b.on_end(end, 0.0, 1.5, zl, zl + 1.6, 0.05)], dmat, name="loft")
                U.boxes(ctx, [b.on_end(end, 0.0, 1.7, zl + 1.6, zl + 1.7, 0.08),
                              b.on_end(end, -0.8, 0.1, zl, zl + 1.6, 0.08),
                              b.on_end(end, 0.8, 0.1, zl, zl + 1.6, 0.08)], trim, name="loft_frame")
                pe = b.W3(end * (L / 2), 0.0, b.ridge - 0.5)
                U.beams(ctx, [(pe, pe + np.r_[b.d * end * 1.0, 0])], 0.15, 0.2, self.post, name="hoist")
            if rng.random() < 0.6:
                b.door(dmat, trim, end=end, v=float(rng.uniform(-W / 4, W / 4)), w=3.4, h=min(h - 0.3, 3.6),
                       brace=trim, depth=0.06)
        self.buildings.append(b)
        return b

    def pole_shed(self, o, d, L, W):
        ctx, rng = self.ctx, self.rng
        k = U.pick(ctx, BARN_WALLS["sheet"][0])
        wall = self.lk.pbr(k, 1.3, tint=[None, (0.85, 0.85, 0.85)][int(rng.integers(2))], bevel=0.01, dirt=0.5)
        roof, rt = self.roof_mat()
        h = float(rng.uniform(3.9, 4.6))
        post = self.post if rng.random() < 0.5 else self.metal
        b = U.Building(ctx, o, d, L, W, h, wall, roof, kind="skillion", pitch=float(rng.uniform(6, 12)),
                       overhang=0.4, roof_tile=rt, open_front=True, post_mat=post)
        ctx.flat([b.W3(-L / 2, -W / 2 - 0.8, 0)[:2], b.W3(L / 2, -W / 2 - 0.8, 0)[:2], b.W3(L / 2, W / 2, 0)[:2],
                  b.W3(-L / 2, W / 2, 0)[:2]], self.concrete, z=0.005, name="shed_floor")
        # contents: bales stacked to the back, machines across a bay; everything stays inside the end walls
        u, u_end = -L / 2 + 0.6, L / 2 - 0.3
        tall = []                                              # (u0, u1, top) of the machines, for the lamps
        while u < u_end - 1.5:
            what = rng.random()
            if what < 0.45:                                    # round bale pyramid
                n0 = int(rng.integers(2, 4))
                r = 0.72
                for lvl in range(2):
                    for i in range(n0 - lvl):
                        cu = u + r + i * 2 * r * 1.01 + lvl * r
                        if cu + r > u_end:
                            continue
                        c = b.W3(cu, W / 2 - 0.9, r + lvl * r * 1.72)
                        a_, b_ = c - np.r_[b.n * 0.6, 0], c + np.r_[b.n * 0.6, 0]
                        U.cyl(ctx, a_, b_, r, self.hay, coll="structure", name="bale", segments=28)
                        U.cyl(ctx, a_ - np.r_[b.n * 0.012, 0], a_, r * 0.97, self.hay_end, coll="structure",
                                     name="bale_end", segments=28)
                u += n0 * 2 * r + 0.8
            elif what < 0.75:                                  # square bale stack to the eaves
                nx = min(int(rng.integers(2, 5)), int((u_end - u - 0.1) / 1.25))
                if nx < 1:
                    break
                c = b.W3(u + nx * 0.62, W / 2 - 1.5, 0)
                U.square_bale_stack(ctx, c[:2], b.yaw, nx, 2, int(rng.integers(3, 7)), (1.2, 0.5, 0.4), self.hay,
                                    coll="structure", claim=False, ragged=0.35)
                u += nx * 1.25 + 0.8
            else:                                              # a machine across the bay, nose out (or in)
                obj = [k for k in U.obj_models(ctx, OBJ_TRACTORS + OBJ_LOADERS + OBJ_WAGONS)
                       if min(ctx.lib.size(k)[:2]) < 3.4 and max(ctx.lib.size(k)[:2]) < W - 1.2]
                what = rng.random()
                k = ctx.lib.pick(rng, obj) if obj and what < 0.55 else None
                hu = 0.5 * min(ctx.lib.size(k)[:2]) if k else (1.15 if what < 0.8 else 1.1)   # half-width along u
                cu = u + 0.3 + hu
                if cu + hu > u_end:
                    break
                c = b.W3(cu, 0.0, 0)
                if k:
                    U.place_model(ctx, k, c[:2], b.yaw + math.pi / 2 * rng.choice([-1, 1]), coll="structure",
                                  check=False, claim=False)
                    top = float(ctx.lib.size(k)[2])
                elif what < 0.8:                              # drawbar out of the open front, body to the back
                    U.wagon(ctx, b.W3(cu, 0.5, 0)[:2], b.yaw - math.pi / 2, self.timber, self.metal, self.tyre,
                            self.hub, coll="structure")
                    top = 1.4
                else:
                    self.a_tractor(c[:2], b.yaw + math.pi / 2 * rng.choice([-1, 1]), coll="structure")
                    top = 2.7
                tall.append((cu - hu, cu + hu, top))
                u = cu + hu + 0.8
        if rng.random() < 0.7:
            lamp = U.have(ctx, ["ph:hanging_industrial_lamp", "ph:caged_hanging_light"])
            if lamp:
                for uu in np.linspace(-L / 3, L / 3, 2):
                    if any(u0 - 0.5 < uu < u1 + 0.5 and top > h - 1.6 for u0, u1, top in tall):
                        continue                               # not into the cab of a tall machine
                    p = b.W3(uu, -W / 2 + 1.2, 0)
                    ctx.instance(lamp[0] + "#0", p[:2], z=h - 1.4, yaw=b.yaw, coll="structure")
        self.buildings.append(b)
        return b

    def byre(self, o, d, L, W, stone=True):
        ctx, rng = self.ctx, self.rng
        wall = self.lk.pbr(U.pick(ctx, STONE if stone else BRICK), 2.2 if stone else 1.6, bevel=0.02, dirt=0.5)
        roof, rt = self.roof_mat(tiles=True)
        h = float(rng.uniform(3.0, 3.8))
        trim = self.post
        b = U.Building(ctx, o, d, L, W, h, wall, roof, kind="gable", pitch=float(rng.uniform(35, 45)), overhang=0.3,
                       roof_tile=rt)
        paint = [self.timber, self.lk.pbr(U.pick(ctx, ["acg:PaintedWood009A", "acg:PaintedWood003", "acg:PaintedWood008A",
                                                       "acg:PaintedWood007A"]), 1.0, bevel=0.01, dirt=0.4)][int(rng.integers(2))]
        for u in np.arange(-L / 2 + 1.5, L / 2 - 1.0, float(rng.uniform(2.6, 3.6))):
            if rng.random() < 0.45:
                # stable door: split in two leaves, the top one open (dark)
                b.door(paint, trim, side=-1, u=u, w=1.2, h=1.1, depth=0.06)
                U.boxes(ctx, [b.on_face(-1, u, 1.2, 1.1, 2.1, 0.02)], self.dark, name="door_open")
                U.boxes(ctx, [b.on_face(-1, u, 1.4, 2.1, 2.22, 0.1)], trim, name="lintel")
            elif rng.random() < 0.6:
                b.window(self.glass, paint, side=-1, u=u, w=0.7, z0=1.2, z1=1.9, panes=(2, 3), sill=True)
        self.buildings.append(b)
        return b

    def grain_bins(self, c, n, r, h, coll="structure", dirn=1):
        """``n`` corrugated bins in a row from ``c`` towards ``dirn`` (+1 / -1
        along x), an auger leaning up the first from the other side; the xy
        box of all of it is ``bins_extent``."""
        ctx = self.ctx
        for i in range(n):
            cc = np.asarray(c, float) + np.array([dirn * i * (2 * r + 1.2), 0.0])
            U.cyl(ctx, (cc[0], cc[1], 0), (cc[0], cc[1], h), r, self.galv, coll=coll, name="bin", segments=40)
            _cone(ctx, cc, r + 0.15, r * 0.55, self.roof_snow or self.galv, h, coll=coll, name="bin_roof")
            U.cyl(ctx, (cc[0], cc[1], h + r * 0.5), (cc[0], cc[1], h + r * 0.62), 0.35, self.metal, coll=coll,
                         name="bin_cap", segments=12)
        # auger leaning up the first bin
        p0 = np.asarray(c, float) + np.array([-dirn * (r + 4.0), -r * 0.3])
        U.tube(ctx, [np.r_[p0, 0.6], np.r_[np.asarray(c) + np.array([-dirn * r * 0.6, 0]), h + r * 0.4]], 0.14,
               self.metal, sides=10, coll=coll, name="auger")

    @staticmethod
    def bins_extent(c, n, r, dirn=1):
        """xy box (lo, hi) of ``grain_bins(c, n, r, .., dirn)``: bins, roofs, the auger."""
        xa, xb = c[0] - dirn * (r + 4.2), c[0] + dirn * ((n - 1) * (2 * r + 1.2) + r + 0.2)
        return np.array([min(xa, xb), c[1] - r - 0.2]), np.array([max(xa, xb), c[1] + r + 0.2])

    def farmhouse(self, o, d, coll="backdrop", L=None, W=None):
        ctx, rng = self.ctx, self.rng
        L = float(rng.uniform(11, 16)) if L is None else L
        W = float(rng.uniform(8, 10)) if W is None else W
        wall = self.lk.pbr(U.pick(ctx, PLASTER + BRICK + STONE), 2.0, bevel=0.02, dirt=0.35)
        roof, rt = self.roof_mat(tiles=True)
        b = U.Building(ctx, o, d, L, W, 5.6, wall, roof, kind="gable", pitch=float(rng.uniform(35, 48)), overhang=0.45,
                       roof_tile=rt, coll=coll, trim=self.white)
        shut = self.lk.pbr(U.pick(ctx, ["acg:PaintedWood009A", "acg:PaintedWood008A", "acg:PaintedWood003",
                                        "acg:PaintedWood005"]), 1.0) if rng.random() < 0.5 else None
        for side in (-1, 1):
            for u in np.linspace(-L / 2 + 1.6, L / 2 - 1.6, 4):
                for z0 in (1.0, 3.6):
                    b.window(self.glass, self.white, side=side, u=u, w=1.0, z0=z0, z1=z0 + 1.3, panes=(2, 2), sill=True,
                             shutters=shut)
        b.door(self.timber, self.white, side=-1, u=0.0, w=1.1, h=2.2)
        ch = b.W3(L / 4, 0.0, 0)
        ctx.box((ch[0], ch[1], b.ridge + 0.2), (0.35, 0.5, 1.2), wall, coll=coll, name="chimney")

    def a_tractor(self, xy, yaw, coll="scatter"):
        col = [(0.12, 0.35, 0.1), (0.6, 0.06, 0.04), (0.08, 0.2, 0.5), (0.8, 0.35, 0.05),
               (0.55, 0.55, 0.52)][int(self.rng.integers(5))]
        k = U.pick(self.ctx, ["acg:PaintedMetal012", "acg:PaintedMetal010"])
        paint = self.lk.pbr(k, 1.0, tint=[min(1.0, 1.25 * c) for c in col], bevel=0.03, dirt=0.6, jitter=0.03) if k \
            else self.lk.paint(col, rough=0.35, name="tractor_paint")
        rim = paint if self.rng.random() < 0.5 else self.lk.paint((0.8, 0.6, 0.08), rough=0.4, name="rim_yellow")
        U.tractor(self.ctx, xy, yaw, paint, self.lk.paint((0.04, 0.04, 0.04), rough=0.5, name="black"), self.tyre,
                  rim, self.glass, coll=coll)


def _dress(ctx, farm, season):
    """The patch obstacles as farm structure, then their frames (caps, posts, ropes)."""
    rng, lk = ctx.rng, ctx.looks
    fam = str(rng.choice(["timber", "stone", "block", "sheet", "siding"], p=[0.3, 0.25, 0.12, 0.15, 0.18]))
    if fam == "timber":
        wall = U.mat(ctx, ["ph:weathered_planks", "ph:brown_planks_03", "ph:wood_plank_wall", "acg:Planks008",
                           "ph:weathered_brown_planks", "ph:medieval_wood"], 2.0, bevel=0.02, dirt=0.5)
    elif fam == "stone":
        wall = U.mat(ctx, STONE, 2.2, bevel=0.03, dirt=0.2)
    elif fam == "block":
        wall = U.mat(ctx, BLOCK, 1.8, bevel=0.02, dirt=0.5)
    elif fam == "sheet":
        wall = U.mat(ctx, BARN_WALLS["sheet"][0], 1.3, bevel=0.01, dirt=0.5)
    else:
        wall = farm.wall_family()[1]
    box = farm.timber if rng.random() < 0.7 else farm.hay
    if rng.random() < 0.55:
        pole = farm.bark
        pole_kind = "trunk"
    else:
        pole = U.mat(ctx, ["ph:rough_wood", "ph:wood_planks_grey", "acg:Wood062", "ph:weathered_planks"], 1.2,
                     tint=(0.55, 0.45, 0.38), bevel=0.01, dirt=0.4)
        pole_kind = "post"
    bar = farm.post if rng.random() < 0.6 else farm.galv
    rock = U.mat(ctx, ROCK, 1.2, dirt=0.2, jitter=0.1)
    ctx.dress_primitives({"wall": wall, "box": box, "pole": pole, "bar": bar, "rock": rock})
    U.claim_openings(ctx, float(ctx.knobs.get("keepout", 2.5)))
    for i, p, objs in U.prim_list(ctx):                   # a floating box is a beam or a board, never hay
        if p["kind"] == "box" and p["c"][2] - p["half"][2] > 0.3 and box is not farm.timber:
            for o in objs:
                o.data.materials[0] = farm.timber
    cap = U.mat(ctx, ["ph:concrete_floor_02", "ph:rough_wood"] if fam != "timber" else PLANKS, 1.0, bevel=0.01, dirt=0.4)
    changed = _frames(ctx, farm, cap, farm.post, lk.paint((0.1, 0.08, 0.06), rough=0.8, name="rope"), pole_kind,
                      lamp=lk.paint((0.9, 0.85, 0.7), rough=0.2, name="lamp_glass", emission=(1.0, 0.85, 0.6, 2.0)),
                      thick=(farm.galv, farm.roof_snow or farm.galv))
    return {"walls": fam, "poles": pole_kind, "semantics_changed": changed}


def _frames(ctx, farm, cap_mat, post_mat, rope_mat, pole_kind, lamp=None, thick=None) -> list[str]:
    """Caps on walls, posts under floating bars/boxes, cords on floating
    spheres, pollard heads or caps on poles. ``thick`` = (material, roof
    material or None): poles wider than 0.6 m radius get that material and a
    cone roof (a grain bin) or a flat coping instead.

    These add solid geometry to the patch's own obstacles (the tier-A
    family then differs from its primitives); returns what was added, for
    the course's ``semantics_changed`` record."""
    rng = ctx.rng
    caps, posts = [], []
    changed = set()
    avoid = [ctx.start, ctx.goal]

    def near(p, r=1.7):
        return any(np.linalg.norm(np.asarray(p)[:2] - np.asarray(q)[:2]) < r for q in avoid)

    for i, p, objs in U.prim_list(ctx):
        k = p["kind"]
        if k == "wall":
            for c, h, R in BC.wall_pieces(p):
                if abs((c[2] + h[2]) - p["z1"]) < 1e-3:
                    caps.append((c + np.array([0, 0, h[2] + 0.04]), (h[0] + 0.02, h[1] + 0.05, 0.04), R))
        elif k == "cyl_h":
            a, b = np.asarray(p["a"], float), np.asarray(p["b"], float)
            if min(a[2], b[2]) - p["r"] > 0.4:
                for e in (a, b):
                    if not near(e):
                        posts.append(((e[0], e[1], (e[2] + p["r"]) / 2), (0.08, 0.08, (e[2] + p["r"]) / 2), 0.0))
                        changed.add("posts under floating bars (0.16 m, at the bar ends)")
        elif k == "box":
            c, h = np.asarray(p["c"], float), np.asarray(p["half"], float)
            rpy = p.get("rpy", [0, 0, 0])
            zb = c[2] - h[2]
            if zb > 0.4 and abs(rpy[0]) < 0.05 and abs(rpy[1]) < 0.05:
                R = BC.rpy_matrix(rpy)
                ax = 0 if h[0] >= h[1] else 1
                for s in (-1, 1):
                    off = np.zeros(3)
                    off[ax] = s * (h[ax] - 0.1)
                    e = c + R @ off
                    if not near(e):
                        posts.append(((e[0], e[1], zb / 2), (0.1, 0.1, zb / 2), float(rpy[2])))
                        changed.add("posts under floating boxes (0.2 m, 0.1 m in from the long-axis ends)")
        elif k == "sphere":
            c, r = np.asarray(p["c"], float), p["r"]
            if c[2] - r > 0.3:
                U.tube(ctx, [c + np.array([0, 0, r * 0.9]), c + np.array([0, 0, 5.5])], 0.012, rope_mat, sides=6,
                       name="rope")
                changed.add("ropes from floating spheres up to 5.5 m (blocks passing over them)")
                if lamp is not None and r < 0.45 and rng.random() < 0.5:
                    for o in objs:
                        o.data.materials[0] = lamp
        elif k == "cyl_v" and thick is not None and p["r"] > 0.6:
            for o in objs:
                o.data.materials[0] = thick[0]
            cx, cy = p["c"]
            if thick[1] is not None:
                _cone(ctx, (cx, cy), p["r"] + 0.12, p["r"] * 0.5, thick[1], p["z1"], coll="structure", name="bin_roof")
                changed.add("cone roofs on poles r > 0.6 m (+0.12 m radius, +0.5 r height)")
            else:
                U.cyl(ctx, (cx, cy, p["z1"]), (cx, cy, p["z1"] + 0.12), p["r"] + 0.06, cap_mat, coll="structure",
                      name="coping", segments=32)
                changed.add("copings on poles r > 0.6 m (+0.06 m radius, +0.12 m)")
        elif k == "cyl_v" and pole_kind == "trunk" and p["z1"] > 4.0:
            # pollarded trunk: knuckles and stubby limbs at the top (mostly above the cloud band)
            cx, cy = p["c"]
            z1, r = p["z1"], p["r"]
            ctx.sphere((cx, cy, z1), r * 1.15, farm.bark, coll="structure", name="knuckle")
            changed.add("pollard knuckles + limbs on trunk tops above 4 m")
            for j in range(int(rng.integers(3, 6))):
                a = rng.uniform(0, 2 * math.pi)
                L = rng.uniform(0.5, 1.2)
                d = np.array([math.cos(a) * 0.6, math.sin(a) * 0.6, 0.8])
                d /= np.linalg.norm(d)
                p0 = np.array([cx, cy, z1 - 0.1])
                U.cyl(ctx, p0, p0 + d * (r + L), r * rng.uniform(0.3, 0.5), farm.bark, coll="structure",
                             name="limb", segments=10)
        elif k == "cyl_v" and pole_kind == "post":
            cx, cy = p["c"]
            U.cyl(ctx, (cx, cy, p["z1"]), (cx, cy, p["z1"] + 0.05), p["r"] + 0.03, cap_mat, coll="structure",
                         name="pcap", segments=20)
            changed.add("caps on posts (+0.03 m radius, +0.05 m)")
    if caps:
        U.boxes(ctx, caps, cap_mat, name="wallcap")
        changed.add("caps on walls (+0.05 m each side, +0.08 m)")
    if posts:
        U.boxes(ctx, posts, post_mat, name="prim_post")
    return sorted(changed)


def _end_scenery(ctx, farm, x_edge, dirn, y_fence, side):
    """Scenery past one end of the yard (``x_edge`` = that end of the zone,
    ``dirn`` = outward along x): a farmhouse, grain bins, a barn, silos or
    nothing. It goes wholly >= 1 m outside the cloud zone (render-only), on
    the yard side of the fence line ``y_fence``, clear of every claimed
    footprint (the building row), moving outward when it does not fit, and is
    claimed (``scenery``) so the fence, the tree rows and the hanging lines
    keep off it. Returns what was built, None if nothing (drawn or fitted)."""
    rng = ctx.rng
    zy0, zy1 = ctx.zone[1], ctx.zone[3]
    what = str(rng.choice(["house", "bins", "barn", "silo", "none"], p=[0.3, 0.2, 0.2, 0.15, 0.15]))
    base = x_edge + dirn * float(rng.uniform(9, 22))

    def centred(dx, half):
        return lambda xe, ye: (np.array([xe + dirn * dx, ye]) - half, np.array([xe + dirn * dx, ye]) + half)

    if what == "house":
        L, W = float(rng.uniform(11, 16)), float(rng.uniform(8, 10))
        d = (0.0, 1.0) if rng.random() < 0.5 else (1.0, 0.0)
        box = centred(6.0, np.array([W, L] if d[1] else [L, W]) / 2 + 0.8)     # + eaves, sills, shutters
    elif what == "bins":
        n, r, h = int(rng.integers(2, 5)), float(rng.uniform(2.5, 3.5)), float(rng.uniform(6, 9))
        box = lambda xe, ye: Farm.bins_extent((xe, ye), n, r, dirn)
    elif what == "barn":
        L, W, h = float(rng.uniform(18, 30)), float(rng.uniform(10, 14)), float(rng.uniform(4.5, 6))
        box = centred(8.0, np.array([W / 2 + 0.6, L / 2 + 0.6]))
    elif what == "silo":
        hs = [float(rng.uniform(14, 20)) for _ in range(int(rng.integers(1, 3)))]
        box = lambda xe, ye: (np.array([xe + dirn * 4 - 3.0, ye - 3.0]),
                              np.array([xe + dirn * 4 + 3.0, ye + (len(hs) - 1) * 6.5 + 3.0]))
    else:
        return None
    for t in range(15):
        xe, ye = base + dirn * 6.0 * (t // 5), float(rng.uniform(zy0, zy1))
        lo, hi = box(xe, ye)
        if ctx.rect_meets_zone(lo - 1.0, hi + 1.0):
            continue
        if (lo[1] < y_fence + 1.0) if side > 0 else (hi[1] > y_fence - 1.0):
            continue                                          # across (or beyond) the paddock fence
        if U.box_free(ctx, lo, hi, 1.0):
            break
    else:
        return None
    if what == "house":
        farm.farmhouse((xe + dirn * 6, ye), d, L=L, W=W)
    elif what == "bins":
        farm.grain_bins((xe, ye), n, r, h, coll="backdrop", dirn=dirn)
    elif what == "barn":
        fam, wall = farm.wall_family()
        roof, rt = farm.roof_mat()
        U.Building(ctx, (xe + dirn * 8, ye), (0.0, 1.0), L, W, h, wall, roof, kind=str(rng.choice(["gable", "gambrel"])),
                   coll="backdrop", roof_tile=rt, trim=farm.white)
    else:
        conc = U.mat(ctx, ["ph:concrete_wall_008", "ph:concrete_wall_006", "ph:preconcrete_wall_001"], 3.0, dirt=0.4)
        for j, hj in enumerate(hs):
            cc = (xe + dirn * 4, ye + j * 6.5)
            U.cyl(ctx, (cc[0], cc[1], 0), (cc[0], cc[1], hj), 2.8, conc, coll="backdrop", name="silo", segments=40)
            ctx.sphere((cc[0], cc[1], hj), 2.8, farm.galv, coll="backdrop", name="silo_dome")
    ctx.claim_rect((lo + hi) / 2, (hi - lo) / 2, 0.0, kind="scenery")
    return what


def build(ctx) -> dict:
    rng, lk = ctx.rng, ctx.looks
    x0, y0, x1, y1 = ctx.bounds
    zx0, zy0, zx1, zy1 = ctx.zone
    season = _pick_season(rng)
    sky = lk.sky(U.pick_hdri(ctx, HDRIS[season], ("farm", "field")), target_ev=float(rng.uniform(0.2, 0.6)))
    farm = Farm(ctx, season)

    # --- ground ---------------------------------------------------------------------------------
    g_dirt, g_grass, g_third = GROUND[season]
    keys = [U.pick(ctx, g_dirt), U.pick(ctx, g_grass), U.pick(ctx, g_third)]
    if rng.random() < 0.5:
        keys = [keys[1], keys[0], keys[2]]
    wet = {"wet": float(rng.uniform(0.5, 1.0)), "summer": float(rng.choice([0.0, 0.0, 0.25])),
           "autumn": float(rng.choice([0.0, 0.3, 0.5]))}.get(season, 0.0)
    ctx.ground(lk.ground_material(keys, tile_m=float(rng.uniform(2.2, 3.2)), puddles=wet,
                                  blend_scale=float(rng.uniform(6.0, 12.0))))

    side = int(rng.choice([-1, 1]))                       # the building row is on this side
    yb_edge = y1 if side > 0 else y0
    yo_edge = y0 if side > 0 else y1

    # farm track along the building row, running on beyond the zone (two ruts, grass crown)
    track_key = U.pick(ctx, TRACK[season])
    track = lk.pbr(track_key, None, proj="uv", vary=0.2, jitter=0.0)
    ttile = float(ctx.lib.items[track_key].get("tile_m") or 2.5)
    y_track = yb_edge - side * float(rng.uniform(0.5, 2.5))
    tw = float(rng.uniform(3.2, 4.4))
    if track_key in ("ph:snow_04", "ph:muddy_tracks"):      # one pair of ruts per strip
        ttile = tw * float(rng.uniform(1.0, 1.15))
    U.jagged_strip(ctx, (zx0 - 80, y_track), (zx1 + 80, y_track), tw, track, name="track", tile=ttile)
    if season != "snow" and track_key != "ph:muddy_tracks" and rng.random() < 0.7:
        crown = lk.pbr(U.pick(ctx, g_grass), 2.5, proj="flat", vary=0.2, jitter=0.0)
        U.jagged_strip(ctx, (zx0 - 80, y_track), (zx1 + 80, y_track), float(rng.uniform(0.5, 0.9)), crown, z=0.006,
                       jit=0.1, name="crown")

    info = {"season": season, "sky": sky, "ground": keys, "track": track_key, "wet": wet, "side": side}
    info.update(_dress(ctx, farm, season))

    # --- the building row ---------------------------------------------------------------------------
    d = np.array([float(side), 0.0])       # local n = d rotated +90 -> v = -W/2 faces the patch
    x = zx0 - float(rng.uniform(4, 14))
    built = []
    while x < zx1 + 10:
        kind = str(rng.choice(["barn", "shed", "byre", "bins", "gap"], p=[0.36, 0.26, 0.14, 0.1, 0.14]))
        if kind == "barn":
            L, W = float(rng.uniform(16, 26)), float(rng.uniform(10, 14))
        elif kind == "shed":
            L, W = float(rng.uniform(12, 20)), float(rng.uniform(7, 9.5))
        elif kind == "byre":
            L, W = float(rng.uniform(10, 16)), float(rng.uniform(6.5, 8.5))
        elif kind == "bins":
            n, r = int(rng.integers(2, 4)), float(rng.uniform(2.3, 3.2))
            L, W = n * (2 * r + 1.2) + 3.6, 2 * r + 0.6                  # + the auger's foot
        else:
            L, W = float(rng.uniform(6, 12)), 4.0
        placed = False
        for g in (float(rng.uniform(1.3, 2.6)), 3.2, 4.4):
            yc = yb_edge + side * (g + W / 2)
            c = np.array([x + L / 2, yc])
            if ctx.free_rect(c, (L / 2, W / 2), 0.0, margin=0.3, step=0.6):
                placed = True
                break
        if placed and kind != "gap":
            bo = None
            if kind == "barn":
                bo = farm.barn(c, d, L, W)
            elif kind == "shed":
                bo = farm.pole_shed(c, d, L, W)
            elif kind == "byre":
                bo = farm.byre(c, d, L, W, stone=rng.random() < 0.7)
            else:
                farm.grain_bins((x + 3.6 + r + 0.6, yc), n, r, float(rng.uniform(5.5, 8.0)))
            ctx.claim_rect(c, (L / 2 + 0.3, W / 2 + 0.3), 0.0, kind="building")
            if bo is not None:
                bo.foot = ctx.foot[-1]                      # a service drop from its eave skips its own footprint
            built.append(kind)
            # concrete apron in front of barns
            if kind in ("barn", "byre") and rng.random() < 0.6:
                ye = yb_edge + side * g
                ctx.flat([(x + 1, ye), (x + L - 1, ye), (x + L - 1, ye - side * 2.2), (x + 1, ye - side * 2.2)],
                         farm.concrete, z=0.005, name="apron")
        elif kind == "gap" or not placed:
            # between buildings: a bale stack against the row or a log pile
            gx = x + L / 2
            yy = yb_edge + side * float(rng.uniform(2.0, 3.5))
            if ctx.free_rect((gx, yy), (2.6, 1.4), 0.0, margin=0.3):
                if rng.random() < 0.6:
                    U.square_bale_stack(ctx, (gx, yy), 0.0, int(rng.integers(2, 5)), 2, int(rng.integers(2, 5)),
                                        (1.2, 0.5, 0.4), farm.hay, coll="structure", ragged=0.4)
                else:
                    U.log_pile(ctx, (gx, yy), 0.0, farm.bark, farm.tree_end, L=float(rng.uniform(2.0, 3.2)),
                               rows=int(rng.integers(2, 4)), coll="structure")
                    ctx.claim_rect((gx, yy), (1.8, 1.0), 0.0, kind="logs")
        x += L + float(rng.uniform(2.5, 7.0))
    info["buildings"] = built

    # --- scenery past the two ends (before the fence and the tree rows, which keep off it) --------------
    y_f = yo_edge - side * float(rng.uniform(0.8, 2.4))          # the paddock fence line
    info["ends"] = [_end_scenery(ctx, farm, zx1, 1, y_f, side), _end_scenery(ctx, farm, zx0, -1, y_f, side)]

    # --- paddock fence along the other side, with a gate ---------------------------------------------
    fkind = str(rng.choice(["post_rail", "post_rail", "wire", "board"]))
    fm = {"post": farm.post, "rail": farm.post if rng.random() < 0.5 else farm.timber, "rails": int(rng.integers(2, 5)),
          "wire": lk.paint((0.35, 0.33, 0.3), rough=0.4, metal=0.8, name="wire"), "strands": int(rng.integers(3, 5)),
          "board": farm.timber, "board_w": 0.15, "board_gap": 0.05}
    gx0 = float(rng.uniform(x0 + 2, x1 - 6))
    U.fence_line(ctx, (zx0 - 60, y_f), (zx1 + 60, y_f), float(rng.uniform(2.2, 3.0)), fkind, fm,
                 h=float(rng.uniform(1.1, 1.45)), gaps=[(gx0 - (zx0 - 60), gx0 - (zx0 - 60) + 4.0)], jitter=0.015)
    gate_mat = farm.galv if rng.random() < 0.6 else lk.paint((0.1, 0.25, 0.12), rough=0.5, name="gate_green")
    if ctx.free_rect((gx0 + 2, y_f), (2.0, 0.15), 0.0, margin=0.2):
        if rng.random() < 0.5:
            U.farm_gate(ctx, (gx0, y_f), (gx0 + 3.9, y_f), gate_mat)
        else:                                            # swung open into the field
            U.farm_gate(ctx, (gx0, y_f), (gx0 + 0.4, y_f - side * 3.8), gate_mat)
        ctx.claim_rect((gx0 + 2, y_f), (2.1, 0.3), 0.0, kind="gate")

    # the fields beyond the fence (flat), and one beyond the building row
    fields = {"summer": ["acg:Grass003", "acg:Grass001", "ph:farm_soil", "acg:Ground037", "ph:sparse_grass", "acg:Grass004"],
              "dry": ["acg:Ground072", "acg:Ground102", "ph:farm_soil", "acg:Ground073", "acg:Ground103"],
              "wet": ["ph:brown_mud_02", "acg:Grass002", "acg:Ground047", "ph:farm_soil"],
              "autumn": ["ph:farm_soil", "acg:Grass003", "acg:Ground013", "ph:brown_mud_leaves_01"],
              "snow": ["ph:snow_03", "ph:snow_field_aerial", "acg:Snow007A", "ph:snow_01"]}[season]
    fx = zx0 - 150
    while fx < zx1 + 150:
        fl = float(rng.uniform(40, 90))
        fmat = lk.pbr(U.pick(ctx, fields), float(rng.uniform(2.0, 3.5)), proj="flat", vary=0.25, jitter=0.0)
        yf1 = y_f - side * 0.3
        ctx.flat([(fx, yf1), (fx + fl, yf1), (fx + fl, yf1 - side * 160), (fx, yf1 - side * 160)], fmat, z=0.002,
                 name="field")
        fx += fl + 0.5

    # --- trees: a few inside the zone margin, tree lines as scenery -------------------------------------
    files = [f for f in TREE_SETS if f in ctx.lib.items]
    rng.shuffle(files)
    files = files[:2]
    tkeys = [k for f in files for k in TREE_SETS[f][0]]
    tscale = TREE_SETS[files[0]][1] if files else (1.0, 1.3)
    far_y = y_f - side * float(rng.uniform(35, 80))
    avoid = ("building", "scenery")                          # no trunk inside a barn, the farmhouse or a bin
    U.tree_row(ctx, tkeys, (zx0 - 160, far_y), (zx1 + 160, far_y), float(rng.uniform(6, 11)), s=tscale, skip=0.15,
               avoid=avoid)
    back_y = yb_edge + side * float(rng.uniform(28, 45))
    U.tree_row(ctx, tkeys, (zx0 - 120, back_y), (zx1 + 120, back_y), float(rng.uniform(7, 13)), s=tscale, skip=0.3,
               avoid=avoid)
    for xe in (zx0 - float(rng.uniform(25, 60)), zx1 + float(rng.uniform(25, 60))):
        U.tree_row(ctx, tkeys, (xe, zy0 - 60), (xe, zy1 + 60), float(rng.uniform(7, 12)), s=tscale, skip=0.35,
                   avoid=avoid)
    # one or two trees in the margin on the fence side (in the cloud: keep few)
    near_keys = [k for k in tkeys if k.split("#")[0] in ("ph:island_tree_01", "ph:tree_small_02", "ph:island_tree_02",
                                                         "ph:searsia_burchellii")]
    n_in = 0
    for _ in range(40):
        if n_in >= int(rng.integers(0, 3)) or not near_keys:
            break
        k = near_keys[int(rng.integers(len(near_keys)))]
        s = float(rng.uniform(0.9, 1.2))
        sx, sy, _ = np.asarray(ctx.lib.size(k)) * s
        rr = 0.5 * max(sx, sy)
        xy = np.array([rng.uniform(zx0 + 2, zx1 - 2), y_f - side * float(rng.uniform(1.0, 4.0))])
        if ctx.free(xy, rr, 0.3):
            ctx.instance(k, xy, 0.0, float(rng.uniform(0, 6.28)), s, coll="structure")
            ctx.claim_disc(xy, rr, kind="tree")
            n_in += 1

    # --- pole line along the fence side (wires overhead, above the flight band) -------------------------
    poles = []
    y_p = y_f - side * float(rng.uniform(0.8, 2.0))
    xp = zx0 - 60 + float(rng.uniform(0, 40))
    while xp < zx1 + 70:
        xy = np.array([xp, y_p])
        if not ctx.in_zone(xy, 0.5) or ctx.free(xy, 0.2, 0.5):
            poles.append(U.utility_pole(ctx, xy, farm.post, farm.metal, h=float(rng.uniform(8.0, 9.5)),
                                        coll="structure" if ctx.in_zone(xy, 1) else "backdrop"))
            if ctx.in_zone(xy, 1):
                ctx.claim_disc(xy, 0.3, kind="pole")
            xp += float(rng.uniform(35, 48))
        else:
            xp += 3.0
    wmat = lk.paint((0.08, 0.08, 0.08), rough=0.5, name="cable")
    for pa, pb in zip(poles[:-1], poles[1:]):
        for a_, b_ in zip(pa, pb):
            U.wire(ctx, a_, b_, float(rng.uniform(0.5, 0.9)), 0.008, wmat, coll="backdrop", name="line")

    # --- the yard: hay, a machine, troughs, clutter ---------------------------------------------------
    ko = float(ctx.knobs.get("keepout", 2.5))
    n_bales = ctx.count(float(rng.uniform(2.0, 4.0)))
    wrapped = rng.random() < 0.35
    for _ in range(n_bales):
        for _t in range(60):
            xy = ctx.uniform_xy(pad=1.0)
            nrow = int(rng.choice([1, 1, 2, 3]))
            yaw = float(rng.choice([0.0, math.pi / 2])) + rng.normal(0, 0.15)
            dvec = np.array([math.cos(yaw + math.pi / 2), math.sin(yaw + math.pi / 2)])
            pts = [xy + dvec * 1.6 * (i - (nrow - 1) / 2) for i in range(nrow)]
            if all(ctx.free(p, 1.0, 0.3 + 0.3 * ko) for p in pts):
                for p in pts:
                    side_m = farm.wrap if wrapped else farm.hay
                    end_m = farm.wrap if wrapped else farm.hay_end
                    U.round_bale(ctx, p, yaw + rng.normal(0, 0.08), side_m, end_m, standing=rng.random() < 0.2)
                break
    for _ in range(ctx.count(float(rng.uniform(0.5, 1.5)))):
        for _t in range(60):
            xy = ctx.uniform_xy(pad=1.5)
            nx, ny = int(rng.integers(2, 4)), int(rng.integers(1, 3))
            half = (nx * 0.62 + 0.1, ny * 0.26 + 0.1)
            yaw = float(rng.choice([0.0, math.pi / 2]))
            if ctx.free_rect(xy, half, yaw, margin=0.3 + 0.3 * ko):
                U.square_bale_stack(ctx, xy, yaw, nx, ny, int(rng.integers(1, 4)), (1.2, 0.5, 0.4), farm.hay)
                break
    # a machine in the yard (big): a real tractor / loader / wagon / pickup, or a built wagon or tractor
    machines = {"tractor": U.obj_models(ctx, OBJ_TRACTORS), "loader": U.obj_models(ctx, OBJ_LOADERS),
                "wagon": U.obj_models(ctx, OBJ_WAGONS), "pickup": U.obj_models(ctx, OBJ_PICKUPS)}
    if rng.random() < 0.3 + 0.3 * float(ctx.knobs.get("clutter", 1.0)):
        for _t in range(80):
            xy = ctx.uniform_xy(pad=2.5)
            yaw = float(rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2])) + rng.normal(0, 0.1)
            what = str(rng.choice(["tractor", "loader", "wagon", "pickup", "built_wagon", "built_tractor"],
                                  p=[0.34, 0.08, 0.12, 0.14, 0.16, 0.16]))
            if machines.get(what):
                if U.place_model(ctx, ctx.lib.pick(rng, machines[what]), xy, yaw, margin=0.4 + 0.3 * ko):
                    info["machine"] = what
                    break
                continue
            if ctx.free_rect(xy, (3.0, 1.4), yaw, margin=0.4 + 0.3 * ko):
                if what in ("built_wagon", "wagon"):          # the kit's own (also without the Objaverse catalog)
                    U.wagon(ctx, xy, yaw, farm.timber, farm.metal, farm.tyre, farm.hub)
                    info["machine"] = "built_wagon"
                else:
                    farm.a_tractor(xy, yaw)
                    info["machine"] = "built_tractor"
                ctx.claim_rect(xy, (3.0, 1.4), yaw, kind="machine")
                break
    # a pickup or a tractor parked along the track by the buildings (clutter-scaled)
    parked = machines["pickup"] + machines["tractor"]
    for _ in range(int(rng.random() < 0.35 * float(ctx.knobs.get("clutter", 1.0))) if parked else 0):
        for _t in range(30):
            xy = np.array([rng.uniform(x0 + 2, x1 - 2), y_track + side * float(rng.uniform(0.0, 1.5))])
            if U.place_model(ctx, ctx.lib.pick(rng, parked), xy, float(rng.choice([0.0, math.pi])) + rng.normal(0, 0.05),
                             margin=0.4 + 0.3 * ko):
                info["parked"] = True
                break
    # troughs
    for _ in range(ctx.count(float(rng.uniform(0.4, 1.2)))):
        for _t in range(40):
            xy = ctx.uniform_xy(pad=1.2)
            if rng.random() < 0.5:
                yaw = float(rng.choice([0.0, math.pi / 2]))
                if ctx.free_rect(xy, (1.3, 0.45), yaw, margin=0.4):
                    U.water_trough(ctx, xy, yaw, farm.galv, farm.water)
                    ctx.claim_rect(xy, (1.3, 0.45), yaw, kind="trough")
                    break
            elif ctx.free(xy, 1.05, 0.4):
                U.stock_tank(ctx, xy, farm.galv, farm.water, r=float(rng.uniform(0.8, 1.2)))
                ctx.claim_disc(xy, 1.2, kind="tank")
                break
    # small clutter: barrels, crates, cans, tyres, pallets, log piles
    props = ctx.lib.models(ids=YARD_PROPS + OBJ_PROPS, h=(0.1, 2.0))
    ctx.scatter(props, ctx.count(float(rng.uniform(4, 8))), margin=0.4)
    for _ in range(ctx.count(1.0)):
        xy = ctx.uniform_xy(pad=1.0)
        yaw = float(rng.uniform(0, math.pi))
        if ctx.free_rect(xy, (0.7, 0.5), yaw, margin=0.4):
            U.pallet_stack(ctx, xy, yaw, farm.timber, n=int(rng.integers(1, 6)))
            ctx.claim_rect(xy, (0.7, 0.5), yaw, kind="pallet")
    if rng.random() < 0.5:
        xy = ctx.uniform_xy(pad=2.0)
        yaw = float(rng.uniform(0, math.pi))
        if ctx.free_rect(xy, (1.6, 0.8), yaw, margin=0.4):
            U.log_pile(ctx, xy, yaw, farm.bark, farm.tree_end, L=float(rng.uniform(2.0, 3.0)), rows=int(rng.integers(2, 4)))
            ctx.claim_rect(xy, (1.6, 0.8), yaw, kind="logs")
    # boulders at the yard edges, weeds along the fence and the building fronts
    rocks = ctx.lib.models(ids=["rock_moss_set_02", "namaqualand_boulder_05", "rock_moss_set_01"], h=(0.3, 1.4))
    ctx.scatter(rocks, ctx.count(1.0), margin=0.5)
    if season != "snow":
        weeds = ctx.lib.models(ids=GROUND_GREEN, h=(0.05, 0.6))
        for yline in (y_f + side * 0.3, yb_edge + side * 1.0):
            for xw in np.arange(zx0, zx1, float(rng.uniform(0.6, 1.4))):
                xy = np.array([xw + rng.normal(0, 0.3), yline + rng.normal(0, 0.35)])
                k = ctx.lib.pick(rng, weeds)
                s = float(rng.uniform(0.9, 1.6))
                if k and ctx.clearance(xy) > 0.1 + _half(ctx, k, s):
                    ctx.instance(k, xy, 0.0, float(rng.uniform(0, 6.28)), s, coll="scatter")

        # tufts and stones scattered over the yard: 3-D detail on the ground near the camera
        tufts = ctx.lib.models(ids=["grass_medium_01", "grass_medium_02"], h=(0.08, 0.5))
        stones = ctx.lib.models(ids=["namaqualand_stones_01", "rock_07", "namaqualand_rocks_01"], h=(0.02, 0.3))
        for _ in range(ctx.count(float(rng.uniform(60, 140)))):
            xy = np.array([rng.uniform(zx0, zx1), rng.uniform(zy0, zy1)])
            k = ctx.lib.pick(rng, stones if rng.random() < 0.25 else tufts)
            s = float(rng.uniform(1.2, 2.4))
            if not k or abs(xy[1] - y_track) < tw / 2 - 0.3 or ctx.clearance(xy) < 0.1 + _half(ctx, k, s):
                continue
            ctx.instance(k, xy, 0.0, float(rng.uniform(0, 6.28)), s, coll="scatter")

    # --- hanging obstacles in the flight band (hard levels) --------------------------------------------
    hang = float(ctx.knobs.get("hang", 0.0))
    n_hang = ctx.count(hang) if hang > 0 else 0
    rope = lk.paint((0.08, 0.075, 0.07), rough=0.6, name="cable_black")
    tape = lk.paint([(0.85, 0.85, 0.8), (0.9, 0.45, 0.05), (0.8, 0.8, 0.15)][int(rng.integers(3))], rough=0.5,
                    name="tape")
    made = 0
    in_poles = [p for p in poles if ctx.in_zone(p[1][:2], 2.0)]
    # eaves a service drop can come to: row buildings whose patch face runs in the zone
    drop_bs = [b for b in farm.buildings if getattr(b, "foot", None) is not None
               and b.o[0] + b.L / 2 - 1 > zx0 and b.o[0] - b.L / 2 + 1 < zx1]
    for _ in range(n_hang * 30):
        if made >= n_hang:
            break
        what = rng.random()
        if what < 0.35 and drop_bs:
            # service drop: pole (or a post by the fence) -> an eave facing the patch, sagging into the band;
            # both ends and the low point in the zone, nothing tall under it but its own building's face
            b = drop_bs[int(rng.integers(len(drop_bs)))]
            xs = float(rng.uniform(max(b.o[0] - b.L / 2 + 1, zx0), min(b.o[0] + b.L / 2 - 1, zx1)))
            pe = b.W3((xs - b.o[0]) * b.d[0], -b.W / 2 - 0.1, b.h - 0.3)
            on_pole = bool(in_poles) and rng.random() < 0.5
            if on_pole:
                pa = in_poles[int(rng.integers(len(in_poles)))][1] + np.array([0, 0, -1.2])
            else:
                pa = np.array([pe[0] + rng.normal(0, 6), y_f + side * 0.3, float(rng.uniform(4.5, 5.5))])
            sag = min(float(rng.uniform(1.0, 2.0)), max(0.3, min(pa[2], pe[2]) - 2.0))
            P = U.catenary(pa, pe, sag, 18)
            if not (abs(pa[0] - pe[0]) <= 20 and ctx.in_zone(pa[:2], 2.0 if on_pole else 0.0)
                    and ctx.in_zone(P[9][:2]) and U.seg_clear(ctx, pa, pe, 2.2)
                    and (on_pole or U.foot_clearance(ctx, pa, U.TALL_KINDS + ("gate",)) >= 0.4)
                    and U.line_free(ctx, P, 0.4, skip=(b.foot,), trim=(0.6, 0.2))):
                continue
            if not on_pole:                                 # its own post by the fence
                U.cyl(ctx, (pa[0], pa[1], 0), (pa[0], pa[1], pa[2] + 0.2), 0.1, farm.post, coll="structure",
                             name="cable_post", segments=12)
                ctx.claim_disc(pa, 0.2, kind="pole")
            U.wire(ctx, pa, pe, sag, 0.012, rope, name="service_drop", n=18)
            U.claim_line(ctx, pa, pe)
            made += 1
        elif what < 0.75:
            # electric-fence tape between light posts
            a = ctx.uniform_xy(pad=1.0)
            ang = float(rng.uniform(0, math.pi))
            L = float(rng.uniform(4, 9))
            b = a + L * np.array([math.cos(ang), math.sin(ang)])
            if not (ctx.free(a, 0.05, 0.3) and ctx.free(b, 0.05, 0.3) and U.seg_clear(ctx, a, b, 2.0)
                    and U.seg_free(ctx, a, b, 0.3, kinds=None)):
                continue
            if not ctx.in_zone(b):
                continue
            nposts = max(2, int(L / 4) + 1)
            for t in np.linspace(0, 1, nposts):
                p = a + (b - a) * t
                U.cyl(ctx, (p[0], p[1], 0), (p[0], p[1], 1.35), 0.012, farm.white, coll="scatter", name="epost",
                             segments=8)
            for z in (0.8, 1.2):
                U.boxes(ctx, [U.beam_item(np.r_[a, z], np.r_[b, z - 0.02], 0.004, 0.035)], tape, name="etape")
            U.claim_line(ctx, a, b, 0.1)
            made += 1
        else:
            # a lamp on a rope between two posts / a crossing rope line with a hanging lamp
            a = ctx.uniform_xy(pad=0.5)
            b = a + rng.normal(0, 1, 2) * 4.0
            z = float(rng.uniform(2.6, 3.4))
            m = (a + b) / 2
            if not (ctx.free(a, 0.1, 0.5) and ctx.free(b, 0.1, 0.5) and U.seg_clear(ctx, a, b, 2.2)
                    and U.line_free(ctx, U.catenary(np.r_[a, z + 0.2], np.r_[b, z + 0.2], 0.3), 0.4, trim=0.1)
                    and U.line_free(ctx, [np.r_[m, z - 0.1], np.r_[m, z - 0.7]], 0.4)):
                continue
            for p in (a, b):
                U.cyl(ctx, (p[0], p[1], 0), (p[0], p[1], z + 0.3), 0.07, farm.post, coll="scatter", name="lpost",
                             segments=10)
            U.wire(ctx, np.r_[a, z + 0.2], np.r_[b, z + 0.2], 0.3, 0.01, rope, name="lamp_rope")
            U.claim_line(ctx, a, b)
            ctx.sphere((m[0], m[1], z - 0.55), 0.14, lk.paint((0.9, 0.85, 0.7), rough=0.2, name="lamp_glass",
                                                               emission=(1.0, 0.85, 0.6, 2.0)), name="lamp")
            U.tube(ctx, [np.r_[m, z - 0.42], np.r_[m, z - 0.1]], 0.008, rope, sides=5, name="lamp_cord")
            made += 1
    info["hanging"] = made
    return info
