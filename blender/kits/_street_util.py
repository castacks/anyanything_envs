"""Shared by the ``street`` and ``alley`` kits: city buildings, street
furniture and the helpers that make them read as real.

* ``Acc`` accumulates quads and boxes per material and emits one mesh per
  material (a building is five objects, not five hundred);
* ``Frame`` is a facade's local frame: ``u`` along the frontage, ``v`` out
  of the wall (towards the street), ``z`` up;
* ``Building`` builds a procedural building on a frame -- a closed volume
  whose front has real window openings (reveals, frames, glass, sills),
  shopfronts with fascia signs (real text), roller shutters, doors,
  cornices, balconies, downpipes -- in scan materials picked from curated
  families (brick, plaster, painted plaster, stone, concrete);
* ``modular_building`` lays Poly Haven's modular factory / apartment
  facade kits (3 m modules) on a frame, recoloured per building;
* furniture: lamp posts, bollards, bins, benches, bike stands, traffic
  signs, cones, dumpsters, pallets, cables (sagging catenaries).

Everything goes through ``ctx`` (so the cloud zone decides what collides)
and every placement inside the patch checks ``ctx`` free space.
"""

from __future__ import annotations

import math

import bpy
import numpy as np

import build_course as BC

# --- curated material families (keys checked against the library at use) ---------------------

BRICK_RED = ["acg:Bricks051", "acg:Bricks054", "acg:Bricks059", "acg:Bricks071", "acg:Bricks074", "acg:Bricks082A",
             "acg:Bricks085", "acg:Bricks086", "acg:Bricks090", "acg:Bricks092", "acg:Bricks094", "acg:Bricks097",
             "acg:Bricks101", "acg:Bricks104", "ph:red_brick_03", "ph:factory_brick", "ph:large_red_bricks",
             "ph:brick_wall_001", "ph:brick_wall_02", "ph:brown_brick_02", "ph:brick_wall_006"]
BRICK_DARK = ["acg:Bricks056", "acg:Bricks058", "acg:Bricks061", "ph:dark_brick_wall", "ph:brick_wall_08"]
BRICK_LIGHT = ["acg:Bricks053", "acg:Bricks060", "acg:Bricks093", "acg:Bricks105", "ph:castle_brick_02_white",
               "ph:painted_brick"]
STONE = ["acg:Bricks019", "acg:Bricks038", "acg:Bricks064", "acg:Bricks066", "acg:Bricks068", "acg:Bricks084",
         "ph:sandstone_blocks_04", "ph:sandstone_blocks_08", "ph:white_sandstone_bricks_03", "ph:medieval_blocks_03"]
PLASTER_WHITE = ["acg:Plaster001", "acg:Plaster002", "acg:Plaster003", "ph:white_stucco", "ph:white_plaster_02",
                 "ph:plastered_wall", "ph:plastered_wall_02", "ph:beige_wall_001", "ph:grey_plaster"]
PLASTER_WORN = ["ph:worn_plaster_wall", "ph:worn_mossy_plasterwall", "ph:worn_cracked_plaster", "ph:damaged_plaster",
                "ph:red_plaster_weathered", "ph:blue_plaster_weathered", "ph:yellow_plaster_02",
                "acg:PaintedPlaster007", "acg:PaintedPlaster008", "acg:PaintedPlaster009", "acg:PaintedPlaster010",
                "acg:PaintedPlaster013", "acg:PaintedPlaster014", "acg:PaintedPlaster015", "acg:PaintedPlaster016",
                "acg:PaintedPlaster018"]
CONCRETE_WALL = ["ph:concrete_wall_001", "ph:concrete_wall_003", "ph:concrete_wall_007", "ph:concrete_wall_008",
                 "ph:concrete_block_wall", "ph:concrete_block_wall_02", "ph:concrete_slab_wall",
                 "acg:Concrete034", "acg:Concrete031", "acg:Concrete022"]
TILE_FACADE = ["ph:concrete_tile_facade", "ph:rectangular_facade_tiles", "ph:rectangular_facade_tiles_02"]
TRIM = ["acg:Concrete017", "acg:Concrete019", "acg:Concrete033", "ph:concrete", "ph:brushed_concrete",
        "acg:Granite001A", "acg:Granite002A", "acg:Travertine004", "acg:Travertine009", "ph:plaster_grey_04"]
ROOF_FLAT = ["ph:bitumen", "ph:roof_07", "acg:Concrete020", "ph:gravel_concrete"]
ROOF_TILES = ["ph:clay_roof_tiles", "ph:clay_roof_tiles_02", "ph:grey_roof_tiles", "ph:roof_slates_02",
              "ph:red_slate_roof_tiles_01", "acg:RoofingTiles006", "acg:RoofingTiles011A", "acg:RoofingTiles013A"]
SHUTTER = ["ph:rusted_shutter", "ph:painted_metal_shutter", "ph:worn_shutter", "ph:rusty_metal_shutter"]
CORRUGATED = ["ph:corrugated_iron", "ph:corrugated_iron_02", "ph:rusty_corrugated_iron", "ph:worn_corrugated_iron",
              "acg:CorrugatedSteel001", "acg:CorrugatedSteel002", "acg:CorrugatedSteel005"]
#: light, lightly worn paint scans -- the base for tinted paint (poles, bins, frames)
PAINTED_METAL = ["acg:PaintedMetal010", "acg:PaintedMetal012", "acg:PaintedMetal013", "acg:PaintedMetal003"]
#: rusty, chipped paint (dumpsters, alley doors)
RUSTY_PAINT = ["acg:PaintedMetal006", "acg:PaintedMetal007", "acg:PaintedMetal008", "acg:PaintedMetal009",
               "acg:PaintedMetal011", "acg:PaintedMetal014"]
#: bare grey steel (Metal007 is brass, Metal034 yellow, SheetMetal004 black)
GALV = ["acg:Metal009", "acg:Metal032", "acg:SheetMetal002"]
PLANKS_PAINTED = ["ph:blue_painted_planks", "ph:distressed_painted_planks", "ph:black_painted_planks",
                  "acg:PaintedWood005", "acg:PaintedWood006A", "acg:PaintedWood007A", "acg:PaintedWood009A",
                  "ph:plywood"]
WOOD = ["acg:Wood049", "acg:Wood051", "acg:Planks012", "acg:Planks021", "acg:Planks023A", "ph:brown_planks_05"]
ASPHALT = ["acg:Asphalt004", "acg:Asphalt010", "acg:Asphalt012", "acg:Asphalt015", "acg:Asphalt018",
           "acg:Asphalt020S", "acg:Asphalt026B", "acg:Asphalt027A", "acg:Asphalt031",   # (029A is red: pink blotches)
           "acg:Road012A", "acg:Road012B", "acg:Road013A", "acg:Road014A", "acg:Road015A",
           "ph:asphalt_01", "ph:asphalt_02", "ph:asphalt_03", "ph:asphalt_04", "ph:asphalt_06", "ph:asphalt_07",
           "ph:road_damaged", "ph:clean_asphalt"]
ASPHALT_PATCHED = ["acg:Road013B", "acg:Road013C", "acg:Road015B", "acg:Road015C", "acg:Road014B",
                   "acg:Road008C", "acg:Road011C"]
SIDEWALK = ["acg:PavingStones029", "acg:PavingStones035", "acg:PavingStones037", "acg:PavingStones038",
            "acg:PavingStones071", "acg:PavingStones077", "acg:PavingStones081", "acg:PavingStones098",
            "acg:PavingStones099", "acg:PavingStones105", "acg:PavingStones108", "acg:PavingStones109",
            "acg:PavingStones111", "acg:PavingStones112", "acg:PavingStones117", "acg:PavingStones131",
            "acg:PavingStones132", "ph:concrete_pavement", "ph:concrete_tiles", "ph:concrete_tiles_02",
            "ph:square_floor", "ph:patio_tiles", "ph:brick_pavement", "ph:herringbone_pavement_02"]
COBBLE = ["ph:cobblestone_floor_001", "ph:cobblestone_floor_05", "ph:cobblestone_floor_08", "ph:cobblestone_05",
          "ph:cobblestone_large_01", "acg:PavingStones046", "acg:PavingStones070", "acg:PavingStones116",
          "acg:PavingStones119", "acg:PavingStones051"]
PLAZA = ["acg:PavingStones107", "acg:PavingStones118", "acg:PavingStones121", "acg:PavingStones125A",
         "acg:PavingStones126A", "acg:PavingStones128", "acg:PavingStones133", "ph:granite_tile_03",
         "ph:grey_cartago_01", "ph:large_square_pattern_01", "ph:checkered_pavement_tiles"] + COBBLE[:4]
KERB = ["acg:Granite001A", "acg:Granite002A", "acg:Granite005A", "acg:Concrete017", "acg:Concrete019",
        "ph:concrete"]
CONCRETE_GROUND = ["ph:concrete_floor_worn_001", "ph:dirty_concrete", "ph:concrete_floor_damaged_01",
                   "ph:damaged_concrete_floor", "ph:concrete_floor_02", "acg:Concrete023", "acg:Concrete036"]

#: facade paint colours (multiplied onto white/light scans)
PASTELS = [(0.96, 0.90, 0.78), (0.93, 0.84, 0.70), (0.98, 0.95, 0.88), (0.86, 0.90, 0.92), (0.92, 0.80, 0.72),
           (0.95, 0.88, 0.62), (0.80, 0.86, 0.78), (0.94, 0.74, 0.62), (0.85, 0.83, 0.80), (0.99, 0.97, 0.93),
           (0.78, 0.66, 0.56), (0.90, 0.72, 0.50), (0.72, 0.76, 0.80), (0.88, 0.62, 0.52)]
FRAME_COLOURS = [(0.92, 0.92, 0.90), (0.92, 0.92, 0.90), (0.08, 0.08, 0.08), (0.20, 0.21, 0.22),
                 (0.30, 0.20, 0.12), (0.12, 0.22, 0.16), (0.55, 0.52, 0.48), (0.15, 0.18, 0.28)]
SIGN_COLOURS = [(0.05, 0.05, 0.05), (0.55, 0.05, 0.05), (0.04, 0.12, 0.30), (0.05, 0.22, 0.12),
                (0.85, 0.85, 0.82), (0.45, 0.30, 0.10), (0.70, 0.45, 0.05), (0.25, 0.25, 0.27),
                (0.50, 0.08, 0.25), (0.06, 0.30, 0.32)]
SHOP_NAMES = ["BAKERY", "PHARMACY", "CAFE", "PIZZERIA", "BAR", "HAIR SALON", "BOOKS", "FLOWERS", "OPTICIAN",
              "LAUNDRY", "DELI", "KEBAB", "SUSHI", "BUTCHER", "MINI MARKET", "HARDWARE", "BANK", "SHOES",
              "TATTOO", "BIKES", "GALLERY", "NAILS", "PHONE REPAIR", "WINE & SPIRITS", "BISTRO", "NOODLE BAR",
              "BARBER", "DENTIST", "ESTATE AGENTS", "BURGERS", "GROCERY", "BACKEREI", "APOTHEKE",
              "BOULANGERIE", "FARMACIA", "TABACCHI", "PANADERIA", "CAFE ROMA", "ANNA'S", "DRY CLEANING",
              "LIQUOR", "PAWN SHOP", "THRIFT STORE", "RECORDS", "TOYS", "PET SHOP", "BEAUTY", "TAILOR",
              "LOCKSMITH", "COFFEE", "TEA HOUSE", "RAMEN", "DONER", "FISH & CHIPS", "OFFICE TO LET",
              "24/7", "BET", "TRAVEL", "OPTIK", "CHEMIST", "NEWSAGENT", "POST OFFICE", "HOTEL", "HOSTEL"]

# --- curated Objaverse models ---------------------------------------------------------------------
# ``obj:<uid>`` files picked from catalog_objaverse.json by their thumbnails (2026-09-27): whole,
# textured, metre scale, CC-BY-4.0, glb <= 15 MB (its textures are packed into the course .blend on
# import), a piece under obj_pieces' 150 k-triangle cap; each course's licences are in course.json "files".
# Blender's glTF import turns a model's front to -y, the piece convention of every kit.

OBJ_CARS = ["4f45aa2e9180437b9687b3d567069ca9", "300e85f5786f48b3abd5f8ab3851c3a1", "f03e5ba5e81b4562958645fe27565f3a",
            "5ce1a4963463439892a5fd523486797a", "e398ad20e83d42e581ab43438d5b6e49", "e5d5f35870344bd4a005ef5599a63313",
            "13cc3ffc69964894a2bc94154aed687f", "c592bf582a1b430daaef753418fab22f", "178f22467bae4c729bdcc15dbc7e445d",
            "192011ef37994e8ca2f09cd1d73ca166", "52f13d6fd3b84e7e88f127cd15f537e2", "bec38f586d9349ee9c1f295b6f5569cb",
            "93fc33be25e34b44b205c5b962d66e61", "2429a825ff1646568e8e6e1453f9c501", "e0bb09118ab84fe0a0530bd0df2fd74a",
            "cb6e82e73386448fb3b4e2dd25255a5e", "3a19f19092614c8cb36d1420288abe21", "bf1fe23e1578417ab11db67dc89fbe32",
            "b8d6c9706f0746628cdf184d268c0e56", "f4eafb342ecb438ba3bff34dbd77cb46", "f809616e219a406a8c2feaf3a00b9fc9",
            "4b36740173d24f73aadda6cb10e7cacc"]
OBJ_VANS = ["7cf3226824a94b0d909a7ba0949bb5fe", "dade78dc96e34f1a8cbcf14dd47d84de", "4258fa5bb1f74a2da7ed2090df0cb55b",
            "f69de1315bb049c8946d57f6006acd73", "ae025c380da7453a8912bec06d3557bb"]
OBJ_TWO_WHEEL = ["e559f6bccecf44a2b73e1da8ffce6b66", "197dd882a69846e0825e80566996cbba",
                 "3bde1ab67c6448169b604dedad6828d8", "f256203558c24ea09385a861b917c75a",
                 "b43c4f717e904d938dcacfe58c345631", "43860f8a9f174cb9b9cc15a3f6786fbf",
                 "99d6dce2e49e4ba185ac108ab044524d", "5e3ba0954ffc4a8f853c86cb7fdaa91b",
                 "f623ab40922944f881caf38d057e5d9d", "d8d0022e70d048b4a87f7248655f73bf",
                 "b4141e61ada9458f9d6c2acaebea6ee8", "b0d453e026b24d8496b14a2fa046f457",
                 "a1b186f04d4b4654ab81b0d829007312", "693e83d86e1e4e5b95e4314dbdd95d40",
                 "d5db61785c4043a094908fe9deedcda5", "7d446ec9135c4e35892714d117b37268",
                 "c02dec85ac0541b8ab6dd56098706b50", "91b031e0cc1b438b8c94bcb6e87aa381"]
OBJ_PHONE = ["e2493ece3b30446086602b8c006d2604", "baf1ea30e7bb4790854619485d62b022",
             "a0339aa975e54bb7ac65982ca3cc5802", "66ab8d583ca24a0fa20f6096f66396b9",
             "8ab0dd710f0f4372bf53a2bd944a783f", "10d9ec53b8a140c7a7785ff54a7ecc22"]
OBJ_MAIL = ["2173ac25461e485aa834f8210bc16e4f", "1d0fe843aee943119c839f747acc7434", "85e4d11083e744efaffb4411307efb7b"]
OBJ_METER = ["a93161d5bafb4d83a73c766aadb97416", "9c91b51de37343d3a63e49e18371e862",
             "a1932b57521c4aa1b0148d1e6ad6b7aa", "df3f36c368e441bd979e0c149ff295a6"]
OBJ_VENDING = ["94c002a46f7f4ac498b294e69e4e62cf", "82cf245dee4b424785bcff3abae609b2",
               "4bfb493631f84f8a9adc00bb69831000", "110e72b7baa04396863b955dde50b954",
               "efb393f5a3bb4bbfbe1bf5c13abb372d", "54127d454edd4e3f982a46f58dd6c816"]
OBJ_BUS_STOP = ["90fbd043c83a423598d425bb3a641dfc", "367bdb5ca8c745978410c6e3ffd9b910"]
OBJ_HYDRANT = ["a3cf6b9b70a042fa85474c5a5f025e6e", "91acc6e821a74ed3a3c0ae72643958a5",
               "d012d05e8bca49928a2944e043d93fb7", "0a83180e31b640a39a83b0da113b00ce",
               "a4ef32d993734d37a8985c36927d4201", "71f34fffd6bd4f1bbb2278fceba686c8",
               "4c11f50141b44f34ba3755850d6e804f", "51c0f9b3d1a7499a8af9558b97f2ba51",
               "dc9cd29a598343d38eae7ba41d311615"]
OBJ_BIN = ["baee1591d1c44f74adf376fb5b6d2edb", "3057e7f5b6c441fcb67c44e68d77dd38", "1b3d199a690f4a58aa39f45c9c040c77",
           "0debf0ba50304951b32b44c7578468dd", "0f1b5d0f8f1d49548580d0722552c0d1", "9044ac8d545747e3959278fbe36fce8b"]
OBJ_TRAFFIC_LIGHT = ["1e120134a77944929519d8f96a5f3998", "525722b75cca448ea57df110a9d5df84",
                     "c351112891524961bf3288f11f5effbf"]
OBJ_BOLLARD = ["cb3dfbc282e74ac5bb0c83e1a6df37b9"]
OBJ_STREET_TREE = ["ef48485b7c534acc905c0d9964c5fe7c"]          # a plane tree, 10.6 m (placed at 0.65-0.8)
OBJ_CROWN = ["cb91cb6149a142a8a196a268dcf36aa8"]                # a 1.3 m leafy bush, 6 k tris: a pollarded crown x2-3
OBJ_DUMPSTER = ["73f80ae78964472bbda19328b354fa35", "135d33df60304481830d6ddd8c43cc47",
                "e8b38df736994cf8b4f7a3a9d2980f6b", "bed7cc2208ae481f800a1d8516839be5"]
OBJ_GAS = ["e788acc021ac45a598178d24ce196737", "4da7e80c3b6a42328d48dd811642db6e", "a68f90323fd44cf7a22c0afe30d6a2f8",
           "f1075a710e554e21b9b678a053e1f9ea"]
OBJ_GENERATOR = ["423cfdbd612c486aada7bb1bc44695a9", "f34ba6c159e04152ac1883caaca2ae18",
                 "f394f4f7f446474fa65e3217923bdaaf", "a6c7477cc905448fa16310be52a989c1",
                 "4b50a06663fe44079503ca53f7eb399f", "05cb0187751541f195d8b7cf3c4736ef"]
OBJ_BARREL = ["c9b66b0949f247c4ae78338053650ee2", "205a7012c31742afb2b81d2cbfe0f2b2",
              "e5fccac2cc97468f90048944188b1b7a"]
OBJ_FORKLIFT = ["b7bea04856e1401898250f313131659f", "d659c366e24643f0b72ad0fe1ef1a60b",
                "060f3f8bc7de4e6ca2f348d414702e9d", "ce8b2c714b2f472bbac58c6eab088ba2",
                "9ad80a7bd90b483f8dbf0e293251cff7"]
OBJ_TRUCK = ["27a0aaa34e7d4ea7877393178d82b01c", "f0b7ca0f694c4213b799677e49041896",
             "56f462ac530840999e3f17a0163fbcac", "95b04d8e016a49d9b487d2c4b9a56819"]
OBJ_LOADER = ["d23c7e5565104a1db2e1c4641d4ed59b", "df9c3cfa98bb40598cfa7b96027db2a5"]


def draw_count(ctx, per_450: float, cap: int | None = None) -> int:
    """``ctx.count`` without the rounding: the whole part of the expected
    count plus one more with the fraction's probability -- ``ctx.count``
    rounds a rate under 0.5 per patch (a bus shelter, a phone booth, a truck
    on a tier-A patch) to none every time."""
    x = per_450 * ctx.area_f * float(ctx.knobs.get("clutter", 1.0))
    n = int(x) + int(ctx.rng.random() < x - int(x))
    return n if cap is None else min(n, cap)


def obj_pieces(ctx, uids, max_tris: int = 150_000) -> list[str]:
    """Piece keys (``obj:<uid>#i``) of the curated Objaverse files the library
    has, one per file (its largest piece), skipping excluded or heavy ones."""
    out = []
    for u in uids:
        fk = f"obj:{u}"
        e = ctx.lib.items.get(fk)
        if e is None or e.get("exclude"):
            continue
        ps = [p for p in ctx.lib.pieces(fk) if not p.get("exclude") and p.get("tris", 0) <= max_tris]
        if ps:
            p = max(ps, key=lambda q: q["size"][0] * q["size"][1] * q["size"][2])
            out.append(f"{fk}#{p['index']}")
    return out


def long_yaw(ctx, pkey, s: float = 1.0):
    """(yaw turning the piece's long side onto +x, length, width) -- for vehicles."""
    sx, sy, _ = (np.asarray(ctx.lib.size(pkey)) * s).tolist()
    return (0.0, sx, sy) if sx >= sy else (math.pi / 2, sy, sx)


def pick_key(ctx, keys, fallback_cats=()):
    ok = [k for k in keys if k in ctx.lib.items]
    if not ok and fallback_cats:
        ok = ctx.lib.materials(cats_any=set(fallback_cats))
    return ctx.lib.pick(ctx.rng, ok)


def scan(ctx, keys, tile, **kw):
    """A pbr material from one of ``keys`` (the key it chose is in ``m['scan']``)."""
    k = pick_key(ctx, keys)
    m = ctx.looks.pbr(k, tile, **kw)
    m["scan"] = k
    return m


def worn_paint(ctx, rgb, name="line", wear=0.35, rough=0.55):
    """Road paint that is worn through in patches (alpha from noise): only
    for non-colliding flat overlays."""
    ck = ("worn_paint", tuple(rgb), wear)
    if ck in ctx.looks.cache:
        return ctx.looks.cache[ck]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    gc = N.new("ShaderNodeNewGeometry")
    nz = N.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 3.0
    nz.inputs["Detail"].default_value = 8.0
    nz.inputs["Roughness"].default_value = 0.7
    L.new(gc.outputs["Position"], nz.inputs["Vector"])
    mr = N.new("ShaderNodeMapRange")
    lo = 0.25 + 0.4 * wear
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = lo, lo + 0.08
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 1.0, 0.0
    L.new(nz.outputs["Fac"], mr.inputs["Value"])
    L.new(mr.outputs["Result"], b.inputs["Alpha"])
    ctx.looks.cache[ck] = m
    return m


def tint_material(m, rgb, sat=1.0, val=1.0, name=None):
    """Copy of an imported material with its base colour desaturated/lifted
    (``sat``, ``val``) and multiplied by ``rgb``."""
    m2 = m.copy()
    if name:
        m2.name = name
    if not m2.use_nodes:
        return m2
    nt = m2.node_tree
    b = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if b is None:
        return m2
    inp = b.inputs["Base Color"]
    hs = nt.nodes.new("ShaderNodeHueSaturation")
    hs.inputs["Saturation"].default_value = sat
    hs.inputs["Value"].default_value = val
    if inp.is_linked:
        nt.links.new(inp.links[0].from_socket, hs.inputs["Color"])
    else:
        hs.inputs["Color"].default_value = inp.default_value
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    nt.links.new(hs.outputs["Color"], mix.inputs[6])
    mix.inputs[7].default_value = (*rgb, 1.0)
    nt.links.new(mix.outputs[2], inp)
    return m2


def piece_variant(ctx, pkey: str, tag: str, fn):
    """A copy of library piece ``pkey``'s collection whose materials are
    mapped by ``fn(material) -> material or None (keep)``; cached per
    (pkey, tag). Place with ``place_variant``."""
    cache = ctx.__dict__.setdefault("_pv_cache", {})
    if (pkey, tag) in cache:
        return cache[(pkey, tag)]
    p = ctx.piece(pkey)
    src = p["coll"]
    dst = bpy.data.collections.new(f"{src.name}_{tag}")
    ctx.lib_coll.children.link(dst)
    mats = ctx.__dict__.setdefault("_pv_mats", {})
    objmap = {}
    for o in src.objects:
        o2 = o.copy()
        if o.type == "MESH" and o.data is not None:
            o2.data = o.data.copy()
            for slot in o2.material_slots:
                m = slot.material
                if m is None:
                    continue
                mk = (m.name, tag)
                if mk not in mats:
                    mats[mk] = fn(m) or m
                slot.material = mats[mk]
        dst.objects.link(o2)
        objmap[o] = o2
    for o, o2 in objmap.items():
        if o.parent in objmap:
            o2.parent = objmap[o.parent]
    dst.instance_offset = src.instance_offset
    cache[(pkey, tag)] = dst
    return dst


def place_variant(ctx, pkey, coll_variant, xy, z=0.0, yaw=0.0, s=1.0, coll="scatter", collide=None, claim=False):
    e = ctx.instance(pkey, xy, z=z, yaw=yaw, s=s, coll=coll, collide=collide, claim=claim)
    if coll_variant is not None:
        e.instance_collection = coll_variant
    return e


def text_obj(ctx, s: str, xyz, yaw: float, size: float, mat, depth=0.008, coll="structure", align="CENTER",
             bold=0.0, spacing=1.0):
    """Upright extruded text facing local -y rotated by ``yaw`` (the piece
    convention), centred on ``xyz``. Render-only (letters are millimetres)."""
    cu = bpy.data.curves.new("sign_text", "FONT")
    cu.body = s
    cu.size = size
    cu.extrude = depth
    cu.align_x, cu.align_y = align, "CENTER"
    cu.offset = bold
    cu.space_character = spacing
    try:
        cu.font = bpy.data.fonts.load("<builtin>", check_existing=True)
    except Exception:  # noqa: BLE001
        pass
    ob = bpy.data.objects.new("sign_text", cu)
    ob.data.materials.append(mat)
    ob.location = tuple(map(float, xyz))
    ob.rotation_euler = (math.pi / 2, 0.0, float(yaw))
    ob["collide"] = 0
    ctx._coll(coll).objects.link(ob)
    return ob


# --- geometry accumulation ------------------------------------------------------------------------

class Acc:
    """Quads and boxes per material, emitted as one mesh per material."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.parts: dict = {}

    def _p(self, mat):
        k = mat.name
        if k not in self.parts:
            self.parts[k] = [mat, [], []]
        return self.parts[k]

    def poly(self, mat, pts, normal=None):
        pts = [np.asarray(p, float) for p in pts]
        if normal is not None and len(pts) >= 3:
            nrm = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            if np.dot(nrm, normal) < 0:
                pts = pts[::-1]
        _, V, F = self._p(mat)
        n = len(V)
        V.extend(tuple(float(c) for c in p) for p in pts)
        F.append(tuple(range(n, n + len(pts))))

    def box(self, mat, c, half, yaw=0.0):
        cy, sy = math.cos(yaw), math.sin(yaw)
        R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
        c = np.asarray(c, float)
        hx, hy, hz = half
        corners = [c + R @ np.array([sx * hx, sy_ * hy, sz * hz]) for sz in (-1, 1) for sy_ in (-1, 1)
                   for sx in (-1, 1)]
        # indices: 0(-,-,-) 1(+,-,-) 2(-,+,-) 3(+,+,-) 4(-,-,+) 5(+,-,+) 6(-,+,+) 7(+,+,+)
        faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (1, 3, 7, 5), (3, 2, 6, 7), (2, 0, 4, 6)]
        _, V, F = self._p(mat)
        n = len(V)
        V.extend(tuple(float(v) for v in p) for p in corners)
        F.extend(tuple(n + i for i in f) for f in faces)

    def emit(self, name="bld", coll="structure", collide=None, roof_mats=(), roof_all=False):
        out = []
        for k, (m, V, F) in self.parts.items():
            if F:
                out.append(self.ctx.mesh(V, F, m, coll=coll, collide=collide, name=name,
                                         roof=roof_all or k in roof_mats))
        self.parts = {}
        return out


class Frame:
    """A frontage: origin ``o`` on the facade line, outward normal ``n``
    (towards the street), ``t`` along the facade (t x z = n)."""

    def __init__(self, o, n):
        self.o = np.asarray(o, float)[:2]
        n = np.asarray(n, float)[:2]
        self.n = n / np.linalg.norm(n)
        self.t = np.array([-self.n[1], self.n[0]])
        self.yaw = math.atan2(self.t[1], self.t[0])
        self.N3 = np.array([self.n[0], self.n[1], 0.0])
        self.T3 = np.array([self.t[0], self.t[1], 0.0])

    def xy(self, u, v=0.0):
        return self.o + self.t * u + self.n * v

    def p(self, u, v, z):
        q = self.xy(u, v)
        return np.array([q[0], q[1], z])

    def quad(self, acc, mat, ua, ub, za, zb, v=0.0):
        """A quad in the plane v, facing +n."""
        acc.poly(mat, [self.p(ua, v, za), self.p(ub, v, za), self.p(ub, v, zb), self.p(ua, v, zb)], normal=self.N3)

    def box(self, acc, mat, ua, ub, va, vb, za, zb):
        c = self.p((ua + ub) / 2, (va + vb) / 2, (za + zb) / 2)
        acc.box(mat, c, (abs(ub - ua) / 2, abs(vb - va) / 2, abs(zb - za) / 2), yaw=self.yaw)


def wall_with_holes(fr: Frame, acc: Acc, mat, u0, u1, z0, z1, holes, v=0.0):
    """The plane v of the facade from (u0, z0) to (u1, z1) minus the
    rectangles ``holes`` [(ua, ub, za, zb), ...] (grid decomposition)."""
    us = sorted({u0, u1, *[h[0] for h in holes], *[h[1] for h in holes]})
    zs = sorted({z0, z1, *[h[2] for h in holes], *[h[3] for h in holes]})
    us = [u for u in us if u0 <= u <= u1]
    zs = [z for z in zs if z0 <= z <= z1]
    for i in range(len(us) - 1):
        for j in range(len(zs) - 1):
            uc, zc = (us[i] + us[i + 1]) / 2, (zs[j] + zs[j + 1]) / 2
            if any(h[0] < uc < h[1] and h[2] < zc < h[3] for h in holes):
                continue
            fr.quad(acc, mat, us[i], us[i + 1], zs[j], zs[j + 1], v=v)


def reveal(fr: Frame, acc: Acc, mat, ua, ub, za, zb, d, back=None, v=0.0):
    """The four reveal faces of an opening ``d`` deep and (optionally) the
    back panel in material ``back``."""
    T, N, Z = fr.T3, fr.N3, np.array([0, 0, 1.0])
    acc.poly(mat, [fr.p(ua, v, za), fr.p(ua, v - d, za), fr.p(ua, v - d, zb), fr.p(ua, v, zb)], normal=T)
    acc.poly(mat, [fr.p(ub, v, za), fr.p(ub, v - d, za), fr.p(ub, v - d, zb), fr.p(ub, v, zb)], normal=-T)
    acc.poly(mat, [fr.p(ua, v, za), fr.p(ub, v, za), fr.p(ub, v - d, za), fr.p(ua, v - d, za)], normal=Z)
    acc.poly(mat, [fr.p(ua, v, zb), fr.p(ub, v, zb), fr.p(ub, v - d, zb), fr.p(ua, v - d, zb)], normal=-Z)
    if back is not None:
        fr.quad(acc, back, ua, ub, za, zb, v=v - d)


def window_frame(fr, acc, mat, ua, ub, za, zb, v, fw=0.06, fd=0.07, mullion=None, transom=None):
    """A frame of bars ``fw`` wide on the inner edge of an opening at depth
    v (its back face), plus an optional mullion (u) and transom (z)."""
    fr.box(acc, mat, ua, ua + fw, v, v + fd, za, zb)
    fr.box(acc, mat, ub - fw, ub, v, v + fd, za, zb)
    fr.box(acc, mat, ua + fw, ub - fw, v, v + fd, za, za + fw)
    fr.box(acc, mat, ua + fw, ub - fw, v, v + fd, zb - fw, zb)
    if mullion is not None:
        for um in (mullion if isinstance(mullion, (list, tuple)) else [mullion]):
            fr.box(acc, mat, um - fw * 0.5, um + fw * 0.5, v, v + fd * 0.8, za + fw, zb - fw)
    if transom is not None:
        fr.box(acc, mat, ua + fw, ub - fw, v, v + fd * 0.8, transom - fw * 0.5, transom + fw * 0.5)


# --- materials of a building ----------------------------------------------------------------------

class Palette:
    """Materials shared by a street's buildings (so a street has a look),
    varied per building by picking among a few of each."""

    def __init__(self, ctx, look: str = "mixed", grime: float = 0.4):
        self.ctx, lk, rng = ctx, ctx.looks, ctx.rng
        self.look = look
        self.grime = grime
        families = {
            "brick": [(BRICK_RED, 1.6), (BRICK_DARK, 1.6), (BRICK_LIGHT, 1.6)],
            "plaster": [(PLASTER_WHITE, 2.5)],
            "worn": [(PLASTER_WORN, 2.5), (BRICK_RED, 1.6)],
            "stone": [(STONE, 2.0), (PLASTER_WHITE, 2.5)],
            "modern": [(CONCRETE_WALL, 3.0), (TILE_FACADE, 2.5), (PLASTER_WHITE, 2.5)],
            "mixed": [(BRICK_RED, 1.6), (PLASTER_WHITE, 2.5), (PLASTER_WHITE, 2.5), (STONE, 2.0),
                      (BRICK_DARK, 1.6), (PLASTER_WORN, 2.5), (CONCRETE_WALL, 3.0)],
        }[look]
        self.walls = []
        for _ in range(int(rng.integers(4, 7))):
            keys, tile = families[int(rng.integers(len(families)))]
            k = pick_key(ctx, keys)
            tint = None
            if keys is PLASTER_WHITE or (k in ("acg:Plaster001", "acg:Plaster002", "acg:Plaster003")):
                tint = PASTELS[int(rng.integers(len(PASTELS)))]
            m = lk.pbr(k, tile * float(rng.uniform(0.85, 1.15)), bevel=0.015, dirt=grime, tint=tint, vary=0.15,
                       jitter=0.05)
            self.walls.append(m)
        self.trim = [lk.pbr(pick_key(ctx, TRIM), 2.0, bevel=0.012, dirt=grime * 0.8, jitter=0.03)
                     for _ in range(2)]
        self.frames = [lk.paint(FRAME_COLOURS[int(i)], rough=0.4, name="frame")
                       for i in rng.choice(len(FRAME_COLOURS), 3, replace=False)]
        self.glass = [self._glass((0.55, 0.60, 0.65), 0.04), self._glass((0.35, 0.38, 0.40), 0.03),
                      self._glass((0.60, 0.58, 0.52), 0.06), self._glass((0.45, 0.50, 0.58), 0.02)]
        # some windows show curtains / blinds behind the glass
        self.curtains = [self._glass(c, 0.08, metal=0.25) for c in
                         [(0.85, 0.80, 0.70), (0.75, 0.72, 0.68), (0.55, 0.30, 0.22), (0.92, 0.92, 0.90)]]
        self.shop_glass = [self._glass((0.10, 0.10, 0.10), 0.02, metal=0.85),
                           self._glass((0.14, 0.13, 0.12), 0.03, metal=0.8)]
        if ctx.__dict__.get("_lit_shops"):
            self.shop_glass.append(self._glass((0.2, 0.18, 0.15), 0.03, metal=0.6, emit=(1.0, 0.85, 0.6, 1.5)))
        self.dark = lk.paint((0.03, 0.03, 0.03), rough=0.8, name="interior")
        self.doors = [lk.pbr(pick_key(ctx, PLANKS_PAINTED + WOOD), 1.5, bevel=0.01, dirt=0.3,
                             tint=None if rng.random() < 0.5 else SIGN_COLOURS[int(rng.integers(len(SIGN_COLOURS)))])
                      for _ in range(2)]
        self.shutter = [lk.pbr(pick_key(ctx, SHUTTER), 1.2, dirt=0.3, bevel=0.005) for _ in range(2)]
        self.roof_flat = lk.pbr(pick_key(ctx, ROOF_FLAT), 3.0, dirt=0.3)
        self.roof_tiles = lk.pbr(pick_key(ctx, ROOF_TILES), 2.0, dirt=0.3, bevel=0.01)
        self.metal = lk.pbr(pick_key(ctx, PAINTED_METAL), 1.0, tint=(0.25, 0.25, 0.25), bevel=0.008, dirt=0.3)
        self.galv = lk.pbr(pick_key(ctx, GALV), 1.0, bevel=0.005, dirt=0.2)
        self.signs = [lk.paint(c, rough=0.35, name="fascia") for c in SIGN_COLOURS]
        self.letters = [lk.paint((0.95, 0.95, 0.92), rough=0.3, name="letters"),
                        lk.paint((0.85, 0.65, 0.2), rough=0.25, metal=0.8, name="letters_gold"),
                        lk.paint((0.03, 0.03, 0.03), rough=0.4, name="letters_black"),
                        lk.paint((1.0, 0.95, 0.85), rough=0.3, name="letters_lit", emission=(1.0, 0.92, 0.8, 4.0))]
        self.awning = [lk.paint(c, rough=0.8, name="awning") for c in
                       [(0.45, 0.05, 0.05), (0.05, 0.18, 0.08), (0.08, 0.10, 0.25), (0.6, 0.55, 0.45),
                        (0.12, 0.12, 0.12), (0.55, 0.30, 0.05)]]

    def _glass(self, rgb, rough, metal=0.9, emit=None):
        m = bpy.data.materials.new("window")
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*rgb, 1.0)
        b.inputs["Roughness"].default_value = rough
        b.inputs["Metallic"].default_value = metal
        if emit is not None:
            b.inputs["Emission Color"].default_value = (*emit[:3], 1.0)
            b.inputs["Emission Strength"].default_value = emit[3]
        return m

    def pick(self, lst):
        return lst[int(self.ctx.rng.integers(len(lst)))]


# --- a procedural building --------------------------------------------------------------------------

class Building:
    """A building on frame ``fr`` from u0 to u1, ``depth`` deep behind the
    facade line. ``ground``: 'shops' | 'res' | 'garage' | 'blank' | 'back'
    (alley backs: few small windows, a back door, a loading door)."""

    def __init__(self, ctx, pal: Palette, fr: Frame, u0, u1, depth, ground="shops", floors=None,
                 v0=0.0, style=None, lit=False, coll="structure", pitched=None, far=False, roof=False):
        self.ctx, self.pal, self.fr = ctx, pal, fr
        self.roof = roof           # hidden in the oblique/top previews (it stands between them and the patch)
        rng = ctx.rng
        self.u0, self.u1, self.depth, self.v0 = u0, u1, depth, v0
        self.coll, self.lit, self.far = coll, lit, far
        self.ground = ground
        self.wall = pal.pick(pal.walls) if style is None else style
        self.trim = pal.pick(pal.trim)
        self.frame = pal.pick(pal.frames)
        self.g = float(rng.uniform(3.6, 4.6)) if ground in ("shops", "garage") else float(rng.uniform(3.0, 3.6))
        self.f = float(rng.uniform(2.9, 3.4))
        self.n = int(floors if floors is not None else rng.integers(1, 6))
        self.parapet = float(rng.uniform(0.4, 1.3))
        self.pitched = (rng.random() < 0.25) if pitched is None else pitched
        if self.pitched:
            self.parapet = 0.0
        self.H = self.g + self.n * self.f + self.parapet
        self.check = None          # check(list of xy) -> bool: may an overhang go there?
        self.bay = float(rng.uniform(2.3, 3.4))
        self.ww = float(rng.uniform(0.38, 0.6)) * self.bay
        self.wh = float(rng.uniform(0.45, 0.62)) * self.f
        self.sill = float(rng.uniform(0.7, 1.0))
        self.d = float(rng.uniform(0.12, 0.25))
        self.shutters = rng.random() < 0.18 and self.ww < 1.5
        self.balconies = float(rng.choice([0.0, 0.0, 0.0, 0.3, 0.6]))
        self.french = rng.random() < 0.15
        self.string_course = rng.random() < 0.6
        self.lintels = rng.random() < 0.5
        self.cornice = rng.random() < 0.75
        self.downpipe = rng.random() < 0.7
        self.acc = Acc(ctx)
        self.holes = []
        self.signs = []   # text jobs after emission

    # ---------------------------------------------------------------------------------------
    def build(self):
        fr, acc, pal, rng = self.fr, self.acc, self.pal, self.ctx.rng
        u0, u1, v0 = self.u0, self.u1, self.v0
        W = u1 - u0
        if not self.far:
            self._ground_floor()
        # upper floors: bays centred on the facade
        nb = max(1, int((W - 0.6) // self.bay))
        margin = (W - nb * self.bay) / 2
        for k in range(self.n):
            zf = self.g + k * self.f
            for b in range(nb):
                uc = u0 + margin + (b + 0.5) * self.bay
                if self.far:
                    self._window(uc, zf, simple=True)
                else:
                    self._window(uc, zf)
        # the front, sides, back, roof
        wall_with_holes(fr, acc, self.wall, u0, u1, 0.0, self.H, [h[:4] for h in self.holes], v=v0)
        for (ua, ub, za, zb, d, back) in self.holes:
            reveal(fr, acc, self.wall, ua, ub, za, zb, d, back=back, v=v0)
        D = self.depth
        T, N, Z = fr.T3, fr.N3, np.array([0, 0, 1.0])
        acc.poly(self.wall, [fr.p(u0, v0, 0), fr.p(u0, v0 - D, 0), fr.p(u0, v0 - D, self.H), fr.p(u0, v0, self.H)],
                 normal=-T)
        acc.poly(self.wall, [fr.p(u1, v0, 0), fr.p(u1, v0 - D, 0), fr.p(u1, v0 - D, self.H), fr.p(u1, v0, self.H)],
                 normal=T)
        fr.quad(acc, self.wall, u0, u1, 0.0, self.H, v=v0 - D)       # back (faces +n; unseen)
        if self.pitched and self.n >= 1:
            self._pitched_roof()
        else:
            acc.poly(pal.roof_flat, [fr.p(u0, v0, self.H - 0.05), fr.p(u1, v0, self.H - 0.05),
                                     fr.p(u1, v0 - D, self.H - 0.05), fr.p(u0, v0 - D, self.H - 0.05)], normal=Z)
            # parapet coping
            fr.box(acc, self.trim, u0 - 0.02, u1 + 0.02, v0 - 0.3, v0 + 0.04, self.H, self.H + 0.08)
            if not self.far:
                self._roof_clutter()
        if self.cornice and not self.pitched:
            zc = self.H - self.parapet
            fr.box(acc, self.trim, u0, u1, v0, v0 + 0.22, zc - 0.28, zc)
            fr.box(acc, self.trim, u0, u1, v0, v0 + 0.1, zc - 0.4, zc - 0.28)
        if self.string_course and self.n > 0:
            fr.box(acc, self.trim, u0, u1, v0, v0 + 0.07, self.g - 0.18, self.g)
        if self.downpipe and not self.far:
            for uu in ([u0 + 0.18] if rng.random() < 0.5 else [u0 + 0.18, u1 - 0.18]):
                a = fr.p(uu, v0 + 0.12, 0.0)
                b = fr.p(uu, v0 + 0.12, self.H - self.parapet * 0.5)
                ob = self.ctx.cylinder(a, b, 0.05, pal.galv, coll=self.coll, name="downpipe", segments=10)
                if self.roof:
                    ob["roof"] = 1
        out = acc.emit(name="bld", coll=self.coll, roof_all=self.roof)
        for job in self.signs:
            ob = text_obj(self.ctx, job["s"], job["xyz"], job["yaw"], job["size"], job["mat"], depth=0.01,
                          coll=self.coll, bold=job["bold"])
            if self.roof:
                ob["roof"] = 1
        return out

    # ---------------------------------------------------------------------------------------
    def _window(self, uc, zf, simple=False):
        fr, acc, pal, rng = self.fr, self.acc, self.pal, self.ctx.rng
        v0 = self.v0
        ww, wh = self.ww, self.wh
        if self.french:
            za, zb = zf + 0.05, zf + min(self.f - 0.35, 2.4)
        else:
            za, zb = zf + self.sill, zf + self.sill + wh
            zb = min(zb, zf + self.f - 0.3)
        ua, ub = uc - ww / 2, uc + ww / 2
        r = rng.random()
        back = pal.pick(pal.curtains) if r < 0.22 else pal.pick(pal.glass)
        if simple:
            self.holes.append((ua, ub, za, zb, 0.08, back))
            return
        d = self.d
        self.holes.append((ua, ub, za, zb, d, back))
        mull = uc if ww > 0.95 else None
        trans = za + 0.72 * (zb - za) if (zb - za) > 1.3 and rng.random() < 0.7 else None
        window_frame(fr, acc, self.frame, ua, ub, za, zb, v0 - d, mullion=mull, transom=trans)
        # sill
        if not self.french:
            fr.box(acc, self.trim, ua - 0.06, ub + 0.06, v0 - 0.02, v0 + 0.07, za - 0.06, za)
        if self.lintels:
            fr.box(acc, self.trim, ua - 0.1, ub + 0.1, v0, v0 + 0.03, zb, zb + 0.2)
        if self.shutters:
            sh = self.frame if rng.random() < 0.5 else pal.pick(pal.doors)
            fr.box(acc, sh, ua - ww / 2 - 0.02, ua - 0.02, v0 + 0.01, v0 + 0.05, za, zb)
            fr.box(acc, sh, ub + 0.02, ub + ww / 2 + 0.02, v0 + 0.01, v0 + 0.05, za, zb)
        if self.french:
            self._railing(ua, ub, v0, v0 + 0.25, za, za + 0.95, pal.metal)
        elif self.balconies and zf > 3.0 and rng.random() < self.balconies:
            self._balcony(ua - 0.5, ub + 0.5, zf)

    def _railing(self, ua, ub, va, vb, za, zb, mat):
        fr, acc = self.fr, self.acc
        fr.box(acc, mat, ua, ub, vb - 0.03, vb, zb - 0.04, zb)
        fr.box(acc, mat, ua, ub, vb - 0.03, vb, za + 0.08, za + 0.11)
        n = max(2, int((ub - ua) / 0.12))
        for u in np.linspace(ua + 0.02, ub - 0.02, n):
            fr.box(acc, mat, u - 0.008, u + 0.008, vb - 0.025, vb - 0.005, za, zb)
        if vb - va > 0.3:
            for u in (ua, ub - 0.03):
                fr.box(acc, mat, u, u + 0.03, va, vb, zb - 0.04, zb)

    def _ok(self, ua, ub, va, vb):
        if self.check is None:
            return True
        fr = self.fr
        pts = [fr.xy(u, v) for u in np.linspace(ua, ub, max(2, int((ub - ua) / 0.5) + 1)) for v in (va, (va + vb) / 2, vb)]
        return self.check(pts)

    def _balcony(self, ua, ub, zf):
        fr, acc, pal = self.fr, self.acc, self.pal
        dep = float(self.ctx.rng.uniform(0.8, 1.3))
        v0 = self.v0
        if not self._ok(ua, ub, v0, v0 + dep):
            return
        fr.box(acc, self.trim, ua, ub, v0, v0 + dep, zf - 0.16, zf)
        # railing on three sides
        self._railing(ua, ub, v0, v0 + dep, zf, zf + 1.0, pal.metal)
        for u in (ua, ub - 0.02):
            fr.box(acc, pal.metal, u, u + 0.02, v0, v0 + dep, zf + 0.96, zf + 1.0)
            for vv in np.linspace(v0 + 0.1, v0 + dep - 0.05, max(2, int(dep / 0.12))):
                fr.box(acc, pal.metal, u, u + 0.016, vv - 0.008, vv + 0.008, zf, zf + 1.0)

    # ---------------------------------------------------------------------------------------
    def _ground_floor(self):
        fr, acc, pal, rng = self.fr, self.acc, self.pal, self.ctx.rng
        u0, u1, v0, g = self.u0, self.u1, self.v0, self.g
        W = u1 - u0
        # plinth
        plinth_h = float(rng.uniform(0.25, 0.6))
        plinth = self.trim if rng.random() < 0.7 else pal.pick(pal.walls)
        kind = self.ground
        if kind == "shops":
            n_units = max(1, int(round(W / float(rng.uniform(4.5, 8.0)))))
            edges = np.linspace(u0, u1, n_units + 1)
            for i in range(n_units):
                self._shopfront(edges[i], edges[i + 1])
            return
        if kind == "garage":
            u = u0 + 0.8
            while u + 3.2 < u1 - 0.6:
                gw = float(rng.uniform(2.6, 4.2))
                if u + gw > u1 - 0.6:
                    break
                gh = min(g - 0.5, float(rng.uniform(2.6, 3.4)))
                back = pal.pick(pal.shutter)
                self.holes.append((u, u + gw, 0.0, gh, 0.12, back))
                fr.box(acc, pal.metal, u - 0.05, u + gw + 0.05, v0, v0 + 0.25, gh, gh + 0.35)  # shutter box
                if rng.random() < 0.5:   # a side door
                    ud = u + gw + 0.6
                    if ud + 1.0 < u1 - 0.5:
                        self._door(ud, ud + 1.0, 2.2)
                        u = ud + 1.0
                u += gw + float(rng.uniform(0.8, 2.5))
            fr.box(acc, plinth, u0, u1, v0, v0 + 0.03, 0.0, plinth_h)
            return
        if kind in ("res", "back", "blank"):
            nb = max(1, int((W - 0.6) // self.bay))
            margin = (W - nb * self.bay) / 2
            door_b = int(rng.integers(nb)) if kind != "blank" else -1
            for b in range(nb):
                uc = u0 + margin + (b + 0.5) * self.bay
                if b == door_b:
                    dw = float(rng.uniform(0.95, 1.3)) if kind != "back" else float(rng.uniform(0.9, 1.1))
                    self._door(uc - dw / 2, uc + dw / 2, float(rng.uniform(2.1, 2.5)))
                elif kind == "blank" or (kind == "back" and rng.random() < 0.6):
                    continue
                else:
                    za = max(self.sill, plinth_h + 0.3)
                    if kind == "back":           # small high windows, some barred
                        ww, za, zb = self.ww * 0.7, 1.6, min(g - 0.4, 2.5)
                    else:
                        ww, zb = self.ww, min(g - 0.35, za + self.wh)
                    ua, ub = uc - ww / 2, uc + ww / 2
                    self.holes.append((ua, ub, za, zb, self.d, pal.pick(pal.glass)))
                    window_frame(fr, acc, self.frame, ua, ub, za, zb, v0 - self.d,
                                 mullion=uc if ww > 1.0 else None)
                    fr.box(acc, self.trim, ua - 0.06, ub + 0.06, v0 - 0.02, v0 + 0.07, za - 0.06, za)
                    if kind == "back" and rng.random() < 0.5:     # security bars
                        for u in np.linspace(ua + 0.08, ub - 0.08, max(2, int(ww / 0.13))):
                            fr.box(acc, pal.metal, u - 0.012, u + 0.012, v0 + 0.02, v0 + 0.045, za, zb)
            if rng.random() < 0.8:
                fr.box(acc, plinth, u0, u1, v0, v0 + 0.03, 0.0, plinth_h)

    def _door(self, ua, ub, h, recess=0.2):
        fr, acc, pal, rng = self.fr, self.acc, self.pal, self.ctx.rng
        v0 = self.v0
        door = pal.pick(pal.doors)
        self.holes.append((ua, ub, 0.0, h, recess, door))
        window_frame(fr, acc, self.frame, ua, ub, 0.0, h, v0 - recess, fw=0.07)
        fr.box(acc, self.trim, ua - 0.12, ub + 0.12, v0, v0 + 0.05, h, h + 0.22)
        if rng.random() < 0.5:     # a step
            fr.box(acc, self.trim, ua - 0.1, ub + 0.1, v0, v0 + 0.3, 0.0, 0.16)

    def _shopfront(self, ua0, ub0):
        fr, acc, pal, rng = self.fr, self.acc, self.pal, self.ctx.rng
        v0, g = self.v0, self.g
        pw = float(rng.uniform(0.25, 0.5))
        ua, ub = ua0 + pw, ub0 - pw
        # pilasters
        for a, b in ((ua0, ua), (ub, ub0)):
            fr.box(acc, self.trim, a, b, v0, v0 + 0.06, 0.0, g - 0.05)
        fascia_top = g - float(rng.uniform(0.15, 0.35))
        fascia_bot = fascia_top - float(rng.uniform(0.5, 0.8))
        top = fascia_bot - 0.12
        riser = float(rng.uniform(0.25, 0.65))
        closed = rng.random() < 0.2
        dw = float(rng.uniform(0.95, 1.4))
        door_left = rng.random() < 0.5
        if ub - ua < dw + 1.2:
            dw = 0.0
        if dw:
            du0, du1 = (ua, ua + dw) if door_left else (ub - dw, ub)
            wa, wb = (ua + dw + 0.1, ub) if door_left else (ua, ub - dw - 0.1)
        else:
            wa, wb = ua, ub
        sg = pal.pick(pal.shop_glass)
        if closed:
            back = pal.pick(pal.shutter)
            self.holes.append((ua, ub, 0.0, top, 0.08, back))
            fr.box(acc, pal.metal, ua - 0.03, ub + 0.03, v0 - 0.02, v0 + 0.22, top - 0.02, top + 0.3)
        else:
            self.holes.append((wa, wb, riser, top, 0.14, sg))
            nm = max(0, int((wb - wa) / 1.8))
            mulls = list(np.linspace(wa, wb, nm + 2)[1:-1]) if nm else None
            window_frame(fr, acc, self.frame, wa, wb, riser, top, v0 - 0.14, fw=0.05, mullion=mulls,
                         transom=top - 0.45 if rng.random() < 0.5 else None)
            fr.box(acc, self.trim, wa - 0.02, wb + 0.02, v0, v0 + 0.02, 0.0, riser)   # stall riser cladding
            if dw:
                self.holes.append((du0, du1, 0.0, top, 0.3, sg if rng.random() < 0.6 else pal.pick(pal.doors)))
                window_frame(fr, acc, self.frame, du0, du1, 0.0, top, v0 - 0.3, fw=0.07,
                             transom=min(top - 0.35, 2.3))
        # fascia sign with the shop's name
        sign = pal.pick(pal.signs)
        fr.box(acc, sign, ua0 + 0.05, ub0 - 0.05, v0, v0 + 0.1, fascia_bot, fascia_top)
        fr.box(acc, self.trim, ua0, ub0, v0, v0 + 0.16, fascia_top, fascia_top + 0.08)
        name = SHOP_NAMES[int(rng.integers(len(SHOP_NAMES)))]
        letters = pal.letters[3] if self.lit and rng.random() < 0.6 else pal.pick(pal.letters[:3])
        hgt = min(0.42, 0.62 * (fascia_top - fascia_bot))
        width_est = 0.62 * hgt * len(name)
        if width_est > 0.85 * (ub0 - ua0):
            hgt *= 0.85 * (ub0 - ua0) / width_est
        c = fr.p((ua0 + ub0) / 2, v0 + 0.105, (fascia_bot + fascia_top) / 2 - 0.3 * hgt)
        self.signs.append({"s": name, "xyz": c, "yaw": fr.yaw, "size": hgt, "mat": letters,
                           "bold": float(rng.choice([0.0, 0.0, 0.004]))})
        # awning over open shops (it projects over the pavement)
        if not closed and rng.random() < 0.3:
            self._awning(ua, ub, fascia_bot - 0.05)

    def _awning(self, ua, ub, z):
        fr, acc, pal, rng = self.fr, self.acc, self.pal, self.ctx.rng
        dep = float(rng.uniform(0.9, 1.4))
        drop = float(rng.uniform(0.4, 0.7))
        v0 = self.v0
        if not self._ok(ua, ub, v0, v0 + dep):
            return
        m = pal.pick(pal.awning)
        # sloping sheet + valance + two side cheeks
        acc.poly(m, [fr.p(ua, v0, z), fr.p(ub, v0, z), fr.p(ub, v0 + dep, z - drop), fr.p(ua, v0 + dep, z - drop)],
                 normal=np.array([0, 0, 1.0]) + fr.N3)
        acc.poly(m, [fr.p(ua, v0 + dep, z - drop), fr.p(ub, v0 + dep, z - drop), fr.p(ub, v0 + dep, z - drop - 0.25),
                     fr.p(ua, v0 + dep, z - drop - 0.25)], normal=fr.N3)
        for u, s in ((ua, -1), (ub, 1)):
            acc.poly(m, [fr.p(u, v0, z), fr.p(u, v0 + dep, z - drop), fr.p(u, v0, z - drop)], normal=fr.T3 * s)

    def _pitched_roof(self):
        fr, acc, pal = self.fr, self.acc, self.pal
        u0, u1, v0, D = self.u0, self.u1, self.v0, self.depth
        zt = self.H - self.parapet
        pitch = float(self.ctx.rng.uniform(0.5, 0.9))
        ov = 0.35
        ridge_v = v0 - D / 2
        zr = zt + pitch * D / 2
        acc.poly(pal.roof_tiles, [fr.p(u0 - 0.1, v0 + ov, zt - ov * pitch), fr.p(u1 + 0.1, v0 + ov, zt - ov * pitch),
                                  fr.p(u1 + 0.1, ridge_v, zr), fr.p(u0 - 0.1, ridge_v, zr)],
                 normal=fr.N3 + np.array([0, 0, 1.0]))
        acc.poly(pal.roof_tiles, [fr.p(u0 - 0.1, v0 - D - ov, zt - ov * pitch),
                                  fr.p(u1 + 0.1, v0 - D - ov, zt - ov * pitch),
                                  fr.p(u1 + 0.1, ridge_v, zr), fr.p(u0 - 0.1, ridge_v, zr)],
                 normal=-fr.N3 + np.array([0, 0, 1.0]))
        for u, s in ((u0, -1), (u1, 1)):
            acc.poly(self.wall, [fr.p(u, v0, zt), fr.p(u, v0 - D, zt), fr.p(u, ridge_v, zr)], normal=fr.T3 * s)
        fr.box(acc, self.trim, u0, u1, v0, v0 + 0.25, zt - 0.2, zt)
        if self.ctx.rng.random() < 0.6:           # a dormer or a chimney
            uc = float(self.ctx.rng.uniform(u0 + 1.5, u1 - 1.5)) if u1 - u0 > 3.5 else (u0 + u1) / 2
            fr.box(acc, pal.pick(pal.walls), uc - 0.35, uc + 0.35, ridge_v - 0.8, ridge_v - 0.1, zt, zr + 0.9)

    def _roof_clutter(self):
        fr, acc, pal, rng = self.fr, self.acc, self.pal, self.ctx.rng
        u0, u1, v0, D, H = self.u0, self.u1, self.v0, self.depth, self.H
        for _ in range(int(rng.integers(0, 3))):
            w, dd, h = rng.uniform(0.6, 2.0), rng.uniform(0.6, 1.5), rng.uniform(0.5, 1.4)
            uc = float(rng.uniform(u0 + 1, max(u0 + 1.1, u1 - 1)))
            vc = float(rng.uniform(v0 - D + 1.5, v0 - 1.5))
            fr.box(acc, pal.galv, uc - w / 2, uc + w / 2, vc - dd / 2, vc + dd / 2, H - 0.05, H + h)


# --- Poly Haven modular facades --------------------------------------------------------------------

FACTORY = "ph:modular_factory_facade"
APTS = "ph:modular_urban_apartments_facade"
FACTORY_GROUND = [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
FACTORY_GARAGE6 = [57, 58, 59]
FACTORY_UPPER = [82, 83, 84, 85, 86, 87, 88, 89, 92, 93, 94, 95, 98, 99, 100, 101, 102, 103, 104, 105, 110, 112,
                 114, 116, 118, 120, 122, 124, 126, 128]
FACTORY_BLIND = [81, 90, 91, 96, 97, 117, 119, 127, 129]
APTS_GROUND = [62, 63, 64, 72]
APTS_UPPER = [104, 107, 110, 115, 116]
APTS_BLIND = [97]


def modular_building(ctx, pal: Palette, fr: Frame, u0, n_mod, floors, kind="factory", depth=10.0, coll="structure",
                     v0=0.0, roof=False):
    """A building of 3 m Poly Haven facade modules on ``fr`` starting at u0,
    ``n_mod`` modules wide, ``floors`` storeys (3 m each), recoloured per
    building; a closed volume behind it (dark front, scan sides/roof)."""
    rng = ctx.rng
    fk = FACTORY if kind == "factory" else APTS
    if fk not in ctx.lib.items:
        return None
    if kind == "factory":
        choice = int(rng.integers(5))
        tints = [((1.0, 1.0, 1.0), 1.0, 1.0), ((0.92, 0.85, 0.80), 1.0, 1.0), ((0.75, 0.72, 0.70), 0.6, 1.0),
                 ((1.0, 0.97, 0.92), 0.08, 1.9), ((0.35, 0.34, 0.33), 0.15, 1.2)][choice]
        wall_key = "brick"
        ground, upper, blind = FACTORY_GROUND, FACTORY_UPPER, FACTORY_BLIND
        side = ctx.looks.pbr(pick_key(ctx, ["ph:red_brick_03", "acg:Bricks090", "ph:factory_brick", "acg:Bricks085"]),
                             1.5, dirt=0.4, tint=(tints[0] if choice in (1, 2) else None))
    else:
        c = PASTELS[int(rng.integers(len(PASTELS)))]
        tints = (c, 0.15, 1.45)
        choice = f"{int(rng.integers(1000))}"
        wall_key = "plaster"
        ground, upper, blind = APTS_GROUND, APTS_UPPER, APTS_BLIND
        side = ctx.looks.pbr(pick_key(ctx, ["acg:Plaster001", "acg:Plaster002", "ph:white_plaster_02"]), 2.5,
                             tint=c, dirt=0.4)
    rgb, sat, val = tints
    tag = f"{kind}{choice}"

    def recolour(m):
        return tint_material(m, rgb, sat=sat, val=val) if wall_key in m.name else None

    upper_pick = [int(i) for i in rng.choice(upper, size=min(3, len(upper)), replace=False)]
    ground_pick = [int(i) for i in rng.choice(ground, size=min(3, len(ground)), replace=False)]

    def put(idx, u, z):
        pk = f"{fk}#{idx}"
        if pk not in {f"{fk}#{p['index']}" for p in ctx.lib.pieces(fk)}:
            return
        var = piece_variant(ctx, pk, tag, recolour)
        size = ctx.lib.size(pk)
        xy = fr.xy(u + size[0] / 2, v0)
        e = place_variant(ctx, pk, var, xy, z=z, yaw=fr.yaw, coll=coll)
        if roof:
            e["roof"] = 1

    # ground floor (garage doors in factories now and then)
    u = u0
    end = u0 + 3 * n_mod
    while u < end - 0.1:
        if kind == "factory" and end - u >= 6 and rng.random() < 0.25:
            put(int(rng.choice(FACTORY_GARAGE6)), u, 0.0)
            u += 6.0
            continue
        put(int(rng.choice(ground_pick)) if rng.random() < 0.75 else int(rng.choice(blind)), u, 0.0)
        u += 3.0
    for k in range(1, floors):
        for i in range(n_mod):
            idx = int(rng.choice(upper_pick)) if rng.random() < 0.85 else int(rng.choice(blind))
            put(idx, u0 + 3 * i, 3.0 * k)
    H = 3.0 * floors
    for i in range(n_mod):
        if kind == "factory":
            put(5, u0 + 3 * i, 0.0)          # base
            put(36, u0 + 3 * i, H)           # crown
            if floors > 1:
                put(23, u0 + 3 * i, 3.0 - 0.16)
        else:
            put(7, u0 + 3 * i, 0.0)
            put(31, u0 + 3 * i, H)
            for k in range(1, floors):
                put(19, u0 + 3 * i, 3.0 * k - 0.2)
    # the volume behind (dark front so windows read as rooms)
    acc = Acc(ctx)
    D = depth
    u1 = end
    fr.quad(acc, pal.dark, u0, u1, 0.0, H + 0.7, v=v0 - 0.35)
    T = fr.T3
    acc.poly(side, [fr.p(u0, v0 + 0.02, 0), fr.p(u0, v0 - D, 0), fr.p(u0, v0 - D, H + 0.7),
                    fr.p(u0, v0 + 0.02, H + 0.7)], normal=-T)
    acc.poly(side, [fr.p(u1, v0 + 0.02, 0), fr.p(u1, v0 - D, 0), fr.p(u1, v0 - D, H + 0.7),
                    fr.p(u1, v0 + 0.02, H + 0.7)], normal=T)
    acc.poly(pal.roof_flat, [fr.p(u0, v0, H + 0.7), fr.p(u1, v0, H + 0.7), fr.p(u1, v0 - D, H + 0.7),
                             fr.p(u0, v0 - D, H + 0.7)], normal=np.array([0, 0, 1.0]))
    fr.quad(acc, side, u0, u1, 0.0, H + 0.7, v=v0 - D)
    acc.emit(name="modbld", coll=coll, roof_all=roof)
    return H


# --- furniture -----------------------------------------------------------------------------------

def catenary(ctx, a, b, sag, r, mat, coll="scatter", n=8, name="cable"):
    """A sagging cable from 3-D point a to b as ``n`` straight segments."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pts = []
    for i in range(n + 1):
        t = i / n
        p = a + (b - a) * t
        p[2] -= sag * 4 * t * (1 - t)
        pts.append(p)
    return [ctx.cylinder(pts[i], pts[i + 1], r, mat, coll=coll, name=name, segments=6) for i in range(n)]


def seg_clear(ctx, a, b, clear=0.5, kinds=None, step=0.4):
    """Every point of the 2-D segment a -> b is ``clear`` metres from the
    footprints of ``kinds`` (default: patch solids and the start/goal)."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    L = float(np.linalg.norm(b - a))
    feet = [f for f in ctx.foot if kinds is None or f.get("kind") in kinds]
    for t in np.linspace(0, 1, max(2, int(L / step) + 1)):
        q = a + (b - a) * t
        for f in feet:
            if BC.footprint_distance(q, f) < clear:
                return False
    return True


def lamp_post(ctx, pal, xy, reach_dir, h=None, z0=0.0, lit=False, coll="scatter", style=None):
    """A street light: tapered pole, an arm over the road, a luminaire."""
    rng = ctx.rng
    h = float(h or rng.uniform(6.0, 8.5))
    style = style or ("arm" if rng.random() < 0.7 else "top")
    x, y = float(xy[0]), float(xy[1])
    mat = pal.metal if rng.random() < 0.6 else pal.galv
    ctx.cylinder((x, y, z0), (x, y, z0 + 0.9), 0.11, mat, coll=coll, name="lamp", segments=16)
    ctx.cylinder((x, y, z0 + 0.9), (x, y, z0 + h), 0.07, mat, coll=coll, name="lamp", segments=16)
    head = ctx.looks.paint((1.0, 0.95, 0.85), rough=0.3, name="lamp_lens",
                           emission=(1.0, 0.85, 0.6, 8.0) if lit else None)
    d = np.asarray(reach_dir, float)
    d = d / max(np.linalg.norm(d), 1e-9)
    if style == "arm":
        L = float(rng.uniform(1.0, 2.0))
        tip = (x + d[0] * L, y + d[1] * L, z0 + h + 0.25)
        ctx.cylinder((x, y, z0 + h - 0.1), tip, 0.045, mat, coll=coll, name="lamp", segments=10)
        yaw = math.atan2(d[1], d[0])
        ctx.box((tip[0], tip[1], tip[2] - 0.05), (0.32, 0.14, 0.06), mat, yaw=yaw, coll=coll, name="lamp")
        ctx.box((tip[0], tip[1], tip[2] - 0.115), (0.26, 0.1, 0.008), head, yaw=yaw, coll=coll, name="lamp")
        if lit:
            ctx.point_light((tip[0], tip[1], tip[2] - 0.3), 250.0, 0.2)
    else:
        ctx.cylinder((x, y, z0 + h), (x, y, z0 + h + 0.5), 0.2, head, coll=coll, name="lamp", segments=16)
        ctx.cylinder((x, y, z0 + h + 0.5), (x, y, z0 + h + 0.6), 0.24, mat, coll=coll, name="lamp", segments=16)
        if lit:
            ctx.point_light((x, y, z0 + h + 0.2), 150.0, 0.2)


def ball(ctx, c, r, mat, coll="scatter", collide=None, name="ball", n=16, hemi=False):
    """A sphere (or its upper half) from an explicit triangle list, in a
    fixed order. ``ctx.sphere`` (bmesh ``create_uvsphere``) gives the same
    vertices every run but its faces in a run-dependent order, and the cloud
    samples triangles in mesh order -- so two builds of one course gave
    different clouds."""
    c = np.asarray(c, float)
    m = n // 2
    rings = [(math.pi / 2) * (1 - k / m) for k in range(m + 1)] if hemi else \
        [math.pi / 2 - math.pi * k / m for k in range(m + 1)]              # latitude, pole first
    V = [tuple(c + [0.0, 0.0, r])]
    for lat in rings[1:-1] if not hemi else rings[1:]:
        for j in range(n):
            lon = 2 * math.pi * j / n
            V.append(tuple(c + r * np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon),
                                             math.sin(lat)])))
    nr = (len(V) - 1) // n
    F = [(0, 1 + j, 1 + (j + 1) % n) for j in range(n)]
    for k in range(nr - 1):
        a, b = 1 + k * n, 1 + (k + 1) * n
        for j in range(n):
            j2 = (j + 1) % n
            F += [(a + j, b + j, b + j2), (a + j, b + j2, a + j2)]
    last = 1 + (nr - 1) * n
    if hemi:                                              # a flat disc closes the half
        V.append(tuple(c))
        F += [(len(V) - 1, last + (j + 1) % n, last + j) for j in range(n)]
    else:
        V.append(tuple(c - [0.0, 0.0, r]))
        F += [(len(V) - 1, last + (j + 1) % n, last + j) for j in range(n)]
    return ctx.mesh(V, F, mat, coll=coll, collide=collide, name=name, smooth=True)


def bollard(ctx, mat, xy, z0=0.0, h=0.95, r=0.09, band=None, coll="scatter", square=False):
    x, y = float(xy[0]), float(xy[1])
    if square:
        ctx.box((x, y, z0 + h / 2), (r, r, h / 2), mat, coll=coll, name="bollard")
    else:
        ctx.cylinder((x, y, z0), (x, y, z0 + h), r, mat, coll=coll, name="bollard", segments=16)
        ball(ctx, (x, y, z0 + h), r * 0.98, mat, coll=coll, name="bollard", hemi=True)
        if band is not None:
            ctx.cylinder((x, y, z0 + h - 0.2), (x, y, z0 + h - 0.12), r + 0.004, band, coll=coll, name="bollard",
                         segments=16)
    ctx.claim_disc((x, y), r + 0.05, kind="bollard")


def bike_stand(ctx, mat, xy, yaw, z0=0.0, coll="scatter"):
    """A Sheffield stand: an inverted U of steel tube."""
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = float(xy[0]), float(xy[1])
    L, h, r = 0.35, 0.8, 0.03
    a = (x - c * L, y - s * L)
    b = (x + c * L, y + s * L)
    ctx.cylinder((a[0], a[1], z0), (a[0], a[1], z0 + h), r, mat, coll=coll, name="bikestand", segments=10)
    ctx.cylinder((b[0], b[1], z0), (b[0], b[1], z0 + h), r, mat, coll=coll, name="bikestand", segments=10)
    ctx.cylinder((a[0], a[1], z0 + h), (b[0], b[1], z0 + h), r, mat, coll=coll, name="bikestand", segments=10)
    ctx.claim_rect((x, y), (L + 0.1, 0.12), yaw, kind="bikestand")


def bench(ctx, wood, metal, xy, yaw, z0=0.0, coll="scatter"):
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = float(xy[0]), float(xy[1])
    L = 1.8
    for k in range(4):          # seat slats + back slats
        off = -0.2 + 0.1 * k if k < 3 else 0.0
        if k < 3:
            ctx.box((x - s * off, y + c * off, z0 + 0.45), (L / 2, 0.04, 0.02), wood, yaw=yaw, coll=coll, name="bench")
    for k in range(2):
        off = 0.26
        ctx.box((x - s * off, y + c * off, z0 + 0.62 + 0.14 * k), (L / 2, 0.02, 0.05), wood, yaw=yaw, coll=coll,
                name="bench")
    for u in (-0.7, 0.7):
        ctx.box((x + c * u, y + s * u, z0 + 0.22), (0.03, 0.25, 0.22), metal, yaw=yaw, coll=coll, name="bench")
        ctx.box((x + c * u - s * 0.25, y + s * u + c * 0.25, z0 + 0.5), (0.03, 0.02, 0.3), metal, yaw=yaw,
                coll=coll, name="bench")
    ctx.claim_rect((x, y), (L / 2 + 0.1, 0.4), yaw, kind="bench")


def litter_bin(ctx, mat, xy, z0=0.0, coll="scatter"):
    x, y = float(xy[0]), float(xy[1])
    ctx.cylinder((x, y, z0), (x, y, z0 + 0.85), 0.24, mat, coll=coll, name="bin", segments=20)
    ctx.cylinder((x, y, z0 + 0.85), (x, y, z0 + 0.95), 0.27, mat, coll=coll, name="bin", segments=20)
    ctx.claim_disc((x, y), 0.3, kind="bin")


def wheelie_bin(ctx, body, lid, xy, yaw, z0=0.0, big=False, coll="scatter"):
    """A 240 l (or 1100 l with ``big``) wheeled bin."""
    x, y = float(xy[0]), float(xy[1])
    w, d, h = (1.3, 1.05, 1.25) if big else (0.58, 0.72, 1.0)
    ctx.box((x, y, z0 + 0.08 + h / 2 - 0.04), (w / 2 - 0.02, d / 2 - 0.03, h / 2 - 0.04), body, yaw=yaw, coll=coll,
            name="wbin")
    ctx.box((x, y, z0 + h + 0.03), (w / 2 + 0.02, d / 2 + 0.02, 0.035), lid, yaw=yaw, coll=coll, name="wbin")
    c, s = math.cos(yaw), math.sin(yaw)
    for u in (-w / 2 + 0.08, w / 2 - 0.08):
        px, py = x + c * u - s * (-d / 2 + 0.08), y + s * u + c * (-d / 2 + 0.08)
        ctx.cylinder((px - c * 0.03, py - s * 0.03, z0 + 0.1), (px + c * 0.03, py + s * 0.03, z0 + 0.1), 0.1,
                     lid, coll=coll, name="wbin", segments=12)
    ctx.claim_rect((x, y), (w / 2 + 0.05, d / 2 + 0.05), yaw, kind="wbin")


def dumpster(ctx, body, xy, yaw, z0=0.0, coll="scatter", open_lid=False):
    """A steel skip bin (1.9 x 1.1 x 1.2) with a two-part lid on castors."""
    x, y = float(xy[0]), float(xy[1])
    L, D, H = 1.9, 1.15, 1.2
    c, s = math.cos(yaw), math.sin(yaw)
    ctx.box((x, y, z0 + 0.15 + H / 2), (L / 2, D / 2, H / 2), body, yaw=yaw, coll=coll, name="dumpster")
    ctx.box((x, y, z0 + 0.15 + H - 0.05), (L / 2 + 0.04, D / 2 + 0.04, 0.05), body, yaw=yaw, coll=coll,
            name="dumpster")
    lid = ctx.looks.paint((0.08, 0.08, 0.08), rough=0.6, name="lid")
    if open_lid:
        ctx.box((x - s * (D / 2 + 0.05), y + c * (D / 2 + 0.05), z0 + 0.15 + H + 0.45), (L / 2, 0.03, 0.5), lid,
                yaw=yaw, coll=coll, name="dumpster")
    else:
        for u in (-L / 4, L / 4):
            ctx.box((x + c * u, y + s * u, z0 + 0.15 + H + 0.04), (L / 4 - 0.01, D / 2 + 0.05, 0.03), lid, yaw=yaw,
                    coll=coll, name="dumpster")
    for u in (-L / 2 + 0.15, L / 2 - 0.15):
        for v in (-D / 2 + 0.12, D / 2 - 0.12):
            px, py = x + c * u - s * v, y + s * u + c * v
            ctx.cylinder((px, py, z0), (px, py, z0 + 0.15), 0.06, lid, coll=coll, name="dumpster", segments=10)
    ctx.claim_rect((x, y), (L / 2 + 0.1, D / 2 + 0.1), yaw, kind="dumpster")


def pallet(ctx, wood, xy, yaw, z0=0.0, coll="scatter", stack=1):
    """EUR pallets (1.2 x 0.8 x 0.144), stacked."""
    x, y = float(xy[0]), float(xy[1])
    c, s = math.cos(yaw), math.sin(yaw)
    for k in range(stack):
        z = z0 + k * 0.144
        for v in (-0.35, 0.0, 0.35):           # blocks + bottom boards
            px, py = x - s * v, y + c * v
            ctx.box((px, py, z + 0.011), (0.6, 0.05, 0.011), wood, yaw=yaw, coll=coll, name="pallet")
            ctx.box((px, py, z + 0.022 + 0.039), (0.6, 0.05, 0.039), wood, yaw=yaw, coll=coll, name="pallet")
        for u in np.linspace(-0.52, 0.52, 5):  # top boards across
            px, py = x + c * u, y + s * u
            ctx.box((px, py, z + 0.1 + 0.011 + 0.022), (0.05, 0.4, 0.011), wood, yaw=yaw, coll=coll, name="pallet")
        ctx.box((x, y, z + 0.1 + 0.0), (0.6, 0.4, 0.011), wood, yaw=yaw, coll=coll, name="pallet")
    ctx.claim_rect((x, y), (0.65, 0.45), yaw, kind="pallet")


def cone(ctx, xy, z0=0.0, h=0.7, coll="scatter"):
    """A traffic cone: a coned mesh with a reflective band, on a square base."""
    x, y = float(xy[0]), float(xy[1])
    orange = ctx.looks.paint((0.9, 0.25, 0.02), rough=0.5, name="cone")
    white = ctx.looks.paint((0.9, 0.9, 0.9), rough=0.3, name="cone_band")
    ctx.box((x, y, z0 + 0.02), (0.2, 0.2, 0.02), orange, coll=coll, name="cone")
    n = 16
    for (za, zb, ra, rb, m) in [(0.04, h * 0.5, 0.15, 0.09, orange), (h * 0.5, h * 0.7, 0.09, 0.06, white),
                                (h * 0.7, h, 0.06, 0.025, orange)]:
        V, F = [], []
        for i in range(n):
            t = 2 * math.pi * i / n
            V.append((x + ra * math.cos(t), y + ra * math.sin(t), z0 + za))
            V.append((x + rb * math.cos(t), y + rb * math.sin(t), z0 + zb))
        for i in range(n):
            j = (i + 1) % n
            F.append((2 * i, 2 * j, 2 * j + 1, 2 * i + 1))
        ctx.mesh(V, F, m, coll=coll, name="cone")
    ctx.claim_disc((x, y), 0.25, kind="cone")


def road_sign(ctx, pal, xy, facing_yaw, z0=0.0, coll="scatter", kind=None):
    """A traffic sign on a post (no entry, no parking, give way, parking, 30)."""
    rng = ctx.rng
    kind = kind or str(rng.choice(["noentry", "noparking", "p", "limit", "giveway", "street"]))
    x, y = float(xy[0]), float(xy[1])
    h = float(rng.uniform(2.2, 2.7))
    ctx.cylinder((x, y, z0), (x, y, z0 + h + 0.35), 0.035, pal.galv, coll=coll, name="sign", segments=10)
    n = np.array([math.cos(facing_yaw), math.sin(facing_yaw)])
    zc = z0 + h
    red = ctx.looks.paint((0.7, 0.03, 0.03), rough=0.35, name="sign_red")
    white = ctx.looks.paint((0.92, 0.92, 0.9), rough=0.35, name="sign_white")
    blue = ctx.looks.paint((0.02, 0.15, 0.55), rough=0.35, name="sign_blue")
    black = ctx.looks.paint((0.02, 0.02, 0.02), rough=0.4, name="sign_black")
    front = np.array([x, y]) + n * 0.05
    yaw_t = facing_yaw + math.pi / 2     # local -y (the face) turned onto n

    def disc(r, m, off):
        p = front + n * off
        ctx.cylinder((p[0] - n[0] * 0.005, p[1] - n[1] * 0.005, zc), (p[0] + n[0] * 0.005, p[1] + n[1] * 0.005, zc),
                     r, m, coll=coll, name="sign", segments=28)

    if kind == "noentry":
        disc(0.32, red, 0.0)
        p = front + n * 0.012
        ctx.box((p[0], p[1], zc), (0.22, 0.004, 0.05), white, yaw=yaw_t, coll=coll, name="sign")
    elif kind == "noparking":
        disc(0.32, red, 0.0)
        disc(0.25, blue, 0.008)
        p = front + n * 0.016
        ctx.box((p[0], p[1], zc), (0.26, 0.004, 0.035), red, yaw=yaw_t + math.pi / 4, coll=coll, name="sign")
    elif kind == "limit":
        disc(0.32, red, 0.0)
        disc(0.26, white, 0.008)
        text_obj(ctx, str(rng.choice(["30", "20", "50", "10"])), (*(front + n * 0.016), zc - 0.1), yaw_t, 0.28,
                 black, coll=coll, bold=0.01)
    elif kind == "giveway":
        p = front
        V = []
        for ang in (90, 210, 330):
            a = math.radians(ang + 180)
            V.append((*(p + np.array([math.cos(yaw_t), math.sin(yaw_t)]) * 0.4 * math.cos(a)), zc + 0.4 * math.sin(a)))
        ctx.mesh(V, [(0, 1, 2)], red, coll=coll, name="sign")
    else:   # rectangular plate: P or a street name
        txt = "P" if kind == "p" else str(rng.choice(["HIGH ST", "MILL LANE", "KING ST", "STATION RD", "CHURCH ST",
                                                        "MARKET ST", "RUE DU PORT", "HAUPTSTR.", "VIA ROMA",
                                                        "ELM AVE", "5TH AVE", "WATER ST"]))
        w = 0.3 if kind == "p" else 0.12 * len(txt) + 0.2
        hh = 0.3 if kind == "p" else 0.22
        m = blue if kind == "p" else (white if rng.random() < 0.6 else pal.pick(pal.signs[:4]))
        ctx.box((front[0], front[1], zc + 0.1), (w / 2, 0.01, hh / 2), m, yaw=yaw_t, coll=coll, name="sign")
        tm = white if m is not white else black
        text_obj(ctx, txt, (*(front + n * 0.012), zc + 0.1 - 0.35 * (0.8 * hh)), yaw_t, 0.8 * hh, tm, coll=coll,
                 bold=0.004 if kind == "p" else 0.0)
    ctx.claim_disc((x, y), 0.12, kind="sign")


POSTER_WORDS = ["CONCERT", "LIVE MUSIC", "FESTIVAL", "SALE", "OPEN AIR", "EXHIBITION", "CIRCUS", "NEW ALBUM",
                "COMING SOON", "FOR LEASE", "VOTE", "CINEMA", "JAZZ NIGHT", "MARKET", "THEATRE", "RAVE",
                "SUMMER FEST", "TOUR 2026", "CLOSING DOWN", "POST NO BILLS", "OPENING", "COMEDY"]
POSTER_COLOURS = [(0.85, 0.15, 0.1), (0.95, 0.8, 0.1), (0.1, 0.3, 0.7), (0.05, 0.05, 0.05), (0.92, 0.9, 0.85),
                  (0.1, 0.55, 0.3), (0.9, 0.45, 0.1), (0.55, 0.1, 0.45), (0.2, 0.7, 0.8), (0.95, 0.5, 0.6)]


def posters(ctx, pal, max_groups=6):
    """Fly-posted bills on the solid stretches of the patch's walls (both
    faces, 4 mm proud, clear of every opening by 0.3 m)."""
    rng, lk = ctx.rng, ctx.looks
    made = 0
    for p in ctx.spec["primitives"]:
        if p["kind"] != "wall" or made >= max_groups:
            continue
        p0, p1 = np.asarray(p["p0"], float), np.asarray(p["p1"], float)
        L = float(np.linalg.norm(p1 - p0))
        d = (p1 - p0) / L
        nrm = np.array([-d[1], d[0]])
        yaw = math.atan2(d[1], d[0])
        t = float(p["t"])
        blocked = [(s - w / 2 - 0.3, s + w / 2 + 0.3) for s, w, zb, zt in p.get("slots", [])]
        for _ in range(int(rng.integers(1, 4))):
            n = int(rng.integers(1, 5))
            w, h = [(0.6, 0.85), (0.84, 1.19), (1.19, 1.68), (0.5, 0.7)][int(rng.integers(4))]
            span = n * (w + 0.01)
            if L < span + 0.4:
                continue
            s0 = float(rng.uniform(0.2, L - span - 0.2))
            if any(not (s0 + span < a or s0 > b) for a, b in blocked):
                continue
            z0 = float(rng.uniform(0.7, max(0.75, min(p["z1"] - 0.2, 2.9) - h)))
            if z0 + h > p["z1"] - 0.05:
                continue
            side = float(rng.choice([1.0, -1.0]))
            col = POSTER_COLOURS[int(rng.integers(len(POSTER_COLOURS)))]
            bg = lk.paint(col, rough=0.5, name="poster")
            ink = lk.paint((0.03, 0.03, 0.03) if sum(col) > 1.6 else (0.95, 0.95, 0.92), rough=0.5, name="ink")
            band = lk.paint(POSTER_COLOURS[int(rng.integers(len(POSTER_COLOURS)))], rough=0.5, name="poster_band")
            word = POSTER_WORDS[int(rng.integers(len(POSTER_WORDS)))]
            for k in range(n):
                s = s0 + k * (w + 0.01) + w / 2
                c = p0 + d * s + nrm * side * (t / 2 + 0.003)
                ctx.box((c[0], c[1], z0 + h / 2), (w / 2, 0.002, h / 2), bg, yaw=yaw, name="poster")
                c2 = p0 + d * s + nrm * side * (t / 2 + 0.0055)
                ctx.box((c2[0], c2[1], z0 + h * 0.3), (w / 2 - 0.04, 0.0005, h * 0.12), band, yaw=yaw, name="poster")
                c3 = p0 + d * s + nrm * side * (t / 2 + 0.006)
                tyaw = yaw if side < 0 else yaw + math.pi
                sz = min(0.16 * h, 0.9 * w / max(3, len(word)) * 1.6)
                text_obj(ctx, word, (c3[0], c3[1], z0 + h * 0.72), tyaw, sz, ink, depth=0.001, coll="structure",
                         bold=0.004)
            made += 1
    return made
