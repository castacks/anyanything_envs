#!/usr/bin/env python3
"""Fetch a curated slice of Objaverse into the v2 asset library: realistic,
textured, real-world-scaled street/port/farm/park/site objects that Poly
Haven lacks (cars, vans, trucks, buses, forklifts, tractors, diggers, bikes,
traffic lights, signs, cones, skips, trolleys, playground kit, benches, ...).

    PY=/mnt/data/blender/objaverse_venv/bin/python        # objaverse trimesh pillow numpy
    $PY blender/fetch_objaverse.py index                  # metadata -> compact index (once)
    $PY blender/fetch_objaverse.py select                 # candidates per category (explore)
    $PY blender/fetch_objaverse.py prescreen              # Sketchfab-thumbnail sheets of the candidates
    $PY blender/fetch_objaverse.py fetch [--candidates]   # download + normalise -> library_objaverse.json
    blender -b --factory-startup -P blender/inspect_assets.py -- --root R --library library_objaverse.json \\
        --catalog-name catalog_objaverse --keys-prefix obj: --thumbs --threads 4 --shard i/3
    $PY blender/inspect_assets.py --merge --root R --library library_objaverse.json --keys-prefix obj: \\
        --catalog-name catalog_objaverse --sheets-dir catalog/objaverse_sheets --sheet-by spec
    $PY blender/fetch_objaverse.py finalise               # junk pieces (stray lights, decals) excluded

Growing the list: ``select --tag r3 --skip-seen [--specs ...] [--licences ...]``,
``prescreen --tag r3`` (numbered sheets in ``<cache>/prescreen_r3/``), write a
keep file ``{"tag": "r3", "keep": "0,4,7-9", "recat": "12:truck"}``, then
``fetch --candidates --keep-file k1.json k2.json ...`` (unreviewed entries are
written with ``exclude``), inspect, look, and paste the survivors into CURATED.

Reproducible result: ``CURATED`` below is the reviewed list (uid -> spec
name) that survived two visual passes (Sketchfab thumbnails, then our own
Cycles thumbnails of the normalised import). ``fetch`` without
``--candidates`` downloads exactly those; ``select``/``prescreen`` are how
they were found and are only needed to grow the list.

Selection (``select``): an object is a candidate for a spec if LVIS put it in
one of the spec's LVIS categories or its name/tags match the spec's keywords;
it must be downloadable, licensed ``--licences`` (default cc0/by/by-sa; the
second round took by-nc/by-nc-sa for thin categories, never ND), have a textured glb of <= 80 MB with 5k-400k faces, and carry none of
the cartoon/low-poly/pack/scene words. Candidates are ranked by likes, views,
staff pick and realism tags (pbr, photogrammetry, scan, realistic).

Normalisation (``fetch``): glbs are in arbitrary units, Y-up. The glTF
importer turns them Z-up, so the Blender box is (x, -z, y) of the glTF one.
Each spec says which dimension to measure (``len`` = largest horizontal side,
``h`` = height) and the real-world range it must match; the target within
the range is fixed per uid (hash), and ``scale`` = target / measured. A spec
also bounds height / largest horizontal side, which rejects objects lying on
their side, flat sign plates without a post, and whole scenes.

Clean-up (``fetch``, Blender on CPU): a file with a baked ground / shadow plane
(few faces, flat, wider than the model) is re-exported without it, and a file
whose sub-meshes sit under non-axis-aligned rotations has every transform baked
into its vertices under one root empty -- otherwise Blender's bound-box-based
sizes (inspect_assets, Library.load) are loose and the model floats above z = 0.
The entry's ``cleaned`` field records which; other files are the original glb.

Writes ``<root>/library_objaverse.json`` (``{"items": {"obj:<uid>": {...}}}``,
same entry format as ``library.json`` plus ``scale``) and copies each glb to
``<root>/objaverse/<uid>/<uid>.glb`` (one directory per file, so a render
node fetching ``files_used`` paths gets only that model). Entries not in
``CURATED`` (a ``--candidates`` run) carry ``"exclude": true`` so kits skip them
until reviewed. ``finalise`` edits ``<root>/catalog_objaverse.json`` written by
the inspect merge (junk pieces excluded).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.request

CACHE = Path("/mnt/data/blender/objaverse_cache")
ROOT = Path("/mnt/data/blender/assets")
HF = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"
LICENCE_NAMES = {"cc0": "CC0-1.0", "by": "CC-BY-4.0", "by-sa": "CC-BY-SA-4.0", "by-nc": "CC-BY-NC-4.0",
                 "by-nc-sa": "CC-BY-NC-SA-4.0", "by-nd": "CC-BY-ND-4.0", "by-nc-nd": "CC-BY-NC-ND-4.0"}


def S(name, cats, lvis=(), kw=(), neg=(), measure="len", rng=(1.0, 1.0), hl=(0.0, 99.0), cap=12, alt=None):
    """One target category. ``kw`` are regexes matched on the name (strong) and
    tags (weak); ``neg`` rejects on name+tags; ``measure``/``rng`` the real
    size in metres; ``hl`` the allowed height / largest-horizontal ratio;
    ``cap`` how many to keep at most; ``alt`` = [(hl_lo, hl_hi, measure, rng)]
    re-targets by shape (a 40 ft vs a 20 ft container)."""
    return dict(name=name, cats=list(cats), lvis=list(lvis), kw=list(kw), neg=list(neg), measure=measure,
                rng=rng, hl=hl, cap=cap, alt=alt or [])


SPECS = [
    # --- vehicles ---------------------------------------------------------------------------
    S("car", ["vehicles", "car"], ["car_(automobile)", "cab_(taxi)", "police_cruiser", "convertible_(automobile)",
                                   "motor_vehicle"],
      [r"\bcar\b", r"sedan", r"hatchback", r"\bsuv\b", r"coupe", r"\btaxi\b", r"estate car", r"station wagon",
       r"crossover", r"police car", r"toyota|honda|volkswagen|\bvw\b|ford|chevrolet|nissan|hyundai|kia|mazda|"
       r"renault|peugeot|audi|bmw|mercedes|skoda|volvo|subaru|lada|fiat|opel|tesla|citroen|dacia|lexus"],
      [r"race|racing|\bf1\b|formula|rally|nascar|drift|tuning|lambo|ferrari|bugatti|mclaren|pagani|koenig|hot ?rod|"
       r"interior|engine|wheel\b|rim\b|tire|tyre|seat|dashboard|steering|wreck|burn|destroy|crash|rusty|"
       r"cockpit|kart|monster|futur|cyber|concept|sci-?fi|batmobile|transformer|toy|carriage|cable car|tram|"
       r"train|wagon wheel|limo|truck|bus\b|van\b|pickup"],
      "len", (4.1, 4.8), (0.18, 0.55), cap=45),
    S("van", ["vehicles", "van"], ["minivan"],
      [r"\bvan\b", r"minivan", r"transit", r"sprinter", r"\bcaddy\b", r"panel van", r"delivery van", r"minibus",
       r"camper ?van", r"kombi", r"\bt[1-6]\b"],
      [r"interior|wreck|burn|destroy|toy|cartoon|futur|concept|ice cream|van gogh|caravan|trailer"],
      "len", (4.7, 5.6), (0.32, 0.62), cap=18),
    S("pickup", ["vehicles", "pickup"], ["pickup_truck"],
      [r"pick-?up", r"\bute\b", r"hilux", r"f-?150", r"ranger", r"tacoma", r"silverado", r"\bram 1500"],
      [r"interior|wreck|burn|destroy|toy|monster|lifted|futur|concept|6x6|rally"],
      "len", (5.0, 5.8), (0.28, 0.58), cap=16),
    S("jeep", ["vehicles", "car", "suv"], ["jeep"],
      [r"\bjeep\b", r"wrangler", r"land ?rover", r"defender", r"land cruiser", r"\buaz\b", r"niva", r"g-?class",
       r"4x4", r"off-?road"],
      [r"interior|wreck|burn|toy|military|army|gun|monster|futur|concept"],
      "len", (4.0, 4.9), (0.3, 0.65), cap=12),
    S("truck", ["vehicles", "truck"], ["truck", "garbage_truck", "tow_truck", "fire_engine", "trailer_truck"],
      [r"\btruck\b", r"\blorry\b", r"box truck", r"garbage truck", r"dump truck", r"tipper", r"fire truck",
       r"fire engine", r"tow truck", r"semi\b", r"tractor unit", r"scania|volvo fh|\bman tg|kamaz|iveco|daf\b|"
       r"freightliner|kenworth|peterbilt|isuzu|hino"],
      [r"toy|cartoon|monster|interior|wreck|burn|futur|concept|pickup|pick-up|food truck|ice cream|military|army"],
      "len", (7.0, 9.5), (0.25, 0.72), cap=30,
      alt=[(0.0, 0.25, "len", (15.0, 16.5))]),           # a tractor unit with its trailer
    S("bus", ["vehicles", "bus"], ["bus_(vehicle)", "school_bus"],
      [r"\bbus\b", r"coach", r"school bus", r"city bus", r"double[- ]?decker", r"trolleybus", r"\bikarus\b"],
      [r"toy|cartoon|interior|wreck|burn|futur|concept|bus stop|busstop|bus shelter|minibus|usb|\bpc\b"],
      "len", (10.5, 12.5), (0.2, 0.46), cap=20,
      alt=[(0.3, 0.46, "len", (10.0, 11.5)), (0.46, 0.68, "len", (8.5, 11.0))]),   # double-decker
    S("ambulance", ["vehicles", "van", "ambulance"], ["ambulance"], [r"ambulance"],
      [r"toy|cartoon|interior|helicopter|futur"], "len", (5.5, 6.6), (0.3, 0.65), cap=6),
    S("camper", ["vehicles", "camper"], ["camper_(vehicle)"],
      [r"camper", r"motorhome", r"\brv\b", r"caravan", r"travel trailer"],
      [r"toy|cartoon|interior|futur"], "len", (5.5, 7.5), (0.3, 0.62), cap=8),
    S("trailer", ["vehicles", "trailer"], [], [r"utility trailer", r"car trailer", r"cargo trailer",
                                                r"box trailer", r"flatbed trailer", r"\btrailer\b"],
      [r"toy|cartoon|interior|truck|semi|movie|film|park"], "len", (3.0, 4.5), (0.15, 0.6), cap=6),
    S("motorcycle", ["vehicles", "motorcycle"], ["motorcycle", "dirt_bike", "motor_scooter"],
      [r"motorcycle", r"motorbike", r"\bscooter\b", r"\bvespa\b", r"moped", r"dirt ?bike", r"harley|ducati|yamaha|"
       r"kawasaki|suzuki|honda cb|triumph|bmw r"],
      [r"toy|cartoon|kick ?scooter|electric scooter|futur|concept|sci-?fi|cyber|helmet|rider|biker|engine\b"],
      "len", (1.8, 2.2), (0.35, 0.85), cap=16),
    S("bicycle", ["vehicles", "bicycle"], ["bicycle"],
      [r"bicycle", r"\bbike\b", r"mountain bike", r"road bike", r"\bbmx\b", r"city bike"],
      [r"toy|cartoon|motor|dirt|exercise|stationary|spin|futur|sci-?fi|tricycle|rack|stand|parking|wheel\b"],
      "len", (1.65, 1.85), (0.45, 0.9), cap=14),
    S("golfcart", ["vehicles", "golf cart"], ["golfcart"], [r"golf ?cart", r"buggy"],
      [r"toy|cartoon|baby|beach|dune"], "len", (2.4, 2.9), (0.5, 0.9), cap=4),
    # --- machinery --------------------------------------------------------------------------
    S("forklift", ["machinery", "forklift"], ["forklift"], [r"fork ?lift", r"reach truck", r"pallet truck",
                                                             r"telehandler"],
      [r"toy|cartoon|futur|sci-?fi"], "len", (2.3, 3.2), (0.6, 1.2), cap=12,
      alt=[(0.2, 0.6, "len", (4.5, 6.0))]),               # telehandler
    S("tractor", ["machinery", "tractor"], ["tractor_(farm_equipment)"],
      [r"tractor", r"john deere", r"massey", r"fendt", r"new holland", r"kubota", r"\bclaas\b", r"\bbelarus\b",
       r"\bmtz\b"],
      [r"toy|cartoon|trailer|tractor unit|semi|\btruck\b|lawn|mower|futur"], "len", (3.6, 4.8), (0.35, 0.95),
      cap=14),
    S("combine", ["machinery", "combine harvester"], [], [r"combine harvester", r"\bharvester\b"],
      [r"toy|cartoon|forest|timber|futur"], "len", (7.5, 9.5), (0.35, 0.6), cap=4),
    S("excavator", ["machinery", "excavator"], ["power_shovel"],
      [r"excavator", r"\bdigger\b", r"backhoe", r"\bjcb\b", r"komatsu|hitachi zx|caterpillar 3|cat 3[0-9]{2}|"
       r"liebherr r"],
      [r"toy|cartoon|futur|sci-?fi|bucket\b|arm\b"], "len", (6.0, 9.0), (0.25, 0.75), cap=14,
      alt=[(0.75, 1.3, "h", (5.5, 7.0))]),               # boom raised: the boom tip is the top
    S("bulldozer", ["machinery", "bulldozer"], ["bulldozer"], [r"bulldozer", r"\bdozer\b", r"crawler tractor"],
      [r"toy|cartoon|futur|sci-?fi"], "len", (5.0, 6.8), (0.45, 0.75), cap=8),
    S("loader", ["machinery", "loader"], [], [r"wheel loader", r"front loader", r"skid steer", r"bobcat",
                                                r"\bloader\b"],
      [r"toy|cartoon|futur|gun|ammo|magazine|log loader"], "len", (5.0, 7.5), (0.35, 0.65), cap=8,
      alt=[(0.55, 1.15, "len", (2.8, 3.6))]),            # skid steer
    S("roller", ["machinery", "road roller"], [], [r"road roller", r"steam roller", r"compactor", r"asphalt roller"],
      [r"toy|cartoon"], "len", (3.5, 5.5), (0.45, 0.8), cap=4),
    S("concrete mixer", ["machinery", "concrete mixer"], [], [r"concrete mixer", r"cement mixer",
                                                              r"mixer truck"],
      [r"toy|cartoon"], "len", (8.0, 9.5), (0.3, 0.55), cap=6,
      alt=[(0.55, 1.6, "h", (1.2, 1.5))]),               # the small drum mixer on legs
    S("generator", ["machinery", "generator"], ["generator"],
      [r"generator", r"genset", r"compressor", r"air compressor"],
      [r"toy|cartoon|wind|steam|van de graaff|electrostatic|sci-?fi|futur|nuclear|shield"],
      "len", (0.7, 1.1), (0.55, 1.2), cap=10,
      alt=[(0.3, 0.55, "len", (2.5, 4.0))]),             # a towed / skid genset
    # --- street furniture ---------------------------------------------------------------------
    S("traffic light", ["street furniture", "traffic light"], ["traffic_light"],
      [r"traffic ?light", r"traffic signal", r"stop ?light", r"semaphore", r"svetofor"],
      [r"toy|cartoon|icon|emoji"], "h", (3.0, 5.0), (1.8, 99.0), cap=14,
      alt=[(0.6, 1.8, "h", (5.5, 7.0))]),                # mast-arm signal
    S("road sign", ["street furniture", "road sign"], ["street_sign", "stop_sign"],
      [r"road ?sign", r"street ?sign", r"traffic ?sign", r"stop ?sign", r"yield sign", r"speed limit",
       r"warning sign", r"sign ?post"],
      [r"toy|cartoon|neon|shop sign|billboard|icon"], "h", (2.2, 3.0), (1.8, 99.0), cap=18),
    S("street light", ["street furniture", "street light"], ["lamppost", "streetlight"],
      [r"street ?light", r"street ?lamp", r"lamp ?post", r"light ?pole", r"lantern pole"],
      [r"toy|cartoon|table|desk|interior|bulb\b|wall lamp|fantasy"], "h", (4.5, 8.0), (1.0, 99.0), cap=12),
    S("utility pole", ["street furniture", "utility pole"], ["telephone_pole"],
      [r"telephone pole", r"utility pole", r"power pole", r"electric pole", r"power line pole"],
      [r"toy|cartoon|tower|pylon"], "h", (8.0, 11.0), (0.5, 99.0), cap=8),
    S("traffic cone", ["street furniture", "traffic cone"], ["cone"],
      [r"traffic ?cone", r"road cone", r"safety cone", r"\bpylon\b", r"construction cone"],
      [r"ice ?cream|pine|toy|cartoon|waffle|electric|power|tower"], "h", (0.5, 0.75), (1.1, 3.0), cap=10),
    S("road barrier", ["street furniture", "barrier"], [],
      [r"jersey barrier", r"road barrier", r"traffic barrier", r"concrete barrier", r"crowd barrier",
       r"crash barrier", r"police barrier", r"construction barrier", r"water barrier", r"barricade",
       r"\bbarrier\b"],
      [r"toy|cartoon|sci-?fi|futur|energy|force|shield|reef"], "h", (0.8, 1.15), (0.2, 1.0), cap=14),
    S("bollard", ["street furniture", "bollard"], [], [r"bollard"], [r"toy|cartoon|mooring|ship|boat"],
      "h", (0.8, 1.1), (0.9, 8.0), cap=6),
    S("bench", ["street furniture", "bench"], ["bench"],
      [r"\bbench\b", r"park bench", r"street bench"],
      [r"work ?bench|workbench|weight|gym|press|piano|pew|church|toy|cartoon|tool|table"],
      "len", (1.5, 1.9), (0.3, 0.75), cap=18),
    S("picnic table", ["street furniture", "picnic table"], [], [r"picnic table", r"picnic bench",
                                                                  r"beer table", r"garden table"],
      [r"toy|cartoon|interior"], "len", (1.8, 2.2), (0.25, 0.55), cap=8),
    S("trash can", ["street furniture", "trash can"], ["trash_can"],
      [r"trash ?can", r"litter ?bin", r"garbage ?can", r"rubbish bin", r"waste ?bin", r"\bbin\b",
       r"wheelie bin", r"recycl(e|ing) bin", r"dust ?bin"],
      [r"toy|cartoon|office|desk|kitchen|pedal|icon|dumpster|skip\b"], "h", (0.9, 1.15), (0.75, 2.6), cap=14),
    S("dumpster", ["street furniture", "dumpster"], ["dumpster"],
      [r"dumpster", r"skip ?bin", r"\bskip\b", r"waste container", r"garbage container", r"trash container",
       r"refuse container", r"roll-?off", r"recycling container"],
      [r"toy|cartoon|fire|burn|diorama"], "len", (1.8, 2.3), (0.45, 0.85), cap=14,
      alt=[(0.2, 0.45, "len", (3.4, 4.2)), (0.85, 1.45, "h", (1.2, 1.45))]),   # builder's skip / 4-wheel bin
    S("fire hydrant", ["street furniture", "fire hydrant"], ["fireplug"], [r"fire ?hydrant", r"\bhydrant\b"],
      [r"toy|cartoon|pokemon"], "h", (0.6, 0.9), (0.8, 3.2), cap=10),
    S("mailbox", ["street furniture", "mailbox"], ["postbox_(public)", "mailbox_(at_home)"],
      [r"post ?box", r"mail ?box", r"pillar box", r"letter ?box"],
      [r"toy|cartoon|wall|icon|email|minecraft"], "h", (1.15, 1.6), (1.1, 4.0), cap=8),
    S("parking meter", ["street furniture", "parking meter"], ["parking_meter"], [r"parking meter",
                                                                                   r"parking machine",
                                                                                   r"ticket machine"],
      [r"toy|cartoon"], "h", (1.3, 1.8), (1.4, 8.0), cap=6),
    S("phone booth", ["street furniture", "phone booth"], ["telephone_booth"],
      [r"phone ?booth", r"telephone ?box", r"phone ?box", r"telephone ?booth", r"payphone"],
      [r"toy|cartoon|tardis|doctor who|police box"], "h", (2.2, 2.6), (1.0, 4.0), cap=8),
    S("vending machine", ["street furniture", "vending machine"], ["vending_machine"],
      [r"vending machine", r"vending", r"soda machine", r"drinks machine"],
      [r"toy|cartoon|futur|sci-?fi"], "h", (1.75, 1.95), (1.2, 3.0), cap=8),
    S("bus stop", ["street furniture", "bus stop"], [], [r"bus ?stop", r"bus shelter", r"tram stop",
                                                          r"bus station shelter"],
      [r"toy|cartoon|station\b(?! shelter)"], "h", (2.4, 2.9), (0.35, 1.2), cap=8),
    S("newsstand", ["street furniture", "kiosk"], ["newsstand"], [r"kiosk", r"newsstand", r"news stand"],
      [r"toy|cartoon|interior|screen|touch|atm"], "h", (2.4, 3.2), (0.5, 1.6), cap=6),
    S("bike rack", ["street furniture", "bike rack"], [], [r"bike rack", r"bicycle rack", r"bicycle parking",
                                                            r"bike stand"],
      [r"toy|cartoon|car|roof"], "len", (1.8, 3.0), (0.25, 0.6), cap=4),
    S("shopping trolley", ["street furniture", "shopping trolley"], ["shopping_cart"],
      [r"shopping (cart|trolley)", r"supermarket (cart|trolley)", r"grocery cart"],
      [r"toy|cartoon|basket"], "len", (0.9, 1.05), (0.9, 1.3), cap=8),
    S("billboard", ["street furniture", "billboard"], ["billboard"], [r"billboard", r"advertising (board|column)"],
      [r"toy|cartoon|interior|poster\b"], "h", (4.0, 8.0), (0.4, 2.5), cap=6),
    # --- site / yard ----------------------------------------------------------------------------
    S("pallet", ["site", "pallet"], [], [r"\bpallet\b", r"euro ?pallet", r"pallets"],
      [r"toy|cartoon|palette|truck|jack|colou?r|paint|forklift"], "len", (1.15, 1.25), (0.08, 1.4), cap=10),
    S("ibc tank", ["site", "ibc tank"], [], [r"\bibc\b", r"intermediate bulk", r"ibc tank", r"water tote",
                                             r"1000 ?l(iter|itre)? tank"],
      [r"toy|cartoon"], "len", (1.15, 1.25), (0.8, 1.2), cap=6),
    S("barrel", ["site", "barrel"], ["barrel"], [r"oil drum", r"\bdrum\b", r"barrel", r"\bdrums\b"],
      [r"toy|cartoon|wine|wood|whisk|beer|gun|drum ?kit|drums? set|drumstick|snare|music|pirate|explosive|"
       r"cannon|rifle|keg"], "h", (0.85, 0.95), (1.0, 1.8), cap=8),
    S("shipping container", ["site", "shipping container"], [],
      [r"shipping container", r"cargo container", r"\bisotank\b", r"container 20|container 40|20 ?ft|40 ?ft"],
      [r"toy|cartoon|house|home|office|cafe|shop|bar\b|ship\b|vessel|crane"], "len", (5.9, 6.2), (0.35, 0.5),
      cap=8, alt=[(0.15, 0.35, "len", (12.0, 12.3))]),
    S("portable toilet", ["site", "portable toilet"], [],
      [r"portable toilet", r"porta ?-?potty", r"porta ?-?loo", r"portaloo", r"chemical toilet", r"outhouse",
       r"toi ?toi", r"portable restroom", r"port-a-john"],
      [r"toy|cartoon"], "h", (2.2, 2.45), (0.8, 2.4), cap=6),
    S("site fence", ["site", "fence"], [], [r"construction fence", r"heras", r"temporary fence", r"mesh fence",
                                            r"fence panel"],
      [r"toy|cartoon|garden|picket|wood"], "len", (3.3, 3.6), (0.4, 0.75), cap=4),
    S("scaffolding", ["site", "scaffolding"], [], [r"scaffold", r"scaffolding", r"scaffold tower"],
      [r"toy|cartoon|building|house|gallows"], "h", (4.0, 6.5), (0.25, 5.0), cap=6),
    S("ladder", ["site", "ladder"], ["ladder", "stepladder"], [r"ladder", r"step ?ladder"],
      [r"toy|cartoon|snake|rope|ship|chute"], "h", (1.8, 3.0), (1.3, 12.0), cap=8,
      alt=[(0.6, 1.3, "h", (1.2, 1.8))]),                # an open step ladder
    S("wheelbarrow", ["site", "wheelbarrow"], ["barrow"], [r"wheel ?barrow"], [r"toy|cartoon|flower|planter"],
      "len", (1.4, 1.65), (0.3, 0.75), cap=8),
    S("hand truck", ["site", "hand truck"], ["handcart"], [r"hand ?truck", r"sack truck", r"dolly", r"hand cart"],
      [r"toy|cartoon|camera|film"], "h", (1.1, 1.4), (1.2, 3.2), cap=4),
    S("gas cylinder", ["site", "gas cylinder"], [], [r"gas cylinder", r"gas bottle", r"propane tank",
                                                      r"oxygen tank", r"lpg"],
      [r"toy|cartoon|scuba|diving"], "h", (0.6, 1.1), (1.1, 6.0), cap=8),
    S("tank", ["site", "tank"], ["tank_(storage_vessel)"], [r"storage tank", r"water tank", r"fuel tank",
                                                             r"oil tank", r"silo"],
      [r"toy|cartoon|army|military|fish|aquarium|toilet|scuba|motorcycle|car\b|gas tank"],
      "h", (2.5, 5.0), (0.4, 3.5), cap=8,
      alt=[(3.5, 8.0, "h", (10.0, 16.0))]),             # farm silo
    # --- farm -----------------------------------------------------------------------------------
    S("hay bale", ["farm", "hay bale"], [], [r"hay ?bale", r"straw ?bale", r"round bale", r"haystack",
                                             r"hay stack", r"\bbale\b"],
      [r"toy|cartoon"], "len", (1.2, 1.8), (0.4, 1.3), cap=8),
    S("water trough", ["farm", "water trough"], [], [r"water trough", r"\btrough\b", r"cattle trough"],
      [r"toy|cartoon|pig|food"], "len", (1.6, 2.8), (0.2, 0.6), cap=4),
    S("farm trailer", ["farm", "farm trailer"], ["cart", "wagon", "plow_(farm_equipment)"],
      [r"farm trailer", r"tipping trailer", r"hay wagon", r"\bplough\b", r"\bplow\b", r"harrow", r"seeder",
       r"farm cart", r"wagon"],
      [r"toy|cartoon|snow|horse|covered|western|station|wheel\b|train|rail"], "len", (4.0, 6.5), (0.25, 0.8),
      cap=6),
    # --- park / garden --------------------------------------------------------------------------
    S("playground", ["playground", "playground"], ["slide"],
      [r"playground", r"play ?ground", r"swing ?set", r"\bswings?\b", r"\bslide\b", r"climbing frame",
       r"jungle gym", r"seesaw", r"see-saw", r"merry-go-round", r"carousel", r"monkey bars", r"sandpit"],
      [r"toy|cartoon|powerpoint|presentation|microscope|water ?slide|aqua|trombone|guitar|slider|golf|jazz"],
      "h", (2.0, 3.2), (0.2, 1.4), cap=16),
    S("trampoline", ["playground", "trampoline"], ["trampoline"], [r"trampoline"], [r"toy|cartoon"],
      "len", (3.0, 4.3), (0.12, 0.9), cap=4),
    S("garden furniture", ["garden furniture", "garden set"], ["deck_chair", "chaise_longue", "grill"],
      [r"garden (chair|furniture|set|table)", r"patio", r"deck ?chair", r"\bbbq\b", r"barbecue", r"\bgrill\b",
       r"adirondack"],
      [r"toy|cartoon|cocktail|interior|kitchen"], "h", (0.85, 1.05), (0.12, 2.0), cap=10),
    S("parasol", ["garden furniture", "parasol"], ["parasol", "umbrella"], [r"parasol", r"sunshade", r"umbrella table"],
      [r"toy|cartoon|cocktail|rain|handheld|closed|folded"], "h", (2.2, 2.6), (0.5, 2.0), cap=4),
    S("gazebo", ["garden furniture", "gazebo"], [], [r"gazebo", r"pergola", r"pavilion"],
      [r"toy|cartoon|interior"], "h", (2.8, 3.6), (0.3, 1.8), cap=6),
    S("roundabout", ["playground", "roundabout"], [], [r"roundabout", r"merry-go-round", r"playground carousel"],
      [r"toy|cartoon|road|traffic"], "len", (1.8, 2.5), (0.2, 0.8), cap=4),
    S("seesaw", ["playground", "seesaw"], [], [r"see-?saw", r"teeter"], [r"toy|cartoon"], "len", (3.0, 4.0),
      (0.08, 0.5), cap=3),
    S("tree", ["plants", "tree"], ["Christmas_tree"],
      [r"\btree\b", r"\boak\b", r"maple", r"birch", r"linden", r"beech", r"\belm\b", r"willow", r"poplar",
       r"chestnut", r"plane tree", r"sycamore", r"ash tree", r"deciduous", r"apple tree", r"cherry tree"],
      [r"toy|cartoon|christmas|xmas|family tree|tree ?house|stump|log\b|trunk|dead|bonsai|palm|lowpoly|"
       r"stylized|forest\b|pack|leaf\b|branch|skill|fractal|decision|minecraft|sapling|potted|pot\b"],
      "h", (6.0, 14.0), (0.7, 5.0), cap=16),
    S("bush", ["plants", "bush"], [], [r"\bbush\b", r"\bshrub\b", r"hedge", r"boxwood", r"\bbushes\b"],
      [r"toy|cartoon|pack|stylized|potted|pot\b|bushcraft|george|flower"], "h", (0.9, 2.2), (0.35, 1.6), cap=10),
    # --- port -----------------------------------------------------------------------------------
    S("small boat", ["boats", "small boat"], ["dinghy", "canoe", "kayak", "rowboat", "river_boat"],
      [r"dinghy", r"rowboat", r"row boat", r"rowing boat", r"canoe", r"kayak", r"fishing boat", r"motorboat",
       r"motor boat", r"skiff", r"\brib\b", r"inflatable boat", r"speedboat"],
      [r"toy|cartoon|paper|origami|interior|ship|sail|pirate|viking|galleon"], "len", (3.5, 6.5), (0.1, 0.5),
      cap=10),
    S("buoy", ["boats", "buoy"], ["buoy"], [r"\bbuoy\b", r"navigation buoy", r"mooring buoy"],
      [r"toy|cartoon|life ?buoy|ring|lifebuoy"], "h", (1.5, 3.0), (0.8, 4.0), cap=4),
]

#: never a candidate: stylised, toy, part or whole-scene words (name + tags + categories)
GLOBAL_NEG = re.compile(
    r"low[\s_-]?poly|lowpoly|cartoon|\btoon|stylized|stylised|cute|chibi|voxel|minecraft|roblox|lego|\btoy\b|toys|"
    r"\bkids?\b|pixel|hand[\s_-]?painted|\bclay\b|paper ?craft|origami|\bsketch\b|\bwip\b|3d[\s_-]?print|"
    r"printable|diorama|"
    r"\bpack\b|collection|\bkit\b|asset ?pack|\bset of|\bscene\b|environment|\bcity\b|\bstreet scene|"
    r"crossy|isometric|\banime\b|fortnite|\bgta\b|simpsons|pokemon|lowres|game ?boy|n64|ps1|psx|retro game|8-?bit|"
    r"fan ?art|cel[\s_-]?shad|\bmini\b|miniature|model kit|scale model|rc car|\brc\b|hot ?wheels|matchbox|"
    r"low[\s_-]?res|papermodel|\bai[\s_-]?generated|\bmeshy\b|\btripo\b|\bluma\b|kaedim|wreck|abandoned|post[\s_-]?apo|"
    r"destroyed|zombie|horror|halloween|christmas|xmas|snow ?man|lantern festival",
    re.I)
REAL_TAGS = re.compile(r"realistic|photoreal|\bpbr\b|photogrammetry|\bscan\b|3d ?scan|scanned|substance|"
                       r"high[\s_-]?poly|highpoly|detailed|\b4k\b|\b8k\b|real[\s_-]?world|reality ?capture|"
                       r"agisoft|metashape|polycam", re.I)
SKIP_SF_CATS = {"characters-creatures", "animals-pets", "fashion-style", "food-drink", "music", "people",
                "weapons-military", "art-abstract", "science-technology"}
JUNK_TAGS = re.compile(r"^(blender|blender3d|b3d|maya|3dsmax|3ds|max|cinema4d|c4d|substance|substancepainter|"
                       r"sketchfab|gameready|game-ready|game|gameasset|game-asset|asset|free|freedownload|download|"
                       r"unity|unity3d|unreal|ue4|ue5|zbrush|model|3d|3dmodel|3d-model|obj|fbx|glb|gltf|pbr|"
                       r"textured|texture|lowpoly|highpoly|noai|.*[0-9]{4}.*|.{1,2}|.{25,})$")

#: the reviewed list: uid -> spec name, 2026-09-25. Two selection rounds (round 1: cc0/by/by-sa,
#: 1407 candidates; round 2: thin categories incl. by-nc/by-nc-sa, 404 candidates), a Sketchfab-
#: thumbnail pass (598 kept), the shape gate (585) and a pass over our own Cycles thumbnails of the
#: normalised import (452 kept: scene clutter, baked ground scans, broken boxes, wrong proportions,
#: cartoonish or untextured models removed). ``fetch`` without --candidates downloads exactly these.
CURATED: dict[str, str] = {
    # ambulance (6)
    "1061168f9ec345db81cffb81c17f3649": "ambulance", "3f2709a3d0be486fb809289122786090": "ambulance",
    "5200983a287d4b9e850bd11236629ec4": "ambulance", "a1c2373a25654117a731a1861403f372": "ambulance",
    "eb3da58fb8bd4408a7901823387d29da": "ambulance", "fcc1aed5629849e6bc6d01bd1ca3c7e6": "ambulance",
    # barrel (7)
    "205a7012c31742afb2b81d2cbfe0f2b2": "barrel", "51582c42a9b14526ae7acb33f536d1ac": "barrel",
    "5edfb4a531bb4b28a005786b6af585dc": "barrel", "7628a1b3f3924562b1e45301fb5398c4": "barrel",
    "c9b66b0949f247c4ae78338053650ee2": "barrel", "e5fccac2cc97468f90048944188b1b7a": "barrel",
    "f11413f39ae34684ac6ecbbbed58621c": "barrel",
    # bench (15)
    "07fb4d364c0c4555a14c86e915542378": "bench", "1587e40d772e4985b11628f550bef5bf": "bench",
    "1985d2c6b0d3455789622e6e599f070d": "bench", "220423b220f246b48ea16efa74cab450": "bench",
    "36dc07c8e85c4b859d78ccd4dd60c63d": "bench", "39ee5c499030472ca7460f3b03077135": "bench",
    "411446b9be884eb59c93885a7ccac923": "bench", "507aa43cecdb432ebe078b84d8027ac9": "bench",
    "82cadef8dfb24c0784eefff7be71610c": "bench", "84426d6537ac4cdc837d602ccabe7036": "bench",
    "95d5315c300945afb0abe31de3ab0903": "bench", "a6da0dff0f9b4a5a8e821c40b370a98f": "bench",
    "af6e758c5ca14e3a929798a950cbbced": "bench", "dcb057ffb7e949a7a1ec5523321d965b": "bench",
    "e6c321d47483484d9828d8ce058cdae1": "bench",
    # bicycle (13)
    "17219902f20d4536beaf383774aba844": "bicycle", "197dd882a69846e0825e80566996cbba": "bicycle",
    "223901fe94de4d55b6324e7d7e273469": "bicycle", "3bde1ab67c6448169b604dedad6828d8": "bicycle",
    "43860f8a9f174cb9b9cc15a3f6786fbf": "bicycle", "89fe9b1ca558441b93975fb715044530": "bicycle",
    "99d6dce2e49e4ba185ac108ab044524d": "bicycle", "b43c4f717e904d938dcacfe58c345631": "bicycle",
    "b5a04eef008e4655bbc48fb616aac401": "bicycle", "c10d7de60f3c4df9a3a8f892aacf18cd": "bicycle",
    "e559f6bccecf44a2b73e1da8ffce6b66": "bicycle", "ee74fcdacb694abfb444cbcc5954526a": "bicycle",
    "f256203558c24ea09385a861b917c75a": "bicycle",
    # billboard (1)
    "bc45c0447d474eba8b5e904ddda1c2ff": "billboard",
    # bollard (6)
    "45d9178dd0cc425fa524c552e5ae8c68": "bollard", "8c517075fdaa4c0fb10d081f12e9fde9": "bollard",
    "9568b99fccd540c7983500fee684e06c": "bollard", "cb3dfbc282e74ac5bb0c83e1a6df37b9": "bollard",
    "dc128321d160460297f0c879a06691f1": "bollard", "dfa2985a7e004239838f3319d2cd1798": "bollard",
    # bulldozer (1)
    "b26ce56081304c1988ec7490e439d882": "bulldozer",
    # buoy (1)
    "cdd198814e874379b6cd2c0cc06bc22d": "buoy",
    # bus (8)
    "08d4c1b35f82404e9f6b50e970baadb9": "bus", "21262813724e401b88333d381c4626f7": "bus",
    "4d32ab9c14c3434f913973f8fe90f85e": "bus", "852a63f852584448a49d643ae944506e": "bus",
    "9237e918de35460baf2568e3e3467b5f": "bus", "cc159e95590b49999c6e08793eb0a76a": "bus",
    "ddfd0b9846c24a058874521e9d170bcb": "bus", "fc497f8e331a4a82895a2d8ee62a8ef8": "bus",
    # bus stop (3)
    "367bdb5ca8c745978410c6e3ffd9b910": "bus stop", "90fbd043c83a423598d425bb3a641dfc": "bus stop",
    "fc632b69dc61498ab4b04139b4adf760": "bus stop",
    # bush (3)
    "8db54bf299954daa9ee29b233e923672": "bush", "8e8fd14f7f1546b9bcf2f5cfee5c116c": "bush",
    "cb91cb6149a142a8a196a268dcf36aa8": "bush",
    # camper (5)
    "0a1703bd068546b2af34a5b5e5b0ceca": "camper", "0d1b1ff7d7b54af9b6b16fd0ef349904": "camper",
    "72208a74546e4e3590bd2acf43dc330a": "camper", "a051cd62b8aa480a93ed0aae845ce1d4": "camper",
    "bc096d0ad5c64ac1a48647200fd559c2": "camper",
    # car (44)
    "0789ab2ece9442de94b3c41595e0ecbd": "car", "0e105d035e4f47f2b42ca6a430b9960a": "car",
    "13cc3ffc69964894a2bc94154aed687f": "car", "192011ef37994e8ca2f09cd1d73ca166": "car",
    "1a950b81ee274a0eb8014bd84ba047f5": "car", "219e25d6174745ae8d9c37f53aafea03": "car",
    "2429a825ff1646568e8e6e1453f9c501": "car", "300e85f5786f48b3abd5f8ab3851c3a1": "car",
    "35d0e4c2310344f88823d7dbf955af26": "car", "3a19f19092614c8cb36d1420288abe21": "car",
    "4223fcc87baa4ac58be2910a5367d7d0": "car", "4b36740173d24f73aadda6cb10e7cacc": "car",
    "507ee2e6d6a34a3db7bf8373c70fecff": "car", "52f13d6fd3b84e7e88f127cd15f537e2": "car",
    "5ce1a4963463439892a5fd523486797a": "car", "6048744a250e44c9bc3081ca1825a907": "car",
    "6e9adfb45e3248f6ba3a9ad735b2caab": "car", "75d636ab26b1423ca61ffd0fd758fd0a": "car",
    "8c653e23d1d74a5991c947e8a57887a3": "car", "8eebf01fb1c14f6d88ddd47fa77f5ccb": "car",
    "90e48e1e38474dc3a7c3d5a42c14c4c2": "car", "91270dab9d1a47c7ba7ffaaf8c8b0beb": "car",
    "936854ba7dde46f09a47d355d433808c": "car", "9f30d360cee343efb5a441978ddb57bd": "car",
    "a47c505c928c4706b32231ac4c8f802a": "car", "a73b3cdeddd846209bf7c73d5f04d20f": "car",
    "ab3547e947e2496e843032684c4b6afe": "car", "b8d6c9706f0746628cdf184d268c0e56": "car",
    "b9604dd71c6546ec842702682019492a": "car", "bec38f586d9349ee9c1f295b6f5569cb": "car",
    "bf1fe23e1578417ab11db67dc89fbe32": "car", "c592bf582a1b430daaef753418fab22f": "car",
    "cb6e82e73386448fb3b4e2dd25255a5e": "car", "d76ad5a0495443eda24a9052565ef9a8": "car",
    "d99bb7b0fc0e4e3c9553246679dd067f": "car", "de1ffd344c41481892511f7fd332c136": "car",
    "e0bb09118ab84fe0a0530bd0df2fd74a": "car", "e398ad20e83d42e581ab43438d5b6e49": "car",
    "e5d5f35870344bd4a005ef5599a63313": "car", "f03e5ba5e81b4562958645fe27565f3a": "car",
    "f4eafb342ecb438ba3bff34dbd77cb46": "car", "f809616e219a406a8c2feaf3a00b9fc9": "car",
    "fa6c6113f1e34d9baaff80b5b586a25d": "car", "ff8fb2251dfa4bb9979e7022c5a6666c": "car",
    # combine (1)
    "50093a1f19264d7db76bc09849612e49": "combine",
    # dumpster (8)
    "083444c1a756495896c8c2fe83d54410": "dumpster", "135d33df60304481830d6ddd8c43cc47": "dumpster",
    "73f80ae78964472bbda19328b354fa35": "dumpster", "989ea7f01b924859b3484ee27103e8d9": "dumpster",
    "bc957334d53c4b2aada512f5213cdebf": "dumpster", "bed7cc2208ae481f800a1d8516839be5": "dumpster",
    "cd72f554120b40acae327f879c15f90d": "dumpster", "e8b38df736994cf8b4f7a3a9d2980f6b": "dumpster",
    # excavator (6)
    "5b2d02e1c1c14deba0f9c8d75fff54b6": "excavator", "5d08010ce3ea44cbb57cf8d86a0403f2": "excavator",
    "63cbe44a46784449907d71c4b6953682": "excavator", "852110f8f33a4b68b548f77fab0505f9": "excavator",
    "940258497f9f4eba9710bb359a9e777e": "excavator", "f6356e9dbec14b548f7e9d36911d71bb": "excavator",
    # farm trailer (5)
    "06d400aa4d2343e490d07c23ff661e4c": "farm trailer", "498b84a5ae0943e8bbad6ab2cee6874c": "farm trailer",
    "91fa7ce2ea9e43a69d723cc3bd08f306": "farm trailer", "b18a7d6210ca466f9dd9ceb8e1675a58": "farm trailer",
    "f47e3429f5de4a2f95ee3f0a82a3edd0": "farm trailer",
    # fire hydrant (17)
    "00ac64cd7a7949b486937cf0d264048b": "fire hydrant", "0a83180e31b640a39a83b0da113b00ce": "fire hydrant",
    "0a9b36d36e904aee8b51e978a7c0acfd": "fire hydrant", "4c11f50141b44f34ba3755850d6e804f": "fire hydrant",
    "51c0f9b3d1a7499a8af9558b97f2ba51": "fire hydrant", "6adac8e59c2949ad8e0acb003f49a7d9": "fire hydrant",
    "71f34fffd6bd4f1bbb2278fceba686c8": "fire hydrant", "7f9164e6018c4ec7a3585bbf6e6d70df": "fire hydrant",
    "91acc6e821a74ed3a3c0ae72643958a5": "fire hydrant", "a3cf6b9b70a042fa85474c5a5f025e6e": "fire hydrant",
    "a4ef32d993734d37a8985c36927d4201": "fire hydrant", "d012d05e8bca49928a2944e043d93fb7": "fire hydrant",
    "d021f42ea8c44afe9c90655beaa36d89": "fire hydrant", "d7883c62770349edbfd19eaf72b39765": "fire hydrant",
    "dc9cd29a598343d38eae7ba41d311615": "fire hydrant", "dd7a6d39655e4a189519fb03dc0ef0ad": "fire hydrant",
    "e5a66a7dd19a4c87bcf37c4a91e36d89": "fire hydrant",
    # forklift (9)
    "060f3f8bc7de4e6ca2f348d414702e9d": "forklift", "7f74cf149f434c378816bbc17084323e": "forklift",
    "9ad80a7bd90b483f8dbf0e293251cff7": "forklift", "b7bea04856e1401898250f313131659f": "forklift",
    "bb8b8dab8b4c4fc1ac6af310b70f1b6e": "forklift", "ce8b2c714b2f472bbac58c6eab088ba2": "forklift",
    "d40cae50e04145dd997cdca415cd72ad": "forklift", "d659c366e24643f0b72ad0fe1ef1a60b": "forklift",
    "e79f0768aec6431d8e6f3874e5acc5f2": "forklift",
    # garden furniture (7)
    "261d81ec28d845eba3bb7373768c60c5": "garden furniture", "31d46740ef6a41b18f67808225cef468": "garden furniture",
    "3e6b91194c6348c0bc0c88d8d2f83cfd": "garden furniture", "76d6011abc8848c89f080df54ed4d0f8": "garden furniture",
    "884d585e35eb4f389b098581246ea265": "garden furniture", "bf4c7bcb6d184667822986614a7c5e97": "garden furniture",
    "d15d2c598774442c91b03478a92cc7b5": "garden furniture",
    # gas cylinder (8)
    "4da7e80c3b6a42328d48dd811642db6e": "gas cylinder", "7f6d2d6a4c774bc48e4ae2ecebf8983b": "gas cylinder",
    "83468fa4ed6742ddae1a9a79c764f883": "gas cylinder", "9800ee3d88004696b4a80b0115fee693": "gas cylinder",
    "a68f90323fd44cf7a22c0afe30d6a2f8": "gas cylinder", "a7bfb8c075f84d659c6b7fef5d4243f2": "gas cylinder",
    "e788acc021ac45a598178d24ce196737": "gas cylinder", "f1075a710e554e21b9b678a053e1f9ea": "gas cylinder",
    # gazebo (6)
    "100de2d5f0b14cb0b371295d07308e1c": "gazebo", "341f8e563e84442984f7286f2266fc68": "gazebo",
    "7a5133a271dc4b90a8e24ad315eeef31": "gazebo", "8fd94526f61f41a9b2f5bccbcfb35b9f": "gazebo",
    "ab95dfb3816e4160989f26972c4c7c39": "gazebo", "ac58fa82394e43f895eb64e50ffbd7a7": "gazebo",
    # generator (7)
    "05cb0187751541f195d8b7cf3c4736ef": "generator", "39df7dcfaec04f20b348d729d3cadf01": "generator",
    "423cfdbd612c486aada7bb1bc44695a9": "generator", "4b50a06663fe44079503ca53f7eb399f": "generator",
    "a6c7477cc905448fa16310be52a989c1": "generator", "f34ba6c159e04152ac1883caaca2ae18": "generator",
    "f394f4f7f446474fa65e3217923bdaaf": "generator",
    # golfcart (2)
    "06f92d4d488d4a048360eb19667e231a": "golfcart", "c47ccbe5107d4d8f92b3caaf93f3d9cc": "golfcart",
    # hand truck (1)
    "8e842e659f7c44e1ba8c4f6981b92e2b": "hand truck",
    # hay bale (3)
    "1148b82e71ba4d6e93f0ea5fc23dc13f": "hay bale", "c9699558a44247b88f73b7690ed11b8b": "hay bale",
    "de765f660b5c4f3a891f78d6ba1c4ec5": "hay bale",
    # ibc tank (1)
    "8b80338880564f6e9c97aa517c60ec37": "ibc tank",
    # jeep (12)
    "13c4d0fbbfb542c0b0614459d2f12324": "jeep", "178f22467bae4c729bdcc15dbc7e445d": "jeep",
    "1ab67bad49c7415daa2c763ac476da96": "jeep", "4bfd57c17fcd4404a19208878ae34192": "jeep",
    "4f45aa2e9180437b9687b3d567069ca9": "jeep", "5cd3ea8eb94c4f5597cc086b2df40e58": "jeep",
    "6fc05e37e8174446884981282dbc7a0d": "jeep", "8630ce940c884057a82c9cd39998fd53": "jeep",
    "8dd9a0b3faa84aaa813431b44c716a6d": "jeep", "93fc33be25e34b44b205c5b962d66e61": "jeep",
    "ccb00f0fb0c94408b60e0397c0c9ef52": "jeep", "d414c185293e46eab32bf539f6fff808": "jeep",
    # ladder (5)
    "0ac651eb518a4bc69873b117cc7e3dd2": "ladder", "12304e30fc9d4c1288f6634dd80a2bb8": "ladder",
    "3770d46e9ea8408d87965bd52da4687e": "ladder", "7f7c32e807e5490b89ebba02c1aed656": "ladder",
    "d9d7989ac61848a09eeb60708dc0d941": "ladder",
    # loader (6)
    "096d89762af14d558355378be409651c": "loader", "0ca90083bdf34659b559b26a2583e619": "loader",
    "3d1fd419dcae41b3a3bc3869d850df79": "loader", "7c1b41c38b3747aeb3d12298aec42376": "loader",
    "d23c7e5565104a1db2e1c4641d4ed59b": "loader", "df9c3cfa98bb40598cfa7b96027db2a5": "loader",
    # mailbox (6)
    "1d0fe843aee943119c839f747acc7434": "mailbox", "2173ac25461e485aa834f8210bc16e4f": "mailbox",
    "600cdd77b2ea4720ab62380bf5f19c2d": "mailbox", "85e4d11083e744efaffb4411307efb7b": "mailbox",
    "b74a4f5f1e8d4d8a8989530c78304b00": "mailbox", "c5b082a6e88744ebaf2eb09ba65c8b9e": "mailbox",
    # motorcycle (20)
    "30be28bffc1c4635b0401c6f00cbbdf0": "motorcycle", "5e3ba0954ffc4a8f853c86cb7fdaa91b": "motorcycle",
    "693e83d86e1e4e5b95e4314dbdd95d40": "motorcycle", "6b5be07a35e449aa97a3001f6dc6a197": "motorcycle",
    "7ca014ea716c4439bbc504ecdcd33629": "motorcycle", "7d446ec9135c4e35892714d117b37268": "motorcycle",
    "89a79e0a3c334b23830561961f9644c8": "motorcycle", "8c5f5943533e47b8af90873f6c6c8563": "motorcycle",
    "91b031e0cc1b438b8c94bcb6e87aa381": "motorcycle", "a1b186f04d4b4654ab81b0d829007312": "motorcycle",
    "a50d882468854f05bd559fe878914d54": "motorcycle", "b0d453e026b24d8496b14a2fa046f457": "motorcycle",
    "b4141e61ada9458f9d6c2acaebea6ee8": "motorcycle", "bf3699774a564ecb9671146281859b41": "motorcycle",
    "c02dec85ac0541b8ab6dd56098706b50": "motorcycle", "d5db61785c4043a094908fe9deedcda5": "motorcycle",
    "d8d0022e70d048b4a87f7248655f73bf": "motorcycle", "dddd830af69a486fbf5e79c8e6515b4f": "motorcycle",
    "e9aa859595f543e7a80e3f4f9d1b1b48": "motorcycle", "f623ab40922944f881caf38d057e5d9d": "motorcycle",
    # newsstand (1)
    "220cc5fddc5d45c59cf61dec5426664f": "newsstand",
    # pallet (4)
    "04b59a0e10c5495a811849797a6f22fa": "pallet", "4f38d1e992e94966b6ceb2754cc16a22": "pallet",
    "ab550a0d20c147a08b9cf0ee97c50188": "pallet", "d47299e449774f40aa62abf4225d3386": "pallet",
    # parasol (4)
    "0f2fb587bd134e7eae6091fa227efcc0": "parasol", "280a65bca20b425e96617b0dd78cce21": "parasol",
    "a9b9af82e36b48f6bffbdaeeb74a5f63": "parasol", "ddb967e02f3c43be9082930d7caae023": "parasol",
    # parking meter (5)
    "6353119ea60d46e6a154374937c7c137": "parking meter", "9c91b51de37343d3a63e49e18371e862": "parking meter",
    "a1932b57521c4aa1b0148d1e6ad6b7aa": "parking meter", "a93161d5bafb4d83a73c766aadb97416": "parking meter",
    "df3f36c368e441bd979e0c149ff295a6": "parking meter",
    # phone booth (10)
    "0fdfe616df064592b8c5ea8ed632e248": "phone booth", "10d9ec53b8a140c7a7785ff54a7ecc22": "phone booth",
    "1367a893a4174000a49f2b6c0ad0b776": "phone booth", "4663dc46ff0c4cbb88e325b5a2b346ca": "phone booth",
    "66ab8d583ca24a0fa20f6096f66396b9": "phone booth", "8ab0dd710f0f4372bf53a2bd944a783f": "phone booth",
    "95086b465a63430cb9dcb124e767300d": "phone booth", "a0339aa975e54bb7ac65982ca3cc5802": "phone booth",
    "baf1ea30e7bb4790854619485d62b022": "phone booth", "e2493ece3b30446086602b8c006d2604": "phone booth",
    # pickup (10)
    "14116da06a65420db32b585466061d31": "pickup", "1986239266ee44c788fd6effe4055704": "pickup",
    "3b9c6eaf305342b48ebcb8295b2d2632": "pickup", "3edae53030ea45269fc63817312af4eb": "pickup",
    "40604d4eb4f3454685e66a77e447f23e": "pickup", "516ebd2fc1a84703b92d69a287ea2f98": "pickup",
    "7cf3226824a94b0d909a7ba0949bb5fe": "pickup", "aa1ff58f50ba4286b4e0c987824ed571": "pickup",
    "bc2c588c8cf9442ca1db7ed0c8cafc8b": "pickup", "dade78dc96e34f1a8cbcf14dd47d84de": "pickup",
    # picnic table (2)
    "629fd23c112a4e79ab8470b3e0d2e2b7": "picnic table", "887796ef013c42c9837320edaa316d07": "picnic table",
    # playground (5)
    "083c55b093ea42d6a6ccb261aff2d6b0": "playground", "8712f53da0e04736802f1bc84ca7df97": "playground",
    "8e116d30fcdd4b81984cf1b8c7c88a28": "playground", "9ea308ef1062419183ec4315449a3967": "playground",
    "e65fec66a8d248dcb1086f47df0f41de": "playground",
    # portable toilet (1)
    "adf13b3e678042c28a3f60483a0b827a": "portable toilet",
    # road barrier (9)
    "10d590da894543a2a9c2f14199c3fd7f": "road barrier", "3119d577027246e99f9deb89ea584d58": "road barrier",
    "8b630eb9f27d4a2b9f1d1629b552b7d3": "road barrier", "925da9cddeec426c9ce900b17cecb0e6": "road barrier",
    "ab21217481094e9a84f989e71fd56a59": "road barrier", "b2dd789e69bd401da49a94790d119079": "road barrier",
    "c566a3d32797473d86bbabb81d0ac155": "road barrier", "e16360dfad7d47cfbbfdffad13190ab7": "road barrier",
    "fe7c25cf56cf4058aaafe89bf5a93ee1": "road barrier",
    # road sign (9)
    "207e60c72ba849509c58959b163c23de": "road sign", "2c7caef93e9e44cabdd267c7221bc50a": "road sign",
    "42d002307cc14dd7ba7efe2a381f4d2b": "road sign", "459c229bb92a4f9481ea42b167082ff8": "road sign",
    "471fe4c7a2a84b7d9c0861577b8c3e7c": "road sign", "54de44f0efed4bb0b2c6388cbbeeb920": "road sign",
    "b6187c37c12541169875a0bdb1a6a821": "road sign", "ce046168e8da425e96e41eb2807ff87c": "road sign",
    "f150e9e0ddd148e8b0e1f2a9a14084b1": "road sign",
    # roundabout (3)
    "25b14611faa14bae992a7e26f1cafc4b": "roundabout", "3b3812979656435b9d705bf7698e0267": "roundabout",
    "834e7c27332a4d2b9cf5735c3762aacb": "roundabout",
    # scaffolding (3)
    "22a33921871448b09b3928a552340920": "scaffolding", "56c63eb16e5b479883cf3189932b38c9": "scaffolding",
    "d2230c5644234b3781cd7b2a6cec9638": "scaffolding",
    # shipping container (8)
    "0e8f2468dac244dc863bf52c628bcb5a": "shipping container", "133b0810d90d4d3992e1f8d67f439e03": "shipping container",
    "731960026f8240b1b5952dec3ec339fa": "shipping container", "7e6a1c61b9274e439b4b198ef67728a3": "shipping container",
    "803fc424bbe04e0a8bae1deb7bcdf196": "shipping container", "b3cf2804b4e5422e925d5f4cd56ab5de": "shipping container",
    "eb4e54e756584a02936bbe168bacbd2d": "shipping container", "f555ce5532d84cac8ad963dec5833934": "shipping container",
    # shopping trolley (3)
    "2d5a9e8a5954436cae975569fed0368d": "shopping trolley", "c22286ade5ea493b890b3269e57b191f": "shopping trolley",
    "fc4cb15fefb44f23bb6b0cd2e8aa1609": "shopping trolley",
    # small boat (10)
    "21bcd49240a54f2b879dd3f91de4fca3": "small boat", "2c603cab67f447d1b8ec8c9f45a30f36": "small boat",
    "392a16e37bf94a9ba74bbc720dee6b67": "small boat", "3e9093b6f0ce474da4d8cfea7e01f420": "small boat",
    "649c006b72e748bf84e7cda011faaff2": "small boat", "7a3bae8253074931a05d45788dee5852": "small boat",
    "8ea6bfc85e5d43b881dec5d224583f19": "small boat", "a4d90a06d2984b57952cc0fab6d78c9b": "small boat",
    "f2759f6bb0174cc3a94448d5a3dc5038": "small boat", "f4b38ccf5ffb46018aa7931c0e106654": "small boat",
    # street light (33)
    "043dfbe13d8041289e277e175df2f279": "street light", "06d6353675c64311860195a98a9655ab": "street light",
    "09c51f811e12462ba38eadf9a87b04a4": "street light", "0d898ab41a5842849f7211a5e0a36c43": "street light",
    "1837c49cd14f4b32bbd82723a702511e": "street light", "1890b4a77b794c6d9db9d2450255422d": "street light",
    "2a240a74f1f24df9af016f97b8e3aada": "street light", "2e59a82ba8af4cc8948ff6dffb6e28fb": "street light",
    "4a902fbbede042499e1c5eca53b6cc69": "street light", "56f6dcb3865144cd84e049ca6a736fae": "street light",
    "626dd7c8f4964b52abd0304d57853e76": "street light", "62ff0ef979144541b07bc8bf06c26582": "street light",
    "6341be23fc704d19967944359a410d68": "street light", "64db60298c1841aba59e23237d61da33": "street light",
    "6881aa1e84b047d79860fa9297e05e22": "street light", "7548a8a72b5e4f788151618ebd441c70": "street light",
    "758491c162e74077950a67fda4953b6c": "street light", "91df2261b9044b4782d25d5e34ae7065": "street light",
    "97624b7ee5594e1c8b24de0a58624d5d": "street light", "99b06af01eae45c09c941255e9bfbb4e": "street light",
    "9e7bd8d5eb144df692175d6aeb7bf879": "street light", "ad9e0859b1b44adf950695ead8c02a9a": "street light",
    "b702a80c6d8b4afe818c05b2cdeba29c": "street light", "b7cebcfb4b1c4d4a97e22072800704ca": "street light",
    "c166623a19574be99e8a417a9e65c45f": "street light", "c43ba8cc502742e5885b154554a1d873": "street light",
    "c47014c9ab5e445d89b7c0456719cf15": "street light", "c8cee098f2cc46b4b30062aa459651a2": "street light",
    "cfaae6a74f8a4483bdb0646961967579": "street light", "dcc7238701624cfc8b2190b7fa09dbf7": "street light",
    "e937afc06a1643f3bac57da9aec9e308": "street light", "e9b769a750b74020973211f39f976037": "street light",
    "fdf1673f1f6d41a99a90312db9bb2351": "street light",
    # tank (3)
    "395938614968466394a45456cbd188f6": "tank", "6c658096fcf5417da204211bedec201f": "tank",
    "b9cf4603062b4b93b10cb83e07ac596b": "tank",
    # tractor (14)
    "068f17e7bcf74d04bacf5524c4a2079d": "tractor", "16d799890d6c499a8762677449b9e179": "tractor",
    "1813d718b3b84a9bbe1a0993e18f84d3": "tractor", "2b7c6999b4f84f408752dd0ac038eb86": "tractor",
    "3ecdd1dd47a8402391ded8f8ab27c5b3": "tractor", "45d5c17faee745f3889074435bede0b2": "tractor",
    "4ff450c68b7143ea9df2e005147c0618": "tractor", "56682eb214214102a00a5e62b391670b": "tractor",
    "6080b7f9e1264520bc70478cfa952307": "tractor", "ae2d1d85b1a34466bf64810b85050a33": "tractor",
    "c333e8395bef42c5b5305c4620ad698c": "tractor", "d891c103cddd44b69d1357c832d9ad56": "tractor",
    "e929f07c86f1435f9ff51808892ce9fc": "tractor", "e9e844f2f9364044a181ce8f38a83dc3": "tractor",
    # traffic cone (12)
    "1a84a1183f1148428f110af1b4358943": "traffic cone", "29c9477325274bdaac844c1a43334a55": "traffic cone",
    "2f42722313cc4c48861f43d1a4950834": "traffic cone", "90d32979dac545c6a465399e453c3484": "traffic cone",
    "a37e20eb85564e9ba21d0569198b20c8": "traffic cone", "bd41ecc23835416bafea042d6a01a07b": "traffic cone",
    "cf5f95bc3d0347499b41bdd8e81ef9e0": "traffic cone", "d0eb4829fc9242dba0e592d553786f3f": "traffic cone",
    "d6c77b5fe7844dee9fbef1ce5390997d": "traffic cone", "e4673fcfd4764b8084465d65fb954980": "traffic cone",
    "f28523a0b7684d6c8972e8a853394708": "traffic cone", "faae6d37ef744cdf9e6325977313a4e5": "traffic cone",
    # traffic light (9)
    "1e120134a77944929519d8f96a5f3998": "traffic light", "525722b75cca448ea57df110a9d5df84": "traffic light",
    "7b86bedf64c04e15a0e01a84d29ea93d": "traffic light", "ac6a796f398e4443a7bf26bf633cb01b": "traffic light",
    "ba5769116fe34c1ea2712242c4f2ca98": "traffic light", "c351112891524961bf3288f11f5effbf": "traffic light",
    "e802a0af37bc41978ca31ba845dbaab5": "traffic light", "ea57a21eb7124457b13b1a9a5bc33625": "traffic light",
    "eb6395bf9a1e43f594fc30dd76004db4": "traffic light",
    # trailer (4)
    "103b3a78116b4abe81010cbd934b9350": "trailer", "594ef36c27924dcaa677781c756fbc7c": "trailer",
    "89a987bbaad04beda9955f70a8c12d90": "trailer", "b21a05025472417c8dc06e31c2481dfb": "trailer",
    # trampoline (1)
    "207a8a10d98e4607a7fb1812ee132b4a": "trampoline",
    # trash can (8)
    "0debf0ba50304951b32b44c7578468dd": "trash can", "0f1b5d0f8f1d49548580d0722552c0d1": "trash can",
    "1b3d199a690f4a58aa39f45c9c040c77": "trash can", "3057e7f5b6c441fcb67c44e68d77dd38": "trash can",
    "85630819475e40248537d2c8ab15caaf": "trash can", "9044ac8d545747e3959278fbe36fce8b": "trash can",
    "baee1591d1c44f74adf376fb5b6d2edb": "trash can", "e2c2ae3cbcb547659866c5a7d7b6a954": "trash can",
    # tree (7)
    "249d8cb553a8469e9d645713f8e96ed1": "tree", "2a0dbcdf8f5d48f5ad79987c7a8170ce": "tree",
    "3dc59560f2d24345bdbe65c44636453b": "tree", "4a201fe7f1a548fa9a0b32388be75499": "tree",
    "b5041d5648984ac2b833e2bde96fae85": "tree", "d841c3bcc5324daebee50f45619e05fc": "tree",
    "ef48485b7c534acc905c0d9964c5fe7c": "tree",
    # truck (8)
    "27a0aaa34e7d4ea7877393178d82b01c": "truck", "37b10f9c92584dfdb3d1337697dc8843": "truck",
    "38e93a0437c84184a141fe44a5125e4f": "truck", "393730a2af3d408c86cd7fa967dda242": "truck",
    "56f462ac530840999e3f17a0163fbcac": "truck", "95b04d8e016a49d9b487d2c4b9a56819": "truck",
    "ee87869496e245bfb4e7e2b0b1339aa6": "truck", "f0b7ca0f694c4213b799677e49041896": "truck",
    # utility pole (2)
    "be438e9c29634d4b8d055c57736c5a87": "utility pole", "da93952e43f6464282f7d4d6a0a88085": "utility pole",
    # van (5)
    "2f20ef6dd367492c885ee754fc7c607b": "van", "4258fa5bb1f74a2da7ed2090df0cb55b": "van",
    "7561ffb58b7e4cd9a25965a08f1d476b": "van", "ae025c380da7453a8912bec06d3557bb": "van",
    "f69de1315bb049c8946d57f6006acd73": "van",
    # vending machine (9)
    "110e72b7baa04396863b955dde50b954": "vending machine", "43280ec89088499a94259c5cb7d28b20": "vending machine",
    "4bfb493631f84f8a9adc00bb69831000": "vending machine", "54127d454edd4e3f982a46f58dd6c816": "vending machine",
    "74b075a5cc414de098ccdecddb355b46": "vending machine", "82cf245dee4b424785bcff3abae609b2": "vending machine",
    "94c002a46f7f4ac498b294e69e4e62cf": "vending machine", "efb393f5a3bb4bbfbe1bf5c13abb372d": "vending machine",
    "f992619d4bfc47bf9c9996ad5a43f8b5": "vending machine",
    # wheelbarrow (7)
    "08c0739e36a74cbc9742125eb1199435": "wheelbarrow", "0fd3dd1da381468c86d3bbe50a4db27b": "wheelbarrow",
    "340a8be1984d49de838ad6eee43a3905": "wheelbarrow", "4fc9d44fa5064daebebefbec98d472cb": "wheelbarrow",
    "68c8d8c7715947d6a393feecc83c0553": "wheelbarrow", "c0dcebcb0d4b47bd98df534334f4caf6": "wheelbarrow",
    "e9c76bd131e74ccbbc36f33f3e515f2c": "wheelbarrow",
}


# --- paths / objaverse --------------------------------------------------------------------------

def meta_dir(cache: Path) -> Path:
    return cache / "hf-objaverse-v1" / "metadata"


def _get(url: str, dst: Path, tries: int = 4) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "anyanything-blender-courses/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(dst)
            return dst
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    return dst


def lvis_map(cache: Path) -> dict[str, str]:
    """uid -> LVIS category (objaverse.load_lvis_annotations, cached under ``cache``)."""
    p = _get(f"{HF}/lvis-annotations.json.gz", cache / "hf-objaverse-v1" / "lvis-annotations.json.gz")
    out = {}
    for c, uids in json.load(gzip.open(p)).items():
        for u in uids:
            out[u] = c
    return out


def object_paths(cache: Path) -> dict[str, str]:
    p = _get(f"{HF}/object-paths.json.gz", cache / "hf-objaverse-v1" / "object-paths.json.gz")
    return json.load(gzip.open(p))


# --- index ----------------------------------------------------------------------------------------

def _index_shard(path: str) -> list[dict]:
    out = []
    for uid, v in json.load(gzip.open(path)).items():
        glb = (v.get("archives") or {}).get("glb") or {}
        if not v.get("isDownloadable", True) or v.get("isAgeRestricted"):
            continue
        thumbs = sorted((t for t in (v.get("thumbnails") or {}).get("images", []) if t.get("width", 0) >= 200),
                        key=lambda t: t["width"])
        out.append({
            "uid": uid, "name": v.get("name") or "", "tags": [t["name"].lower() for t in v.get("tags", [])],
            "sf_cats": [c["name"] for c in v.get("categories", [])], "licence": v.get("license"),
            "likes": v.get("likeCount", 0), "views": v.get("viewCount", 0), "staff": bool(v.get("staffpickedAt")),
            "faces": glb.get("faceCount", v.get("faceCount", 0)), "tex": glb.get("textureCount", 0),
            "tex_res": glb.get("textureMaxResolution", 0), "glb_mb": round(glb.get("size", 0) / 1e6, 2),
            "anim": v.get("animationCount", 0), "author": (v.get("user") or {}).get("displayName"),
            "url": v.get("viewerUrl"), "thumb": thumbs[0]["url"] if thumbs else None,
            "desc": (v.get("description") or "")[:400],
        })
    return out


def cmd_index(a) -> None:
    """All 160 metadata shards (fetched if missing) -> ``<cache>/index.json.gz``."""
    md = meta_dir(a.cache)
    ids = [f"000-{i:03d}" for i in range(160)]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda i: _get(f"{HF}/metadata/{i}.json.gz", md / f"{i}.json.gz"), ids))
    recs = []
    with ProcessPoolExecutor(12) as ex:
        for r in ex.map(_index_shard, [str(md / f"{i}.json.gz") for i in ids]):
            recs += r
    with gzip.open(a.cache / "index.json.gz", "wt") as f:
        json.dump(recs, f)
    print(f"{len(recs)} objects indexed -> {a.cache / 'index.json.gz'}")


def load_index(cache: Path) -> dict[str, dict]:
    return {r["uid"]: r for r in json.load(gzip.open(cache / "index.json.gz"))}


# --- select --------------------------------------------------------------------------------------

def _compile():
    for s in SPECS:
        s["_kw"] = re.compile("|".join(f"(?:{k})" for k in s["kw"]), re.I) if s["kw"] else None
        s["_neg"] = re.compile("|".join(f"(?:{k})" for k in s["neg"]), re.I) if s["neg"] else None
        s["_lvis"] = set(s["lvis"])


def spec(name: str) -> dict:
    return next(s for s in SPECS if s["name"] == name)


def score(r: dict, lvis: bool) -> float:
    txt = " ".join([r["name"], *r["tags"], r["desc"][:200]])
    sc = 1.0 * math.log1p(r["likes"]) + 0.35 * math.log1p(r["views"]) + 1.5 * r["staff"]
    sc += 1.2 * bool(REAL_TAGS.search(txt)) + 0.6 * lvis
    sc += 0.4 * min(r["tex_res"], 4096) / 4096
    if r["faces"] < 15000:
        sc -= 1.0
    return sc


def eligible(r: dict, a) -> str | None:
    """None if the object passes the global filters, else why not."""
    if r["licence"] not in a.licences:
        return "licence"
    if not (a.min_faces <= r["faces"] <= a.max_faces):
        return "faces"
    if r["tex"] < 1:
        return "untextured"
    if not (0 < r["glb_mb"] <= a.max_mb):
        return "size"
    if set(r["sf_cats"]) & SKIP_SF_CATS:
        return "sf-category"
    if GLOBAL_NEG.search(" ".join([r["name"], *r["tags"]])):
        return "style"
    return None


def classify(r: dict, lvis_cat: str | None) -> tuple[str, bool] | None:
    """(spec name, matched by LVIS) for the first spec an object fits."""
    name, tags = r["name"], " ".join(r["tags"])
    both = f"{name} {tags}"
    for strong in (True, False):
        for s in SPECS:
            if s["_neg"] and s["_neg"].search(both):
                continue
            if lvis_cat in s["_lvis"]:
                return s["name"], True
            if s["_kw"] and s["_kw"].search(name if strong else tags):
                return s["name"], False
    return None


def _tagged(a, name: str, ext: str) -> Path:
    """``<cache>/<name>[_<tag>][.<ext>]`` -- one set of selection files per round."""
    stem = f"{name}_{a.tag}" if a.tag else name
    return a.cache / (f"{stem}.{ext}" if ext else stem)


def cmd_select(a) -> dict[str, list[dict]]:
    _compile()
    idx = load_index(a.cache)
    lvis = lvis_map(a.cache)
    by = {s["name"]: [] for s in SPECS}
    why = {}
    seen = set()
    if a.skip_seen:                                          # earlier rounds' candidates
        for f in a.cache.glob("candidates*.json"):
            if f.name != _tagged(a, "candidates", "json").name:
                seen |= {c["uid"] for v in json.loads(f.read_text()).values() for c in v}
    for uid, r in idx.items():
        if uid in seen:
            continue
        w = eligible(r, a)
        if w:
            why[w] = why.get(w, 0) + 1
            continue
        c = classify(r, lvis.get(uid))
        if c:
            by[c[0]].append({**r, "spec": c[0], "lvis": lvis.get(uid), "score": round(score(r, c[1]), 2)})
    out = {}
    for s in SPECS:
        if a.specs and s["name"] not in a.specs:
            continue
        cands = sorted(by[s["name"]], key=lambda x: -x["score"])
        out[s["name"]] = cands[: int(s["cap"] * a.over)]
        print(f"{s['name']:>20}: {len(cands):5d} eligible, keep {len(out[s['name']])}")
    print("filtered:", why)
    _tagged(a, "candidates", "json").write_text(json.dumps(out, indent=1))
    print(f"{sum(map(len, out.values()))} candidates -> {_tagged(a, 'candidates', 'json')}")
    return out


# --- prescreen -------------------------------------------------------------------------------------

def cmd_prescreen(a) -> None:
    """Contact sheets of the candidates' Sketchfab thumbnails, labelled ``<n> <spec>`` --
    the n is the candidate's position in ``candidates.json`` so a reviewer can reject by number."""
    from PIL import Image, ImageDraw
    cands = json.loads(_tagged(a, "candidates", "json").read_text())
    flat = [c for s in SPECS for c in cands.get(s["name"], [])]
    tdir = a.cache / "sf_thumbs"

    def one(c):
        if not c["thumb"]:
            return None
        try:
            return _get(c["thumb"], tdir / f"{c['uid']}.jpg")
        except Exception:  # noqa: BLE001
            return None
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, flat))
    out = _tagged(a, "prescreen", "")
    out.mkdir(exist_ok=True)
    cols, cw, ch, per = 8, 180, 196, 48
    numbered = []
    for n, c in enumerate(flat):
        numbered.append((n, c))
    _tagged(a, "prescreen_index", "json").write_text(json.dumps({n: c["uid"] for n, c in numbered}))
    for page in range(0, len(numbered), per):
        chunk = numbered[page:page + per]
        rows = (len(chunk) + cols - 1) // cols
        im = Image.new("RGB", (cols * cw, rows * ch), (24, 24, 24))
        dr = ImageDraw.Draw(im)
        for q, (n, c) in enumerate(chunk):
            x, y = (q % cols) * cw, (q // cols) * ch
            p = tdir / f"{c['uid']}.jpg"
            try:
                t = Image.open(p).convert("RGB")
                t.thumbnail((cw - 4, cw - 4))
                im.paste(t, (x + 2, y + 2))
            except Exception:  # noqa: BLE001
                pass
            dr.rectangle((x, y + cw - 18, x + cw, y + ch), fill=(0, 0, 0))
            dr.text((x + 3, y + cw - 16), f"{n} {c['spec']}"[:28], fill=(255, 255, 90))
            dr.text((x + 3, y + cw - 2), f"{c['name'][:26]}", fill=(200, 200, 200))
        im.save(out / f"pre_{page // per:03d}.jpg", quality=80)
    print(f"{len(flat)} candidates on {(len(flat) + per - 1) // per} sheets in {out}")


# --- fetch + normalise ------------------------------------------------------------------------------

def _planes(parts) -> set[int]:
    """Indices of baked ground / shadow planes among ``parts`` = [(lo, hi, n_faces)]:
    few faces, flat, and wider than everything that is not such a slab."""
    flat = set()
    for i, (lo, hi, nf) in enumerate(parts):
        e = [hi[k] - lo[k] for k in range(3)]
        if nf <= 64 and e[2] <= 0.05 * max(e[0], e[1], 1e-9):
            flat.add(i)
    rest = [p for i, p in enumerate(parts) if i not in flat]
    if not rest:
        return set()
    rhor = max(max(p[1][k] for p in rest) - min(p[0][k] for p in rest) for k in (0, 1))
    return {i for i in flat if max(parts[i][1][k] - parts[i][0][k] for k in (0, 1)) > 1.2 * rhor}


def glb_box(path: str) -> tuple[list[float], int, bool]:
    """(Blender-frame (x, y, z) extent, number of ground planes, needs a transform bake)
    of a glb. glTF is Y-up and the importer maps (x, y, z) -> (x, -z, y), so height is
    the glTF y extent. The box leaves out baked ground/shadow planes (``_planes``).

    Blender (inspect_assets, Library.load) boxes an object by its local bound box
    carried to world space, which over-states the extent of a mesh under a rotation
    that is not a multiple of 90 degrees -- and a too-low box bottom makes the model
    float when a kit stands it on z = 0. ``needs bake`` flags files where that loose
    box differs from the exact one by more than 2 %."""
    import numpy as np
    import trimesh
    sc = trimesh.load(path, force="scene", process=False)
    parts, loose = [], []
    for node in sc.graph.nodes_geometry:
        T, g = sc.graph[node]
        m = sc.geometry[g]
        if not hasattr(m, "vertices") or len(m.vertices) == 0:
            continue
        vl = np.asarray(m.vertices)
        v = trimesh.transform_points(vl, T)[:, [0, 2, 1]]
        parts.append((v.min(0), v.max(0), len(getattr(m, "faces", []))))
        c = np.array([[x, y, z] for x in (vl[:, 0].min(), vl[:, 0].max()) for y in (vl[:, 1].min(), vl[:, 1].max())
                      for z in (vl[:, 2].min(), vl[:, 2].max())])
        c = trimesh.transform_points(c, T)[:, [0, 2, 1]]
        loose.append((c.min(0), c.max(0)))
    if not parts:
        raise ValueError("no geometry")
    planes = _planes(parts)
    keep = [i for i in range(len(parts)) if i not in planes]
    lo, hi = np.min([parts[i][0] for i in keep], 0), np.max([parts[i][1] for i in keep], 0)
    llo, lhi = np.min([loose[i][0] for i in keep], 0), np.max([loose[i][1] for i in keep], 0)
    ext = hi - lo
    bake = bool(np.max(np.abs((lhi - llo) - ext)) > 0.02 * max(ext.max(), 1e-9) or
                abs(llo[2] - lo[2]) > 0.01 * max(ext.max(), 1e-9))
    return [float(x) for x in ext], len(planes), bake


def _box_job(path: str):
    try:
        return glb_box(path), None
    except Exception as ex:  # noqa: BLE001
        return None, f"{type(ex).__name__}: {ex}"


def blender_clean(src: Path, dst: Path, blender: Path, strip: bool, bake: bool) -> bool:
    """Re-export ``src`` without its ground planes and/or with every mesh's transform
    baked into its vertices (this file run inside Blender, CPU)."""
    import subprocess
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(blender), "-b", "--factory-startup", "-P", str(Path(__file__).resolve()), "--",
                        "clean", str(src), str(dst), str(int(strip)), str(int(bake))],
                       capture_output=True, text=True, timeout=900)
    ok = r.returncode == 0 and dst.exists() and "CLEANED" in r.stdout
    if not ok:
        print(f"clean failed {src.name}: {r.stdout[-400:]} {r.stderr[-400:]}", flush=True)
    return ok


def _bl_clean(src: str, dst: str, strip: bool, bake: bool) -> None:
    import bpy
    from mathutils import Matrix, Vector
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)
    bpy.context.view_layer.update()
    gone = set()
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if strip:
        boxes = []
        for o in meshes:
            pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
            boxes.append(([min(p[k] for p in pts) for k in range(3)], [max(p[k] for p in pts) for k in range(3)],
                          len(o.data.polygons)))
        gone = _planes(boxes)
        for i in gone:
            bpy.data.objects.remove(meshes[i], do_unlink=True)
    baked = False
    skinned = any(o.type == "ARMATURE" for o in bpy.data.objects)
    if bake and not skinned:
        live = [o for o in bpy.data.objects if o.type == "MESH"]
        world = {o: o.matrix_world.copy() for o in live}
        for o in live:
            if o.data.users > 1:
                o.data = o.data.copy()
        for o in live:
            o.parent = None
            o.data.transform(world[o])
            if world[o].determinant() < 0:
                o.data.flip_normals()
            o.matrix_world = Matrix.Identity(4)
            o.data.update()
        for o in [o for o in bpy.data.objects if o.type != "MESH"]:
            bpy.data.objects.remove(o, do_unlink=True)
        root = bpy.data.objects.new("root", None)         # one root keeps the model one catalog piece
        bpy.context.scene.collection.objects.link(root)
        for o in live:
            o.parent = root
        baked = True
    bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB")
    print(f"CLEANED planes={len(gone)} baked={baked} skinned={skinned}", flush=True)


def target(s: dict, uid: str, box: list[float]) -> tuple[str, tuple, float] | None:
    """(measure, range, target size) for a spec given the object's shape, or None
    when the shape is outside every allowed height/width ratio."""
    L = max(box[0], box[1])
    hl = box[2] / max(L, 1e-9)
    choice = None
    for lo, hi, m, rng in s["alt"]:
        if lo <= hl < hi:
            choice = (m, rng)
    if choice is None and s["hl"][0] <= hl <= s["hl"][1]:
        choice = (s["measure"], s["rng"])
    if choice is None:
        return None
    frac = int(hashlib.md5(uid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    m, rng = choice
    return m, rng, rng[0] + frac * (rng[1] - rng[0])


def clean_tags(r: dict, s: dict) -> list[str]:
    """Spec words + the object's own name words and tags, minus words that name
    *another* spec (an Objaverse car tagged "tree" must not answer tags_any={"tree"})."""
    own = {w for w in [s["name"], *s["cats"]]}
    others = set()
    for t in SPECS:
        if t["name"] != s["name"]:
            others |= {t["name"], t["cats"][-1]}
    others -= own
    words = set()
    for t in r["tags"] + re.findall(r"[a-z][a-z-]+", r["name"].lower()):
        t = t.strip().lower().replace("_", " ")
        if not t or JUNK_TAGS.match(t) or t in others:
            continue
        words.add(t)
    return sorted(words | own | {"objaverse"})


def _numbers(txt: str) -> set[int]:
    """'3,7,10-14' -> {3, 7, 10, 11, 12, 13, 14}"""
    out = set()
    for part in filter(None, (x.strip() for x in (txt or "").split(","))):
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            out |= set(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def cmd_fetch(a) -> None:
    import objaverse
    _compile()
    objaverse.BASE_PATH = str(a.cache)
    objaverse._VERSIONED_PATH = str(a.cache / "hf-objaverse-v1")
    idx = load_index(a.cache)
    lvis = lvis_map(a.cache)
    if a.candidates:
        want = {}
        for kf in a.keep_file or [None]:                     # one keep file per selection round
            k = json.loads(kf.read_text()) if kf else {}
            a.tag = k.get("tag", a.tag)
            cands = json.loads(_tagged(a, "candidates", "json").read_text())
            pre = json.loads(_tagged(a, "prescreen_index", "json").read_text())
            drop = {pre[str(n)] for n in _numbers(k.get("drop", a.drop_numbers)) if str(n) in pre}
            w = {c["uid"]: c["spec"] for s in SPECS for c in cands.get(s["name"], []) if c["uid"] not in drop}
            if k.get("keep", a.keep_numbers):
                keep = {pre[str(n)] for n in _numbers(k.get("keep", a.keep_numbers)) if str(n) in pre}
                w = {u: s for u, s in w.items() if u in keep}
            for part in filter(None, (x.strip() for x in k.get("recat", a.recat).split(","))):
                n, name = part.split(":")
                spec(name)                                   # must exist
                w[pre[n]] = name
            want.update(w)
    else:
        want = dict(CURATED)
    rej_path = a.cache / "rejected.json"
    rejected = {}
    print(f"{len(want)} objects to fetch", flush=True)
    paths = objaverse.load_objects(sorted(want), download_processes=a.processes)
    uids = sorted(u for u in want if u in paths)
    with ProcessPoolExecutor(a.processes) as ex:
        boxes = dict(zip(uids, ex.map(_box_job, [paths[u] for u in uids])))
    # ground / shadow planes baked into the file, rotated sub-meshes: re-export (Blender, CPU)
    todo = {u: (boxes[u][0][1] > 0, boxes[u][0][2]) for u in uids if boxes[u][0] and (boxes[u][0][1] or boxes[u][0][2])}
    cleaned = {}
    if todo:
        print(f"cleaning {len(todo)} files ({sum(v[0] for v in todo.values())} with ground planes, "
              f"{sum(v[1] for v in todo.values())} with rotated sub-meshes)", flush=True)

        def one(u):
            strip, bake = todo[u]
            dst = a.root / "objaverse" / u / f"{u}.glb"
            marker = dst.with_suffix(".cleaned")
            want_m = {"strip_ground_planes": strip, "bake_transforms": bake, "v": 2}
            if marker.exists() and dst.exists() and json.loads(marker.read_text()) == want_m:
                return u, want_m
            ok = blender_clean(Path(paths[u]), dst, a.blender, strip, bake)
            if ok:
                marker.write_text(json.dumps(want_m))
            return u, want_m if ok else None
        with ThreadPoolExecutor(4) as ex:
            for u, m in ex.map(one, sorted(todo)):
                if m:
                    cleaned[u] = m
                    boxes[u] = _box_job(str(a.root / "objaverse" / u / f"{u}.glb"))

    out_path = a.root / "library_objaverse.json"
    lib = {"items": {}}
    n_rej = 0
    for uid in uids:
        s, r = spec(want[uid]), idx[uid]
        res, err = boxes[uid]
        box = res[0] if res else None
        if err or not box or max(box) <= 0:
            rejected[uid] = f"unreadable: {err}"
            n_rej += 1
            continue
        t = target(s, uid, box)
        if t is None:
            rejected[uid] = f"shape: {s['name']} h/L={box[2] / max(box[0], box[1], 1e-9):.2f} outside {s['hl']}"
            n_rej += 1
            continue
        m, rng, tgt = t
        cur = box[2] if m == "h" else max(box[0], box[1])
        scale = tgt / cur
        rel = Path("objaverse") / uid / f"{uid}.glb"
        dst = a.root / rel
        if uid not in cleaned and (not dst.exists() or dst.stat().st_size != Path(paths[uid]).stat().st_size):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(paths[uid], dst)
        lib["items"][f"obj:{uid}"] = {
            "kind": "model", "source": "objaverse", "licence": LICENCE_NAMES.get(r["licence"], r["licence"]),
            "licence_short": r["licence"], "authors": [r["author"]] if r["author"] else [],
            "url": r["url"], "name": r["name"], "categories": list(s["cats"]), "tags": clean_tags(r, s),
            "src_tags": r["tags"], "spec": s["name"], "lvis": lvis.get(uid),
            "path": str(rel), "scale": round(scale, 6), "dims_m": [round(x * scale, 3) for x in box],
            "size_rule": {"measure": m, "range": list(rng), "target": round(tgt, 3)},
            "faces": r["faces"], "textures": r["tex"], "cleaned": cleaned.get(uid, {}),
            "exclude": uid not in CURATED,                   # not reviewed yet: kits must not place it
        }
    rej_path.write_text(json.dumps(rejected, indent=1, sort_keys=True))
    lib["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = out_path.with_suffix(".json.tmp")                # atomic: builds may be reading it
    tmp.write_text(json.dumps(lib, indent=1, sort_keys=True))
    tmp.replace(out_path)
    print(f"{len(lib['items'])} models -> {out_path}; {n_rej} rejected by shape/read ({rej_path})")


# --- finalise ----------------------------------------------------------------------------------------

def cmd_finalise(a) -> None:
    """After the inspect merge: mark ``exclude`` on the pieces of an Objaverse file that
    are not objects of their own -- empty, or small next to the file's biggest piece (a
    stray light cone, a decal, a loose bolt) -- and drop catalog entries whose file left
    the library. Substantial extra pieces stay (a second hydrant, the chairs of a set)."""
    cat_path = a.root / "catalog_objaverse.json"
    cat = json.loads(cat_path.read_text())
    items = json.loads((a.root / "library_objaverse.json").read_text())["items"]
    files = {k: v for k, v in cat["files"].items() if k in items}
    n_x = 0
    for k, v in files.items():
        ps = v.get("pieces", [])
        if len(ps) <= 1:
            continue
        vol = [max(p["size"][0], 1e-3) * max(p["size"][1], 1e-3) * max(p["size"][2], 1e-3) for p in ps]
        best = max(range(len(ps)), key=lambda i: vol[i])
        big = max(ps[best]["size"])
        for i, p in enumerate(ps):
            p["exclude"] = i != best and (p.get("tris", 0) < 0.02 * ps[best].get("tris", 0) or
                                          vol[i] < 0.1 * vol[best] or max(p["size"]) < 0.3 * big)
            n_x += p["exclude"]
    cat["files"] = files
    tmp = cat_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cat, indent=1, sort_keys=True))
    tmp.replace(cat_path)
    print(f"{len(files)} files in {cat_path}; {n_x} secondary pieces excluded")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["index", "select", "prescreen", "fetch", "finalise"])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--licences", nargs="*", default=["cc0", "by", "by-sa"])
    ap.add_argument("--min-faces", type=int, default=5000)
    ap.add_argument("--max-faces", type=int, default=400000)
    ap.add_argument("--max-mb", type=float, default=80.0)
    ap.add_argument("--over", type=float, default=2.5, help="candidates kept per spec = cap x over")
    ap.add_argument("--candidates", action="store_true", help="fetch the select output, not CURATED")
    ap.add_argument("--drop-numbers", default="", help="prescreen numbers to skip, e.g. '3,7,10-14'")
    ap.add_argument("--keep-numbers", default="", help="prescreen numbers to keep (all others skipped)")
    ap.add_argument("--keep-file", type=Path, nargs="*", default=None,
                    help="json {tag, keep, drop, recat} per selection round (instead of the flags)")
    ap.add_argument("--tag", default="", help="selection round: candidates_<tag>.json, prescreen_<tag>/ ...")
    ap.add_argument("--specs", nargs="*", default=None, help="select only these specs")
    ap.add_argument("--skip-seen", action="store_true", help="select: skip earlier rounds' candidates")
    ap.add_argument("--recat", default="", help="prescreen number -> spec, e.g. '8:motorcycle,491:pallet'")
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--blender", type=Path, default=Path("/mnt/data/blender/blender-5.2.2-linux-x64/blender"))
    a = ap.parse_args(argv)
    a.cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", "/mnt/data/hf-home")
    {"index": cmd_index, "select": cmd_select, "prescreen": cmd_prescreen, "fetch": cmd_fetch,
     "finalise": cmd_finalise}[a.step](a)
    return 0


if __name__ == "__main__":
    if "bpy" in sys.modules or Path(sys.executable).name.startswith("blender") or "--" in sys.argv:
        _argv = sys.argv[sys.argv.index("--") + 1:]
        if _argv and _argv[0] == "clean":        # blender -b -P fetch_objaverse.py -- clean in out strip bake
            _bl_clean(_argv[1], _argv[2], bool(int(_argv[3])), bool(int(_argv[4])))
            sys.exit(0)
    sys.exit(main())
