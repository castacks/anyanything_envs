#!/usr/bin/env python3
"""Fetch the asset library the course builder draws from, and describe it.

    python blender/assets.py --root ~/.cache/blender_assets --theme yard

Poly Haven (CC0, real-world scale in metres) through its public API: models
as glTF with 1k textures, HDRIs as 2k ``.hdr``, PBR textures as 1k jpg
(diffuse, OpenGL normal, roughness). Every file is checked against the md5
the API publishes, so a re-run downloads only what is missing or corrupt.

The output is ``<root>/manifest.json``: one entry per asset with its kind,
the role(s) the course builder may use it in, the source and the licence.
The roles are hand-curated below per theme -- the API's categories say what
an object *is*, not whether a drone can meaningfully fly round it.

Standard library only: it runs on any system Python, without a venv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://api.polyhaven.com"
UA = {"User-Agent": "anyanything-blender-courses/0.1 (research dataset generator)"}

#: role -> Poly Haven model ids. Roles are what ``build_course.py`` places:
#: ``prop`` stands on the ground as a scattered obstacle, ``stack`` is
#: stacked into columns, ``tall`` is a slender vertical obstacle (posts,
#: lamps, shelving), ``large`` a bulky one.
#: Texture roles: ``wall`` dresses walls and boxes, ``pole`` vertical
#: cylinders (trunks in the woodland), ``bar`` horizontal ones, ``rock``
#: spheres, ``ground`` the floor, ``ceiling`` the warehouse roof. Model roles
#: add ``tree`` (scattered like ``large``, may carry alpha-cut leaves) and
#: ``hang`` (hung from the warehouse ceiling into the flight band, hard levels).
THEMES = {
    "yard": {
        "models": {
            # stacked 2-4 high by the builder (build_course.scatter), so they reach the flight band
            "stack": ["cardboard_box_01", "wooden_crate_01", "wooden_crate_02", "plastic_crate_01",
                      "Barrel_01", "Barrel_02", "barrel_03", "wooden_military_crate",
                      "industrial_pastic_container", "metal_tool_chest"],
            "prop": ["fire_hydrant", "WetFloorSign_01", "old_tyre", "hand_truck", "tool_cart",
                     "portable_welding_cart", "small_lpg_tank", "propane_tank", "painted_wooden_bench",
                     "utility_box_01", "utility_box_02", "portable_generator"],
            "large": ["concrete_road_barrier", "concrete_road_barrier_02", "covered_car",
                      "industrial_storage_cart", "exterior_aircon_unit", "metal_trash_can", "boulder_01",
                      "dead_tree_trunk_02"],
            "tall": ["street_lamp_01", "Shelf_01", "ladder_sectioned_01"],
        },
        # outdoor, urban/industrial, daylight
        "hdris": ["factory_yard", "abandoned_parking", "construction_yard", "derelict_airfield_02",
                  "mall_parking_lot", "overcast_industrial_courtyard", "abandoned_tank_farm_01",
                  "pump_station", "urban_courtyard", "wide_street_01", "skidpan", "driving_school",
                  "docklands_01", "pylons", "urban_alley_01"],
        "textures": {
            "wall": ["brick_wall_001", "brick_wall_02", "concrete_block_wall", "concrete_wall_003",
                     "corrugated_iron", "painted_brick", "plastered_wall_02", "rusty_metal_sheet"],
            "pole": ["concrete_wall_003", "rusty_metal_sheet", "metal_plate"],
            "bar": ["rusty_metal_sheet", "metal_plate", "blue_metal_plate"],
            "rock": ["mossy_rock", "rock_pitted_mossy"],
            "ground": ["asphalt_02", "concrete_floor_worn_001", "gravel_road", "road_damaged",
                       "concrete_pavement"],
        },
    },
    "woodland": {
        "models": {
            "tree": ["tree_small_02", "island_tree_02", "island_tree_03", "searsia_burchellii", "searsia_lucida",
                     "quiver_tree_01", "othonna_cerarioides"],
            "large": ["namaqualand_boulder_04", "namaqualand_boulder_03", "boulder_01", "coast_rocks_05",
                      "rock_face_02", "root_cluster_01", "dead_tree_trunk_02", "shrub_02", "pachira_aquatica_01"],
            "tall": ["dead_quiver_trunk", "quiver_tree_02", "fir_sapling", "pine_sapling_small"],
            "prop": ["tree_stump_01", "tree_stump_02", "namaqualand_boulder_02", "namaqualand_boulder_06",
                     "namaqualand_boulder_05", "grass_medium_02", "didelta_spinosa", "celandine_01"],
        },
        "hdris": ["autumn_forest_01", "autumn_forest_02", "autumn_forest_03", "epping_forest_01", "autumn_park",
                  "borghese_gardens", "ballawley_park", "charolettenbrunn_park", "cloudy_vondelpark",
                  "alps_field", "autumn_meadow", "abandoned_pathway"],
        "textures": {
            "wall": ["mossy_stone_wall", "old_stone_wall", "stacked_stone_wall", "rustic_stone_wall", "moss_wood"],
            "pole": ["bark_brown_01", "bark_brown_02", "bark_willow", "bark_platanus", "eucalyptus_bark",
                     "japanese_cedar_bark"],
            "bar": ["bark_brown_02", "bark_willow_02", "moss_wood"],
            "rock": ["mossy_rock", "rock_pitted_mossy"],
            "ground": ["forest_floor", "forest_ground_04", "forest_leaves_02", "leaves_forest_ground",
                       "brown_mud_leaves_01", "sparse_grass", "grass_path_2"],
        },
    },
    "warehouse": {
        "models": {
            "stack": ["cardboard_box_01", "wooden_crate_01", "wooden_crate_02", "plastic_crate_01", "Barrel_01",
                      "Barrel_02", "wooden_military_crate", "industrial_pastic_container"],
            "tall": ["steel_frame_shelves_01", "steel_frame_shelves_02", "steel_frame_shelves_03", "Shelf_01",
                     "worn_metal_rack", "wooden_bookshelf_worn", "drawer_cabinet", "ladder_sectioned_01",
                     "modular_industrial_pipes_01"],
            "large": ["industrial_storage_cart", "tool_cart", "metal_trash_can", "CoffeeCart_01",
                      "painted_wooden_table", "metal_office_desk", "covered_car"],
            "prop": ["hand_truck", "WetFloorSign_01", "metal_tool_chest", "old_drill_press", "drill_press_01",
                     "barrel_03", "utility_box_01", "portable_generator", "small_lpg_tank"],
            "hang": ["hanging_industrial_lamp", "caged_hanging_light", "modern_ceiling_lamp_01"],
        },
        # never seen through the shell, but they set the ambient colour of the gaps
        "hdris": ["empty_warehouse_01", "aircraft_workshop_01", "carpentry_shop_01"],
        "textures": {
            "wall": ["corrugated_iron_02", "concrete_block_wall", "painted_brick", "grey_plaster_02",
                     "worn_corrugated_iron", "plaster_brick_01"],
            "pole": ["concrete_wall_003", "metal_plate", "blue_metal_plate"],
            "bar": ["metal_plate", "rusty_metal_sheet"],
            "rock": ["concrete_wall_003"],
            "ground": ["concrete_floor_01", "hangar_concrete_floor", "smooth_concrete_floor",
                       "concrete_floor_painted", "worn_concrete_floor"],
            "ceiling": ["corrugated_iron", "metal_plate", "grey_plaster"],
        },
    },
    "plaza": {
        "models": {
            "tree": ["tree_small_02", "searsia_lucida", "island_tree_03"],
            "tall": ["street_lamp_01", "gothic_statue", "potted_plant_01", "modular_industrial_pipes_01"],
            "large": ["modular_street_seating", "wooden_picnic_table", "outdoor_table_chair_set_01", "covered_car",
                      "concrete_road_barrier", "concrete_road_barrier_02", "bronze_whale_statue", "metal_trash_can"],
            "prop": ["painted_wooden_bench", "planter_box_01", "planter_box_02", "planter_box_03", "fire_hydrant",
                     "potted_plant_02", "plastic_monobloc_chair_01", "round_wooden_table_01", "utility_box_02"],
        },
        "hdris": ["urban_courtyard_02", "piazza_bologni", "piazza_martin_lutero", "dresden_square",
                  "schadowplatz", "museumplein", "potsdamer_platz", "stone_alley_02", "venetian_crossroads",
                  "palermo_sidewalk", "german_town_street", "plac_wolnosci"],
        "textures": {
            "wall": ["plaster_stone_wall_01", "red_sandstone_wall", "stone_wall", "painted_plaster_wall",
                     "brick_wall_02", "patterned_plaster_wall"],
            "pole": ["stone_wall_02", "grey_plaster", "metal_plate"],
            "bar": ["metal_plate", "blue_metal_plate"],
            "rock": ["stone_wall"],
            "ground": ["cobblestone_floor_01", "herringbone_pavement", "brick_pavement_02", "cobblestone_05",
                       "checkered_pavement_tiles", "concrete_pavement_02"],
        },
    },
}

TEXTURE_MAPS = {"Diffuse": "diff", "nor_gl": "nor_gl", "Rough": "rough"}


def get_json(url: str, tries: int = 4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, want_md5: str | None, tries: int = 4) -> int:
    """Download ``url`` to ``dest`` unless it is already there with the right
    md5. Returns the bytes downloaded (0 when cached)."""
    if dest.exists() and (want_md5 is None or md5(dest) == want_md5):
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, \
                    open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if want_md5 and md5(tmp) != want_md5:
                raise IOError(f"md5 mismatch for {url}")
            tmp.replace(dest)
            return dest.stat().st_size
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    return 0


def model_jobs(aid: str, root: Path, res: str) -> tuple[list, dict]:
    """The glTF and everything it includes, plus every ``*alpha`` map the
    asset has: Poly Haven's glTF exports mark leaf materials as blended but
    ship them with JPG colour only, so the cut-out lives in a separate file the
    builder wires in (``build_course.attach_alpha``)."""
    files = get_json(f"{API}/files/{aid}")
    g = files["gltf"][res]["gltf"]
    base = root / "models" / aid
    jobs = [(g["url"], base / f"{aid}.gltf", g.get("md5"))]
    for rel, inc in g.get("include", {}).items():
        jobs.append((inc["url"], base / rel, inc.get("md5")))
    alpha = {}
    for key, per_res in files.items():
        if not key.lower().endswith("alpha") or res not in per_res:
            continue
        fmt = "png" if "png" in per_res[res] else next(iter(per_res[res]))
        f = per_res[res][fmt]
        dest = base / "alpha" / f"{aid}_{key}_{res}.{fmt}"
        jobs.append((f["url"], dest, f.get("md5")))
        part = key[:-len("alpha")].rstrip("_").lower()
        alpha[part] = str(dest.relative_to(root))
    entry = {"kind": "model", "path": str((base / f"{aid}.gltf").relative_to(root))}
    if alpha:
        entry["alpha_maps"] = alpha
    return jobs, entry


def hdri_jobs(aid: str, root: Path, res: str) -> tuple[list, dict]:
    h = get_json(f"{API}/files/{aid}")["hdri"][res]["hdr"]
    dest = root / "hdris" / f"{aid}_{res}.hdr"
    return [(h["url"], dest, h.get("md5"))], {"kind": "hdri", "path": str(dest.relative_to(root))}


def texture_jobs(aid: str, root: Path, res: str) -> tuple[list, dict]:
    files = get_json(f"{API}/files/{aid}")
    jobs, maps = [], {}
    for key, short in TEXTURE_MAPS.items():
        if key not in files:
            continue
        j = files[key][res]["jpg"]
        dest = root / "textures" / aid / f"{aid}_{short}_{res}.jpg"
        jobs.append((j["url"], dest, j.get("md5")))
        maps[short] = str(dest.relative_to(root))
    return jobs, {"kind": "texture", "maps": maps}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--theme", default="all", choices=sorted(THEMES) + ["all"])
    ap.add_argument("--model-res", default="1k")
    ap.add_argument("--hdri-res", default="2k")
    ap.add_argument("--tex-res", default="1k")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    root = a.root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    themes = sorted(THEMES) if a.theme == "all" else [a.theme]
    info = get_json(f"{API}/assets")                 # names, categories, dimensions, authors

    man_path = root / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"assets": {}}
    for e in manifest["assets"].values():          # the listed themes' lists are authoritative
        e["themes"] = [t for t in e.get("themes", []) if t not in themes]
        e["roles_by_theme"] = {t: r for t, r in e.get("roles_by_theme", {}).items() if t not in themes}
    todo = []
    for th in themes:
        theme = THEMES[th]
        for role, ids in theme["models"].items():
            todo += [(aid, "model", role, th) for aid in ids]
        todo += [(aid, "hdri", "sky", th) for aid in theme["hdris"]]
        for role, ids in theme["textures"].items():
            todo += [(aid, "texture", role, th) for aid in ids]

    jobs, failed = [], []
    for aid, kind, role, th in todo:
        try:
            fj, entry = {"model": model_jobs, "hdri": hdri_jobs, "texture": texture_jobs}[kind](
                aid, root, {"model": a.model_res, "hdri": a.hdri_res, "texture": a.tex_res}[kind])
        except Exception as e:  # noqa: BLE001
            failed.append(f"{aid}: {type(e).__name__}: {e}")
            continue
        meta = info.get(aid, {})
        old = manifest["assets"].get(aid, {})
        rbt = dict(old.get("roles_by_theme", {}))
        rbt[th] = sorted(set(rbt.get(th, [])) | {role})
        manifest["assets"][aid] = {**entry, "roles_by_theme": rbt, "themes": sorted(rbt),
                                   "roles": sorted({r for rs in rbt.values() for r in rs}), "source": "polyhaven",
                                   "licence": "CC0-1.0", "url": f"https://polyhaven.com/a/{aid}",
                                   "authors": sorted(meta.get("authors", {})),
                                   "dimensions_mm": meta.get("dimensions")}
        jobs += fj

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        got = list(ex.map(lambda j: fetch(*j), jobs))
    manifest["assets"] = {k: v for k, v in manifest["assets"].items() if v.get("roles_by_theme")}
    manifest["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    n = sum(1 for g in got if g)
    print(f"{len(manifest['assets'])} assets in {man_path}; {n}/{len(jobs)} files downloaded "
          f"({sum(got) / 1e6:.0f} MB) in {time.time() - t0:.0f} s")
    for f in failed:
        print("FAILED", f, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
