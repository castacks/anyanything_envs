"""An Infinigen-Indoors house generated for drone flights: one furnished single-storey house, every room real.

Unlike the nature pipeline, Infinigen-Indoors does not populate by camera visibility: the solver places every object
of every room and populate_state_placeholders spawns each one at the fixed detail distance, with no camera. What
depends on the camera is optional, and this driver forces it off (all as gin overrides, nothing patched):

- the outdoor terrain backdrop (only the camera's view gets grass and rocks): terrain off, no outdoors at all
- OcMesher room meshes around the cameras (real_geometry*.gin): off, Infinigen's uniform room meshes
- overhead.gin's hidden rooms / invisible ceilings / overhead camera: off, every room and ceiling renders
- floating objects placed in the camera's view: off
- Infinigen's own camera search (pose_cameras), which nothing here uses: skipped
- the 20 % chance that every light is deleted: 0, so the drone views are lit

Not about the camera, but about flying: Infinigen leaves half its doors nearly shut (0-9 deg), which seals rooms off.
--doors open (the default) opens every door with Infinigen's own all_open switch; --doors infinigen keeps its mix.

    python infinigen/indoor_box.py --seed 1 --out out/house_1 [--doors open] [--configs extra.gin] [-p gin.param=value]

fast_solve.gin is on by default (fewer solver steps and Infinigen's reduced room program: living room, kitchen,
dining room, bedroom(s), bathroom; about 5-7 rooms). Writes <out>/scene/scene.blend (+ Infinigen's solve_state.json,
MaskTag.json) and <out>/box.json (rooms with outlines, interior bounds, floor and ceiling heights, object counts).
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Applied after --configs, before -p: what makes the house the same from every pose (see the docstring).
FLIGHT_OVERRIDES = [
    "RoomConstants.n_stories=1",
    "compose_indoors.terrain_enabled=False",
    "compose_indoors.enable_ocmesh_room=False",
    "compose_indoors.hide_other_rooms_enabled=False",
    "compose_indoors.invisible_room_ceilings_enabled=False",
    "compose_indoors.overhead_cam_enabled=False",
    "compose_indoors.floating_objs_enabled=False",
    "compose_indoors.pose_cameras_enabled=False",
    "compose_indoors.animate_cameras_enabled=False",
    "compose_indoors.lights_off_chance=0.0",
]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-fast-solve", action="store_true", help="full solver steps and room program (hours)")
    ap.add_argument("--doors", choices=["open", "infinigen"], default="open",
                    help="open: every door open (84-174 deg); infinigen: half of them nearly shut")
    ap.add_argument("--configs", nargs="*", default=[], help="extra configs_indoor gin files")
    ap.add_argument("-p", "--overrides", nargs="*", default=[])
    ap.add_argument("--layout-only", action="store_true",
                    help="recompute box.json's layout from <out>/scene/scene.blend")
    return ap.parse_args()


def world_triangles(o):
    """World-space triangles of a mesh object, [n, 3, 3]."""
    me = o.data
    me.calc_loop_triangles()
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    M = np.array([list(r) for r in o.matrix_world])
    V = V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    T = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", T)
    return V[T.reshape(-1, 3)]


def house_layout():
    """Rooms of the house in the open scene, from Infinigen's split room meshes (collections
    unique_assets:room_floor / room_ceiling, objects '<room>.floor' / '<room>.ceiling'): each room's floor outline,
    area and floor height from its floor faces, its ceiling height from its ceiling faces."""
    import bpy
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    rooms = {}
    for part in ("floor", "ceiling"):
        col = bpy.data.collections.get(f"unique_assets:room_{part}")
        for o in (col.objects if col is not None else []):
            if o.type != "MESH" or o.hide_render:
                continue
            tri = world_triangles(o)
            if len(tri) == 0:
                continue
            name = o.name.rsplit(".", 1)[0]
            r = rooms.setdefault(name, {"name": name, "type": name.split("_")[0]})
            if part == "floor":
                shape = unary_union([p for p in (Polygon(t[:, :2]) for t in tri) if p.area > 1e-6])
                shape = shape.buffer(0.005).buffer(-0.005).simplify(0.01)
                polys = list(shape.geoms) if hasattr(shape, "geoms") else [shape]
                polys.sort(key=lambda p: -p.area)
                r["polygon"] = [[round(x, 3), round(y, 3)] for x, y in polys[0].exterior.coords]
                r["extra_polygons"] = len(polys) - 1
                r["area_m2"] = round(shape.area, 2)
                r["bounds"] = [round(v, 3) for v in shape.bounds]
                r["floor_z"] = round(float(np.median(tri[:, :, 2])), 3)
            else:
                r["ceiling_z"] = round(float(np.median(tri[:, :, 2])), 3)
    rooms = [r for r in rooms.values() if "polygon" in r]
    rooms.sort(key=lambda r: r["name"])
    for r in rooms:
        if "ceiling_z" in r:
            r["height_m"] = round(r["ceiling_z"] - r["floor_z"], 3)
    b = np.array([r["bounds"] for r in rooms])
    walls = []
    for part in ("wall", "exterior"):
        col = bpy.data.collections.get(f"unique_assets:room_{part}")
        for o in (col.objects if col is not None else []):
            V = world_triangles(o).reshape(-1, 3) if o.type == "MESH" else np.zeros((0, 3))
            if len(V):
                walls.append(np.r_[V.min(0), V.max(0)])
    walls = np.array(walls)
    floor_z = float(np.median([r["floor_z"] for r in rooms]))
    ceil_z = float(np.median([r["ceiling_z"] for r in rooms if "ceiling_z" in r]))
    return {
        "rooms": rooms,
        "interior_bounds": [round(float(v), 3) for v in (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())],
        "house_bounds": ([round(float(v), 3) for v in np.r_[walls[:, :3].min(0), walls[:, 3:].max(0)]]
                         if len(walls) else None),
        "floor_z": round(floor_z, 3), "ceiling_z": round(ceil_z, 3), "ceiling_height_m": round(ceil_z - floor_z, 3),
        "floor_area_m2": round(float(unary_union([Polygon(r["polygon"]) for r in rooms]).area), 2),
    }


def object_census(rooms):
    """Populated asset roots per factory and per room (render-visible, outside the placeholder collections; a root
    belongs to the room whose floor outline holds its origin, else to the nearest one within 0.5 m: doors and windows
    sit in the walls), and any placeholder left unpopulated (should be none)."""
    import bpy
    from shapely.geometry import Point, Polygon
    hidden = set()
    for c in bpy.data.collections:
        if c.name.startswith("placeholders") or c.hide_render:
            hidden.update(o.name for o in c.all_objects)
    polys = [Polygon(r["polygon"]) for r in rooms]
    roots, per_room, stray = Counter(), Counter(), []
    for o in bpy.data.objects:
        if o.name in hidden or o.hide_render:
            continue
        if ".spawn_asset(" in o.name and (o.parent is None or ".spawn_asset(" not in o.parent.name):
            roots[o.name.split("(")[0]] += 1
            pt = Point(o.matrix_world.translation[:2])
            d, k = min((poly.distance(pt), k) for k, poly in enumerate(polys))
            per_room[rooms[k]["name"] if d <= 0.5 else "none"] += 1
        elif ".spawn_placeholder(" in o.name and o.type == "MESH" and o.parent is None:
            stray.append(o.name)
    for r in rooms:
        r["n_objects"] = per_room.get(r["name"], 0)
    return dict(sorted(roots.items())), per_room.get("none", 0), stray


def describe(rec):
    layout = house_layout()
    objects, outside, stray = object_census(layout["rooms"])
    rec.update(layout)
    rec.update({"flight_band_m": [0.4, round(layout["ceiling_height_m"] - 0.4, 3)], "objects": objects,
                "n_objects": int(sum(objects.values())), "objects_outside_rooms": outside,
                "unpopulated_placeholders": stray})
    return rec


def main():
    a = parse()
    a.out.mkdir(parents=True, exist_ok=True)
    box_path = a.out / "box.json"
    if a.layout_only:
        import bpy
        bpy.ops.wm.open_mainfile(filepath=str(a.out / "scene" / "scene.blend"))
        rec = describe(json.loads(box_path.read_text()) if box_path.exists() else {"kind": "indoors", "seed": a.seed})
        box_path.write_text(json.dumps(rec, indent=1))
        print("INDOOR_LAYOUT_DONE", len(rec["rooms"]), "rooms")
        return

    import logging

    import bpy
    from infinigen.core.constraints.example_solver import populate
    from infinigen_examples import generate_indoors

    logging.getLogger("infinigen").setLevel(logging.INFO)
    logging.getLogger("infinigen.core.nodes.node_wrangler").setLevel(logging.CRITICAL)

    # Log the asset detail Infinigen uses here: populate gives every object the fixed scatter_res_distance, and
    # target_face_size falls back to its clip minimum when no scene camera is set (none is, before populate).
    _populate = populate.populate_state_placeholders

    def populate_logged(state, filter=None, final=True):
        print(f"populate (final={final}): scene camera = {bpy.context.scene.camera}", flush=True)
        return _populate(state, filter=filter, final=final)

    populate.populate_state_placeholders = populate_logged

    configs = ([] if a.no_fast_solve else ["fast_solve.gin"]) + a.configs
    overrides = FLIGHT_OVERRIDES + (["populate_doors.all_open=True"] if a.doors == "open" else []) + a.overrides
    t0 = time.time()
    args = argparse.Namespace(seed=str(a.seed), output_folder=a.out / "scene", input_folder=None, task=["coarse"],
                              task_uniqname=None, debug=None, configs=configs, overrides=overrides)
    generate_indoors.main(args)
    rec = {"kind": "indoors", "seed": a.seed, "doors": a.doors, "configs": ["base_indoors.gin"] + configs,
           "overrides": overrides,
           "seconds": round(time.time() - t0, 1),
           "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}
    box_path.write_text(json.dumps(rec, indent=1))  # before the layout pass: --layout-only can redo that one
    describe(rec)
    box_path.write_text(json.dumps(rec, indent=1))
    print("INDOOR_BOX_DONE", json.dumps({k: v for k, v in rec.items() if k not in ("rooms", "objects")}))


if __name__ == "__main__":
    sys.exit(main())
