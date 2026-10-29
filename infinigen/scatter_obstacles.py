"""Scatter extra obstacles through a finished nature_box scene: Wenshan-style floaters (objects at random heights
and yaw anywhere in the flight volume) and grounded clutter, made by Infinigen's own procedural asset factories
(furniture, appliances, rocks, cacti) so nothing is an asset our training data has seen.

    python infinigen/scatter_obstacles.py --scene <box>/scene/scene.blend --box <box>/box.json \
        --cloud <box>/views/cloud.npz --out <box>/floaters_80 --floaters 80 --ground 20 [--seed 1]

Placement uses the scene's collision cloud (box_views.py): a candidate is rejected if any existing obstacle point,
or an already placed obstacle, lies within its bounding radius plus --gap. Floaters sit at --agl metres over the
local ground (uniform); grounded clutter stands on the terrain. Each factory makes --unique assets; every placement
is a collection instance of one of them (random yaw, scale), so a density ladder costs seconds per level.
Writes <out>/scene.blend (a copy of the scene with the obstacles) and <out>/obstacles.json (every placement).
Run box_views.py on <out>/scene.blend for the new cloud, plan map and renders.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import pkgutil
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from scipy.spatial import cKDTree

FLOATERS = ["ChairFactory", "SofaFactory", "TableDiningFactory", "LampFactory", "TVFactory", "MicrowaveFactory",
            "DishwasherFactory", "BarChairFactory", "SimpleBookcaseBaseFactory", "BlenderRockFactory"]
GROUND = ["BoulderFactory", "BlenderRockFactory", "CactusFactory", "SofaFactory", "TableDiningFactory"]
MAX_FACES = 150_000          # heavier assets get a Decimate modifier down to about this


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--cloud", type=Path, required=True, help="the scene's views/cloud.npz (box_views.py)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--floaters", type=int, default=60)
    ap.add_argument("--ground", type=int, default=15)
    ap.add_argument("--agl", type=float, nargs=2, default=(0.8, 6.0), help="floater centre height over ground, m")
    ap.add_argument("--size", type=float, nargs=2, default=(0.6, 2.0), help="longest side after scaling, m")
    ap.add_argument("--gap", type=float, default=0.4, help="extra clearance around each placement, m")
    ap.add_argument("--floater-factories", nargs="*", default=FLOATERS)
    ap.add_argument("--ground-factories", nargs="*", default=GROUND)
    ap.add_argument("--unique", type=int, default=3, help="unique assets per factory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--in-box", action="store_true", help="place only inside the flight box (default: the populated square)")
    return ap.parse_args(argv)


def find_factories(names):
    import infinigen.assets.objects as objs
    from infinigen.core.placement.factory import AssetFactory
    found = {}
    for m in pkgutil.walk_packages(objs.__path__, objs.__name__ + "."):
        if len(found) == len(set(names)):
            break
        try:
            mod = importlib.import_module(m.name)
        except Exception:  # noqa: BLE001
            continue
        for k, v in vars(mod).items():
            if k in names and k not in found and isinstance(v, type) and issubclass(v, AssetFactory):
                found[k] = v
    return found


def hierarchy(o):
    return [o] + list(o.children_recursive)


def world_bounds(objs):
    pts = []
    for o in objs:
        if o.type == "MESH":
            pts += [o.matrix_world @ Vector(c) for c in o.bound_box]
    P = np.array([list(p) for p in pts]) if pts else np.zeros((1, 3))
    return P.min(0), P.max(0)


def make_proto(fac, name, i, holder):
    """Spawn one asset, decimate it if heavy, move it into its own (unlinked) collection to instance."""
    o = fac.spawn_asset(i)
    objs = hierarchy(o)
    faces = sum(len(c.data.polygons) for c in objs if c.type == "MESH")
    if faces > MAX_FACES:
        for c in objs:
            if c.type == "MESH" and len(c.data.polygons) > 1000:
                m = c.modifiers.new("decimate", "DECIMATE")
                m.ratio = max(0.02, MAX_FACES / faces)
    bpy.context.view_layer.update()
    lo, hi = world_bounds(objs)
    col = bpy.data.collections.new(f"obstacle_proto_{name}_{i}")
    for c in objs:
        for uc in list(c.users_collection):
            uc.objects.unlink(c)
        col.objects.link(c)
    holder.children.link(col)
    return {"col": col, "lo": lo, "hi": hi, "faces": faces, "factory": name, "i": i}


def main():
    a = parse()
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    box = json.loads(a.box.read_text())
    half, region = box["half"], box["region"]
    lim = half if a.in_box else region - 1.0
    rng = np.random.default_rng(a.seed)
    bpy.ops.wm.open_mainfile(filepath=str(a.scene))
    z = np.load(a.cloud)
    P = np.asarray(z["points"], float)
    g, GZ = np.asarray(z["grid_x"], float), np.asarray(z["ground_grid"], float)

    def ground(x, y):
        i = int(np.clip(round((x + region) / 0.5), 0, len(g) - 1))
        j = int(np.clip(round((y + region) / 0.5), 0, len(g) - 1))
        return float(GZ[j, i])

    tree = cKDTree(P)
    placed_c, placed_r = [], []

    # prototypes live in one collection that is excluded from the view layer: only their instances render
    holder = bpy.data.collections.new("obstacle_protos")
    bpy.context.scene.collection.children.link(holder)
    bpy.context.view_layer.layer_collection.children[holder.name].exclude = True
    facs = find_factories(set(a.floater_factories) | set(a.ground_factories))
    protos = {}
    for name in sorted(facs):
        protos[name] = []
        for i in range(a.unique):
            try:
                protos[name].append(make_proto(facs[name](int(rng.integers(1e7))), name, i, holder))
            except Exception as e:  # noqa: BLE001
                print(f"{name}[{i}] failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
        print(f"{name}: {len(protos[name])} prototypes", flush=True)

    out_col = bpy.data.collections.new("scattered_obstacles")
    bpy.context.scene.collection.children.link(out_col)
    records = []

    def place(kind, factories, n):
        pool = [p for f in factories for p in protos.get(f, [])]
        if not pool:
            return 0
        done = tries = 0
        while done < n and tries < 60 * n:
            tries += 1
            pr = pool[int(rng.integers(len(pool)))]
            ext = pr["hi"] - pr["lo"]
            s = float(rng.uniform(*a.size)) / max(float(ext.max()), 1e-3)
            radius = 0.5 * float(np.linalg.norm(ext)) * s + a.gap
            x, y = (float(v) for v in rng.uniform(-lim, lim, 2))
            zg = ground(x, y)
            if kind == "floater":
                zc = zg + float(rng.uniform(*a.agl))
                if zc - 0.5 * ext[2] * s < zg + 0.3:
                    continue
            else:
                zc = zg + 0.5 * ext[2] * s - 0.05
            c = np.array([x, y, zc])
            if tree.query_ball_point(c, radius, return_length=True) > 0:
                continue
            if any(np.linalg.norm(c - pc) < radius + pr_ for pc, pr_ in zip(placed_c, placed_r)):
                continue
            emp = bpy.data.objects.new(f"obstacle_{kind}_{done}", None)
            emp.instance_type = "COLLECTION"
            emp.instance_collection = pr["col"]
            pr["col"].instance_offset = Vector(list(0.5 * (pr["lo"] + pr["hi"])))
            emp.location = Vector(list(c))
            yaw = float(rng.uniform(0, 2 * math.pi))
            tilt = (float(rng.normal(0, 0.15)), float(rng.normal(0, 0.15))) if kind == "floater" else (0.0, 0.0)
            emp.rotation_euler = (tilt[0], tilt[1], yaw)
            emp.scale = (s, s, s)
            out_col.objects.link(emp)
            placed_c.append(c)
            placed_r.append(radius - a.gap)
            records.append({"kind": kind, "factory": pr["factory"], "proto": pr["i"], "centre": [round(float(v), 3) for v in c],
                            "agl": round(float(zc - zg), 3), "yaw": round(yaw, 4), "tilt": [round(v, 4) for v in tilt],
                            "scale": round(s, 4), "radius": round(radius - a.gap, 3)})
            done += 1
        return done

    nf = place("floater", a.floater_factories, a.floaters)
    ng = place("ground", a.ground_factories, a.ground)
    bpy.ops.wm.save_as_mainfile(filepath=str(a.out / "scene.blend"), copy=True)
    rec = {"source_scene": str(a.scene), "box": str(a.box), "seed": a.seed, "floaters": nf, "ground": ng,
           "requested": {"floaters": a.floaters, "ground": a.ground}, "agl": list(a.agl), "size": list(a.size),
           "gap": a.gap, "area": "flight box" if a.in_box else "populated square",
           "prototypes": {k: [{"faces": p["faces"], "dims": [round(float(v), 2) for v in p["hi"] - p["lo"]]} for p in v]
                          for k, v in protos.items()},
           "placements": records, "seconds": round(time.time() - t0, 1)}
    (a.out / "obstacles.json").write_text(json.dumps(rec, indent=1))
    print("SCATTER_DONE", json.dumps({k: rec[k] for k in ("floaters", "ground", "seconds")}))


if __name__ == "__main__":
    main()
