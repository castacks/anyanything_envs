"""Where a scene's per-frame Cycles sync cost comes from: the depsgraph instances Cycles walks on every render,
grouped by instancer and instanced object, split into inside the flight box / the rest of the populated square /
beyond it, with their sizes. --backdrop adds box_views' render-time backdrop first (flight frames need it too).

    python infinigen/tools/probe_cost.py --scene S/scene/scene.blend --box S/box.json \
        [--band S/views_ae/band.json --backdrop] [--out probe.json] [--top 40]
"""
import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import box_views as FV  # noqa: E402


def family(name):
    n = re.sub(r"\.\d{3,}$", "", name)
    n = re.sub(r"\(\d+\)", "", n)
    n = re.sub(r"[_.]\d+(?=$|[_.])", "", n)
    return n[:56]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--box", required=True)
    ap.add_argument("--band", default=None)
    ap.add_argument("--backdrop", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args(argv)
    box = json.loads(Path(a.box).read_text())
    half, region = box["half"], box["region"]
    t0 = time.time()
    bpy.ops.wm.open_mainfile(filepath=a.scene)
    FV.show_render_visible()
    for o in bpy.data.objects:
        if o.name.startswith("liquid"):
            o.hide_render = o.hide_viewport = True
    load_s = time.time() - t0
    backdrop = None
    if a.backdrop:
        band = json.loads(Path(a.band).read_text())
        zg = FV.ground_fn(FV.terrain_objects())
        td = band["assets_in_box"].get("TreeFactory", 0) / band["box_area_m2"]
        backdrop = FV.add_backdrop(zg, box.get("terrain_bound", region + 4.0), region, np.random.default_rng(0), td)
        bpy.context.view_layer.update()
    t1 = time.time()
    deps = bpy.context.evaluated_depsgraph_get()
    eval_s = time.time() - t1
    groups = collections.defaultdict(lambda: {"n": 0, "box": 0, "square": 0, "beyond": 0, "ext": []})
    bbox_cache = {}
    tot = collections.Counter()
    t2 = time.time()
    for inst in deps.object_instances:
        ob = inst.object
        tot["all"] += 1
        tot["type:" + ob.type] += 1
        if inst.is_instance:
            tot["instances"] += 1
            par = inst.parent.original.name if inst.parent is not None else "?"
            key = (family(par), family(ob.original.name))
        else:
            key = ("-", family(ob.original.name))
        M = inst.matrix_world
        x, y = M[0][3], M[1][3]
        g = groups[key]
        g["n"] += 1
        r = max(abs(x), abs(y))
        g["box" if r <= half else ("square" if r <= region else "beyond")] += 1
        if len(g["ext"]) < 2000 and ob.type == "MESH":
            nm = ob.name
            if nm not in bbox_cache:
                bb = np.array([list(v) for v in ob.bound_box])
                bbox_cache[nm] = np.ptp(bb, axis=0)
            s = max(M.col[0].xyz.length, M.col[1].xyz.length, M.col[2].xyz.length)
            g["ext"].append(float(bbox_cache[nm].max() * s))
    walk_s = time.time() - t2
    rows = []
    for (par, obn), g in groups.items():
        e = np.array(g["ext"]) if g["ext"] else np.zeros(1)
        rows.append({"instancer": par, "object": obn, "n": g["n"], "box": g["box"], "square": g["square"],
                     "beyond": g["beyond"], "ext_med": round(float(np.median(e)), 3),
                     "ext_p90": round(float(np.percentile(e, 90)), 3)})
    rows.sort(key=lambda r: -r["n"])
    byinst = collections.Counter()
    for r in rows:
        byinst[r["instancer"]] += r["n"]
    rec = {"scene": a.scene, "load_s": round(load_s, 1), "eval_s": round(eval_s, 1), "walk_s": round(walk_s, 1),
           "totals": dict(tot), "backdrop": backdrop, "by_instancer": byinst.most_common(a.top), "rows": rows}
    for r in rows[:a.top]:
        print(f"{r['n']:9d}  box {r['box']:8d} sq {r['square']:8d} out {r['beyond']:8d}  ext {r['ext_med']:6.2f}/"
              f"{r['ext_p90']:6.2f}  {r['instancer']}  <-  {r['object']}")
    print("by instancer:")
    for k, v in byinst.most_common(a.top):
        print(f"{v:9d}  {k}")
    print("PROBE_COST", json.dumps({k: rec[k] for k in ("scene", "load_s", "eval_s", "walk_s", "totals", "backdrop")}),
          flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
