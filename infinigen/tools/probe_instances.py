"""Why are depsgraph instances skipped by box_views.sample_scene? Tally by reason, with example names."""
import sys, collections
from pathlib import Path
import numpy as np
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import box_views as FV

scene = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.open_mainfile(filepath=scene)
FV.show_render_visible()
deps = bpy.context.evaluated_depsgraph_get()
placeholder_objs = set()
for c in bpy.data.collections:
    if c.name.startswith("placeholders"):
        placeholder_objs.update(o.name for o in c.all_objects)
coll_hidden = {c.name for c in bpy.data.collections if c.hide_render}
obj_colls = {}
for c in bpy.data.collections:
    for o in c.objects:
        obj_colls.setdefault(o.name, []).append(c.name)

def reason(orig):
    nm = orig.name
    if any(s in nm for s in FV.SKIP_NAMES): return "skipname"
    if nm in placeholder_objs: return "placeholder_coll"
    if nm.startswith("OpaqueTerrain"): return "terrain"
    if orig.hide_render: return "hide_render"
    if any(cn in coll_hidden for cn in obj_colls.get(nm, [])): return "coll_hidden"
    return None

tally = collections.Counter(); ex = collections.defaultdict(set); sizes = collections.Counter()
for inst in deps.object_instances:
    ob = inst.object
    if ob.type != "MESH":
        tally["type:" + ob.type] += 1; continue
    r = reason(ob.original)
    key = None
    if r is None and inst.is_instance and inst.parent is not None:
        rp = reason(inst.parent.original)
        if rp is not None:
            r = "parent_" + rp
            key = inst.parent.original.name
    k = (r or "kept") + (":inst" if inst.is_instance else ":obj")
    tally[k] += 1
    if len(ex[k]) < 4: ex[k].add((ob.original.name[:60], (key or "")[:60]))
    if r is None:
        bb = np.array([list(v) for v in ob.bound_box]); M = np.array(inst.matrix_world)
        ext = float(np.ptp(bb, axis=0).max() * np.linalg.norm(M[:3, :3], axis=0).max())
        sizes["<0.25" if ext < 0.25 else ">=0.25"] += 1
for k, v in tally.most_common():
    print(f"{v:9d}  {k}  e.g. {sorted(ex[k])[:3]}")
print("kept sizes", dict(sizes))
