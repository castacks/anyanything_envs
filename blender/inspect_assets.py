"""Split every model file of the v2 library into pieces and thumbnail them.

    blender -b --factory-startup -P blender/inspect_assets.py -- \
        --root /mnt/data/blender/assets [--shard 0/4] [--thumbs] [--threads 4]
    python3 blender/inspect_assets.py --merge --root ...      # shards -> catalog.json + sheets

Another library file (e.g. ``fetch_objaverse.py``'s) goes to its own catalog:
``--library library_objaverse.json --catalog-name catalog_objaverse
--keys-prefix obj: --sheets-dir catalog/objaverse_sheets [--sheet-by spec]``
keeps its shards in ``catalog/catalog_objaverse_shards/`` and writes
``<root>/catalog_objaverse.json``; the defaults are the v2 library as before.

A model file often carries several objects: a rock set, three tree variants,
a modular facade kit. A *piece* is one root object with its children, and
roots whose bounding boxes overlap (a car body and its unparented wheels)
are merged into one piece. Per piece: the root names, size (x, y, z) in
metres, triangles, whether it has alpha-cut materials, and with ``--thumbs``
a 256 px Cycles CPU thumbnail from a three-quarter view.

Shard output: ``<root>/catalog/shard_<i>.json``; ``--merge`` (plain python,
PIL) writes ``<root>/catalog.json`` and one contact sheet per category under
``<root>/catalog/sheets/`` (each cell labelled with the piece key and size),
which is how a person or an agent chooses what a kit may place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--thumbs", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="file keys")
    ap.add_argument("--library", default="library.json", help="library file under --root")
    ap.add_argument("--catalog-name", default="catalog", help="writes <root>/<name>.json")
    ap.add_argument("--keys-prefix", default=None, help="only file keys starting with this, e.g. obj:")
    ap.add_argument("--sheets-dir", default="catalog/sheets", help="contact sheets, relative to --root")
    ap.add_argument("--sheet-by", choices=["categories", "spec"], default="categories",
                    help="one sheet per category, or per entry 'spec' (fetch_objaverse)")
    return ap.parse_args(argv)


def _shard_dir(a) -> Path:
    """``catalog/`` for the default catalog (as before), ``catalog/<name>_shards/`` otherwise."""
    return a.root / "catalog" if a.catalog_name == "catalog" else a.root / "catalog" / f"{a.catalog_name}_shards"


# --- inside Blender -----------------------------------------------------------------------------

def inspect(a) -> None:
    import bpy
    from mathutils import Vector
    import numpy as np

    lib = json.loads((a.root / a.library).read_text())["items"]
    keys = sorted(k for k, e in lib.items() if e["kind"] == "model")
    if a.only:
        keys = [k for k in keys if k in a.only]
    if a.keys_prefix:
        keys = [k for k in keys if k.startswith(a.keys_prefix)]
    i, n = map(int, a.shard.split("/"))
    keys = keys[i::n]
    out_dir = a.root / "catalog"
    (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    shard_path = _shard_dir(a) / f"shard_{i}.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(shard_path.read_text()) if shard_path.exists() else {}
    t0 = time.time()

    for j, fk in enumerate(keys):
        if fk in done and (not a.thumbs or all(Path(a.root / p["thumb"]).exists() for p in done[fk]["pieces"] if p.get("thumb"))):
            continue
        e = lib[fk]
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        try:
            p = str(a.root / e["path"])
            if p.endswith((".gltf", ".glb")):
                bpy.ops.import_scene.gltf(filepath=p)
            elif p.endswith(".fbx"):
                bpy.ops.import_scene.fbx(filepath=p)
            else:
                bpy.ops.wm.obj_import(filepath=p)
        except Exception as ex:  # noqa: BLE001
            done[fk] = {"error": f"{type(ex).__name__}: {ex}", "pieces": []}
            continue
        scale = float(e.get("scale", 1.0))
        objs = list(bpy.data.objects)
        if scale != 1.0:
            for o in objs:
                if o.parent is None:
                    o.scale = tuple(scale * s for s in o.scale)
                    o.location = tuple(scale * c for c in o.location)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        roots = [o for o in objs if o.parent is None]
        info = []
        for r in roots:
            sub = [r, *r.children_recursive]
            pts = [list(o.matrix_world @ Vector(c)) for o in sub if o.type == "MESH" for c in o.bound_box]
            if not pts:
                continue
            pts = np.array(pts)
            tris = 0
            alpha = False
            for o in sub:
                if o.type != "MESH":
                    continue
                ev = o.evaluated_get(dg)
                tris += sum(len(pl.vertices) - 2 for pl in ev.data.polygons)
                for s in o.material_slots:
                    m = s.material
                    if m and m.use_nodes:
                        b = next((nd for nd in m.node_tree.nodes if nd.type == "BSDF_PRINCIPLED"), None)
                        if b is not None and b.inputs["Alpha"].is_linked:
                            alpha = True
            info.append({"root": r.name, "lo": pts.min(0), "hi": pts.max(0), "tris": tris, "alpha": alpha})
        # union roots whose boxes overlap (1 cm slack): a body and its loose wheels are one piece
        parent = list(range(len(info)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u in range(len(info)):
            for v in range(u + 1, len(info)):
                a_, b_ = info[u], info[v]
                if np.all(a_["lo"] - 0.01 <= b_["hi"]) and np.all(b_["lo"] - 0.01 <= a_["hi"]):
                    lo = np.maximum(a_["lo"], b_["lo"]); hi = np.minimum(a_["hi"], b_["hi"])
                    inter = float(np.prod(np.maximum(hi - lo, 0.0)))
                    va = float(np.prod(np.maximum(a_["hi"] - a_["lo"], 1e-3)))
                    vb = float(np.prod(np.maximum(b_["hi"] - b_["lo"], 1e-3)))
                    if inter > 0.2 * min(va, vb) or min(va, vb) < 1e-4:
                        parent[find(u)] = find(v)
        groups = {}
        for u in range(len(info)):
            groups.setdefault(find(u), []).append(info[u])
        pieces = []
        for k, (_, g) in enumerate(sorted(groups.items(), key=lambda kv: min(x["root"] for x in kv[1]))):
            lo = np.min([x["lo"] for x in g], 0); hi = np.max([x["hi"] for x in g], 0)
            pieces.append({"index": k, "roots": sorted(x["root"] for x in g), "size": np.round(hi - lo, 3).tolist(),
                           "centre": np.round((lo + hi) / 2, 3).tolist(), "tris": int(sum(x["tris"] for x in g)),
                           "alpha": any(x["alpha"] for x in g)})
        if a.thumbs:
            thumb_pieces(bpy, a, fk, pieces, info)
        done[fk] = {"pieces": pieces, "n_roots": len(info)}
        if j % 10 == 9:
            shard_path.write_text(json.dumps(done))
            print(f"[{i}] {j + 1}/{len(keys)}  {time.time() - t0:.0f} s", flush=True)
    shard_path.write_text(json.dumps(done))
    print(f"[{i}] done {len(keys)} files in {time.time() - t0:.0f} s", flush=True)


def thumb_pieces(bpy, a, fk, pieces, info) -> None:
    """One 256 px Cycles CPU still per piece: the other pieces hidden, a grey
    floor, a sky, a three-quarter view framing the piece's box."""
    from mathutils import Vector
    import math
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 12
    scene.cycles.use_denoising = True
    scene.render.threads_mode, scene.render.threads = "FIXED", a.threads
    scene.render.resolution_x = scene.render.resolution_y = 256
    scene.render.image_settings.file_format = "JPEG"
    scene.view_settings.view_transform = "AgX"
    w = bpy.data.worlds.new("w"); scene.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.62, 0.72, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(40), 0, math.radians(30))
    scene.collection.objects.link(sun)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam); scene.camera = cam
    cam.data.lens = 50
    roots = {o.name: o for o in bpy.data.objects if o.parent is None and o.type != "LIGHT" and o is not cam}
    for p in pieces:
        keep = set(p["roots"])
        for nm, o in roots.items():
            for x in [o, *o.children_recursive]:
                x.hide_render = nm not in keep
        c = Vector(p["centre"]); s = max(p["size"]) or 1.0
        d = 1.35 * s
        eye = c + Vector((d * 0.75, -d * 0.9, d * 0.55))
        cam.location = eye
        cam.rotation_euler = (c - eye).to_track_quat("-Z", "Y").to_euler()
        cam.data.clip_end = 10 * d + 100
        rel = f"catalog/thumbs/{fk.replace(':', '_')}_{p['index']}.jpg"
        scene.render.filepath = str(a.root / rel)
        bpy.ops.render.render(write_still=True)
        p["thumb"] = rel


# --- merge (plain python) -----------------------------------------------------------------------

def merge(a) -> None:
    from PIL import Image, ImageDraw
    lib = json.loads((a.root / a.library).read_text())["items"]
    files = {}
    for f in sorted(_shard_dir(a).glob("shard_*.json")):
        files.update(json.loads(f.read_text()))
    files = {k: v for k, v in files.items() if k in lib and (not a.keys_prefix or k.startswith(a.keys_prefix))}
    out = a.root / f"{a.catalog_name}.json"
    tmp = out.with_suffix(".json.tmp")            # atomic: builds may be reading it
    tmp.write_text(json.dumps({"files": files, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1, sort_keys=True))
    tmp.replace(out)
    n_p = sum(len(v["pieces"]) for v in files.values())
    print(f"{len(files)} files, {n_p} pieces -> {out.name}")
    by_cat = {}
    for fk, v in files.items():
        cats = [lib[fk]["spec"]] if a.sheet_by == "spec" and lib[fk].get("spec") else \
            (lib[fk].get("categories", ["uncategorised"]) or ["uncategorised"])
        for c in cats:
            by_cat.setdefault(c, []).extend((fk, p) for p in v["pieces"] if p.get("thumb"))
    sheets = a.root / a.sheets_dir
    sheets.mkdir(parents=True, exist_ok=True)
    for cat, cells in sorted(by_cat.items()):
        cols, cw, ch = 8, 200, 222
        for page in range(0, len(cells), 48):
            chunk = cells[page:page + 48]
            rows = (len(chunk) + cols - 1) // cols
            im = Image.new("RGB", (cols * cw, rows * ch), (24, 24, 24))
            dr = ImageDraw.Draw(im)
            for q, (fk, p) in enumerate(chunk):
                x, y = (q % cols) * cw, (q // cols) * ch
                try:
                    t = Image.open(a.root / p["thumb"]).convert("RGB").resize((cw - 4, cw - 4))
                    im.paste(t, (x + 2, y + 2))
                except Exception:  # noqa: BLE001
                    pass
                sx, sy, sz = p["size"]
                dr.text((x + 3, y + cw - 1), f"{fk.split(':', 1)[-1]}#{p['index']}"[:30], fill=(255, 255, 255))
                dr.text((x + 3, y + cw + 10), f"{sx:.1f}x{sy:.1f}x{sz:.1f} m {p['tris'] // 1000}k", fill=(190, 190, 190))
            name = cat.replace(":", "").replace(" ", "_")
            im.save(sheets / f"{name}_{page // 48}.jpg", quality=82)
    print(f"sheets in {sheets}")
    if any(e["kind"] == "material" for e in lib.values()):
        material_sheets(a, lib)


def material_sheets(a, lib) -> None:
    """One contact sheet per material category: each scan's colour map at
    160 px, labelled with its key -- ``catalog/materials/<category>_<page>.jpg``."""
    from PIL import Image, ImageDraw
    out = a.root / "catalog" / "materials"
    out.mkdir(parents=True, exist_ok=True)
    thumbs = a.root / "catalog" / "mat_thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    by_cat = {}
    for k, e in lib.items():
        if e["kind"] != "material" or "diff" not in e.get("maps", {}):
            continue
        cats = [c for c in e.get("categories", []) if c not in ("outdoor", "man made", "natural", "indoor")] or ["other"]
        for c in cats:
            by_cat.setdefault(c, []).append(k)
    cw = 160
    for cat, keys in sorted(by_cat.items()):
        keys = sorted(keys)
        for page in range(0, len(keys), 60):
            chunk = keys[page:page + 60]
            cols = 10
            rows = (len(chunk) + cols - 1) // cols
            im = Image.new("RGB", (cols * cw, rows * (cw + 14)), (24, 24, 24))
            dr = ImageDraw.Draw(im)
            for q, k in enumerate(chunk):
                x, y = (q % cols) * cw, (q // cols) * (cw + 14)
                tp = thumbs / f"{k.replace(':', '_')}.jpg"
                try:
                    if not tp.exists():
                        Image.open(a.root / lib[k]["maps"]["diff"]).convert("RGB").resize((cw - 4, cw - 4)).save(tp, quality=80)
                    im.paste(Image.open(tp), (x + 2, y + 2))
                except Exception:  # noqa: BLE001
                    pass
                dr.text((x + 3, y + cw), k[:26], fill=(255, 255, 255))
            im.save(out / f"{cat.replace(':', '').replace(' ', '_').replace('/', '-')}_{page // 60}.jpg", quality=82)
    print(f"material sheets in {out}")


if __name__ == "__main__":
    args = _args()
    if args.merge:
        merge(args)
    else:
        inspect(args)
