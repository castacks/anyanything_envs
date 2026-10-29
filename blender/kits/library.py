"""The v2 asset library as a place kit sees it (runs inside Blender).

``library.json`` (``fetch_library.py``) lists what was fetched; ``catalog.json``
(``inspect_assets.py``) splits every model file into *pieces* -- the separate
objects a file carries (a rock set is six rocks, a tree file three trees, a
modular facade kit a dozen modules) -- with each piece's size, triangle count
and thumbnail. Kits place pieces, never whole files.

Keys: ``ph:<id>`` / ``acg:<id>`` / ``obj:<uid>`` for files, ``<file key>#<n>``
for pieces. A file that is one object has the single piece ``#0``.

    lib = Library(root)
    keys = lib.models(tags_any={"tree"}, h=(4, 30))           # piece keys
    lib.material("acg:Asphalt012")                              # maps dict
    lib.hdri(rng, cats_all={"outdoor", "urban"}, cats_none={"night"})
"""

from __future__ import annotations

import json
from pathlib import Path

import bpy
from mathutils import Vector
import numpy as np


class Library:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser()
        #: the index files read, in order (``course.json`` ``library_sha`` hashes them)
        self.index_files: list[Path] = []
        # library.json (+ library_<source>.json from other fetchers, e.g. Objaverse)
        self.items = {}
        for f in [self.root / "library.json", *sorted(self.root.glob("library_*.json"))]:
            self.items.update(json.loads(f.read_text())["items"])
            self.index_files.append(f)
        self.catalog = {}
        for f in [self.root / "catalog.json", *sorted(self.root.glob("catalog_*.json"))]:
            if f.exists():
                self.catalog.update(json.loads(f.read_text())["files"])
                self.index_files.append(f)
        self._loaded: dict[str, dict] = {}          # file key -> {piece index -> {"coll", "size"}}
        self.used_files: set[str] = set()

    # --- queries ------------------------------------------------------------------------------

    def _words(self, e: dict) -> set[str]:
        return set(e.get("categories", [])) | set(e.get("tags", []))

    def _match(self, e, cats_any, cats_all, cats_none, tags_any, source) -> bool:
        cats, words = set(e.get("categories", [])), self._words(e)
        if source and e.get("source") != source:
            return False
        if cats_all and not set(cats_all) <= cats:
            return False
        if cats_any and not set(cats_any) & cats:
            return False
        if cats_none and set(cats_none) & words:
            return False
        if tags_any and not set(tags_any) & words:
            return False
        return True

    def files(self, kind: str, cats_any=(), cats_all=(), cats_none=(), tags_any=(), source=None) -> list[str]:
        return sorted(k for k, e in self.items.items()
                      if e["kind"] == kind and self._match(e, cats_any, cats_all, cats_none, tags_any, source))

    def pieces(self, fkey: str) -> list[dict]:
        """The catalogued pieces of a model file (``[]`` if not inspected yet)."""
        return self.catalog.get(fkey, {}).get("pieces", [])

    def models(self, ids=None, cats_any=(), cats_all=(), cats_none=(), tags_any=(), source=None,
               h=None, w=None, max_tris=None, exclude=()) -> list[str]:
        """Piece keys matching the file filters and the piece's own size:
        ``h`` = (lo, hi) height in metres, ``w`` = (lo, hi) largest footprint
        side. ``ids`` restricts to those files (bare Poly Haven ids allowed)."""
        if ids is not None:
            fkeys = [i if ":" in i else f"ph:{i}" for i in ids]
            fkeys = [k for k in fkeys if k in self.items]
        else:
            fkeys = self.files("model", cats_any, cats_all, cats_none, tags_any, source)
        out = []
        for fk in fkeys:
            if fk in exclude or self.items[fk].get("exclude"):
                continue
            for p in self.pieces(fk):
                sx, sy, sz = p["size"]
                if p.get("exclude"):
                    continue
                if h and not (h[0] <= sz <= h[1]):
                    continue
                if w and not (w[0] <= max(sx, sy) <= w[1]):
                    continue
                if max_tris and p.get("tris", 0) > max_tris:
                    continue
                out.append(f"{fk}#{p['index']}")
        return out

    def size(self, pkey: str) -> list[float]:
        fk, i = pkey.split("#")
        return next(p["size"] for p in self.pieces(fk) if p["index"] == int(i))

    def material(self, key: str) -> dict:
        """``{"maps": {...}, "tile_m": ...}`` of a material (``ph:`` or ``acg:``)."""
        e = self.items[key]
        self.used_files.add(key)
        return {"maps": {k: str(self.root / v) for k, v in e["maps"].items()}, "tile_m": e.get("tile_m"),
                "key": key}

    def materials(self, cats_any=(), cats_all=(), cats_none=(), tags_any=(), source=None) -> list[str]:
        return [k for k in self.files("material", cats_any, cats_all, cats_none, tags_any, source)
                if "diff" in self.items[k].get("maps", {})]

    def hdris(self, cats_any=(), cats_all=(), cats_none=(), tags_any=()) -> list[str]:
        return self.files("hdri", cats_any, cats_all, cats_none, tags_any)

    def pick(self, rng, keys):
        return str(keys[int(rng.integers(len(keys)))]) if len(keys) else None

    def path(self, key: str) -> Path:
        self.used_files.add(key)
        return self.root / self.items[key]["path"]

    # --- loading ------------------------------------------------------------------------------

    def load(self, pkey: str, lib_coll) -> dict:
        """Import the piece's file once (into collections under the excluded
        ``lib_coll``) and return ``{"coll", "size"}`` for the piece: its own
        collection, offset so its footprint centre sits on the instance origin
        and its base on z = 0. ``size`` is the piece's (x, y, z) in metres
        after the file's ``scale`` (Objaverse items carry one)."""
        fk, idx = pkey.split("#")
        idx = int(idx)
        if fk not in self._loaded:
            self._loaded[fk] = self._import(fk, lib_coll)
        return self._loaded[fk][idx]

    def _import(self, fk: str, lib_coll) -> dict:
        from build_course import attach_alpha  # noqa: PLC0415 (bpy-side helper)
        e = self.items[fk]
        self.used_files.add(fk)
        before = set(bpy.data.objects)
        path = str(self.root / e["path"])
        if path.lower().endswith((".glb", ".gltf")):
            bpy.ops.import_scene.gltf(filepath=path)
        elif path.lower().endswith(".fbx"):
            bpy.ops.import_scene.fbx(filepath=path)
        elif path.lower().endswith(".obj"):
            bpy.ops.wm.obj_import(filepath=path)
        else:
            raise ValueError(f"cannot import {path}")
        new = [o for o in bpy.data.objects if o not in before]
        if e.get("alpha_maps"):
            attach_alpha(new, e["alpha_maps"], self.root)
        scale = float(e.get("scale", 1.0))
        if scale != 1.0:
            for o in new:
                if o.parent is None:
                    o.scale = tuple(scale * s for s in o.scale)
                    o.location = tuple(scale * c for c in o.location)
        bpy.context.view_layer.update()
        groups = self.catalog.get(fk, {}).get("pieces")
        by_name = {o.name: o for o in new}
        if not groups:                                  # not inspected: the whole file is one piece
            groups = [{"index": 0, "roots": [o.name for o in new if o.parent is None]}]
        out = {}
        for g in groups:
            roots = [by_name[n] for n in g["roots"] if n in by_name]
            if not roots:
                roots = [by_name[n] for n in by_name if n.split(".")[0] in {r.split(".")[0] for r in g["roots"]}]
            objs = []
            for r in roots:
                objs += [r, *r.children_recursive]
            coll = bpy.data.collections.new(f"lib_{fk.replace(':', '_')}_{g['index']}")
            lib_coll.children.link(coll)
            for o in objs:
                for c in list(o.users_collection):
                    c.objects.unlink(o)
                coll.objects.link(o)
            pts = np.array([list(o.matrix_world @ Vector(c)) for o in objs if o.type == "MESH" for c in o.bound_box])
            if not len(pts):
                continue
            lo, hi = pts.min(0), pts.max(0)
            coll.instance_offset = Vector(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]))
            out[g["index"]] = {"coll": coll, "size": (hi - lo).tolist(), "key": f"{fk}#{g['index']}"}
        return out

    def prune(self, used_pieces: set) -> None:
        """Delete every imported piece no instance uses."""
        for fk, pieces in self._loaded.items():
            for i, p in pieces.items():
                if p["key"] in used_pieces:
                    continue
                for o in list(p["coll"].objects):
                    bpy.data.objects.remove(o, do_unlink=True)
                bpy.data.collections.remove(p["coll"])
        bpy.data.orphans_purge(do_recursive=True)

    def files_used(self) -> list[dict]:
        """The library files this course reads (so a render node can fetch only these)."""
        out = []
        for k in sorted(self.used_files):
            e = self.items[k]
            if e["kind"] == "model":                     # a glTF brings its .bin and textures along
                paths = [str(Path(e["path"]).parent)]
            else:
                paths = [e["path"]] if "path" in e else list(e.get("maps", {}).values())
            paths += list(e.get("alpha_maps", {}).values())
            out.append({"key": k, "source": e.get("source"), "licence": e.get("licence"), "url": e.get("url"),
                        "paths": paths})
        return out
