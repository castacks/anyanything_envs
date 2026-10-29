"""What every place kit builds with (runs inside Blender): the ``Ctx``.

A kit turns a patch spec into a *place* -- a street, a container yard, a
pine forest -- around the patch's own obstacles. ``build_course.py --kit
<name>`` makes a ``Ctx`` and calls ``kits.<name>.build(ctx)``; the kit
lights the scene, lays the ground, dresses the patch primitives, and adds
structure (buildings, racks, fences), props and far scenery through the
methods here, which keep the labeller's contract for it:

**The cloud zone.** ``ctx.zone`` is the patch bounds grown by
``CLOUD_MARGIN`` (6 m). Everything that stands inside the zone collides
and is in the cloud (up to ``CLOUD_ZMAX``, 4.75 m: the v7 envelope, see
``build_course.CLOUD_ZMAX``); anything wholly outside is scenery,
render-only. ``instance``/``box``/... decide this from the footprint, so a
kit never sets it by hand: a facade row 3 m outside the patch is a wall
the labeller sees, the skyline 30 m out is a picture. ``finish_frames``
checks cloud completeness only inside the zone for the same reason.

**The floor.** The ground is the plane z = 0 (the labeller's floor grid).
Flat overlays (roads, lawns, markings) sit < 5 cm above it and never
collide. Anything taller than 5 cm inside the zone collides.

**Free space.** Every placement is checked against ``ctx.foot`` (patch
solids, the approach to every opening, the start and the goal) and claims
its own footprint. The start (origin) and the patch goal are always kept
clear by ``START_CLEAR`` metres (more with ``--start-keepout``).

**Collections.** ``obstacles`` holds the patch primitives only (the
camera-parity render shows just these and the ground); ``structure``,
``scatter``, ``backdrop``, ``ground``, ``lights`` the rest.
"""

from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Vector
import numpy as np

import build_course as BC

CLOUD_MARGIN = 6.0
#: the cloud's top (``build_course.CLOUD_ZMAX``): a kit that decides what collides by
#: height (a crown above it, a tank split at it) must use this, not a copy of the number
CLOUD_ZMAX = BC.CLOUD_ZMAX
START_CLEAR = 1.5
COLLS = ("obstacles", "structure", "scatter", "backdrop", "ground", "lights")


class Ctx:
    def __init__(self, spec: dict, lib, looks, rng, knobs: dict, level: str = ""):
        self.spec, self.lib, self.looks, self.rng = spec, lib, looks, rng
        self.knobs, self.level = knobs, level
        self.bounds = tuple(float(v) for v in spec["bounds"])
        x0, y0, x1, y1 = self.bounds
        m = CLOUD_MARGIN
        self.zone = (x0 - m, y0 - m, x1 + m, y1 + m)
        self.start = np.zeros(2)
        self.goal = np.asarray(spec["goal_pos"][:2], float)
        self.scene = bpy.context.scene
        self.colls = {}
        for name in COLLS:
            c = bpy.data.collections.new(name)
            self.scene.collection.children.link(c)
            self.colls[name] = c
        self.lib_coll = bpy.data.collections.new("library")
        self.scene.collection.children.link(self.lib_coll)
        bpy.context.view_layer.layer_collection.children["library"].exclude = True
        self.foot: list[dict] = []
        self.placed: list[dict] = []
        #: the piece catalogue (blender/piece_ids.py): one row per placed primitive / kit object, the object
        #: stamped with ``ob["piece"] = id``; read by build_course --piece-ids, inert otherwise
        self.pieces: list[dict] = []
        self.used_pieces: set[str] = set()
        self.info: dict = {"kit": None, "notes": []}
        self.n = 0
        r = max(START_CLEAR, float(knobs.get("start_keepout", 0.0)))
        self.claim_disc(self.start, r, kind="start")
        self.claim_disc(self.goal, r, kind="goal")

    # --- geometry of the patch -------------------------------------------------------------

    @property
    def size(self):
        x0, y0, x1, y1 = self.bounds
        return x1 - x0, y1 - y0

    @property
    def area_f(self) -> float:
        """Patch area over 450 m^2: scale counts by this (and by ``clutter``)."""
        w, h = self.size
        return w * h / 450.0

    def count(self, per_450: float) -> int:
        """A scatter count for this patch: ``per_450`` per 450 m^2 at clutter 1."""
        return int(round(per_450 * self.area_f * float(self.knobs.get("clutter", 1.0))))

    def in_zone(self, xy, pad: float = 0.0) -> bool:
        x0, y0, x1, y1 = self.zone
        return x0 - pad <= xy[0] <= x1 + pad and y0 - pad <= xy[1] <= y1 + pad

    def rect_meets_zone(self, lo, hi) -> bool:
        x0, y0, x1, y1 = self.zone
        return not (hi[0] < x0 or lo[0] > x1 or hi[1] < y0 or lo[1] > y1)

    def uniform_xy(self, rect=None, pad: float = 0.5) -> np.ndarray:
        x0, y0, x1, y1 = rect or self.bounds
        return np.array([self.rng.uniform(x0 + pad, x1 - pad), self.rng.uniform(y0 + pad, y1 - pad)])

    # --- free space --------------------------------------------------------------------------

    def claim_disc(self, c, r, kind="disc"):
        self.foot.append({"type": "disc", "c": np.asarray(c, float)[:2], "r": float(r), "kind": kind})

    def claim_rect(self, c, half, yaw: float = 0.0, kind="rect"):
        cy, sy = math.cos(yaw), math.sin(yaw)
        self.foot.append({"type": "rect", "c": np.asarray(c, float)[:2], "half": np.asarray(half, float)[:2],
                          "R": np.array([[cy, -sy], [sy, cy]]), "kind": kind})

    def clearance(self, xy) -> float:
        """Distance from ``xy`` to the nearest claimed footprint (negative inside)."""
        xy = np.asarray(xy, float)[:2]
        return min((BC.footprint_distance(xy, f) for f in self.foot), default=1e9)

    def free(self, xy, r: float, margin: float = 0.4) -> bool:
        return self.clearance(xy) >= r + margin

    def free_rect(self, c, half, yaw=0.0, margin=0.3, step=0.5) -> bool:
        """Every point of a rectangle (sampled every ``step`` m) is ``margin`` clear."""
        c = np.asarray(c, float)[:2]
        hx, hy = half[0], half[1]
        cy, sy = math.cos(yaw), math.sin(yaw)
        R = np.array([[cy, -sy], [sy, cy]])
        us = np.linspace(-hx, hx, max(2, int(2 * hx / step) + 1))
        vs = np.linspace(-hy, hy, max(2, int(2 * hy / step) + 1))
        for u in us:
            for v in vs:
                if self.clearance(c + R @ np.array([u, v])) < margin:
                    return False
        return True

    # --- the patch primitives -------------------------------------------------------------------

    def dress_primitives(self, mats: dict, keepout: float | None = None) -> None:
        """Build the patch's obstacles (exact geometry) in the kit's materials:
        ``mats`` has ``wall``, ``box`` (default wall), ``pole``, ``bar``, ``rock``."""
        mats = {**mats, "box": mats.get("box", mats["wall"])}
        ko = float(self.knobs.get("keepout", 3.0) if keepout is None else keepout)
        before = len(self.foot)
        # build_course.build_primitives dresses boxes with "wall"; swap in "box" after
        foot = BC.build_primitives(self.spec["primitives"], self.colls["obstacles"], mats, keepout=ko)
        for o in self.colls["obstacles"].objects:
            if "_box" in o.name and mats["box"] is not mats["wall"]:
                o.data.materials[0] = mats["box"]
        for f in foot:
            f.setdefault("kind", "prim")
        self.foot.extend(foot)
        self.info["primitives"] = len(self.spec["primitives"])
        # one piece id per primitive (a slotted wall is several boxes named prim<i>_wall<j>)
        for o in self.colls["obstacles"].objects:
            if o.get("piece") is None and o.name.startswith("prim"):
                try:
                    i = int(o.name[4:6])
                except ValueError:
                    continue
                prim = self.spec["primitives"][i] if i < len(self.spec["primitives"]) else {}
                row = next((r for r in self.pieces if r["kind"] == "prim" and r.get("prim") == i), None)
                if row is None:
                    row = self._piece(o, "prim", cls="insert", key=prim.get("kind"), prim=i)
                else:
                    o["piece"] = row["id"]
        assert len(self.foot) >= before

    # --- adding geometry ------------------------------------------------------------------------

    def ground(self, mat, half: float = 2000.0):
        """The floor: one plane at z = 0, ``half`` metres each way from the
        patch centre (well past the horizon), non-colliding (the labeller's
        floor grid stands for it)."""
        x0, y0, x1, y1 = self.bounds
        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=half)
        bmesh.ops.translate(bm, vec=Vector(((x0 + x1) / 2, (y0 + y1) / 2, 0.0)), verts=bm.verts)
        me = bpy.data.meshes.new("ground_plane")
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new("ground_plane", me)
        ob.data.materials.append(mat)
        ob["collide"] = 0
        self.colls["ground"].objects.link(ob)
        return ob

    def _coll(self, coll):
        return self.colls[coll] if isinstance(coll, str) else coll

    def _tag(self, ob, lo, hi, collide, roof=False, kind="object", cls="building", key=None):
        if collide is None:
            collide = self.rect_meets_zone(lo, hi) and hi[2] > 0.05
        ob["collide"] = int(bool(collide))
        if roof:
            ob["roof"] = 1
        self._piece(ob, kind, cls=cls, key=key, lo=lo, hi=hi)
        return ob

    def _piece(self, ob, kind: str, cls: str = "building", key=None, lo=None, hi=None, **extra) -> dict:
        """Stamp ``ob`` with the next piece id and keep its catalogue row (piece_ids.py)."""
        row = {"id": len(self.pieces), "name": ob.name, "kind": str(kind), "cls": cls, "key": key,
               "collide": int(ob.get("collide", 0)), **extra}
        if lo is not None and hi is not None:
            row["bbox"] = [[round(float(v), 3) for v in lo], [round(float(v), 3) for v in hi]]
        ob["piece"] = row["id"]
        self.pieces.append(row)
        return row

    def box(self, centre, half, mat, yaw: float = 0.0, coll="structure", collide=None, name="box",
            roof=False, claim: bool = False):
        """An axis-aligned-in-yaw box (centre, half extents). ``claim`` adds its footprint."""
        cy, sy = math.cos(yaw), math.sin(yaw)
        R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
        self.n += 1
        ob = BC.add_box(f"{name}_{self.n:04d}", centre, half, R, self._coll(coll), mat)
        c, h = np.asarray(centre, float), np.asarray(half, float)
        ext = np.abs(R) @ h
        self._tag(ob, c - ext, c + ext, collide, roof, kind=name)
        if claim:
            self.claim_rect(c, h, yaw, kind=name)
        return ob

    def cylinder(self, a, b, r, mat, coll="structure", collide=None, name="cyl", segments=24, claim=False):
        self.n += 1
        ob = BC.add_cylinder(f"{name}_{self.n:04d}", a, b, r, self._coll(coll), mat, segments=segments)
        a_, b_ = np.asarray(a, float), np.asarray(b, float)
        lo, hi = np.minimum(a_, b_) - r, np.maximum(a_, b_) + r
        self._tag(ob, lo, hi, collide, kind=name)
        if claim:
            self.claim_disc((a_ + b_)[:2] / 2, r + 0.5 * float(np.linalg.norm((b_ - a_)[:2])), kind=name)
        return ob

    def sphere(self, c, r, mat, coll="scatter", collide=None, name="sph", claim=False):
        self.n += 1
        ob = BC.add_sphere(f"{name}_{self.n:04d}", c, r, self._coll(coll), mat)
        c_ = np.asarray(c, float)
        self._tag(ob, c_ - r, c_ + r, collide, kind=name)
        if claim:
            self.claim_disc(c_, r, kind=name)
        return ob

    def mesh(self, verts, faces, mat, uvs=None, coll="structure", collide=None, name="mesh", roof=False,
             smooth=False):
        """A mesh from world-space vertices and faces; ``uvs`` per face-corner
        (list per face) for materials that read UVs (alpha-cut cards)."""
        self.n += 1
        me = bpy.data.meshes.new(f"{name}_{self.n:04d}")
        me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(f) for f in faces])
        if uvs is not None:
            layer = me.uv_layers.new(name="UVMap")
            k = 0
            for poly, fuv in zip(me.polygons, uvs):
                for li, uv in zip(poly.loop_indices, fuv):
                    layer.data[li].uv = (float(uv[0]), float(uv[1]))
                k += 1
        me.update()
        if smooth:
            for p in me.polygons:
                p.use_smooth = True
        ob = bpy.data.objects.new(me.name, me)
        ob.data.materials.append(mat)
        self._coll(coll).objects.link(ob)
        V = np.asarray(verts, float)
        self._tag(ob, V.min(0), V.max(0), collide, roof, kind=name)
        return ob

    def panel(self, p0, p1, z0, z1, mat, tile_m=2.0, coll="structure", collide=None, name="panel",
              thickness: float = 0.0):
        """A vertical rectangle from ``p0`` to ``p1`` (xy) between heights
        ``z0``..``z1``, UVs in units of ``tile_m`` (so alpha-cut fence and net
        scans tile at their real size); ``thickness`` > 0 makes it a slab."""
        p0, p1 = np.asarray(p0, float)[:2], np.asarray(p1, float)[:2]
        L = float(np.linalg.norm(p1 - p0))
        if thickness > 0:
            d = (p1 - p0) / max(L, 1e-9)
            c = (p0 + p1) / 2
            return self.box((c[0], c[1], (z0 + z1) / 2), (L / 2, thickness / 2, (z1 - z0) / 2), mat,
                            yaw=math.atan2(d[1], d[0]), coll=coll, collide=collide, name=name)
        V = [(*p0, z0), (*p1, z0), (*p1, z1), (*p0, z1)]
        u1, v1 = L / tile_m, (z1 - z0) / tile_m
        return self.mesh(V, [(0, 1, 2, 3)], mat, uvs=[[(0, 0), (u1, 0), (u1, v1), (0, v1)]], coll=coll,
                         collide=collide, name=name)

    def flat(self, poly_xy, mat, z: float = 0.003, name="overlay"):
        """A flat, non-colliding ground overlay (road, lawn, painted line) at
        height ``z`` (< 5 cm), from a convex or simple polygon in xy."""
        assert z < 0.05
        V = [(float(x), float(y), z) for x, y in poly_xy]
        return self.mesh(V, [tuple(range(len(V)))], mat, coll="ground", collide=False, name=name)

    def strip(self, a, b, width, mat, z=0.003, name="strip"):
        """A flat rectangle of ``width`` along the segment a -> b (roads, lines)."""
        a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([-d[1], d[0]]) * width / 2
        return self.flat([a - n, b - n, b + n, a + n], mat, z=z, name=name)

    # --- library pieces ----------------------------------------------------------------------

    def piece(self, pkey: str) -> dict:
        return self.lib.load(pkey, self.lib_coll)

    def piece_class(self, pkey: str) -> str:
        """``tree`` when the library file carries a tree category / tag, else ``prop`` (the catalogue)."""
        fk = pkey.split("#")[0]
        e = getattr(self.lib, "items", {}).get(fk, {}) or {}
        words = {str(w).lower() for w in (list(e.get("categories", [])) + list(e.get("tags", [])))}
        return "tree" if any("tree" in w for w in words) else "prop"

    def instance(self, pkey: str, xy, z: float = 0.0, yaw: float = 0.0, s: float = 1.0, coll="scatter",
                 collide=None, name=None, claim: bool = False, margin: float = 0.0, tilt=(0.0, 0.0)):
        """Place a library piece: footprint centre at ``xy``, base at ``z``.
        ``claim`` adds its footprint (as a disc of its half-diagonal + margin)."""
        p = self.piece(pkey)
        self.used_pieces.add(p["key"])
        self.n += 1
        e = bpy.data.objects.new(name or f"i{self.n:04d}_{pkey.split(':')[1].split('#')[0][:24]}", None)
        e.instance_type, e.instance_collection = "COLLECTION", p["coll"]
        e.location = (float(xy[0]), float(xy[1]), float(z))
        e.rotation_euler = (float(tilt[0]), float(tilt[1]), float(yaw))
        e.scale = (s, s, s)
        self._coll(coll).objects.link(e)
        sx, sy, sz = (np.asarray(p["size"]) * s).tolist()
        r = 0.5 * math.hypot(sx, sy)
        lo = np.array([xy[0] - r, xy[1] - r, z]); hi = np.array([xy[0] + r, xy[1] + r, z + sz])
        self._tag(e, lo, hi, collide, kind="piece", cls=self.piece_class(p["key"]), key=p["key"])
        if claim:
            self.claim_disc(xy, r + margin, kind="piece")
        self.placed.append({"piece": p["key"], "xy": [round(float(xy[0]), 3), round(float(xy[1]), 3)],
                            "z": round(float(z), 3), "yaw": round(float(yaw), 3), "scale": round(float(s), 3),
                            "size": [round(v, 2) for v in (sx, sy, sz)], "collide": e["collide"]})
        return e

    def scatter(self, pkeys, n: int, rect=None, margin: float = 0.5, s=(0.9, 1.1), tries: int = 300,
                yaw=None, coll="scatter", avoid_start: bool = True, where=None) -> list:
        """Rejection-sample ``n`` placements of random pieces from ``pkeys``
        inside ``rect`` (default: the patch bounds), clear of every footprint
        by ``margin``; each claims its footprint. ``where(xy) -> bool`` can
        restrict further (e.g. only on the lawn). Returns the objects."""
        out = []
        if not pkeys:
            return out
        for _ in range(n):
            for _t in range(tries):
                k = self.lib.pick(self.rng, pkeys)
                sc = float(self.rng.uniform(*s)) if isinstance(s, tuple) else float(s)
                sx, sy, _ = np.asarray(self.lib.size(k)) * sc
                r = 0.5 * math.hypot(sx, sy)
                xy = self.uniform_xy(rect, pad=min(r, 1.0))
                if where is not None and not where(xy):
                    continue
                if not self.free(xy, r, margin):
                    continue
                y = float(self.rng.uniform(0, 2 * math.pi)) if yaw is None else float(yaw)
                out.append(self.instance(k, xy, 0.0, y, sc, coll=coll, claim=True))
                break
        return out

    def row(self, pkeys, a, b, spacing: float, offset: float = 0.0, jitter: float = 0.0, yaw_off: float = 0.0,
            s=(0.95, 1.05), coll="scatter", check: bool = True, margin: float = 0.3, skip: float = 0.0) -> list:
        """Pieces every ``spacing`` metres along a -> b (lamps along a kerb,
        trees in a row), ``offset`` to the left, facing along the row +
        ``yaw_off``. Skips spots that are not free (``check``) and a random
        ``skip`` share of them."""
        a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
        L = float(np.linalg.norm(b - a))
        if L < 1e-6 or not pkeys:
            return []
        d = (b - a) / L
        nrm = np.array([-d[1], d[0]])
        out = []
        for t in np.arange(spacing / 2, L, spacing):
            if skip and self.rng.random() < skip:
                continue
            xy = a + d * (t + self.rng.normal(0, jitter)) + nrm * offset
            k = self.lib.pick(self.rng, pkeys)
            sc = float(self.rng.uniform(*s)) if isinstance(s, tuple) else float(s)
            sx, sy, _ = np.asarray(self.lib.size(k)) * sc
            r = 0.5 * math.hypot(sx, sy)
            if check and not self.free(xy, r, margin):
                continue
            out.append(self.instance(k, xy, 0.0, math.atan2(d[1], d[0]) + yaw_off, sc, coll=coll, claim=True))
        return out

    # --- light --------------------------------------------------------------------------------

    def area_light(self, xyz, size: float, energy: float, rgb=(1.0, 0.95, 0.88), direction=None, shape="SQUARE"):
        L = bpy.data.lights.new(f"area_{self.n}", "AREA")
        L.size, L.energy, L.color, L.shape = size, energy, rgb, shape
        ob = bpy.data.objects.new(L.name, L)
        ob.location = tuple(map(float, xyz))
        if direction is not None:
            ob.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()
        self.colls["lights"].objects.link(ob)
        self.n += 1
        return ob

    def point_light(self, xyz, energy: float, radius: float = 0.1, rgb=(1.0, 0.85, 0.65)):
        L = bpy.data.lights.new(f"point_{self.n}", "POINT")
        L.energy, L.shadow_soft_size, L.color = energy, radius, rgb
        ob = bpy.data.objects.new(L.name, L)
        ob.location = tuple(map(float, xyz))
        self.colls["lights"].objects.link(ob)
        self.n += 1
        return ob

    def sun(self, elevation_deg: float, azimuth_deg: float, strength: float, angle_deg: float = 0.6,
            rgb=(1.0, 0.96, 0.9)):
        L = bpy.data.lights.new("sun", "SUN")
        L.energy, L.angle, L.color = strength, math.radians(angle_deg), rgb
        ob = bpy.data.objects.new("sun", L)
        el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
        d = Vector((-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)))
        ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        self.colls["lights"].objects.link(ob)
        return ob

    # --- bookkeeping --------------------------------------------------------------------------

    def note(self, s: str) -> None:
        self.info["notes"].append(s)

    def finish(self) -> dict:
        """Drop unused library pieces; report what was built."""
        self.lib.prune(self.used_pieces)
        coll_counts = {k: len(c.all_objects) for k, c in self.colls.items()}
        n_coll = sum(int(o.get("collide", 0)) for c in self.colls.values() for o in c.objects)
        return {**self.info, "zone": [round(v, 3) for v in self.zone], "cloud_margin_m": CLOUD_MARGIN,
                "objects": coll_counts, "colliding_objects": n_coll, "placed": self.placed, "pieces": len(self.pieces),
                "pieces_used": sorted(self.used_pieces)}
