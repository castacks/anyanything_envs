"""Private builders for the ``industrial`` and ``construction`` kits.

Everything here builds through ``ctx`` (``ctx.mesh``/``ctx.claim_*``), so the
labeller contract of ``kits.base`` holds: a part that meets the cloud zone
collides, a part wholly outside it is render-only scenery.

* ``Batch`` gathers boxes, cylinders, cones, domes and panels into a few
  meshes (one per material and zone class), so a pipe rack or a scaffold is a
  handful of objects instead of hundreds. Every part is routed by its own
  bounding box: ``far`` (wholly outside the zone: backdrop, non-colliding),
  ``high`` (in the zone but wholly above ``ZSPLIT``: above the cloud band,
  non-colliding) or ``near`` (collides). Tall vertical boxes and cylinders in
  the zone are split at ``ZSPLIT`` so the cloud sampler never integrates a
  12 m tank wall it then throws away.
* Structural members (I-beams, hollow sections), tanks, sheds, fences,
  masts, stairs and decals (oil stains, worn paint) used by both kits.
* **The family's own space** (``reserve_core``): both kits claim it before
  they place anything, so no dressing enters a wall slot's approach, the
  space under / over a floating box or bar, the inside of a ring, room or
  stand, the strip between a lintel and its bars, or the 1 m go-around at a
  wall end, a pillar or a slab. Racks stop short of patch primitives
  (``free_runs``) and overhead runs keep clear of that space
  (``seg_clear``). What a kit still adds *on* a primitive (wall trim within
  0.12 m of its faces, copings only on walls too tall to fly over) is listed
  in the build's ``semantics`` record.
"""

from __future__ import annotations

import math

import bpy
import numpy as np

import build_course as BC

#: faces wholly above this never reach the cloud (it follows the cloud top, 4.5 m or v7's 4.75 m)
ZSPLIT = float(BC.CLOUD_ZMAX) + 0.1
UP = np.array([0.0, 0.0, 1.0])
#: the tallest wall top the v7 drone still clears (flight z 1.75 + climb-max 2.5 - r_body 0.35):
#: a coping or starter bars go only on walls at least this tall, where they change no route
REACH_TOP = BC._band.current().reach_top_v7      # band.Band.reach_top_v7: ceiling - 0.35 (v9: 3.9)
CORE_PAD = 0.3          # around the family's face-span rectangles
END_R = 1.0             # go-around kept clear at a wall end inside the patch, round a pillar / slab / float
CORE_KINDS = ("prim", "core", "start", "goal")


def rotz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def frame(d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed (x, y, d) with d the unit axis."""
    d = np.asarray(d, float)
    d = d / max(np.linalg.norm(d), 1e-12)
    ref = UP if abs(d[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, d)
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    return x, y, d


def keys_in(lib, keys) -> list[str]:
    return [k for k in keys if k in lib.items]


def pick(ctx, keys, fallback=()):
    ks = keys_in(ctx.lib, keys) or list(fallback)
    return ctx.lib.pick(ctx.rng, ks)


# --- mesh builder ------------------------------------------------------------------------------

class MB:
    """Faces with per-corner UVs (metres / tile) and per-face smooth flags."""

    def __init__(self):
        self.V: list = []
        self.F: list = []
        self.UV: list = []
        self.S: list = []
        self.lo = np.full(3, np.inf)
        self.hi = np.full(3, -np.inf)

    def add(self, verts, faces, uvs, smooth):
        n = len(self.V)
        V = np.asarray(verts, float)
        self.lo = np.minimum(self.lo, V.min(0))
        self.hi = np.maximum(self.hi, V.max(0))
        self.V.extend(V.tolist())
        for f, uv in zip(faces, uvs):
            self.F.append(tuple(i + n for i in f))
            self.UV.append(uv)
            self.S.append(smooth if isinstance(smooth, bool) else smooth(f))

    def box(self, c, half, R, tile=1.0):
        c, half = np.asarray(c, float), np.asarray(half, float)
        sgn = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], float)
        loc = sgn * half
        P = c + loc @ np.asarray(R).T
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        drop = [2, 2, 1, 0, 1, 0]
        uvs = []
        for f, ax in zip(faces, drop):
            keep = [i for i in range(3) if i != ax]
            uvs.append([(loc[i][keep[0]] / tile, loc[i][keep[1]] / tile) for i in f])
        self.add(P, faces, uvs, False)

    def cyl(self, a, b, r0, r1=None, seg=12, caps=(True, True), tile=1.0, smooth=True, phase=0.0):
        a, b = np.asarray(a, float), np.asarray(b, float)
        r1 = r0 if r1 is None else r1
        x, y, d = frame(b - a)
        L = float(np.linalg.norm(b - a))
        th = phase + 2 * math.pi * np.arange(seg) / seg
        cs, sn = np.cos(th), np.sin(th)
        ring0 = a + r0 * (cs[:, None] * x + sn[:, None] * y)
        ring1 = b + r1 * (cs[:, None] * x + sn[:, None] * y)
        V = np.concatenate([ring0, ring1])
        faces, uvs, sm = [], [], []
        circ = 2 * math.pi * max(r0, r1) / tile
        for i in range(seg):
            j = (i + 1) % seg
            faces.append((i, j, seg + j, seg + i))
            u0, u1 = circ * i / seg, circ * (i + 1) / seg
            uvs.append([(u0, 0.0), (u1, 0.0), (u1, L / tile), (u0, L / tile)])
        n_side = len(faces)
        if caps[0] and r0 > 1e-4:
            faces.append(tuple(range(seg - 1, -1, -1)))
            uvs.append([(r0 * cs[i] / tile, r0 * sn[i] / tile) for i in range(seg - 1, -1, -1)])
        if caps[1] and r1 > 1e-4:
            faces.append(tuple(seg + i for i in range(seg)))
            uvs.append([(r1 * cs[i] / tile, r1 * sn[i] / tile) for i in range(seg)])
        base = len(self.F)
        self.add(V, faces, uvs, False)
        for k in range(n_side):
            self.S[base + k] = smooth

    def dome(self, c, r, h, seg=24, rings=5, tile=1.0):
        """Elliptic cap on a circle of radius ``r`` at ``c`` (z up), height ``h``."""
        c = np.asarray(c, float)
        V, faces, uvs = [], [], []
        for k in range(rings):
            al = (k / rings) * math.pi / 2
            rr, zz = r * math.cos(al), h * math.sin(al)
            for i in range(seg):
                t = 2 * math.pi * i / seg
                V.append(c + np.array([rr * math.cos(t), rr * math.sin(t), zz]))
        V.append(c + np.array([0, 0, h]))
        top = len(V) - 1
        for k in range(rings - 1):
            for i in range(seg):
                j = (i + 1) % seg
                f = (k * seg + i, k * seg + j, (k + 1) * seg + j, (k + 1) * seg + i)
                faces.append(f)
                uvs.append([(V[q][0] / tile, V[q][1] / tile) for q in f])
        k = rings - 1
        for i in range(seg):
            j = (i + 1) % seg
            f = (k * seg + i, k * seg + j, top)
            faces.append(f)
            uvs.append([(V[q][0] / tile, V[q][1] / tile) for q in f])
        self.add(V, faces, uvs, True)

    def hexa(self, P, tile=1.0):
        """Eight corners in box order (bottom 0-3 ccw, top 4-7): skips, heaps, tapered blocks."""
        P = np.asarray(P, float)
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        drop = [2, 2, 1, 0, 1, 0]
        uvs = [[(P[i][[k for k in range(3) if k != ax][0]] / tile, P[i][[k for k in range(3) if k != ax][1]] / tile)
                for i in f] for f, ax in zip(faces, drop)]
        self.add(P, faces, uvs, False)

    def mound(self, c, R, h, rng, seg=20, rings=6, tile=2.0, ry=None, yaw=0.0):
        """A heap: radial height field, ragged by noise, base ring just below z = 0."""
        c = np.asarray(c, float)
        ry = R if ry is None else ry
        Rm = rotz(yaw)[:2, :2]
        bumps = rng.normal(0, 1, (rings, seg))
        V = [np.array([c[0], c[1], h * (1 + 0.05 * rng.normal())])]
        for k in range(1, rings + 1):
            rho = k / rings
            for i in range(seg):
                t = 2 * math.pi * i / seg
                w = 1 + 0.12 * bumps[k - 1, i]
                q = Rm @ np.array([R * rho * math.cos(t) * w, ry * rho * math.sin(t) * w])
                z = h * max(0.0, 1 - rho ** 2) ** 0.75 * (1 + 0.1 * bumps[k - 1, i]) if k < rings else -0.02
                V.append(np.array([c[0] + q[0], c[1] + q[1], z]))
        faces = []
        for i in range(seg):
            faces.append((0, 1 + i, 1 + (i + 1) % seg))
        for k in range(1, rings):
            for i in range(seg):
                j = (i + 1) % seg
                a0, a1 = 1 + (k - 1) * seg + i, 1 + (k - 1) * seg + j
                b0, b1 = 1 + k * seg + i, 1 + k * seg + j
                faces.append((a0, b0, b1, a1))
        uvs = [[(V[q][0] / tile, V[q][1] / tile) for q in f] for f in faces]
        self.add(V, faces, uvs, True)

    def quad(self, p0, p1, p2, p3, uv=None):
        P = [np.asarray(p, float) for p in (p0, p1, p2, p3)]
        if uv is None:
            uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.add(P, [(0, 1, 2, 3)], [uv], False)

    def tri(self, p0, p1, p2, tile=1.0):
        P = [np.asarray(p, float) for p in (p0, p1, p2)]
        self.add(P, [(0, 1, 2)], [[(p[0] / tile, p[2] / tile) for p in P]], False)

    def panel(self, p0, p1, z0, z1, tile=1.0):
        """A vertical rectangle (UVs in metres / tile) -- alpha-cut fences, nets."""
        p0, p1 = np.asarray(p0, float)[:2], np.asarray(p1, float)[:2]
        L = float(np.linalg.norm(p1 - p0))
        P = [(*p0, z0), (*p1, z0), (*p1, z1), (*p0, z1)]
        self.add(P, [(0, 1, 2, 3)], [[(0, z0 / tile), (L / tile, z0 / tile), (L / tile, z1 / tile), (0, z1 / tile)]],
                 False)


class Batch:
    """Collects parts per (material, zone class, collection, roof flag); ``flush`` makes the objects."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.parts: dict = {}
        self.n_parts = 0

    def _cls(self, lo, hi, far=False):
        if far or not self.ctx.rect_meets_zone(lo, hi):
            return "far"
        if lo[2] > ZSPLIT:
            return "high"
        return "near"

    def mb(self, mat, lo, hi, coll=None, roof=False, far=False, name="part") -> MB:
        cls = self._cls(lo, hi, far)
        coll = coll or ("backdrop" if cls == "far" else "structure")
        key = (mat.name, cls, coll, bool(roof))
        if key not in self.parts:
            self.parts[key] = {"mat": mat, "cls": cls, "coll": coll, "roof": bool(roof), "mb": MB(), "name": name}
        self.n_parts += 1
        return self.parts[key]["mb"]

    # --- primitives ---------------------------------------------------------------------------

    def box(self, mat, c, half, yaw=0.0, R=None, tile=1.0, coll=None, roof=False, far=False, name="part",
            split=True):
        c, half = np.asarray(c, float), np.asarray(half, float)
        R = rotz(yaw) if R is None else np.asarray(R, float)
        ext = np.abs(R) @ half
        lo, hi = c - ext, c + ext
        if (split and not far and lo[2] < ZSPLIT - 0.3 and hi[2] > ZSPLIT + 0.3 and abs(R[2, 2]) > 0.999
                and self.ctx.rect_meets_zone(lo, hi)):
            z0, z1 = lo[2], hi[2]
            self.box(mat, (c[0], c[1], (z0 + ZSPLIT) / 2), (half[0], half[1], (ZSPLIT - z0) / 2), R=R, tile=tile,
                     coll=coll, roof=roof, name=name, split=False)
            self.box(mat, (c[0], c[1], (ZSPLIT + z1) / 2), (half[0], half[1], (z1 - ZSPLIT) / 2), R=R, tile=tile,
                     coll=coll, roof=roof, name=name, split=False)
            return
        self.mb(mat, lo, hi, coll, roof, far, name).box(c, half, R, tile)

    def cyl(self, mat, a, b, r0, r1=None, seg=12, caps=(True, True), tile=1.0, coll=None, roof=False, far=False,
            name="part", smooth=True, split=True):
        a, b = np.asarray(a, float), np.asarray(b, float)
        rr = max(r0, r0 if r1 is None else r1)
        lo, hi = np.minimum(a, b) - rr, np.maximum(a, b) + rr
        vert = abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6
        if (split and vert and not far and r1 is None and lo[2] < ZSPLIT - 0.3 and hi[2] > ZSPLIT + 0.3
                and self.ctx.rect_meets_zone(lo, hi)):
            lo_z, hi_z = (a, b) if a[2] < b[2] else (b, a)
            m = np.array([a[0], a[1], ZSPLIT])
            self.cyl(mat, lo_z, m, r0, None, seg, (caps[0], False), tile, coll, roof, far, name, smooth, False)
            self.cyl(mat, m, hi_z, r0, None, seg, (False, caps[1]), tile, coll, roof, far, name, smooth, False)
            return
        self.mb(mat, lo, hi, coll, roof, far, name).cyl(a, b, r0, r1, seg, caps, tile, smooth)

    def dome(self, mat, c, r, h, seg=24, rings=5, tile=1.0, coll=None, roof=False, far=False, name="part"):
        c = np.asarray(c, float)
        lo, hi = c - np.array([r, r, 0]), c + np.array([r, r, h])
        self.mb(mat, lo, hi, coll, roof, far, name).dome(c, r, h, seg, rings, tile)

    def hexa(self, mat, P, tile=1.0, coll=None, roof=False, far=False, name="part"):
        P = np.asarray(P, float)
        self.mb(mat, P.min(0), P.max(0), coll, roof, far, name).hexa(P, tile)

    def mound(self, mat, c, R, h, seg=20, rings=6, tile=2.0, ry=None, yaw=0.0, coll=None, roof=False, far=False,
              name="heap"):
        c = np.asarray(c, float)
        rr = 1.4 * max(R, R if ry is None else ry)
        lo, hi = np.array([c[0] - rr, c[1] - rr, -0.02]), np.array([c[0] + rr, c[1] + rr, h * 1.3])
        self.mb(mat, lo, hi, coll, roof, far, name).mound(c, R, h, self.ctx.rng, seg, rings, tile, ry, yaw)

    def quad(self, mat, p0, p1, p2, p3, uv=None, coll=None, roof=False, far=False, name="part"):
        P = np.array([p0, p1, p2, p3], float)
        self.mb(mat, P.min(0), P.max(0), coll, roof, far, name).quad(p0, p1, p2, p3, uv)

    def tri(self, mat, p0, p1, p2, tile=1.0, coll=None, roof=False, far=False, name="part"):
        P = np.array([p0, p1, p2], float)
        self.mb(mat, P.min(0), P.max(0), coll, roof, far, name).tri(p0, p1, p2, tile)

    def panel(self, mat, p0, p1, z0, z1, tile=1.0, coll=None, roof=False, far=False, name="panel"):
        P = np.array([[*p0[:2], z0], [*p1[:2], z1]], float)
        self.mb(mat, P.min(0), P.max(0), coll, roof, far, name).panel(p0, p1, z0, z1, tile)

    # --- structural members -------------------------------------------------------------------

    def member(self, mat, a, b, w=0.2, h=0.2, kind="I", roll=None, **kw):
        """A steel member a -> b: ``I`` (web + two flanges), ``box`` (hollow
        section as a solid), ``L`` (angle). ``roll``: the section's 'up'
        (default world z, or x for a vertical member)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            return
        d = (b - a) / L
        up = np.asarray(roll, float) if roll is not None else (UP if abs(d[2]) < 0.95 else np.array([1.0, 0, 0]))
        e_up = up - d * float(up @ d)
        e_up /= np.linalg.norm(e_up)
        e_sd = np.cross(e_up, d)
        R = np.stack([d, e_sd, e_up], 1)
        c = (a + b) / 2
        if kind == "box":
            self.box(mat, c, (L / 2, w / 2, h / 2), R=R, **kw)
        elif kind == "L":
            t = max(0.008, 0.1 * w)
            self.box(mat, c + e_up * (h / 2 - t / 2), (L / 2, w / 2, t / 2), R=R, **kw)
            self.box(mat, c + e_sd * (w / 2 - t / 2), (L / 2, t / 2, h / 2), R=R, **kw)
        else:
            tf, tw = max(0.01, 0.08 * h), max(0.008, 0.05 * h)
            self.box(mat, c + e_up * (h / 2 - tf / 2), (L / 2, w / 2, tf / 2), R=R, **kw)
            self.box(mat, c - e_up * (h / 2 - tf / 2), (L / 2, w / 2, tf / 2), R=R, **kw)
            self.box(mat, c, (L / 2, tw / 2, h / 2 - tf), R=R, **kw)

    # --- output -------------------------------------------------------------------------------

    def flush(self) -> list:
        out = []
        for p in self.parts.values():
            mb = p["mb"]
            if not mb.F:
                continue
            collide = False if p["cls"] in ("far", "high") else None
            ob = self.ctx.mesh(mb.V, mb.F, p["mat"], uvs=mb.UV, coll=p["coll"], collide=collide,
                               name=f"{p['name']}_{p['cls']}", roof=p["roof"])
            ob.data.polygons.foreach_set("use_smooth", np.asarray(mb.S, bool))
            ob.data.update()
            out.append(ob)
        self.parts = {}
        return out


class Xf:
    """A ``Batch`` seen through a local frame (origin ``o`` in xy, rotation
    ``yaw``): build a shed or a rack in its own coordinates, place it anywhere."""

    def __init__(self, B, o, yaw=0.0):
        self.B, self.ctx = B, B.ctx
        self.o = np.array([o[0], o[1], 0.0])
        self.yaw = float(yaw)
        self.R = rotz(yaw)

    def p(self, v):
        v = np.asarray(v, float)
        if v.shape[0] == 2:
            return (self.R[:2, :2] @ v) + self.o[:2]
        return self.R @ v + self.o

    def box(self, mat, c, half, yaw=0.0, R=None, **kw):
        R = rotz(yaw) if R is None else np.asarray(R, float)
        self.B.box(mat, self.p(c), half, R=self.R @ R, **kw)

    def cyl(self, mat, a, b, *args, **kw):
        self.B.cyl(mat, self.p(a), self.p(b), *args, **kw)

    def dome(self, mat, c, *args, **kw):
        self.B.dome(mat, self.p(c), *args, **kw)

    def hexa(self, mat, P, *args, **kw):
        self.B.hexa(mat, [self.p(p) for p in P], *args, **kw)

    def quad(self, mat, p0, p1, p2, p3, *args, **kw):
        self.B.quad(mat, self.p(p0), self.p(p1), self.p(p2), self.p(p3), *args, **kw)

    def tri(self, mat, p0, p1, p2, *args, **kw):
        self.B.tri(mat, self.p(p0), self.p(p1), self.p(p2), *args, **kw)

    def panel(self, mat, p0, p1, *args, **kw):
        self.B.panel(mat, self.p(p0[:2]), self.p(p1[:2]), *args, **kw)

    def member(self, mat, a, b, *args, roll=None, **kw):
        self.B.member(mat, self.p(a), self.p(b), *args, roll=None if roll is None else self.R @ np.asarray(roll),
                      **kw)


# --- the patch --------------------------------------------------------------------------------

def prim_boxes(spec) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """3-D bounding boxes (lo, hi, prim) of the patch primitives."""
    out = []
    for p in spec["primitives"]:
        k = p["kind"]
        if k == "wall":
            p0, p1 = np.asarray(p["p0"], float)[:2], np.asarray(p["p1"], float)[:2]
            t = p["t"] / 2
            lo = np.array([*(np.minimum(p0, p1) - t), p["z0"]])
            hi = np.array([*(np.maximum(p0, p1) + t), p["z1"]])
        elif k == "box":
            R = BC.rpy_matrix(p["rpy"])
            ext = np.abs(R) @ np.asarray(p["half"], float)
            c = np.asarray(p["c"], float)
            lo, hi = c - ext, c + ext
        elif k == "cyl_v":
            c = np.asarray(p["c"], float)[:2]
            lo = np.array([*(c - p["r"]), p["z0"]])
            hi = np.array([*(c + p["r"]), p["z1"]])
        elif k == "cyl_h":
            a, b = np.asarray(p["a"], float), np.asarray(p["b"], float)
            lo, hi = np.minimum(a, b) - p["r"], np.maximum(a, b) + p["r"]
        elif k == "sphere":
            c = np.asarray(p["c"], float)
            lo, hi = c - p["r"], c + p["r"]
        else:
            continue
        out.append((lo, hi, p))
    return out


def box_clear(boxes, lo, hi, pad=0.3) -> bool:
    lo, hi = np.asarray(lo, float) - pad, np.asarray(hi, float) + pad
    for blo, bhi, _ in boxes:
        if np.all(bhi >= lo) and np.all(blo <= hi):
            return False
    return True


def reserve_core(ctx) -> dict:
    """Claim the patch family's own space (kind ``core``) before anything is
    placed: the face-span rectangles of a tier-A family or a corridor's legs
    (+``CORE_PAD``, clipped to the patch bounds; not an arena's, whose cells
    are the clutter), an ``END_R`` disc at every wall end inside the bounds,
    and ``END_R`` round a pillar, slab or floating box / bar. Returns counts."""
    spec = ctx.spec
    x0, y0, x1, y1 = ctx.bounds
    fam = spec.get("family")
    out = {"rects": 0, "ends": 0}
    spans = [] if fam == "arena" else (spec.get("params", {}).get("face_spans") or [])
    for fx, gx, yc, hw in spans:
        lo = (max(fx - CORE_PAD, x0), max(yc - hw - CORE_PAD, y0))
        hi = (min(gx + CORE_PAD, x1), min(yc + hw + CORE_PAD, y1))
        if hi[0] - lo[0] < 0.05 or hi[1] - lo[1] < 0.05:
            continue
        ctx.claim_rect(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2), ((hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2),
                       0.0, kind="core")
        out["rects"] += 1
    for p in spec["primitives"]:
        if p["kind"] == "wall":
            for e in (p["p0"], p["p1"]):
                e = np.asarray(e, float)[:2]
                if x0 < e[0] < x1 and y0 < e[1] < y1:
                    ctx.claim_disc(e, END_R + p["t"] / 2, kind="core")
                    out["ends"] += 1
        elif p["kind"] == "cyl_v" and fam == "pillar":
            ctx.claim_disc(p["c"], p["r"] + END_R, kind="core")
            out["ends"] += 1
        elif p["kind"] == "box" and fam in ("slab", "float_box", "float_bar"):
            ext = np.abs(BC.rpy_matrix(p["rpy"])) @ np.asarray(p["half"], float)
            ctx.claim_rect(np.asarray(p["c"], float)[:2], ext[:2] + END_R, 0.0, kind="core")
            out["ends"] += 1
    return out


def walled(ctx) -> bool:
    """The patch is enclosed by perimeter walls too tall to fly over (an
    arena): nothing between them and the zone edge can be reached or seen
    below their top, so the kits leave that ring empty (it would only cost
    cloud points) and let the render-only backdrop make the skyline."""
    per = [p for p in ctx.spec["primitives"] if p["kind"] == "wall" and p.get("region") == "perimeter"]
    return len(per) >= 4 and min(p["z1"] for p in per) >= REACH_TOP


def clear_of(ctx, xy, kinds=CORE_KINDS) -> float:
    """Distance from ``xy`` to the nearest claimed footprint of ``kinds`` (the
    patch, its reserved core, the start and goal discs)."""
    xy = np.asarray(xy, float)[:2]
    return min((BC.footprint_distance(xy, f) for f in ctx.foot if f.get("kind") in kinds), default=1e9)


def seg_clear(ctx, p, q, margin=0.5, kinds=CORE_KINDS, step=0.3) -> bool:
    """An overhead run p -> q (xy) keeps ``margin`` from the patch, its core,
    the start and the goal all along (sampled every ``step`` m)."""
    p, q = np.asarray(p, float)[:2], np.asarray(q, float)[:2]
    n = max(2, int(np.linalg.norm(q - p) / step) + 1)
    return all(clear_of(ctx, p + (q - p) * t, kinds) >= margin for t in np.linspace(0.0, 1.0, n))


def rect_clear(ctx, c, half, yaw=0.0, margin=0.3, kinds=CORE_KINDS, step=0.4) -> bool:
    """``ctx.free_rect`` restricted to footprints of ``kinds``."""
    c = np.asarray(c, float)[:2]
    R = rotz(yaw)[:2, :2]
    us = np.linspace(-half[0], half[0], max(2, int(2 * half[0] / step) + 1))
    vs = np.linspace(-half[1], half[1], max(2, int(2 * half[1] / step) + 1))
    return all(clear_of(ctx, c + R @ np.array([u, v]), kinds) >= margin for u in us for v in vs)


def free_runs(ctx, a, b, half_w, z0, z1, pad=0.5, min_len=3.0) -> list:
    """The stretches of a straight run a -> b (a rack, a duct), ``half_w`` wide
    and spanning heights z0..z1, that pass no patch primitive (its box grown
    by ``pad``): [(a', b'), ...], each at least ``min_len`` long."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    L = float(np.linalg.norm(b - a))
    if L < 1e-6:
        return []
    d = (b - a) / L
    n = np.array([-d[1], d[0]])
    blocked = []
    for lo, hi, _ in prim_boxes(ctx.spec):
        if hi[2] < z0 or lo[2] > z1:
            continue
        C = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]) - a
        u, v = C @ d, C @ n
        if v.max() < -half_w - pad or v.min() > half_w + pad or u.max() < -pad or u.min() > L + pad:
            continue
        blocked.append((u.min() - pad, u.max() + pad))
    runs, t = [], 0.0
    for s0, s1 in sorted(blocked):
        if s0 > t:
            runs.append((t, s0))
        t = max(t, s1)
    if t < L:
        runs.append((t, L))
    return [(a + d * s0, a + d * s1) for s0, s1 in runs if s1 - s0 >= min_len]


def wall_caps(ctx, B, mat, spec, overhang=0.03, thick=0.06):
    """Copings on top of patch walls (only the pieces that reach the wall
    top, so no slot shrinks), and only on walls too tall to fly over
    (``REACH_TOP``): on a lower wall they would raise the over-the-top route."""
    for p in spec["primitives"]:
        if p["kind"] != "wall" or p["z1"] < REACH_TOP:
            continue
        for c, h, R in BC.wall_pieces(p):
            if abs((c[2] + h[2]) - p["z1"]) > 1e-3:
                continue
            B.box(mat, (c[0], c[1], p["z1"] + thick / 2), (h[0] + overhang, h[1] + overhang, thick / 2), R=R,
                  name="coping")


# --- materials ----------------------------------------------------------------------------------

PAINT_CLEAN = ["acg:PaintedMetal012", "acg:Plastic013A"]     # white paint / plastic, tinted to colour
PAINT_WORN = ["acg:PaintedMetal010"]                           # white paint with rust spots
PAINT_PLAIN = ["acg:Plastic013A"]                              # no chips: large painted surfaces
PAINT_OLD = ["acg:PaintedMetal014", "ph:green_metal_rust"]     # dull, weathered
GALV = ["acg:Metal011", "acg:Metal009", "acg:Metal032", "acg:MetalPlates014", "acg:MetalPlates009"]


def paint(ctx, rgb, rusty: float = 0.3, tile=1.5, bevel=0.01, dirt=0.35, plain: bool = False):
    """Painted steel: a clean white paint scan tinted to ``rgb``; rust-spotted
    paint only for darker colours (on light paint the spots read as birch
    bark) and at a larger tile. ``plain``: the unchipped scan only -- big
    surfaces (tank shells), where chips at their real size read as a cow
    pattern from 20 m."""
    rng = ctx.rng
    lum = float(np.mean(rgb))
    r = rng.random()
    if plain and keys_in(ctx.lib, PAINT_PLAIN):
        k, t, scale = pick(ctx, PAINT_PLAIN), tile, 1.05
    elif r < 0.6 * rusty and lum < 0.45 and keys_in(ctx.lib, PAINT_WORN):
        k, t, scale = pick(ctx, PAINT_WORN), max(tile, 2.5), 1.15
    else:
        k, t, scale = pick(ctx, PAINT_CLEAN), tile, 1.1
    if k == "acg:Plastic013A":
        scale = 1.05
    dirt = (dirt + 0.3 * rusty) * (0.35 if lum > 0.5 else 1.0)     # grime blotches on white paint read as bark
    return ctx.looks.pbr(k, t, tint=tuple(min(1.0, scale * c) for c in rgb), bevel=bevel, dirt=dirt, jitter=0.05)


def native(ctx, key, default=2.0):
    """The scan's own tile size (bricks, blocks, panels have a real module size)."""
    return float(ctx.lib.items.get(key, {}).get("tile_m") or default)


#: scans much darker than the thing they dress (mean sRGB albedo ~0.21-0.24 for concrete block):
#: a linear gain brings them to a plausible mid grey under the auto exposure
GAIN = {"ph:concrete_block_wall": 2.2, "ph:concrete_block_wall_02": 2.5, "ph:concrete_tile_facade": 1.8,
        "ph:hangar_concrete_floor": 1.8}


def scan_native(ctx, keys, default=2.0, fallback=(), **kw):
    k = pick(ctx, keys, fallback)
    if k in GAIN and "tint" not in kw:
        kw["tint"] = (GAIN[k],) * 3
    return ctx.looks.pbr(k, native(ctx, k, default), **kw)


def galv(ctx, tile=1.2, bevel=0.005, dirt=0.3, tint=None):
    return ctx.looks.pbr(pick(ctx, GALV), tile, bevel=bevel, dirt=dirt, tint=tint, jitter=0.04)


def scan(ctx, keys, tile=None, fallback_cats=(), **kw):
    k = pick(ctx, keys, ctx.lib.materials(cats_any=set(fallback_cats)) if fallback_cats else ())
    if k in GAIN and "tint" not in kw:
        kw["tint"] = (GAIN[k],) * 3
    return ctx.looks.pbr(k, tile, **kw)


def decal_material(name, rgb=(0.02, 0.018, 0.015), rough=0.2, opacity=0.8, scale=1.2, soft=0.35):
    """A blotch (oil stain, water mark, tyre smear) on a unit-UV quad: radial
    falloff broken up by 4-D noise seeded per object, alpha-blended."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    tc = N.new("ShaderNodeTexCoord")
    oi = N.new("ShaderNodeObjectInfo")
    sub = N.new("ShaderNodeVectorMath"); sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = (0.5, 0.5, 0.0)
    L.new(tc.outputs["UV"], sub.inputs[0])
    ln = N.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"
    L.new(sub.outputs["Vector"], ln.inputs[0])
    nz = N.new("ShaderNodeTexNoise")
    nz.noise_dimensions = "4D"
    nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = 8.0
    nz.inputs["Roughness"].default_value = 0.6
    L.new(tc.outputs["Object"], nz.inputs["Vector"])
    w = N.new("ShaderNodeMath"); w.operation = "MULTIPLY"; w.inputs[1].default_value = 97.0
    L.new(oi.outputs["Random"], w.inputs[0])
    L.new(w.outputs["Value"], nz.inputs["W"])
    # field = 1 - 2*d + (noise - 0.5) * 1.4
    f1 = N.new("ShaderNodeMath"); f1.operation = "MULTIPLY_ADD"
    f1.inputs[1].default_value = -2.0; f1.inputs[2].default_value = 1.0
    L.new(ln.outputs["Value"], f1.inputs[0])
    f2 = N.new("ShaderNodeMath"); f2.operation = "MULTIPLY_ADD"
    f2.inputs[1].default_value = 1.4; f2.inputs[2].default_value = -0.7
    L.new(nz.outputs["Fac"], f2.inputs[0])
    f3 = N.new("ShaderNodeMath"); f3.operation = "ADD"
    L.new(f1.outputs["Value"], f3.inputs[0]); L.new(f2.outputs["Value"], f3.inputs[1])
    mr = N.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.0, soft
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 0.0, opacity
    L.new(f3.outputs["Value"], mr.inputs["Value"])
    L.new(mr.outputs["Result"], b.inputs["Alpha"])
    return m


def worn_paint_material(name, rgb, rough=0.55, wear=0.35, scale=2.5):
    """Road paint that has worn through in patches (alpha from world noise)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    tc = N.new("ShaderNodeTexCoord")
    nz = N.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = 10.0
    nz.inputs["Roughness"].default_value = 0.7
    L.new(tc.outputs["Object"], nz.inputs["Vector"])
    mr = N.new("ShaderNodeMapRange")
    lo = 0.25 + 0.3 * wear
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = lo, lo + 0.12
    L.new(nz.outputs["Fac"], mr.inputs["Value"])
    L.new(mr.outputs["Result"], b.inputs["Alpha"])
    return m


def decal(ctx, mat, c, rx, ry, yaw=0.0, z=0.004, name="decal"):
    """A flat (non-colliding) quad with unit UVs for ``decal_material``."""
    R = rotz(yaw)[:2, :2]
    P = [np.asarray(c, float)[:2] + R @ np.array(v) for v in ((-rx, -ry), (rx, -ry), (rx, ry), (-rx, ry))]
    V = [(p[0], p[1], z) for p in P]
    return ctx.mesh(V, [(0, 1, 2, 3)], mat, uvs=[[(0, 0), (1, 0), (1, 1), (0, 1)]], coll="ground", collide=False,
                    name=name)


# --- structures ---------------------------------------------------------------------------------

def ladder(B, mat, p, yaw, z0, z1, cage=True, **kw):
    """A fixed steel ladder at ``p`` (xy) facing along ``yaw``, optional cage above 2.5 m."""
    R = rotz(yaw)
    s = R @ np.array([0.0, 1.0, 0.0])
    for sgn in (-1, 1):
        q = np.array([p[0], p[1], 0.0]) + sgn * 0.22 * s
        B.box(mat, (q[0], q[1], (z0 + z1) / 2), (0.03, 0.01, (z1 - z0) / 2), yaw=yaw, **kw)
    for z in np.arange(z0 + 0.3, z1, 0.3):
        B.box(mat, (p[0], p[1], z), (0.012, 0.22, 0.012), yaw=yaw, **kw)
    if cage and z1 - z0 > 3.0:
        f = R @ np.array([1.0, 0.0, 0.0])
        for z in np.arange(max(z0 + 2.5, 2.5), z1, 1.0):
            pts = [np.array([p[0], p[1], z]) + 0.35 * math.cos(t) * f + 0.35 * math.sin(t) * s
                   for t in np.linspace(-math.pi / 2, math.pi / 2, 6)]
            for u, v in zip(pts[:-1], pts[1:]):
                B.cyl(mat, u, v, 0.01, seg=4, caps=(False, False), **kw)


def handrail(B, mat, pts, z, h=1.05, post=1.6, closed=False, **kw):
    """Posts + top rail + knee rail along a polyline at deck height ``z``."""
    pts = [np.asarray(p, float)[:2] for p in pts]
    if closed:
        pts = pts + [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.linalg.norm(b - a))
        if L < 1e-3:
            continue
        for zz in (z + h, z + h / 2):
            B.cyl(mat, (*a, zz), (*b, zz), 0.022, seg=6, **kw)
        for t in np.linspace(0, 1, max(2, int(L / post) + 1)):
            q = a + (b - a) * t
            B.cyl(mat, (*q, z), (*q, z + h), 0.022, seg=6, **kw)


def stair(B, mat_str, mat_tread, a, yaw, z1, width=0.9, rail=None, **kw):
    """A straight steel stair from ground at ``a`` rising along ``yaw`` to ``z1`` (40 deg)."""
    run = z1 / math.tan(math.radians(40))
    R = rotz(yaw)
    f, s = R @ np.array([1.0, 0, 0]), R @ np.array([0, 1.0, 0])
    a3 = np.array([a[0], a[1], 0.0])
    for sgn in (-1, 1):
        p0 = a3 + sgn * s * width / 2
        B.member(mat_str, p0, p0 + f * run + UP * z1, 0.03, 0.2, kind="box", **kw)
    n = max(2, int(z1 / 0.19))
    for i in range(1, n):
        t = i / n
        c = a3 + f * run * t + UP * (z1 * t)
        B.box(mat_tread, c, (0.13, width / 2, 0.02), yaw=yaw, **kw)
    if rail is not None:
        for sgn in (-1, 1):
            p0 = a3 + sgn * s * (width / 2 + 0.02)
            p1 = p0 + f * run + UP * z1
            B.cyl(rail, p0 + UP * 0.95, p1 + UP * 0.95, 0.022, seg=6, **kw)
            for t in (0.05, 0.5, 0.95):
                q = p0 + (p1 - p0) * t
                B.cyl(rail, q, q + UP * 0.95, 0.022, seg=6, **kw)
    return run


def fence_line(ctx, B, a, b, h, mesh_mat, post_mat, spacing=2.5, tile=1.0, barbed=False, rail=True,
               check=True, **kw):
    """Chain-link / weld-mesh fence a -> b: posts, top rail, alpha-cut panels.
    Stops at the first post that is not free (``check``)."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    L = float(np.linalg.norm(b - a))
    n = max(1, int(round(L / spacing)))
    pts = [a + (b - a) * i / n for i in range(n + 1)]
    ok = [(not check) or ctx.free(p, 0.05, 0.2) for p in pts]
    for i, p in enumerate(pts):
        if not ok[i]:
            continue
        B.cyl(post_mat, (*p, 0.0), (*p, h + 0.05), 0.03, seg=8, **kw)
        if barbed:
            B.cyl(post_mat, (*p, h), (p[0], p[1] + 0.0, h + 0.45), 0.012, seg=4, **kw)
    for i in range(n):
        if not (ok[i] and ok[i + 1]):
            continue
        p, q = pts[i], pts[i + 1]
        B.panel(mesh_mat, p, q, 0.03, h, tile=tile, name="fence", **kw)
        if rail:
            B.cyl(post_mat, (*p, h), (*q, h), 0.021, seg=6, caps=(False, False), **kw)
        if barbed:
            for dz in (0.15, 0.3, 0.44):
                B.cyl(post_mat, (*p, h + dz), (*q, h + dz), 0.003, seg=3, caps=(False, False), **kw)
        if check:
            d = (q - p) / max(np.linalg.norm(q - p), 1e-9)
            ctx.claim_rect((p + q) / 2, (np.linalg.norm(q - p) / 2, 0.1), math.atan2(d[1], d[0]), kind="fence")


def light_mast(B, mat, lamp, xy, h, heads=3, yaw=0.0, **kw):
    x, y = xy
    B.cyl(mat, (x, y, 0.0), (x, y, h), 0.22, 0.09, seg=12, **kw)
    B.box(mat, (x, y, h + 0.1), (0.9, 0.1, 0.06), yaw=yaw, **kw)
    R = rotz(yaw)
    for i in range(heads):
        o = R @ np.array([-0.7 + 1.4 * i / max(1, heads - 1), 0.0, 0.0])
        B.box(lamp, (x + o[0], y + o[1], h + 0.35), (0.22, 0.08, 0.16), yaw=yaw, **kw)


def pole_line(ctx, B, mat, wire, pts, h=8.0, sag=0.6, n_wires=3, **kw):
    """Timber/steel poles with crossarms and sagging wires through ``pts``."""
    pts = [np.asarray(p, float)[:2] for p in pts]
    tops = []
    for i, p in enumerate(pts):
        d = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        yaw = math.atan2(d[1], d[0]) + math.pi / 2
        B.cyl(mat, (*p, 0.0), (*p, h), 0.14, 0.11, seg=10, **kw)
        B.box(mat, (p[0], p[1], h - 0.4), (0.06, 0.9, 0.06), yaw=yaw - math.pi / 2, **kw)
        s = rotz(yaw) @ np.array([1.0, 0, 0])
        tops.append([np.array([p[0], p[1], h - 0.3]) + s[:3] * o for o in np.linspace(-0.8, 0.8, n_wires)])
    for t0, t1 in zip(tops[:-1], tops[1:]):
        for a, b in zip(t0, t1):
            catenary(B, wire, a, b, sag, r=0.008, **kw)


def catenary(B, mat, a, b, sag, r=0.01, n=10, **kw):
    a, b = np.asarray(a, float), np.asarray(b, float)
    prev = a
    for i in range(1, n + 1):
        t = i / n
        p = a + (b - a) * t - UP * (4 * sag * t * (1 - t))
        B.cyl(mat, prev, p, r, seg=4, caps=(False, False), **kw)
        prev = p


def gable_shed(B, wall, roof, trim, x0, x1, yf, depth, side, h, pitch=10.0, plinth=None, door=None, glass=None,
               bay=6.0, doors=(), roof_flag=False, gutter=None, far=False, rng=None):
    """A portal-frame shed: long facade on ``y = yf`` facing the patch
    (``side`` = +1: the building extends to +y), walls ``h`` high, a
    ridge along x. ``doors``: list of (x_centre, width, height) roller doors."""
    kw = {"roof": roof_flag, "far": far}
    yb = yf + side * depth
    yc = (yf + yb) / 2
    xc = (x0 + x1) / 2
    L = x1 - x0
    t = 0.2
    # walls (split into bays so nothing is one giant face)
    for yy in (yf, yb):
        for bx0 in np.arange(x0, x1 - 1e-6, bay * 2):
            bx1 = min(x1, bx0 + bay * 2)
            B.box(wall, ((bx0 + bx1) / 2, yy, h / 2), ((bx1 - bx0) / 2, t / 2, h / 2), name="shed_wall", **kw)
    rise = math.tan(math.radians(pitch)) * depth / 2
    for xx in (x0, x1):
        for by0 in np.arange(min(yf, yb), max(yf, yb) - 1e-6, bay * 2):
            by1 = min(max(yf, yb), by0 + bay * 2)
            B.box(wall, (xx, (by0 + by1) / 2, h / 2), (t / 2, (by1 - by0) / 2, h / 2), name="shed_wall", **kw)
        # gable triangle
        B.tri(wall, (xx, yf, h), (xx, yb, h), (xx, yc, h + rise), tile=2.0, name="gable", **kw)
    # roof: two slopes, 0.3 m eave overhang
    o = 0.35
    for ya in (yf - side * o, yb + side * o):
        p0 = (x0 - o, ya, h - 0.06)
        p1 = (x1 + o, ya, h - 0.06)
        p2 = (x1 + o, yc, h + rise)
        p3 = (x0 - o, yc, h + rise)
        B.quad(roof, p0, p1, p2, p3, name="shed_roof", **kw)
        B.quad(roof, (p0[0], p0[1], p0[2] + 0.08), (p3[0], p3[1], p3[2] + 0.08), (p2[0], p2[1], p2[2] + 0.08),
               (p1[0], p1[1], p1[2] + 0.08), name="shed_roof", **kw)
    if gutter is not None:
        for ya in (yf - side * (o + 0.08), yb + side * (o + 0.08)):
            B.box(gutter, (xc, ya, h - 0.12), (L / 2 + o, 0.08, 0.07), name="gutter", **kw)
    if plinth is not None:
        B.box(plinth, (xc, yf - side * 0.02, 0.5), (L / 2 + 0.02, t / 2 + 0.02, 0.5), name="plinth", **kw)
    for xd, wd, hd in doors:
        if door is not None:
            B.box(door, (xd, yf - side * (t / 2 + 0.03), hd / 2), (wd / 2, 0.03, hd / 2), name="door", **kw)
        B.box(trim, (xd, yf - side * (t / 2 + 0.2), hd + 0.25), (wd / 2 + 0.15, 0.25, 0.25), name="door_box", **kw)
        for sx in (-1, 1):
            B.box(trim, (xd + sx * (wd / 2 + 0.06), yf - side * (t / 2 + 0.05), hd / 2), (0.06, 0.05, hd / 2),
                  name="door_jamb", **kw)
    if glass is not None and h > 5:
        zw = h - 1.4
        for bx0 in np.arange(x0 + 1.0, x1 - 1.5, bay):
            if any(abs(bx0 + bay / 2 - xd) < wd / 2 + bay / 2 for xd, wd, _ in doors):
                continue
            B.box(glass, (bx0 + bay / 2, yf - side * (t / 2 + 0.01), zw), (bay / 2 - 0.6, 0.02, 0.45),
                  name="window", **kw)
    # portal columns proud of the facade every bay
    for bx in np.arange(x0, x1 + 1e-6, bay):
        B.box(trim, (bx, yf - side * (t / 2 + 0.1), h / 2), (0.1, 0.1, h / 2), name="pier", **kw)
    return rise


def tank(B, shell, roof, rail, c, r, h, kind="cone", stair_mat=None, rings=True, far=False, roof_flag=False,
         seg=40):
    """A vertical storage tank: shell, cone / dome / flat roof, stiffener
    rings, rim handrail and a helical stair."""
    kw = {"far": far, "roof": roof_flag}
    x, y = c
    B.cyl(shell, (x, y, 0.0), (x, y, h), r, seg=seg, caps=(False, kind == "flat"), tile=2.0, name="tank", **kw)
    if kind == "cone":
        B.cyl(roof, (x, y, h), (x, y, h + 0.12 * r), r + 0.05, 0.25, seg=seg, caps=(True, True), tile=2.0,
              name="tank_roof", smooth=False, **kw)
    elif kind == "dome":
        B.dome(roof, (x, y, h), r + 0.03, 0.22 * r, seg=seg, rings=5, tile=2.0, name="tank_roof", **kw)
    if rings:
        for z in np.arange(2.4, h - 0.5, 2.4):
            B.cyl(shell, (x, y, z), (x, y, z + 0.06), r + 0.025, seg=seg, caps=(False, False), name="tank", **kw)
    if rail is not None:
        pts = [(x + (r - 0.2) * math.cos(t), y + (r - 0.2) * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 25)]
        zt = h + (0.1 * r if kind == "cone" else 0.12 * r if kind == "dome" else 0.0)
        for a, b in zip(pts[:-1], pts[1:]):
            B.cyl(rail, (*a, zt + 1.0), (*b, zt + 1.0), 0.025, seg=5, caps=(False, False), **kw)
        for a in pts[:-1:2]:
            B.cyl(rail, (*a, zt), (*a, zt + 1.0), 0.025, seg=5, **kw)
    if stair_mat is not None and h > 3:
        n = int(h / 0.2)
        th0 = 2 * math.pi * float((np.random.default_rng(int(abs(x * 13 + y * 7)) % 1000)).random())
        sweep = min(1.6 * math.pi, h / (0.9 * r) * 1.2)
        prev = None
        for i in range(1, n + 1):
            t = i / n
            th = th0 + sweep * t
            z = h * t
            rc = r + 0.5
            p = np.array([x + rc * math.cos(th), y + rc * math.sin(th), z])
            B.box(stair_mat, p, (0.12, 0.45, 0.02), yaw=th, name="tank_stair", **kw)
            po = np.array([x + (r + 0.97) * math.cos(th), y + (r + 0.97) * math.sin(th), z + 1.0])
            if prev is not None:
                B.cyl(rail or stair_mat, prev, po, 0.022, seg=5, caps=(False, False), **kw)
            if i % 5 == 0 and rail is not None:
                B.cyl(rail, po - UP * 1.0, po, 0.022, seg=5, **kw)
            prev = po


def vessel_h(B, shell, saddle, c, yaw, r, L, far=False, roof_flag=False):
    """A horizontal 'bullet' tank on two concrete saddles."""
    kw = {"far": far, "roof": roof_flag}
    R = rotz(yaw)
    f = R @ np.array([1.0, 0, 0])
    zc = r + 0.45
    a = np.array([c[0], c[1], zc]) - f * L / 2
    b = np.array([c[0], c[1], zc]) + f * L / 2
    B.cyl(shell, a, b, r, seg=28, caps=(False, False), tile=2.0, name="vessel", **kw)
    for p, sgn in ((a, -1), (b, 1)):
        # an elliptic head along the axis: a dome built in a rotated frame
        mb = B.mb(shell, p - 1.6 * r, p + 1.6 * r, name="vessel", **kw)
        loc = MB()
        loc.dome(np.zeros(3), r, 0.6 * r, seg=28, rings=5, tile=2.0)
        x_, y_, d_ = frame(sgn * f)
        M = np.stack([x_, y_, d_], 1)
        V = np.asarray(loc.V) @ M.T + p
        mb.add(V, [tuple(ff) for ff in loc.F], loc.UV, True)
    for s in (-0.3, 0.3):
        q = np.array([c[0], c[1], 0.0]) + f * L * s
        B.box(saddle, (q[0], q[1], zc / 2), (0.25, r * 0.8, zc / 2), yaw=yaw, name="saddle", **kw)
    return zc + r


def tapered(c, yaw, hb, ht, z0, z1):
    """Eight corners of a block with bottom half-size ``hb`` and top ``ht`` (xy), yawed."""
    R = rotz(yaw)
    c = np.asarray(c, float)
    out = []
    for (hx, hy), z in ((hb, z0), (ht, z1)):
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            q = R @ np.array([sx * hx, sy * hy, 0.0])
            out.append((c[0] + q[0], c[1] + q[1], z))
    return out


def soften(mat, mode="blob", width=0.3, noise_scale=0.8, noise_amp=0.5):
    """Give a flat overlay material (a ground scan from ``looks.pbr``) a
    ragged, soft alpha edge from its UVs: ``blob`` (unit-UV quad, radial) or
    ``strip`` (v = 0..1 across). Only for non-colliding ground overlays."""
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    b = N.get("Principled BSDF")
    if b is None:
        return mat
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["UV"], sep.inputs["Vector"])
    if mode == "strip":
        m1 = N.new("ShaderNodeMath"); m1.operation = "MULTIPLY_ADD"
        m1.inputs[1].default_value = 2.0; m1.inputs[2].default_value = -1.0
        L.new(sep.outputs["Y"], m1.inputs[0])
        ab = N.new("ShaderNodeMath"); ab.operation = "ABSOLUTE"
        L.new(m1.outputs["Value"], ab.inputs[0])
        dist = ab.outputs["Value"]
    else:
        sub = N.new("ShaderNodeVectorMath"); sub.operation = "SUBTRACT"
        sub.inputs[1].default_value = (0.5, 0.5, 0.0)
        L.new(tc.outputs["UV"], sub.inputs[0])
        sc = N.new("ShaderNodeVectorMath"); sc.operation = "MULTIPLY"
        sc.inputs[1].default_value = (2.0, 2.0, 0.0)
        L.new(sub.outputs["Vector"], sc.inputs[0])
        ln = N.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"
        L.new(sc.outputs["Vector"], ln.inputs[0])
        dist = ln.outputs["Value"]
    inv = N.new("ShaderNodeMath"); inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    L.new(dist, inv.inputs[1])
    nz = N.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = noise_scale
    nz.inputs["Detail"].default_value = 6.0
    L.new(tc.outputs["Object"], nz.inputs["Vector"])
    na = N.new("ShaderNodeMath"); na.operation = "MULTIPLY_ADD"
    na.inputs[1].default_value = noise_amp; na.inputs[2].default_value = -0.5 * noise_amp
    L.new(nz.outputs["Fac"], na.inputs[0])
    add = N.new("ShaderNodeMath"); add.operation = "ADD"
    L.new(inv.outputs["Value"], add.inputs[0]); L.new(na.outputs["Value"], add.inputs[1])
    mr = N.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.0, width
    L.new(add.outputs["Value"], mr.inputs["Value"])
    L.new(mr.outputs["Result"], b.inputs["Alpha"])
    return mat


def overlay_strip(ctx, mat, a, b, width, z=0.004, name="overlay"):
    """A flat, non-colliding strip a -> b with UVs (u metres along, v 0..1 across) for ``soften``."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    L = float(np.linalg.norm(b - a))
    d = (b - a) / max(L, 1e-9)
    n = np.array([-d[1], d[0]]) * width / 2
    P = [a - n, b - n, b + n, a + n]
    V = [(p[0], p[1], z) for p in P]
    return ctx.mesh(V, [(0, 1, 2, 3)], mat, uvs=[[(0, 0), (L, 0), (L, 1), (0, 1)]], coll="ground", collide=False,
                    name=name)


def seg_dist(p, a, b) -> float:
    """Distance from point p to segment a-b (xy)."""
    p, a, b = (np.asarray(v, float)[:2] for v in (p, a, b))
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + ab * t)))


# --- Objaverse pieces (library_objaverse.json / catalog_objaverse.json) --------------------------

#: curated by role from the Objaverse contact sheets: real metre scale, textured, <= 80k triangles,
#: CC-BY / CC-BY-SA only (the NC files are left out). The licence of every file a course uses is in
#: course.json ``files``; the kits list their Objaverse placements in ``chosen["objaverse"]``.
OBJ = {
    "truck": ["27a0aaa34e7d4ea7877393178d82b01c", "393730a2af3d408c86cd7fa967dda242",
              "56f462ac530840999e3f17a0163fbcac", "f0b7ca0f694c4213b799677e49041896",
              "37b10f9c92584dfdb3d1337697dc8843"],
    "van": ["4258fa5bb1f74a2da7ed2090df0cb55b", "f69de1315bb049c8946d57f6006acd73"],
    "pickup": ["7cf3226824a94b0d909a7ba0949bb5fe", "aa1ff58f50ba4286b4e0c987824ed571",
               "dade78dc96e34f1a8cbcf14dd47d84de", "1986239266ee44c788fd6effe4055704",
               "3b9c6eaf305342b48ebcb8295b2d2632"],
    "forklift": ["060f3f8bc7de4e6ca2f348d414702e9d", "9ad80a7bd90b483f8dbf0e293251cff7",
                 "b7bea04856e1401898250f313131659f", "ce8b2c714b2f472bbac58c6eab088ba2",
                 "d40cae50e04145dd997cdca415cd72ad", "d659c366e24643f0b72ad0fe1ef1a60b",
                 "bb8b8dab8b4c4fc1ac6af310b70f1b6e"],
    "generator": ["423cfdbd612c486aada7bb1bc44695a9", "4b50a06663fe44079503ca53f7eb399f",
                  "f34ba6c159e04152ac1883caaca2ae18", "f394f4f7f446474fa65e3217923bdaaf",
                  "a6c7477cc905448fa16310be52a989c1", "05cb0187751541f195d8b7cf3c4736ef"],
    "ibc": ["8b80338880564f6e9c97aa517c60ec37"],
    "dumpster": ["135d33df60304481830d6ddd8c43cc47", "73f80ae78964472bbda19328b354fa35",
                 "bed7cc2208ae481f800a1d8516839be5", "e8b38df736994cf8b4f7a3a9d2980f6b",
                 "989ea7f01b924859b3484ee27103e8d9"],
    "container": ["133b0810d90d4d3992e1f8d67f439e03", "731960026f8240b1b5952dec3ec339fa",
                  "b3cf2804b4e5422e925d5f4cd56ab5de", "7e6a1c61b9274e439b4b198ef67728a3"],
    "gas": ["4da7e80c3b6a42328d48dd811642db6e", "e788acc021ac45a598178d24ce196737",
            "a68f90323fd44cf7a22c0afe30d6a2f8", "f1075a710e554e21b9b678a053e1f9ea"],
    "barrel": ["205a7012c31742afb2b81d2cbfe0f2b2", "c9b66b0949f247c4ae78338053650ee2",
               "e5fccac2cc97468f90048944188b1b7a"],
    "excavator": ["5d08010ce3ea44cbb57cf8d86a0403f2", "f6356e9dbec14b548f7e9d36911d71bb",
                  "940258497f9f4eba9710bb359a9e777e", "63cbe44a46784449907d71c4b6953682"],
    "loader": ["7c1b41c38b3747aeb3d12298aec42376", "d23c7e5565104a1db2e1c4641d4ed59b",
               "df9c3cfa98bb40598cfa7b96027db2a5"],
    "dozer": ["b26ce56081304c1988ec7490e439d882"],
    "skip": ["bc957334d53c4b2aada512f5213cdebf"],
    "wheelbarrow": ["08c0739e36a74cbc9742125eb1199435", "0fd3dd1da381468c86d3bbe50a4db27b",
                    "c0dcebcb0d4b47bd98df534334f4caf6", "4fc9d44fa5064daebebefbec98d472cb",
                    "68c8d8c7715947d6a393feecc83c0553"],
    "barrier": ["3119d577027246e99f9deb89ea584d58", "b2dd789e69bd401da49a94790d119079",
                "925da9cddeec426c9ce900b17cecb0e6", "10d590da894543a2a9c2f14199c3fd7f",
                "8b630eb9f27d4a2b9f1d1629b552b7d3", "ab21217481094e9a84f989e71fd56a59",
                "e16360dfad7d47cfbbfdffad13190ab7"],
    "cone": ["cf5f95bc3d0347499b41bdd8e81ef9e0", "90d32979dac545c6a465399e453c3484",
             "d0eb4829fc9242dba0e592d553786f3f", "f28523a0b7684d6c8972e8a853394708",
             "d6c77b5fe7844dee9fbef1ce5390997d"],
    "ladder": ["12304e30fc9d4c1288f6634dd80a2bb8", "d9d7989ac61848a09eeb60708dc0d941"],
}


def obj_keys(ctx, *roles) -> list[str]:
    """Piece keys of the curated files for ``roles`` that the library has (not excluded, not NC)."""
    out = []
    for role in roles:
        for uid in OBJ.get(role, ()):
            fk = f"obj:{uid}"
            e = ctx.lib.items.get(fk)
            if e is None or e.get("exclude") or "nc" in str(e.get("licence_short", "")).lower():
                continue
            out += [f"{fk}#{p['index']}" for p in ctx.lib.pieces(fk) if not p.get("exclude")]
    return out


def along_yaw(ctx, pkey, direction: float) -> float:
    """The yaw that lays a piece's long side along ``direction``, facing either way."""
    sx, sy, _ = ctx.lib.size(pkey)
    return float(direction) + (0.0 if sx >= sy else math.pi / 2) + (math.pi if ctx.rng.random() < 0.5 else 0.0)


def long_half(ctx, pkey) -> tuple[float, float]:
    """(long, short) half sides of a piece's footprint."""
    sx, sy, _ = ctx.lib.size(pkey)
    return max(sx, sy) / 2, min(sx, sy) / 2


def place_obj(ctx, pkey, xy, yaw, margin=0.3, coll="scatter", check=True, ignore_from=None):
    """An Objaverse piece standing on the ground at ``xy`` (footprint centre),
    ``yaw``: its real footprint rectangle must be ``margin`` clear (``check``;
    of the footprints claimed before index ``ignore_from`` when given, so a
    group can touch) and is claimed. Skipped (None) when the loaded piece
    disagrees with the catalogue size by more than 25 % (a bad scale)."""
    sx, sy, _ = ctx.lib.size(pkey)
    if check and ignore_from is None and not ctx.free_rect(xy, (sx / 2, sy / 2), yaw, margin=margin):
        return None
    if check and ignore_from is not None:
        foot = ctx.foot[:ignore_from]
        R = rotz(yaw)[:2, :2]
        for u in np.linspace(-sx / 2, sx / 2, max(2, int(sx / 0.4) + 1)):
            for v in np.linspace(-sy / 2, sy / 2, max(2, int(sy / 0.4) + 1)):
                q = np.asarray(xy, float)[:2] + R @ np.array([u, v])
                if min((BC.footprint_distance(q, f) for f in foot), default=1e9) < margin:
                    return None
    got = np.asarray(ctx.piece(pkey)["size"], float)
    want = np.asarray(ctx.lib.size(pkey), float)
    if np.any(np.abs(got - want) > 0.25 * np.maximum(want, 0.2)):
        return None
    ob = ctx.instance(pkey, xy, 0.0, yaw, 1.0, coll=coll)
    ctx.claim_rect(xy, (got[0] / 2, got[1] / 2), yaw, kind="obj")
    return ob


def obj_record(ctx) -> list[dict]:
    """The Objaverse pieces a course placed: key, name, licence, author, count, colliding count
    (course.json ``chosen["objaverse"]``; ``files`` carries the same licences per file)."""
    out = {}
    for p in ctx.placed:
        if not p["piece"].startswith("obj:"):
            continue
        e = ctx.lib.items[p["piece"].split("#")[0]]
        r = out.setdefault(p["piece"], {"key": p["piece"], "name": str(e.get("name", ""))[:60],
                                        "licence": e.get("licence"), "authors": e.get("authors", [])[:3],
                                        "n": 0, "colliding": 0})
        r["n"] += 1
        r["colliding"] += int(p["collide"])
    return sorted(out.values(), key=lambda r: r["key"])
