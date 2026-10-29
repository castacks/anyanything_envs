"""Indoor helpers shared by the ``garage`` and ``warehouse`` kits (runs inside Blender).

* ``Batch``: many small boxes / lofts / cylinders merged into one mesh per
  material, split into a colliding part (footprint meets the cloud zone) and
  a render-only part (wholly outside it) -- racks with thousands of beams and
  cartons stay a handful of objects, and ``ctx.mesh`` still tags them.
  ``Batch.solid`` draws a group of touching boxes (a carton stack) whole but
  lets the cloud sample only its outer faces: the render is unchanged and the
  insides (3-4x the hull in points) never reach the labeller;
* the kit side of the cloud budget (``area_budget``, ``cloud_area``), of the
  family's own free space (``claim_passages``), of hanging a floating primitive
  without closing its over route (``hang_rods``: splayed bridles), and of what
  a ceiling does to the family, measured on the built scene (``overhead``,
  ``ceiling_semantics``);
* custom materials the scan library has no scan for: car paint (clear coat),
  worn floor paint (a decal that lets the concrete show through where it is
  worn), oil stains, stretch wrap;
* procedural vehicles (cars, forklifts), pallets and pallet loads, light
  fixtures with their area lights, floor markings (arrows, hatching, joints).

Everything here goes through the ``Ctx`` (``ctx.mesh``, ``ctx.flat``,
``ctx.area_light``), so the labeller contract holds: in-zone geometry
collides, floor paint is a < 5 cm non-colliding overlay.
"""

from __future__ import annotations

import math
import re

import bpy
import numpy as np

import build_course as BC


# --- material tweaks -------------------------------------------------------------------------

def set_rough(mat, mul: float = 1.0, add: float = 0.0):
    """Scale every Principled BSDF's roughness in ``mat`` (polished floors,
    satin paint): a linked roughness map gets a multiply-add node."""
    nt = mat.node_tree
    for b in [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"]:
        inp = b.inputs["Roughness"]
        if inp.is_linked:
            src = inp.links[0].from_socket
            m = nt.nodes.new("ShaderNodeMath")
            m.operation = "MULTIPLY_ADD"
            m.use_clamp = True
            m.inputs[1].default_value = mul
            m.inputs[2].default_value = add
            nt.links.new(src, m.inputs[0])
            nt.links.new(m.outputs[0], inp)
        else:
            inp.default_value = float(np.clip(inp.default_value * mul + add, 0.0, 1.0))
    return mat


def car_paint(rgb, metallic: float = 0.7, rough: float = 0.32, name: str = "car_paint"):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    for k, v in (("Coat Weight", 1.0), ("Coat Roughness", 0.04)):
        if k in b.inputs:
            b.inputs[k].default_value = v
    nt = m.node_tree
    bv = nt.nodes.new("ShaderNodeBevel")
    bv.samples = 4
    bv.inputs["Radius"].default_value = 0.03
    nt.links.new(bv.outputs["Normal"], b.inputs["Normal"])
    return m


def plain(rgb, rough=0.5, metal=0.0, name="plain", emission=None, bevel=0.0):
    """A plain material with an optional bevel (plastic, rubber, lamp housings)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emission is not None:
        b.inputs["Emission Color"].default_value = (*emission[:3], 1.0)
        b.inputs["Emission Strength"].default_value = emission[3]
    if bevel > 0:
        nt = m.node_tree
        bv = nt.nodes.new("ShaderNodeBevel")
        bv.samples = 4
        bv.inputs["Radius"].default_value = bevel
        nt.links.new(bv.outputs["Normal"], b.inputs["Normal"])
    return m


def wrap_film(name="stretch_wrap"):
    """Stretch-wrapped pallet film: milky, glossy, crinkled (opaque for depth)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.78, 0.8, 0.8, 1.0)
    b.inputs["Roughness"].default_value = 0.18
    if "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = 0.6
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 6.0
    nz.inputs["Detail"].default_value = 8.0
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    bp = nt.nodes.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = 0.35
    nt.links.new(nz.outputs["Fac"], bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    # milky: a little of the cardboard shows through as a darker, warmer tone
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 0.55, 0.95
    nz2 = nt.nodes.new("ShaderNodeTexNoise")
    nz2.inputs["Scale"].default_value = 1.5
    nt.links.new(tc.outputs["Object"], nz2.inputs["Vector"])
    nt.links.new(nz2.outputs["Fac"], mr.inputs["Value"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(mr.outputs["Result"], mix.inputs["Factor"])
    mix.inputs[6].default_value = (0.55, 0.45, 0.32, 1.0)
    mix.inputs[7].default_value = (0.85, 0.87, 0.88, 1.0)
    nt.links.new(mix.outputs[2], b.inputs["Base Color"])
    return m


def decal(rgb, rough=0.55, wear=0.35, seed=0.0, name="floor_paint", scale=1.2, emission=None):
    """Worn floor paint for flat overlays: paint mixed with a transparent BSDF
    by a noise mask, so the concrete shows through where traffic wore it off.
    (Overlays never collide, so the alpha never reaches the cloud.)"""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    if emission is not None:
        b.inputs["Emission Color"].default_value = (*emission[:3], 1.0)
        b.inputs["Emission Strength"].default_value = emission[3]
    out = N["Material Output"]
    tr = N.new("ShaderNodeBsdfTransparent")
    tc = N.new("ShaderNodeTexCoord")
    mp = N.new("ShaderNodeMapping")
    mp.inputs["Location"].default_value = (seed, 2 * seed, 0.0)
    L.new(tc.outputs["Object"], mp.inputs["Vector"])
    nz = N.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = 10.0
    nz.inputs["Roughness"].default_value = 0.7
    L.new(mp.outputs["Vector"], nz.inputs["Vector"])
    mr = N.new("ShaderNodeMapRange")
    lo = 0.62 - 0.25 * wear
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = lo, lo - 0.08
    L.new(nz.outputs["Fac"], mr.inputs["Value"])
    # fine speckle: paint is chipped at small scale too
    nz2 = N.new("ShaderNodeTexNoise")
    nz2.inputs["Scale"].default_value = 40.0
    nz2.inputs["Detail"].default_value = 4.0
    L.new(mp.outputs["Vector"], nz2.inputs["Vector"])
    mr2 = N.new("ShaderNodeMapRange")
    mr2.inputs["From Min"].default_value, mr2.inputs["From Max"].default_value = 0.30 + 0.1 * wear, 0.36 + 0.1 * wear
    L.new(nz2.outputs["Fac"], mr2.inputs["Value"])
    mul = N.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    L.new(mr.outputs["Result"], mul.inputs[0])
    L.new(mr2.outputs["Result"], mul.inputs[1])
    mix = N.new("ShaderNodeMixShader")
    L.new(mul.outputs[0], mix.inputs["Fac"])
    L.new(tr.outputs["BSDF"], mix.inputs[1])
    L.new(b.outputs["BSDF"], mix.inputs[2])
    L.new(mix.outputs["Shader"], out.inputs["Surface"])
    return m


def stain(dark=0.03, rough=0.25, strength=0.8, name="oil_stain", tone=(0.03, 0.028, 0.025)):
    """An oil / water stain decal: dark and glossier, fading out radially over
    the overlay's own bounding box (Generated coordinates) with a noisy edge."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*tone, 1.0)
    b.inputs["Roughness"].default_value = rough
    out = N["Material Output"]
    tr = N.new("ShaderNodeBsdfTransparent")
    tc = N.new("ShaderNodeTexCoord")
    sub = N.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = (0.5, 0.5, 0.0)
    L.new(tc.outputs["Generated"], sub.inputs[0])
    ln = N.new("ShaderNodeVectorMath")
    ln.operation = "LENGTH"
    L.new(sub.outputs[0], ln.inputs[0])
    nz = N.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 3.0
    nz.inputs["Detail"].default_value = 8.0
    L.new(tc.outputs["Generated"], nz.inputs["Vector"])
    add = N.new("ShaderNodeMath")
    add.operation = "MULTIPLY_ADD"
    add.inputs[1].default_value = 0.35
    L.new(nz.outputs["Fac"], add.inputs[0])
    L.new(ln.outputs["Value"], add.inputs[2])
    mr = N.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.62, 0.32
    mr.inputs["To Max"].default_value = strength
    L.new(add.outputs[0], mr.inputs["Value"])
    mix = N.new("ShaderNodeMixShader")
    L.new(mr.outputs["Result"], mix.inputs["Fac"])
    L.new(tr.outputs["BSDF"], mix.inputs[1])
    L.new(b.outputs["BSDF"], mix.inputs[2])
    L.new(mix.outputs["Shader"], out.inputs["Surface"])
    return m


# --- batching ---------------------------------------------------------------------------------

def _rotz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


_BOX_F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
_BOX_N = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], float)   # per _BOX_F
_BOX_C = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


class Batch:
    """Accumulates geometry per (material, colliding?, collection, roof, smooth)
    and writes one ``ctx.mesh`` per key on ``flush``."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.parts: dict = {}
        self.area_in = 0.0          # m^2 of colliding faces in the zone and under the cloud top (cloud estimate)
        self._zmax = cloud_zmax()

    def _key(self, mat, lo, hi, coll, roof, smooth, collide):
        if collide is None:
            collide = bool(self.ctx.rect_meets_zone(lo, hi) and hi[2] > 0.05)
        return (mat.name, bool(collide), coll, bool(roof), bool(smooth)), collide

    def add(self, V, F, mat, coll="structure", roof=False, smooth=False, collide=None):
        V = np.asarray(V, float)
        key, col = self._key(mat, V.min(0), V.max(0), coll, roof, smooth, collide)
        p = self.parts.setdefault(key, {"V": [], "F": [], "n": 0, "mat": mat})
        n = p["n"]
        p["V"].append(V)
        p["F"].extend(tuple(i + n for i in f) for f in F)
        p["n"] += len(V)
        if col:
            lo, hi = V.min(0), V.max(0)
            if hi[2] > 0.05 and lo[2] < self._zmax:
                self.area_in += _approx_area(V, F, self.ctx.zone, self._zmax)

    def box(self, c, half, mat, yaw=0.0, R=None, **kw):
        c = np.asarray(c, float)
        h = np.asarray(half, float)
        R = _rotz(yaw) if R is None else np.asarray(R, float)
        V = _BOX_C * h @ R.T + c
        self.add(V, _BOX_F, mat, **kw)

    def solid(self, boxes, coll="structure", rest=None, push=0.06, **kw):
        """A group of touching or near-touching boxes ``[(centre, half, R, mat)]`` -- a carton
        stack, sacks, bins: every box is drawn whole, but a face whose outside (its centre
        pushed ``push`` m out) lies in another box of the group, or that faces down onto
        ``rest`` (the pallet deck it stands on), is render-only. The cloud then samples the
        group's outer surface only; the image is unchanged."""
        if not boxes:
            return
        C = np.array([b[0] for b in boxes], float)
        H = np.array([b[1] for b in boxes], float)
        R = np.array([b[2] for b in boxes], float)
        n = len(boxes)
        # face centres pushed out, (n, 6, 3)
        loc = _BOX_N[None] * H[:, None, :]                              # local face centres
        nrm = np.einsum("nij,fj->nfi", R, _BOX_N)                        # world normals
        P = C[:, None, :] + np.einsum("nij,nfj->nfi", R, loc) + push * nrm
        rel = P.reshape(-1, 1, 3) - C[None]                              # (6n, n, 3)
        q = np.abs(np.einsum("mji,kmj->kmi", R, rel))                    # in each box's frame
        inside = (q <= H[None] + 1e-4).all(-1)
        inside[np.arange(6 * n), np.repeat(np.arange(n), 6)] = False     # not in itself
        covered = inside.any(1).reshape(n, 6)
        if rest is not None:
            zf = C[:, 2][:, None] + np.einsum("nij,nfj->nfi", R, loc)[:, :, 2]
            covered |= (nrm[:, :, 2] < -0.9) & (np.abs(zf - rest) < 0.03)
        for i, (c, h, Ri, mat) in enumerate(boxes):
            V = _BOX_C * np.asarray(h, float) @ np.asarray(Ri, float).T + np.asarray(c, float)
            ext = [f for f, cv in zip(_BOX_F, covered[i]) if not cv]
            inn = [f for f, cv in zip(_BOX_F, covered[i]) if cv]
            if ext:
                self.add(V, ext, mat, coll=coll, **kw)
            if inn:
                self.add(V, inn, mat, coll=coll, collide=False, **kw)

    def box_lohi(self, lo, hi, mat, **kw):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        self.box((lo + hi) / 2, (hi - lo) / 2, mat, **kw)

    def cyl(self, a, b, r, mat, seg=12, cap=True, **kw):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ax = b - a
        L = float(np.linalg.norm(ax))
        if L < 1e-6:
            return
        z = ax / L
        x = np.cross(z, [0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        ang = np.linspace(0, 2 * math.pi, seg, endpoint=False)
        ring = np.cos(ang)[:, None] * x + np.sin(ang)[:, None] * y
        V = np.concatenate([a + r * ring, b + r * ring])
        F = [(i, (i + 1) % seg, seg + (i + 1) % seg, seg + i) for i in range(seg)]
        if cap:
            F += [tuple(range(seg))[::-1], tuple(range(seg, 2 * seg))]
        self.add(V, F, mat, smooth=kw.pop("smooth", True), **kw)

    def loft(self, stations, mat, xy, yaw, z0=0.0, k=3, **kw):
        """A car-body-like loft: ``stations`` = [(u, half_w, z_bot, z_top, r)]
        along the local x axis, rounded-rectangle sections, capped; placed at
        ``xy`` turned by ``yaw``."""
        rings = []
        for st in stations:
            u, w, zb, zt, r = st[:5]
            wt = st[5] if len(st) > 5 else w            # top width (tumblehome)
            r = max(0.005, min(r, w * 0.95, wt * 0.95, (zt - zb) * 0.49))
            pts = []
            for cy, cz, a0 in ((w - r, zb + r, -90), (wt - r, zt - r, 0), (-(wt - r), zt - r, 90),
                               (-(w - r), zb + r, 180)):
                for j in range(k + 1):
                    a = math.radians(a0 + 90 * j / k)
                    pts.append((u, cy + r * math.cos(a), cz + r * math.sin(a)))
            rings.append(pts)
        m = len(rings[0])
        V = np.array([p for ring in rings for p in ring], float)
        F = []
        for i in range(len(rings) - 1):
            for j in range(m):
                F.append((i * m + j, i * m + (j + 1) % m, (i + 1) * m + (j + 1) % m, (i + 1) * m + j))
        F.append(tuple(range(m))[::-1])
        F.append(tuple(range((len(rings) - 1) * m, len(rings) * m)))
        V = V @ _rotz(yaw).T + np.array([xy[0], xy[1], z0])
        self.add(V, F, mat, smooth=kw.pop("smooth", True), **kw)

    def flush(self):
        out = []
        for (mname, col, coll, roof, smooth), p in self.parts.items():
            V = np.concatenate(p["V"])
            ob = self.ctx.mesh(V, p["F"], p["mat"], coll=coll, collide=(None if col else False), roof=roof,
                               smooth=smooth, name=f"b_{mname[:20]}")
            if not col:
                ob["collide"] = 0
            out.append(ob)
        self.parts = {}
        return out


def _inside_share(lo, hi, zone, zmax):
    """Share of a face's bounding box inside the zone and under ``zmax``, per axis
    (a 200 m ceiling or a duct run across the hall counts only where the cloud samples it)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    box_lo = np.array([zone[0], zone[1], -np.inf]) if zone is not None else np.full(3, -np.inf)
    box_hi = np.array([zone[2], zone[3], np.inf if zmax is None else zmax]) if zone is not None else \
        np.array([np.inf, np.inf, np.inf if zmax is None else zmax])
    ext = hi - lo
    ov = np.minimum(hi, box_hi) - np.maximum(lo, box_lo)
    flat = ext < 1e-9
    inside = (lo >= box_lo - 1e-9) & (hi <= box_hi + 1e-9)
    share = np.where(flat, (ov >= -1e-9).astype(float), np.clip(ov, 0.0, None) / np.where(flat, 1.0, ext))
    share = np.where(inside, 1.0, share)
    return float(np.prod(share, axis=-1)) if share.ndim == 1 else np.prod(share, axis=-1)


def _approx_area(V, F, zone=None, zmax=None) -> float:
    """Area of the faces (fans) inside ``zone`` and under ``zmax`` (bounding-box share)."""
    a = 0.0
    for f in F:
        if len(f) < 3:
            continue
        Q = V[list(f)]
        s = _inside_share(Q.min(0), Q.max(0), zone, zmax) if (zone is not None or zmax is not None) else 1.0
        if s <= 0.0:
            continue
        p0 = Q[0]
        fa = 0.0
        for i in range(1, len(f) - 1):
            fa += 0.5 * float(np.linalg.norm(np.cross(Q[i] - p0, Q[i + 1] - p0)))
        a += s * fa
    return a


# --- floor markings ----------------------------------------------------------------------------

def line(ctx, a, b, w, mat, z=0.004):
    return ctx.strip(a, b, w, mat, z=z, name="marking")


def dashed(ctx, a, b, w, mat, dash=3.0, gap=3.0, z=0.004):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 1e-6:
        return
    d = (b - a) / L
    t = 0.0
    while t < L:
        ctx.strip(a + d * t, a + d * min(t + dash, L), w, mat, z=z, name="marking")
        t += dash + gap


def arrow(ctx, c, direction, mat, length=3.0, width=0.35, head=1.1, z=0.004):
    """A painted lane arrow (shaft + head) centred at ``c`` pointing ``direction``."""
    d = np.asarray(direction, float)[:2]
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    c = np.asarray(c, float)[:2]
    tail = c - d * length / 2
    neck = c + d * (length / 2 - head)
    tip = c + d * length / 2
    ctx.flat([tail - n * width / 2, neck - n * width / 2, neck + n * width / 2, tail + n * width / 2], mat, z=z,
             name="arrow")
    ctx.flat([neck - n * width * 1.6, tip, neck + n * width * 1.6], mat, z=z, name="arrow")


def hatch(ctx, c, half, yaw, mat, pitch=0.9, w=0.15, z=0.004):
    """A diagonally hatched keep-clear box: border + 45 degree stripes."""
    c = np.asarray(c, float)[:2]
    hx, hy = half
    R = _rotz(yaw)[:2, :2]
    P = lambda u, v: c + R @ np.array([u, v])  # noqa: E731
    for (u0, v0), (u1, v1) in (((-hx, -hy), (hx, -hy)), ((hx, -hy), (hx, hy)), ((hx, hy), (-hx, hy)),
                               ((-hx, hy), (-hx, -hy))):
        ctx.strip(P(u0, v0), P(u1, v1), w, mat, z=z, name="hatch")
    t = -hx - hy + pitch
    while t < hx + hy:
        # the line u - v = t clipped to the box
        us = [u for u in (t - hy, t + hy, -hx, hx) if -hx - 1e-6 <= u <= hx + 1e-6 and -hy - 1e-6 <= u - t <= hy + 1e-6]
        if len(us) >= 2:
            ua, ub = min(us), max(us)
            if ub - ua > 0.05:
                ctx.strip(P(ua, ua - t), P(ub, ub - t), w, mat, z=z, name="hatch")
        t += pitch


def joints(ctx, rect, spacing, mat, w=0.012, z=0.0025, jitter=0.0):
    """Saw-cut slab joints: a thin dark grid over ``rect`` (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = rect
    ox = ctx.rng.uniform(0, spacing)
    oy = ctx.rng.uniform(0, spacing)
    for x in np.arange(x0 + ox, x1, spacing):
        ctx.strip((x, y0), (x, y1), w, mat, z=z, name="joint")
    for y in np.arange(y0 + oy, y1, spacing):
        ctx.strip((x0, y), (x1, y), w, mat, z=z, name="joint")


def blob(ctx, c, rx, ry, yaw, mat, z=0.0035, n=10, name="stain"):
    """An irregular flat blob (stain decal carrier)."""
    c = np.asarray(c, float)[:2]
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rr = 1.0 + ctx.rng.normal(0, 0.12, n)
    P = np.stack([rx * rr * np.cos(ang), ry * rr * np.sin(ang)], 1) @ _rotz(yaw)[:2, :2].T + c
    return ctx.flat(P, mat, z=z, name=name)


# --- lights ------------------------------------------------------------------------------------

class Lamps:
    """Light fixtures: an emissive body the camera sees + an area light that
    lights the scene. ``gain`` scales every power (it follows the exposure so
    lamps render alike under any HDRI)."""

    def __init__(self, ctx, gain: float, rgb=(1.0, 0.96, 0.9)):
        self.ctx, self.gain, self.rgb = ctx, gain, rgb
        self.n = 0
        self._emit = {}

    def emit_mat(self, strength, rgb=None):
        rgb = rgb or self.rgb
        k = (round(strength, 3), tuple(rgb))
        if k not in self._emit:
            self._emit[k] = plain((0.9, 0.9, 0.9), rough=0.4, name="lamp_emit",
                                  emission=(*rgb, strength * self.gain))
        return self._emit[k]

    def tube(self, B, c, yaw, housing, length=1.25, power=60.0, strength=25.0, light=True, coll="structure",
             roof=True, twin=False):
        """A fluorescent / LED batten under a ceiling at height c[2] (its top)."""
        x, y, z = c
        w = 0.16 if twin else 0.1
        B.box((x, y, z - 0.035), (length / 2, w / 2, 0.035), housing, yaw=yaw, coll=coll, roof=roof)
        B.box((x, y, z - 0.08), (length / 2 - 0.03, w / 2 - 0.02, 0.012), self.emit_mat(strength), yaw=yaw,
              coll=coll, roof=roof)
        if light:
            ob = self.ctx.area_light((x, y, z - 0.1), length, power * self.gain, rgb=self.rgb,
                                     direction=(0, 0, -1), shape="RECTANGLE")
            ob.data.size_y = w
            ob.rotation_euler[2] = yaw
            self.n += 1

    def disc(self, B, c, housing, r=0.24, power=300.0, strength=30.0, light=True, coll="structure", roof=True,
             drop=0.0, rod_mat=None):
        """A round high-bay (UFO) fixture hanging ``drop`` m below height c[2]."""
        x, y, z = c
        zt = z - drop
        if drop > 0 and rod_mat is not None:
            B.cyl((x, y, zt), (x, y, z), 0.012, rod_mat, seg=6, coll=coll, roof=roof)
        B.cyl((x, y, zt - 0.16), (x, y, zt), r, housing, seg=16, coll=coll, roof=roof)
        B.cyl((x, y, zt - 0.18), (x, y, zt - 0.16), r * 0.9, self.emit_mat(strength), seg=16, coll=coll, roof=roof)
        if light:
            self.ctx.area_light((x, y, zt - 0.2), 2 * r, power * self.gain, rgb=self.rgb, direction=(0, 0, -1),
                                shape="DISK")
            self.n += 1


# --- vehicles ----------------------------------------------------------------------------------

#: car colours roughly as parked cars show them (white, black, greys, silver lead)
CAR_COLOURS = [((0.80, 0.80, 0.79), 0.2, 20), ((0.015, 0.015, 0.017), 0.6, 18), ((0.22, 0.23, 0.24), 0.8, 12),
               ((0.52, 0.53, 0.55), 0.9, 12), ((0.05, 0.09, 0.22), 0.7, 7), ((0.45, 0.03, 0.03), 0.6, 6),
               ((0.10, 0.12, 0.10), 0.6, 2), ((0.38, 0.30, 0.22), 0.6, 3), ((0.62, 0.60, 0.55), 0.7, 5),
               ((0.10, 0.25, 0.40), 0.7, 3), ((0.70, 0.35, 0.05), 0.5, 1), ((0.02, 0.10, 0.05), 0.6, 1)]

CAR_KINDS = {  # length, width, height, weight
    "hatch": ((3.9, 4.3), (1.70, 1.80), (1.42, 1.52), 5),
    "sedan": ((4.5, 4.95), (1.78, 1.86), (1.40, 1.48), 6),
    "wagon": ((4.6, 4.9), (1.78, 1.86), (1.46, 1.52), 2),
    "suv": ((4.4, 4.9), (1.82, 1.95), (1.62, 1.78), 6),
    "van": ((4.9, 5.4), (1.9, 2.0), (1.90, 2.05), 1),
    "pickup": ((5.2, 5.7), (1.9, 2.0), (1.75, 1.90), 1),
}


class Fleet:
    """Procedural cars. Materials are made once per build. ``rng`` (default ``ctx.rng``): a kit
    passes its own generator so a bay taken by a library car does not shift the scene's stream."""

    def __init__(self, ctx, rng=None):
        self.ctx = ctx
        self.rng = rng = ctx.rng if rng is None else rng
        w = np.array([c[2] for c in CAR_COLOURS], float)
        self.paints = []
        for i in range(len(CAR_COLOURS)):
            rgb, met, _ = CAR_COLOURS[i]
            self.paints.append(car_paint(rgb, metallic=met, rough=float(rng.uniform(0.22, 0.38)), name=f"car_{i}"))
        self.pw = w / w.sum()
        self.glass = plain((0.02, 0.025, 0.03), rough=0.03, metal=0.0, name="car_glass")
        gb = self.glass.node_tree.nodes["Principled BSDF"]
        if "Specular IOR Level" in gb.inputs:
            gb.inputs["Specular IOR Level"].default_value = 0.8
        self.tyre = plain((0.025, 0.025, 0.025), rough=0.75, name="tyre")
        self.rim = plain((0.55, 0.56, 0.58), rough=0.25, metal=1.0, name="rim")
        self.trim = plain((0.03, 0.03, 0.03), rough=0.5, name="car_trim")
        self.chrome = plain((0.8, 0.8, 0.8), rough=0.1, metal=1.0, name="chrome")
        self.head = plain((0.9, 0.9, 0.88), rough=0.05, name="headlamp")
        self.tail = plain((0.5, 0.02, 0.02), rough=0.15, name="taillamp")
        self.plate = plain((0.85, 0.85, 0.8), rough=0.4, name="plate")
        kinds = list(CAR_KINDS)
        self.kinds = kinds
        kw = np.array([CAR_KINDS[k][3] for k in kinds], float)
        self.kw = kw / kw.sum()

    def pick_kind(self):
        return self.kinds[int(self.rng.choice(len(self.kinds), p=self.kw))]

    def dims(self, kind):
        rng = self.rng
        Lr, Wr, Hr, _ = CAR_KINDS[kind]
        return float(rng.uniform(*Lr)), float(rng.uniform(*Wr)), float(rng.uniform(*Hr))

    def car(self, B, xy, yaw, kind=None, dims=None, paint=None, coll="scatter"):
        """One car, footprint centred at ``xy``, nose along +yaw. Returns its
        half extents (L/2, W/2) for claims."""
        rng = self.rng
        kind = kind or self.pick_kind()
        L, W, H = dims or self.dims(kind)
        paint = paint or self.paints[int(rng.choice(len(self.paints), p=self.pw))]
        w = W / 2
        # body: stations along u in [-L/2, L/2] (+u = front)
        s = lambda f: -L / 2 + f * L  # noqa: E731  (f = 0 rear .. 1 front)
        belt = {"hatch": 0.92, "sedan": 0.90, "wagon": 0.92, "suv": 1.05, "van": 1.05, "pickup": 1.05}[kind]
        belt = min(belt, H * 0.62)
        wr, wf = 0.19, 0.81                     # wheel centres (fractions)
        rw = {"suv": 0.36, "van": 0.34, "pickup": 0.38}.get(kind, 0.32)
        zb = 0.22 if kind not in ("suv", "pickup") else 0.30
        arch = rw + 0.06
        body = [
            (s(0.0), w * 0.86, zb + 0.10, belt * 0.88, 0.14),
            (s(0.02), w * 0.95, zb + 0.02, belt * 0.97, 0.14),
            (s(wr - 0.10), w, zb, belt, 0.12),
            (s(wr - 0.085), w, arch + rw * 0.2, belt, 0.10),
            (s(wr + 0.085), w, arch + rw * 0.2, belt, 0.10),
            (s(wr + 0.10), w, zb, belt, 0.12),
            (s(wf - 0.10), w, zb, belt * 0.98, 0.12),
            (s(wf - 0.085), w, arch + rw * 0.2, belt * 0.97, 0.10),
            (s(wf + 0.085), w, arch + rw * 0.2, belt * 0.92, 0.10),
            (s(wf + 0.10), w * 0.98, zb, belt * 0.9, 0.12),
            (s(0.98), w * 0.93, zb + 0.03, belt * 0.84, 0.14),
            (s(1.0), w * 0.84, zb + 0.10, belt * 0.76, 0.14),
        ]
        if kind == "pickup":
            body[0] = (s(0.0), w * 0.95, zb + 0.1, belt * 0.98, 0.06)
            body[1] = (s(0.02), w * 0.98, zb + 0.02, belt, 0.06)
        body = [(*st, st[1] * 0.95) for st in body]
        B.loft(body, paint, xy, yaw, coll=coll)
        # greenhouse (glass) and roof
        gh = {"hatch": (0.03, 0.08, 0.55, 0.68), "sedan": (0.14, 0.24, 0.55, 0.70), "wagon": (0.03, 0.08, 0.55, 0.70),
              "suv": (0.03, 0.07, 0.58, 0.72), "van": (0.02, 0.03, 0.70, 0.80), "pickup": (0.40, 0.42, 0.60, 0.67)}[kind]
        r0, r1, f1, f0 = gh                      # rear base, rear roof, front roof, front base (fractions)
        top = H
        gw = w * 0.9
        tw = 0.8 if kind not in ("van",) else 0.9
        cab = [(s(r0), gw * 0.98, belt - 0.02, belt + 0.02, 0.03, gw * 0.97),
               (s(r1), gw * 0.95, belt - 0.02, top - 0.02, 0.10, gw * tw),
               (s(f1), gw * 0.95, belt - 0.02, top - 0.02, 0.10, gw * tw),
               (s(f0), gw * 0.98, belt - 0.02, belt + 0.02, 0.03, gw * 0.97)]
        B.loft(cab, self.glass, xy, yaw, coll=coll)
        roof = [(s(r1 + 0.01), gw * tw * 0.98, top - 0.05, top + 0.005, 0.04),
                (s(f1 - 0.01), gw * tw * 0.98, top - 0.05, top + 0.005, 0.04)]
        B.loft(roof, paint, xy, yaw, coll=coll)
        R = _rotz(yaw)
        c3 = np.array([xy[0], xy[1], 0.0])
        P = lambda u, v, z: R @ np.array([u, v, z]) + c3  # noqa: E731
        # B-pillars (body colour) so the glass reads as windows
        for f in ((r1 + f1) / 2 + 0.03,):
            for sv in (-1, 1):
                Rp = R @ np.array([[1.0, 0, 0], [0, math.cos(sv * 0.22), -math.sin(sv * 0.22)],
                                   [0, math.sin(sv * 0.22), math.cos(sv * 0.22)]])
                B.box(P(s(f), sv * gw * (0.95 + tw) / 2, (belt + top) / 2), (0.05, 0.015, (top - belt) / 2 + 0.01),
                      paint, R=Rp, coll=coll)
        if kind == "pickup":                     # load bed walls
            for sv in (-1, 1):
                B.box(P(s(0.21), sv * (w - 0.04), belt + 0.2), (L * 0.2, 0.04, 0.2), paint, R=R, coll=coll)
            B.box(P(s(0.01), 0, belt + 0.2), (0.04, w - 0.05, 0.2), paint, R=R, coll=coll)
        # wheels
        for f in (wr, wf):
            for sv in (-1, 1):
                yc = sv * (w - 0.13)
                a = P(s(f), yc - 0.11, rw)
                b = P(s(f), yc + 0.11, rw)
                B.cyl(a, b, rw, self.tyre, seg=18, coll=coll)
                ya, yb = (yc + sv * 0.11, yc + sv * 0.115)
                B.cyl(P(s(f), ya, rw), P(s(f), yb, rw), rw * 0.62, self.rim, seg=14, coll=coll)
        # lamps, grille, plates, mirrors
        for sv in (-1, 1):
            B.box(P(s(0.997), sv * w * 0.62, belt * 0.8), (0.03, w * 0.17, 0.05), self.head, R=R, coll=coll)
            B.box(P(s(0.003), sv * w * 0.66, belt * 0.83), (0.03, w * 0.16, 0.05), self.tail, R=R, coll=coll)
            B.box(P(s(wf - 0.14), sv * (w + 0.08), belt + 0.08), (0.05, 0.08, 0.05), self.trim, R=R, coll=coll)
        B.box(P(s(1.0) + 0.0, 0, zb + 0.28), (0.03, w * 0.35, 0.09), self.trim, R=R, coll=coll)
        B.box(P(s(1.0) + 0.02, 0, zb + 0.13), (0.02, 0.26, 0.06), self.plate, R=R, coll=coll)
        B.box(P(s(0.0) - 0.02, 0, belt * 0.62), (0.02, 0.26, 0.06), self.plate, R=R, coll=coll)
        return L / 2, W / 2, H


class Forklift:
    """A counterbalance forklift (or a reach truck) from boxes and cylinders (``rng`` as ``Fleet``)."""

    def __init__(self, ctx, rng=None):
        self.rng = rng = ctx.rng if rng is None else rng
        self.ctx = ctx
        cols = [(0.85, 0.55, 0.02), (0.80, 0.30, 0.03), (0.55, 0.05, 0.04), (0.05, 0.20, 0.45), (0.9, 0.75, 0.05),
                (0.15, 0.35, 0.15)]
        lk = ctx.looks
        pm = ctx.lib.materials(cats_any={"painted metal"})
        base = lk.pick_material(prefer=["acg:PaintedMetal001", "acg:PaintedMetal004", "acg:PaintedMetal002"]) or \
            ctx.lib.pick(rng, pm)
        self.bodies = [lk.pbr(base, 1.0, tint=[min(1.0, 1.3 * c + 0.05) for c in rgb], bevel=0.03, dirt=0.4,
                              name="forklift_body") for rgb in cols]
        # the scan is itself coloured: grey it out so the tint decides
        for m in self.bodies:
            _desaturate(m)
        self.dark = plain((0.05, 0.05, 0.05), rough=0.5, name="fl_dark", bevel=0.01)
        self.steel = plain((0.25, 0.25, 0.26), rough=0.35, metal=1.0, name="fl_steel", bevel=0.01)
        self.tyre = plain((0.02, 0.02, 0.02), rough=0.8, name="fl_tyre")
        self.seat = plain((0.03, 0.03, 0.03), rough=0.6, name="fl_seat", bevel=0.03)

    def build(self, B, xy, yaw, coll="scatter", fork_h=None):
        rng = self.rng
        body = self.bodies[int(rng.integers(len(self.bodies)))]
        R = _rotz(yaw)
        c3 = np.array([xy[0], xy[1], 0.0])
        P = lambda u, v, z: R @ np.array([u, v, z]) + c3  # noqa: E731
        L, W = 2.3, 1.15
        # chassis (u = 0 at the centre, +u = forks)
        B.box(P(0.0, 0, 0.45), (0.85, W / 2, 0.3), body, R=R, coll=coll)
        B.box(P(-0.75, 0, 0.72), (0.3, W / 2, 0.42), self.steel if rng.random() < 0.3 else body, R=R, coll=coll)
        B.box(P(-0.2, 0, 0.9), (0.2, 0.25, 0.12), self.seat, R=R, coll=coll)
        B.box(P(-0.2, 0, 1.12), (0.05, 0.25, 0.2), self.seat, R=R, coll=coll)
        # overhead guard
        for (u, v) in ((-0.6, -0.5), (-0.6, 0.5), (0.45, -0.5), (0.45, 0.5)):
            B.box(P(u, v, 1.35), (0.03, 0.03, 0.6), self.dark, R=R, coll=coll)
        B.box(P(-0.07, 0, 2.08), (0.56, 0.54, 0.03), self.dark, R=R, coll=coll)
        for v in np.linspace(-0.4, 0.4, 5):
            B.box(P(-0.07, v, 2.12), (0.52, 0.02, 0.02), self.dark, R=R, coll=coll)
        # mast, carriage, forks
        mh = float(rng.uniform(2.1, 2.4))
        for v in (-0.35, 0.35):
            B.box(P(0.95, v, mh / 2 + 0.05), (0.05, 0.05, mh / 2), self.steel, R=R, coll=coll)
        fh = float(rng.choice([0.08, 0.08, 0.3, 0.9])) if fork_h is None else fork_h
        B.box(P(1.02, 0, fh + 0.35), (0.02, 0.45, 0.35), self.dark, R=R, coll=coll)
        for v in (-0.3, 0.3):
            B.box(P(1.6, v, fh + 0.02), (0.58, 0.05, 0.022), self.steel, R=R, coll=coll)
        # wheels
        for u, rw in ((0.6, 0.3), (-0.65, 0.24)):
            for sv in (-1, 1):
                B.cyl(P(u, sv * (W / 2 - 0.02), rw), P(u, sv * (W / 2 + 0.06), rw), rw, self.tyre, seg=14, coll=coll)
        return 1.95, W / 2 + 0.1, fh


def _desaturate(mat):
    """Put a Hue/Saturation node after the base colour texture: saturation 0,
    so a coloured paint scan becomes a grey the tint recolours."""
    nt = mat.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    # find the first image texture feeding colour (diff): walk back from Base Color to an image
    todo = [b.inputs["Base Color"].links[0].from_node] if b.inputs["Base Color"].is_linked else []
    seen = set()
    while todo:
        n = todo.pop(0)
        if n.name in seen:
            continue
        seen.add(n.name)
        if n.type == "TEX_IMAGE":
            for l in list(n.outputs["Color"].links):
                hs = nt.nodes.new("ShaderNodeHueSaturation")
                hs.inputs["Saturation"].default_value = 0.0
                hs.inputs["Value"].default_value = 1.6
                nt.links.new(n.outputs["Color"], hs.inputs["Color"])
                nt.links.new(hs.outputs["Color"], l.to_socket)
            continue
        for i in n.inputs:
            for l in i.links:
                todo.append(l.from_node)


# --- pallets and loads ---------------------------------------------------------------------------

class Loads:
    """Pallets and what stands on them: carton stacks, wrapped loads, drums,
    sacks, crates, bins. ``put(B, x, y, z, yaw, h_max)`` builds one."""

    def __init__(self, ctx, rng=None):
        self.ctx = ctx
        rng = rng or ctx.rng
        lk = ctx.looks
        wood = [k for k in ("acg:Wood026", "acg:Wood027", "acg:Wood048", "acg:Wood049", "acg:Planks012",
                            "acg:Planks020", "ph:plywood", "ph:wood_planks_dirt") if k in ctx.lib.items]
        self.pallet = [lk.pbr(ctx.lib.pick(rng, wood), 0.9, bevel=0.005, dirt=0.3, name="pallet_wood")]
        self.pallet.append(lk.pbr(ctx.lib.pick(rng, wood), 0.9, tint=(0.7, 0.66, 0.6), bevel=0.005, dirt=0.5,
                                  name="pallet_wood_old"))
        self.pallet_blue = plain((0.05, 0.18, 0.45), rough=0.5, name="chep_blue", bevel=0.005)
        cards = [k for k in ctx.lib.materials(cats_any={"cardboard"})]
        self.carton = [lk.pbr(k, 0.8, bevel=0.006, jitter=0.1, vary=0.1, name="carton") for k in cards] or \
            [plain((0.55, 0.42, 0.28), rough=0.8, name="carton")]
        self.carton += [lk.pbr(ctx.lib.pick(rng, cards), 0.8, tint=t, bevel=0.006, jitter=0.1, name="carton_t")
                        for t in ((1.25, 1.25, 1.25), (0.8, 0.75, 0.7))] if cards else []
        self.film = wrap_film()
        self.drum = [plain(c, rough=0.35, name="drum", bevel=0.01) for c in
                     ((0.05, 0.15, 0.45), (0.02, 0.02, 0.02), (0.5, 0.05, 0.03), (0.08, 0.3, 0.12))]
        self.sack = [plain(c, rough=0.85, name="sack", bevel=0.05) for c in
                     ((0.75, 0.72, 0.62), (0.85, 0.85, 0.83), (0.35, 0.4, 0.25))]
        self.bins = [plain(c, rough=0.45, name="bin", bevel=0.01) for c in
                     ((0.05, 0.2, 0.55), (0.6, 0.08, 0.05), (0.1, 0.1, 0.1), (0.1, 0.4, 0.15), (0.85, 0.6, 0.05))]
        self.ibc = plain((0.85, 0.87, 0.85), rough=0.3, name="ibc")
        self.cage = plain((0.6, 0.6, 0.62), rough=0.35, metal=1.0, name="cage")
        self.kinds = ["cartons", "cartons", "wrapped", "wrapped", "wrapped", "drums", "sacks", "bins", "ibc", "mixed"]

    def pallet_at(self, B, x, y, z, yaw, coll, pd=(1.2, 1.0)):
        rng = self.ctx.rng
        mat = self.pallet_blue if rng.random() < 0.15 else self.pallet[int(rng.random() < 0.4)]
        R = _rotz(yaw)
        c = np.array([x, y, z])
        P = lambda u, v, w: R @ np.array([u, v, w]) + c  # noqa: E731
        lx, ly = pd[0] / 2, pd[1] / 2
        B.box(P(0, 0, 0.133), (lx, ly, 0.011), mat, R=R, coll=coll)
        for v in (-ly + 0.05, 0, ly - 0.05):
            B.box(P(0, v, 0.072), (lx, 0.05, 0.05), mat, R=R, coll=coll)
        for u in (-lx + 0.05, 0, lx - 0.05):
            B.box(P(u, 0, 0.011), (0.05, ly, 0.011), mat, R=R, coll=coll)
        return 0.144

    def put(self, B, x, y, z, yaw, h_max, coll="structure", kind=None, pallet=True, pd=(1.2, 1.0)):
        """A pallet and its load at (x, y, z), load height <= h_max. Returns the top."""
        rng = self.ctx.rng
        kind = kind or self.kinds[int(rng.integers(len(self.kinds)))]
        z0 = z + (self.pallet_at(B, x, y, z, yaw, coll, pd) if pallet else 0.0)
        h = float(np.clip(rng.uniform(0.6, 1.5), 0.3, h_max - (z0 - z)))
        if h < 0.25:
            return z0
        R = _rotz(yaw)
        c = np.array([x, y, 0.0])
        P = lambda u, v, w: R @ np.array([u, v, w]) + c  # noqa: E731
        lx, ly = pd[0] / 2 - 0.02, pd[1] / 2 - 0.02
        if kind in ("cartons", "wrapped", "mixed"):
            mat = self.carton[int(rng.integers(len(self.carton)))]
            nu, nv = int(rng.choice([2, 2, 3])), int(rng.choice([2, 2, 3]))
            ch = float(rng.uniform(0.22, 0.42))
            nl = max(1, int(h // ch))
            du, dv = 2 * lx / nu, 2 * ly / nv
            if kind == "wrapped":
                hh = nl * ch
                B.box(P(0, 0, z0 + hh / 2), (lx + 0.01, ly + 0.01, hh / 2 + 0.005), self.film, R=R, coll=coll)
                return z0 + hh
            top = z0
            boxes = []
            for k in range(nl):
                if kind == "mixed" and k == nl - 1 and rng.random() < 0.6:
                    cnt = int(rng.integers(1, nu * nv))
                else:
                    cnt = nu * nv
                cells = rng.permutation(nu * nv)[:cnt]
                for cidx in cells:
                    i, j = divmod(int(cidx), nv)
                    jit = rng.normal(0, 0.008, 2)
                    boxes.append((P(-lx + du * (i + 0.5) + jit[0], -ly + dv * (j + 0.5) + jit[1], z0 + ch * (k + 0.5)),
                                  (du / 2 - 0.006, dv / 2 - 0.006, ch / 2 - 0.002), R, mat))
                top = z0 + ch * (k + 1)
            B.solid(boxes, coll=coll, rest=z0)
            return top
        if kind == "drums":
            mat = self.drum[int(rng.integers(len(self.drum)))]
            hd = min(0.88, h)
            for u in (-0.3, 0.3):
                for v in (-0.25, 0.25):
                    B.cyl(P(u, v, z0), P(u, v, z0 + hd), 0.28, mat, seg=16, coll=coll)
            return z0 + hd
        if kind == "sacks":
            mat = self.sack[int(rng.integers(len(self.sack)))]
            nl = max(1, int(h // 0.16))
            boxes = []
            for k in range(nl):
                rot = (k % 2) * math.pi / 2
                for i in (-1, 1):
                    jit = rng.normal(0, 0.02, 2)
                    Rk = _rotz(yaw + rot)
                    cc = np.array([x, y, 0.0]) + Rk @ np.array([0.0, i * ly / 2, 0.0])
                    boxes.append((cc + np.array([jit[0], jit[1], z0 + 0.08 + 0.16 * k]),
                                  (lx * 0.95, ly / 2 - 0.02, 0.075), Rk, mat))
            B.solid(boxes, coll=coll, rest=z0)
            return z0 + 0.16 * nl
        if kind == "bins":
            mat = self.bins[int(rng.integers(len(self.bins)))]
            nl = max(1, int(h // 0.3))
            B.solid([(P(u, v, z0 + 0.15 + 0.3 * k), (lx / 2 - 0.01, ly / 2 - 0.01, 0.145), R, mat)
                     for k in range(nl) for u in (-lx / 2, lx / 2) for v in (-ly / 2, ly / 2)], coll=coll, rest=z0)
            return z0 + 0.3 * nl
        if kind == "ibc":
            hd = min(1.15, h)
            B.box(P(0, 0, z0 + hd / 2), (lx - 0.03, ly - 0.03, hd / 2 - 0.02), self.ibc, R=R, coll=coll)
            for u in (-lx, lx):
                for v in (-ly, ly):
                    B.box(P(u, v, z0 + hd / 2), (0.015, 0.015, hd / 2), self.cage, R=R, coll=coll)
            for zz in np.linspace(z0 + 0.2, z0 + hd, 4):
                for sv in (-1, 1):
                    B.box(P(0, sv * ly, zz), (lx, 0.01, 0.01), self.cage, R=R, coll=coll)
                    B.box(P(sv * lx, 0, zz), (0.01, ly, 0.01), self.cage, R=R, coll=coll)
            return z0 + hd
        return z0


def cone(ctx, B, xy, mat, stripe, h=0.75, r=0.17, coll="scatter"):
    """A traffic cone: square base + tapered cone with a reflective band."""
    x, y = xy
    B.box((x, y, 0.02), (0.19, 0.19, 0.02), mat, coll=coll)
    seg = 16
    ang = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    for z0, z1, m in ((0.04, h * 0.45, mat), (h * 0.45, h * 0.62, stripe), (h * 0.62, h, mat)):
        r0 = r * (1 - (z0 - 0.04) / (h - 0.04)) + 0.02
        r1 = r * (1 - (z1 - 0.04) / (h - 0.04)) + 0.02
        V = np.concatenate([np.stack([x + r0 * np.cos(ang), y + r0 * np.sin(ang), np.full(seg, z0)], 1),
                            np.stack([x + r1 * np.cos(ang), y + r1 * np.sin(ang), np.full(seg, z1)], 1)])
        F = [(i, (i + 1) % seg, seg + (i + 1) % seg, seg + i) for i in range(seg)] + [tuple(range(seg, 2 * seg))]
        B.add(V, F, m, coll=coll, smooth=True)


def prim_index(name: str) -> int:
    """The patch primitive an ``obstacles`` object dresses (``prim<i>_wall<j>``, ``prim<i>_box``,
    ...: ``build_course.build_primitives`` pads ``i`` to two digits, so an arena's 100+ run
    to three), or -1."""
    m = re.match(r"prim(\d+)_", name)
    return int(m.group(1)) if m else -1


def prim_tops(spec) -> list[tuple[str, float, dict]]:
    """(kind, top z, primitive) for every patch primitive."""
    out = []
    for p in spec["primitives"]:
        k = p["kind"]
        if k == "wall":
            t = p["z1"]
        elif k == "box":
            R = _rpy(p.get("rpy", (0, 0, 0)))
            t = float(p["c"][2] + (np.abs(R) @ np.asarray(p["half"], float))[2])
        elif k == "cyl_v":
            t = p["z1"]
        elif k == "cyl_h":
            t = max(p["a"][2], p["b"][2]) + p["r"]
        elif k == "sphere":
            t = p["c"][2] + p["r"]
        else:
            t = 0.0
        out.append((k, float(t), p))
    return out


def _rpy(rpy):
    """A box primitive's rotation, in the convention its mesh is built with (``build_course.rpy_matrix``)."""
    return BC.rpy_matrix(rpy)


#: a hung primitive's bridle cables lean out from vertical by this much, degrees: enough to open
#: the over route (the anchor at the end corner decides the room at the lowest crossing, the lean
#: the room above it), little enough to keep them near the end's vertical (<= ~0.3 m out in the band)
BRIDLE_SPLAY = 15.0


def hang_rods(ctx, B, z_ceil, mat, r=0.012, gap=0.3, splay=BRIDLE_SPLAY) -> int:
    """Hang every floating patch primitive -- a level box (float_box, float_bar, a clutter
    box) or a sphere whose underside is above 0.3 m -- from the ceiling at ``z_ceil`` on a
    splayed bridle: two cables from the top edges of a box's end faces (its long axis), or
    from a sphere's sides 30 deg above its equator, leaning *out* ``splay`` deg from vertical.
    A box floating in a building reads as a hung sign or duct section instead of a levitating
    block. Nothing enters the prism over the primitive or the side passages beside it at its
    height, and the over route stays open at every width: over a 1.0 m box a 0.35 m body
    clears the cables within +-0.2 m of the middle at any height (rods 0.2 m in from the ends
    closed it under 1.1 m). Any hanger must cross the space beside the ends above the top;
    these take a sliver of it next to the end's vertical. Skipped where the top is within
    ``gap`` of the ceiling (closed up or touching) and on spheres under 0.3 m. The cables
    collide (``semantics_changed``). Returns how many primitives were hung."""
    ts = math.tan(math.radians(splay))
    n = 0
    for p in ctx.spec["primitives"]:
        if p["kind"] == "box":
            R = _rpy(p.get("rpy", (0, 0, 0)))
            if R[2, 2] < 0.999:
                continue
            h = np.asarray(p["half"], float)
            c = np.asarray(p["c"], float)
            bot, top = c[2] - h[2], c[2] + h[2]
            if bot < 0.3 or top > z_ceil - gap:
                continue
            ax = 1 if h[1] >= h[0] else 0
            e = np.zeros(3)
            e[ax] = 1.0
            ends = [(c + R @ (s * (h[ax] + r) * e) + np.array([0.0, 0.0, h[2]]), R @ (s * e)) for s in (-1, 1)]
        elif p["kind"] == "sphere":
            c, rr = np.asarray(p["c"], float), float(p["r"])
            if rr < 0.3 or c[2] - rr < 0.3 or c[2] + rr > z_ceil - gap:
                continue
            ends = [(c + np.array([0.0, s * rr * math.cos(math.pi / 6), rr * math.sin(math.pi / 6)]),
                     np.array([0.0, float(s), 0.0])) for s in (-1, 1)]
        else:
            continue
        for a, out in ends:
            b = a + (z_ceil - a[2]) * np.array([ts * out[0], ts * out[1], 1.0])
            B.cyl(a, b, r, mat, seg=6)
        n += 1
    return n


def clearance_many(foot, P) -> np.ndarray:
    """``Ctx.clearance`` of many points at once: distance from each xy in ``P`` (n, 2) to the
    nearest claimed footprint (discs and yawed rects, ``build_course.footprint_distance``)."""
    P = np.atleast_2d(np.asarray(P, float))[:, :2]
    best = np.full(len(P), 1e9)
    discs = [f for f in foot if f["type"] == "disc"]
    rects = [f for f in foot if f["type"] != "disc"]
    if discs:
        C = np.array([f["c"] for f in discs], float)
        r = np.array([f["r"] for f in discs], float)
        d = np.sqrt(((P[:, None, :] - C[None]) ** 2).sum(-1)) - r[None]
        best = np.minimum(best, d.min(1))
    if rects:
        C = np.array([f["c"] for f in rects], float)
        H = np.array([f["half"] for f in rects], float)
        R = np.array([f["R"] for f in rects], float)
        q = np.abs(np.einsum("nji,pnj->pni", R, P[:, None, :] - C[None])) - H[None]    # R^T (xy - c)
        d = np.sqrt((np.maximum(q, 0.0) ** 2).sum(-1)) + np.minimum(q.max(-1), 0.0)
        best = np.minimum(best, d.min(1))
    return best


def free_rect(ctx, c, half, yaw=0.0, margin=0.3, step=0.5) -> bool:
    """``Ctx.free_rect`` (the same samples, the same distances), vectorised over the claims: a
    garage's bays tried one by one against ~600 claims were most of an arena's build."""
    c = np.asarray(c, float)[:2]
    hx, hy = half[0], half[1]
    cy, sy = math.cos(yaw), math.sin(yaw)
    R = np.array([[cy, -sy], [sy, cy]])
    us = np.linspace(-hx, hx, max(2, int(2 * hx / step) + 1))
    vs = np.linspace(-hy, hy, max(2, int(2 * hy / step) + 1))
    UV = np.stack(np.meshgrid(us, vs, indexing="ij"), -1).reshape(-1, 2)
    return bool((clearance_many(ctx.foot, c + UV @ R.T) >= margin).all())


def seg_clear(ctx, a, b, margin, step=0.4) -> bool:
    """Every point of the xy segment a -> b is ``margin`` clear of claims."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    L = float(np.linalg.norm(b - a))
    n = max(2, int(L / step) + 1)
    return all(ctx.clearance(a + (b - a) * t) >= margin for t in np.linspace(0, 1, n))


def exposure_gain(ctx, sky: dict, d_ev: float) -> float:
    """Raise the scene exposure by ``d_ev`` over the HDRI's auto exposure (an
    interior is darker than the sky outside) and return the gain that makes a
    lamp of nominal power render the same under any HDRI: 2^-(final EV)."""
    ev = float(sky["exposure"]) + d_ev
    bpy.context.scene.view_settings.exposure = ev
    sky["exposure"] = round(ev, 3)
    sky["interior_ev"] = round(d_ev, 3)
    return 2.0 ** (-ev)


# --- the labeller's contract, kit side ------------------------------------------------------------

#: cloud points a scene aims for (the ~700k budget with a margin): kits turn it into m^2 of in-zone
#: surface with the live voxel -- a surface gives ~1.05 points per voxel face (measured on 21 builds)
PTS_TARGET = 600_000
#: in-zone surface of one procedural car, m^2 (``Fleet.car``; the garage notes the measured mean)
CAR_AREA = 45.0


def cloud_zmax() -> float:
    return float(BC.CLOUD_ZMAX)


def area_budget(pts: float = PTS_TARGET) -> float:
    """m^2 of in-zone surface under the cloud top that makes ``pts`` cloud points."""
    return float(pts) * float(BC.CLOUD_VOXEL) ** 2 / 1.05


def cloud_area(ctx, B=None) -> float:
    """m^2 of colliding surface the cloud will sample so far: every face of a collide-tagged
    mesh by the share of its bounding box in the zone and under the cloud top, colliding
    library pieces by their box, plus what ``B`` holds unflushed."""
    zmax = cloud_zmax()
    a = 0.0
    for c in ctx.colls.values():
        for o in c.objects:
            fb = _face_boxes(o)
            if fb is not None:
                ar, lo, hi = fb
                a += float((ar * _inside_share(lo, hi, ctx.zone, zmax)).sum())
    for p in ctx.placed:
        if p["collide"] and ctx.in_zone(p["xy"]):
            sx, sy, sz = p["size"]
            sz = min(sz, zmax - p["z"])
            if sz > 0:
                a += 1.4 * (sx * sy + sx * sz + sy * sz)
    return a + (B.area_in if B is not None else 0.0)


def _face_boxes(o):
    """(area, lo, hi) per face, world bounding boxes, of a collide-tagged mesh object, else None."""
    if o.type != "MESH" or not int(o.get("collide", 0)):
        return None
    me = o.data
    n = len(me.polygons)
    if not n:
        return None
    ar = np.empty(n)
    me.polygons.foreach_get("area", ar)
    V = np.empty(3 * len(me.vertices))
    me.vertices.foreach_get("co", V)
    M = np.array(o.matrix_world)
    V = V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    vi = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", vi)
    start = np.empty(n, np.int64)
    me.polygons.foreach_get("loop_start", start)
    P = V[vi]
    return ar, np.minimum.reduceat(P, start, axis=0), np.maximum.reduceat(P, start, axis=0)


#: the lowest a garage ceiling's services hang (``svc`` >= 3.8 m); kit surfaces from here up are "overhead"
OVERHEAD_Z = 3.6
#: share of the patch a kit ceiling inside the cloud must cover to cap the flight band
CAP_SHARE = 0.98


def raster_min(G, lo, hi, rect, cell, z_lo, zmax) -> None:
    """Lower ``G`` (cells of ``rect`` = (x0, y0, x1, y1), ``cell`` m, x by y) to the underside of
    every face box (lo, hi) that starts between ``z_lo`` and ``zmax`` over the cells it meets."""
    x0, y0, x1, y1 = rect
    lo, hi = np.atleast_2d(lo), np.atleast_2d(hi)
    m = (lo[:, 2] >= z_lo) & (lo[:, 2] <= zmax) & (hi[:, 0] >= x0) & (lo[:, 0] <= x1) & \
        (hi[:, 1] >= y0) & (lo[:, 1] <= y1)
    if not m.any():
        return
    nx, ny = G.shape
    i0 = np.clip(np.floor((lo[m, 0] - x0) / cell).astype(np.int64), 0, nx - 1)
    j0 = np.clip(np.floor((lo[m, 1] - y0) / cell).astype(np.int64), 0, ny - 1)
    # a face ending on a cell edge does not reach into the next cell
    i1 = np.maximum(i0, np.clip(np.ceil((hi[m, 0] - x0) / cell).astype(np.int64) - 1, 0, nx - 1))
    j1 = np.maximum(j0, np.clip(np.ceil((hi[m, 1] - y0) / cell).astype(np.int64) - 1, 0, ny - 1))
    for a, b, c, d, z in zip(i0, i1, j0, j1, lo[m, 2]):
        sl = G[a:b + 1, c:d + 1]
        np.minimum(sl, z, out=sl)


def overhead(ctx, rect=None, z_lo=OVERHEAD_Z, cell=0.25) -> np.ndarray:
    """What the cloud will hold over the patch: the lowest colliding *kit* surface between
    ``z_lo`` and the cloud top over each ``cell`` of ``rect`` (default the patch bounds),
    +inf where the cloud is open from ``z_lo`` to its top. Faces by their bounding boxes (the
    indoor ceilings are axis-aligned); patch primitives, the ground and library instances are
    left out. Call after ``Batch.flush``: it reads the scene, not the plan."""
    rect = tuple(ctx.bounds) if rect is None else tuple(rect)
    x0, y0, x1, y1 = rect
    G = np.full((max(1, int(math.ceil((x1 - x0) / cell))), max(1, int(math.ceil((y1 - y0) / cell)))), np.inf)
    zmax = cloud_zmax()
    for name, c in ctx.colls.items():
        if name in ("obstacles", "ground"):
            continue
        for o in c.objects:
            fb = _face_boxes(o)
            if fb is not None:
                raster_min(G, fb[1], fb[2], rect, cell, z_lo, zmax)
    return G


def ceiling_semantics(G, tops, slab, rect=None, cell=0.25, what="ceiling", parts="structure, services",
                      z_lo=OVERHEAD_Z):
    """How a kit's ceiling changes the tier-A family, read off what the cloud samples
    (``overhead`` raster ``G`` over ``rect``, the patch), not off the plan. ``capped`` when kit
    surfaces under the cloud top cover >= ``CAP_SHARE`` of the patch: no route over anything,
    and the room left over each primitive is the ceiling over its footprint less its top -- the
    note gives it (lowest-median of the cells over the footprint) for the primitive with the
    least, and how many primitives the ceiling closes over (median room < 0.1 m: they run up
    into it or were closed up to it). Otherwise the
    cloud is open up to its top: whatever ceiling structure is in it narrows the top of the
    band locally, and a primitive taller than the cloud top is cut at it, so a route over it
    is not in the cloud. ``slab`` = the ceiling's/roof's underside; ``tops`` = ``prim_tops``.
    Returns (notes, stats)."""
    zc = cloud_zmax()
    cov = np.isfinite(G)
    share = float(cov.mean()) if G.size else 0.0
    zs = G[cov]
    st = {"cover": round(share, 3), "capped": bool(share >= CAP_SHARE), "slab": round(float(slab), 3)}
    if zs.size:
        st["underside_p10"], st["underside_p90"] = (round(float(v), 3) for v in np.percentile(zs, [10, 90]))
        st["underside_min"] = round(float(zs.min()), 3)
    if share >= CAP_SHARE:
        p10, p90 = np.percentile(zs, [10, 90])
        s = (f"{what} in the cloud over the whole patch (underside {p10:.2f}-{p90:.2f} m, lowest {zs.min():.2f} m; "
             f"cloud top {zc:g} m): flight band capped, no over-the-top route")
        rooms = []
        if rect is not None:
            x0, y0 = rect[0], rect[1]
            nx, ny = G.shape
            for k, t, p in tops:
                if t >= zc or "kind" not in p:
                    continue
                lo, hi = _prim_xy_box(p)
                i0, j0 = max(0, int((lo[0] - x0) // cell)), max(0, int((lo[1] - y0) // cell))
                i1, j1 = min(nx, int(math.ceil((hi[0] - x0) / cell))), min(ny, int(math.ceil((hi[1] - y0) / cell)))
                over = G[i0:i1, j0:j1]
                if over.size:
                    rooms.append((float(np.median(over)) - t, float(over.min()) - t, k, t))
        closed = [r for r in rooms if r[0] < 0.1]
        free = [r for r in rooms if r[0] >= 0.1]
        if free:
            med, low, k, t = min(free)
            span = f"{med:.2f}" if med - low < 0.02 else f"{low:.2f}-{med:.2f}"
            s += f"; least room over a primitive {span} m ({k} top {t:.2f} m; lowest-median ceiling over it)"
        if closed:
            s += f"; closed over {len(closed)} primitive(s) (they reach the ceiling)"
        st["room_min"] = round(min(free)[1], 3) if free else None
        st["room_median"] = round(min(free)[0], 3) if free else None
    else:
        s = (f"{what} {slab:.2f} m, above the cloud top ({zc:g} m): the cloud is open up to its top over "
             f"{1 - share:.0%} of the patch")
        if zs.size:
            s += (f"; {parts} above {z_lo:g} m in the cloud over {share:.0%} (lowest {zs.min():.2f} m) narrow "
                  f"the top of the band there, not cap it")
        tall = sum(1 for _, t, _ in tops if t > zc)
        if tall:
            s += f"; {tall} primitive(s) taller than the cloud top are cut at it (a route over them is not in the cloud)"
    return [s], st


def passage_m(ctx) -> float:
    """Metres kept free round a family's passages: the drone's 0.7 m and margins, more at L1."""
    return max(1.2, 0.5 * float(ctx.knobs.get("keepout", 2.5)))


def _circle(xy):
    """Least-squares circle (centre, radius) through points (Kasa)."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    cx, cy, c = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0] * np.array([0.5, 0.5, 1.0])
    return np.array([cx, cy]), float(math.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def _group_kind(name: str) -> str:
    n = name.rstrip("0123456789").rstrip("_")
    return {"forest": "stand", "stand": "stand", "stand_sparse": "stand", "stand_medium": "stand",
            "stand_dense": "stand", "ring": "ring", "room": "room", "pillar": "pillar", "wall_offset": "wall_end",
            "wall_sym": "wall_end", "float_box": "box_side", "float_bar": "box_side", "slab": "box_side",
            "bars_lintel": "lintel", "bars": "lintel"}.get(n, "")


def claim_passages(ctx) -> list[str]:
    """Claim the free space a tier-A family is *about*, on top of ``dress_primitives``' solids
    and slot approaches, so no kit column, car, rack or prop lands in it: the passage past each
    end of a wall_offset/wall_sym wall and each side of a floating box/bar or slab, round a
    pillar, a ring with its inside and gap, a room's inside, the approaches under a lintel, a
    stand's whole grove. ``passage_m`` wide. Corridor and arena primitives are grouped by their
    ``region`` tag (``ring_03``, ``forest_13``, ...) and treated the same. Returns notes."""
    spec = ctx.spec
    P = passage_m(ctx)
    groups: dict = {}
    for p in spec["primitives"]:
        g = spec["family"] if spec["family"] not in ("corridor", "arena") else p.get("region", "")
        groups.setdefault(g, []).append(p)
    notes = []
    for g, ps in groups.items():
        kind = _group_kind(g)
        n = 0
        if kind == "wall_end":
            for p in ps:
                if p["kind"] == "wall":
                    for e in (p["p0"], p["p1"]):
                        ctx.claim_disc(e, P + p["t"] / 2, kind="passage")
                        n += 1
        elif kind == "pillar":
            for p in ps:
                if p["kind"] == "cyl_v":
                    ctx.claim_disc(p["c"], p["r"] + P, kind="passage")
                    n += 1
        elif kind == "box_side":
            for p in ps:
                if p["kind"] == "box":
                    R = _rpy(p.get("rpy", (0, 0, 0)))
                    for s in (-1, 1):
                        e = np.asarray(p["c"], float) + R @ np.array([0.0, s * p["half"][1], 0.0])
                        ctx.claim_disc(e[:2], P, kind="passage")
                        n += 1
        elif kind == "lintel":
            for p in ps:
                if p["kind"] == "box" and _rpy(p.get("rpy", (0, 0, 0)))[2, 2] > 0.999:
                    ctx.claim_rect(p["c"], (p["half"][0] + P, p["half"][1]), p["rpy"][2], kind="passage")
                    n += 1
        elif kind == "ring":
            xy = np.array([p["c"] for p in ps if p["kind"] == "cyl_v"], float)
            if len(xy) >= 3:
                c, r = _circle(xy)
                ctx.claim_disc(c, r + P, kind="passage")
                n += 1
        elif kind in ("stand", "room") and ps:
            lo = np.min([_prim_xy_box(p)[0] for p in ps], 0)
            hi = np.max([_prim_xy_box(p)[1] for p in ps], 0)
            pad = 0.3 if kind == "stand" else 0.0
            ctx.claim_rect((lo + hi) / 2, (hi - lo) / 2 + pad, 0.0, kind="passage")
            n += 1
        if n:                                           # only what was actually kept free
            notes.append(f"{g}:{kind}")
    return notes


def _prim_xy_box(p):
    """(lo, hi) xy of one primitive."""
    k = p["kind"]
    if k == "wall":
        q = np.array([p["p0"], p["p1"]], float)[:, :2]
        return q.min(0) - p["t"] / 2, q.max(0) + p["t"] / 2
    if k == "box":
        e = (np.abs(_rpy(p.get("rpy", (0, 0, 0)))) @ np.asarray(p["half"], float))[:2]
        return np.asarray(p["c"], float)[:2] - e, np.asarray(p["c"], float)[:2] + e
    if k == "cyl_v":
        return np.asarray(p["c"], float) - p["r"], np.asarray(p["c"], float) + p["r"]
    if k == "cyl_h":
        q = np.array([p["a"], p["b"]], float)[:, :2]
        return q.min(0) - p["r"], q.max(0) + p["r"]
    c = np.asarray(p["c"], float)[:2]
    return c - p["r"], c + p["r"]
