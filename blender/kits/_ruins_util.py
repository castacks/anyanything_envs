"""Private helpers for the ``ruins`` and ``coast`` kits (runs inside Blender).

Geometry that the base ``Ctx`` does not have, all built through ``ctx.mesh``
(so the zone decides what collides, exactly as for ``ctx.box``):

* ``Parts`` -- many boxes / frusta / arbitrary quads gathered into ONE mesh
  (a broken wall of 40 stones is one object, not 40), with full 3-D
  rotation per box (tumbled rubble);
* ``arch_block`` -- a masonry arch: semicircular opening in a rectangular
  block, the piers below it are ordinary boxes;
* ``frustum`` -- a tapered cylinder (tree limbs, lighthouse, columns with
  entasis);
* ``boat_hull`` -- a clinker-ish rowing/fishing boat hull shell;
* ``heightfield_ring`` -- low rolling hills far out, render-only.

Free space for things that are not a disc round their base (``Ctx.free``
checks one point): ``clearance_many`` (vectorised ``Ctx.clearance``),
``seg_free`` (a beam / rope / line along its length), ``parts_free`` (every
vertex of a ``Parts``, e.g. a dead tree's limbs), ``fit_scale`` (shrink a
plant into the free space round it), ``clear_of_ends`` /
``rect_clear_of_ends`` (dressing on a primitive stays out of the start /
goal keep-out). ``claim_openings`` claims the openings a family implies that
the shared ``build_primitives`` leaves unclaimed (a ring's interior and gap,
the passages past a short wall's ends, the approach under a floating box),
so every later placement keeps out of them. ``cloud_guard`` keeps the
obstacle cloud under budget from an estimate made on the evaluated scene,
cropped as the cloud is.

Materials that ``looks.py`` does not have, built on its scans:

* ``zoned_ground`` -- the floor as several scans mixed by noise AND by
  world-space region masks (a paved courtyard fraying into grass, a beach
  going from dry sand through wet sand to the waterline);
* ``water`` -- a sea surface: Fresnel-reflective, rough-noise waves,
  shallow-to-deep colour with foam along a (wavy) shoreline.
"""

from __future__ import annotations

import math

import bpy
import numpy as np


#: the highest point the drone's body reaches under recipe v7 (flight_z 1.75 +
#: climb-max 2.5 + body radius 0.35): dressing added to a patch primitive
#: wholly above this leaves the labels' route topology alone
V7_REACH_TOP = 4.6
V7_BODY_R = 0.35


# --- rotations -------------------------------------------------------------------------------

def rot(yaw=0.0, pitch=0.0, roll=0.0) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


_BOX_F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


class Parts:
    """Collects geometry, emits one mesh through ``ctx.mesh``."""

    def __init__(self):
        self.V: list = []
        self.F: list = []

    def __len__(self):
        return len(self.F)

    def box(self, c, half, R=None, yaw: float = 0.0):
        R = rot(yaw) if R is None else np.asarray(R, float)
        c, h = np.asarray(c, float), np.asarray(half, float)
        n0 = len(self.V)
        for i in (-1, 1):
            for j in (-1, 1):
                for k in (-1, 1):
                    self.V.append(c + R @ (h * np.array([i, j, k])))
        # vertex index = 4*(i>0) + 2*(j>0) + (k>0)
        self.F += [tuple(n0 + v for v in f) for f in _BOX_F]

    def frustum(self, a, b, r0, r1, seg: int = 10, caps: bool = True, twist: float = 0.0):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ax = b - a
        L = float(np.linalg.norm(ax))
        if L < 1e-6:
            return
        z = ax / L
        x = np.cross(z, [0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.cross(z, [1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        n0 = len(self.V)
        for (p, r) in ((a, r0), (b, r1)):
            for s in range(seg):
                t = 2 * math.pi * s / seg + twist
                self.V.append(p + r * (math.cos(t) * x + math.sin(t) * y))
        for s in range(seg):
            s2 = (s + 1) % seg
            self.F.append((n0 + s, n0 + s2, n0 + seg + s2, n0 + seg + s))
        if caps:
            self.F.append(tuple(n0 + s for s in reversed(range(seg))))
            self.F.append(tuple(n0 + seg + s for s in range(seg)))

    def quads(self, V, F):
        n0 = len(self.V)
        self.V += [np.asarray(v, float) for v in V]
        self.F += [tuple(n0 + i for i in f) for f in F]

    def emit(self, ctx, mat, name="parts", coll="structure", collide=None, roof=False, smooth=False):
        if not self.F:
            return None
        return ctx.mesh(np.asarray(self.V), self.F, mat, coll=coll, collide=collide, name=name, roof=roof,
                        smooth=smooth)


# --- masonry ---------------------------------------------------------------------------------

def arch_block(parts: Parts, c, yaw, ri, ro, zs, depth, top: float = 0.35, seg: int = 14,
               jag=None):
    """A block with a semicircular opening: the opening is ``2 ri`` wide and
    springs at height ``zs``; the block is ``2 ro`` wide, reaches
    ``zs + ro + top`` and is ``depth`` thick. ``c`` = centre of the opening
    at ground level, ``yaw`` = direction along the wall. ``jag`` (optional
    list of per-segment height losses) breaks the top."""
    c = np.asarray(c, float)[:2]
    d = np.array([math.cos(yaw), math.sin(yaw)])
    n = np.array([-d[1], d[0]])
    H = ro + top
    # outer points: ray from (0, 0) (arch centre, local u/v) at angle th hits the rectangle
    th = np.linspace(0.0, math.pi, seg + 1)
    corner = math.atan2(H, ro)
    th = np.unique(np.concatenate([th, [corner, math.pi - corner]]))
    inner, outer = [], []
    for t in th:
        cu, sv = math.cos(t), math.sin(t)
        inner.append((ri * cu, ri * sv))
        # hit the rectangle |u| <= ro, 0 <= v <= H
        su = ro / abs(cu) if abs(cu) > 1e-9 else 1e9
        sw = H / sv if sv > 1e-9 else 1e9
        s = min(su, sw)
        outer.append((s * cu, s * sv))
    if jag is not None:
        m = len(outer)
        for i in range(m):
            u, v = outer[i]
            if v >= H - 1e-6:
                v = max(v - jag[i % len(jag)], inner[i][1] + 0.25)
                outer[i] = (u, v)

    def P(uv, side):
        u, v = uv
        xy = c + d * u + n * side * depth / 2
        return (xy[0], xy[1], zs + v)

    V, F = [], []
    m = len(th)
    for side in (-1, 1):
        for i in range(m):
            V.append(P(inner[i], side))
        for i in range(m):
            V.append(P(outer[i], side))

    def idx(side, ring, i):  # side 0/1, ring 0 inner / 1 outer
        return side * 2 * m + ring * m + i

    for i in range(m - 1):
        # front (side -1) and back faces
        F.append((idx(0, 0, i), idx(0, 1, i), idx(0, 1, i + 1), idx(0, 0, i + 1)))
        F.append((idx(1, 0, i), idx(1, 0, i + 1), idx(1, 1, i + 1), idx(1, 1, i)))
        # intrados and extrados
        F.append((idx(0, 0, i), idx(0, 0, i + 1), idx(1, 0, i + 1), idx(1, 0, i)))
        F.append((idx(0, 1, i), idx(1, 1, i), idx(1, 1, i + 1), idx(0, 1, i + 1)))
    # springing faces (facing down at both ends: the second winds the other way round)
    F.append((idx(0, 0, 0), idx(1, 0, 0), idx(1, 1, 0), idx(0, 1, 0)))
    F.append((idx(0, 1, m - 1), idx(1, 1, m - 1), idx(1, 0, m - 1), idx(0, 0, m - 1)))
    parts.quads(V, F)


def broken_profile(rng, L, h_hi, h_lo, course=0.3, chunk=(0.35, 0.9), ends_high=True):
    """Chunk widths and heights of a ruined wall's stepped top: masonry courses
    lost unevenly, usually highest at the ends (corners survive)."""
    ws = []
    u = 0.0
    while u < L - 1e-6:
        w = float(min(rng.uniform(*chunk), L - u))
        if L - u - w < 0.2:
            w = L - u
        ws.append(w)
        u += w
    k = len(ws)
    us = np.cumsum([0.0] + ws[:-1]) + np.asarray(ws) / 2
    t = us / max(L, 1e-6)
    # envelope: a few random bumps plus high ends
    env = np.zeros(k)
    for _ in range(int(rng.integers(1, 4))):
        mu, sg = rng.uniform(0, 1), rng.uniform(0.08, 0.3)
        env += rng.uniform(0.3, 1.0) * np.exp(-0.5 * ((t - mu) / sg) ** 2)
    if ends_high:
        env += 0.9 * np.exp(-0.5 * (t / 0.08) ** 2) + 0.9 * np.exp(-0.5 * ((1 - t) / 0.08) ** 2)
    env = env / max(env.max(), 1e-6)
    h = h_lo + (h_hi - h_lo) * env + rng.normal(0, 0.12 * (h_hi - h_lo), k)
    h = np.clip(np.round(h / course) * course, course, h_hi)
    return ws, h


def broken_wall(ctx, parts: Parts, p0, p1, thick, h_hi, h_lo, course=0.3, rng=None, ends_high=True,
                stones=True):
    """Add a ruined wall p0 -> p1 to ``parts``: stepped chunks, a few stones
    proud of the face and a few missing. Returns (chunk widths, heights)."""
    rng = rng or ctx.rng
    p0, p1 = np.asarray(p0, float)[:2], np.asarray(p1, float)[:2]
    L = float(np.linalg.norm(p1 - p0))
    d = (p1 - p0) / max(L, 1e-9)
    n = np.array([-d[1], d[0]])
    yaw = math.atan2(d[1], d[0])
    ws, hs = broken_profile(rng, L, h_hi, h_lo, course, ends_high=ends_high)
    u = 0.0
    for w, h in zip(ws, hs):
        c = p0 + d * (u + w / 2)
        parts.box((c[0], c[1], h / 2), (w / 2, thick / 2, h / 2), yaw=yaw)
        if stones and h > course * 1.5 and rng.random() < 0.55:
            # a loose top stone, slightly askew
            sw = min(w * 0.8, rng.uniform(0.25, 0.5))
            sc = c + d * rng.uniform(-w / 2 + sw / 2, w / 2 - sw / 2) + n * rng.normal(0, 0.04)
            sh = rng.uniform(0.12, 0.2)
            parts.box((sc[0], sc[1], h + sh / 2 - 0.02), (sw / 2, thick / 2 * rng.uniform(0.6, 0.95), sh / 2),
                      R=rot(yaw + rng.normal(0, 0.06), rng.normal(0, 0.05), rng.normal(0, 0.05)))
        u += w
    return ws, hs


def mound(parts: Parts, c, R, h, rng, n_th=14, n_r=4, squash=(1.0, 1.0)):
    """An irregular low dome (a heap of earth/rubble) of radius R, height h."""
    c = np.asarray(c, float)[:2]
    ph = rng.uniform(0, 6.3, 3)
    V = [(c[0], c[1], h)]
    for i in range(1, n_r + 1):
        rr = i / n_r
        for j in range(n_th):
            t = 2 * math.pi * j / n_th
            wob = 1 + 0.18 * math.sin(3 * t + ph[0]) + 0.1 * math.sin(5 * t + ph[1]) + rng.normal(0, 0.05)
            z = h * (1 - rr ** 2) * (1 + rng.normal(0, 0.08)) if i < n_r else -0.02
            V.append((c[0] + R * rr * wob * squash[0] * math.cos(t), c[1] + R * rr * wob * squash[1] * math.sin(t),
                      max(-0.02, z)))
    F = []
    for j in range(n_th):
        F.append((0, 1 + j, 1 + (j + 1) % n_th))
    for i in range(n_r - 1):
        for j in range(n_th):
            a = 1 + i * n_th + j
            b = 1 + i * n_th + (j + 1) % n_th
            F.append((a, a + n_th, b + n_th, b))
    parts.quads(V, F)


def rubble(ctx, parts: Parts, c, R, n, size=(0.15, 0.45), rng=None, height=0.6, base=None):
    """A heap of tumbled blocks round ``c`` (radius R), resting on a low
    mound (``base`` Parts gets the mound; default: same ``parts``)."""
    rng = rng or ctx.rng
    c = np.asarray(c, float)[:2]
    mound(base if base is not None else parts, c, R * 1.05, height, rng)
    for _ in range(n):
        r = R * math.sqrt(rng.random()) * 0.95
        a = rng.uniform(0, 2 * math.pi)
        xy = c + r * np.array([math.cos(a), math.sin(a)])
        s = rng.uniform(*size)
        half = np.array([s, s * rng.uniform(0.5, 0.9), s * rng.uniform(0.35, 0.7)]) / 2
        zs = height * (1 - (r / R) ** 2)
        parts.box((xy[0], xy[1], zs + half[2] * 0.1), half,
                  R=rot(rng.uniform(0, 6.3), rng.normal(0, 0.3), rng.normal(0, 0.3)))


# --- boats -----------------------------------------------------------------------------------

def boat_hull(L, B, D, rng, n_st=14, n_sec=9, transom=0.62, sheer=0.18, rocker=0.08, p=1.7):
    """Hull shell in local coordinates (x along the boat, stern at -L/2,
    keel at z = 0): verts, faces. Open at the top."""
    V, F = [], []
    ts = np.linspace(0, 1, n_st)
    for t in ts:
        x = -L / 2 + t * L
        s = 2 * t - 1
        # half beam: blunt stern (transom), sharp bow
        if t < 0.5:
            b = B / 2 * (transom + (1 - transom) * math.sin(math.pi * t) ** 0.6)
        else:
            b = B / 2 * max(0.0, math.sin(math.pi * t)) ** 0.75
        zk = rocker * (s ** 2) + (0.25 * D * max(0.0, s) ** 3)
        zg = D + sheer * (s ** 2) * (1.4 if s > 0 else 0.6)
        for k in range(n_sec):
            q = 2 * k / (n_sec - 1) - 1
            y = b * q
            z = zk + (zg - zk) * abs(q) ** p
            V.append((x, y, z))
    for i in range(n_st - 1):
        for k in range(n_sec - 1):
            a = i * n_sec + k
            F.append((a, a + 1, a + n_sec + 1, a + n_sec))
    F.append(tuple(range(n_sec - 1, -1, -1)))           # transom
    return np.asarray(V), F


# --- tapered limbs ---------------------------------------------------------------------------

def dead_tree(parts: Parts, base, height, r0, rng, depth: int = 3, min_z: float = 0.0):
    """A leafless tree: tapered trunk and recursive limbs, into ``parts``."""
    base = np.asarray(base, float)

    def limb(a, d, L, r, lev):
        b = a + d * L
        if b[2] < min_z + 0.2:
            b[2] = min_z + 0.2
        r1 = r * rng.uniform(0.55, 0.72)
        parts.frustum(a, b, r, r1, seg=max(5, 9 - 2 * (depth - lev)), caps=lev == 0)
        if lev == 0 or r1 < 0.02:
            return
        nk = int(rng.integers(2, 4))
        for _ in range(nk):
            az = rng.uniform(0, 2 * math.pi)
            el = rng.uniform(0.2, 0.75)
            nd = d * rng.uniform(0.4, 0.8) + np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el),
                                                        math.sin(el)])
            nd /= np.linalg.norm(nd)
            limb(b - d * rng.uniform(0, 0.25) * L, nd, L * rng.uniform(0.5, 0.75), r1 * rng.uniform(0.6, 0.9),
                 lev - 1)

    d0 = np.array([rng.normal(0, 0.08), rng.normal(0, 0.08), 1.0])
    d0 /= np.linalg.norm(d0)
    limb(base, d0, height * 0.5, r0, depth)


# --- far hills -------------------------------------------------------------------------------

def heightfield_ring(ctx, centre, r_in, r_out, h_max, mat, rng, arc=(0.0, 2 * math.pi), n_th=96, n_r=10,
                     name="hills"):
    """Rolling hills between radii ``r_in`` and ``r_out`` over an azimuth
    ``arc`` (render-only: everything is far outside the zone)."""
    c = np.asarray(centre, float)[:2]
    th = np.linspace(arc[0], arc[1], n_th)
    rs = np.linspace(r_in, r_out, n_r)
    ph = rng.uniform(0, 2 * math.pi, 4)
    fr = [2, 3, 5, 9]
    amp = rng.uniform(0.3, 1.0, 4)
    # a partial arc fades to the plain at both ends (cut off, its end reads as a mesa's edge)
    span = arc[1] - arc[0]
    fade = 0.18 * span if span < 2 * math.pi - 1e-3 else 0.0

    def ss(x):
        x = min(1.0, max(0.0, x))
        return x * x * (3 - 2 * x)

    V = []
    for r in rs:
        u = (r - r_in) / (r_out - r_in)
        for t in th:
            prof = sum(a * (0.5 + 0.5 * math.sin(f * t + p + 0.004 * r)) for a, f, p in zip(amp, fr, ph)) / sum(amp)
            # a ridge, not a plateau: up to a crest whose distance wanders with azimuth, then down
            # to a lower far rim (rising all the way out, the rim read as a flat-topped mesa)
            crest = 0.45 + 0.2 * math.sin(2 * t + ph[0])
            rise = ss(u / crest) if u < crest else 1.0 - 0.6 * ss((u - crest) / (1.0 - crest))
            end = min(1.0, (t - arc[0]) / fade, (arc[1] - t) / fade) if fade else 1.0
            h = h_max * rise * ss(end) * (0.25 + 0.75 * prof)
            # (flat parts sink under the ground plane: at z = 0 they z-fought with it, dark bands)
            V.append((c[0] + r * math.cos(t), c[1] + r * math.sin(t), h - 0.5))
    F = []
    for i in range(n_r - 1):
        for j in range(n_th - 1):
            a = i * n_th + j
            F.append((a, a + 1, a + n_th + 1, a + n_th))
    return ctx.mesh(V, F, mat, coll="backdrop", collide=False, name=name, smooth=True)


# --- shader helpers --------------------------------------------------------------------------

class NK:
    """Tiny node-building helper for one node tree."""

    def __init__(self, nt):
        self.nt, self.N, self.L = nt, nt.nodes, nt.links
        self._tc = None

    def pos(self):
        if self._tc is None:
            self._tc = self.N.new("ShaderNodeTexCoord")
        return self._tc.outputs["Object"]

    def sep(self, v):
        s = self.N.new("ShaderNodeSeparateXYZ")
        self.L.new(v, s.inputs["Vector"])
        return s.outputs

    def math(self, op, a, b=None, clamp=False):
        m = self.N.new("ShaderNodeMath")
        m.operation = op
        m.use_clamp = clamp
        for i, x in enumerate((a, b)):
            if x is None:
                continue
            if isinstance(x, (int, float)):
                m.inputs[i].default_value = float(x)
            else:
                self.L.new(x, m.inputs[i])
        return m.outputs[0]

    def mrange(self, v, fmin, fmax, tmin=0.0, tmax=1.0, smooth=True):
        m = self.N.new("ShaderNodeMapRange")
        if smooth:
            m.interpolation_type = "SMOOTHSTEP"
        m.inputs["From Min"].default_value, m.inputs["From Max"].default_value = fmin, fmax
        m.inputs["To Min"].default_value, m.inputs["To Max"].default_value = tmin, tmax
        self.L.new(v, m.inputs["Value"])
        return m.outputs["Result"]

    def noise(self, scale, detail=4.0, rough=0.55, offset=(0, 0, 0), stretch=(1, 1, 1), vec=None):
        mp = self.N.new("ShaderNodeMapping")
        mp.inputs["Location"].default_value = tuple(float(v) for v in offset)
        mp.inputs["Scale"].default_value = tuple(float(v) for v in stretch)
        self.L.new(vec if vec is not None else self.pos(), mp.inputs["Vector"])
        nz = self.N.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = scale
        nz.inputs["Detail"].default_value = detail
        nz.inputs["Roughness"].default_value = rough
        self.L.new(mp.outputs["Vector"], nz.inputs["Vector"])
        return nz.outputs["Fac"]

    def mixsh(self, fac, a, b):
        m = self.N.new("ShaderNodeMixShader")
        if isinstance(fac, (int, float)):
            m.inputs["Fac"].default_value = fac
        else:
            self.L.new(fac, m.inputs["Fac"])
        self.L.new(a, m.inputs[1])
        self.L.new(b, m.inputs[2])
        return m.outputs["Shader"]

    def mixrgb(self, fac, a, b):
        m = self.N.new("ShaderNodeMix")
        m.data_type = "RGBA"
        if isinstance(fac, (int, float)):
            m.inputs["Factor"].default_value = fac
        else:
            self.L.new(fac, m.inputs["Factor"])
        for i, x in ((6, a), (7, b)):
            if isinstance(x, (tuple, list)):
                m.inputs[i].default_value = (*x[:3], 1.0)
            else:
                self.L.new(x, m.inputs[i])
        return m.outputs[2]

    def shore_dist(self, shore):
        """Signed distance to a wavy shoreline, positive on the sea side.
        ``shore`` = dict(n=(nx, ny), off, A, k, phi)."""
        nx, ny = shore["n"]
        tx, ty = -ny, nx
        s = self.sep(self.pos())
        pn = self.math("ADD", self.math("MULTIPLY", s[0], nx), self.math("MULTIPLY", s[1], ny))
        pt = self.math("ADD", self.math("MULTIPLY", s[0], tx), self.math("MULTIPLY", s[1], ty))
        wav = self.math("MULTIPLY", self.math("SINE", self.math("ADD", self.math("MULTIPLY", pt, shore["k"]),
                                                                 shore["phi"])), shore["A"])
        wav2 = self.math("MULTIPLY", self.math("SINE", self.math("ADD", self.math("MULTIPLY", pt, shore["k"] * 2.7),
                                                                  shore["phi"] * 1.9)), shore["A"] * 0.3)
        return self.math("SUBTRACT", pn, self.math("ADD", self.math("ADD", wav, wav2), shore["off"]))

    def rect_dist(self, c, half):
        """Approximate signed distance to an axis-aligned rectangle (negative inside)."""
        s = self.sep(self.pos())
        dx = self.math("SUBTRACT", self.math("ABSOLUTE", self.math("SUBTRACT", s[0], c[0])), half[0])
        dy = self.math("SUBTRACT", self.math("ABSOLUTE", self.math("SUBTRACT", s[1], c[1])), half[1])
        return self.math("MAXIMUM", dx, dy)


def shore_y(shore, t):
    """Shoreline offset along its normal at tangent coordinate ``t`` (matches NK.shore_dist)."""
    return (shore["off"] + shore["A"] * math.sin(shore["k"] * t + shore["phi"])
            + shore["A"] * 0.3 * math.sin(shore["k"] * 2.7 * t + shore["phi"] * 1.9))


def shore_signed(shore, xy):
    nx, ny = shore["n"]
    tx, ty = -ny, nx
    pn = xy[0] * nx + xy[1] * ny
    pt = xy[0] * tx + xy[1] * ty
    return pn - shore_y(shore, pt)


def zoned_ground(looks, base, layers=(), puddles: float = 0.0, blend_scale: float = 9.0, name="ground_zoned",
                 rng=None):
    """Ground material: ``base`` = list of (scan key, tile_m or None) mixed by
    noise; ``layers`` = list of dicts ``{"key", "tile", "mask": fn(nk) ->
    socket 0..1}`` painted over it in order. ``puddles`` as in
    ``looks.ground_material``."""
    rng = rng if rng is not None else looks.rng
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nk = NK(nt)
    N = nt.nodes
    N.remove(N["Principled BSDF"])
    out = N["Material Output"]

    def sh(key, tile, tint=None):
        return looks.pbr(key, tile, proj="flat", vary=0.2, jitter=0.0, into=nt, tint=tint).outputs["BSDF"]

    cur = sh(*base[0])
    for i, (k, t) in enumerate(base[1:3], 1):
        f = nk.noise(1.0 / (blend_scale * (0.7 + 0.6 * i)), detail=6.0, rough=0.65,
                     offset=rng.uniform(-50, 50, 3))
        c0 = 0.5 + 0.06 * (i - 1)
        cur = nk.mixsh(nk.mrange(f, c0 - 0.05, c0 + 0.05), cur, sh(k, t))
    for lay in layers:
        cur = nk.mixsh(lay["mask"](nk), cur, sh(lay["key"], lay.get("tile"), lay.get("tint")))
    if puddles > 0:
        wet = N.new("ShaderNodeBsdfPrincipled")
        wet.inputs["Base Color"].default_value = (0.06, 0.055, 0.05, 1)
        wet.inputs["Roughness"].default_value = 0.06
        f = nk.noise(0.5, detail=5.0, offset=rng.uniform(-50, 50, 3))
        lo = 0.70 - 0.12 * puddles
        cur = nk.mixsh(nk.mrange(f, lo, lo + 0.03), cur, wet.outputs["BSDF"])
    nt.links.new(cur, out.inputs["Surface"])
    return m


def water(name="sea", shore=None, deep=(0.012, 0.045, 0.06), shallow=(0.06, 0.16, 0.14), rough=0.035,
          wave=1.0, foam=0.7, rng=None, murky=0.0, sand=(0.36, 0.3, 0.22)):
    """Sea surface: dark, glossy, noise-wave normals; shallow tint and foam
    near ``shore`` (dict as ``NK.shore_dist``)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nk = NK(nt)
    b = nt.nodes["Principled BSDF"]
    b.inputs["IOR"].default_value = 1.333
    deep = tuple(np.asarray(deep) * (1 - murky) + np.array([0.05, 0.05, 0.035]) * murky)
    shallow = tuple(np.asarray(shallow) * (1 - murky) + np.array([0.11, 0.1, 0.07]) * murky)
    col = None
    rgh = None
    if shore is not None:
        d = nk.shore_dist(shore)
        depth_f = nk.mrange(d, 0.0, 14.0)
        col = nk.mixrgb(depth_f, shallow, deep)
        # the first metre or two: the sand shows through
        col = nk.mixrgb(nk.mrange(d, 2.5, 0.0), col, sand)
        if foam > 0:
            fz = nk.noise(2.2, detail=8, rough=0.7, offset=rng.uniform(-40, 40, 3))
            band = nk.mrange(d, 1.6, 0.1)                      # 1 at the waterline, 0 beyond ~1.5 m
            f = nk.math("MULTIPLY", band, nk.mrange(fz, 0.42, 0.6), clamp=True)
            # a thin broken line of foam a few metres out (a spent wave)
            band2 = nk.math("MULTIPLY", nk.mrange(d, 3.6, 4.0), nk.mrange(d, 4.6, 4.2))
            f2 = nk.math("MULTIPLY", band2, nk.mrange(nk.noise(2.8, detail=6, offset=rng.uniform(-40, 40, 3)),
                                                      0.45, 0.6), clamp=True)
            fm = nk.math("MULTIPLY", nk.math("MAXIMUM", f, nk.math("MULTIPLY", f2, 0.7)), foam, clamp=True)
            col = nk.mixrgb(fm, col, (0.72, 0.74, 0.72))
            rr = nt.nodes.new("ShaderNodeMapRange")
            rr.inputs["To Min"].default_value, rr.inputs["To Max"].default_value = rough, 0.55
            nt.links.new(fm, rr.inputs["Value"])
            rgh = rr.outputs["Result"]
    if col is None:
        b.inputs["Base Color"].default_value = (*deep, 1.0)
    else:
        nt.links.new(col, b.inputs["Base Color"])
    if rgh is None:
        b.inputs["Roughness"].default_value = rough
    else:
        nt.links.new(rgh, b.inputs["Roughness"])
    # waves: long swell + chop, stretched along one axis
    ang = float(rng.uniform(0, math.pi))
    stretch = (1.0, 2.5, 1.0)
    w1 = nk.noise(0.12 * wave, detail=3, rough=0.5, offset=rng.uniform(-50, 50, 3), stretch=stretch)
    w2 = nk.noise(0.9 * wave, detail=8, rough=0.62, offset=rng.uniform(-50, 50, 3))
    w3 = nk.noise(4.0 * wave, detail=4, rough=0.5, offset=rng.uniform(-50, 50, 3))
    h = nk.math("ADD", nk.math("ADD", w1, nk.math("MULTIPLY", w2, 0.45)), nk.math("MULTIPLY", w3, 0.12))
    bp = nt.nodes.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = 0.35
    bp.inputs["Distance"].default_value = 0.08
    nt.links.new(h, bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    del ang
    return m


def sea_mesh(ctx, shore, mat, far=1800.0, z=0.012, step=1.0, name="sea"):
    """The sea as a flat overlay from the wavy shoreline out to ``far``
    (non-colliding: it is floor, 1 cm up)."""
    nx, ny = shore["n"]
    n = np.array([nx, ny], float)
    t = np.array([-ny, nx], float)
    zx0, zy0, zx1, zy1 = ctx.zone
    cz = np.array([(zx0 + zx1) / 2, (zy0 + zy1) / 2])
    tc = float(cz @ t)
    near = np.arange(tc - 90, tc + 90, step)
    ts = np.concatenate([np.linspace(tc - far, tc - 90 - 5, 40), near, np.linspace(tc + 90 + 5, tc + far, 40)])
    V, F = [], []
    for i, tt in enumerate(ts):
        s = shore_y(shore, tt)
        a = t * tt + n * s
        b = t * tt + n * max(far, s + 50)
        V.append((a[0], a[1], z))
        V.append((b[0], b[1], z))
        if i:
            k = 2 * i
            F.append((k - 2, k, k + 1, k - 1))
    # orient faces up
    ob = ctx.mesh(V, F, mat, coll="ground", collide=False, name=name)
    me = ob.data
    if len(me.polygons) and me.polygons[0].normal.z < 0:
        me.flip_normals()
    return ob


def weather(mat, rng, moss=(0.16, 0.2, 0.08), damp: float = 0.5, streaks: float = 0.5, lichen: float = 0.3,
            base_h: float = 0.8):
    """Age a ``looks.pbr`` material in place: rising damp and moss low down,
    rain streaks, lichen/moss spots (all world-space, so neighbouring
    blocks share them). Returns the material."""
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    bsdf = next(n for n in N if n.type == "BSDF_PRINCIPLED")
    if not bsdf.inputs["Base Color"].links:
        return mat
    src = bsdf.inputs["Base Color"].links[0].from_socket
    nk = NK(nt)
    g = N.new("ShaderNodeNewGeometry")
    pos = g.outputs["Position"]
    z = nk.sep(pos)[2]
    off = rng.uniform(-60, 60, 3)
    # rising damp: dark band at the foot with a noisy top edge
    zn = nk.noise(0.9, detail=4, offset=off, vec=pos)
    edge = nk.math("ADD", base_h * 0.6, nk.math("MULTIPLY", zn, base_h))
    low = nk.math("SUBTRACT", 1.0, nk.mrange(nk.math("SUBTRACT", z, edge), -0.15, 0.15))
    col = nk.mixrgb(nk.math("MULTIPLY", low, damp), src, (0.09, 0.085, 0.06))
    # streaks: noise stretched vertically
    st = nk.noise(2.2, detail=3, offset=off + 11, stretch=(1.0, 1.0, 0.08), vec=pos)
    stm = nk.mrange(st, 0.5, 0.75)
    col = nk.mixrgb(nk.math("MULTIPLY", stm, streaks * 0.55), col, (0.12, 0.11, 0.09))
    # lichen / moss spots
    sp = nk.noise(1.4, detail=8, rough=0.7, offset=off + 23, vec=pos)
    spm = nk.mrange(sp, 0.6, 0.7)
    col = nk.mixrgb(nk.math("MULTIPLY", spm, lichen), col, moss)
    L.new(col, bsdf.inputs["Base Color"])
    return mat


def _local_tris(ob) -> np.ndarray | None:
    """Local-space triangles (n, 3, 3) of an evaluated mesh object."""
    me = ob.to_mesh()
    try:
        me.calc_loop_triangles()
        n = len(me.loop_triangles)
        if not n:
            return None
        V = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", V)
        F = np.empty(n * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", F)
        return V.reshape(-1, 3)[F.reshape(-1, 3)]
    finally:
        ob.to_mesh_clear()


def _area_in(T, M, zone, zmax) -> float:
    """Area of the triangles ``T`` (local) placed by ``M`` whose centroid lies
    in ``zone`` below ``zmax``."""
    W = T @ M[:3, :3].T + M[:3, 3]
    c = W.mean(1)
    keep = (c[:, 0] >= zone[0]) & (c[:, 0] <= zone[2]) & (c[:, 1] >= zone[1]) & (c[:, 1] <= zone[3]) & \
        (c[:, 2] <= zmax)
    if not keep.any():
        return 0.0
    W = W[keep]
    return float(0.5 * np.linalg.norm(np.cross(W[:, 1] - W[:, 0], W[:, 2] - W[:, 0]), axis=1).sum())


def cloud_estimate(ctx, voxel=None, zmax=None):
    """Rough obstacle-cloud size per colliding object: the surface area it
    puts in the zone under the cloud top, over voxel^2 -- {object name:
    points}, an instance counted by its instancing object. Walks the evaluated
    depsgraph exactly as ``build_course.collidable_triangles`` does (instances
    placed, cropped to the zone and the cloud top), so a tall fort module or a
    tree crown is charged only for what the cloud will hold. ``voxel`` and
    ``zmax`` default to ``build_course``'s ``CLOUD_VOXEL`` / ``CLOUD_ZMAX``."""
    import build_course as BC  # noqa: PLC0415 (bpy side)
    voxel = float(voxel or BC.CLOUD_VOXEL)
    zmax = float(zmax or BC.CLOUD_ZMAX)
    zone = ctx.zone
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    cache: dict = {}
    est: dict = {}
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type != "MESH":
            continue
        tag = (inst.parent if inst.is_instance else ob).original
        if not int(tag.get("collide", 0)):
            continue
        key = ob.original.name
        if key not in cache:
            cache[key] = _local_tris(ob)
        if cache[key] is None:
            continue
        a = _area_in(cache[key], np.array(inst.matrix_world), zone, zmax)
        est[tag.name] = est.get(tag.name, 0.0) + a / voxel ** 2
    return est


def cloud_guard(ctx, budget=600_000, protect=("obstacles",)):
    """Keep the obstacle cloud under ``budget`` points: drop the biggest
    optional colliding objects (scatter first, then structure) until the
    estimate fits. Returns (estimate before, after, objects removed)."""
    import bpy  # noqa: PLC0415
    est = cloud_estimate(ctx)
    total = before = sum(est.values())
    removed = 0
    for cname in ("scatter", "structure"):
        if total <= budget:
            break
        obs = sorted((o for o in ctx.colls[cname].objects if o.name in est and not o.name.startswith("prim_")),
                     key=lambda o: -est[o.name])
        for o in obs:
            if total <= budget:
                break
            total -= est[o.name]
            bpy.data.objects.remove(o, do_unlink=True)
            removed += 1
    return int(before), int(total), removed


def piece_tris(lib, pkey: str) -> int:
    """Catalogued triangle count of a library piece (0 if not inspected)."""
    fk, i = pkey.split("#")
    return int(next((p.get("tris", 0) for p in lib.pieces(fk) if p["index"] == int(i)), 0))


def pick(rng, keys, lib=None):
    keys = [k for k in keys if lib is None or k in lib.items]
    return str(keys[int(rng.integers(len(keys)))]) if keys else None


# --- free space for long / spreading things ----------------------------------------------------

def clearance_many(ctx, xy, ignore=()) -> np.ndarray:
    """``ctx.clearance`` for many xy points at once (vectorised over the
    claimed footprints): distance to the nearest claim, negative inside.
    ``ignore``: claims (the dicts in ``ctx.foot``) left out, by identity."""
    P = np.asarray(xy, float).reshape(-1, 2)
    d = np.full(len(P), 1e9)
    foot = [f for f in ctx.foot if not any(f is g for g in ignore)]
    discs = [f for f in foot if f["type"] == "disc"]
    if discs:
        C = np.array([f["c"] for f in discs])
        R = np.array([f["r"] for f in discs])
        for i in range(0, len(P), 4096):
            Q = P[i:i + 4096]
            d[i:i + 4096] = (np.linalg.norm(Q[:, None] - C[None], axis=2) - R[None]).min(1)
    for f in foot:
        if f["type"] != "rect":
            continue
        q = np.abs((P - f["c"]) @ f["R"]) - f["half"]
        d = np.minimum(d, np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0))
    return d


def seg_free(ctx, a, b, r: float = 0.1, margin: float = 0.3, step: float = 0.2, ignore=()) -> bool:
    """Every point of the segment a -> b (xy) is ``r + margin`` clear of every
    claim -- a beam, rope or line checked along its length, not at its ends.
    ``ignore``: claims the line may touch -- the posts it is strung between,
    when they are claimed first (else their discs fail it at both ends)."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    n = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / step)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return bool((clearance_many(ctx, a + (b - a) * t, ignore) >= r + margin).all())


def parts_free(ctx, parts: "Parts", margin: float = 0.3, z_min: float = 0.05, n0: int = 0) -> bool:
    """Every vertex of ``parts`` (from index ``n0``) above ``z_min`` is
    ``margin`` clear of every claim: for things whose reach is not a disc
    round their base (a dead tree's limbs)."""
    V = np.asarray(parts.V[n0:], float)
    if not len(V):
        return True
    V = V[V[:, 2] > z_min]
    return bool((clearance_many(ctx, V[:, :2]) >= margin).all()) if len(V) else True


def clear_of_ends(ctx, xy, reach: float) -> bool:
    """Something reaching ``reach`` m round ``xy`` stays out of the start and
    goal keep-out discs (the dressing of a patch primitive that stands close
    to them must not bring it closer)."""
    r = next((f["r"] for f in ctx.foot if f.get("kind") == "start"), 1.5)
    xy = np.asarray(xy, float)[:2]
    return min(float(np.linalg.norm(xy - ctx.start)), float(np.linalg.norm(xy - ctx.goal))) - reach >= r


def rect_clear_of_ends(ctx, c, half, yaw: float = 0.0) -> bool:
    """A rect footprint (centre, half extents, yaw) lies wholly outside the
    start and goal keep-out discs -- ``clear_of_ends`` for dressing that is
    long rather than round (a crown block, a snow cap)."""
    r = next((f["r"] for f in ctx.foot if f.get("kind") == "start"), 1.5)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rt = np.array([[cy, sy], [-sy, cy]])
    h = np.asarray(half, float)[:2]
    for e in (ctx.start, ctx.goal):
        q = np.abs(Rt @ (np.asarray(e, float)[:2] - np.asarray(c, float)[:2])) - h
        if float(np.linalg.norm(np.maximum(q, 0.0))) + min(float(q.max()), 0.0) < r:
            return False
    return True


def top_cap(c, half, R, t: float):
    """A slab ``2 t`` thick lying on the top face of the box (centre ``c``,
    half extents ``half``, rotation ``R``): (centre, half) to build with the
    same ``R`` -- the box's own footprint, tilt and yaw (a cap sized from the
    rotated box's axis-aligned extents overhangs a yawed box)."""
    c, h, R = np.asarray(c, float), np.asarray(half, float), np.asarray(R, float)
    up = int(np.argmax(np.abs(R[2])))              # the box axis nearest world z
    hs = h.copy()
    hs[up] = t
    return c + R[:, up] * np.sign(R[2, up]) * (h[up] + t), hs


# --- the openings a family implies ---------------------------------------------------------------

def _clusters(P, link: float) -> list:
    """Single-linkage groups (index arrays) of 2-D points closer than ``link``."""
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    lab, k = -np.ones(len(P), int), 0
    for i in range(len(P)):
        if lab[i] >= 0:
            continue
        lab[i], stack = k, [i]
        while stack:
            j = stack.pop()
            for m in np.nonzero((D[j] < link) & (lab < 0))[0]:
                lab[m] = k
                stack.append(int(m))
        k += 1
    return [np.nonzero(lab == g)[0] for g in range(k)]


def ring_openings(prims) -> list:
    """Every ring of poles (``cyl_v`` of a ``ring*`` region, split into rings
    by 1 m linkage): {centre, r, pole_r, top, idx, gap} with ``gap`` None for
    a closed ring, else {phi (bisector), theta (between the edge poles)}.
    Measured off the poles (least-squares circle, largest angular step), so
    corridor legs and rotated arena cells read the same as tier A."""
    idx = [i for i, p in enumerate(prims) if p["kind"] == "cyl_v" and str(p.get("region", "")).startswith("ring")]
    out = []
    if not idx:
        return out
    C = np.array([prims[i]["c"][:2] for i in idx], float)
    for g in _clusters(C, 1.0):
        if len(g) < 8:
            continue
        P = C[g]
        sol = np.linalg.lstsq(np.c_[2 * P, np.ones(len(P))], (P ** 2).sum(1), rcond=None)[0]
        c = sol[:2]
        r = float(np.sqrt(sol[2] + c @ c))
        ang = np.sort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
        dif = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
        j = int(np.argmax(dif))
        gap = None
        if dif[j] > 2.5 * float(np.median(dif)):
            gap = {"phi": float(ang[j] + dif[j] / 2), "theta": float(dif[j])}
        out.append({"centre": c, "r": r, "pole_r": max(float(prims[idx[i]]["r"]) for i in g),
                    "top": max(float(prims[idx[i]]["z1"]) for i in g), "idx": [idx[i] for i in g], "gap": gap})
    return out


def claim_openings(ctx, keepout: float | None = None) -> list:
    """Claim the openings a family implies that ``build_primitives`` does not
    (it claims wall slots and the band round a bar), at the level's keep-out
    ``ko`` and in the slot approach's proportions, so no structure, clutter or
    plant lands in them:

    * a ring of poles: its interior (to the poles' inner face) and, when open,
      its gap -- from the chord between the edge poles out to r + ko, the
      chord + 0.4 ko either side wide;
    * a short wall (``wall_offset*`` / ``wall_sym*``): the passage past each
      end, ko along the wall and ko either side of it;
    * a level floating box (underside > 0.4 m: float box, float bar, lintel):
      the approach to the space under it, ko in front and behind, the box +
      0.4 ko either side wide (+ ko: the way round a ``float_box*``).

    Call right after ``dress_primitives``. Returns the claims made, for the
    course record ({kind, prims, c, half, yaw} / {kind, prims, c, r})."""
    ko = float(ctx.knobs.get("keepout", 3.0) if keepout is None else keepout)
    prims = ctx.spec["primitives"]
    made = []

    def rect(kind, ii, c, half, yaw):
        ctx.claim_rect(c, half, yaw, kind=kind)
        made.append({"kind": kind, "prims": [int(i) for i in ii], "c": [round(float(v), 3) for v in c],
                     "half": [round(float(v), 3) for v in half], "yaw": round(float(yaw), 4)})

    for rg in ring_openings(prims):
        c, r, pr = rg["centre"], rg["r"], rg["pole_r"]
        ctx.claim_disc(c, r - pr, kind="ring_inside")
        made.append({"kind": "ring_inside", "prims": rg["idx"], "c": [round(float(v), 3) for v in c],
                     "r": round(r - pr, 3)})
        if rg["gap"] is not None:
            phi, th = rg["gap"]["phi"], rg["gap"]["theta"]
            u = np.array([math.cos(phi), math.sin(phi)])
            s0, s1 = r * math.cos(th / 2), r + ko
            rect("ring_gap", rg["idx"], c + u * (s0 + s1) / 2, ((s1 - s0) / 2, r * math.sin(th / 2) + 0.4 * ko), phi)
    for i, p in enumerate(prims):
        reg = str(p.get("region", ""))
        if p["kind"] == "wall" and reg.startswith(("wall_offset", "wall_sym")):
            p0, p1 = np.asarray(p["p0"], float)[:2], np.asarray(p["p1"], float)[:2]
            d = (p1 - p0) / max(float(np.linalg.norm(p1 - p0)), 1e-9)
            for e, o in ((p0, -d), (p1, d)):
                rect("passage", [i], e + o * ko / 2, (ko / 2, ko), math.atan2(o[1], o[0]))
        elif p["kind"] == "box":
            c, h = np.asarray(p["c"], float), np.asarray(p["half"], float)
            if c[2] - h[2] <= 0.4 or abs(p["rpy"][0]) > 1e-3 or abs(p["rpy"][1]) > 1e-3:
                continue
            k = 0 if h[0] <= h[1] else 1                    # the thin axis: under is crossed along it
            half = h[:2].copy()
            half[k] += ko
            half[1 - k] += ko if reg.startswith("float_box") else 0.4 * ko
            rect("float_approach", [i], c[:2], half, float(p["rpy"][2]))
    return made


def fit_scale(r0: float, sc: float, cl: float, pad: float = 0.05, s_min: float = 0.5):
    """The largest scale <= ``sc`` at which a piece of footprint radius ``r0``
    stays ``pad`` inside the free space ``cl`` round it; None if even
    ``s_min`` does not fit."""
    if r0 * sc <= cl - pad:
        return sc
    s = (cl - pad) / max(r0, 1e-3)
    return s if s >= s_min else None
