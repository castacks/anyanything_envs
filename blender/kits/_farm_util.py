"""Shared helpers for the ``farm`` and ``backyard`` kits (rural/suburban
timber-and-sheet-metal places): merged box meshes, oriented beams, tubes
and sagging wires, gable/gambrel/skillion buildings with doors and windows,
post-and-rail / board / wire fences, round and square bales, wagons, tree
rows. Everything goes through the ``Ctx`` (``ctx.mesh`` / ``ctx.box`` /
``ctx.cylinder`` / ``ctx.instance``) so the cloud-zone contract holds: what
stands in the zone collides, what is wholly outside is scenery.

Axis convention for buildings: a building is a rectangle centred at ``o``
(xy) with its ridge along the unit vector ``d`` (length ``L``) and depth
``W`` across it (along ``n`` = d rotated +90 deg). Local ``(u, v, z)`` ->
world ``o + u d + v n``. Kits keep ``d`` axis-aligned so the box-projected
scans (siding boards, bricks) stay level.

The module itself needs no ``bpy`` (only the ``ctx`` does): the free-space
helpers (``ring_openings``, ``line_free``, ``box_free``, ...) import and run
outside Blender, for tests and gates.
"""

from __future__ import annotations

import math

import numpy as np

# --- material helpers ---------------------------------------------------------------------------


def have(ctx, keys) -> list[str]:
    """The keys of ``keys`` that exist in the library (materials must have a colour map)."""
    out = []
    for k in keys:
        e = ctx.lib.items.get(k)
        if e is None:
            continue
        if e["kind"] == "material" and "diff" not in e.get("maps", {}):
            continue
        out.append(k)
    return out


def pick(ctx, keys, fallback=None) -> str | None:
    ks = have(ctx, keys)
    if not ks and fallback:
        ks = have(ctx, fallback)
    return ctx.lib.pick(ctx.rng, ks)


def mat(ctx, keys, tile, **kw):
    """A pbr material from a random existing key of ``keys``."""
    k = pick(ctx, keys)
    if k is None:
        return ctx.looks.paint(kw.get("tint") or (0.5, 0.5, 0.5), name="fallback")
    return ctx.looks.pbr(k, tile, **kw)


def pick_hdri(ctx, keys, fallback_tags=("field",)) -> str:
    ks = [k for k in keys if k in ctx.lib.items]
    if not ks:
        ks = ctx.lib.hdris(cats_all={"outdoor"}, cats_none={"night", "indoor"}, tags_any=set(fallback_tags))
    return ctx.lib.pick(ctx.rng, ks)


# --- low-level meshes ---------------------------------------------------------------------------


def smooth_sides(ob):
    """Shade a cylinder's side quads smooth (its n-gon caps stay flat): same geometry, no facets."""
    for poly in ob.data.polygons:
        poly.use_smooth = len(poly.vertices) == 4
    return ob


def cyl(ctx, a, b, r, mat_, **kw):
    return smooth_sides(ctx.cylinder(a, b, r, mat_, **kw))


_CUBE_F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
_CUBE_S = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)


def yaw_R(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box_verts(c, half, R=None) -> np.ndarray:
    R = np.eye(3) if R is None else np.asarray(R, float)
    return (_CUBE_S * np.asarray(half, float)) @ R.T + np.asarray(c, float)


def boxes(ctx, items, mat_, coll="structure", name="boxes", collide=None, roof=False):
    """Many boxes as ONE mesh object: ``items`` = [(centre, half, R_or_yaw), ...]."""
    if not items:
        return None
    V, F = [], []
    for it in items:
        c, h = it[0], it[1]
        R = it[2] if len(it) > 2 else None
        if R is not None and np.isscalar(R):
            R = yaw_R(float(R))
        base = len(V)
        V.extend(box_verts(c, h, R).tolist())
        F.extend([tuple(base + i for i in f) for f in _CUBE_F])
    return ctx.mesh(V, F, mat_, coll=coll, name=name, collide=collide, roof=roof)


def beam_frame(a, b):
    """Right-handed frame (t, s, u) for a beam from a to b (t along it)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = b - a
    L = float(np.linalg.norm(t))
    t = t / max(L, 1e-9)
    up = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    s = np.cross(up, t)
    s /= np.linalg.norm(s)
    u = np.cross(t, s)
    return t, s, u, L


def beam_item(a, b, w, h):
    """(centre, half, R) of an oriented box beam from a to b, cross-section w (side) x h (up)."""
    t, s, u, L = beam_frame(a, b)
    R = np.stack([t, s, u], 1)
    c = (np.asarray(a, float) + np.asarray(b, float)) / 2
    return (c, (L / 2, w / 2, h / 2), R)


def beams(ctx, segs, w, h, mat_, coll="structure", name="beams", collide=None):
    return boxes(ctx, [beam_item(a, b, w, h) for a, b in segs], mat_, coll=coll, name=name, collide=collide)


def tube(ctx, pts, r, mat_, sides=8, closed=False, coll="structure", name="tube", collide=None, caps=True):
    """A round tube along a 3-D polyline (wires, pipes, rings) as one mesh."""
    P = np.asarray(pts, float)
    n = len(P)
    if n < 2:
        return None
    V, F = [], []
    prev = None
    for i in range(n):
        if closed:
            tan = P[(i + 1) % n] - P[i - 1]
        else:
            tan = P[min(i + 1, n - 1)] - P[max(i - 1, 0)]
        tan /= max(np.linalg.norm(tan), 1e-9)
        if prev is None:
            ref = np.array([0.0, 0.0, 1.0]) if abs(tan[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            x = np.cross(ref, tan)
        else:
            x = prev - tan * float(prev @ tan)
        x /= max(np.linalg.norm(x), 1e-9)
        prev = x
        y = np.cross(tan, x)
        for k in range(sides):
            a = 2 * math.pi * k / sides
            V.append((P[i] + r * (math.cos(a) * x + math.sin(a) * y)).tolist())
    rings = n if closed else n - 1
    for i in range(rings):
        j = (i + 1) % n
        for k in range(sides):
            k2 = (k + 1) % sides
            F.append((i * sides + k, i * sides + k2, j * sides + k2, j * sides + k))
    if caps and not closed:
        F.append(tuple(range(sides))[::-1])
        F.append(tuple((n - 1) * sides + k for k in range(sides)))
    return ctx.mesh(V, F, mat_, coll=coll, name=name, collide=collide, smooth=True)


def catenary(a, b, sag, n=14):
    """Points of a wire from a to b sagging ``sag`` metres at mid-span (parabola)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.linspace(0, 1, n)
    P = a[None] + (b - a)[None] * t[:, None]
    P[:, 2] -= 4 * sag * t * (1 - t)
    return P


def wire(ctx, a, b, sag, r, mat_, coll="structure", name="wire", n=14):
    return tube(ctx, catenary(a, b, sag, n), r, mat_, sides=6, coll=coll, name=name)


def prism(ctx, poly_vz, u0, u1, o, d, mat_, coll="structure", name="prism", roof=False):
    """Extrude a (v, z) polygon along u in [u0, u1] (building local frame)."""
    o, d = np.asarray(o, float)[:2], np.asarray(d, float)[:2]
    nrm = np.array([-d[1], d[0]])
    m = len(poly_vz)
    V = []
    for u in (u0, u1):
        for v, z in poly_vz:
            p = o + u * d + v * nrm
            V.append((p[0], p[1], z))
    F = [tuple(range(m))[::-1], tuple(m + i for i in range(m))]
    for i in range(m):
        j = (i + 1) % m
        F.append((i, j, m + j, m + i))
    return ctx.mesh(V, F, mat_, coll=coll, name=name, roof=roof)


def roof_slabs(ctx, profile, u0, u1, o, d, mat_, thick=0.12, tile=2.0, coll="structure", name="roof"):
    """Roof sheets over a (v, z) profile polyline (eave -> ridge -> eave),
    from u0 to u1, as one mesh with UVs in metres / ``tile`` (u along the
    eave, s up the slope) so tile rows run parallel to the eave."""
    o, d = np.asarray(o, float)[:2], np.asarray(d, float)[:2]
    nrm = np.array([-d[1], d[0]])
    V, F, UV = [], [], []

    def W(u, v, z):
        p = o + u * d + v * nrm
        return (float(p[0]), float(p[1]), float(z))

    s_acc = 0.0
    for (va, za), (vb, zb) in zip(profile[:-1], profile[1:]):
        seg = np.array([vb - va, zb - za])
        Ls = float(np.linalg.norm(seg))
        up = np.array([-seg[1], seg[0]]) / max(Ls, 1e-9)        # outward normal of the sheet in (v, z)
        if up[1] < 0:
            up = -up
        dv, dz = up * thick
        base = len(V)
        # bottom face corners then top face corners
        V += [W(u0, va, za), W(u1, va, za), W(u1, vb, zb), W(u0, vb, zb),
              W(u0, va + dv, za + dz), W(u1, va + dv, za + dz), W(u1, vb + dv, zb + dz), W(u0, vb + dv, zb + dz)]
        uu0, uu1 = u0 / tile, u1 / tile
        ss0, ss1 = s_acc / tile, (s_acc + Ls) / tile
        faces = [((0, 3, 2, 1), [(uu0, ss0), (uu0, ss1), (uu1, ss1), (uu1, ss0)]),
                 ((4, 5, 6, 7), [(uu0, ss0), (uu1, ss0), (uu1, ss1), (uu0, ss1)]),
                 ((0, 1, 5, 4), [(uu0, ss0), (uu1, ss0), (uu1, ss0 + 0.05), (uu0, ss0 + 0.05)]),
                 ((1, 2, 6, 5), [(0, ss0), (0, ss1), (0.05, ss1), (0.05, ss0)]),
                 ((2, 3, 7, 6), [(uu1, ss1), (uu0, ss1), (uu0, ss1 + 0.05), (uu1, ss1 + 0.05)]),
                 ((3, 0, 4, 7), [(0, ss1), (0, ss0), (0.05, ss0), (0.05, ss1)])]
        for f, uv in faces:
            F.append(tuple(base + i for i in f))
            UV.append(uv)
        s_acc += Ls
    return ctx.mesh(V, F, mat_, uvs=UV, coll=coll, name=name, roof=True)


# --- buildings ------------------------------------------------------------------------------------


class Building:
    """A rectangular building: body box to the eaves, an attic prism under
    the roof profile, roof sheets with overhang. ``kind``: gable, gambrel,
    skillion (mono-pitch, high side at v = -W/2) or flat."""

    def __init__(self, ctx, o, d, L, W, h_eave, wall, roof, kind="gable", pitch=35.0, overhang=0.35,
                 coll="structure", roof_tile=2.0, trim=None, open_front=False, post_mat=None, drop=0.0,
                 roof_thick=0.12):
        self.ctx, self.o, self.L, self.W, self.h = ctx, np.asarray(o, float)[:2], L, W, h_eave
        self.d = np.asarray(d, float)[:2] / np.linalg.norm(d)
        self.n = np.array([-self.d[1], self.d[0]])
        self.yaw = math.atan2(self.d[1], self.d[0])
        self.coll, self.wall, self.trim = coll, wall, trim
        oh = overhang
        t = math.tan(math.radians(pitch))
        hw = W / 2
        if kind == "gable":
            ridge = h_eave + hw * t
            attic = [(-hw, h_eave), (0.0, ridge), (hw, h_eave)]
            prof = [(-hw - oh, h_eave - oh * t), (0.0, ridge), (hw + oh, h_eave - oh * t)]
        elif kind == "gambrel":
            k = 0.3 * W                                    # knee inset from the wall
            hk = h_eave + k * math.tan(math.radians(62))
            ridge = hk + (hw - k) * math.tan(math.radians(24))
            attic = [(-hw, h_eave), (-hw + k, hk), (0.0, ridge), (hw - k, hk), (hw, h_eave)]
            s1 = (hk - h_eave) / k
            prof = [(-hw - oh, h_eave - oh * s1), (-hw + k, hk), (0.0, ridge), (hw - k, hk), (hw + oh, h_eave - oh * s1)]
        elif kind == "skillion":
            ridge = h_eave + W * t
            attic = [(-hw, ridge), (hw, h_eave), (-hw, h_eave)]
            attic = [(-hw, h_eave), (-hw, ridge), (hw, h_eave)]
            prof = [(-hw - oh, ridge + oh * t), (hw + oh, h_eave - oh * t)]
        else:                                              # flat
            ridge = h_eave
            attic = None
            prof = [(-hw - oh, h_eave + 0.02), (hw + oh, h_eave + 0.02)]
        self.ridge = ridge
        self.profile = prof
        if open_front:                                     # pole shed: back + ends walls, posts along the front
            self._open_shell(post_mat or wall, attic)
        else:
            ctx.box((self.o[0], self.o[1], h_eave / 2), (L / 2, W / 2, h_eave / 2), wall, yaw=self.yaw, coll=coll,
                    name="bldg")
            if attic:
                prism(ctx, attic, -L / 2, L / 2, self.o, self.d, wall, coll=coll, name="attic")
        roof_slabs(ctx, prof, -L / 2 - oh, L / 2 + oh, self.o, self.d, roof, thick=roof_thick, tile=roof_tile,
                   coll=coll)
        if trim is not None and kind in ("gable", "gambrel"):      # barge boards along the gable edges
            segs = []
            for u in (-L / 2 - oh, L / 2 + oh):
                for (va, za), (vb, zb) in zip(prof[:-1], prof[1:]):
                    segs.append((self.W3(u, va, za - 0.05), self.W3(u, vb, zb - 0.05)))
            beams(ctx, segs, 0.05, 0.22, trim, coll=coll, name="barge")

    def W3(self, u, v, z):
        p = self.o + u * self.d + v * self.n
        return np.array([p[0], p[1], z])

    def _open_shell(self, post_mat, attic):
        ctx, L, W, h = self.ctx, self.L, self.W, self.h
        t = 0.08
        items = [(self.W3(0, W / 2 - t / 2, h / 2), (L / 2, t / 2, h / 2), self.yaw)]      # back wall
        for u in (-L / 2 + t / 2, L / 2 - t / 2):                                          # end walls
            items.append((self.W3(u, 0, h / 2), (t / 2, W / 2, h / 2), self.yaw))
        boxes(ctx, items, self.wall, coll=self.coll, name="shed_wall")
        if attic:
            prism(ctx, attic, -L / 2, L / 2, self.o, self.d, self.wall, coll=self.coll, name="attic")
        nb = max(2, int(round(L / 4.5)))
        posts = []
        for i in range(nb + 1):
            u = -L / 2 + 0.12 + (L - 0.24) * i / nb
            posts.append((self.W3(u, -W / 2 + 0.12, h / 2), (0.1, 0.1, h / 2), self.yaw))
        posts.append((self.W3(0, -W / 2 + 0.12, h - 0.15), (L / 2, 0.1, 0.15), self.yaw))  # front beam
        boxes(ctx, posts, post_mat, coll=self.coll, name="shed_post")

    # features on the long faces (side = -1: the v = -W/2 face, +1 the other) or the ends (end = -1/+1)
    def on_face(self, side, u, w, z0, z1, depth):
        c = self.W3(u, side * (self.W / 2 + depth / 2 - 0.005), (z0 + z1) / 2)
        return (c, (w / 2, depth / 2, (z1 - z0) / 2), self.yaw)

    def on_end(self, end, v, w, z0, z1, depth):
        c = self.W3(end * (self.L / 2 + depth / 2 - 0.005), v, (z0 + z1) / 2)
        return (c, (depth / 2, w / 2, (z1 - z0) / 2), self.yaw)

    def window(self, glass, frame, side=None, u=0.0, w=1.0, z0=1.0, z1=2.2, end=None, v=0.0, panes=(2, 2),
               sill=None, shutters=None):
        """A framed window proud of the wall: frame, recessed-looking glass, glazing bars, a sill."""
        fr, gl = [], []
        mk = (lambda uu, ww, a, b, dp: self.on_face(side, uu, ww, a, b, dp)) if end is None else \
            (lambda uu, ww, a, b, dp: self.on_end(end, uu, ww, a, b, dp))
        c0 = u if end is None else v
        fw = 0.07
        gl.append(mk(c0, w, z0, z1, 0.03))
        fr.append(mk(c0, w + 2 * fw, z1, z1 + fw, 0.07))
        fr.append(mk(c0, w + 2 * fw, z0 - fw, z0, 0.07))
        fr.append(mk(c0 - w / 2 - fw / 2, fw, z0, z1, 0.07))
        fr.append(mk(c0 + w / 2 + fw / 2, fw, z0, z1, 0.07))
        nx, nz = panes
        for i in range(1, nx):
            fr.append(mk(c0 - w / 2 + w * i / nx, 0.035, z0, z1, 0.05))
        for i in range(1, nz):
            zz = z0 + (z1 - z0) * i / nz
            fr.append(mk(c0, w, zz - 0.0175, zz + 0.0175, 0.05))
        if sill is not None:
            fr.append(mk(c0, w + 0.25, z0 - fw - 0.05, z0 - fw, 0.14))
        if shutters is not None:
            for sgn in (-1, 1):
                boxes(self.ctx, [mk(c0 + sgn * (w / 2 + fw + w * 0.27), w * 0.5, z0, z1, 0.04)], shutters,
                      coll=self.coll, name="shutter")
        boxes(self.ctx, gl, glass, coll=self.coll, name="glass")
        boxes(self.ctx, fr, frame, coll=self.coll, name="wframe")

    def door(self, panel, frame, side=None, u=0.0, w=1.0, h=2.1, end=None, v=0.0, brace=None, depth=0.05):
        """A door panel with a frame; ``brace`` adds a barn-door Z/X brace in that material."""
        mk = (lambda uu, ww, a, b, dp: self.on_face(side, uu, ww, a, b, dp)) if end is None else \
            (lambda uu, ww, a, b, dp: self.on_end(end, uu, ww, a, b, dp))
        c0 = u if end is None else v
        boxes(self.ctx, [mk(c0, w, 0.0, h, depth)], panel, coll=self.coll, name="door")
        fw = 0.1
        boxes(self.ctx, [mk(c0, w + 2 * fw, h, h + fw, depth + 0.03), mk(c0 - w / 2 - fw / 2, fw, 0, h, depth + 0.03),
                         mk(c0 + w / 2 + fw / 2, fw, 0, h, depth + 0.03)], frame, coll=self.coll, name="dframe")
        if brace is not None:
            # X brace + rails: 0.14 m boards, 0.04 m thick, laid on the panel (2 mm in)
            if end is None:
                off = side * (self.W / 2 + depth + 0.013)
                p = lambda a, z: self.W3(c0 + a, off, z)
            else:
                off = end * (self.L / 2 + depth + 0.013)
                p = lambda a, z: self.W3(off, c0 + a, z)
            segs = [(p(-w / 2 + 0.1, 0.15), p(w / 2 - 0.1, h - 0.15)), (p(w / 2 - 0.1, 0.15), p(-w / 2 + 0.1, h - 0.15)),
                    (p(-w / 2 + 0.05, h / 2), p(w / 2 - 0.05, h / 2)),
                    (p(-w / 2 + 0.05, 0.12), p(w / 2 - 0.05, 0.12)), (p(-w / 2 + 0.05, h - 0.12), p(w / 2 - 0.05, h - 0.12))]
            beams(self.ctx, segs, 0.04, 0.14, brace, coll=self.coll, name="brace")

    def corner_trim(self, mat_, w=0.12):
        items = []
        for su in (-1, 1):
            for sv in (-1, 1):
                c = self.W3(su * (self.L / 2 + 0.01), sv * (self.W / 2 + 0.01), self.h / 2)
                items.append((c, (w / 2, w / 2, self.h / 2), self.yaw))
        boxes(self.ctx, items, mat_, coll=self.coll, name="ctrim")

    def footprint(self, pad=0.0):
        return self.o, (self.L / 2 + pad, self.W / 2 + pad), self.yaw


# --- fences ---------------------------------------------------------------------------------------


def _seg_ok(ctx, a, b, halfw, margin, check):
    if not check:
        return True
    c = (a + b) / 2
    L = float(np.linalg.norm(b - a))
    yaw = math.atan2(b[1] - a[1], b[0] - a[0])
    return ctx.free_rect(c, (L / 2, halfw), yaw, margin=margin, step=0.4)


def fence_line(ctx, a, b, spacing, kind, mats, h=1.3, check=True, margin=0.25, coll="structure", gaps=(),
               claim=True, jitter=0.0):
    """A fence from a to b: ``kind`` post_rail (2-4 timber rails), wire
    (posts + strands), board (solid boards), picket. Segments that are not
    free are left out (a gap), and a list of (t0, t1) ``gaps`` along the
    line (metres) are left open for gates. Returns the posts placed."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    L = float(np.linalg.norm(b - a))
    if L < 0.5:
        return []
    dvec = (b - a) / L
    nrm = np.array([-dvec[1], dvec[0]])
    yaw = math.atan2(dvec[1], dvec[0])
    n = max(1, int(round(L / spacing)))
    ts = np.linspace(0, L, n + 1)
    rng = ctx.rng
    posts, rails, boards, strands = [], [], [], []
    placed_posts = set()
    pw = mats.get("post_w", 0.11)
    ok = [not any(g0 < (ts[i] + ts[i + 1]) / 2 < g1 for g0, g1 in gaps)
          and _seg_ok(ctx, a + dvec * ts[i], a + dvec * ts[i + 1], 0.12, margin, check) for i in range(n)]
    for i in range(n):
        t0, t1 = ts[i], ts[i + 1]
        if not ok[i]:
            continue
        p0, p1 = a + dvec * t0, a + dvec * t1
        for k, p in ((i, p0), (i + 1, p1)):
            if k not in placed_posts:
                placed_posts.add(k)
                hh = h + 0.08 + rng.uniform(-0.04, 0.04)
                lean = rng.normal(0, jitter)
                posts.append(((p[0], p[1], hh / 2), (pw / 2, pw / 2, hh / 2), yaw + lean))
        if kind == "post_rail":
            nr = mats.get("rails", 3)
            for r in range(nr):
                z = h * (0.35 + 0.6 * r / max(1, nr - 1)) if nr > 1 else h * 0.8
                z += rng.normal(0, 0.015)
                q0 = np.array([p0[0], p0[1], z]) + np.r_[nrm * (pw / 2 + 0.03), 0]
                q1 = np.array([p1[0], p1[1], z + rng.normal(0, 0.02)]) + np.r_[nrm * (pw / 2 + 0.03), 0]
                rails.append(beam_item(q0 - np.r_[dvec * 0.05, 0], q1 + np.r_[dvec * 0.05, 0], 0.045, 0.11))
        elif kind == "wire":
            ns = mats.get("strands", 4)
            for r in range(ns):
                z = h * (0.3 + 0.68 * r / max(1, ns - 1))
                strands.append((np.array([p0[0], p0[1], z]), np.array([p1[0], p1[1], z])))
        elif kind in ("board", "picket"):
            # rails behind, boards in front (on the -n side)
            for z in (0.3, h - 0.3):
                q0 = np.array([p0[0], p0[1], z]) + np.r_[nrm * 0.07, 0]
                q1 = np.array([p1[0], p1[1], z]) + np.r_[nrm * 0.07, 0]
                rails.append(beam_item(q0, q1, 0.04, 0.09))
            bw = mats.get("board_w", 0.14)
            gap = mats.get("board_gap", 0.01 if kind == "board" else 0.07)
            nb = max(1, int((t1 - t0 - pw) / (bw + gap)))
            for j in range(nb):
                tt = t0 + pw / 2 + (j + 0.5) * (t1 - t0 - pw) / nb
                p = a + dvec * tt - nrm * 0.0
                hb = h + (0.0 if kind == "board" else 0.1) + rng.normal(0, 0.006)
                boards.append(((p[0], p[1], 0.04 + hb / 2), (bw / 2, 0.011, hb / 2), yaw))
        if claim:
            c = (p0 + p1) / 2
            ctx.claim_rect(c, ((t1 - t0) / 2 + 0.1, 0.15), yaw, kind="fence")
    if posts:
        boxes(ctx, posts, mats["post"], coll=coll, name="fpost")
    if rails:
        boxes(ctx, rails, mats.get("rail", mats["post"]), coll=coll, name="frail")
    if boards:
        boxes(ctx, boards, mats.get("board", mats["post"]), coll=coll, name="fboard")
    for s0, s1 in strands:
        wire(ctx, s0, s1, 0.03, 0.006, mats.get("wire", mats["post"]), coll=coll, name="fwire", n=6)
    return posts


def fence_path(ctx, pts, spacing, kind, mats, **kw):
    out = []
    for p, q in zip(pts[:-1], pts[1:]):
        out += fence_line(ctx, p, q, spacing, kind, mats, **kw)
    return out


def farm_gate(ctx, a, b, mat_, h=1.15, coll="structure"):
    """A tubular 5-bar farm gate from a to b (xy), hung on a post at a."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    segs = []
    for k in range(5):
        z = 0.15 + (h - 0.15) * k / 4
        segs.append((np.r_[a, z], np.r_[b, z]))
    segs += [(np.r_[a, 0.1], np.r_[a, h + 0.03]), (np.r_[b, 0.1], np.r_[b, h + 0.03]),
             (np.r_[a, 0.2], np.r_[b, h - 0.05])]
    mid = (a + b) / 2
    segs.append((np.r_[mid, 0.12], np.r_[mid, h]))
    for s0, s1 in segs:
        tube(ctx, [s0, s1], 0.022, mat_, sides=8, coll=coll, name="gate")


# --- farm objects ---------------------------------------------------------------------------------


def round_bale(ctx, xy, yaw, mat_side, mat_end, r=0.75, w=1.2, standing=False, coll="scatter", claim=True):
    x, y = xy
    if standing:
        cyl(ctx, (x, y, 0.0), (x, y, w), r, mat_side, coll=coll, name="bale", segments=28)
        cyl(ctx, (x, y, w), (x, y, w + 0.012), r * 0.97, mat_end, coll=coll, name="bale_end", segments=28)
        rad = r
    else:
        d = np.array([math.cos(yaw), math.sin(yaw), 0.0]) * w / 2
        c = np.array([x, y, r])
        cyl(ctx, c - d, c + d, r, mat_side, coll=coll, name="bale", segments=28)
        e = d / np.linalg.norm(d) * 0.012
        cyl(ctx, c + d, c + d + e, r * 0.97, mat_end, coll=coll, name="bale_end", segments=28)
        cyl(ctx, c - d - e, c - d, r * 0.97, mat_end, coll=coll, name="bale_end", segments=28)
        rad = math.hypot(r, w / 2)
    if claim:
        ctx.claim_disc(xy, rad + 0.1, kind="bale")
    return rad


def square_bale_stack(ctx, c, yaw, nx, ny, nz, size, mat_, coll="scatter", claim=True, ragged=0.3):
    """Stacked square bales (``size`` = (l, w, h)) in an nx x ny grid, nz high, topping out raggedly."""
    l, w, h = size
    R = yaw_R(yaw)
    items = []
    rng = ctx.rng
    for k in range(nz):
        for i in range(nx):
            for j in range(ny):
                if k > 0 and rng.random() < ragged * k / nz:
                    continue
                off = np.array([(i - (nx - 1) / 2) * (l + 0.02), (j - (ny - 1) / 2) * (w + 0.02), 0])
                off[:2] += rng.normal(0, 0.03, 2)
                p = np.asarray([c[0], c[1], 0.0]) + R @ off
                p[2] = h / 2 + k * h
                items.append((p, (l / 2, w / 2, h / 2), yaw + rng.normal(0, 0.03)))
    boxes(ctx, items, mat_, coll=coll, name="sqbale")
    half = (nx * (l + 0.02) / 2 + 0.1, ny * (w + 0.02) / 2 + 0.1)
    if claim:
        ctx.claim_rect(c, half, yaw, kind="bales")
    return half


def wheel(ctx, c, axis, r, width, tyre, hub, coll="scatter"):
    c, axis = np.asarray(c, float), np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    cyl(ctx, c - axis * width / 2, c + axis * width / 2, r, tyre, coll=coll, name="tyre", segments=28)
    cyl(ctx, c - axis * (width / 2 + 0.01), c + axis * (width / 2 + 0.01), r * 0.5, hub, coll=coll,
        name="hub", segments=20)
    cyl(ctx, c - axis * (width / 2 + 0.03), c + axis * (width / 2 + 0.03), r * 0.12, hub, coll=coll,
        name="hubcap", segments=10)


def wagon(ctx, xy, yaw, deck, chassis, tyre, hub, load=None, L=4.2, W=2.1, coll="scatter"):
    """A flatbed hay wagon: deck, low side boards, chassis, 4 wheels, drawbar."""
    x, y = xy
    R = yaw_R(yaw)
    d, n = R[:, 0], R[:, 1]
    c = np.array([x, y, 0.0])
    zd = 0.95
    items = [(c + np.r_[0, 0, zd], (L / 2, W / 2, 0.05), yaw)]
    for s in (-1, 1):
        items.append((c + n * s * (W / 2 - 0.03) + np.r_[0, 0, zd + 0.2], (L / 2, 0.03, 0.15), yaw))
    for s in (-1, 1):
        items.append((c + d * s * (L / 2 - 0.03) + np.r_[0, 0, zd + 0.2], (0.03, W / 2, 0.15), yaw))
    boxes(ctx, items, deck, coll=coll, name="wagon_deck")
    ch = [(c + n * s * 0.5 + np.r_[0, 0, zd - 0.14], (L / 2 - 0.2, 0.06, 0.09), yaw) for s in (-1, 1)]
    for s in (-1, 1):
        ch.append((c + d * s * (L / 2 - 0.8) + np.r_[0, 0, 0.45], (0.05, W / 2 - 0.1, 0.05), yaw))
    boxes(ctx, ch, chassis, coll=coll, name="wagon_chassis")
    beams(ctx, [(c + d * (L / 2 - 0.3) + np.r_[0, 0, 0.5], c + d * (L / 2 + 1.4) + np.r_[0, 0, 0.35])], 0.09, 0.09,
          chassis, coll=coll, name="drawbar")
    for su in (-1, 1):
        for sv in (-1, 1):
            wheel(ctx, c + d * su * (L / 2 - 0.8) + n * sv * (W / 2 - 0.12) + np.r_[0, 0, 0.45], n, 0.45, 0.28, tyre,
                  hub, coll=coll)
    if load == "square":
        return zd + 0.05
    return None


def tractor(ctx, xy, yaw, paint, dark, tyre, hub, glass, coll="scatter"):
    """A compact farm tractor built from boxes: bonnet, cab with pillars and
    roof, big rear / small front wheels, mudguards, exhaust."""
    x, y = xy
    R = yaw_R(yaw)
    d, n = R[:, 0], R[:, 1]
    c = np.array([x, y, 0.0])
    rr, rf = 0.78, 0.48
    xr, xf = -0.75, 1.35                               # axle positions along d
    body = [(c + d * 0.55 + np.r_[0, 0, 1.05], (1.05, 0.42, 0.38), yaw),        # bonnet
            (c + d * 1.62 + np.r_[0, 0, 0.95], (0.05, 0.4, 0.33), yaw),         # grille
            (c + d * (-0.35) + np.r_[0, 0, 0.85], (0.55, 0.5, 0.25), yaw)]     # transmission
    boxes(ctx, body, paint, coll=coll, name="tractor_body")
    under = [(c + d * 0.4 + np.r_[0, 0, 0.62], (1.1, 0.3, 0.14), yaw),
             (c + d * 1.62 + np.r_[0, 0, 0.93], (0.02, 0.3, 0.22), yaw)]
    boxes(ctx, under, dark, coll=coll, name="tractor_chassis")
    # cab
    zc0, zc1 = 1.12, 2.55
    cab = []
    for su in (-1, 1):
        for sv in (-1, 1):
            p = c + d * (-0.75 + su * 0.55) + n * sv * 0.6
            cab.append((p + np.r_[0, 0, (zc0 + zc1) / 2], (0.035, 0.035, (zc1 - zc0) / 2), yaw))
    cab.append((c + d * -0.75 + np.r_[0, 0, zc1 + 0.05], (0.68, 0.72, 0.06), yaw))
    boxes(ctx, cab, dark, coll=coll, name="tractor_cab")
    gl = [(c + d * (-0.75 + 0.55) + np.r_[0, 0, 1.85], (0.012, 0.56, 0.62), yaw),
          (c + d * (-0.75 - 0.55) + np.r_[0, 0, 1.85], (0.012, 0.56, 0.62), yaw)]
    for sv in (-1, 1):
        gl.append((c + d * -0.75 + n * sv * 0.6 + np.r_[0, 0, 1.85], (0.52, 0.012, 0.62), yaw))
    boxes(ctx, gl, glass, coll=coll, name="tractor_glass")
    for sv in (-1, 1):                                   # mudguards over the rear wheels
        boxes(ctx, [(c + d * xr + n * sv * 0.82 + np.r_[0, 0, 2 * rr - 0.02], (0.55, 0.26, 0.03), yaw)], paint,
              coll=coll, name="guard")
    for sv in (-1, 1):
        wheel(ctx, c + d * xr + n * sv * 0.82 + np.r_[0, 0, rr], n, rr, 0.42, tyre, hub, coll=coll)
        wheel(ctx, c + d * xf + n * sv * 0.72 + np.r_[0, 0, rf], n, rf, 0.28, tyre, hub, coll=coll)
    ex = c + d * 0.95 + n * 0.3
    cyl(ctx, ex + np.r_[0, 0, 1.4], ex + np.r_[0, 0, 2.5], 0.045, dark, coll=coll, name="exhaust", segments=12)


def water_trough(ctx, xy, yaw, metal, water, L=2.4, W=0.7, H=0.6, coll="scatter"):
    x, y = xy
    R = yaw_R(yaw)
    d, n = R[:, 0], R[:, 1]
    c = np.array([x, y, 0.0])
    t = 0.025
    items = [(c + np.r_[0, 0, 0.08], (L / 2, W / 2, 0.02), yaw)]
    for s in (-1, 1):
        items.append((c + n * s * (W / 2 - t / 2) + np.r_[0, 0, H / 2], (L / 2, t / 2, H / 2), yaw))
        items.append((c + d * s * (L / 2 - t / 2) + np.r_[0, 0, H / 2], (t / 2, W / 2, H / 2), yaw))
    boxes(ctx, items, metal, coll=coll, name="trough")
    boxes(ctx, [(c + np.r_[0, 0, H - 0.09], (L / 2 - t, W / 2 - t, 0.01), yaw)], water, coll=coll, name="water")


def stock_tank(ctx, xy, metal, water, r=1.0, H=0.65, coll="scatter"):
    x, y = xy
    cyl(ctx, (x, y, 0), (x, y, H), r, metal, coll=coll, name="tank", segments=36)
    cyl(ctx, (x, y, H), (x, y, H + 0.004), r - 0.05, water, coll=coll, name="water", segments=36)


def pallet_stack(ctx, xy, yaw, wood, n=1, coll="scatter"):
    x, y = xy
    R = yaw_R(yaw)
    items = []
    for k in range(n):
        z0 = k * 0.145
        for j in (-1, 0, 1):
            items.append((np.array([x, y, 0]) + R @ np.array([0, j * 0.36, 0]) + np.r_[0, 0, z0 + 0.05],
                          (0.6, 0.05, 0.05), yaw))
        for i in range(7):
            items.append((np.array([x, y, 0]) + R @ np.array([-0.55 + i * 0.183, 0, 0]) + np.r_[0, 0, z0 + 0.12],
                          (0.045, 0.4, 0.011), yaw))
    boxes(ctx, items, wood, coll=coll, name="pallet")


def log_pile(ctx, xy, yaw, bark, ends, L=2.5, rows=3, r=0.14, coll="scatter"):
    """Logs stacked in a pyramid, bark sides and sawn end discs."""
    x, y = xy
    R = yaw_R(yaw)
    d, n = R[:, 0], R[:, 1]
    c0 = np.array([x, y, 0.0])
    rng = ctx.rng
    k = 0
    for row in range(rows):
        cnt = rows + 2 - row
        for i in range(cnt):
            rr = r * rng.uniform(0.8, 1.15)
            v = (i - (cnt - 1) / 2) * 2 * r * 1.02
            z = r + row * 2 * r * 0.88
            c = c0 + n * v + np.r_[0, 0, z] + d * rng.normal(0, 0.08)
            a, b = c - d * (L / 2 + rng.normal(0, 0.05)), c + d * (L / 2 + rng.normal(0, 0.05))
            cyl(ctx, a, b, rr, bark, coll=coll, name="log", segments=14)
            e = d * 0.006
            cyl(ctx, b, b + e, rr * 0.93, ends, coll=coll, name="logend", segments=14)
            cyl(ctx, a - e, a, rr * 0.93, ends, coll=coll, name="logend", segments=14)
            k += 1
    return (L / 2 + 0.2, (rows + 2) * r * 1.05 + 0.1)


def utility_pole(ctx, xy, wood, metal, h=8.5, arm=1.8, yaw=0.0, coll="structure", r=0.13):
    """A wooden distribution pole with a crossarm and insulators; returns the
    attachment points of its three conductors."""
    x, y = xy
    cyl(ctx, (x, y, 0), (x, y, h), r, wood, coll=coll, name="upole", segments=14)
    R = yaw_R(yaw)
    n = R[:, 1]
    za = h - 0.45
    boxes(ctx, [((x, y, za), (0.05, arm / 2, 0.05), yaw)], wood, coll=coll, name="crossarm")
    pts = []
    for s in (-0.42, 0.0, 0.42):
        p = np.array([x, y, za]) + np.r_[n[:2] * s * arm, 0]
        if s == 0.0:
            p = np.array([x, y, h + 0.1])
        cyl(ctx, p, p + np.r_[0, 0, 0.14], 0.035, metal, coll=coll, name="insul", segments=8)
        pts.append(p + np.r_[0, 0, 0.14])
    return pts


def tree_row(ctx, pkeys, a, b, spacing, s=(0.8, 1.2), jitter=1.5, offset_j=2.0, skip=0.1, coll="backdrop",
             avoid=None):
    """Trees along a -> b (scenery). ``avoid`` = footprint kinds (buildings,
    scenery) a trunk must not stand in: a tree is left out unless 70 % of its
    crown radius keeps clear of them; no other free-space checks."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    L = float(np.linalg.norm(b - a))
    if L < 1e-6 or not pkeys:
        return []
    d = (b - a) / L
    nrm = np.array([-d[1], d[0]])
    out = []
    rng = ctx.rng
    for t in np.arange(spacing * rng.uniform(0.2, 0.8), L, spacing):
        if rng.random() < skip:
            continue
        xy = a + d * (t + rng.normal(0, jitter)) + nrm * rng.normal(0, offset_j)
        k = ctx.lib.pick(rng, pkeys)
        sc = float(rng.uniform(*s))
        if avoid and foot_clearance(ctx, xy, avoid) < 0.35 * max(ctx.lib.size(k)[:2]) * sc:
            continue
        out.append(ctx.instance(k, xy, 0.0, float(rng.uniform(0, 2 * math.pi)), sc, coll=coll))
    return out


def jagged_strip(ctx, a, b, width, mat_, z=0.004, step=1.5, jit=0.12, name="track", tile=None):
    """A flat, non-colliding ground strip with ragged edges (farm tracks, worn
    paths). ``tile`` (m) gives it UVs running along the strip (u across, v
    along), for ``proj="uv"`` scans whose pattern has a direction (tyre ruts)."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    L = float(np.linalg.norm(b - a))
    d = (b - a) / L
    nrm = np.array([-d[1], d[0]])
    n = max(2, int(L / step) + 1)
    rng = ctx.rng
    V, F = [], []
    wob = 0.0
    for i in range(n):
        t = L * i / (n - 1)
        wob = 0.85 * wob + rng.normal(0, jit * 0.6)
        c = a + d * t + nrm * wob
        for s in (-1, 1):
            p = c + nrm * s * (width / 2 + rng.normal(0, jit))
            V.append((p[0], p[1], z))
    UV = []
    for i in range(n - 1):
        F.append((2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1))
        if tile:
            t0, t1 = L * i / (n - 1) / tile, L * (i + 1) / (n - 1) / tile
            wu = width / tile
            UV.append([(0.5 - wu / 2, t0), (0.5 - wu / 2, t1), (0.5 + wu / 2, t1), (0.5 + wu / 2, t0)])
    return ctx.mesh(V, F, mat_, uvs=UV if tile else None, coll="ground", collide=False, name=name)


def blob(ctx, c, rx, ry, mat_, z=0.004, k=18, jit=0.18, name="patch"):
    """A flat irregular ground patch (a worn spot, a puddle, a bed)."""
    rng = ctx.rng
    V = []
    ph = rng.uniform(0, 2 * math.pi)
    for i in range(k):
        a = 2 * math.pi * i / k
        f = 1 + jit * math.sin(3 * a + ph) + rng.normal(0, jit * 0.4)
        V.append((c[0] + rx * f * math.cos(a), c[1] + ry * f * math.sin(a), z))
    return ctx.mesh(V, [tuple(range(k))], mat_, coll="ground", collide=False, name=name)


def ring_openings(prims, min_gap_deg: float = 25.0) -> list[dict]:
    """The entrance of every post ring (tier-A ``ring`` and the arenas'
    ``ring_*`` cells): posts grouped by ``region``, the widest angular gap
    between neighbours. Each: centre, radius, the two bounding posts' angles
    (rad, a0 -> a1 counter-clockwise across the gap) and the gap in degrees."""
    groups = {}
    for p in prims:
        if p["kind"] == "cyl_v" and str(p.get("region", "")).startswith("ring"):
            groups.setdefault(p["region"], []).append(p)
    out = []
    for posts in groups.values():
        if len(posts) < 6:
            continue
        C = np.array([p["c"] for p in posts], float)
        A = np.c_[C, np.ones(len(C))]                   # algebraic circle fit (the mean is pulled off the gap)
        sol = np.linalg.lstsq(A, (C ** 2).sum(1), rcond=None)[0]
        c = sol[:2] / 2
        ang = np.sort(np.arctan2(C[:, 1] - c[1], C[:, 0] - c[0]))
        gaps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
        i = int(np.argmax(gaps))
        if math.degrees(gaps[i]) < min_gap_deg:
            continue                                    # a closed ring
        out.append({"c": c, "r": float(np.linalg.norm(C - c, axis=1).mean()), "a0": float(ang[i]),
                    "a1": float(ang[i] + gaps[i]), "gap_deg": math.degrees(gaps[i])})
    return out


def claim_openings(ctx, keepout: float) -> int:
    """Keep the way into every post ring clear, as ``build_primitives`` does
    for wall slots: a rectangle across the entrance chord, ``keepout`` deep on
    both sides. Returns how many were claimed."""
    rs = ring_openings(ctx.spec["primitives"])
    for o in rs:
        p0 = o["c"] + o["r"] * np.array([math.cos(o["a0"]), math.sin(o["a0"])])
        p1 = o["c"] + o["r"] * np.array([math.cos(o["a1"]), math.sin(o["a1"])])
        t = p1 - p0
        L = float(np.linalg.norm(t))
        ctx.claim_rect((p0 + p1) / 2, (L / 2 + 0.4 * keepout, max(keepout, 1.0)), math.atan2(t[1], t[0]),
                       kind="ring_gap")
    return len(rs)


#: footprints a hanging wire / line must not cross: the patch primitives (and their
#: opening approaches) and the post-ring entrances
PRIM_KINDS = ("prim", "ring_gap")
#: footprints of what stands above head height (and of the lines already strung): a
#: wire, festoon or branch in the flight band must not pass through them. Library
#: pieces ("piece") are judged by their own height instead (``line_free``).
TALL_KINDS = PRIM_KINDS + ("building", "house", "shed", "scenery", "tree", "pole", "model", "machine", "swings",
                           "trampoline", "parasol", "rope_swing", "hedge", "rotary", "line")


def _foot_dist(Q, f) -> np.ndarray:
    """Signed xy distances of the points ``Q`` (n x 2) to one footprint (``footprint_distance``, vectorised)."""
    if f["type"] == "disc":
        return np.linalg.norm(Q - f["c"], axis=1) - f["r"]
    q = np.abs((Q - f["c"]) @ f["R"]) - f["half"]
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)


def _feet(ctx, kinds, skip=()):
    ids = {id(x) for x in skip}
    return [f for f in ctx.foot if (kinds is None or f.get("kind") in kinds) and id(f) not in ids]


def _clear(ctx, Q, kinds, skip=()) -> np.ndarray:
    """Per point of ``Q`` (n x 2): distance to the nearest footprint of ``kinds`` (None = all), not ``skip``."""
    Q = np.atleast_2d(np.asarray(Q, float))[:, :2]
    out = np.full(len(Q), 1e9)
    for f in _feet(ctx, kinds, skip):
        np.minimum(out, _foot_dist(Q, f), out=out)
    return out


def foot_clearance(ctx, xy, kinds=PRIM_KINDS, skip=()) -> float:
    """Distance from ``xy`` to the nearest claimed footprint of ``kinds`` (None = all), ``skip`` left out."""
    return float(_clear(ctx, xy, kinds, skip)[0])


def seg_free(ctx, a, b, margin=0.3, kinds=PRIM_KINDS, step=0.3, skip=()) -> bool:
    """True if the xy segment a-b keeps ``margin`` from every claimed footprint
    of ``kinds`` (None = every footprint; ``skip`` = footprints left out):
    lines, tapes and cables strung in the flight band must not cut through
    the patch's obstacles or openings."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    n = max(2, int(float(np.linalg.norm(b - a)) / step) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return bool((_clear(ctx, a + (b - a) * t, kinds, skip) >= margin).all())


def line_free(ctx, P, margin=0.3, kinds=TALL_KINDS, z_clear=0.3, skip=(), trim=(0.0, 0.0), step=0.25) -> bool:
    """True if the polyline ``P`` (n x 3: a sagging wire, a festoon, a branch)
    keeps ``margin`` in xy from every claimed footprint of ``kinds`` (None =
    all; footprints carry no height, so they count at any height) and passes
    ``z_clear`` above, or ``margin`` beside, every placed library piece.
    ``skip`` = footprints / ``ctx.placed`` entries it hangs from; ``trim`` =
    metres left unchecked at (start, end): where it is fixed to an eave, a
    pole or a trunk."""
    P = np.asarray(P, float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    ta, tb = (trim, trim) if np.isscalar(trim) else trim
    if s[-1] - ta - tb <= 0:
        return True
    q = np.linspace(ta, s[-1] - tb, max(2, int((s[-1] - ta - tb) / step) + 1))
    Q = np.stack([np.interp(q, s, P[:, k]) for k in range(3)], 1)
    if (_clear(ctx, Q, kinds, skip) < margin).any():
        return False
    ids = {id(x) for x in skip}
    pl = [p for p in ctx.placed if id(p) not in ids]
    if not pl:
        return True
    xy = np.array([p["xy"] for p in pl], float)
    r = np.array([0.5 * math.hypot(p["size"][0], p["size"][1]) for p in pl])
    top = np.array([p["z"] + p["size"][2] for p in pl])
    near = np.linalg.norm(Q[:, None, :2] - xy[None], axis=2) < r[None] + margin
    return not (near & (Q[:, None, 2] < top[None] + z_clear)).any()


def claim_line(ctx, a, b, half_w=0.15, kind="line"):
    """Claim the xy strip under a line strung a -> b (later lines keep off it)."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    t = b - a
    ctx.claim_rect((a + b) / 2, (float(np.linalg.norm(t)) / 2 + half_w, half_w), math.atan2(t[1], t[0]), kind=kind)


def box_free(ctx, lo, hi, margin=1.0, kinds=None) -> bool:
    """True if no claimed footprint of ``kinds`` (None = all) comes within
    ``margin`` of the axis-aligned xy box ``lo``..``hi`` (footprints judged by
    their bounding boxes: cheap, for big scenery)."""
    lo, hi = np.asarray(lo, float)[:2] - margin, np.asarray(hi, float)[:2] + margin
    for f in _feet(ctx, kinds):
        ext = np.full(2, f["r"]) if f["type"] == "disc" else np.abs(f["R"]) @ f["half"]
        if np.all(f["c"] - ext < hi) and np.all(f["c"] + ext > lo):
            return False
    return True


def obj_models(ctx, uids, h=None) -> list[str]:
    """Piece keys of the Objaverse models ``uids`` (``obj:<uid>`` or bare
    uids) that the library holds and does not exclude; ``[]`` without the
    Objaverse catalog, so a kit falls back to its own pieces."""
    return ctx.lib.models(ids=[u if u.startswith("obj:") else f"obj:{u}" for u in uids], h=h)


def place_model(ctx, pkey, xy, heading, margin=0.4, s=1.0, coll="scatter", check=True, claim=True):
    """A library piece with its long axis along ``heading`` (either way round),
    base on the ground, if its oriented footprint rectangle keeps ``margin``
    from every footprint (``check``); claims that rectangle. The object or None."""
    sx, sy, _ = np.asarray(ctx.lib.size(pkey), float) * s
    yaw = float(heading) if sx >= sy else float(heading) - math.pi / 2
    half = (sx / 2, sy / 2)
    if check and not ctx.free_rect(xy, half, yaw, margin=margin, step=0.4):
        return None
    ob = ctx.instance(pkey, xy, 0.0, yaw, s, coll=coll)
    if claim:
        ctx.claim_rect(xy, (half[0] + 0.1, half[1] + 0.1), yaw, kind="model")
    return ob


def seg_clear(ctx, a, b, r, pts=None):
    """True if the xy segment a-b keeps ``r`` metres from the start and the goal (and ``pts``)."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    for p in [ctx.start, ctx.goal] + list(pts or []):
        p = np.asarray(p, float)[:2]
        ab = b - a
        t = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-9), 0, 1))
        if np.linalg.norm(a + t * ab - p) < r:
            return False
    return True


def prim_list(ctx):
    """(index, primitive, obstacle objects) for the patch primitives after dress_primitives."""
    objs = list(ctx.colls["obstacles"].objects)
    out = []
    for i, p in enumerate(ctx.spec["primitives"]):
        tag = f"prim{i:02d}_"
        out.append((i, p, [o for o in objs if o.name.startswith(tag)]))
    return out
