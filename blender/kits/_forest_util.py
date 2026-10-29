"""Nature helpers shared by the forest and park kits (runs inside Blender).

* ``clear_many`` -- ``Ctx.clearance`` for many points at once (numpy), so a
  kit can test hundreds of ground-cover spots against the claimed footprints
  without a Python loop per point;
* ``trunk`` / ``limb`` -- procedural tree trunks and branches: a curved,
  tapering tube with a lobed root flare (the part of a tree a drone at
  1.5 m actually sees), built through ``ctx.mesh`` so the labeller's
  contract holds (in the zone it collides and is in the cloud);
* ``coat`` / ``coat_piece`` -- lay a second scan over a material where the
  surface faces up (snow on rocks and logs, moss on boulder tops and wall
  copings) or low down (char on burnt trunks, damp at the foot of a wall),
  patchy by world-space noise;
* ``crown`` -- a library tree set on top of a trunk (a "crown on a pole"):
  the canopy of a 15-20 m tree without importing a 5 M-triangle tree, and,
  set wholly above the cloud ceiling (``cloud_top``), without touching the
  labeller.
"""

from __future__ import annotations

import math
import sys

import bpy
import numpy as np

import build_course as BC


# --- free space, vectorised ------------------------------------------------------------------

def clear_many(ctx, P, skip=(), skip_foot=()) -> np.ndarray:
    """Distance from each xy in ``P`` (n, 2) to the nearest claimed footprint
    (footprints whose kind is in ``skip``, and the footprint dicts in
    ``skip_foot`` themselves -- a host tree's own claim -- do not count)."""
    P = np.asarray(P, float).reshape(-1, 2)
    d = np.full(len(P), 1e9)
    for f in ctx.foot:
        if skip and f.get("kind") in skip:
            continue
        if skip_foot and any(f is g for g in skip_foot):
            continue
        if f["type"] == "disc":
            dd = np.linalg.norm(P - f["c"], axis=1) - f["r"]
        else:
            q = np.abs((P - f["c"]) @ f["R"]) - f["half"]
            dd = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
        d = np.minimum(d, dd)
    return d


#: the footprints nothing may reach into, whatever its size: the start and goal
#: discs and the patch primitives with their opening approaches (``Ctx.dress_primitives``)
HARD = ("start", "goal", "prim")


def clear_of(ctx, P, kinds=HARD) -> np.ndarray:
    """``clear_many`` against only the footprints whose kind is in ``kinds``
    (default ``HARD``): lets a root clump overlap its own tree's claim but not
    the start disc or a wall slot's approach."""
    P = np.asarray(P, float).reshape(-1, 2)
    d = np.full(len(P), 1e9)
    for f in ctx.foot:
        if f.get("kind") not in kinds:
            continue
        if f["type"] == "disc":
            dd = np.linalg.norm(P - f["c"], axis=1) - f["r"]
        else:
            q = np.abs((P - f["c"]) @ f["R"]) - f["half"]
            dd = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
        d = np.minimum(d, dd)
    return d


#: ground-level dressing (plants under knee height, litter, twigs) may lie inside a family's region
LOW = ("family",)


def claim_family_regions(ctx, keepout: float = 1.5) -> list[dict]:
    """Claim (kind ``family``) the space a family's obstacles define, so the
    kit adds nothing standing inside it: a stand's or a tier-C forest cell's
    tree field (the bbox of its trunks -- a kit tree among them would make a
    sparse stand dense), a ring's disc and the approach to its gap, a room's
    interior. Only ``LOW`` dressing (ground cover, litter) may go there.
    Returns what was claimed, for the course record."""
    prims = ctx.spec["primitives"]
    groups: dict = {}
    for p in prims:
        groups.setdefault(p.get("region", ""), []).append(p)
    out = []
    for name, ps in sorted(groups.items()):
        cyl = [p for p in ps if p["kind"] == "cyl_v"]
        walls = [p for p in ps if p["kind"] == "wall"]
        if name.startswith(("stand", "forest")) and cyl:
            C = np.array([p["c"] for p in cyl], float)
            R = np.array([p["r"] for p in cyl], float)
            lo, hi = (C - R[:, None]).min(0) - 0.3, (C + R[:, None]).max(0) + 0.3
            ctx.claim_rect((lo + hi) / 2, (hi - lo) / 2, 0.0, kind="family")
            out.append({"region": name, "claim": "stand", "rect": [round(float(v), 2) for v in (*lo, *hi)]})
        elif name.startswith("ring") and len(cyl) >= 6:
            C = np.array([p["c"] for p in cyl], float)
            A = np.column_stack([2 * C, np.ones(len(C))])      # the circle through the posts (least squares):
            c = np.linalg.lstsq(A, (C ** 2).sum(1), rcond=None)[0][:2]   # their mean sits off-centre, away from the gap
            rad = float(np.linalg.norm(C - c, axis=1).max() + max(p["r"] for p in cyl))
            ctx.claim_disc(c, rad + 0.3, kind="family")
            ang = np.sort(np.arctan2(C[:, 1] - c[1], C[:, 0] - c[0]))
            gaps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
            j = int(np.argmax(gaps))
            rec = {"region": name, "claim": "ring", "c": [round(float(v), 2) for v in c], "r": round(rad + 0.3, 2)}
            if gaps[j] > math.radians(30):         # the way in: a band out from the gap
                a = ang[j] + gaps[j] / 2
                dv = np.array([math.cos(a), math.sin(a)])
                chord = 2 * rad * math.sin(gaps[j] / 2)
                ctx.claim_rect(c + dv * (rad + keepout / 2), (keepout / 2 + 0.3, chord / 2), a, kind="family")
                rec["gap_deg"] = round(math.degrees(float(gaps[j])), 1)
            out.append(rec)
        elif name.startswith("room") and len(walls) >= 3:
            xy = np.array([q for w in walls for q in (w["p0"][:2], w["p1"][:2])], float)
            lo, hi = xy.min(0), xy.max(0)
            ctx.claim_rect((lo + hi) / 2, (hi - lo) / 2, 0.0, kind="family")
            out.append({"region": name, "claim": "room", "rect": [round(float(v), 2) for v in (*lo, *hi)]})
    return out


def piece_r(ctx, pkey, s=1.0) -> float:
    """Half the footprint diagonal of library piece ``pkey`` at scale ``s``."""
    sx, sy, _ = ctx.lib.size(pkey)
    return 0.5 * math.hypot(sx, sy) * s


def claim_segment(ctx, a, b, half_w, kind="hang"):
    """Claim the xy band of half-width ``half_w`` along a -> b (a branch, a
    cable), so later placements keep off it."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        ctx.claim_disc(a, half_w, kind=kind)
        return
    ctx.claim_rect((a + b) / 2, (L / 2 + half_w, half_w), math.atan2(d[1], d[0]), kind=kind)


def segment_clear(ctx, a, b, margin: float, step: float = 0.25, skip_ends: float = 0.0, skip=(),
                  skip_foot=()) -> bool:
    """Every point of the xy segment a -> b (minus ``skip_ends`` metres at
    either end) is ``margin`` clear of the claimed footprints (``skip`` /
    ``skip_foot`` as in ``clear_many``)."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    L = float(np.linalg.norm(b - a))
    if L < 1e-6:
        return bool(clear_many(ctx, a[None], skip, skip_foot)[0] >= margin)
    ts = np.linspace(skip_ends / L, 1 - skip_ends / L, max(2, int(L / step) + 1)) if L > 2 * skip_ends else [0.5]
    P = np.array([a + (b - a) * t for t in ts])
    return bool(clear_many(ctx, P, skip, skip_foot).min() >= margin)


def free_spots(ctx, n, rect, r, margin, rng, where=None, tries=6, skip=()):
    """Up to ``n`` random xy in ``rect`` clear of footprints by r + margin
    (checked in batches; the spots do not claim, and may overlap each other;
    footprints of a kind in ``skip`` do not count)."""
    x0, y0, x1, y1 = rect
    out = []
    for _ in range(tries):
        m = max(8, 2 * (n - len(out)))
        P = np.column_stack([rng.uniform(x0, x1, m), rng.uniform(y0, y1, m)])
        ok = clear_many(ctx, P, skip) >= r + margin
        if where is not None:
            ok &= np.array([bool(where(p)) for p in P])
        out.extend(P[ok][: n - len(out)])
        if len(out) >= n:
            break
    return out


# --- materials --------------------------------------------------------------------------------

def _coat_fac(nt, mode, lo, hi, patchy, patch_scale, rng):
    N, L = nt.nodes, nt.links
    geo = N.new("ShaderNodeNewGeometry")
    sep = N.new("ShaderNodeSeparateXYZ")
    mr = N.new("ShaderNodeMapRange")
    if mode == "up":                               # surfaces facing the sky
        L.new(geo.outputs["Normal"], sep.inputs["Vector"])
        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = lo, hi
    else:                                          # "low": near the ground (z from lo to hi fades out)
        L.new(geo.outputs["Position"], sep.inputs["Vector"])
        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = hi, lo
    L.new(sep.outputs["Z"], mr.inputs["Value"])
    fac = mr.outputs["Result"]
    if patchy > 0:
        nz = N.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = 1.0 / patch_scale
        nz.inputs["Detail"].default_value = 4.0
        mp = N.new("ShaderNodeMapping")
        mp.inputs["Location"].default_value = tuple(rng.uniform(-40, 40, 3))
        L.new(geo.outputs["Position"], mp.inputs["Vector"])
        L.new(mp.outputs["Vector"], nz.inputs["Vector"])
        m2 = N.new("ShaderNodeMapRange")
        c = 0.5 - 0.12 * (1.0 - patchy)
        m2.inputs["From Min"].default_value, m2.inputs["From Max"].default_value = c - 0.08, c + 0.08
        m2.inputs["To Min"].default_value = 1.0 - patchy
        L.new(nz.outputs["Fac"], m2.inputs["Value"])
        mul = N.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.use_clamp = True
        L.new(fac, mul.inputs[0])
        L.new(m2.outputs["Result"], mul.inputs[1])
        fac = mul.outputs["Value"]
    return fac


def _coat_bsdf(lk, nt, coat, tile):
    """The coat's BSDF inside ``nt``: a library scan key, or an RGB (paint)."""
    if isinstance(coat, str):
        return lk.pbr(coat, tile, into=nt, vary=0.1, jitter=0.0).outputs["BSDF"]
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (*coat[:3], 1.0)
    b.inputs["Roughness"].default_value = coat[3] if len(coat) > 3 else 0.85
    return b.outputs["BSDF"]


def coat(lk, base, coat, tile=1.0, mode="up", lo=0.45, hi=0.8, patchy=0.5, patch_scale=1.2, name=None):
    """A copy of material ``base`` with ``coat`` (scan key or RGB) laid over
    where the surface faces up (``mode="up"``: normal z from lo to hi) or
    near the ground (``mode="low"``: full below z=lo, gone above z=hi)."""
    m = base.copy()
    if name:
        m.name = name
    nt = m.node_tree
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    src = out.inputs["Surface"].links[0].from_socket
    cb = _coat_bsdf(lk, nt, coat, tile)
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(_coat_fac(nt, mode, lo, hi, patchy, patch_scale, lk.rng), mix.inputs["Fac"])
    nt.links.new(src, mix.inputs[1])
    nt.links.new(cb, mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return m


def alpha_cut(bsdf) -> bool:
    """A material is alpha-cut (leaves, cards) when its Principled BSDF's
    Alpha is wired or below 1 -- how the cloud reads it
    (``build_course._alpha_source``). Not ``blend_method``: in Blender
    4.2+ every opaque material, fresh or imported, reports ``HASHED``, so
    the old test left crown stems unrebarked, hue-shifted autumn bark to
    purple and never snowed on a rock (review of cd304aa)."""
    a = bsdf.inputs["Alpha"]
    return bool(a.is_linked or a.default_value < 0.999)


def coat_piece(ctx, pkey, coat_, tile=1.0, mode="up", lo=0.45, hi=0.8, patchy=0.4, patch_scale=1.0):
    """Coat a library piece's own (opaque) materials in place -- snow on a
    rock set, moss on a stump. Alpha-cut materials are left alone (the
    cloud reads their alpha from the first BSDF)."""
    p = ctx.piece(pkey)
    lk = ctx.looks
    for o in p["coll"].all_objects:
        for slot in getattr(o, "material_slots", []):
            m = slot.material
            if m is None or not m.use_nodes or m.get("coated"):
                continue
            bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None or alpha_cut(bsdf):
                continue
            nt = m.node_tree
            out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
            if out is None or not out.inputs["Surface"].is_linked:
                continue
            src = out.inputs["Surface"].links[0].from_socket
            cb = _coat_bsdf(lk, nt, coat_, tile)
            mix = nt.nodes.new("ShaderNodeMixShader")
            nt.links.new(_coat_fac(nt, mode, lo, hi, patchy, patch_scale, lk.rng), mix.inputs["Fac"])
            nt.links.new(src, mix.inputs[1])
            nt.links.new(cb, mix.inputs[2])
            nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
            m["coated"] = 1


def rebark_piece(ctx, pkey, mat):
    """Give library tree ``pkey``'s opaque materials (its stem and limbs)
    material ``mat`` -- the bark of the procedural trunk a crown sits on, so
    trunk and crown stem are one tree, not a pale pole with a dark stub on
    top. Alpha-cut materials (the leaves) are left alone. Changes the piece
    for every instance."""
    p = ctx.piece(pkey)
    for o in p["coll"].all_objects:
        for slot in getattr(o, "material_slots", []):
            m = slot.material
            if m is None or m is mat or not m.use_nodes:
                continue
            bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None or alpha_cut(bsdf):
                continue
            slot.material = mat


def tint_piece(ctx, pkey, rgb, amount=1.0):
    """Multiply a library piece's base colours by ``rgb`` (autumn foliage,
    dried grass, burnt stumps). Changes the piece for every instance."""
    p = ctx.piece(pkey)
    for o in p["coll"].all_objects:
        for slot in getattr(o, "material_slots", []):
            m = slot.material
            if m is None or not m.use_nodes or m.get("tinted"):
                continue
            nt = m.node_tree
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                continue
            inp = bsdf.inputs["Base Color"]
            mt = nt.nodes.new("ShaderNodeMix")
            mt.data_type = "RGBA"
            mt.blend_type = "MULTIPLY"
            mt.inputs["Factor"].default_value = amount
            mt.inputs[7].default_value = (*rgb, 1.0)
            if inp.is_linked:
                nt.links.new(inp.links[0].from_socket, mt.inputs[6])
            else:
                mt.inputs[6].default_value = inp.default_value
            nt.links.new(mt.outputs[2], inp)
            m["tinted"] = 1


# --- trees -----------------------------------------------------------------------------------

def trunk_axis(xy, h, lean=(0.0, 0.0), bend=0.0, bdir=(1.0, 0.0), z=0.0) -> np.ndarray:
    """Where the centre line of a ``trunk`` (same ``xy, h, lean, bend, bdir``)
    passes at height ``z`` above its foot: a crown set on a bowed trunk sits
    there, not over the trunk's top."""
    t = min(max(z / h, 0.0), 1.0)
    return np.asarray(xy, float)[:2] + np.asarray(lean, float) * z + \
        np.asarray(bdir, float) * bend * math.sin(math.pi * t)


def trunk(ctx, xy, h, r0, mat, *, r_top=None, lean=(0.0, 0.0), bend=0.0, flare=0.35, segs=14, lobes=5,
          rough=0.05, cap=True, coll="structure", name="trunk", z0=0.0, collide=None, bdir=None):
    """A tree trunk from (xy, z0) up ``h`` metres: radius ``r0`` at breast
    height tapering to ``r_top``, a lobed root flare (``flare`` extra radius
    at the ground), a lean (dx, dy per metre) and a bow (``bend`` metres at
    mid height, toward the unit xy ``bdir``: drawn when None; ``trunk_axis``
    gives the centre line). World-space vertices, smooth shaded, box-mapped
    bark."""
    rng = ctx.rng
    r_top = r0 * 0.55 if r_top is None else r_top
    zs = [0.0, 0.06, 0.16, 0.32, 0.55, 0.85]
    z = 1.3
    while z < h:
        zs.append(z)
        z += max(0.8, h / 9)
    zs = [v for v in zs if v < h - 0.05] + [h]
    th = np.linspace(0, 2 * math.pi, segs, endpoint=False)
    ph = rng.uniform(0, 2 * math.pi)
    lob = rng.uniform(0.5, 1.0, segs)
    if bdir is None:
        bdir = rng.normal(0, 1, 2)
        bdir /= np.linalg.norm(bdir) + 1e-9
    V, F = [], []
    for i, z in enumerate(zs):
        t = z / h
        c = trunk_axis(xy, h, lean, bend, bdir, z)
        r = r0 + (r_top - r0) * (t ** 0.9)
        fl = flare * math.exp(-z / 0.35)
        for j, a in enumerate(th):
            lobe = 1.0 + fl * (0.6 + 0.8 * lob[j] * max(0.0, math.cos(lobes * a + ph)))
            rr = r * lobe * (1.0 + rng.normal(0, rough))
            V.append((c[0] + rr * math.cos(a), c[1] + rr * math.sin(a), z0 + z))
    n = segs
    for i in range(len(zs) - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            F.append((a, b, b + n, a + n))
    if cap:
        top = len(zs) - 1
        V.append(tuple(np.mean(np.asarray(V[top * n:(top + 1) * n]), axis=0)))
        ci = len(V) - 1
        for j in range(n):
            F.append((top * n + j, top * n + (j + 1) % n, ci))
    return ctx.mesh(V, F, mat, coll=coll, name=name, smooth=True, collide=collide)


def root_flare(ctx, xy, r, mat, h=0.8, flare=0.35, coll="structure"):
    """A lobed root collar round the foot of a patch pole of radius ``r`` (a
    trunk does not stand in the ground like a post): out to <= 1.45 r at the
    ground, 1.16 r at 0.3 m, 1.07 r at 0.55 m, flush with the pole at ``h``
    (it only ever adds, and only below h). None -- no collar -- where it
    would reach into the start or goal disc (a pole right behind the goal)."""
    if clear_of(ctx, xy, ("start", "goal"))[0] < 1.45 * r + 0.05:
        return None
    return trunk(ctx, xy, h, r * 0.97, mat, r_top=r * 0.97, flare=flare, rough=0.0, cap=False, coll=coll,
                 name="flare")


def limb(ctx, a, b, r0, r1, mat, segs=7, coll="scatter", name="limb", collide=None, droop=0.0):
    """A tapered branch from point a (radius r0) to b (radius r1), sagging
    ``droop`` metres at the middle."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    z = d / max(L, 1e-9)
    x = np.cross(z, [0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.cross(z, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    k = 4 if L > 1.5 else 2
    V, F = [], []
    th = np.linspace(0, 2 * math.pi, segs, endpoint=False)
    for i in range(k + 1):
        t = i / k
        c = a + d * t - np.array([0, 0, droop * 4 * t * (1 - t)])
        r = r0 + (r1 - r0) * t
        for q in th:
            V.append(tuple(c + r * (math.cos(q) * x + math.sin(q) * y)))
    for i in range(k):
        for j in range(segs):
            p, q = i * segs + j, i * segs + (j + 1) % segs
            F.append((p, q, q + segs, p + segs))
    V.append(tuple(a)); V.append(tuple(b - np.array([0, 0, 0.0])))
    ia, ib = len(V) - 2, len(V) - 1
    for j in range(segs):
        F.append(((j + 1) % segs, j, ia))
        F.append((k * segs + j, k * segs + (j + 1) % segs, ib))
    return ctx.mesh(V, F, mat, coll=coll, name=name, smooth=True, collide=collide)


#: the labeller's default cloud ceiling (``build_course.CLOUD_ZMAX``): nothing above it is in the
#: cloud. A build may cut elsewhere (``--cloud-zmax``): decide by ``cloud_top(ctx)``, not this
CLOUD_TOP = float(getattr(BC, "CLOUD_ZMAX", 4.5))


def _argv_cloud_zmax(argv=None):
    """``--cloud-zmax`` from build_course's own command line (after ``--``), or None."""
    argv = sys.argv if argv is None else argv
    if "--" not in argv:
        return None
    args = argv[argv.index("--") + 1:]
    for i, v in enumerate(args):
        if v == "--cloud-zmax" and i + 1 < len(args):
            return float(args[i + 1])
        if v.startswith("--cloud-zmax="):
            return float(v.split("=", 1)[1])
    return None


def cloud_top(ctx) -> float:
    """The cloud ceiling this build cuts at -- what decides whether a crown or
    a limb is in the cloud: ``ctx.cloud_zmax`` when the build sets it, else
    ``--cloud-zmax`` on build_course's command line (the value it will cut
    at), else ``build_course.CLOUD_ZMAX``. Never a copy frozen at import."""
    v = getattr(ctx, "cloud_zmax", None)
    if v is None:
        v = _argv_cloud_zmax()
    return float(v) if v is not None else CLOUD_TOP
_BASE: dict = {}
_VERTS: dict = {}


def _piece_verts(ctx, pkey) -> np.ndarray:
    """The piece's vertices relative to its instance origin (footprint centre, base at z = 0)."""
    if pkey not in _VERTS:
        p = ctx.piece(pkey)
        lo = np.array(p["coll"].instance_offset)
        pts = []
        for o in p["coll"].all_objects:
            if o.type != "MESH":
                continue
            V = np.empty(len(o.data.vertices) * 3, np.float64)
            o.data.vertices.foreach_get("co", V)
            M = np.array(o.matrix_world)
            pts.append(V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3] - lo)
        _VERTS[pkey] = np.concatenate(pts) if pts else np.zeros((0, 3))
    return _VERTS[pkey]


def base_offset(ctx, pkey) -> np.ndarray:
    """Where a library piece's stem stands, relative to its footprint centre
    (the instance origin): the mean xy of its vertices within 25 cm of its
    base. A leaning or one-sided tree's bbox centre is up to a metre off."""
    if pkey in _BASE:
        return _BASE[pkey]
    P = _piece_verts(ctx, pkey)
    P = P[P[:, 2] < 0.25]
    off = P[:, :2].mean(0) if len(P) else np.zeros(2)
    _BASE[pkey] = off
    return off


_FOL: dict = {}


def foliage_z(ctx, pkey, s, stem_r=0.3) -> float:
    """Height above its base (world metres, at scale ``s``) where library tree
    ``pkey`` first reaches more than ``stem_r`` from its stem: below it the
    piece is only its stem, which a procedural trunk covers. (Per piece the
    vertices are sorted by distance from the stem once, with the running
    minimum height from the far end, so a query is a bisection.)"""
    if pkey not in _FOL:
        V = _piece_verts(ctx, pkey)
        off = base_offset(ctx, pkey)
        rho = np.hypot(V[:, 0] - off[0], V[:, 1] - off[1])
        o = np.argsort(rho)
        _FOL[pkey] = (rho[o], np.minimum.accumulate(V[o, 2][::-1])[::-1], float(V[:, 2].max()))
    rho, zmin, top = _FOL[pkey]
    i = int(np.searchsorted(rho, stem_r / s, side="right"))
    return float(zmin[i]) * s if i < len(rho) else top * s


def crown_collides(ctx, pkey, z, s, covered_to=None, cover_r=0.0) -> bool | None:
    """The ``collide`` tag for crown ``pkey`` at base height ``z``, scale
    ``s``: False (render only) when nothing of it is below the cloud ceiling
    ``cloud_top(ctx)`` outside its trunk -- it starts above the ceiling, or
    the trunk under it (up to ``covered_to``) fills ``cover_r`` metres round
    the crown's stem axis everywhere from ``z`` to the ceiling and the crown
    reaches no farther than that from its stem below the ceiling
    (``foliage_z``). Else None: the footprint decides (in the zone: collides,
    realised for the cloud as rendered). ``cover_r`` must be what the trunk
    really fills (its radius less bow, lean and roughness), not a guess: the
    old fixed 0.3 m left crown stems and first limbs rendered 0.1-0.28 m
    outside 0.11-0.17 m trunk tops and missing from the cloud (review of
    cd304aa). At 0 (the default) a crown is render-only only above the
    ceiling."""
    top = cloud_top(ctx)
    if z >= top:
        return False
    if covered_to is not None and covered_to >= top and cover_r > 0 and z >= 1.0 and \
            z + foliage_z(ctx, pkey, s, stem_r=cover_r) >= top:
        return False
    return None


def crown(ctx, pkey, xy, z, s, yaw=None, coll="scatter", covered_to=None, cover_r=0.0):
    """Put library tree ``pkey`` with its base at height ``z`` over ``xy``
    (its own stem on ``xy``). Render-only when ``crown_collides`` says so --
    the cloud keeps nothing above the ceiling anyway, and realising a
    2 M-triangle crown per tree for nothing is what made builds slow; a crown
    reaching below the ceiling collides (in the cloud, as rendered). Kits
    set a crown's base above ``cloud_top(ctx)`` to keep it render-only."""
    yaw = float(ctx.rng.uniform(0, 2 * math.pi)) if yaw is None else yaw
    off = base_offset(ctx, pkey) * s                  # put the tree's own stem, not its bbox centre, on xy
    cy, sy = math.cos(yaw), math.sin(yaw)
    xy = np.asarray(xy, float)[:2] - np.array([cy * off[0] - sy * off[1], sy * off[0] + cy * off[1]])
    collide = crown_collides(ctx, pkey, z, s, covered_to, cover_r)
    return ctx.instance(pkey, xy, z=z, yaw=yaw, s=s, coll=coll, collide=collide)


def flat_z(ctx, base=0.003, step=8e-5) -> float:
    """The next height for a flat ground overlay, each one its own: two
    overlapping overlays at exactly one height render black where they
    overlap in Cycles (each shadows the other: the r3 park preview's winter
    lawn islands). Later overlays lie on top; stays under the 5 cm the cloud
    ignores."""
    k = getattr(ctx, "_flat_k", 0)
    ctx._flat_k = k + 1
    return min(base + step * k, 0.045)


def poly_blob(c, r, n=14, rough=0.25, rng=None):
    """A roughly round polygon (a patch of moss, a lawn bed)."""
    rng = rng or np.random.default_rng()
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rr = r * (1 + rough * rng.uniform(-1, 1, n))
    return [(c[0] + a * math.cos(t), c[1] + a * math.sin(t)) for a, t in zip(rr, th)]


def ribbon(ctx, pts, width, mat, z=0.004, jitter=0.15, name="trail", rng=None):
    """A flat, non-colliding band along the polyline ``pts`` (a path),
    ``width`` wide with ragged edges; one mesh, so no overlaps flicker."""
    rng = rng or ctx.rng
    P = np.asarray(pts, float)
    V, F = [], []
    for i, p in enumerate(P):
        d = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
        d /= np.linalg.norm(d) + 1e-9
        nrm = np.array([-d[1], d[0]])
        w = width / 2 * (1 + rng.uniform(-jitter, jitter))
        w2 = width / 2 * (1 + rng.uniform(-jitter, jitter))
        V.append((*(p + nrm * w), z))
        V.append((*(p - nrm * w2), z))
    for i in range(len(P) - 1):
        F.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
    return ctx.mesh(V, F, mat, coll="ground", collide=False, name=name)


def wander(a, b, rng, amp=2.0, step=1.0, waves=2.0):
    """A meandering polyline from a to b (a path through the trees)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    d = (b - a) / max(L, 1e-9)
    nrm = np.array([-d[1], d[0]])
    ph = rng.uniform(0, 2 * math.pi, 2)
    out = []
    for t in np.linspace(0, 1, max(3, int(L / step))):
        off = amp * (0.7 * math.sin(2 * math.pi * waves * t + ph[0]) + 0.3 * math.sin(2 * math.pi * 2.3 * waves * t + ph[1]))
        out.append(a + d * L * t + nrm * off)
    return np.array(out)


def hdri_pick(ctx, prefer=(), **query):
    """An HDRI from ``prefer`` (keys) that exist, else from the query."""
    keys = [k for k in prefer if k in ctx.lib.items]
    if not keys:
        keys = ctx.lib.hdris(**query)
    return ctx.lib.pick(ctx.rng, keys)


def mats_existing(ctx, keys):
    return [k for k in keys if k in ctx.lib.items]


def litter_materials(lib) -> list[str]:
    """The leaf-litter scans that are cut-outs (an ``alpha`` map): the only
    ones a litter card may use. ScatteredLeaves008/009 are full-coverage
    carpets with no alpha -- under ``soft_decal``'s fade each card renders
    as an opaque disc of flat leaf texture (yellow ellipses in a broadleaf
    forest, review of cd304aa)."""
    return [k for k in lib.materials(cats_any={"scattered leaves"}) if "alpha" in (lib.items[k].get("maps") or {})]


# --- merged props ------------------------------------------------------------------------------

class Buf:
    """Many boxes/tubes/rings merged into one ``ctx.mesh`` (a bench is one
    object, not thirty), with a material slot per part and smooth shading
    only where a part asks for it."""

    def __init__(self):
        self.V, self.F, self.M, self.S = [], [], [], []

    def _add(self, V, F, m, smooth):
        o = len(self.V)
        self.V.extend(tuple(map(float, v)) for v in V)
        for f in F:
            self.F.append(tuple(i + o for i in f))
            self.M.append(m)
            self.S.append(smooth)

    def box(self, c, half, yaw=0.0, R=None, m=0):
        if R is None:
            cy, sy = math.cos(yaw), math.sin(yaw)
            R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
        c, h = np.asarray(c, float), np.asarray(half, float)
        V = [c + R @ (h * np.array(s)) for s in ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                                                   (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))]
        F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        self._add(V, F, m, False)

    def beam(self, a, b, hw, ht, up=(0, 0, 1), m=0):
        """A box from a to b (length axis), ``hw`` half-width, ``ht`` half-thickness along ``up``."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        x = (b - a) / max(L, 1e-9)
        z = np.asarray(up, float) - x * float(np.dot(up, x))
        if np.linalg.norm(z) < 1e-6:
            z = np.cross(x, [1.0, 0, 0]) if abs(x[0]) < 0.9 else np.cross(x, [0, 1.0, 0])
        z /= np.linalg.norm(z)
        y = np.cross(z, x)
        self.box((a + b) / 2, (L / 2, hw, ht), R=np.stack([x, y, z], 1), m=m)

    def tube(self, a, b, r, seg=10, m=0, r1=None, cap=True):
        a, b = np.asarray(a, float), np.asarray(b, float)
        r1 = r if r1 is None else r1
        d = b - a
        L = float(np.linalg.norm(d))
        z = d / max(L, 1e-9)
        x = np.cross(z, [0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.cross(z, [1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        th = np.linspace(0, 2 * math.pi, seg, endpoint=False)
        V = [a + r * (math.cos(t) * x + math.sin(t) * y) for t in th] + \
            [b + r1 * (math.cos(t) * x + math.sin(t) * y) for t in th]
        F = [(j, (j + 1) % seg, seg + (j + 1) % seg, seg + j) for j in range(seg)]
        self._add(V, F, m, True)
        if cap:
            o = len(self.V) - 2 * seg
            self.F.append(tuple(o + j for j in reversed(range(seg)))); self.M.append(m); self.S.append(False)
            self.F.append(tuple(o + seg + j for j in range(seg))); self.M.append(m); self.S.append(False)

    def ring(self, c, r_out, r_in, z0, z1, seg=48, m=0):
        """An annular wall (a fountain basin, a tree grate surround)."""
        th = np.linspace(0, 2 * math.pi, seg, endpoint=False)
        V = []
        for r, z in ((r_out, z0), (r_out, z1), (r_in, z1), (r_in, z0)):
            V += [(c[0] + r * math.cos(t), c[1] + r * math.sin(t), z) for t in th]
        F = []
        for k in range(4):
            for j in range(seg):
                a0, a1 = k * seg + j, k * seg + (j + 1) % seg
                b0, b1 = ((k + 1) % 4) * seg + j, ((k + 1) % 4) * seg + (j + 1) % seg
                F.append((a0, a1, b1, b0))
        self._add(V, F, m, False)

    def disc(self, c, r, z, seg=48, m=0):
        th = np.linspace(0, 2 * math.pi, seg, endpoint=False)
        self._add([(c[0] + r * math.cos(t), c[1] + r * math.sin(t), z) for t in th], [tuple(range(seg))], m, False)

    def flush(self, ctx, mats, coll="scatter", name="prop", collide=None):
        if not self.F:
            return None
        mats = mats if isinstance(mats, (list, tuple)) else [mats]
        ob = ctx.mesh(self.V, self.F, mats[0], coll=coll, name=name, collide=collide)
        for m in mats[1:]:
            ob.data.materials.append(m)
        for p, mi, sm in zip(ob.data.polygons, self.M, self.S):
            p.material_index = min(mi, len(mats) - 1)
            p.use_smooth = sm
        return ob


def bare_tree(ctx, xy, h, r0, mat, *, depth=3, spread=0.9, coll="structure", name="baretree", clear_z=2.2,
              n_main=None, rng=None, reach_ok=None):
    """A leafless deciduous tree (winter park, dead or burnt wood): trunk to
    a fork at ~``clear_z``..0.5 h, then limbs branching ``depth`` levels,
    all in one mesh. The limbs a drone meets are the ones below the cloud top: their
    largest xy distance from the trunk is the record's ``reach``, and
    ``reach_ok(reach) -> bool`` can veto the tree before it is built (None)."""
    rng = rng or ctx.rng
    buf = Buf()
    fork = float(rng.uniform(clear_z, max(clear_z + 0.5, 0.45 * h)))
    lean = rng.normal(0, 0.015, 2)
    base = np.array([xy[0], xy[1], 0.0])
    top = base + np.array([lean[0] * fork, lean[1] * fork, fork])
    # the trunk as a flared tube pair
    buf.tube(base + [0, 0, -0.05], base + [0, 0, 0.5], r0 * 1.35, seg=12, r1=r0 * 1.02, cap=False)
    buf.tube(base + [0, 0, 0.5], top, r0, seg=12, r1=r0 * 0.8, cap=False)

    def grow(p, d, r, L, lvl):
        q = p + d * L
        buf.tube(p, q, r, seg=8 if lvl < 2 else 5, r1=r * 0.62, cap=lvl == depth)
        if lvl >= depth:
            return
        n = int(rng.integers(2, 4))
        for _ in range(n):
            nd = d + rng.normal(0, spread * 0.6, 3)
            nd[2] = abs(nd[2]) * 0.7 + 0.25
            nd /= np.linalg.norm(nd)
            grow(q - d * L * float(rng.uniform(0.0, 0.3)), nd, r * 0.6, L * float(rng.uniform(0.55, 0.75)),
                 lvl + 1)

    n_main = n_main or int(rng.integers(3, 6))
    Lm = max(1.2, (h - fork) * 0.55)
    for k in range(n_main):
        a = 2 * math.pi * k / n_main + float(rng.normal(0, 0.4))
        d = np.array([math.cos(a) * spread, math.sin(a) * spread, float(rng.uniform(0.8, 1.6))])
        d /= np.linalg.norm(d)
        grow(top - [0, 0, float(rng.uniform(0, 0.4))], d, r0 * 0.62, Lm * float(rng.uniform(0.8, 1.1)), 1)
    buf.tube(top, top + np.array([lean[0], lean[1], 1.0]) * (h - fork) * 0.7, r0 * 0.7, seg=8, r1=r0 * 0.2)
    V = np.asarray(buf.V)
    low = V[V[:, 2] <= cloud_top(ctx) + 0.2]
    reach = float(np.hypot(low[:, 0] - xy[0], low[:, 1] - xy[1]).max()) if len(low) else r0
    if reach_ok is not None and not reach_ok(reach):
        return None
    ob = buf.flush(ctx, mat, coll=coll, name=name)
    return {"xy": np.asarray(xy, float), "r": r0, "h": h, "obj": ob, "reach": reach}


def bare_top(ctx, xy, z0, r0, mat, *, h=None, spread=0.9, coll="structure", name="baretop", rng=None):
    """The leafless crown of a tree whose trunk is a patch pole ending at
    ``z0``: the trunk tapers on up and forks into limbs that only grow
    upward from ``z0`` -- nothing below ``z0``, so a pole reaching the cloud
    ceiling stays the family's pole in the cloud."""
    rng = rng or ctx.rng
    h = float(rng.uniform(3.0, 6.0)) if h is None else h
    buf = Buf()
    base = np.array([xy[0], xy[1], z0 - 0.02])
    fork = base + np.array([0.0, 0.0, float(rng.uniform(0.3, 0.6)) * h])
    buf.tube(base, fork, r0 * 0.97, seg=14, r1=r0 * 0.75, cap=False)

    def grow(p, d, r, L, lvl):
        q = p + d * L
        buf.tube(p, q, r, seg=8 if lvl < 2 else 5, r1=r * 0.62, cap=lvl == 3)
        if lvl >= 3:
            return
        for _ in range(int(rng.integers(2, 4))):
            nd = d + rng.normal(0, spread * 0.6, 3)
            nd[2] = abs(nd[2]) * 0.7 + 0.3
            nd /= np.linalg.norm(nd)
            grow(q - d * L * float(rng.uniform(0.0, 0.3)), nd, r * 0.6, L * float(rng.uniform(0.55, 0.75)), lvl + 1)

    n_main = int(rng.integers(3, 6))
    for k in range(n_main):
        a = 2 * math.pi * k / n_main + float(rng.normal(0, 0.4))
        d = np.array([math.cos(a) * spread, math.sin(a) * spread, float(rng.uniform(0.9, 1.6))])
        grow(fork, d / np.linalg.norm(d), r0 * 0.5, max(1.2, 0.5 * h) * float(rng.uniform(0.8, 1.1)), 1)
    buf.tube(fork, fork + np.array([0.0, 0.0, 0.6 * h]), r0 * 0.6, seg=8, r1=r0 * 0.15)
    return buf.flush(ctx, mat, coll=coll, name=name)


def autumn_piece(ctx, pkey, hue=0.4, sat=1.1, val=1.0):
    """Turn a library tree's leaves (its alpha-cut materials) toward autumn:
    Blender hue 0.5 = unchanged, 0.4 pushes green toward yellow/orange."""
    p = ctx.piece(pkey)
    for o in p["coll"].all_objects:
        for slot in getattr(o, "material_slots", []):
            m = slot.material
            if m is None or not m.use_nodes or m.get("autumn"):
                continue
            nt = m.node_tree
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None or not alpha_cut(bsdf):
                continue
            inp = bsdf.inputs["Base Color"]
            if not inp.is_linked:
                continue
            hs = nt.nodes.new("ShaderNodeHueSaturation")
            hs.inputs["Hue"].default_value = hue
            hs.inputs["Saturation"].default_value = sat
            hs.inputs["Value"].default_value = val
            nt.links.new(inp.links[0].from_socket, hs.inputs["Color"])
            nt.links.new(hs.outputs["Color"], inp)
            m["autumn"] = 1


def spline(anchors, step=0.8, rng=None, wobble=0.0):
    """A smooth Catmull-Rom polyline through ``anchors`` (a trail, a park
    path), sampled every ~``step`` m, with an optional small lateral wobble."""
    A = np.asarray(anchors, float)
    A = np.vstack([2 * A[0] - A[1], A, 2 * A[-1] - A[-2]])
    out = []
    for i in range(1, len(A) - 2):
        p0, p1, p2, p3 = A[i - 1], A[i], A[i + 1], A[i + 2]
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(A[-2])
    P = np.array(out)
    if wobble > 0 and rng is not None and len(P) > 2:
        d = np.gradient(P, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        nrm = np.column_stack([-d[:, 1], d[:, 0]])
        ph = rng.uniform(0, 6.283, 2)
        s = np.arange(len(P)) * step
        P = P + nrm * (wobble * (np.sin(s / 3.1 + ph[0]) * 0.6 + np.sin(s / 1.3 + ph[1]) * 0.4))[:, None]
    return P


def soft_decal(m, inner=0.15, outer=0.5, ragged=0.22):
    """Fade a decal material's alpha to nothing toward the edge of its UV
    square (radius ``inner`` .. ``outer`` in UV units, the rim roughened by
    noise), so a leaf-litter card reads as a drift, not a square. Only for
    non-colliding ground decals: the cloud reads a collider's alpha through
    its first BSDF, and this would change it."""
    if m.get("soft"):
        return m
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = next((n for n in N if n.type == "BSDF_PRINCIPLED"), None)
    if b is None:
        return m
    tc = N.new("ShaderNodeTexCoord")
    d = N.new("ShaderNodeVectorMath")
    d.operation = "DISTANCE"
    d.inputs[1].default_value = (0.5, 0.5, 0.0)
    L.new(tc.outputs["UV"], d.inputs[0])
    nz = N.new("ShaderNodeTexNoise")                   # a low-frequency wobble: a drift, not a coin
    nz.inputs["Scale"].default_value = 2.5
    nz.inputs["Detail"].default_value = 4.0
    nz.inputs["Roughness"].default_value = 0.6
    L.new(tc.outputs["UV"], nz.inputs["Vector"])
    wob = N.new("ShaderNodeMapRange")
    wob.inputs["To Min"].default_value, wob.inputs["To Max"].default_value = -ragged, ragged
    L.new(nz.outputs["Fac"], wob.inputs["Value"])
    add = N.new("ShaderNodeMath")
    add.operation = "ADD"
    L.new(d.outputs["Value"], add.inputs[0])
    L.new(wob.outputs["Result"], add.inputs[1])
    fade = N.new("ShaderNodeMapRange")
    fade.interpolation_type = "SMOOTHSTEP"
    fade.inputs["From Min"].default_value, fade.inputs["From Max"].default_value = inner, outer
    fade.inputs["To Min"].default_value, fade.inputs["To Max"].default_value = 1.0, 0.0
    L.new(add.outputs["Value"], fade.inputs["Value"])
    a = b.inputs["Alpha"]
    if a.is_linked:
        mul = N.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        L.new(a.links[0].from_socket, mul.inputs[0])
        L.new(fade.outputs["Result"], mul.inputs[1])
        L.new(mul.outputs["Value"], a)
    else:
        L.new(fade.outputs["Result"], a)
    m["soft"] = 1
    return m


def matte(m, lo=0.75):
    """Raise every Principled BSDF's roughness in material ``m`` to at least
    ``lo`` (a lawn scan's roughness map makes grass shine like plastic at a
    grazing angle)."""
    nt = m.node_tree
    for b in [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"]:
        inp = b.inputs["Roughness"]
        if inp.is_linked:
            src = inp.links[0].from_socket
            mr = nt.nodes.new("ShaderNodeMapRange")
            mr.inputs["To Min"].default_value = lo
            nt.links.new(src, mr.inputs["Value"])
            nt.links.new(mr.outputs["Result"], inp)
        else:
            inp.default_value = max(inp.default_value, lo)
        if "Specular IOR Level" in b.inputs:
            b.inputs["Specular IOR Level"].default_value = 0.3
    return m


def foliage(name, rng, dark=(0.018, 0.05, 0.014), mid=(0.05, 0.12, 0.03), light=(0.13, 0.2, 0.05),
            leaf=0.045, bevel=0.12):
    """A clipped-hedge / topiary surface without leaf geometry: every Voronoi
    cell is a leaf (its own shade between ``dark`` and ``light``), bumped
    into a leafy relief, patchy at the metre scale, edges rounded by
    ``bevel`` -- reads as privet or box from a few metres away."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    geo = N.new("ShaderNodeNewGeometry")
    vo = N.new("ShaderNodeTexVoronoi")
    vo.voronoi_dimensions = "3D"
    vo.inputs["Scale"].default_value = 1.0 / leaf
    vo.inputs["Randomness"].default_value = 1.0
    L.new(geo.outputs["Position"], vo.inputs["Vector"])
    sep = N.new("ShaderNodeSeparateColor")
    L.new(vo.outputs["Color"], sep.inputs["Color"])
    ramp = N.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position, cr.elements[0].color = 0.0, (*dark, 1)
    cr.elements[1].position, cr.elements[1].color = 1.0, (*light, 1)
    e = cr.elements.new(0.55)
    e.color = (*mid, 1)
    L.new(sep.outputs[0], ramp.inputs["Fac"])
    # metre-scale patchiness (sun-bleached, thin spots)
    nz = N.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 0.9
    nz.inputs["Detail"].default_value = 3.0
    mp = N.new("ShaderNodeMapping")
    mp.inputs["Location"].default_value = tuple(rng.uniform(-30, 30, 3))
    L.new(geo.outputs["Position"], mp.inputs["Vector"])
    L.new(mp.outputs["Vector"], nz.inputs["Vector"])
    mr = N.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 0.55, 1.35
    L.new(nz.outputs["Fac"], mr.inputs["Value"])
    mul = N.new("ShaderNodeMix")
    mul.data_type, mul.blend_type = "RGBA", "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    L.new(ramp.outputs["Color"], mul.inputs[6])
    L.new(mr.outputs["Result"], mul.inputs[7])
    # deep gaps between leaves are dark
    gap = N.new("ShaderNodeMapRange")
    gap.inputs["From Min"].default_value, gap.inputs["From Max"].default_value = 0.0, 0.5
    gap.inputs["To Min"].default_value, gap.inputs["To Max"].default_value = 0.25, 1.0
    L.new(vo.outputs["Distance"], gap.inputs["Value"])
    inv = N.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.25
    L.new(gap.outputs["Result"], inv.inputs[1])
    dk = N.new("ShaderNodeMix")
    dk.data_type, dk.blend_type = "RGBA", "MULTIPLY"
    dk.inputs["Factor"].default_value = 1.0
    L.new(mul.outputs[2], dk.inputs[6])
    L.new(inv.outputs["Value"], dk.inputs[7])
    L.new(dk.outputs[2], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.6
    # relief: leaves as domes + fine noise, on a bevelled normal
    nz2 = N.new("ShaderNodeTexNoise")
    nz2.inputs["Scale"].default_value = 1.0 / (leaf * 3)
    nz2.inputs["Detail"].default_value = 6.0
    L.new(geo.outputs["Position"], nz2.inputs["Vector"])
    h = N.new("ShaderNodeMath")
    h.operation = "ADD"
    L.new(inv.outputs["Value"], h.inputs[0])
    L.new(nz2.outputs["Fac"], h.inputs[1])
    bp = N.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = 0.9
    bp.inputs["Distance"].default_value = leaf * 0.6
    L.new(h.outputs["Value"], bp.inputs["Height"])
    if bevel > 0:
        bv = N.new("ShaderNodeBevel")
        bv.samples = 6
        bv.inputs["Radius"].default_value = bevel
        L.new(bv.outputs["Normal"], bp.inputs["Normal"])
    L.new(bp.outputs["Normal"], b.inputs["Normal"])
    return m


# --- fast placement (the Ctx versions loop over every footprint in Python) --------------------------

def free_rect(ctx, c, half, yaw=0.0, margin=0.3, step=0.5) -> bool:
    """``Ctx.free_rect`` with one vectorised clearance call."""
    c = np.asarray(c, float)[:2]
    hx, hy = half[0], half[1]
    cy, sy = math.cos(yaw), math.sin(yaw)
    R = np.array([[cy, -sy], [sy, cy]])
    us = np.linspace(-hx, hx, max(2, int(2 * hx / step) + 1))
    vs = np.linspace(-hy, hy, max(2, int(2 * hy / step) + 1))
    U_, V_ = np.meshgrid(us, vs)
    P = c + np.column_stack([U_.ravel(), V_.ravel()]) @ R.T
    return bool(clear_many(ctx, P).min() >= margin)


def scatter(ctx, pkeys, n, rect=None, margin=0.5, s=(0.9, 1.1), where=None, coll="scatter", batch=48, tries=6):
    """``Ctx.scatter`` with batched clearance: each placement draws up to
    ``tries`` x ``batch`` candidates and takes the first that is clear; it
    claims its footprint like ``Ctx.scatter``."""
    rng = ctx.rng
    out = []
    if not pkeys:
        return out
    x0, y0, x1, y1 = rect or ctx.bounds
    for _ in range(n):
        k = ctx.lib.pick(rng, pkeys)
        sc = float(rng.uniform(*s)) if isinstance(s, tuple) else float(s)
        sx, sy, _ = np.asarray(ctx.lib.size(k)) * sc
        r = 0.5 * math.hypot(sx, sy)
        pad = min(r, 1.0)
        for _t in range(tries):
            P = np.column_stack([rng.uniform(x0 + pad, x1 - pad, batch), rng.uniform(y0 + pad, y1 - pad, batch)])
            ok = clear_many(ctx, P) >= r + margin
            if where is not None:
                ok &= np.array([bool(where(p)) for p in P])
            idx = np.flatnonzero(ok)
            if len(idx):
                xy = P[idx[0]]
                out.append(ctx.instance(k, xy, 0.0, float(rng.uniform(0, 2 * math.pi)), sc, coll=coll, claim=True))
                break
    return out


def claim_polyline(ctx, pts, half_w, seg=3.0, kind="path"):
    """Claim a band along a polyline in ~``seg`` m pieces (fewer footprints)."""
    P = np.asarray(pts, float)
    a = P[0]
    acc = 0.0
    for i in range(1, len(P)):
        acc += float(np.linalg.norm(P[i] - P[i - 1]))
        if acc >= seg or i == len(P) - 1:
            b = P[i]
            if ctx.rect_meets_zone(np.minimum(a, b) - half_w, np.maximum(a, b) + half_w):
                d = b - a
                L = float(np.linalg.norm(d))
                if L > 1e-6:
                    ctx.claim_rect((a + b) / 2, (L / 2 + 0.1, half_w), math.atan2(d[1], d[0]), kind=kind)
            a, acc = b, 0.0
