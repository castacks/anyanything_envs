"""Flight copy of a nature_box scene, for renderers that fly many frames through one scene (the closed-loop bench,
timing): what box_views.py adds at render time, Infinigen's time dependence baked, and far fewer instances.

    python infinigen/tools/flight_copy.py --scene S/scene/scene.blend --box S/box.json --band S/views_ae/band.json \
        --out S/flight/scene_flight.blend [--keep-water] [--no-cut]

Why: Cycles walks every depsgraph instance on every frame, persistent data or not (~13 us each on gs2's CPU), and an
Infinigen square carries millions. plain 201 has 3.2 M grass tufts and 0.23 M twigs, fruit and branches on its trees;
box_views' backdrop adds 1.6 M (its trees are full copies of the scene's own, ~12k instances each); forest 222's
backdrop alone is 5.9 M. A flight frame of plain 201 spent 50 s in that sync on gs2 (1 sample, 64 x 48).

What the copy holds (--no-cut keeps 1 only, the like-for-like reference):
1. The backdrop, water and terrain shading as box_views.py renders them (water hidden unless --keep-water, which
   re-shades it); drivers, Scene Time nodes and object animation baked at the current frame (tools/freeze.py).
2. Dense scatters (>= --patch-min instances: grass) tiled as patches: the scatter's points binned into cells of ~--patch
   points, --patches prototype meshes realized from real cells (decimated copies of the scatter's assets, positions
   relative to the cell's terrain frame), one patch instance per occupied cell, picked by point count and turned by a
   multiple of 90 degrees about the local terrain normal. The density field and its mask follow the original at the
   cell scale (~0.16 m for plain 201's grass); 3.2 M instances become ~0.1 M.
3. The backdrop's trees: copies of --backdrop-protos of the scene's trees whose twigs are thinned like 4 (15 % kept, 8 %
   beyond 50 m, where half the backdrop trees are dropped), instead of the scene's full trees (~12k instances each).
4. Trees and bushes standing more than 3 m outside the flight box keep half their twigs, more than 8 m a third (--thin),
   the kept ones scaled up by p^-1/2 so the canopy keeps its cover. Trees in and near the box are untouched.

Looks (box_views' A1-A3) are applied at render time and are not baked; the copy renders A0.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_views as FV  # noqa: E402
from freeze import freeze_objects, freeze_time  # noqa: E402


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--band", type=Path, required=True, help="views band.json (backdrop tree density)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-water", action="store_true")
    ap.add_argument("--no-backdrop", action="store_true", help="the scene already has one (a scene_wires.blend)")
    ap.add_argument("--no-cut", action="store_true", help="backdrop/water/shading/freeze only (reference copy)")
    ap.add_argument("--patch", type=int, default=64, help="points per patch cell")
    ap.add_argument("--patch-min", type=int, default=20000, help="scatters with at least this many instances")
    ap.add_argument("--patches", type=int, default=48, help="prototype patches per scatter")
    ap.add_argument("--asset-polys", type=int, default=300, help="decimate a scatter asset to this many faces")
    ap.add_argument("--backdrop-protos", type=int, default=6)
    ap.add_argument("--backdrop-thin", type=float, nargs=3, default=[0.15, 0.08, 0.5], metavar=("P_NEAR", "P_FAR", "KEEP"),
                    help="backdrop trees keep this fraction of their twigs (near, beyond --backdrop-far), and this "
                    "fraction of the trees beyond --backdrop-far")
    ap.add_argument("--backdrop-far", type=float, default=50.0)
    ap.add_argument("--thin", default="3:0.5,8:0.33", help="trees this far outside the box (m) keep this fraction "
                    "of their twigs: d:p,d:p ('' = none)")
    ap.add_argument("--smooth-angle", type=float, default=35.0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


# ---------------------------------------------------------------- geometry-node helpers

RAND_ATTR = "fc_rand"

def gn_group(name):
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    return ng, ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")


def instance_group(name, coll, realize):
    """Instance coll's objects (alphabetical order, as Collection Info separates them) on the points: point
    attributes pid (int), rot (XYZ Euler), scl (vector)."""
    ng, gi, go = gn_group(name)
    nodes, links = ng.nodes, ng.links
    ci = nodes.new("GeometryNodeCollectionInfo")
    ci.transform_space = "ORIGINAL"
    ci.inputs["Collection"].default_value = coll
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    iop = nodes.new("GeometryNodeInstanceOnPoints")
    iop.inputs["Pick Instance"].default_value = True

    def attr(nm, dt):
        n = nodes.new("GeometryNodeInputNamedAttribute")
        n.data_type = dt
        n.inputs["Name"].default_value = nm
        return n.outputs["Attribute"]
    links.new(gi.outputs[0], iop.inputs["Points"])
    links.new(ci.outputs["Instances"], iop.inputs["Instance"])
    links.new(attr("pid", "INT"), iop.inputs["Instance Index"])
    rot = attr("rot", "FLOAT_VECTOR")
    if iop.inputs["Rotation"].type == "ROTATION":
        e2r = nodes.new("FunctionNodeEulerToRotation")
        links.new(rot, e2r.inputs[0])
        rot = e2r.outputs[0]
    links.new(rot, iop.inputs["Rotation"])
    links.new(attr("scl", "FLOAT_VECTOR"), iop.inputs["Scale"])
    out = iop.outputs["Instances"]
    if realize:
        # each instance's own random value, kept through the realize (Object Info > Random would be the patch's)
        rnd = nodes.new("FunctionNodeRandomValue")
        rnd.data_type = "FLOAT"
        st = nodes.new("GeometryNodeStoreNamedAttribute")
        st.data_type, st.domain = "FLOAT", "INSTANCE"
        st.inputs["Name"].default_value = RAND_ATTR
        links.new(out, st.inputs["Geometry"])
        links.new(rnd.outputs["Value"], st.inputs["Value"])
        rz = nodes.new("GeometryNodeRealizeInstances")
        links.new(st.outputs["Geometry"], rz.inputs[0])
        out = rz.outputs[0]
    links.new(out, go.inputs[0])
    return ng


def per_instance_random(me):
    """Materials of a realized patch: copies whose Object Info > Random reads the stored per-instance value."""
    for i, m in enumerate(me.materials):
        if m is None or not m.use_nodes:
            continue
        mc = m.copy()
        nt = mc.node_tree
        for n in [n for n in nt.nodes if n.bl_idname == "ShaderNodeObjectInfo" and n.outputs["Random"].links]:
            at = nt.nodes.new("ShaderNodeAttribute")
            at.attribute_type, at.attribute_name = "GEOMETRY", RAND_ATTR
            for l in list(n.outputs["Random"].links):
                nt.links.new(at.outputs["Fac"], l.to_socket)
        me.materials[i] = mc


def points_object(name, P, pid, rot, scl):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(P))
    me.vertices.foreach_set("co", np.ascontiguousarray(P, np.float32).ravel())
    me.attributes.new("pid", "INT", "POINT").data.foreach_set("value", np.ascontiguousarray(pid, np.int32))
    me.attributes.new("rot", "FLOAT_VECTOR", "POINT").data.foreach_set("vector",
                                                                        np.ascontiguousarray(rot, np.float32).ravel())
    me.attributes.new("scl", "FLOAT_VECTOR", "POINT").data.foreach_set("vector",
                                                                        np.ascontiguousarray(scl, np.float32).ravel())
    me.update()
    return bpy.data.objects.new(name, me)


def hidden_collection(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    c.hide_render = c.hide_viewport = True
    return c


def evaluated_mesh(ob, name):
    """ob's evaluated geometry as a new mesh (ob linked to the scene for the evaluation, then removed)."""
    bpy.context.scene.collection.objects.link(ob)
    deps = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(deps), preserve_all_data_layers=True, depsgraph=deps)
    me.name = name
    bpy.data.objects.remove(ob)
    return me


def decimated_mesh(me, target, name):
    """A copy of mesh me with about target faces (collapse decimation); me itself when it is already that small."""
    n = len(me.polygons)
    if n <= target:
        return me
    ob = bpy.data.objects.new("fc_tmp", me)
    m = ob.modifiers.new("dec", "DECIMATE")
    m.ratio = max(target / n, 1e-5)
    return evaluated_mesh(ob, name)


def euler_xyz(R):
    """Blender XYZ Euler angles of rotation matrices R (..., 3, 3) (R = Rz Ry Rx)."""
    b = np.arcsin(np.clip(-R[..., 2, 0], -1.0, 1.0))
    a = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    c = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.stack([a, b, c], -1)


def frames(normals, turns=None):
    """Rotation matrices with z along the normals and x along world x projected onto the tangent plane, turned by
    turns * 90 deg about z."""
    z = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    ex = np.array([1.0, 0.0, 0.0])
    x = ex - (z @ ex)[:, None] * z
    bad = np.linalg.norm(x, axis=1) < 1e-6
    x[bad] = np.array([0.0, 1.0, 0.0]) - z[bad, 1:2] * z[bad]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)
    R = np.stack([x, y, z], -1)
    if turns is not None:
        t = turns * (math.pi / 2)
        c, s = np.cos(t), np.sin(t)
        Rz = np.zeros((len(t), 3, 3))
        Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = c, -s, s, c, 1.0
        R = R @ Rz
    return R


def ground_with_normal(objs):
    """(x, y) -> (z, unit normal) of the highest terrain surface, or None."""
    deps = bpy.context.evaluated_depsgraph_get()
    items = [(BVHTree.FromObject(o, deps), o.matrix_world.copy()) for o in objs]
    items = [(t, M, M.inverted(), M.to_3x3().inverted().transposed()) for t, M in items]

    def f(x, y):
        best = None
        for t, M, Mi, Mn in items:
            o = Mi @ Vector((x, y, 500.0))
            d = (Mi.to_3x3() @ Vector((0, 0, -1))).normalized()
            loc, nrm, _, _ = t.ray_cast(o, d)
            if loc is not None:
                z = (M @ loc).z
                if best is None or z > best[0]:
                    n = (Mn @ nrm).normalized()
                    best = (z, n if n.z >= 0 else -n)
        return best
    return f


# ---------------------------------------------------------------- instances

def instances_by_parent(min_count, prefix="scatter:"):
    """Two walks over the evaluated depsgraph: count instances per instancing object, then collect world matrices and
    instanced objects for the scatters with >= min_count of them. Returns (counts, {scatter name: (M (N,4,4),
    proto objects, pid (N,))})."""
    deps = bpy.context.evaluated_depsgraph_get()
    counts = {}
    for inst in deps.object_instances:
        if inst.is_instance and inst.parent is not None:
            k = inst.parent.original.name
            counts[k] = counts.get(k, 0) + 1
    dense = {k: n for k, n in counts.items() if k.startswith(prefix) and n >= min_count}
    buf = {k: [np.empty((n, 4, 4), np.float64), np.empty(n, np.int32), {}, 0] for k, n in dense.items()}
    for inst in deps.object_instances:
        if not inst.is_instance or inst.parent is None:
            continue
        b = buf.get(inst.parent.original.name)
        if b is None:
            continue
        nm = inst.object.original.name
        q = b[2].setdefault(nm, len(b[2]))
        b[0][b[3]] = inst.matrix_world
        b[1][b[3]] = q
        b[3] += 1
    res = {}
    for k, (M, pid, pmap, n) in buf.items():
        protos = [bpy.data.objects[nm] for nm, _ in sorted(pmap.items(), key=lambda kv: kv[1])]
        res[k] = (M[:n], protos, pid[:n])
    return counts, res


def patch_scatter(scatter, M, protos, pid, a, zgn, rng, half):
    """Replace scatter (an Infinigen geo_instance_scatter object) by patch instances; see the module docstring."""
    tag = scatter.name.split(":")[-1].split("(")[0][:24]
    P, A = M[:, :3, 3], M[:, :3, :3]
    # cell size from the typical point density (the scatter's density is uniform where its mask allows)
    _, c0 = np.unique(np.floor(P[:, :2] / 0.25).astype(np.int64), axis=0, return_counts=True)
    rho = float(np.median(c0)) / 0.0625
    cell = float(np.clip(math.sqrt(a.patch / rho), 0.05, 1.0))
    uij, inv, cnt = np.unique(np.floor(P[:, :2] / cell).astype(np.int64), axis=0, return_inverse=True,
                              return_counts=True)
    inv = inv.ravel()
    nc = len(uij)
    C = (uij + 0.5) * cell
    # terrain height and normal at every cell centre (fallback: the cell's points, normal up)
    Z = np.empty(nc)
    N = np.tile([0.0, 0.0, 1.0], (nc, 1))
    zsum = np.bincount(inv, weights=P[:, 2], minlength=nc)
    for c in range(nc):
        g = zgn(C[c, 0], C[c, 1])
        if g is None:
            Z[c] = zsum[c] / cnt[c]
        else:
            Z[c] = g[0]
            N[c] = g[1]
    O = np.concatenate([C, Z[:, None]], 1)
    # prototype patches: cells at evenly spaced point-count quantiles
    order = np.argsort(cnt, kind="stable")
    src = np.unique(order[np.linspace(0, nc - 1, a.patches).round().astype(int)])
    dcoll = hidden_collection(f"fc_{tag}_assets")
    for q, o in enumerate(protos):
        dob = bpy.data.objects.new(f"fc_{tag}_d{q:03d}", decimated_mesh(o.data, a.asset_polys, f"fc_{tag}_d{q:03d}"))
        dcoll.objects.link(dob)
    grp = instance_group(f"fc_{tag}_realize", dcoll, realize=True)
    pcoll = hidden_collection(f"fc_{tag}_patches")
    members = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(cnt)])
    Rs = frames(N[src])
    pcount = []
    for q, (c, R) in enumerate(zip(src, Rs)):
        idx = members[starts[c]:starts[c + 1]]
        rel = (P[idx] - O[c]) @ R                       # R^T (p - o), row vectors
        Arel = np.einsum("ji,njk->nik", R, A[idx])      # R^T A
        s = np.linalg.norm(Arel, axis=1)                # column norms = scale
        tmp = points_object("fc_tmp_pts", rel, pid[idx], euler_xyz(Arel / s[:, None, :]), s)
        tmp.modifiers.new("inst", "NODES").node_group = grp
        pme = evaluated_mesh(tmp, f"fc_{tag}_p{q:03d}")
        if q == 0:
            per_instance_random(pme)
            mats = list(pme.materials)
        else:
            for i in range(min(len(mats), len(pme.materials))):
                if pme.materials[i] is not None and pme.materials[i].name.split(".")[0] == mats[i].name.split(".")[0]:
                    pme.materials[i] = mats[i]
            if len(pme.materials) != len(mats):
                per_instance_random(pme)
        pob = bpy.data.objects.new(f"fc_{tag}_p{q:03d}", pme)
        pcoll.objects.link(pob)
        pcount.append(len(idx))
    pcount = np.array(pcount)
    # one patch per occupied cell: nearest point count, ties broken at random; a random quarter turn
    porder = np.argsort(pcount, kind="stable")
    pos = np.searchsorted(pcount[porder], cnt)
    lo, hi = np.clip(pos - 1, 0, len(porder) - 1), np.clip(pos, 0, len(porder) - 1)
    near = np.where(np.abs(pcount[porder][lo] - cnt) <= np.abs(pcount[porder][hi] - cnt), lo, hi)
    jitter = rng.integers(-1, 2, size=nc)
    near = np.clip(near + np.where(cnt > 8, jitter, 0), 0, len(porder) - 1)
    choice = porder[near]
    R = frames(N, rng.integers(0, 4, size=nc))
    ob = points_object(f"fc_{tag}_patches", O, choice, euler_xyz(R), np.ones((nc, 3)))
    ob.modifiers.new("patches", "NODES").node_group = instance_group(f"fc_{tag}_place", pcoll, realize=False)
    for c in scatter.users_collection:
        c.objects.link(ob)
    bpy.data.objects.remove(scatter)
    inbox = (np.abs(C[:, 0]) < half) & (np.abs(C[:, 1]) < half)
    return {"scatter": tag, "instances": int(len(P)), "cell_m": round(cell, 3),
            "density_per_m2": round(rho, 1), "patch_instances": int(nc), "in_box": int(inbox.sum()),
            "prototypes": int(len(src)), "faces_per_patch": int(np.mean([len(o.data.polygons)
                                                                         for o in pcoll.objects]))}


# ---------------------------------------------------------------- trees outside the box

def thin_group(p):
    """Appended after a tree's twig instancer: keep each instance (twig, leaf cluster, fruit) with probability p and
    scale the kept ones about their own origin by p^-1/2, so the canopy keeps its cover with fewer, larger twigs."""
    name = f"fc_thin_{p:.2f}"
    ng = bpy.data.node_groups.get(name)
    if ng is not None:
        return ng
    ng, gi, go = gn_group(name)
    nodes, links = ng.nodes, ng.links
    rnd = nodes.new("FunctionNodeRandomValue")
    rnd.data_type = "BOOLEAN"
    [i for i in rnd.inputs if i.name == "Probability"][0].default_value = 1.0 - p
    dg = nodes.new("GeometryNodeDeleteGeometry")
    dg.domain = "INSTANCE"
    links.new(gi.outputs[0], dg.inputs["Geometry"])
    links.new([o for o in rnd.outputs if o.type == "BOOLEAN"][0], dg.inputs["Selection"])
    sc = nodes.new("GeometryNodeScaleInstances")
    sc.inputs["Scale"].default_value = (p ** -0.5,) * 3
    links.new(dg.outputs[0], sc.inputs["Instances"])
    links.new(sc.outputs[0], go.inputs[0])
    return ng


def thin_outer_trees(half, schedule):
    """Trees and bushes whose trunk base is more than d m outside the flight box keep a fraction p of their twigs
    (schedule: [(d, p), ...], the last d passed wins). Returns {p: trees}."""
    roots = []
    for c in bpy.data.collections:
        if c.name.startswith(("unique_assets:TreeFactory", "unique_assets:BushFactory")):
            members = set(c.objects)
            roots += [o for o in c.objects if o.parent not in members]
    done = {}
    for r in roots:
        x, y = r.matrix_world.translation.x, r.matrix_world.translation.y
        d = max(abs(x), abs(y)) - half
        p = 1.0
        for dd, pp in schedule:
            if d > dd:
                p = pp
        if p >= 1.0:
            continue
        stack = list(r.children)
        while stack:
            o = stack.pop()
            stack += list(o.children)
            if any(m.type == "NODES" and m.show_render for m in o.modifiers):
                m = o.modifiers.new("fc_thin", "NODES")
                m.node_group = thin_group(p)
        done[p] = done.get(p, 0) + 1
    return done


# ---------------------------------------------------------------- backdrop trees

def thinned_copies(roots, p, tag):
    """Collections (not linked to the scene, not hidden: a collection's own "disable in renders" flag also drops it
    from collection instances) holding a copy of each root's hierarchy, data and node groups shared, every instancer
    thinned to a fraction p of its twigs (thin_group). Collection offset = the root's location, as box_views'
    prototypes. Realized low-detail trees do not work: a leafy twig asset is ~1e5 separate leaf pieces, which collapse
    decimation cannot take below one triangle each."""
    protos = []
    for k, r in enumerate(roots):
        pc = bpy.data.collections.new(f"fc_backdrop_{tag}_{k:02d}")
        copies = {}
        stack = [r]
        while stack:
            o = stack.pop()
            stack += list(o.children)
            copies[o] = c = o.copy()
            pc.objects.link(c)
        for o, c in copies.items():
            if o.parent in copies:
                c.parent = copies[o.parent]
                c.matrix_parent_inverse = o.matrix_parent_inverse.copy()
            else:
                c.parent = None
                c.matrix_world = o.matrix_world.copy()
            if c.type == "MESH" and any(m.type == "NODES" and m.show_render for m in c.modifiers):
                c.modifiers.new("fc_thin", "NODES").node_group = thin_group(p)
        pc.instance_offset = r.matrix_world.translation.copy()
        protos.append(pc)
    return protos


# ---------------------------------------------------------------- main

def main():
    a = parse()
    t0 = time.time()
    box = json.loads(a.box.read_text())
    half, region = box["half"], box["region"]
    rng = np.random.default_rng(a.seed)
    bpy.ops.wm.open_mainfile(filepath=str(a.scene))
    FV.show_render_visible()
    rec = {"scene": str(a.scene), "cut": not a.no_cut}
    for o in bpy.data.objects:
        if o.name.startswith("liquid") and o.type == "MESH":
            if a.keep_water:
                FV.water_material(o)
            else:
                o.hide_render = o.hide_viewport = True
    tobjs = FV.terrain_objects()
    if a.smooth_angle > 0 and "terrain_smooth" not in box:
        for o in tobjs:
            o.data.polygons.foreach_set("use_smooth", np.ones(len(o.data.polygons), bool))
            o.data.set_sharp_from_angle(angle=math.radians(a.smooth_angle))
    s = bpy.context.scene
    rec["frozen"] = dict(zip(("drivers", "scene_time"), freeze_time(s)))
    rec["frozen"]["objects"] = freeze_objects()
    bpy.context.view_layer.update()
    zgn = ground_with_normal(tobjs)

    if not a.no_cut:
        t1 = time.time()
        counts, dense = instances_by_parent(a.patch_min)
        rec["instances_before"] = {k: v for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:12]}
        rec["patched"] = []
        for name, (M, protos, pid) in dense.items():
            r = patch_scatter(bpy.data.objects[name], M, protos, pid, a, zgn, rng, half)
            print("patched", name, r, flush=True)
            rec["patched"].append(r)
        rec["patch_s"] = round(time.time() - t1, 1)

    if not a.no_backdrop:
        band = json.loads(a.band.read_text())
        td = band["assets_in_box"].get("TreeFactory", 0) / band["box_area_m2"]
        protos = None
        if not a.no_cut:
            t1 = time.time()
            roots = FV.tree_roots()
            pick = [roots[i] for i in np.unique(np.linspace(0, len(roots) - 1, min(a.backdrop_protos, len(roots)))
                                                .round().astype(int))]
            protos = thinned_copies(pick, a.backdrop_thin[0], "near")
            far = thinned_copies(pick, a.backdrop_thin[1], "far")
            rec["backdrop_protos"] = [r.name for r in pick]
        rec["backdrop"] = FV.add_backdrop(lambda x, y: (lambda g: None if g is None else g[0])(zgn(x, y)),
                                          box.get("terrain_bound", region + 4.0), region, rng, td, protos=protos)
        if not a.no_cut:
            # beyond --backdrop-far m: keep a fraction of the trees, each thinner still
            n_far = n_drop = 0
            for e in [o for o in bpy.data.objects if o.name.startswith("backdrop_tree_")]:
                if max(abs(e.location.x), abs(e.location.y)) < a.backdrop_far:
                    continue
                if rng.random() > a.backdrop_thin[2]:
                    bpy.data.objects.remove(e)
                    n_drop += 1
                else:
                    e.instance_collection = far[protos.index(e.instance_collection)]
                    n_far += 1
            rec["backdrop"].update(far_m=a.backdrop_far, far_kept=n_far, far_dropped=n_drop)
            rec["backdrop_s"] = round(time.time() - t1, 1)
    if not a.no_cut and a.thin:
        rec["thinned_trees"] = thin_outer_trees(half, [tuple(map(float, t.split(":"))) for t in a.thin.split(",")])
    FV.ground_below_horizon()
    for c in bpy.data.collections:
        if c.name.startswith("placeholders"):
            c.hide_render = True
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(a.out), copy=True, compress=False)
    rec["seconds"] = round(time.time() - t0, 1)
    a.out.with_suffix(".json").write_text(json.dumps(rec, indent=1))
    print("FLIGHT_COPY_DONE", json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
