"""Views of one nature_box scene: collision cloud + flight-band map, drone-eye renders in several looks,
true depth, a top-down render. Optional thin wires (pole line with guy wires, ropes between trees).

    python infinigen/box_views.py --scene <out>/scene/scene.blend --box <out>/box.json --out <out>/views \
        [--wires] [--n-views 4] [--looks A0 A1 A2 A3] [--samples 64] [--threads 30]

Camera = the Starling front camera of our Blender screen (640 x 480, 87 deg horizontal FOV, level).
Writes <out>/{cloud.npz, band.json, band_map.png?, views.json, v<k>_<look>.png, v<k>_depth.npy, top.png}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

W, H, HFOV = 640, 480, 87.0
BAND = (0.4, 5.5)          # flight band, m above local ground (labeller band v12)
FLY_SLICE = (1.0, 3.0)     # under-canopy slice the smoke views fly in (cameras at 1.6-2.4 m)
VOX = 0.1                  # cloud voxel, m


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--box", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wires", action="store_true")
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--looks", nargs="*", default=["A0", "A1", "A2", "A3"])
    ap.add_argument("--look-views", type=int, default=1, help="render every look for this many views")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--threads", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-res", type=int, default=1024)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-backdrop", action="store_true")
    ap.add_argument("--depth-only", action="store_true", help="re-render v<k>_depth.npy for <out>/views.json")
    ap.add_argument("--keep-water", action="store_true", help="keep Infinigen's water body (coarsely meshed by the "
                    "UniformMesher: renders as mirror facets)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- scene helpers

def terrain_objects():
    """The rendered ground: Infinigen's fine terrain mesh, OpaqueTerrain_fine in most scene types,
    CollectiveTransparentTerrain_fine (ice) in arctic."""
    objs = [o for o in bpy.data.objects if o.type == "MESH" and "Terrain" in o.name]
    fine = [o for o in objs if o.name.endswith("_fine")]
    return fine or [o for o in objs if o.name.startswith("OpaqueTerrain")]


def ground_fn(objs):
    deps = bpy.context.evaluated_depsgraph_get()
    trees = [BVHTree.FromObject(o, deps) for o in objs]
    mats = [o.matrix_world.copy() for o in objs]

    def zg(x, y):
        best = None
        for t, M in zip(trees, mats):
            Mi = M.inverted()
            o = Mi @ Vector((x, y, 200.0))
            d = (Mi.to_3x3() @ Vector((0, 0, -1))).normalized()
            hit = t.ray_cast(o, d)
            if hit[0] is not None:
                z = (M @ hit[0]).z
                best = z if best is None else max(best, z)
        return best
    return zg


def is_hidden(o):
    if o.hide_render:
        return True
    for c in o.users_collection:
        if c.hide_render:
            return True
    return False


# placeholders are recognised by their collection, not by name: a populated tree's skeleton (which carries the
# twig and leaf instances) keeps its spawn_placeholder name
SKIP_NAMES = ("atmosphere", "liquid", "cloud", "Cloud", "camera", "collider", "smoke_fog", "backdrop")


def show_render_visible():
    """Infinigen saves its asset collections hidden in the viewport, which keeps them out of the view-layer
    depsgraph we sample: un-hide everything that renders, hide every placeholder collection (render too)."""
    for c in bpy.data.collections:
        if c.name.startswith("placeholders"):
            c.hide_viewport = c.hide_render = True
        elif not c.hide_render:
            c.hide_viewport = False

    def walk(lc):
        if lc.collection.name.startswith("placeholders"):
            lc.exclude = True
        else:
            if not lc.collection.hide_render:
                lc.exclude = False
                lc.hide_viewport = False
            for ch in lc.children:
                walk(ch)
    walk(bpy.context.view_layer.layer_collection)
    for o in bpy.data.objects:
        if not o.hide_render:
            o.hide_viewport = False
    bpy.context.view_layer.update()


def mat_world(M):
    return np.array([list(r) for r in M], float)


def sample_mesh_local(me, density, rng, max_verts=5000):
    """Area-weighted samples on an evaluated mesh (local coordinates), plus its vertices when it is small; at least
    one point per mesh."""
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    nv = len(me.vertices)
    if nt == 0:
        return np.zeros((0, 3))
    V = np.empty(nv * 3)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    T = np.empty(nt * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", T)
    T = T.reshape(-1, 3)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    k = rng.poisson(area * density)
    idx = np.repeat(np.arange(nt), k)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    P = a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])
    parts = [P]
    if nv <= max_verts:
        parts.append(V)
    if sum(len(x) for x in parts) == 0:
        parts.append(V.mean(0, keepdims=True))
    return np.concatenate(parts, 0)


_OFF = 1 << 20
_MASK = (1 << 21) - 1


def vox_keys(P):
    k = np.floor(P / VOX).astype(np.int64) + _OFF
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def vox_centres(keys):
    k = np.stack([(keys >> 42) & _MASK, (keys >> 21) & _MASK, keys & _MASK], 1) - _OFF
    return (k + 0.5) * VOX


class VoxAcc:
    """Running set of occupied voxels (first label wins), compacted every few million points."""

    def __init__(self):
        self.keys = np.zeros(0, np.int64)
        self.lab = np.zeros(0, np.int16)
        self.pk, self.pl, self.n = [], [], 0

    def add(self, P, label):
        kk = vox_keys(P)
        self.pk.append(kk)
        self.pl.append(np.full(len(kk), label, np.int16))
        self.n += len(kk)
        if self.n > 4_000_000:
            self.compact()

    def compact(self):
        if not self.pk:
            return
        K = np.concatenate([self.keys] + self.pk)
        L = np.concatenate([self.lab] + self.pl)
        self.keys, first = np.unique(K, return_index=True)
        self.lab = L[first]
        self.pk, self.pl, self.n = [], [], 0

    def result(self):
        self.compact()
        return vox_centres(self.keys), self.lab


def sample_scene(region, zcap, rng, density=200.0, skip_terrain=True, ground=None, min_top_agl=0.3, only_prefix=None):
    """Every render-visible mesh instance whose bounds meet the region: points in world coordinates + labels.
    With ground(x, y) given, instances lying wholly below min_top_agl over the ground (grass, leaves, twigs) are
    skipped: they are under the flight band and would dominate the cloud."""
    deps = bpy.context.evaluated_depsgraph_get()
    cache = {}
    acc, names = VoxAcc(), {}
    # per source object: skip or not (users_collection scans every collection, so never call it per instance)
    placeholder_objs = set()
    for c in bpy.data.collections:
        if c.name.startswith("placeholders"):
            placeholder_objs.update(o.name for o in c.all_objects)
    coll_hidden = {c.name for c in bpy.data.collections if c.hide_render}
    obj_colls = {}
    for c in bpy.data.collections:
        for o in c.objects:
            obj_colls.setdefault(o.name, []).append(c.name)
    skip_cache = {}
    binfo = {}
    small_pts, small_lab = [], []

    def skip_obj(orig):
        nm = orig.name
        v = skip_cache.get(nm)
        if v is None:
            v = (any(s in nm for s in SKIP_NAMES) or nm in placeholder_objs
                 or (skip_terrain and "Terrain" in nm) or orig.hide_render
                 or any(cn in coll_hidden for cn in obj_colls.get(nm, [])))
            skip_cache[nm] = v
        return v

    n_inst = n_big = n_small = 0
    for inst in deps.object_instances:
        ob = inst.object
        if ob.type != "MESH":
            continue
        n_inst += 1
        orig = ob.original
        nm = orig.name
        if only_prefix is not None and not nm.startswith(only_prefix):
            continue
        if inst.is_instance:
            # a geometry-nodes / collection instance renders wherever its source lives (the sources sit in
            # render-hidden prototype collections): judge it by its instancer, and by name only
            if any(t in nm for t in SKIP_NAMES):
                continue
            if inst.parent is not None and skip_obj(inst.parent.original):
                continue
        elif skip_obj(orig):
            continue
        key = ob.data.as_pointer() if ob.data is not None else id(ob)
        bi = binfo.get(key)
        if bi is None:
            bb = np.array([list(v) for v in ob.bound_box])
            bi = (bb, float(np.ptp(bb, axis=0).max()))
            binfo[key] = bi
        bb, ext = bi
        M = np.array(inst.matrix_world)
        bw = bb @ M[:3, :3].T + M[:3, 3]
        if ext * np.linalg.norm(M[:3, :3], axis=0).max() < 0.25:
            # leaves, twigs, needles, pebbles: one voxel's worth each; keep the canopy, drop the ground litter
            if ground is not None:
                c = bw.mean(0)
                zg0 = ground(c[0], c[1])
                if zg0 is None or bw[:, 2].max() < zg0 + min_top_agl:
                    continue
            if np.abs(bw[:, 0]).max() <= region and np.abs(bw[:, 1]).max() <= region and bw[:, 2].min() <= zcap:
                small_pts.append(bw.mean(0))
                n_small += 1
                small_lab.append(names.setdefault(nm.split("(")[0].split(".")[0], len(names)))
                if len(small_pts) >= 200_000:
                    for li in set(small_lab):
                        m = np.array(small_lab) == li
                        acc.add(np.array(small_pts)[m], li)
                    small_pts, small_lab = [], []
            continue
        lo, hi = bw.min(0), bw.max(0)
        if hi[0] < -region or lo[0] > region or hi[1] < -region or lo[1] > region or lo[2] > zcap:
            continue
        if ground is not None:
            zg0 = ground(0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]))
            if zg0 is not None and hi[2] < zg0 + min_top_agl:
                continue
        if key not in cache:
            try:
                cache[key] = sample_mesh_local(ob.data, density, rng)
            except Exception as e:  # noqa: BLE001
                print("sample skip", nm, e)
                cache[key] = np.zeros((0, 3))
        P = cache[key]
        if len(P) == 0:
            continue
        Pw = P @ M[:3, :3].T + M[:3, 3]
        keep = (np.abs(Pw[:, 0]) <= region) & (np.abs(Pw[:, 1]) <= region) & (Pw[:, 2] <= zcap)
        if keep.any():
            fam = nm.split("(")[0].split(".")[0]
            li = names.setdefault(fam, len(names))
            acc.add(Pw[keep], li)
            n_big += 1
    if small_pts:
        for li in set(small_lab):
            m = np.array(small_lab) == li
            acc.add(np.array(small_pts)[m], li)
    P, L = acc.result()
    print(f"sample_scene: {n_inst} mesh instances, {len(cache)} unique meshes, {n_big} sampled instances, "
          f"{n_small} small instances kept as points, {len(P)} occupied voxels", flush=True)
    return P, L, names


def voxelize(P, lab):
    k = np.floor(P / VOX).astype(np.int64)
    _, first = np.unique(k, axis=0, return_index=True)
    return P[first], lab[first]


# ---------------------------------------------------------------- wires

def wire_material():
    m = bpy.data.materials.new("wire_black")
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1)
    bsdf.inputs["Roughness"].default_value = 0.45
    return m


def wood_material():
    m = bpy.data.materials.new("pole_wood")
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.16, 0.11, 0.07, 1)
    bsdf.inputs["Roughness"].default_value = 0.85
    return m


def catenary(p0, p1, sag, n=48):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    t = np.linspace(0, 1, n)
    P = p0[None] + t[:, None] * (p1 - p0)[None]
    P[:, 2] -= 4 * sag * t * (1 - t)
    return P


def add_curve(name, P, radius, mat):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 2
    sp = cu.splines.new("POLY")
    sp.points.add(len(P) - 1)
    for i, p in enumerate(P):
        sp.points[i].co = (p[0], p[1], p[2], 1)
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def add_pole(name, x, y, z0, h, r, mat):
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=r, depth=h, location=(x, y, z0 + h / 2 - 0.3))
    ob = bpy.context.active_object
    ob.name = name
    ob.data.materials.append(mat)
    return ob


def add_wires(zg, half, rng, P_obst):
    """A pole line across the box (2 wires + guy wires) and two ropes between free points; returns the objects."""
    wm, pm = wire_material(), wood_material()
    objs = []
    ang = rng.uniform(0, math.pi)
    d = np.array([math.cos(ang), math.sin(ang)])
    nrm = np.array([-d[1], d[0]])
    off = rng.uniform(-0.3, 0.3) * half
    span = 2 * half / 2.0
    poles = []
    for i, s in enumerate(np.linspace(-half - 2, half + 2, 4)):
        x, y = (s * d + off * nrm)
        z0 = zg(x, y) or 0.0
        objs.append(add_pole(f"wire_pole_{i}", x, y, z0, 6.5, 0.12, pm))
        poles.append((x, y, z0))
        # guy wire from pole top to the ground, 3 m out sideways
        side = nrm if i % 2 else -nrm
        gx, gy = np.array([x, y]) + 3.0 * side
        objs.append(add_curve(f"guy_wire_{i}", catenary((x, y, z0 + 5.8), (gx, gy, (zg(gx, gy) or z0) + 0.05), 0.05),
                              0.006, wm))
    for i in range(len(poles) - 1):
        (x0, y0, z0), (x1, y1, z1) = poles[i], poles[i + 1]
        for k, hgt in enumerate((5.6, 4.9)):
            objs.append(add_curve(f"line_wire_{i}_{k}",
                                  catenary((x0 + 0.25 * k * nrm[0], y0 + 0.25 * k * nrm[1], z0 + hgt),
                                           (x1 + 0.25 * k * nrm[0], y1 + 0.25 * k * nrm[1], z1 + hgt), 0.9),
                                  0.008, wm))
    # ropes: between two random trunk-ish obstacle points 6-12 m apart at 1.5-3 m above ground
    trunks = P_obst
    if len(trunks):
        for r in range(2):
            for _ in range(200):
                i, j = rng.integers(len(trunks), size=2)
                a, b = trunks[i], trunks[j]
                dist = np.hypot(*(a[:2] - b[:2]))
                if 6 < dist < 12 and max(abs(a[0]), abs(a[1]), abs(b[0]), abs(b[1])) < half:
                    za, zb = (zg(a[0], a[1]) or 0) + rng.uniform(1.5, 3.0), (zg(b[0], b[1]) or 0) + rng.uniform(1.5, 3.0)
                    objs.append(add_curve(f"rope_{r}", catenary((a[0], a[1], za), (b[0], b[1], zb), 0.3), 0.01, wm))
                    break
    return objs


# ---------------------------------------------------------------- looks

def sky_node():
    w = bpy.context.scene.world
    for n in w.node_tree.nodes:
        if n.type == "TEX_SKY":
            return n
    return None


def world_background():
    w = bpy.context.scene.world
    for n in w.node_tree.nodes:
        if n.type == "BACKGROUND":
            return n
    return None


FOG_GEOM = {"half_xy": 60.0, "z0": -10.0, "z1": 40.0}


def set_fog(density):
    """Haze in a closed box around the scene. A world volume would also absorb the sky light itself (all black)."""
    w = bpy.context.scene.world
    out = [n for n in w.node_tree.nodes if n.type == "OUTPUT_WORLD"][0]
    for l in list(out.inputs["Volume"].links):
        w.node_tree.links.remove(l)
    ob = bpy.data.objects.get("smoke_fog_box")
    if density <= 0:
        if ob is not None:
            ob.hide_render = True
        return
    if ob is None:
        me = bpy.data.meshes.new("smoke_fog_box")
        h, z0, z1 = FOG_GEOM["half_xy"], FOG_GEOM["z0"], FOG_GEOM["z1"]
        V = [(x, y, z) for z in (z0, z1) for y in (-h, h) for x in (-h, h)]
        F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        me.from_pydata(V, [], F)
        me.update()
        ob = bpy.data.objects.new("smoke_fog_box", me)
        bpy.context.scene.collection.objects.link(ob)
        m = bpy.data.materials.new("smoke_fog")
        m.use_nodes = True
        nt = m.node_tree
        for n in list(nt.nodes):
            if n.type != "OUTPUT_MATERIAL":
                nt.nodes.remove(n)
        vol = nt.nodes.new("ShaderNodeVolumePrincipled")
        vol.name = "fog"
        vol.inputs["Color"].default_value = (0.86, 0.88, 0.9, 1)
        vol.inputs["Anisotropy"].default_value = 0.3
        mo = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]
        nt.links.new(vol.outputs[0], mo.inputs["Volume"])
        try:
            m.cycles.homogeneous_volume = True
        except AttributeError:
            pass
        me.materials.append(m)
    ob.hide_render = False
    ob.data.materials[0].node_tree.nodes["fog"].inputs["Density"].default_value = density


def apply_look(look, base, view_yaw):
    sky, bg = sky_node(), world_background()
    if sky is not None and sky.id_data.animation_data:
        sky.id_data.animation_data_clear()
    if sky is None or bg is None:
        print("no sky texture / background node; look", look, "skipped")
        return False
    sky.sun_elevation, sky.sun_rotation, sky.sun_intensity = base["elev"], base["rot"], base["sun"]
    bg.inputs["Strength"].default_value = base["strength"]
    sky.air_density, sky.dust_density = base["air"], base["dust"]
    set_fog(0.0)
    scene = bpy.context.scene
    scene.view_settings.exposure = 0.0
    if look == "A0":          # Infinigen's own sky
        pass
    elif look == "A1":        # low sun against the camera: glare, long shadows, dark trunks
        sky.sun_elevation = math.radians(4.0)
        sky.sun_rotation = (math.pi / 2 - view_yaw) % (2 * math.pi)  # sun roughly ahead of the camera
        sky.sun_intensity = 1.0
        scene.view_settings.exposure = 1.0
    elif look == "A2":        # overcast haze / fog
        sky.sun_elevation = math.radians(35.0)
        sky.sun_intensity = 0.15
        sky.dust_density = 8.0
        bg.inputs["Strength"].default_value = base["strength"] * 1.4
        set_fog(0.04)
    elif look == "A3":        # dusk, sun just below the horizon
        sky.sun_elevation = math.radians(-2.0)
        sky.sun_intensity = 1.0
        scene.view_settings.exposure = 2.0
    return True


# ---------------------------------------------------------------- rendering

def setup_render(samples, threads):
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.cycles.device = "CPU"
    s.cycles.samples = samples
    s.cycles.use_adaptive_sampling = True
    s.cycles.adaptive_threshold = 0.02
    s.cycles.use_denoising = True
    s.cycles.denoiser = "OPENIMAGEDENOISE"
    s.render.threads_mode, s.render.threads = "FIXED", threads
    s.render.use_persistent_data = True
    s.render.image_settings.file_format = "PNG"
    s.render.film_transparent = False
    s.view_layers[0].use_pass_z = True
    try:
        s.view_settings.view_transform = "Filmic"
    except TypeError:
        pass
    s.render.use_motion_blur = False


def make_camera(name="smoke_cam"):
    cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    bpy.context.scene.collection.objects.link(cam)
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = 36.0
    fx = (W / 2) / math.tan(math.radians(HFOV / 2))
    cam.data.lens = fx * 36.0 / W
    cam.data.clip_start, cam.data.clip_end = 0.05, 1000.0
    return cam


def pose_camera(cam, p, yaw, pitch=0.0):
    # Blender camera looks down -Z with +Y up; build world rotation: yaw about Z, level horizon
    cam.location = Vector(p)
    cam.rotation_euler = (math.pi / 2 + pitch, 0.0, yaw - math.pi / 2)


def render_to(path, res=(W, H)):
    s = bpy.context.scene
    s.render.resolution_x, s.render.resolution_y = res
    s.render.resolution_percentage = 100
    s.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_depth(path_npy):
    """Z pass through a multilayer EXR (the compositor viewer is not filled in background mode): metric z-depth."""
    import OpenEXR
    s = bpy.context.scene
    old = (s.cycles.samples, s.cycles.use_denoising, s.render.image_settings.file_format)
    s.cycles.samples, s.cycles.use_denoising = 1, False
    s.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    s.render.image_settings.color_depth = "32"
    tmp = Path(str(path_npy)[:-4] + ".exr")
    vols = [o for o in bpy.data.objects if (o.name.startswith("atmosphere") or o.name == "smoke_fog_box")
            and not o.hide_render]
    for o in vols:
        o.hide_render = True
    render_to(tmp)
    for o in vols:
        o.hide_render = False
    f = OpenEXR.File(str(tmp))
    chans = f.channels()
    key = [k for k in chans if k.endswith("Depth.Z") or k.endswith("Depth.V") or k == "Depth"]
    if not key:
        key = [k for k in chans if "Depth" in k]
    px = np.asarray(chans[key[0]].pixels, np.float32)
    px = px.reshape(px.shape[0], px.shape[1]) if px.ndim > 2 else px
    np.save(path_npy, np.minimum(px, 65000.0).astype(np.float16))
    tmp.unlink()
    s.cycles.samples, s.cycles.use_denoising, s.render.image_settings.file_format = old
    s.render.image_settings.color_depth = "8"
    return px


def asset_counts(half):
    """Populated asset roots (unique or instanced) per factory family whose base lies inside the flight box."""
    out = {}
    for c in bpy.data.collections:
        if not c.name.startswith("unique_assets:"):
            continue
        fam = c.name.split(":", 1)[1].split("(")[0]
        members = set(c.objects)
        n = 0
        for o in c.objects:
            if o.parent in members:
                continue
            x, y, _ = o.matrix_world.translation
            if abs(x) < half and abs(y) < half:
                n += 1
        if n:
            out[fam] = out.get(fam, 0) + n
    return out


def water_material(o):
    """Infinigen's water shader renders as the error colour (magenta) on the UniformMesher's liquid mesh, and its
    flat facets mirror the sky in shards: a plain transmissive water with a ripple bump, smooth-shaded."""
    m = bpy.data.materials.get("envgen_water")
    if m is None:
        m = bpy.data.materials.new("envgen_water")
        m.use_nodes = True
        nt = m.node_tree
        bsdf = nt.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (0.02, 0.045, 0.045, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.04
        bsdf.inputs["IOR"].default_value = 1.333
        for k in ("Transmission Weight", "Transmission"):
            if k in bsdf.inputs:
                bsdf.inputs[k].default_value = 0.9
                break
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 6.0
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.15
        bump.inputs["Distance"].default_value = 0.02
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    o.data.materials.clear()
    o.data.materials.append(m)
    for poly in o.data.polygons:
        poly.use_smooth = True


# ---------------------------------------------------------------- backdrop

def ground_below_horizon(color=(0.035, 0.04, 0.025)):
    """Rays below the horizon that miss all geometry see the Nishita sky's black lower half; give them a dark
    forest-floor tone instead."""
    w = bpy.context.scene.world
    nt = w.node_tree
    bg = world_background()
    if bg is None or nt.nodes.get("smoke_ground_mix") is not None:
        return
    src = bg.inputs["Color"].links[0].from_socket if bg.inputs["Color"].links else None
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    lt = nt.nodes.new("ShaderNodeMath")
    lt.operation = "LESS_THAN"
    lt.inputs[1].default_value = 0.0
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.name = "smoke_ground_mix"
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], lt.inputs[0])
    nt.links.new(lt.outputs[0], mix.inputs["Fac"])
    if src is not None:
        nt.links.new(src, mix.inputs["Color1"])
    else:
        mix.inputs["Color1"].default_value = bg.inputs["Color"].default_value
    mix.inputs["Color2"].default_value = (*color, 1.0)
    nt.links.new(mix.outputs[0], bg.inputs["Color"])


def copy_edge_attributes(src, me, V, e, band=2.0):
    """Infinigen's terrain material mixes its surfaces by per-vertex attributes of the terrain mesh; on a mesh
    without them it renders the error colour (magenta). Give each skirt vertex the attributes of the nearest
    terrain vertex near the square's edge, so the ground continues outward."""
    from scipy.spatial import cKDTree
    sm = src.data
    nv = len(sm.vertices)
    co = np.empty(nv * 3)
    sm.vertices.foreach_get("co", co)
    M = np.array([list(r) for r in src.matrix_world])
    co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    edge = np.nonzero(np.maximum(np.abs(co[:, 0]), np.abs(co[:, 1])) > e - band)[0]
    if len(edge) == 0:
        return 0
    _, nn = cKDTree(co[edge, :2]).query(np.asarray(V)[:, :2])
    src_idx = edge[nn]
    width = {"FLOAT": 1, "INT": 1, "BOOLEAN": 1, "INT8": 1, "FLOAT_VECTOR": 3, "FLOAT_COLOR": 4, "BYTE_COLOR": 4,
             "FLOAT2": 2}
    key = {"FLOAT_COLOR": "color", "BYTE_COLOR": "color", "FLOAT_VECTOR": "vector", "FLOAT2": "vector"}
    n = 0
    for at in list(sm.attributes):
        if at.domain != "POINT" or at.name.startswith(".") or at.name == "position" or at.data_type not in width:
            continue
        w = width[at.data_type]
        dt = np.int32 if at.data_type in ("INT", "INT8") else (bool if at.data_type == "BOOLEAN" else np.float32)
        buf = np.empty(nv * w, dtype=dt)
        at.data.foreach_get(key.get(at.data_type, "value"), buf)
        na = me.attributes.new(at.name, at.data_type, "POINT")
        na.data.foreach_set(key.get(at.data_type, "value"), buf.reshape(nv, w)[src_idx].reshape(-1))
        n += 1
    print(f"skirt: {n} terrain attributes copied from the edge", flush=True)
    return n


def add_backdrop(zg, tbound, region, rng, tree_density, ring_out=90.0, step=1.5):
    """Scenery outside the populated square (none of it is in the collision cloud or the flight box):
    a terrain skirt that continues the fine terrain's edge heights outward, and instanced copies of the scene's own
    trees at the scene's density between the populated square and ring_out."""
    t0 = time.time()
    e = tbound - 0.6
    # edge height profile along the square (every 1 m), for the skirt's height extension
    s = np.arange(-e, e + 1e-6, 1.0)
    sides = {"S": [zg(x, -e) for x in s], "N": [zg(x, e) for x in s], "W": [zg(-e, y) for y in s],
             "E": [zg(e, y) for y in s]}
    allz = np.array([z for v in sides.values() for z in v if z is not None])
    zmean = float(np.median(allz)) if len(allz) else 0.0
    prof = {k: np.array([zmean if z is None else z for z in v]) for k, v in sides.items()}

    def z_edge(x, y):
        cx, cy = np.clip(x, -e, e), np.clip(y, -e, e)
        dx, dy = abs(x) - e, abs(y) - e
        if dx >= dy:
            k, t = ("E" if x > 0 else "W"), cy
        else:
            k, t = ("N" if y > 0 else "S"), cx
        zz = np.interp(t, s, prof[k])
        d = max(dx, dy, 0.0)
        w = min(1.0, d / 40.0)
        w = w * w * (3 - 2 * w)
        return zz + (zmean - zz) * w - 0.04

    xs = np.arange(-ring_out, ring_out + 1e-6, step)
    V, F, idx = [], [], {}
    for j, y in enumerate(xs):
        for i, x in enumerate(xs):
            if max(abs(x), abs(y)) < e - step:
                continue
            idx[(i, j)] = len(V)
            V.append((x, y, z_edge(x, y)))
    for (i, j), v0 in idx.items():
        q = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        if all(k in idx for k in q):
            F.append(tuple(idx[k] for k in q))
    me = bpy.data.meshes.new("backdrop_skirt")
    me.from_pydata(V, [], F)
    me.update()
    sk = bpy.data.objects.new("backdrop_skirt", me)
    bpy.context.scene.collection.objects.link(sk)
    fine = terrain_objects()
    if fine and fine[0].data.materials:
        me.materials.append(fine[0].data.materials[0])
        copy_edge_attributes(fine[0], me, V, e)
    for p in me.polygons:
        p.use_smooth = True

    def zskirt(x, y):
        if max(abs(x), abs(y)) <= e:
            z = zg(x, y)
            return z if z is not None else z_edge(x, y)
        return z_edge(x, y)

    # tree prototypes: every populated tree hierarchy as a collection to instance
    roots = []
    for c in bpy.data.collections:
        if c.name.startswith("unique_assets:TreeFactory"):
            members = set(c.objects)
            roots += [o for o in c.objects if o.parent not in members]  # asset roots hang off their placeholders
    protos = []
    for k, r in enumerate(roots[:40]):
        pc = bpy.data.collections.new(f"backdrop_proto_{k}")
        stack = [r]
        while stack:
            o = stack.pop()
            pc.objects.link(o)
            stack += list(o.children)
        pc.instance_offset = r.matrix_world.translation.copy()
        protos.append(pc)
    n_inst = 0
    if protos:
        holder = bpy.data.collections.new("backdrop_trees")
        bpy.context.scene.collection.children.link(holder)
        area = (2 * ring_out) ** 2 - (2 * region) ** 2
        n = int(tree_density * area)
        pts = rng.uniform(-ring_out, ring_out, size=(int(n * 1.6) + 10, 2))
        pts = pts[np.maximum(np.abs(pts[:, 0]), np.abs(pts[:, 1])) > region + 1.0][:n]
        for (x, y) in pts:
            emp = bpy.data.objects.new(f"backdrop_tree_{n_inst}", None)
            emp.instance_type = "COLLECTION"
            emp.instance_collection = protos[int(rng.integers(len(protos)))]
            emp.location = (float(x), float(y), float(zskirt(x, y)) - 0.1)
            emp.rotation_euler = (0.0, 0.0, float(rng.uniform(0, 2 * math.pi)))
            sc = float(rng.uniform(0.85, 1.15))
            emp.scale = (sc, sc, sc)
            holder.objects.link(emp)
            n_inst += 1
    ground_below_horizon()
    print(f"backdrop: skirt {len(V)} verts, {len(protos)} tree prototypes, {n_inst} instanced trees, "
          f"{time.time() - t0:.0f} s", flush=True)
    return {"skirt_verts": len(V), "prototypes": len(protos), "instances": n_inst, "ring_out_m": ring_out}


# ---------------------------------------------------------------- main

def main():
    a = parse()
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    box = json.loads(a.box.read_text())
    half, region = box["half"], box["region"]
    rng = np.random.default_rng(a.seed)
    bpy.ops.wm.open_mainfile(filepath=str(a.scene))
    show_render_visible()
    if a.depth_only:
        for o in bpy.data.objects:
            if o.name.startswith("liquid") and not a.keep_water:
                o.hide_render = True
        setup_render(a.samples, a.threads)
        cam = make_camera()
        bpy.context.scene.camera = cam
        for k, v in enumerate(json.loads((a.out / "views.json").read_text())["views"]):
            pose_camera(cam, v["p"], v["yaw"])
            t1 = time.time()
            render_depth(a.out / f"v{k}_depth.npy")
            print("depth", k, round(time.time() - t1, 1), "s", flush=True)
        print("FOREST_VIEWS_DONE", json.dumps({"seconds": round(time.time() - t0, 1), "depth_only": True}))
        return
    if not a.keep_water:
        for o in bpy.data.objects:
            if o.name.startswith("liquid"):
                o.hide_render = True
                print("hid water body", o.name, flush=True)
    else:
        for o in bpy.data.objects:
            if o.name.startswith("liquid") and o.type == "MESH" and not o.hide_render:
                water_material(o)
                print("water body re-shaded", o.name, flush=True)
    tobjs = terrain_objects()
    print("terrain objects", [o.name for o in tobjs])
    zg = ground_fn(tobjs)

    # ground grid over the region (for AGL and the band map)
    g = np.arange(-region, region + 1e-6, 0.5)
    GZ = np.array([[zg(x, y) if zg(x, y) is not None else np.nan for x in g] for y in g])
    zcap = np.nanmax(GZ) + 12.0
    FOG_GEOM.update(half_xy=region + 40.0, z0=float(np.nanmin(GZ)) - 5.0, z1=float(np.nanmax(GZ)) + 35.0)
    def ground_grid(x, y):
        i = int(round((x + region) / 0.5))
        j = int(round((y + region) / 0.5))
        if 0 <= i < len(g) and 0 <= j < len(g):
            z = GZ[j, i]
            return None if np.isnan(z) else float(z)
        return None

    P, L, names = sample_scene(region, zcap, rng, ground=ground_grid)
    print(f"sampled {len(P)} obstacle samples from {len(names)} families in {time.time() - t0:.0f} s")

    wire_objs = []
    if a.wires:
        # trunk candidates: obstacle points 1-3 m above ground inside the box
        gi = np.clip(((P[:, :2] + region) / 0.5).round().astype(int), 0, len(g) - 1)
        agl = P[:, 2] - GZ[gi[:, 1], gi[:, 0]]
        cand = P[(agl > 1.0) & (agl < 3.0) & (np.abs(P[:, 0]) < half) & (np.abs(P[:, 1]) < half)]
        cand = cand[rng.choice(len(cand), size=min(len(cand), 3000), replace=False)] if len(cand) else cand
        wire_objs = add_wires(zg, half, rng, cand)
        bpy.context.view_layer.update()
        Pw, Lw, nw = sample_scene(region, zcap, rng, density=4000.0, ground=None,
                                  only_prefix=("line_wire", "guy_wire", "rope", "wire_pole"))
        sel = np.isin(Lw, [nw[k] for k in nw if k.startswith(("line_wire", "guy_wire", "rope", "wire_pole"))])
        wl = len(names)
        names["wires"] = wl
        P = np.concatenate([P, Pw[sel]])
        L = np.concatenate([L, np.full(int(sel.sum()), wl, np.int16)])

    P, L = voxelize(P, L)
    gi = np.clip(((P[:, :2] + region) / 0.5).round().astype(int), 0, len(g) - 1)
    agl = P[:, 2] - GZ[gi[:, 1], gi[:, 0]]
    np.savez_compressed(a.out / "cloud.npz", points=P.astype(np.float32), label=L, agl=agl.astype(np.float32),
                        names=json.dumps(names), ground_grid=GZ.astype(np.float32), grid_x=g.astype(np.float32))

    # occupancy on a 0.25 m grid inside the box, per height slice over the local ground. In a forest the crowns
    # start a few metres up, so the whole 0.4-5.5 m band projected flat is mostly canopy; the slice the drone
    # flies in under the canopy (FLY_SLICE) is the one views and clearances use.
    cell = 0.25
    nb = int(round(2 * half / cell))
    inbox = (np.abs(P[:, 0]) < half) & (np.abs(P[:, 1]) < half)
    from scipy.ndimage import distance_transform_edt

    def slice_occ(lo, hi):
        m = inbox & (agl >= lo) & (agl <= hi)
        o = np.zeros((nb, nb), bool)
        ij = np.clip(((P[m, :2] + half) / cell).astype(int), 0, nb - 1)
        o[ij[:, 1], ij[:, 0]] = True
        return o

    slices = {}
    for lo, hi in [(BAND[0], BAND[1]), (0.4, 1.0), FLY_SLICE, (3.0, 5.5)]:
        o = slice_occ(lo, hi)
        cl = distance_transform_edt(~o) * cell
        slices[f"{lo:g}-{hi:g}"] = {"blocked_frac": float(o.mean()), "free_frac_clear_1m": float((cl >= 1.0).mean()),
                                    "median_clearance_m": float(np.median(cl[~o])) if (~o).any() else 0.0}
    occ = slice_occ(*FLY_SLICE)
    clear = distance_transform_edt(~occ) * cell
    # canopy: cells with anything 3-15 m up; base = lowest such voxel per cell (median over covered cells)
    up = inbox & (agl > 3.0) & (agl < 15.0)
    base = {}
    ij = np.clip(((P[up, :2] + half) / cell).astype(int), 0, nb - 1)
    for (cx, cy), z in zip(map(tuple, ij), agl[up]):
        if z < base.get((cx, cy), 1e9):
            base[(cx, cy)] = z
    band = {"band_m": BAND, "fly_slice_m": FLY_SLICE, "slices": slices,
            "blocked_frac": slices[f"{FLY_SLICE[0]:g}-{FLY_SLICE[1]:g}"]["blocked_frac"],
            "free_frac_clear_1m": float((clear >= 1.0).mean()),
            "median_clearance_m": float(np.median(clear[~occ])) if (~occ).any() else 0.0,
            "canopy_cover_frac": float(len(base) / nb ** 2),
            "canopy_base_m": float(np.median(list(base.values()))) if base else None,
            "points": int(len(P)), "families": names,
            "thin_points_frac": float(np.isin(L, [names.get("wires", -1)]).mean()),
            "ground_relief_m": float(np.nanmax(GZ) - np.nanmin(GZ)),
            "assets_in_box": asset_counts(half), "box_area_m2": float((2 * half) ** 2)}
    (a.out / "band.json").write_text(json.dumps(band, indent=1))
    np.save(a.out / "band_occ.npy", occ)
    print("band", json.dumps({k: v for k, v in band.items() if k != "families"}))

    if a.no_render:
        return

    backdrop = None
    if not a.no_backdrop:
        # backdrop trees at the density measured inside the box (Infinigen's nominal density overstates it)
        td = band["assets_in_box"].get("TreeFactory", 0) / band["box_area_m2"]
        backdrop = add_backdrop(zg, box.get("terrain_bound", region + 4.0), region, rng, td)

    # view poses: free spots (clearance >= 1.5 m in FLY_SLICE), 1.6-2.4 m AGL, facing the longest free sightline
    setup_render(a.samples, a.threads)
    cam = make_camera()
    bpy.context.scene.camera = cam
    free = np.argwhere(clear >= 1.5)
    views = []
    tries = 0
    while len(views) < a.n_views and tries < 500 and len(free):
        tries += 1
        r, c = free[rng.integers(len(free))]
        x, y = -half + (c + 0.5) * cell, -half + (r + 0.5) * cell
        if abs(x) > half - 3 or abs(y) > half - 3:
            continue
        if any(math.hypot(x - v["p"][0], y - v["p"][1]) < 0.4 * half for v in views):
            continue
        z0 = zg(x, y)
        if z0 is None:
            continue
        h = float(rng.uniform(1.6, 2.4))
        # yaw with the longest free ray (in the band map) among 16 directions, so the frame looks into the forest
        best, byaw = -1, 0.0
        for yaw in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            dd = 0.0
            while dd < 2 * half:
                xx, yy = x + dd * math.cos(yaw), y + dd * math.sin(yaw)
                ci, ri = int((xx + half) / cell), int((yy + half) / cell)
                if not (0 <= ci < nb and 0 <= ri < nb) or occ[ri, ci]:
                    break
                dd += cell
            if dd > best:
                best, byaw = dd, yaw
        views.append({"p": [x, y, z0 + h], "agl": h, "yaw": float(byaw), "free_ray_m": best})

    base = None
    sky, bg = sky_node(), world_background()
    if sky is not None and bg is not None:
        base = {"elev": sky.sun_elevation, "rot": sky.sun_rotation, "sun": sky.sun_intensity,
                "strength": bg.inputs["Strength"].default_value, "air": sky.air_density, "dust": sky.dust_density}
    timings = []
    for k, v in enumerate(views):
        pose_camera(cam, v["p"], v["yaw"])
        looks = a.looks if k < a.look_views else ["A0"]
        for look in looks:
            if base is not None:
                apply_look(look, base, v["yaw"])
            t1 = time.time()
            render_to(a.out / f"v{k}_{look}.png")
            timings.append({"view": k, "look": look, "s": round(time.time() - t1, 1)})
            print("rendered", k, look, timings[-1]["s"], "s", flush=True)
        if base is not None:
            apply_look("A0", base, v["yaw"])
        render_depth(a.out / f"v{k}_depth.npy")

    # top-down orthographic render over the region (A0)
    if base is not None:
        apply_look("A0", base, 0.0)
    top = bpy.data.objects.new("top_cam", bpy.data.cameras.new("top_cam"))
    bpy.context.scene.collection.objects.link(top)
    top.data.type = "ORTHO"
    top.data.ortho_scale = 2 * region
    top.data.clip_end = 1000.0
    top.location = (0.0, 0.0, float(np.nanmax(GZ)) + 60.0)
    top.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = top
    bpy.context.scene.cycles.samples = 16
    t1 = time.time()
    render_to(a.out / "top.png", (a.top_res, a.top_res))
    timings.append({"view": "top", "look": "A0", "s": round(time.time() - t1, 1)})

    rec = {"views": views, "timings": timings, "camera": {"w": W, "h": H, "hfov_deg": HFOV},
           "samples": a.samples, "threads": a.threads, "wires": [o.name for o in wire_objs], "backdrop": backdrop,
           "seconds": round(time.time() - t0, 1)}
    (a.out / "views.json").write_text(json.dumps(rec, indent=1))
    if a.wires:
        bpy.ops.wm.save_as_mainfile(filepath=str(a.out / "scene_wires.blend"), copy=True)
    print("FOREST_VIEWS_DONE", json.dumps({"seconds": rec["seconds"]}))


if __name__ == "__main__":
    main()
