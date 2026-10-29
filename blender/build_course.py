"""Build one Blender course from a patch spec: the scene, its mesh, its cloud.

    blender -b --factory-startup -P blender/build_course.py -- \
        --spec <spec.json> --assets ~/.cache/blender_assets --out <course dir> [--preview]
    blender -b --factory-startup -t 6 -P blender/build_course.py -- \
        --spec <spec.json> --kit street --assets /mnt/data/blender/assets --out <course dir> \
        --cpu --threads 6 --pack --pack-max-res 1024 --require-clean      # a pipeline build

The patch spec (``blender/dump_spec.py``, i.e. ``draw_states.PatchSpec``) is
the layout: its primitives are rebuilt as meshes **with the same geometry**
(walls with their slots, boxes, cylinders, spheres) and dressed in PBR
materials, so the obstacle the goal regime was drawn for is exactly there.
Real objects from the asset library are then scattered round it as further
obstacles, an HDRI lights the scene and a textured ground runs well past
the camera's 20 m. ``--theme`` picks the look (yard, woodland, warehouse --
an enclosed shell with ceiling lights -- or plaza); ``--clutter``,
``--keepout`` and ``--hang`` set how crowded and how hard it is. ``--kit``
builds a place kit instead (``blender/kits/``, the v2 library).

Writes into ``--out``:

    course.blend    the scene (textures referenced from --assets; with --pack every
                    image is inside it, --pack-max-res downscaling the 8-bit ones first)
    mesh.npz        (--export-mesh) every collidable triangle in world coordinates
    mesh.ply        (--export-mesh) the same, for any mesh viewer
    cloud.npz       ``points`` (float64), ``region`` (int16) -- the labeller's cloud
                    contract (patch.sample): obstacle surface samples one per
                    ``CLOUD_VOXEL`` (0.10 m) voxel (region 0) + the floor grid at
                    0.25 m over the patch bounds (-1)
    course.json     what was placed where, the HDRI, the materials, the counts, and
                    the provenance (``git_sha``, ``dirty``, ``kits_order``,
                    ``library_sha``, Blender version, argv; ``blender/provenance.py``);
                    kit courses: ``kit`` and ``semantics_changed`` -- the kit's changes to
                    the patch geometry as ``[{prim, kind, change, detail, z_top}]``
                    (``blender/course_semantics.py``; ``chosen`` keeps the kit's own record)
    preview_*.png   (--preview) overview renders

Geometry rule: everything that collides is in the mesh and the cloud; the
ground plane is the floor (z = 0), represented by the floor grid exactly as
for a primitive patch. Nothing render-only stands above the floor, and the
cloud stops at ``CLOUD_ZMAX`` = 4.75 m, above anything the drone can reach
(see the constant). Alpha-cut surfaces (leaf cards) are in the cloud only
where their alpha reaches 0.5, the threshold the depth render uses, so the
cloud holds the leaves, not the cards.

Reproducibility: the same spec, arguments, code and library give the same
``cloud.npz`` byte for byte (numpy writes the zip with a fixed timestamp)
and the same ``course.json`` apart from timings, device and argv (checked
09-27 on port, forest and street: two builds each, same ``cloud.npz`` sha256).
Spheres are written in a canonical face order (``create_uvsphere`` is not
stable); ``batch.py build`` also pins Blender's Python ``str`` hash
(``PYTHONHASHSEED=0`` with ``--python-use-system-env``; ``--require-clean``
refuses a salted one) so a set of names cannot reorder a scene. Batch 1
(09-23) was built at ``--cloud-voxel 0.08 --cloud-zmax 4.5``; those flags
rebuild its clouds byte for byte only for scenes *without spheres* (37 of
42; checked on wall_slot_3022, corridor_3000 and ring_3008 against S3).
Batch 1 predates the canonical sphere order, so its sphere clouds were one
draw of an unstable face order: arena_3000-3004 rebuild with the same
placements but a different cloud (arena_3003: 7,556 of 340,626 points
differ, every one on a sphere; two rebuilds at this code agree), and their
S3 routes and labels do not carry over to a rebuild.

``seconds_build`` is wall time (scene + cloud); ``cpu_seconds_build`` the
process CPU time to the same point, which a loaded box inflates far less
(the build gate reads it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zlib

import bmesh
import bpy
from mathutils import Matrix, Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bl_common import script_args, use_gpu  # noqa: E402
import course_semantics as CS  # noqa: E402  (stdlib only)
import provenance as PV  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
#: provenance fields ``course.json`` carries at its top level (contract 6); the rest is under ``provenance``
PROVENANCE_KEYS = ("git_sha", "dirty", "kits_order", "library_sha")

#: obstacle points: one per voxel of this size -- the primitive contract (patch.sample's
#: 0.1 m grid). Batch 1 and the 09-25 kit previews used 0.08 m; 0.10 m has ~35 % fewer
#: points for the same surfaces, and the labeller's clearance cost is linear in them.
CLOUD_VOXEL = 0.10
CLOUD_OVERSAMPLE = 600.0    # area samples per m^2 before the voxel thinning
FLOOR_SPACING = 0.25        # patch.sample's floor grid
#: Obstacle surfaces above this are left out of the cloud: an enclosed warehouse's roof
#: and upper walls would otherwise be ~90 % of the points and make labelling ~10x slower;
#: the depth images still show them. It must clear the v7 envelope (recipe v7m): labels
#: fly under a ceiling of ``patch.FLIGHT_Z`` + ``--climb-max`` = ``band.Band.ceiling`` (v9: 4.25 m)
#: (``data_make_dataset.base_params`` z_band), and the labeller inflates the cloud by
#: ``value_cost.Params.r_body`` = 0.35 m, so the body top reaches 4.60 m. 4.75 m keeps
#: 0.15 m (1.5 voxels) above that. Batch 1 was cut at 4.5 m -- 0.10 m *below* the body top.
CLOUD_ZMAX = 4.75
from band_shim import band as _band  # noqa: E402
_band.bind(globals(), CLOUD_ZMAX=lambda b: b.cloud_top)


# --- materials ------------------------------------------------------------------------------

def pbr_material(name: str, maps: dict, root: Path, tile_m: float) -> bpy.types.Material:
    """Principled BSDF on box-projected 1k maps, tiled every ``tile_m`` metres
    of *object* coordinates -- the procedural meshes are built in world
    coordinates on an identity object, so this is world-scale tiling."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / tile_m,) * 3
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    def image(key, colour):
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(root / maps[key]), check_existing=True)
        tex.image.colorspace_settings.name = "sRGB" if colour else "Non-Color"
        tex.projection, tex.projection_blend = "BOX", 0.25
        nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
        return tex

    if "diff" in maps:
        nt.links.new(image("diff", True).outputs["Color"], bsdf.inputs["Base Color"])
    if "rough" in maps:
        nt.links.new(image("rough", False).outputs["Color"], bsdf.inputs["Roughness"])
    if "nor_gl" in maps:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(image("nor_gl", False).outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def world_hdri(path: Path, rotation: float, strength: float) -> None:
    w = bpy.data.worlds.new("world")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(path))
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0.0, 0.0, rotation)
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength


# --- primitive meshes -----------------------------------------------------------------------

def _mesh_object(name: str, bm: bmesh.types.BMesh, coll, material, collide: bool = True,
                 canonical: bool = False):
    """``bm`` as a mesh object. ``canonical`` writes the faces in a fixed order,
    each from its lowest vertex: ``bmesh.ops.create_uvsphere`` emits its faces
    in an order that changes from call to call (26 orders in 30 calls; the
    vertices, ``create_cube`` and ``create_cone`` are stable), and the cloud
    samples a mesh's triangles in order, so two builds of one course put
    different points on every sphere (the street kit's bollard caps, 09-27)."""
    me = bpy.data.meshes.new(name)
    if canonical:
        bm.verts.index_update()
        faces = []
        for f in bm.faces:
            idx = [v.index for v in f.verts]
            k = idx.index(min(idx))
            faces.append(idx[k:] + idx[:k])
        faces.sort()
        me.from_pydata([tuple(v.co) for v in bm.verts], [], faces)
        me.update()
    else:
        bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(material)
    ob["collide"] = int(collide)
    coll.objects.link(ob)
    return ob


def _affine(R, t) -> Matrix:
    M = Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            M[i][j] = float(R[i][j])
        M[i][3] = float(t[i])
    return M


def add_box(name, centre, half, R, coll, mat):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bmesh.ops.transform(bm, matrix=_affine(np.asarray(R) @ np.diag(half), centre), verts=bm.verts)
    return _mesh_object(name, bm, coll, mat)


def add_cylinder(name, a, b, r, coll, mat, segments=48):
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = b - a
    L = float(np.linalg.norm(axis))
    z = axis / L
    x = np.cross(z, [0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z], 1)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments, radius1=r, radius2=r, depth=L)
    bmesh.ops.transform(bm, matrix=_affine(R, (a + b) / 2), verts=bm.verts)
    return _mesh_object(name, bm, coll, mat)


#: primitive sphere tessellation: ``(u, v)`` segments, or ``"auto"`` -- per sphere, enough segments that the
#: chord sag r (1 - cos(pi / v)) stays under SPHERE_SAG_M (round 2: sphere-heavy patches -- clutter_mixed,
#: low_line, low_post mounds, r up to 3.9 m -- failed the render depth-parity gate at 48 x 24, a 3 cm sag).
#: ``--sphere-segments``; the default keeps every existing build byte for byte.
SPHERE_SEGMENTS = (48, 24)
SPHERE_SAG_M = 0.002


def sphere_segments(r: float, spec=None) -> tuple[int, int]:
    spec = SPHERE_SEGMENTS if spec is None else spec
    if spec == "auto":
        v = math.ceil(math.pi / math.acos(max(-1.0, 1.0 - SPHERE_SAG_M / max(float(r), 1e-6))))
        v = int(min(128, max(24, v)))
        return 2 * v, v
    return int(spec[0]), int(spec[1])


def add_sphere(name, c, r, coll, mat):
    bm = bmesh.new()
    u, v = sphere_segments(r)
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=r)
    bmesh.ops.translate(bm, vec=Vector(c), verts=bm.verts)
    return _mesh_object(name, bm, coll, mat, canonical=True)


def wall_pieces(p: dict) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A wall with slots as solid boxes ``(centre, half, R)``: the along-wall
    axis is cut at every slot edge, and each piece keeps the heights no slot
    covers. Same solid as ``render_depth.hit_wall``."""
    p0, p1 = np.asarray(p["p0"], float), np.asarray(p["p1"], float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    d /= L
    yaw = math.atan2(d[1], d[0])
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0, 0, 1.0]])
    z0, z1, t = p["z0"], p["z1"], p["t"]
    slots = [(s - w / 2, s + w / 2, sz0, sz1) for s, w, sz0, sz1 in p.get("slots", [])]
    cuts = sorted({0.0, L, *[min(max(v, 0.0), L) for s in slots for v in s[:2]]})
    out = []
    for u0, u1 in zip(cuts[:-1], cuts[1:]):
        if u1 - u0 < 1e-6:
            continue
        um = (u0 + u1) / 2
        covered = sorted((sz0, sz1) for a, b, sz0, sz1 in slots if a < um < b)
        free, lo = [], z0
        for c0, c1 in covered:
            if c0 > lo:
                free.append((lo, min(c0, z1)))
            lo = max(lo, c1)
        if lo < z1:
            free.append((lo, z1))
        for f0, f1 in free:
            if f1 - f0 < 1e-6:
                continue
            c2 = p0[:2] + d[:2] * um
            out.append((np.array([c2[0], c2[1], (f0 + f1) / 2]), np.array([(u1 - u0) / 2, t / 2, (f1 - f0) / 2]), R))
    return out


def rpy_matrix(rpy) -> np.ndarray:
    """``render_depth.euler_xyz``: R = Rx(roll) Ry(pitch) Rz(yaw)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rx @ ry @ rz


#: per-theme scatter counts per 450 m^2 of patch at clutter 1.0, and the look
THEMES = {
    "yard":      {"counts": {"large": 5, "tall": 5, "stack": 9, "prop": 10}},
    "woodland":  {"counts": {"tree": 7, "large": 6, "tall": 5, "prop": 10}},
    "warehouse": {"counts": {"tall": 8, "large": 4, "stack": 10, "prop": 8}, "shell": True},
    "plaza":     {"counts": {"tree": 4, "large": 6, "tall": 6, "prop": 10}},
}
SHELL_MARGIN, SHELL_H = 3.0, 7.0
ROLE_ORDER = ("large", "tree", "tall", "stack", "prop")


def build_primitives(prims: list, coll, mats: dict, keepout: float = 3.0) -> list[dict]:
    """Every primitive as a mesh, dressed by kind. Returns 2-D keep-out
    footprints for the scatter: every solid, plus ``keepout`` metres in front
    of and behind every wall opening."""
    foot = []
    for i, p in enumerate(prims):
        k = p["kind"]
        if k == "wall":
            for j, (c, h, R) in enumerate(wall_pieces(p)):
                add_box(f"prim{i:02d}_wall{j}", c, h, R, coll, mats["wall"])
                foot.append({"type": "rect", "c": c[:2], "half": h[:2], "R": R[:2, :2]})
            p0, p1 = np.asarray(p["p0"], float), np.asarray(p["p1"], float)
            d = (p1 - p0) / np.linalg.norm(p1 - p0)
            for s, w, _, _ in p.get("slots", []):
                c = p0[:2] + d[:2] * s
                Rw = np.array([[d[0], -d[1]], [d[1], d[0]]])
                foot.append({"type": "rect", "c": c, "half": np.array([w / 2 + 0.4 * keepout, keepout]), "R": Rw})
        elif k == "box":
            R = rpy_matrix(p["rpy"])
            add_box(f"prim{i:02d}_box", p["c"], p["half"], R, coll, mats["wall"])
            foot.append({"type": "rect", "c": np.asarray(p["c"][:2]), "half": np.asarray(p["half"][:2]), "R": R[:2, :2]})
        elif k == "cyl_v":
            cx, cy = p["c"]
            add_cylinder(f"prim{i:02d}_cyl", (cx, cy, p["z0"]), (cx, cy, p["z1"]), p["r"], coll, mats["pole"])
            foot.append({"type": "disc", "c": np.array([cx, cy]), "r": p["r"]})
        elif k == "cyl_h":
            add_cylinder(f"prim{i:02d}_bar", p["a"], p["b"], p["r"], coll, mats["bar"])
            a, b = np.asarray(p["a"][:2]), np.asarray(p["b"][:2])
            L = float(np.linalg.norm(b - a))
            d = (b - a) / max(L, 1e-9)
            foot.append({"type": "rect", "c": (a + b) / 2, "half": np.array([L / 2, p["r"] + 0.6 * keepout]),
                         "R": np.array([[d[0], -d[1]], [d[1], d[0]]])})
        elif k == "sphere":
            add_sphere(f"prim{i:02d}_sph", p["c"], p["r"], coll, mats["rock"])
            foot.append({"type": "disc", "c": np.asarray(p["c"][:2]), "r": p["r"]})
        else:
            raise ValueError(f"unknown primitive kind {k!r}")
    return foot


def footprint_distance(xy: np.ndarray, f: dict) -> float:
    if f["type"] == "disc":
        return float(np.linalg.norm(xy - f["c"]) - f["r"])
    q = np.abs(f["R"].T @ (xy - f["c"])) - f["half"]
    return float(np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1]), 0.0))


# --- the warehouse shell ------------------------------------------------------------------------

def build_shell(bounds, coll, wall_mat, ceil_mat, h: float = SHELL_H, margin: float = SHELL_MARGIN,
                light_scale: float = 1.0, light_color=(1.0, 0.96, 0.9), rect=None):
    """Four walls and a roof round the patch, with light panels under the roof.
    All of it collides: it is within 20 m of the camera, so it is in the cloud.
    ``rect`` (x0, y0, x1, y1): the walls' inner faces, instead of ``bounds`` + ``margin``
    (re-skin: outside every label's map box)."""
    if rect is not None:
        x0, y0, x1, y1 = (float(v) for v in rect)
    else:
        x0, y0, x1, y1 = bounds[0] - margin, bounds[1] - margin, bounds[2] + margin, bounds[3] + margin
    t = 0.3
    I = np.eye(3)
    add_box("shell_wall_s", ((x0 + x1) / 2, y0 - t / 2, h / 2), ((x1 - x0) / 2 + t, t / 2, h / 2), I, coll, wall_mat)
    add_box("shell_wall_n", ((x0 + x1) / 2, y1 + t / 2, h / 2), ((x1 - x0) / 2 + t, t / 2, h / 2), I, coll, wall_mat)
    add_box("shell_wall_w", (x0 - t / 2, (y0 + y1) / 2, h / 2), (t / 2, (y1 - y0) / 2, h / 2), I, coll, wall_mat)
    add_box("shell_wall_e", (x1 + t / 2, (y0 + y1) / 2, h / 2), (t / 2, (y1 - y0) / 2, h / 2), I, coll, wall_mat)
    roof = add_box("shell_roof", ((x0 + x1) / 2, (y0 + y1) / 2, h + t / 2), ((x1 - x0) / 2 + t, (y1 - y0) / 2 + t, t / 2),
                   I, coll, ceil_mat)
    lamp = bpy.data.materials.new("light_panel")
    lamp.use_nodes = True
    bs = lamp.node_tree.nodes["Principled BSDF"]
    bs.inputs["Emission Color"].default_value = (*light_color, 1.0)
    bs.inputs["Emission Strength"].default_value = 12.0 * light_scale
    n_lights = 0
    for xc in np.arange(x0 + 3.0, x1 - 1.0, 6.0):
        for yc in np.arange(y0 + 3.0, y1 - 1.0, 6.0):
            add_box(f"shell_panel_{n_lights}", (xc, yc, h - 0.05), (0.6, 0.6, 0.05), I, coll, lamp)
            L = bpy.data.lights.new(f"shell_light_{n_lights}", "AREA")
            L.size, L.energy = 1.2, 900.0 * light_scale
            L.color = light_color
            ob = bpy.data.objects.new(L.name, L)
            ob.location = (float(xc), float(yc), h - 0.15)
            coll.objects.link(ob)
            n_lights += 1
    return {"rect": [x0, y0, x1, y1], "height": h, "lights": n_lights, "roof": roof.name, "margin": margin,
            "light_scale": light_scale, "light_color": list(light_color)}


# --- the asset library ----------------------------------------------------------------------

def attach_alpha(objs, alpha_maps: dict, root: Path) -> int:
    """Wire a separate alpha map into every material whose name carries its
    part (``leaves`` -> ``tree_small_02_leaves``; ``""`` -> every blended
    material), through the UVs the colour texture uses. Returns how many."""
    n = 0
    mats = {s.material for o in objs if o.type == "MESH" for s in o.material_slots if s.material}
    for m in mats:
        if not m.use_nodes:
            continue
        nt = m.node_tree
        bsdf = next((x for x in nt.nodes if x.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        # "blended" as the glTF importer marks a cut-out: Blender 5.2 reports every opaque material as
        # HASHED / DITHERED, so ``blend_method != "OPAQUE"`` wired the leaf map into bark and kettle bodies too
        blended = (m.blend_method == "BLEND" or getattr(m, "surface_render_method", "") == "BLENDED"
                   or bsdf.inputs["Alpha"].is_linked)
        part = next((p for p in sorted(alpha_maps, key=len, reverse=True)
                     if (p and p in m.name.lower()) or (not p and blended)), None)
        if part is None:
            continue
        col = bsdf.inputs["Base Color"]
        src_vec = None
        if col.is_linked and col.links[0].from_node.type == "TEX_IMAGE":
            v = col.links[0].from_node.inputs["Vector"]
            src_vec = v.links[0].from_socket if v.is_linked else None
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(root / alpha_maps[part]), check_existing=True)
        tex.image.colorspace_settings.name = "Non-Color"
        if src_vec is not None:
            nt.links.new(src_vec, tex.inputs["Vector"])
        for l in list(bsdf.inputs["Alpha"].links):
            nt.links.remove(l)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Alpha"])
        n += 1
    return n


def import_model(aid: str, entry: dict, root: Path, lib) -> dict:
    """Import a glTF into its own (excluded) collection, offset so that its
    footprint centre sits on the instance origin and its base on z = 0."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(root / entry["path"]))
    new = [o for o in bpy.data.objects if o not in before]
    if entry.get("alpha_maps"):
        attach_alpha(new, entry["alpha_maps"], root)
    coll = bpy.data.collections.new(f"lib_{aid}")
    lib.children.link(coll)
    for o in new:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        coll.objects.link(o)
    bpy.context.view_layer.update()
    pts = np.array([list(o.matrix_world @ Vector(c)) for o in new if o.type == "MESH" for c in o.bound_box])
    lo, hi = pts.min(0), pts.max(0)
    coll.instance_offset = Vector(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]))
    return {"coll": coll, "size": (hi - lo).tolist()}


def _instance(name, lib, xyz, yaw, s, coll):
    e = bpy.data.objects.new(name, None)
    e.instance_type, e.instance_collection = "COLLECTION", lib["coll"]
    e.location = tuple(float(v) for v in xyz)
    e.rotation_euler = (0.0, 0.0, float(yaw))
    e.scale = (s, s, s)
    e["collide"] = 1
    coll.objects.link(e)
    return e


def scatter(rng, rect, foot, roles_of, n_by_role, library, placed_coll, margin=0.5,
            stack_max_h=3.4, hang=None) -> list[dict]:
    """Rejection-sample placements: inside ``rect``, clear of every primitive
    footprint and opening approach, and of each other.

    ``stack`` assets go down as columns 2-4 units high (one or two columns
    side by side, at most ``stack_max_h``), which is how crates and barrels
    reach the 1-2.5 m flight band in a real yard. ``hang`` assets (with
    ``hang={"ceiling": h, "bottom": (lo, hi)}``) hang from the ceiling on a
    cable, their underside inside the flight band."""
    x0, y0, x1, y1 = rect
    placed = []
    for role in ROLE_ORDER + (("hang",) if hang else ()):
        ids = sorted(a for a in roles_of if role in roles_of[a])
        for _ in range(n_by_role.get(role, 0) if ids else 0):
            for _try in range(400):
                aid = str(rng.choice(ids))
                lib = library[aid]
                sx, sy, sz = lib["size"]
                s = float(rng.uniform(0.95, 1.05)) if role in ("stack", "hang") else float(rng.uniform(0.9, 1.15))
                cols, levels = 1, 1
                if role == "stack":
                    cols = 2 if rng.random() < 0.4 else 1
                    levels = int(np.clip(rng.integers(2, 5), 1, max(1, int(stack_max_h // (sz * s)))))
                r = 0.5 * s * math.hypot(sx * cols * 1.03, sy)
                xy = np.array([rng.uniform(x0 + 0.5, x1 - 0.5), rng.uniform(y0 + 0.5, y1 - 0.5)])
                if any(footprint_distance(xy, f) < r + 0.4 for f in foot):
                    continue
                if any(np.linalg.norm(xy - q["xy"]) < r + q["r"] + margin for q in placed):
                    continue
                yaw = float(rng.uniform(0, 2 * math.pi))
                ax = np.array([math.cos(yaw), math.sin(yaw)])
                n = len(placed)
                if role == "hang":
                    z = float(rng.uniform(*hang["bottom"]))
                    _instance(f"hang_{n:02d}_{aid}", lib, (*xy, z), yaw, s, placed_coll)
                    top = z + sz * s
                    if top < hang["ceiling"] - 0.02:
                        cable = add_cylinder(f"hang_{n:02d}_cable", (*xy, top), (*xy, hang["ceiling"]), 0.012,
                                             placed_coll, hang["cable_mat"], segments=8)
                        cable["collide"] = 1
                    placed.append({"asset": aid, "role": role, "xy": xy, "r": r, "yaw": yaw, "scale": s,
                                   "bottom_z": z, "size": [sx * s, sy * s, sz * s]})
                    break
                for c in range(cols):
                    base = xy + ax * (c - (cols - 1) / 2) * sx * s * 1.03
                    h = levels if cols == 1 else max(1, levels - int(rng.integers(0, 2)))
                    for k in range(h):
                        jit = rng.normal(0, 0.015, 2)
                        _instance(f"{role}_{n:02d}_{aid}_{c}{k}", lib, (*(base + jit), k * sz * s),
                                  yaw + rng.normal(0, 0.04), s, placed_coll)
                placed.append({"asset": aid, "role": role, "xy": xy, "r": r, "yaw": yaw, "scale": s,
                               "columns": cols, "levels": levels,
                               "size": [sx * s * cols, sy * s, sz * s * levels]})
                break
    return placed


def prune_library(library: dict, used: set) -> None:
    """Drop the models no instance uses, so the saved .blend holds only the scene."""
    for aid, lib in library.items():
        if aid in used:
            continue
        for o in list(lib["coll"].objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(lib["coll"])
    bpy.data.orphans_purge(do_recursive=True)


# --- mesh + cloud export --------------------------------------------------------------------

#: the whole surface shader decides alpha (``alpha_expr``: Mix/Add Shader + Transparent BSDF,
#: node groups, glTF MASK cut-off chains) instead of the first image upstream of the Principled
#: Alpha. Off by default so existing kit clouds rebuild byte for byte; the Wenshan kit turns it on.
ALPHA_EXACT = False
#: ``pack_images`` also downscales *16-bit* PNG / TIFF maps (Blender loads them as float buffers,
#: so the 8-bit rule skipped them): Wenshan's scenes carry dozens of 4096^2 16-bit normal and
#: roughness maps, 1.3 GB of a 2.2 GB packed course. True HDR files (.exr, .hdr: the sky, the
#: terrain height field) are never touched. Off by default (existing courses pack as before); a
#: kit turns it on with ``PACK_16BIT = True``.
PACK_16BIT = False
_HDR_EXT = (".exr", ".hdr")


def _alpha_source(mat, exact: bool | None = None):
    """How a material's alpha is decided, as the renderer sees it:
    ``None`` (opaque), ``("const", a)`` or ``("image", image, use_alpha_channel, uv_name)``
    -- the first image texture upstream of the Principled BSDF's Alpha input.
    With ``ALPHA_EXACT`` (or ``exact``): ``alpha_expr.alpha_source`` (``("expr", key, expr, uv)``)."""
    if ALPHA_EXACT if exact is None else exact:
        import alpha_expr as AE  # noqa: PLC0415
        return AE.alpha_source(mat, lambda m: _alpha_source(m, exact=False))
    if mat is None or not mat.use_nodes:
        return None
    bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return None
    inp = bsdf.inputs["Alpha"]
    if not inp.is_linked:
        return None if inp.default_value >= 0.999 else ("const", float(inp.default_value))
    todo, seen = [(inp.links[0].from_node, inp.links[0].from_socket.name)], set()
    while todo:
        node, sock = todo.pop(0)
        if node.name in seen:
            continue
        seen.add(node.name)
        if node.type == "TEX_IMAGE" and node.image is not None:
            uv = None
            vec = node.inputs["Vector"]
            if vec.is_linked and vec.links[0].from_node.type == "UVMAP":
                uv = vec.links[0].from_node.uv_map or None
            return ("image", node.image, sock == "Alpha", uv)
        for i in node.inputs:
            for l in i.links:
                todo.append((l.from_node, l.from_socket.name))
    return None


class AlphaLookup:
    """Texture alpha at surface samples, the way Cycles' Depth pass sees it
    with ``pass_alpha_threshold`` = 0.5 (``render_frames``)."""

    def __init__(self):
        self.images = {}

    def pixels(self, img):
        if img.name not in self.images:
            W, H = img.size
            px = np.empty(W * H * img.channels, np.float32)
            img.pixels.foreach_get(px)
            self.images[img.name] = (px.reshape(H, W, img.channels), W, H)
        return self.images[img.name]

    def alpha(self, src, uv: np.ndarray) -> np.ndarray:
        kind = src[0]
        if kind == "const":
            return np.full(len(uv), src[1], np.float32)
        if kind == "expr":
            import alpha_expr as AE  # noqa: PLC0415
            return AE.evaluate(src[2], uv, self)
        px, W, H = self.pixels(src[1])
        u = np.mod(uv[:, 0], 1.0); v = np.mod(uv[:, 1], 1.0)
        i = np.clip((v * H).astype(int), 0, H - 1); j = np.clip((u * W).astype(int), 0, W - 1)
        if src[2] and px.shape[2] >= 4:
            return px[i, j, 3]
        return px[i, j, :3].mean(-1)


def collidable_triangles(zone=None, zmax=None):
    """World-space triangles of everything tagged ``collide``, instances
    realised, with what the cloud needs to honour alpha cut-outs: per face its
    alpha source id (-1 = opaque) and its three corner UVs.

    ``zone`` (x0, y0, x1, y1) and ``zmax`` drop faces wholly outside the
    zone or wholly above ``zmax`` (the cloud keeps neither); a library object
    instanced many times is read once."""
    dg = bpy.context.evaluated_depsgraph_get()
    Vs, Fs, ids, names, srcs_f, uvs_f, nv = [], [], [], [], [], [], 0
    sources, src_index = [], {}
    cache = {}
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type != "MESH":
            continue
        tag = inst.parent if inst.is_instance else ob
        if not int(tag.original.get("collide", 0)):
            continue
        ck = ob.original.name if inst.is_instance else None
        if ck is not None and ck in cache:
            got = cache[ck]
        else:
            got = _mesh_arrays(ob, sources, src_index)
            if ck is not None:
                cache[ck] = got
        if got is None:
            continue
        V, F, fsrc, fuv = got
        M = np.array(inst.matrix_world)
        V = V @ M[:3, :3].T + M[:3, 3]
        if zone is not None or zmax is not None:
            T = V[F]
            keep = np.ones(len(F), bool)
            if zmax is not None:
                keep &= T[:, :, 2].min(1) <= zmax
            if zone is not None:
                keep &= (T[:, :, 0].max(1) >= zone[0]) & (T[:, :, 0].min(1) <= zone[2]) & \
                        (T[:, :, 1].max(1) >= zone[1]) & (T[:, :, 1].min(1) <= zone[3])
            if not keep.any():
                continue
            Fk = F[keep]
            used, inv = np.unique(Fk.ravel(), return_inverse=True)
            V, F, fsrc_i, fuv_i = V[used], inv.reshape(-1, 3), fsrc[keep], fuv[keep]
        else:
            fsrc_i, fuv_i = fsrc, fuv
        nt = len(F)
        Vs.append(V)
        Fs.append(F + nv)
        ids.append(np.full(nt, len(names), np.int32))
        srcs_f.append(fsrc_i)
        uvs_f.append(fuv_i)
        names.append(f"{tag.name}/{ob.name}" if inst.is_instance else ob.name)
        nv += len(V)
    return (np.concatenate(Vs), np.concatenate(Fs), np.concatenate(ids), names,
            np.concatenate(srcs_f), np.concatenate(uvs_f), sources)


def _mesh_arrays(ob, sources, src_index):
    """Local-space (V, F, per-face alpha source, per-face corner UVs) of an
    evaluated mesh object, registering its alpha sources; None if empty."""
    me = ob.to_mesh()
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    V = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", V)
    F = np.empty(nt * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", F)
    mi = np.empty(nt, np.int64)
    me.loop_triangles.foreach_get("material_index", mi)
    loops = np.empty(nt * 3, np.int64)
    me.loop_triangles.foreach_get("loops", loops)
    per_slot = []
    for slot in ob.material_slots:
        src = _alpha_source(slot.material)
        if src is None:
            per_slot.append((-1, None))
            continue
        key = (src[0], src[1] if src[0] in ("const", "expr") else src[1].name, src[2] if src[0] == "image" else None)
        if key not in src_index:
            src_index[key] = len(sources)
            sources.append(src)
        per_slot.append((src_index[key], src[3] if src[0] in ("image", "expr") else None))
    fsrc = np.full(nt, -1, np.int32)
    fuv = np.zeros((nt, 3, 2), np.float32)
    for si, (sid, uvname) in enumerate(per_slot):
        sel = mi == si
        if sid < 0 or not sel.any():
            continue
        fsrc[sel] = sid
        layer = me.uv_layers.get(uvname) if uvname else me.uv_layers.active
        if layer is not None:
            uvall = np.empty(len(layer.data) * 2, np.float32)
            layer.data.foreach_get("uv", uvall)
            uvall = uvall.reshape(-1, 2)
            fuv[sel] = uvall[loops.reshape(-1, 3)[sel]]
    ob.to_mesh_clear()
    if not nt:
        return None
    return V.reshape(-1, 3), F.reshape(-1, 3), fsrc, fuv


def sample_cloud(V, F, rng, bounds, fsrc=None, fuv=None, sources=(), threshold=0.5, zone=None,
                 voxel: float | None = None, zmax: float | None = None, face_label=None, ground=None):
    """Area-weighted surface samples thinned to one per ``voxel`` (default
    ``CLOUD_VOXEL``) voxel, cut at ``zmax`` (default ``CLOUD_ZMAX``), plus
    ``patch.sample``'s floor grid. Samples on an alpha-cut surface are
    kept only where its alpha reaches ``threshold`` -- the leaves a camera
    sees, not the transparent card they are painted on. ``face_label`` (int per face):
    also return the label of every obstacle point (-1 for the floor grid) as a fifth value;
    the points are the same with or without it.

    ``ground`` (terrain mode, a ``wenshan_scene.HeightGrid`` in the patch frame; None = the flat
    floor, byte for byte as before): the cut is ``zmax`` above the grid's HIGHEST ground, and the floor
    grid becomes the terrain sheet -- the same 0.25 m grid laid on the ground over the whole cloud
    ``zone`` (not only the bounds: the labeller's own floor rule no longer covers the zone, see
    ``terrain_floor``), plus a skirt down the zone's edge to the lowest ground."""
    voxel = CLOUD_VOXEL if voxel is None else float(voxel)
    zmax = CLOUD_ZMAX if zmax is None else float(zmax)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    total = float(area.sum())
    n = int(total * CLOUD_OVERSAMPLE) + 1
    tri = rng.choice(len(F), size=n, p=area / total)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    P = w[:, :1] * A[tri] + w[:, 1:2] * B[tri] + w[:, 2:] * C[tri]
    keep = np.ones(n, bool)
    dropped = 0
    if fsrc is not None and (fsrc >= 0).any():
        look = AlphaLookup()
        ts = fsrc[tri]
        for sid, src in enumerate(sources):
            sel = ts == sid
            if not sel.any():
                continue
            uv = np.einsum("nk,nkd->nd", w[sel], fuv[tri[sel]])
            keep[sel] = look.alpha(src, uv) >= threshold
        dropped = int((~keep).sum())
    opaque_v = np.zeros(len(V), bool)
    opaque_v[F[(fsrc < 0) if fsrc is not None else slice(None)].ravel()] = True
    P = np.concatenate([P[keep], V[opaque_v]])    # vertices too (opaque faces): thin parts keep their tips
    lab = None
    if face_label is not None:
        vlab = np.full(len(V), -1, np.int32)
        fo = (fsrc < 0) if fsrc is not None else np.ones(len(F), bool)
        vlab[F[fo].ravel()] = np.repeat(np.asarray(face_label, np.int32)[fo], 3)
        lab = np.concatenate([np.asarray(face_label, np.int32)[tri][keep], vlab[opaque_v]])
    # terrain mode: one absolute top at the zone's HIGHEST ground + zmax. The labeller's per-state ceiling is
    # the higher of the ground under the state and its goal + 4.25 (terrain.z_band), so a row can pass over a
    # dip at up to (highest ground + 4.25): a cut relative to the local ground (W17 1be4150..301e192) left
    # geometry above a dip out of the cloud the labeller planned in (W18, w17p1: 20 rows over the cut)
    m = P[:, 2] <= zmax if ground is None else P[:, 2] <= zmax + ground.zmax
    if zone is not None:                        # place kits: scenery beyond the cloud zone is render-only
        m &= (P[:, 0] >= zone[0]) & (P[:, 0] <= zone[2]) & (P[:, 1] >= zone[1]) & (P[:, 1] <= zone[3])
    P = P[m]
    _, first = np.unique(np.floor(P / voxel).astype(np.int64), axis=0, return_index=True)
    obs = P[np.sort(first)]
    if ground is None:
        x0, y0, x1, y1 = bounds
        xs = np.arange(x0, x1 + FLOOR_SPACING / 2, FLOOR_SPACING)
        ys = np.arange(y0, y1 + FLOOR_SPACING / 2, FLOOR_SPACING)
        X, Y = np.meshgrid(xs, ys)
        floor = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    else:
        floor = terrain_floor(ground, zone if zone is not None else bounds)
    pts = np.concatenate([obs, floor])
    reg = np.concatenate([np.zeros(len(obs), np.int16), np.full(len(floor), -1, np.int16)])
    if lab is not None:
        inst = np.concatenate([lab[m][np.sort(first)], np.full(len(floor), -1, np.int32)])
        return pts, reg, total, dropped, inst
    return pts, reg, total, dropped


def terrain_floor(ground, rect, spacing: float = FLOOR_SPACING) -> np.ndarray:
    """Terrain mode's floor (region -1): the 0.25 m grid over ``rect`` laid on ``ground``, plus a skirt of
    points down every edge of ``rect`` from the sheet to the lowest ground under it. The skirt closes the
    pockets under a rise from the side: with the sheet alone the space under a hill reaches the open
    world past the zone's edge, and a roadmap (TopologyPRM's box reaches 2 m past the cloud) could route
    beneath the ground."""
    x0, y0, x1, y1 = (float(v) for v in rect)
    xs = np.arange(x0, x1 + spacing / 2, spacing)
    ys = np.arange(y0, y1 + spacing / 2, spacing)
    X, Y = np.meshgrid(xs, ys)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    Z = ground(XY)
    sheet = np.column_stack([XY, Z])
    zlo = float(Z.min())
    edge = np.concatenate([np.column_stack([xs, np.full(len(xs), y0)]), np.column_stack([xs, np.full(len(xs), y1)]),
                           np.column_stack([np.full(len(ys) - 2, x0), ys[1:-1]]),
                           np.column_stack([np.full(len(ys) - 2, x1), ys[1:-1]])])
    ze = ground(edge)
    skirt = [np.column_stack([e[None].repeat(len(zz), 0), zz])
             for e, top in zip(edge, ze) if (zz := np.arange(top - spacing, zlo - 1e-9, -spacing)).size]
    return np.concatenate([sheet, *skirt]) if skirt else sheet


def write_ply(path: Path, V: np.ndarray, F: np.ndarray) -> None:
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n").encode()
    faces = np.zeros(len(F), dtype=[("n", "u1"), ("v", "<i4", (3,))])
    faces["n"], faces["v"] = 3, F
    with open(path, "wb") as f:
        f.write(head)
        f.write(V.astype("<f4").tobytes())
        f.write(faces.tobytes())


# --- previews -------------------------------------------------------------------------------

def look_at(ob, eye, target) -> None:
    ob.location = Vector(eye)
    ob.rotation_euler = (Vector(target) - Vector(eye)).to_track_quat("-Z", "Y").to_euler()


def previews(out: Path, spec: dict, shell: dict | None, eye_shots: bool = False, res=(1280, 720),
             samples: int = 128, indoor: bool = False) -> list[str]:
    """Chase (behind the patch, low), oblique and top. Indoors the roof and its
    lights are hidden for the oblique and top shots, and the chase stays low.
    ``eye_shots`` adds the drone's own view: 1.5 m up at the start looking at
    the goal (``eye``) and from the goal back (``eye_back``), 90 deg wide.
    Objects carrying a ``roof`` property are hidden like the shell's roof."""
    scene = bpy.context.scene
    cam = bpy.data.objects.new("preview_cam", bpy.data.cameras.new("preview_cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    x0, y0, x1, y1 = spec["bounds"]
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    span = max(x1 - x0, y1 - y0)
    low = bool(shell) or indoor
    shots = {"chase": ((x0 - (1.0 if low else 3.0), yc, 3.2 if low else 4.0), (xc, yc, 1.2), 30.0 if low else 50.0),
             "side": ((xc, y0 - 0.5 * span, 0.35 * span + 3), (xc, yc, 0.5), 35.0),
             "top": ((xc, yc, 1.6 * span), (xc, yc, 0.0), 35.0)}
    if eye_shots:
        g = spec["goal_pos"]
        shots["eye"] = ((0.0, 0.0, 1.5), (g[0], g[1], 1.5), 18.0)
        shots["eye_back"] = ((g[0], g[1], 1.5), (0.0, 0.0, 1.5), 18.0)
    roof = [o for o in bpy.data.objects if o.name.startswith(("shell_roof", "shell_panel", "shell_wall_s"))
            and o.type == "MESH"] if shell else []
    roof += [o for o in bpy.data.objects if o.get("roof")]
    saved = []
    old = (scene.render.resolution_x, scene.render.resolution_y, scene.cycles.samples)
    scene.render.resolution_x, scene.render.resolution_y, scene.cycles.samples = res[0], res[1], samples
    for name, (eye, target, lens) in shots.items():
        for o in roof:
            o.hide_render = name in ("side", "top")
        cam.data.lens = lens
        look_at(cam, eye, target)
        scene.render.filepath = str(out / f"preview_{name}.png")
        bpy.ops.render.render(write_still=True)
        saved.append(f"preview_{name}.png")
    for o in roof:
        o.hide_render = False
    scene.render.resolution_x, scene.render.resolution_y, scene.cycles.samples = old
    bpy.data.objects.remove(cam)
    return saved


# --- packing + provenance -------------------------------------------------------------------

def _pack_cache_file(cache: Path, src: str, max_res: int, quality: int, fmt: str) -> Path:
    """Where the downscaled copy of library file ``src`` lives in ``--pack-cache``:
    keyed by its real path, size, mtime and the downscale settings."""
    st = os.stat(src)
    key = hashlib.sha256(f"{os.path.realpath(src)}|{st.st_size}|{st.st_mtime_ns}|{max_res}|{quality}|{fmt}"
                         .encode()).hexdigest()[:32]
    return cache / key[:2] / f"{key}.{'jpg' if fmt == 'JPEG' else 'png'}"


def pack_images(max_res: int = 0, quality: int = 90, tmp: Path | None = None, cache: Path | None = None) -> dict:
    """Pack every external file into the .blend (``--pack``), so a render node
    needs only ``course.blend``. With ``max_res`` > 0 every 8-bit image whose
    longer side exceeds it is first downscaled (aspect kept) and re-encoded in
    its own format -- JPEG at ``quality``, anything else as PNG (alpha kept) --
    under ``tmp``, and that file is what gets packed. Float images (the HDRI
    sky, EXR/16-bit maps) are packed as they are: the sky is seen at the
    camera's own resolution. The camera is 640x480 at 90 deg (~7 px/deg), so a
    1k texel on a 2 m tile (2 mm) is finer than a pixel beyond ~0.5 m.

    Runs after the cloud is sampled: the cloud's alpha cut-outs read the
    full-resolution maps, so ``cloud.npz`` is the same with or without it (the
    render's 0.5 alpha threshold on the downscaled map can move a leaf edge by
    a texel). ``cache`` (``--pack-cache``) keeps each library file's downscaled
    copy for the next build (the same bytes a fresh downscale writes; written
    by atomic rename, so parallel builds may share it): the downscale is most
    of the packing time (~0.2 s an image, 60-160 images a kit scene). Returns
    the stats recorded in ``course.json`` ``pack``."""
    st = {"max_res": int(max_res), "quality": int(quality), "images": 0, "downscaled": 0, "float_kept": 0,
          "alpha_png": 0, "bytes_in": 0, "bytes_out": 0, "cache_hits": 0}
    imgs = sorted((im for im in bpy.data.images if im.source == "FILE" and im.users), key=lambda im: im.name)
    st["images"] = len(imgs)
    if max_res > 0:
        tmp.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(imgs):
            if img.is_float and not (PACK_16BIT and not bpy.path.abspath(img.filepath).lower().endswith(_HDR_EXT)):
                st["float_kept"] += 1
                continue
            w, h = img.size
            if max(w, h) <= max_res:
                continue
            src = bpy.path.abspath(img.filepath)
            st["bytes_in"] += img.packed_file.size if img.packed_file else (Path(src).stat().st_size
                                                                            if Path(src).exists() else 0)
            fmt = "JPEG" if img.file_format == "JPEG" else "PNG"
            if fmt == "PNG" and img.channels == 4:
                st["alpha_png"] += 1
            cached = (_pack_cache_file(cache, src, max_res, quality, fmt)
                      if cache is not None and not img.packed_file and Path(src).exists() else None)
            if cached is not None and cached.exists():
                p = cached
                st["cache_hits"] += 1
            else:
                f = max_res / max(w, h)
                img.scale(max(1, round(w * f)), max(1, round(h * f)))
                p = tmp / f"{i:05d}.{'jpg' if fmt == 'JPEG' else 'png'}"
                img.file_format = fmt
                img.save(filepath=str(p), quality=int(quality))
                if cached is not None:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    part = cached.with_name(f"{cached.name}.{os.getpid()}.part")
                    shutil.copyfile(p, part)
                    os.replace(part, cached)
            if img.packed_file:
                img.unpack(method="REMOVE")
            img.filepath = str(p)
            img.reload()
            st["bytes_out"] += p.stat().st_size
            st["downscaled"] += 1
    bpy.ops.file.pack_all()
    st["unpacked_left"] = sorted(im.name for im in bpy.data.images
                                 if im.source in ("FILE", "SEQUENCE", "TILED") and im.users and not im.packed_file)
    return st


def save_blend(out: Path, a) -> dict:
    """``course.blend`` (compressed, no ``.blend1``), packed when ``--pack``.
    Returns ``{"packed", "blend_bytes", ...pack_images stats}``."""
    bpy.context.preferences.filepaths.save_version = 0        # no course.blend1 next to it
    st = {"packed": bool(a.pack)}
    tmp = out / ".pack_tmp"
    if a.pack:
        st.update(pack_images(a.pack_max_res, a.pack_quality, tmp, a.pack_cache))
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "course.blend"), compress=True)
    shutil.rmtree(tmp, ignore_errors=True)
    st["blend_bytes"] = (out / "course.blend").stat().st_size
    return st


def blender_info() -> dict:
    h = bpy.app.build_hash
    return {"version": bpy.app.version_string, "build_hash": h.decode() if isinstance(h, bytes) else str(h)}


def cloud_info(out: Path, pts, reg, voxel: float, zmax: float, dropped: int) -> dict:
    obs = pts[reg >= 0]
    return {"points": int(len(pts)), "obstacle_points": int(len(obs)), "floor_points": int((reg < 0).sum()),
            "voxel_m": voxel, "floor_spacing_m": FLOOR_SPACING, "zmax_m": zmax,
            "obstacle_zmax_m": round(float(obs[:, 2].max()), 4) if len(obs) else None,
            "samples_dropped_by_alpha": dropped, "sha256": PV.file_sha256(out / "cloud.npz")}


# --- place kits (v2) ------------------------------------------------------------------------

def setup_render(samples: int, cpu: bool, threads: int = 0) -> str:
    """Cycles for the per-state RGB: denoised, adaptive, clamped indirect
    light (no fireflies off wet asphalt), AgX. Returns the device backend."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if cpu:
        scene.cycles.device = "CPU"
        backend = "CPU"
    else:
        backend = use_gpu()
    if threads:
        scene.render.threads_mode, scene.render.threads = "FIXED", threads
    scene.cycles.samples = samples
    scene.cycles.adaptive_threshold = 0.015
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPTIX" if backend == "OPTIX" else "OPENIMAGEDENOISE"
    except TypeError:
        pass
    scene.cycles.sample_clamp_indirect = 8.0
    scene.cycles.max_bounces, scene.cycles.diffuse_bounces, scene.cycles.glossy_bounces = 8, 3, 3
    scene.cycles.transmission_bounces, scene.cycles.volume_bounces = 6, 0
    scene.cycles.caustics_reflective = scene.cycles.caustics_refractive = False
    scene.view_settings.view_transform = "AgX"
    scene.render.image_settings.file_format = "PNG"
    return backend


def kit_args_of(v: str | None) -> dict:
    """``--kit-args``: a JSON object, or ``@path`` of one."""
    if not v:
        return {}
    txt = Path(v[1:]).read_text() if v.startswith("@") else v
    d = json.loads(txt)
    if not isinstance(d, dict):
        raise ValueError("--kit-args must be a JSON object")
    return d


def build_kit(a, spec: dict, seed: int, t0: float) -> None:
    """Build ``spec`` as a place with kit ``a.kit`` (see blender/kits/)."""
    import kits  # noqa: PLC0415
    from kits.base import CLOUD_MARGIN, Ctx  # noqa: PLC0415
    from kits.library import Library  # noqa: PLC0415
    from kits.looks import Looks  # noqa: PLC0415

    global ALPHA_EXACT
    kit = kits.load(a.kit)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.spec.resolve() != (out / "spec.json").resolve():
        (out / "spec.json").write_text(a.spec.read_text())
    kit_args = kit_args_of(a.kit_args)
    if getattr(kit, "BACKDROP", False):             # a backdrop kit opens its own scene (kits/wenshan.py)
        kit.open_base(a, kit_args)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    ALPHA_EXACT = bool(getattr(kit, "ALPHA_EXACT", False))
    global PACK_16BIT
    PACK_16BIT = bool(getattr(kit, "PACK_16BIT", False))
    samples = a.samples if a.samples != 64 else 128           # kit courses default to 128 spp
    backend = setup_render(samples, a.cpu, a.threads)
    fam = zlib.crc32(spec["family"].encode())          # corridor_5001 and arena_5001 share a seed
    kit_i = kits.KITS.index(a.kit) if a.kit in kits.KITS else len(kits.KITS) + kits.BACKDROPS.index(a.kit)
    rng = np.random.default_rng([seed, 101, kit_i, a.variant, fam])
    lib = Library(a.assets)
    looks = Looks(lib, rng)
    knobs = {"clutter": a.clutter, "keepout": a.keepout, "hang": a.hang, "start_keepout": a.start_keepout}
    ctx = Ctx(spec, lib, looks, rng, knobs, a.level)
    ctx.cloud_zmax = float(a.cloud_zmax)             # kits that decide collision by height read it (cloud_top)
    ctx.kit_args = kit_args
    ctx.info["kit"] = a.kit
    chosen = kit.build(ctx) or {}
    if spec["primitives"] and not len(ctx.colls["obstacles"].objects):
        raise RuntimeError(f"kit {a.kit} did not dress the patch primitives")
    kinfo = ctx.finish()
    shutter = a.shutter if a.shutter is not None else float(rng.choice([0.0, 1 / 500, 1 / 250, 1 / 125, 1 / 60]))
    bpy.context.view_layer.update()

    ground = getattr(ctx, "ground", None)               # terrain mode (kits/wenshan.py): the floor follows it
    if ground is not None and not hasattr(ground, "zmax"):
        ground = None                                   # a plain kit: Ctx.ground is the floor-plane METHOD, not a grid
    V, F, face_obj, names, fsrc, fuv, sources = collidable_triangles(
        zone=ctx.zone, zmax=a.cloud_zmax if ground is None else a.cloud_zmax + ground.zmax)
    inst_of = getattr(ctx, "inst_of_name", None)        # backdrop kits: the scene's own instance ids
    piece_cat = None
    if inst_of is None and getattr(a, "piece_ids", False):
        import piece_ids as PI  # noqa: PLC0415
        piece_of_name = {o.name: int(o["piece"]) for o in bpy.data.objects if o.get("piece") is not None}
        inst_of, cat_rows = PI.labels_for(names, piece_of_name, getattr(ctx, "pieces", []))
        piece_cat = PI.catalogue(a.kit, cat_rows)
        (out / "catalogue.json").write_text(json.dumps(piece_cat, indent=1))
    face_label = None if inst_of is None else np.array(
        [inst_of.get(n, inst_of.get(n.split("/")[0], inst_of.get(n.split("/")[-1], -1))) for n in names],
        np.int32)[face_obj]
    got = sample_cloud(V, F, np.random.default_rng([seed, 78]), spec["bounds"], fsrc, fuv, sources, zone=ctx.zone,
                       voxel=a.cloud_voxel, zmax=a.cloud_zmax, face_label=face_label,
                       **({} if ground is None else {"ground": ground}))
    pts, reg, area, dropped = got[:4]
    inst = got[4] if len(got) > 4 else None
    floor_drop = float(getattr(kit, "CLOUD_FLOOR_DROP_M", 0.0))
    if inst is not None and floor_drop > 0:          # backdrop plinths / pads at the floor: the floor grid's, not obstacles
        zf = pts[:, 2] if ground is None else pts[:, 2] - ground(pts[:, :2])     # terrain: above the local ground
        k = ~((reg >= 0) & (inst >= 0) & (zf < floor_drop))
        pts, reg, inst = pts[k], reg[k], inst[k]
    np.savez_compressed(out / "cloud.npz", points=pts, region=reg, **({} if inst is None else {"inst": inst}))
    if ground is not None:                           # terrain mode: the height grid the floor was laid on
        np.savez_compressed(out / "terrain.npz", **ground.to_npz_dict())
    if hasattr(kit, "after_cloud"):
        kit.after_cloud(ctx, pts, reg, inst, chosen)
    if a.export_mesh:
        np.savez_compressed(out / "mesh.npz", V=V.astype(np.float32), F=F.astype(np.int32),
                            face_obj=face_obj, names=np.array(names), alpha_cut=(fsrc >= 0))
    t_build, c_build = time.time() - t0, time.process_time()
    blend = save_blend(out, a)
    t_blend = time.time() - t0 - t_build
    shots = []
    if a.preview:
        w, h = map(int, a.preview_res.split("x"))
        shots = previews(out, spec, None, eye_shots=True, res=(w, h), samples=a.preview_samples,
                         indoor=bool(getattr(kit, "INDOOR", False)))
    sky = chosen.get("sky", {})
    source = chosen.pop("source", None) or {"kind": "kit"}
    if piece_cat is not None:
        source = {**source, "piece_ids": True, "catalogue": "catalogue.json", "n_pieces": piece_cat["n_pieces"],
                  "catalogue_classes": piece_cat["classes"],
                  "conventions": {**(source.get("conventions") or {}), "inst": piece_cat["conventions"]["inst"]}}
    prov = PV.provenance(REPO, lib.index_files, kits_order=kits.KITS, argv=sys.argv, blender=blender_info(),
                         git=a.git)
    info = {
        "patch": f"{spec['family']}_{spec['seed']}", "seed": seed, "kit": a.kit, "variant": a.variant,
        "theme": a.kit, "level": a.level, "indoor": bool(getattr(kit, "INDOOR", False)),
        "knobs": {**knobs}, "blender": bpy.app.version_string, "device": backend,
        **{k: prov[k] for k in PROVENANCE_KEYS}, "provenance": prov,
        "hdri": sky.get("hdri"), "exposure": sky.get("exposure", bpy.context.scene.view_settings.exposure),
        "shutter_s": shutter, "chosen": chosen, "kit_info": kinfo, "materials": looks.used,
        # the kit's changes to the patch's own geometry, one shape for every kit (course_semantics);
        # chosen.semantics_changed keeps the kit's own format
        "semantics_changed": CS.from_chosen(chosen),
        "source": source, "alpha_rule": "exact" if ALPHA_EXACT else "legacy",
        "primitives": len(spec["primitives"]), "cloud_zone": list(ctx.zone), "cloud_margin_m": CLOUD_MARGIN,
        "mesh": {"vertices": int(len(V)), "triangles": int(len(F)), "objects": len(names),
                 "area_m2": round(area, 2), "alpha_cut_triangles": int((fsrc >= 0).sum())},
        "cloud": cloud_info(out, pts, reg, a.cloud_voxel, a.cloud_zmax, dropped),
        "files": lib.files_used(), "pack": blend,
        "rgb_samples": samples, "seconds_build": round(t_build, 1), "cpu_seconds_build": round(c_build, 1),
        "seconds_blend": round(t_blend, 1),
        "previews": shots, "seconds_total": round(time.time() - t0, 1),
    }
    (out / "course.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("patch", "kit", "hdri", "exposure", "mesh", "cloud", "seconds_build",
                                           "seconds_total")}))


# --- main -----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--theme", default="yard", choices=sorted(THEMES))
    ap.add_argument("--seed", type=int, default=None, help="default: the patch seed")
    ap.add_argument("--clutter", type=float, default=1.0, help="scatter count multiplier (per 450 m^2)")
    ap.add_argument("--keepout", type=float, default=3.0,
                    help="metres kept clear in front of and behind every wall opening (and round bars)")
    ap.add_argument("--start-keepout", type=float, default=0.0,
                    help="metres kept clear round the patch's start (the origin) and its goal; 0 = none (the default, as batch 1 built)")
    ap.add_argument("--hang", type=float, default=0.0,
                    help="warehouse: hanging obstacles per 450 m^2, underside in the flight band")
    ap.add_argument("--samples", type=int, default=64, help="Cycles samples for the per-state RGB")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--export-mesh", action="store_true", help="also write mesh.npz / mesh.ply (large with trees)")
    ap.add_argument("--level", default="", help="recorded only")
    ap.add_argument("--kit", default=None, help="a place kit (blender/kits/<kit>.py) instead of --theme; "
                    "--assets is then the v2 library root (library.json + catalog.json)")
    ap.add_argument("--kit-args", default=None,
                    help="kit courses: a JSON object (or @file) the kit reads as ctx.kit_args (the wenshan backdrop: "
                    "seq, t, window, root)")
    ap.add_argument("--cpu", action="store_true", help="Cycles on the CPU (previews on a box with no GPU)")
    ap.add_argument("--threads", type=int, default=0, help="CPU render threads (0 = all)")
    ap.add_argument("--preview-res", default="1280x720")
    ap.add_argument("--preview-samples", type=int, default=128)
    ap.add_argument("--shutter", type=float, default=None,
                    help="kit courses: RGB exposure time in seconds for camera motion blur (default: drawn, 0 = none)")
    ap.add_argument("--variant", type=int, default=0,
                    help="a second look at the same patch: kit courses, and (2026-10-01) theme courses -- "
                         "variant > 0 draws its own HDRI, rotation and strength, textures and (warehouse) "
                         "lights from its own stream; variant 0 is the build as before, byte for byte")
    ap.add_argument("--sphere-segments", default=None,
                    help="primitive sphere tessellation: 'U,V' or 'auto' (chord sag <= 2 mm per sphere, 48x24 .. "
                         "256x128); default 48,24 (every existing build unchanged)")
    ap.add_argument("--look-variant", type=int, default=0,
                    help="theme courses: a looks-only variant (HDRI, rotation, strength, textures, warehouse lights "
                         "from their own stream) with the scatter, cloud and labels of the variant-0 build; 0 = off")
    ap.add_argument("--shell-rect", default=None,
                    help="warehouse: x0,y0,x1,y1 of the shell walls' inner faces (overrides --shell-margin; re-skin: "
                         "the union of the labeller's map boxes + 0.5 m)")
    ap.add_argument("--shell-height", type=float, default=SHELL_H, help="warehouse: the shell's roof height (m)")
    ap.add_argument("--shell-margin", type=float, default=SHELL_MARGIN,
                    help="warehouse: the shell's walls this far outside the patch bounds (re-skin: 8 m, beyond "
                         "the labels' bounds + 6 m zone)")
    ap.add_argument("--pack", action="store_true",
                    help="pack every image into course.blend, so a render node needs only the .blend")
    ap.add_argument("--pack-max-res", type=int, default=0,
                    help="with --pack: downscale 8-bit images whose longer side exceeds this before packing "
                    "(0 = pack as they are; 1024 for the 640x480 camera). The cloud is sampled first, unaffected")
    ap.add_argument("--pack-quality", type=int, default=90, help="JPEG quality of the downscaled maps")
    ap.add_argument("--pack-cache", type=Path, default=None,
                    help="keep downscaled library maps here for later builds (e.g. /mnt/data/blender/pack_cache)")
    ap.add_argument("--cloud-voxel", type=float, default=CLOUD_VOXEL,
                    help=f"obstacle cloud voxel, m (default {CLOUD_VOXEL}; batch 1 was 0.08)")
    ap.add_argument("--cloud-zmax", type=float, default=CLOUD_ZMAX,
                    help=f"obstacle cloud top, m (default {CLOUD_ZMAX}; batch 1 was 4.5)")
    ap.add_argument("--piece-ids", action="store_true",
                    help="kit courses: write cloud.npz inst (one id per placed primitive / kit object, as Wenshan courses "
                         "carry metadata instance ids) and catalogue.json (blender/piece_ids.py); default off")
    ap.add_argument("--require-clean", action="store_true",
                    help="pipeline mode: refuse (exit 3) from a dirty checkout, with a kit test hook set, or with "
                    "Python's str hash salted (run as batch.py build does: PYTHONHASHSEED=0 --python-use-system-env)")
    a = ap.parse_args(script_args())
    global SPHERE_SEGMENTS
    if a.sphere_segments:
        SPHERE_SEGMENTS = "auto" if a.sphere_segments == "auto" else tuple(int(x) for x in a.sphere_segments.split(","))
    t0 = time.time()
    a.git = PV.git_state(REPO)                  # the code this build loads, recorded in course.json
    if a.require_clean:
        why = PV.check_clean({**a.git, "env_hooks": PV.env_hooks(),
                              "hash_randomization": bool(sys.flags.hash_randomization)})
        if why:
            print("build_course: REFUSED (--require-clean): " + " | ".join(why), flush=True)
            sys.exit(3)
    if a.pack_max_res and not a.pack:
        ap.error("--pack-max-res needs --pack")
    spec = json.loads(a.spec.read_text())
    seed = spec["seed"] if a.seed is None else a.seed
    if a.kit:
        build_kit(a, spec, seed, t0)
        return
    rng = np.random.default_rng([seed, 77, sorted(THEMES).index(a.theme)] + ([a.variant] if a.variant else []))
    root = a.assets.expanduser()
    manifest = json.loads((root / "manifest.json").read_text())["assets"]
    cfg = THEMES[a.theme]
    roles = {k: set(v.get("roles_by_theme", {}).get(a.theme, [])) for k, v in manifest.items()}
    roles = {k: r for k, r in roles.items() if r}
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.spec.resolve() != (out / "spec.json").resolve():
        (out / "spec.json").write_text(a.spec.read_text())      # the course carries its own layout

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    backend = use_gpu()
    scene.cycles.samples = a.samples
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPTIX" if backend == "OPTIX" else "OPENIMAGEDENOISE"
    except TypeError:
        pass
    scene.render.image_settings.file_format = "PNG"

    obst = bpy.data.collections.new("obstacles")
    scene.collection.children.link(obst)
    props = bpy.data.collections.new("scatter")
    scene.collection.children.link(props)
    shell_coll = bpy.data.collections.new("shell")
    scene.collection.children.link(shell_coll)
    lib = bpy.data.collections.new("library")
    scene.collection.children.link(lib)
    bpy.context.view_layer.layer_collection.children["library"].exclude = True

    def pick(role, kind):
        ids = sorted(k for k, r in roles.items() if manifest[k]["kind"] == kind and role in r)
        return str(rng.choice(ids)) if ids else None

    hdri = pick("sky", "hdri")
    tex = {r: pick(r, "texture") for r in ("wall", "pole", "bar", "rock", "ground", "ceiling")}
    tex = {r: t or tex["wall"] for r, t in tex.items()}
    hdri_rot = float(rng.uniform(0, 2 * math.pi))
    look_variant = int(getattr(a, "look_variant", 0) or 0)
    if look_variant and not a.variant:
        # a looks-only variant: the main stream is consumed exactly as variant 0's (so the scatter, the
        # cloud and every label stay the base course's), the looks come from their own stream
        lrng = np.random.default_rng([seed, 77, sorted(THEMES).index(a.theme), 9001, look_variant])
        rng, rng_main = lrng, rng
        hdri = pick("sky", "hdri")
        tex = {r: pick(r, "texture") for r in ("wall", "pole", "bar", "rock", "ground", "ceiling")}
        tex = {r: t or tex["wall"] for r, t in tex.items()}
        hdri_rot = float(rng.uniform(0, 2 * math.pi))
    # variant > 0 (re-skin looks): also the sky's strength and the warehouse lights; variant 0 draws
    # nothing more from rng here, so its build is unchanged
    look = {"hdri_strength": 1.0, "light_scale": 1.0, "light_color": (1.0, 0.96, 0.9)}
    if a.variant or (look_variant and not a.variant):
        look = {"hdri_strength": float(rng.uniform(0.6, 1.6)), "light_scale": float(rng.uniform(0.6, 1.4)),
                "light_color": tuple(float(c) for c in rng.choice([(1.0, 0.96, 0.9), (1.0, 0.85, 0.7),
                                                                    (0.9, 0.95, 1.0)]))}
    if look_variant and not a.variant:
        rng = rng_main
    world_hdri(root / manifest[hdri]["path"], hdri_rot, look["hdri_strength"])
    tiles = {"wall": 2.5, "pole": 1.5, "bar": 1.0, "rock": 2.0, "ground": 4.0, "ceiling": 3.0}
    mats = {r: pbr_material(f"{r}_{t}", manifest[t]["maps"], root, tile_m=tiles[r]) for r, t in tex.items()}

    x0, y0, x1, y1 = spec["bounds"]
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=2000.0)
    bmesh.ops.translate(bm, vec=Vector(((x0 + x1) / 2, (y0 + y1) / 2, 0.0)), verts=bm.verts)
    ground = _mesh_object("ground", bm, scene.collection, mats["ground"], collide=False)

    shell = build_shell(spec["bounds"], shell_coll, mats["wall"], mats["ceiling"], h=a.shell_height,
                        margin=a.shell_margin,
                        rect=[float(v) for v in a.shell_rect.split(",")] if a.shell_rect else None,
                        light_scale=look["light_scale"], light_color=look["light_color"]) if cfg.get("shell") else None
    foot = build_primitives(spec["primitives"], obst, mats, keepout=a.keepout)
    if a.start_keepout > 0:                         # the start (origin) and the patch's goal
        foot.append({"type": "disc", "c": np.zeros(2), "r": a.start_keepout})
        foot.append({"type": "disc", "c": np.asarray(spec["goal_pos"][:2], float), "r": a.start_keepout})
    wanted = {k for k, r in roles.items() if manifest[k]["kind"] == "model"}
    area_f = (x1 - x0) * (y1 - y0) / 450.0
    counts = {r: int(round(n * a.clutter * area_f)) for r, n in cfg["counts"].items()}
    if not any(counts.values()) and not (shell and a.hang > 0):
        wanted = set()                      # --clutter 0 (re-skin): nothing to scatter, import no models
    library = {aid: import_model(aid, manifest[aid], root, lib) for aid in sorted(wanted)}
    hang = None
    if shell and a.hang > 0:
        counts["hang"] = int(round(a.hang * area_f))
        hang = {"ceiling": shell["height"], "bottom": (2.2, 2.9), "cable_mat": mats["bar"]}
    placed = scatter(rng, spec["bounds"], foot, {k: roles[k] for k in wanted}, counts, library, props, hang=hang)
    prune_library(library, {p["asset"] for p in placed})
    bpy.context.view_layer.update()

    V, F, face_obj, names, fsrc, fuv, sources = collidable_triangles()
    if a.export_mesh:                                 # trees make this hundreds of MB: opt-in
        np.savez_compressed(out / "mesh.npz", V=V.astype(np.float32), F=F.astype(np.int32),
                            face_obj=face_obj, names=np.array(names), alpha_cut=(fsrc >= 0))
        write_ply(out / "mesh.ply", V, F)
    pts, reg, area, dropped = sample_cloud(V, F, np.random.default_rng([seed, 78]), spec["bounds"], fsrc, fuv, sources,
                                           voxel=a.cloud_voxel, zmax=a.cloud_zmax)
    np.savez_compressed(out / "cloud.npz", points=pts, region=reg)
    t_build, c_build = time.time() - t0, time.process_time()

    blend = save_blend(out, a)
    t_blend = time.time() - t0 - t_build
    shots = previews(out, spec, shell) if a.preview else []
    prov = PV.provenance(REPO, [root / "manifest.json"], kits_order=None, argv=sys.argv, blender=blender_info(),
                         git=a.git)
    info = {
        "patch": f"{spec['family']}_{spec['seed']}", "seed": seed, "theme": a.theme, "level": a.level,
        "variant": a.variant, "look": {**look, "light_color": list(look["light_color"]), "hdri_rotation": hdri_rot},
        **({"look_variant": look_variant} if look_variant else {}),
        "knobs": {"clutter": a.clutter, "keepout": a.keepout, "hang": a.hang, "start_keepout": a.start_keepout},
        "blender": bpy.app.version_string, "device": backend,
        **{k: prov[k] for k in PROVENANCE_KEYS}, "themes_order": sorted(THEMES), "provenance": prov,
        "hdri": hdri, "textures": tex, "wall_texture": tex["wall"], "ground_texture": tex["ground"],
        "primitives": len(spec["primitives"]), "ground_object": ground.name, "ground_half_m": 2000.0,
        "shell": shell, "counts_asked": counts,
        "placed": [{**{k: v for k, v in p.items() if k != "xy"}, "xy": [float(v) for v in p["xy"]]} for p in placed],
        "mesh": {"vertices": int(len(V)), "triangles": int(len(F)), "objects": len(names),
                 "area_m2": round(area, 2), "alpha_cut_triangles": int((fsrc >= 0).sum())},
        "cloud": cloud_info(out, pts, reg, a.cloud_voxel, a.cloud_zmax, dropped), "pack": blend,
        "rgb_samples": a.samples, "seconds_build": round(t_build, 1), "cpu_seconds_build": round(c_build, 1),
        "seconds_blend": round(t_blend, 1),
        "previews": shots, "seconds_total": round(time.time() - t0, 1),
    }
    (out / "course.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("patch", "theme", "hdri", "mesh", "cloud", "seconds_build",
                                           "seconds_total")}))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:                   # blender -P exits 0 on an exception unless told otherwise
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
