"""An Infinigen nature scene generated for drone flights: one fixed square flight box, populated everywhere.

Infinigen's nature pipeline is camera-centric: it only turns placeholders into real trees (and only scatters ground
cover) where one rendered camera can see. A flight test needs the whole box real, from every pose. This driver runs
Infinigen's own generate_nature tasks unchanged, except that its "in view of a camera" test is replaced by a crop:
everything whose xy lies inside the box plus a margin counts as in view, at a fixed design distance (which sets the
asset detail). Terrain is meshed with the view-independent UniformMesher over the same square.

    python infinigen/nature_box.py --seed 1 --out out/forest_1 --scene-type forest --half 15 --margin 10 \
        --design-dist 8 --no-monocots --configs fast_terrain_assets.gin [--tree-density 0.06] [-p gin.param=value]

Scene types are Infinigen's ``configs_nature/scene_types`` (forest, plain, river, canyon, cliff, mountain,
desert, arctic, snowy_mountain, coast; cave needs its own ground rule, see docs/infinigen.md).

Writes <out>/scene/scene.blend (+ Infinigen's assets/ and logs) and <out>/box.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--half", type=float, default=15.0, help="flight box half-width, m")
    ap.add_argument("--margin", type=float, default=12.0, help="real assets this far outside the box, m")
    ap.add_argument("--scene-type", default="forest", help="Infinigen nature scene type (configs_nature/scene_types)")
    ap.add_argument("--tree-density", type=float, default=None, help="trees per m2 (default: the scene type's own)")
    ap.add_argument("--water", choices=["off", "type"], default="off",
                    help="off: no water body (the UniformMesher meshes it coarsely); type: the scene type's own")
    ap.add_argument("--bush-density", type=float, default=None)
    ap.add_argument("--design-dist", type=float, default=4.0, help="asset detail as if seen from this far, m")
    ap.add_argument("--no-monocots", action="store_true", help="monocot grasses take ~27 min per model here")
    ap.add_argument("--voxel", type=float, default=0.15, help="terrain mesher voxel, m")
    ap.add_argument("--near-subsurf-max-faces", type=int, default=600_000,
                    help="leave Infinigen's near terrain unsubdivided above this many faces (steep types)")
    ap.add_argument("--unique-trees", type=int, default=12, help="unique trees per species; the rest are copies")
    ap.add_argument("--unique-bushes", type=int, default=10, help="unique bushes per species; the rest are copies")
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("-p", "--overrides", nargs="*", default=[])
    return ap.parse_args()


def main():
    a = parse()
    a.out.mkdir(parents=True, exist_ok=True)
    region = a.half + a.margin            # real assets inside |x|,|y| <= region
    tbound = region + 4.0                 # fine terrain a little wider than the assets

    import logging
    from infinigen.core.placement import split_in_view
    from infinigen_examples import generate_nature

    def box_inview(points, cameras, dist_max, vis_margin, frame_start=None, frame_end=None, verbose=False):
        p = np.asarray(points, float)[:, :3]
        inside = (np.abs(p[:, 0]) <= region) & (np.abs(p[:, 1]) <= region)
        out_box = np.maximum(np.abs(p[:, :2]) - a.half, 0.0)
        dist = a.design_dist + np.hypot(out_box[:, 0], out_box[:, 1])
        return inside, np.where(inside, dist, 1e7), np.where(inside, 0.0, 1e7)

    split_in_view.compute_inview_distances = box_inview

    # Every bpy.ops call during asset generation updates the view layer, which re-evaluates the ground-cover
    # scatters; over the whole box (instead of Infinigen's 10 m near patch) that made each plant asset take
    # minutes (monocots 27 min each). Keep the "scatters" collection out of the viewport depsgraph; rendering
    # uses the render depsgraph, so nothing changes in the images.
    from infinigen.core.placement import instance_scatter
    from infinigen.core.util import blender as butil
    _scatter = instance_scatter.scatter_instances

    def scatter_hidden(*args, **kwargs):
        obj = _scatter(*args, **kwargs)
        butil.get_collection("scatters").hide_viewport = True
        return obj

    n_patched = 0
    for mod in list(sys.modules.values()):
        if getattr(mod, "scatter_instances", None) is _scatter:
            mod.scatter_instances = scatter_hidden
            n_patched += 1
    print(f"scatter_instances rebound in {n_patched} modules", flush=True)

    # Infinigen puts a COLLISION modifier on the near-camera terrain for falling-leaf particles (off here). With the
    # box crop that terrain is the whole subdivided box, and the modifier's BVH is rebuilt on depsgraph updates.
    # It also subdivides that terrain twice (SUBSURF levels=2), sized for its own 0.8 m coarse mesh near one camera.
    # Ours is the whole square, already at --voxel, and OpenSubdiv needs ~50 KB per face: on steep terrain
    # (mountain: 1.1 M faces in the square) that alone passed 45 GB. Above the limit the near terrain stays as meshed.
    _modify_mesh = butil.modify_mesh
    near = {}

    def modify_mesh_box(obj, type, *args, **kwargs):
        if type == "COLLISION":
            kwargs["show_viewport"] = False  # keep the settings Infinigen touches next, skip viewport evaluation
        if type == "SUBSURF" and obj.name.endswith(".inview_near"):
            near.update(faces=len(obj.data.polygons), subdivided=len(obj.data.polygons) <= a.near_subsurf_max_faces)
            if not near["subdivided"]:
                print(f"{obj.name}: {near['faces']} faces > --near-subsurf-max-faces, not subdivided", flush=True)
                return obj
        return _modify_mesh(obj, type, *args, **kwargs)

    butil.modify_mesh = modify_mesh_box

    # Trees and bushes: the first K placeholders of each species get unique assets, the rest get linked copies of
    # them (random yaw, +-10 % scale) -- a few hundred unique trees would take hours to grow.
    from infinigen.core.placement import placement as plc
    from infinigen.core.placement import detail
    _populate = plc.populate_collection
    unique_k = {"TreeFactory": a.unique_trees, "BushFactory": a.unique_bushes}
    crng = np.random.default_rng(a.seed + 7)

    def clone_tree(src, col):
        mapping = {}
        for o in [src] + list(src.children_recursive):
            c = o.copy()
            col.objects.link(c)
            mapping[o] = c
        for o, c in mapping.items():
            if o.parent in mapping:
                c.parent = mapping[o.parent]
        return mapping[src]

    def populate_instanced(factory, placeholder_col, cameras=None, asset_col_target=None, dist_cull=None,
                           vis_cull=None, verbose=True, cache_system=None, **asset_kwargs):
        k = unique_k.get(type(factory).__name__)
        if k is None or cache_system is not None:
            return _populate(factory, placeholder_col, cameras=cameras, asset_col_target=asset_col_target,
                             dist_cull=dist_cull, vis_cull=vis_cull, verbose=verbose, cache_system=cache_system,
                             **asset_kwargs)
        if asset_col_target is None:
            asset_col_target = butil.get_collection(f"unique_assets:{repr(factory)}")
        placeholders = [o for o in placeholder_col.objects if o.parent is None]
        if cameras is not None:
            targets = plc.filter_populate_targets(placeholders, cameras, dist_cull, vis_cull, False)
        else:
            targets = [(p, detail.scatter_res_distance(), 0) for p in placeholders]
        print(f"Populating {len(targets)} placeholders for {factory=} out of {len(placeholders)} total "
              f"({min(k, len(targets))} unique, rest instanced)", flush=True)
        protos, all_objs, updated = [], [], []
        for i, (p, dist, vis_dist) in enumerate(targets):
            _, inst_seed, *_ = plc.parse_asset_name(p.name)
            if len(protos) < k:
                obj = factory.spawn_asset(i, placeholder=p, distance=dist, vis_distance=vis_dist, **asset_kwargs)
                protos.append(obj)
            else:
                src = protos[int(crng.integers(len(protos)))]
                obj = clone_tree(src, asset_col_target)
                for ch in list(p.children):
                    ch.hide_render = True
                    ch.hide_viewport = True
                obj.parent = p
                obj.rotation_euler.z += float(crng.uniform(0, 2 * np.pi))
                sc = float(crng.uniform(0.9, 1.1))
                obj.scale = (obj.scale[0] * sc, obj.scale[1] * sc, obj.scale[2] * sc)
                obj.name = f"{repr(factory)}.spawn_asset({i})"
            if p is not obj:
                p.hide_render = True
            for o in butil.iter_object_tree(obj):
                butil.put_in_collection(o, asset_col_target)
            obj["dist"] = dist
            obj["vis_dist"] = vis_dist
            updated.append((inst_seed, p))
            all_objs.append((inst_seed, obj))
            if (i + 1) % 10 == 0:
                print(f"  {repr(factory)}: {i + 1}/{len(targets)} placed", flush=True)
        asset_col_target.hide_viewport = False
        factory.finalize_assets([r for _, r in all_objs])
        asset_col_target.hide_viewport = True
        return all_objs, updated

    plc.populate_collection = populate_instanced
    logging.getLogger("infinigen").setLevel(logging.INFO)
    logging.getLogger("infinigen.core.nodes.node_wrangler").setLevel(logging.CRITICAL)

    n = int(round(2 * tbound / a.voxel / 3))  # UniformMesher upscales 3x
    overrides = [
        'fine_terrain.mesher_backend="UniformMesher"',
        f"Terrain.populated_bounds=({-tbound}, {tbound}, {-tbound}, {tbound}, -15, 35)",
        f"UniformMesher.subdivisions=({n}, -1, -1)",
        "compose_nature.leaf_particles_chance=0.0",
        "compose_nature.rain_particles_chance=0.0",
        "compose_nature.ground_creatures_chance=0.0",
        "compose_nature.flying_creatures_chance=0.0",
        "execute_tasks.optimize_terrain_diskusage=False",
        # the camera no longer decides what is populated, but its pose search must still finish: snowy_mountain wants
        # all visible terrain >= 40 m away, which the square almost never offers. base.gin's 2 m (the other types').
        "keep_cam_pose_proposal.min_terrain_distance=2",
    ]
    if a.tree_density is not None:
        overrides.append(f"compose_nature.tree_density={a.tree_density}")
    if a.bush_density is not None:
        overrides.append(f"compose_nature.bush_density={a.bush_density}")
    if a.water == "off":
        overrides.append("scene.waterbody_chance=0.0")
    if a.no_monocots:
        overrides.append("compose_nature.monocots_chance=0.0")
    overrides += a.overrides

    scene_dir = a.out / "scene"
    t0 = time.time()
    args = argparse.Namespace(
        seed=str(a.seed), output_folder=scene_dir, input_folder=None,
        task=["coarse", "populate", "fine_terrain"], task_uniqname=None, debug=None,
        configs=[f"{a.scene_type}.gin", "no_creatures.gin", "no_particles.gin"] + a.configs, overrides=overrides)
    generate_nature.main(args)
    rec = {"seed": a.seed, "scene_type": a.scene_type, "water": a.water, "half": a.half, "margin": a.margin, "region": region, "terrain_bound": tbound,
           "tree_density": a.tree_density, "bush_density": a.bush_density, "design_dist": a.design_dist,
           "voxel": a.voxel, "mesher_subdiv": n, "configs": args.configs, "overrides": overrides,
           "unique_trees": a.unique_trees, "unique_bushes": a.unique_bushes, "near_terrain": near,
           "seconds": round(time.time() - t0, 1)}
    (a.out / "box.json").write_text(json.dumps(rec, indent=1))
    print("FOREST_BOX_DONE", json.dumps(rec))


if __name__ == "__main__":
    sys.exit(main())
