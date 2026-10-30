"""Bake an Infinigen scene's time dependence, for renderers that render many frames of one scene (flights, timing).
Plain bpy, so it runs in Blender 5.2 (bl52_frame.py) and in the Infinigen environment (flight_copy.py)."""
import bpy


def freeze_time(scene):
    """Infinigen scenes depend on time beyond object animation: a driver on the cloud shader (frame * k) and Scene
    Time nodes in the grass and fern scatters (wind). Blender re-evaluates them on every render, which recompiles
    every shader and rebuilds the cloud volume and the scene BVH each frame. Bake their values at the current frame.
    Returns (drivers baked, Scene Time nodes replaced).
    """
    deps = bpy.context.evaluated_depsgraph_get()
    pairs = [(ng, ng.evaluated_get(deps)) for ng in bpy.data.node_groups]
    for idb in list(bpy.data.materials) + list(bpy.data.worlds) + list(bpy.data.lights):
        ev = idb.evaluated_get(deps)
        pairs.append((idb, ev))
        if idb.node_tree is not None:
            pairs.append((idb.node_tree, ev.node_tree))
    n_drivers = n_time = 0
    for orig, ev in pairs:
        if orig.animation_data is None:
            continue
        for d in list(orig.animation_data.drivers):
            owner, attr = d.data_path.rsplit(".", 1)
            val = getattr(ev.path_resolve(owner), attr)
            orig.driver_remove(d.data_path, d.array_index)
            if hasattr(val, "__len__"):
                getattr(orig.path_resolve(owner), attr)[d.array_index] = val[d.array_index]
            else:
                setattr(orig.path_resolve(owner), attr, val)
            n_drivers += 1
    fps = scene.render.fps / scene.render.fps_base
    for ng in bpy.data.node_groups:
        for n in [n for n in ng.nodes if n.bl_idname == "GeometryNodeInputSceneTime"]:
            for out in n.outputs:
                if out.links:
                    v = ng.nodes.new("ShaderNodeValue")
                    v.outputs[0].default_value = scene.frame_current / fps if out.name == "Seconds" else scene.frame_current
                    for link in list(out.links):
                        ng.links.new(v.outputs[0], link.to_socket)
            ng.nodes.remove(n)
            n_time += 1
    return n_drivers, n_time


def freeze_objects(keep=()):
    """Object animation baked at the current frame (the bench's freeze_scene). Returns the number of objects."""
    n = 0
    for o in bpy.data.objects:
        if o.animation_data is not None and o not in keep:
            mw = o.matrix_world.copy()
            o.animation_data_clear()
            o.matrix_world = mw
            n += 1
    return n
