"""Blender 5.2 (the screen's flyer version) render check: open a scene, put the Starling front camera at a pose,
render twice (first = sync + render, second = steady state with persistent data), CPU Cycles.

    blender -b --factory-startup --python bl52_frame.py -- --scene X.blend --out frame.png \
        [--pose x y z yaw | --views views.json --view 0] [--samples 32] [--threads 30] [--hide-prefix liquid]
"""
import argparse
import json
import math
import sys
import time

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pose", nargs=4, type=float, default=None)
ap.add_argument("--views", default=None)
ap.add_argument("--view", type=int, default=0)
ap.add_argument("--samples", type=int, default=32)
ap.add_argument("--threads", type=int, default=30)
ap.add_argument("--hide-prefix", nargs="*", default=[])
a = ap.parse_args(argv)

t0 = time.time()
bpy.ops.wm.open_mainfile(filepath=a.scene)
load_s = time.time() - t0
s = bpy.context.scene
for o in bpy.data.objects:
    if any(o.name.startswith(p) for p in a.hide_prefix):
        o.hide_render = True
for c in bpy.data.collections:
    if c.name.startswith("placeholders"):
        c.hide_render = True

if a.views:
    v = json.load(open(a.views))["views"][a.view]
    x, y, z, yaw = v["p"][0], v["p"][1], v["p"][2], v["yaw"]
elif a.pose:
    x, y, z, yaw = a.pose
else:
    deps = bpy.context.evaluated_depsgraph_get()
    hit, loc, *_ = s.ray_cast(deps, Vector((0.0, 0.0, 200.0)), Vector((0, 0, -1)))
    x, y, z, yaw = 0.0, 0.0, (loc.z if hit else 0.0) + 2.0, 0.0

W, H, HFOV = 640, 480, 87.0
cam = bpy.data.objects.new("check_cam", bpy.data.cameras.new("check_cam"))
s.collection.objects.link(cam)
cam.data.sensor_fit, cam.data.sensor_width = "HORIZONTAL", 36.0
cam.data.lens = (W / 2) / math.tan(math.radians(HFOV / 2)) * 36.0 / W
cam.data.clip_start, cam.data.clip_end = 0.05, 1000.0
cam.location = (x, y, z)
cam.rotation_euler = (math.pi / 2, 0.0, yaw - math.pi / 2)
s.camera = cam
s.render.engine = "CYCLES"
s.cycles.device = "CPU"
s.cycles.samples = a.samples
s.cycles.use_adaptive_sampling = True
s.cycles.adaptive_threshold = 0.02
s.cycles.use_denoising = True
s.render.threads_mode, s.render.threads = "FIXED", a.threads
s.render.use_persistent_data = True
s.render.resolution_x, s.render.resolution_y, s.render.resolution_percentage = W, H, 100
s.render.image_settings.file_format = "PNG"
times = []
for k in range(2):
    s.render.filepath = a.out if k == 0 else a.out.replace(".png", "_2.png")
    t = time.time()
    bpy.ops.render.render(write_still=True)
    times.append(round(time.time() - t, 1))
sky = [n.sky_type for w in [s.world] if w and w.node_tree for n in w.node_tree.nodes if n.type == "TEX_SKY"]
print("BL52_FRAME", json.dumps({"scene": a.scene, "blender": bpy.app.version_string, "load_s": round(load_s, 1),
                                "render_s": times, "pose": [x, y, z, yaw], "samples": a.samples,
                                "threads": a.threads, "sky": sky}), flush=True)
