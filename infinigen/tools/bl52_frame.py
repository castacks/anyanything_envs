"""Blender 5.2 (the screen's flyer version) render check: open a scene, put the Starling front camera at a pose,
render --renders times per sample count (the first render of the session includes the scene sync; the rest are
steady state with persistent data), on the CPU or a GPU.

    blender -b --factory-startup --python bl52_frame.py -- --scene X.blend --out frame.png \
        [--pose x y z yaw | --views views.json --view 0] [--samples 64 32] [--renders 2] [--device CPU|CUDA|OPTIX] \
        [--threads 30] [--hide-prefix liquid] [--bench] [--denoiser OPTIX] [--step 0.05] [--res 640 480]

--bench renders the way the closed-loop bench's blender_server does (bench-students setup_cycles_mppi +
freeze_scene): adaptive threshold 0.015, bounces 8/3/3/6/0, indirect clamp 8, no caustics, animated objects
frozen, every pass but Combined off, no compositor; and, for Infinigen, the drivers and Scene Time nodes baked at
the current frame (freeze_time). The denoiser matters most on a GPU: Blender's default OIDN
runs on the CPU (~2.5 s a frame on airstation03), the OptiX denoiser ~0.3 s. --step moves the camera forward by
that many metres before every render after the first, so the steady-state renders see a camera-only update, as
in a flight.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze import freeze_objects, freeze_time  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pose", nargs=4, type=float, default=None)
ap.add_argument("--views", default=None)
ap.add_argument("--view", type=int, default=0)
ap.add_argument("--samples", type=int, nargs="+", default=[32])
ap.add_argument("--renders", type=int, default=2, help="renders per sample count")
ap.add_argument("--device", default="CPU", choices=["CPU", "CUDA", "OPTIX"])
ap.add_argument("--threads", type=int, default=30)
ap.add_argument("--hide-prefix", nargs="*", default=[])
ap.add_argument("--bench", action="store_true", help="the closed-loop bench's Cycles settings")
ap.add_argument("--denoiser", default=None, choices=["OPTIX", "OPENIMAGEDENOISE", "NONE"],
                help="default: Blender's own (OIDN)")
ap.add_argument("--step", type=float, default=0.0, help="metres forward per render after the first")
ap.add_argument("--res", type=int, nargs=2, default=[640, 480], help="W H (a small frame isolates the per-frame sync)")
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

W, H, HFOV = a.res[0], a.res[1], 87.0
cam = bpy.data.objects.new("check_cam", bpy.data.cameras.new("check_cam"))
s.collection.objects.link(cam)
cam.data.sensor_fit, cam.data.sensor_width = "HORIZONTAL", 36.0
cam.data.lens = (W / 2) / math.tan(math.radians(HFOV / 2)) * 36.0 / W
cam.data.clip_start, cam.data.clip_end = 0.05, 1000.0
cam.location = (x, y, z)
cam.rotation_euler = (math.pi / 2, 0.0, yaw - math.pi / 2)
s.camera = cam
s.render.engine = "CYCLES"
devices = []
if a.device == "CPU":
    s.cycles.device = "CPU"
else:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = a.device
    prefs.refresh_devices()
    for d in prefs.devices:
        d.use = d.type == a.device
        if d.use:
            devices.append(d.name)
    if not devices:
        raise SystemExit(f"no {a.device} device")
    s.cycles.device = "GPU"
s.cycles.use_adaptive_sampling = True
s.cycles.adaptive_threshold = 0.02
s.cycles.use_denoising = a.denoiser != "NONE"
if a.denoiser not in (None, "NONE"):
    s.cycles.denoiser = a.denoiser
frozen = None
if a.bench:
    s.cycles.adaptive_threshold = 0.015
    s.cycles.sample_clamp_indirect = 8.0
    s.cycles.max_bounces, s.cycles.diffuse_bounces, s.cycles.glossy_bounces = 8, 3, 3
    s.cycles.transmission_bounces, s.cycles.volume_bounces = 6, 0
    s.cycles.caustics_reflective = s.cycles.caustics_refractive = False
    s.cycles.filter_width = 1.5
    s.render.use_motion_blur = False
    frozen = dict(zip(("drivers", "scene_time"), freeze_time(s)))
    frozen["objects"] = freeze_objects(keep=(cam,))
    vl = bpy.context.view_layer
    for p in dir(vl):
        if p.startswith("use_pass_") and p != "use_pass_combined" and isinstance(getattr(vl, p, None), bool) \
                and getattr(vl, p):
            try:
                setattr(vl, p, False)
            except (AttributeError, TypeError):
                pass
    s.render.use_compositing = False
s.render.threads_mode, s.render.threads = "FIXED", a.threads
s.render.use_persistent_data = True
s.render.resolution_x, s.render.resolution_y, s.render.resolution_percentage = W, H, 100
s.render.image_settings.file_format = "PNG"
fwd = Vector((math.cos(yaw), math.sin(yaw), 0.0)) * a.step
times = {}
for spp in a.samples:
    s.cycles.samples = spp
    times[spp] = []
    for k in range(a.renders):
        cam.location = Vector((x, y, z)) + fwd * k
        s.render.filepath = a.out.replace(".png", f"_{spp}spp_{k}.png") if (len(a.samples) > 1 or k) else a.out
        t = time.time()
        bpy.ops.render.render(write_still=k == 0)
        times[spp].append(round(time.time() - t, 2))
sky = [n.sky_type for w in [s.world] if w and w.node_tree for n in w.node_tree.nodes if n.type == "TEX_SKY"]
print("BL52_FRAME", json.dumps({"scene": a.scene, "blender": bpy.app.version_string, "load_s": round(load_s, 1),
                                "render_s": times if len(a.samples) > 1 else times[a.samples[0]], "pose": [x, y, z, yaw],
                                "samples": a.samples, "device": a.device, "devices": devices,
                                "threads": a.threads, "sky": sky, "bench": a.bench,
                                "denoiser": s.cycles.denoiser if s.cycles.use_denoising else "NONE",
                                "step": a.step, "res": [W, H], "frozen": frozen}), flush=True)
