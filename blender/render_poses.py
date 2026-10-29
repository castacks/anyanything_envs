"""Render a camera path through a course: RGB, depth, and a top-down map.

    blender -b --factory-startup -P blender/render_poses.py -- \
        --course <course dir> --poses <poses.json> --out <dir> [--depth-scale 0.5] [--samples 64]

``poses.json`` is ``flythrough_path.py``'s: fps, width, height, hfov and per
frame the body pose (p, R_wb) on ``flowlib``'s level camera mount. Writes

    rgb/%05d.png        Cycles RGB at width x height
    depth/%05d.npy      float32 z-depth (m, inf = sky) at depth-scale x the RGB size
    topmap.png          an orthographic top-down render of the course
    render.json         timings, the depth camera's K, the map's world -> pixel mapping

Frames already rendered are skipped, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bl_common import camera_intrinsics, script_args, use_gpu  # noqa: E402
import flowlib as FL  # noqa: E402
import render_frames as RF  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", type=Path, required=True)
    ap.add_argument("--poses", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--depth-scale", type=float, default=0.5)
    ap.add_argument("--every", type=int, default=1, help="render every n-th frame (a quick look)")
    ap.add_argument("--rgb-only", action="store_true", help="no depth, no map (a montage clip)")
    a = ap.parse_args(script_args())
    t0 = time.time()
    P = json.loads(a.poses.read_text())
    bpy.ops.wm.open_mainfile(filepath=str(a.course / "course.blend"))
    scene = bpy.context.scene
    backend = use_gpu()
    scene.render.use_persistent_data = True           # keep the BVH between frames
    (a.out / "rgb").mkdir(parents=True, exist_ok=True)
    (a.out / "depth").mkdir(parents=True, exist_ok=True)
    W, H, hfov = P["width"], P["height"], P["hfov_deg"]
    fx = 0.5 * W / math.tan(math.radians(hfov) / 2)
    Wd, Hd = int(round(W * a.depth_scale)), int(round(H * a.depth_scale))
    cam = bpy.data.objects.new("fly_cam", bpy.data.cameras.new("fly_cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    frames = list(enumerate(P["frames"]))[::a.every]

    def put(fr):
        cam.matrix_world = Matrix([list(map(float, r)) for r in
                                   FL.blender_matrix(np.array(fr["p"]), np.array(fr["R_wb"]))])

    timings = {}
    modes = RF.Modes(scene, a.samples)
    modes.rgb()
    camera_intrinsics(cam, W, H, fx)
    t1, n = time.time(), 0
    for k, fr in frames:
        path = a.out / "rgb" / f"{k:05d}.png"
        if path.exists():
            continue
        put(fr)
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        n += 1
    timings["rgb_s_per_frame"] = round((time.time() - t1) / max(n, 1), 3)

    if a.rgb_only:
        print(json.dumps({"frames": len(frames), **timings, "seconds": round(time.time() - t0, 1)}))
        return
    modes.depth()
    camera_intrinsics(cam, Wd, Hd, fx * a.depth_scale)
    tmp = a.out / "__depth.exr"
    t1, n = time.time(), 0
    for k, fr in frames:
        path = a.out / "depth" / f"{k:05d}.npy"
        if path.exists():
            continue
        put(fr)
        np.save(path, RF.render_depth(scene, tmp))
        n += 1
    timings["depth_s_per_frame"] = round((time.time() - t1) / max(n, 1), 3)

    # the map: orthographic, straight down, image up = world +y
    x0, y0, x1, y1 = P["bounds"]
    low = [f["p"] for f in P["frames"] if f.get("phase", "low") == "low"]   # the overview arc may leave the map
    xs = [x0, x1] + [p[0] for p in low]
    ys = [y0, y1] + [p[1] for p in low]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    Wm, Hm = 960, 540
    span = max(max(xs) - min(xs) + 4, (max(ys) - min(ys) + 4) * Wm / Hm)
    modes.rgb()
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = span
    cam.data.sensor_fit = "HORIZONTAL"
    scene.render.resolution_x, scene.render.resolution_y = Wm, Hm
    cam.matrix_world = Matrix.Translation((cx, cy, 80.0))
    cam.rotation_euler = (0.0, 0.0, 0.0)
    scene.render.filepath = str(a.out / "topmap.png")
    bpy.ops.render.render(write_still=True)

    (a.out / "render.json").write_text(json.dumps({
        "blender": bpy.app.version_string, "device": backend, "samples": a.samples,
        "rgb": [W, H], "depth": [Wd, Hd], "K_depth": FL.intrinsics(Wd, Hd, hfov).tolist(),
        "map": {"W": Wm, "H": Hm, "centre": [cx, cy], "metres_per_px": span / Wm},
        "frames": len(P["frames"]), "every": a.every, **timings, "seconds": round(time.time() - t0, 1)}, indent=1))
    print(json.dumps({"frames": len(frames), **timings, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
