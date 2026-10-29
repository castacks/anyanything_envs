"""Helpers shared by the scripts that run inside Blender (``bpy`` available)."""

from __future__ import annotations

import sys

import bpy


def script_args(argv=None) -> list[str]:
    """The arguments after ``--`` on ``blender -b -P script.py -- ...``."""
    argv = sys.argv if argv is None else argv
    return argv[argv.index("--") + 1:] if "--" in argv else []


def use_gpu(prefer: str = "OPTIX") -> str:
    """Point Cycles at the NVIDIA GPU(s) only. Returns the backend in use."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in (prefer, "CUDA"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == backend
            bpy.context.scene.cycles.device = "GPU"
            return backend
    bpy.context.scene.cycles.device = "CPU"
    return "CPU"


def camera_intrinsics(cam_obj, width: int, height: int, fx: float) -> None:
    """A pinhole with focal ``fx`` px, square pixels, principal point at the
    image centre. Blender's centre is ``(w - 1) / 2`` in pixel-centre
    coordinates -- the convention ``render_depth.Camera`` uses -- so no shift."""
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    cam = cam_obj.data
    cam.type = "PERSP"
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = 36.0
    cam.lens = fx * cam.sensor_width / width
    cam.shift_x = cam.shift_y = 0.0
    cam.clip_start, cam.clip_end = 0.01, 1000.0
