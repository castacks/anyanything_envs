"""Render every state of a scene in its Blender course.

    # a labelled rollout (its render_manifest.jsonl + odometry.csv)
    blender -b --factory-startup -P blender/render_frames.py -- \
        --course <course dir> --rollout <data>/train/rollout_<seed>_<family> [--dt 0.0667]
    # the drawn states, beside labelling (a draw / routes record, or a states file with whole states)
    blender -b --factory-startup -P blender/render_frames.py -- \
        --course <course dir> --states <draw>/<pid>.json --out <render>/<pid>

Rollout mode reads the rollout's ``render_manifest.jsonl`` (pose of every
frame, the camera) and ``odometry.csv`` (velocity, drawn acceleration, body
rates); drawn mode reads the same fields from the drawn states
(``render_io.load_drawn``: flattened and numbered ``k = 0..N-1`` the way the
labeller numbers them) with the named camera ``--cam`` (``render_io.CAMS``; default
``starling_front_v1``, the Starling 2 Max nose camera: 640x480, 87 deg, mounted at
(0.071, 0, -0.007) m FLU, pitch 0; ``agile_91`` is the 91 deg pinhole at the body origin
every render before 2026-10-01 used). Rollout mode takes the camera of the rollout's
manifest. ``--no-prev`` (the rgb-v9 renders) skips the previous frame: no ``rgbprev``,
``depthprev`` or ``vector`` (the depth is rendered at the state alone).
Both write raw renders into ``<rollout or out>/blender_raw/``:

    rgb_%08d.png, rgbprev_%08d.png      Cycles RGB at the state and dt earlier
    depth_%08d.npy, depthprev_%08d.npy  float32 z-depth (m), inf where no surface
    depthprim_%08d.npy                  z-depth of the course's primitives alone
                                        (scatter hidden): the camera parity check
    vector_%08d.npy                     Blender's Vector pass at the state, (H, W, 4) px:
                                        the camera is keyframed at t - dt (frame 1), t
                                        (frame 2) and t + dt (frame 3) and frame 2 rendered,
                                        so it is Blender's own flow to the previous frame
                                        -- the independent check on flowlib's
    poses.json                          both poses per frame, K, dt, settings

Frame numbering is the writer's (``k + 1``). Depth is Cycles' Depth pass
(z-depth), one sample through a near-zero pixel filter -- the ray through
each pixel centre, the way ``render_depth.render`` casts -- with the scene's
own materials and ``pass_alpha_threshold`` 0.5, so an alpha-cut leaf card is
seen through where its alpha is under a half, exactly as
``build_course.sample_cloud`` leaves those parts out of the cloud. (Until
2026-09-23 depth was a view-z emission with every material overridden; the
two agreed to 0.0 mm on opaque scenes, and the override cannot see alpha.) ``finish_frames.py`` (repo python)
turns this into the stored frames and flow (layout: ``render_io``).

``--device`` OPTIX | CUDA | CPU | auto (OPTIX, then CUDA, then CPU); asking for
a GPU and getting none is an error, not a silent CPU render. An image the
.blend neither packs nor finds on disk (an unpacked build moved off the box
that built it: Cycles renders it pink and still exits 0) is an error too,
unless ``--allow-missing-images``; either way poses.json lists them. ``--motion-blur on``
blurs RGB over the course's drawn ``shutter_s`` (0 on some courses: no blur
there), ``off`` never blurs. ``--samples`` defaults to the course's
``rgb_samples``; ``--limit`` / ``--only`` render a subset (checks, timing).

Which course build: ``poses.json`` records ``course_source`` (``render_io.course_source``:
the sha256 of ``cloud.npz``, ``spec.json``, ``course.json`` and ``course.blend``, and
``course.json``'s git_sha / dirty / library_sha). The drawn states' ``cloud_source`` (a
draw / routes record, contract 1) -- or in rollout mode the labels' -- must name those
files; states drawn on another build or in the primitives are refused before anything
renders (``--no-course-check`` skips it; a record without the key is ``unverified``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bl_common import camera_intrinsics, script_args, use_gpu  # noqa: E402
import flowlib as FL  # noqa: E402
import render_io as RIO  # noqa: E402


def set_format(scene, fmt: str, multilayer: bool = False) -> None:
    """Blender 5 gates the formats by ``media_type`` (multi-layer images only
    offer ``OPEN_EXR_MULTILAYER``); 4.x lists every format at once."""
    im = scene.render.image_settings
    if hasattr(im, "media_type"):
        im.media_type = "MULTI_LAYER_IMAGE" if multilayer else "IMAGE"
        im.file_format = "OPEN_EXR_MULTILAYER" if multilayer else fmt
    else:
        im.file_format = "OPEN_EXR_MULTILAYER" if multilayer else fmt


class Modes:
    """Switch the scene between the RGB render and the depth render."""

    def __init__(self, scene, samples: int):
        self.scene, self.samples = scene, samples
        self.vl = bpy.context.view_layer

    def rgb(self, exposure: float | None = None):
        s = self.scene
        self.vl.material_override = None
        s.cycles.samples, s.cycles.use_denoising = self.samples, True
        s.cycles.filter_width = 1.5
        s.render.film_transparent = False
        s.view_settings.view_transform = "AgX"
        if exposure is not None:
            s.view_settings.exposure = exposure
        set_format(s, "PNG")
        s.render.image_settings.color_mode = "RGB"
        s.render.image_settings.color_depth = "8"

    def depth(self):
        s = self.scene
        self.vl.material_override = None
        self.vl.pass_alpha_threshold = 0.5
        s.cycles.samples, s.cycles.use_denoising = 1, False
        s.cycles.use_adaptive_sampling = False
        s.cycles.filter_width = 0.01
        s.cycles.max_bounces = 0
        s.render.film_transparent = True
        s.view_settings.view_transform = "Standard"
        s.view_settings.look = "None"
        s.view_settings.exposure, s.view_settings.gamma = 0.0, 1.0
        s.render.use_motion_blur = False              # Cycles writes no Vector pass with blur on
        self.vl.use_pass_vector = True
        self.vl.use_pass_z = True
        set_format(s, "OPEN_EXR", multilayer=True)
        s.render.image_settings.color_depth = "32"


def render_passes(scene, tmp: Path) -> dict[str, np.ndarray]:
    """Render to a multilayer EXR and read it back with Blender's bundled
    OpenImageIO: ``depth`` (the view-z emission, inf where no surface),
    ``zpass`` (Blender's Depth pass), ``vector`` (H, W, 4). OIIO reads rows
    top-down, so no flip here."""
    import OpenImageIO as oiio
    scene.render.filepath = str(tmp)
    bpy.ops.render.render(write_still=True)
    # Blender 5 writes each pass as its own part (subimage) of the EXR
    inp = oiio.ImageInput.open(str(tmp))
    planes, i = {}, 0
    while inp.seek_subimage(i, 0):
        names = list(inp.spec().channelnames)
        px = np.asarray(inp.read_image("float"), np.float32).reshape(inp.spec().height, inp.spec().width, len(names))
        planes.update({n: px[..., j] for j, n in enumerate(names)})
        i += 1
    inp.close()
    tmp.unlink()

    def ch(suffix):
        hits = [n for n in planes if n.endswith(suffix)]
        if not hits:
            raise KeyError(f"no channel *{suffix} in {sorted(planes)}")
        return planes[hits[0]]

    z = ch("Depth.Z")
    out = {"depth": np.where(z < 1e9, z, np.inf).astype(np.float32)}   # no hit: Cycles writes 1e10
    try:
        out["vector"] = np.stack([ch(f"Vector.{c}") for c in "XYZW"], -1).astype(np.float32)
    except KeyError:
        pass
    return out


def render_depth(scene, tmp: Path) -> np.ndarray:
    return render_passes(scene, tmp)["depth"]


def key_camera(cam, frames_poses, t_bc=None) -> None:
    """Keyframe the camera through ``[(frame, (p, R_wb)), ...]``, keeping the
    quaternions on one hemisphere so the interpolation takes the short way."""
    cam.animation_data_clear()
    cam.rotation_mode = "QUATERNION"
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    last = None
    for f, pose in frames_poses:
        cam.matrix_world = Matrix([list(map(float, row)) for row in FL.blender_matrix(*pose, t_bc)])
        q = cam.rotation_quaternion.copy()
        if last is not None and q.dot(last) < 0:
            q.negate()
            cam.rotation_quaternion = q
        last = q
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
    for fc in cam.animation_data.action.fcurves if hasattr(cam.animation_data.action, "fcurves") else []:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def missing_images() -> list[str]:
    """File images neither packed nor on disk (paths resolved like Blender does)."""
    out = []
    for im in bpy.data.images:
        if im.source != "FILE" or im.packed_file is not None or not im.filepath:
            continue
        path = bpy.path.abspath(im.filepath, library=im.library)
        if not os.path.exists(path):
            out.append(path)
    return sorted(set(out))


def pick_device(want: str) -> str:
    """Cycles on ``want``; auto = OPTIX, then CUDA, then CPU."""
    if want == "CPU":
        bpy.context.scene.cycles.device = "CPU"
        return "CPU"
    got = use_gpu("CUDA" if want == "CUDA" else "OPTIX")
    if want != "auto" and got == "CPU":
        raise SystemExit(f"--device {want}: Cycles found no NVIDIA GPU")
    return got


def rekey_camera(cam, frames_poses, t_bc) -> bool:
    """Move the camera's existing keys (``key_camera``'s layout) to new poses in place: inserting
    or clearing keyframes rebuilds the depsgraph relations, and on a heavy course that re-syncs
    the whole scene (BVH, light tree) every frame. False when there are no such keys yet."""
    ad = cam.animation_data
    act = ad.action if ad is not None else None
    if act is None:
        return False
    if hasattr(act, "fcurves"):                     # Blender 4.x
        curves = act.fcurves
    else:                                           # Blender 5: slotted actions
        from bpy_extras import anim_utils  # noqa: PLC0415
        cb = anim_utils.action_get_channelbag_for_slot(act, ad.action_slot)
        if cb is None:
            return False
        curves = cb.fcurves
    fcs = {(fc.data_path, fc.array_index): fc for fc in curves}
    if len(fcs) != 7 or any(len(fc.keyframe_points) != len(frames_poses) for fc in fcs.values()):
        return False
    last = None
    for j, (f, pose) in enumerate(frames_poses):
        M = Matrix([list(map(float, row)) for row in FL.blender_matrix(*pose, t_bc)])
        loc, q, _ = M.decompose()
        if last is not None and q.dot(last) < 0:
            q.negate()
        last = q
        for i in range(3):
            kp = fcs[("location", i)].keyframe_points[j]
            kp.co = (f, loc[i])
        for i in range(4):
            fcs[("rotation_quaternion", i)].keyframe_points[j].co = (f, q[i])
    for fc in fcs.values():
        for kp in fc.keyframe_points:
            kp.handle_left = kp.co
            kp.handle_right = kp.co
        fc.update()
    return True


def blur_camera(scene, cam, t, shutter: float, dt: float, t_bc) -> None:
    """Camera motion blur over ``shutter`` seconds centred on the state: keyed through
    prev -> state (velocity and body rates, ``dt`` apart), extrapolated one step either side."""
    p0 = 2 * t["prev"][0] - t["pose"][0]
    R0 = t["prev"][1] @ (t["pose"][1].T @ t["prev"][1])
    p3 = 2 * t["pose"][0] - t["prev"][0]
    R3 = t["pose"][1] @ (t["prev"][1].T @ t["pose"][1])
    keys = [(1, (p0, R0)), (2, t["prev"]), (3, t["pose"]), (4, (p3, R3))]
    if not rekey_camera(cam, keys, t_bc):
        key_camera(cam, keys, t_bc)
    scene.render.use_motion_blur = True
    scene.render.motion_blur_position = "CENTER"
    scene.render.motion_blur_shutter = shutter / dt
    scene.frame_set(3)


def freeze_scene(scene, cam) -> dict:
    """``--looks``: a heavy course renders as fast as its geometry allows.

    * Every animated object but our camera is frozen at the current frame (the frame the course
      was built and its cloud sampled at): Wenshan's floating props and her stereo rig carry
      animation, so each render re-evaluated them, rebuilt the BVH and light tree (~6 s a frame
      on low_post_6105, 4,962 objects) -- and the blur path's frame_set moved the props away
      from where the cloud has them.
    * Every view-layer pass but Combined is off (her layer writes Vector, Normal, Position, Z,
      Object Index, Cryptomatte; the Vector pass alone costs a motion sync per frame), and the
      compositor is off. ``Modes.depth`` turns Z (and Vector with a previous frame) back on.
    Measured on gs2's CPU, 64x48 at 1 spp: 9.5 s -> 3.8 s (frozen) -> 0.2 s (+ passes) a frame."""
    f = scene.frame_current
    n = 0
    for o in bpy.data.objects:
        if o.animation_data is not None and o != cam:
            mw = o.matrix_world.copy()
            o.animation_data_clear()
            o.matrix_world = mw
            n += 1
    vl = bpy.context.view_layer
    off = []
    for p in dir(vl):
        if p.startswith("use_pass_") and p != "use_pass_combined" and isinstance(getattr(vl, p, None), bool) and getattr(vl, p):
            try:
                setattr(vl, p, False)
                off.append(p[9:])
            except (AttributeError, TypeError):
                pass
    comp = bool(scene.render.use_compositing)
    scene.render.use_compositing = False
    return {"frame": f, "animated_frozen": n, "passes_off": off, "compositing_was": comp}


def render_looks(a, scene, modes, course, cam, todo, raw: Path, t_bc, put):
    """``--looks``: every look for every state, RGB only (the depth passes follow as before).
    Per frame: one render to the Render Result, its linear pixels metered (EXR, no second
    render), then the PNG saved through the view transform at the look's exposure and white
    balance. Returns (per-look records, {frame i: {variant: per-frame record}})."""
    import OpenImageIO as oiio  # noqa: PLC0415
    import render_looks as RL  # noqa: PLC0415
    spec = json.loads(a.looks.read_text())
    looks = spec["looks"] if isinstance(spec, dict) else spec
    if len({str(L["variant"]) for L in looks}) != len(looks):
        raise SystemExit("--looks: variants must be distinct")
    prefix = a.sample_prefix or course.get("patch") or "scene"
    vs = scene.view_settings
    frozen = freeze_scene(scene, cam)
    per = {t["i"]: {} for t in todo}
    meta = []
    tmp = raw / "__meter.exr"
    for L in looks:
        v = str(L["variant"])
        t0 = time.time()
        swaps = []
        if L.get("light"):
            if a.light_root is None:
                raise SystemExit(f"--looks: look {v} has a light, --light-root is not set")
            swaps.append(RL.LightSwap(scene, L["light"], a.light_root))
        if L.get("plain"):
            swaps.append(RL.PlainSwap(scene, f"{prefix}|{v}"))
        modes.rgb(course.get("exposure"))
        if spec.get("denoiser") if isinstance(spec, dict) else False:
            # OptiX: 0.3 s a frame on the 5090 where OIDN (CPU; its GPU path did not engage) took 2.5 s
            scene.cycles.denoiser = spec["denoiser"]
        ev_course = float(vs.exposure)
        wb0 = (bool(vs.use_white_balance), float(vs.white_balance_temperature), float(vs.white_balance_tint))
        mode = (L.get("exposure") or {}).get("mode", "course")
        t_render = t_post = 0.0
        for t in todo:
            dr = RL.draw_frame(L, f"{prefix}/{t['k']}")
            ta = time.time()
            if dr["shutter_s"] > 0:     # keys moved in place, blur on through the look (no re-sync per frame)
                blur_camera(scene, cam, t, dr["shutter_s"], a.dt, t_bc)
            else:
                if cam.animation_data is not None:
                    cam.animation_data_clear()
                put(t["pose"])
            bpy.ops.render.render()
            tb = time.time()
            rr = bpy.data.images["Render Result"]
            rec = {**dr, "name": L.get("name")}
            if mode == "auto":
                set_format(scene, "OPEN_EXR")
                scene.render.image_settings.color_depth = "32"
                rr.save_render(str(tmp), scene=scene)
                inp = oiio.ImageInput.open(str(tmp))
                px = np.asarray(inp.read_image("float"), np.float32).reshape(inp.spec().height, inp.spec().width, -1)
                inp.close()
                tmp.unlink()
                m = RL.meter(px, (L.get("exposure") or {}).get("metering", "wenshan"))
                ev, clamped = RL.ae_ev(m["meter"], dr["ev_jitter"])
                rec.update({k: round(x, 6) for k, x in m.items()})
                rec.update({"ev_metered": round(ev - dr["ev_jitter"], 4) if not clamped else None, "clamped": clamped})
            else:
                ev = ev_course
            rec["exposure_ev"] = round(float(ev), 4)
            rec["render_s"] = round(tb - ta, 2)
            set_format(scene, "PNG")
            scene.render.image_settings.color_mode = "RGB"
            scene.render.image_settings.color_depth = "8"
            if L.get("also_course_exposure") and mode == "auto":     # the old rule, same pixels (comparison)
                vs.exposure = ev_course
                rr.save_render(str(raw / f"rgbcourse_v{v}_{t['i']:08d}.png"), scene=scene)
            vs.exposure = ev
            if "wb_dtemp_k" in dr:
                vs.use_white_balance = True
                vs.white_balance_temperature = wb0[1] + dr["wb_dtemp_k"]
                vs.white_balance_tint = wb0[2] + dr["wb_dtint"]
                rec["wb"] = {"temperature": round(float(vs.white_balance_temperature), 1),
                             "tint": round(float(vs.white_balance_tint), 3), "neutral": [wb0[1], wb0[2]]}
            rr.save_render(str(raw / f"rgb_v{v}_{t['i']:08d}.png"), scene=scene)
            vs.use_white_balance, vs.white_balance_temperature, vs.white_balance_tint = wb0
            vs.exposure = ev_course
            per[t["i"]][v] = rec
            t_render += tb - ta
            t_post += time.time() - tb
        if scene.render.use_motion_blur or cam.animation_data is not None:
            scene.render.use_motion_blur = False
            cam.animation_data_clear()
            scene.frame_set(frozen["frame"])
        rec_l = {"variant": L["variant"], "name": L.get("name"), "exposure_mode": mode, "ev_course": round(ev_course, 4),
                 "jitter_ev": (L.get("exposure") or {}).get("jitter_ev"), "wb": L.get("wb"), "shutter_ms": L.get("shutter_ms"),
                 "light": swaps[0].record if L.get("light") else None,
                 "plain": next((s.record for s in swaps if isinstance(s, RL.PlainSwap)), None),
                 "denoiser": scene.cycles.denoiser, "samples": scene.cycles.samples,
                 "render_s_per_frame": round(t_render / len(todo), 3), "post_s_per_frame": round(t_post / len(todo), 3),
                 "seconds": round(time.time() - t0, 1)}
        for s in reversed(swaps):
            s.restore()
        rec_l["frozen"] = frozen
        meta.append(rec_l)
        print("LOOK " + json.dumps({k: rec_l[k] for k in ("variant", "name", "render_s_per_frame", "post_s_per_frame",
                                                           "seconds")}), flush=True)
    return meta, per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rollout", type=Path, help="a labelled rollout dir (renders into it)")
    src.add_argument("--states", type=Path, help="drawn states: a draw / routes record or a states file")
    ap.add_argument("--out", type=Path, default=None, help="(--states) the scene's render dir")
    ap.add_argument("--dt", type=float, default=1.0 / 15.0, help="previous frame, seconds before the state")
    ap.add_argument("--samples", type=int, default=None, help="default: the course's rgb_samples")
    ap.add_argument("--device", default="auto", choices=["auto", "OPTIX", "CUDA", "CPU"])
    ap.add_argument("--threads", type=int, default=0, help="CPU render threads (0: Blender's choice)")
    ap.add_argument("--motion-blur", default="on", choices=["on", "off"],
                    help="on: the course's shutter_s; off: never")
    ap.add_argument("--limit", type=int, default=0, help="render only the first N states")
    ap.add_argument("--only", default="", help="render only these state indices k (comma list)")
    ap.add_argument("--allow-missing-images", action="store_true",
                    help="render even if the .blend references image files that are not there")
    ap.add_argument("--no-course-check", action="store_true",
                    help="render states / labels whose cloud_source names another build of the course")
    ap.add_argument("--cam", default=RIO.RGB_V9_CAM, choices=sorted(RIO.CAMS),
                    help="(--states) the named camera; rollout mode uses the rollout manifest's")
    ap.add_argument("--no-prev", action="store_true",
                    help="no previous frame: skip rgbprev, depthprev and the Vector pass (rgb-v9)")
    ap.add_argument("--variant", type=int, default=None, help="the appearance variant, recorded only")
    ap.add_argument("--looks", type=Path, default=None,
                    help="render-time looks (render_looks.py): a JSON {'looks': [...]}; every look is rendered for every "
                         "state in this one session (RGB once per look, depth once), each frame metered on its own "
                         "(auto exposure) -- img/frame_left_* is the first look, aux/rgb_v<variant>_* the others")
    ap.add_argument("--light-root", type=Path, default=None,
                    help="(--looks) the light packs: <root>/<seq>/{hdri,fill}.* for looks with a 'light'")
    ap.add_argument("--sample-prefix", default=None,
                    help="(--looks) the per-frame draws are keyed '<prefix>/<k>' (default: the course's patch)")
    ap.add_argument("--geometry-check", action="store_true",
                    help="re-skin: the states were labelled in the primitives (cloud_source null / absent); "
                         "accept them on the course's re-skin geometry check (course.json reskin.ok) instead "
                         "of the cloud_source match")
    a = ap.parse_args(script_args())
    if a.states is not None and a.out is None:
        ap.error("--states needs --out")
    t0 = time.time()
    course = json.loads((a.course / "course.json").read_text())
    if a.rollout is not None:
        root = a.rollout
        todo, cam_info, patch = RIO.rollout_todo(a.rollout, a.dt)
        what = "the labels"
    else:
        root = a.out
        todo, cam_info, patch = RIO.drawn_todo(a.states, a.dt, a.cam)
        what = "the drawn states"
    if patch and course.get("patch") and patch != course["patch"]:
        raise SystemExit(f"states are for {patch}, the course is {course['patch']}")
    src = RIO.course_source(a.course, blend=True)
    try:
        theirs = RIO.rollout_cloud_source(a.rollout) if a.rollout is not None else RIO.load_drawn(a.states)["cloud_source"]
        if a.no_course_check:
            course_check = "off"
        elif a.geometry_check:
            course_check = RIO.check_reskin(a.course, RIO.load_drawn(a.states)["geometry_hash"] if a.states else
                                            RIO.read_jsonl(a.rollout / RIO.MANIFEST)[0].get("geometry_hash"), what)
        else:
            course_check = RIO.check_course(theirs, src, what)
    except RIO.CourseMismatch as e:
        raise SystemExit(f"{e} (--no-course-check renders anyway)")
    print(f"course {RIO.source_str(src)}: {what} {course_check}", flush=True)
    states_total = len(todo)
    if a.only:
        want = {int(x) for x in a.only.split(",") if x.strip()}
        todo = [t for t in todo if t["k"] in want]
    if a.limit:
        todo = todo[:a.limit]
    if not todo:
        raise SystemExit("no states to render")
    bpy.ops.wm.open_mainfile(filepath=str(a.course / "course.blend"))
    missing = missing_images()
    if missing:
        print(f"{len(missing)} image file(s) neither packed nor found, e.g. {missing[:3]}", flush=True)
        if not a.allow_missing_images:
            raise SystemExit(f"course.blend is not self-contained ({len(missing)} missing images): build it with "
                             f"--pack, or pass --allow-missing-images")
    scene = bpy.context.scene
    backend = pick_device(a.device)
    if a.threads:
        scene.render.threads_mode, scene.render.threads = "FIXED", a.threads
    scene.render.use_persistent_data = True           # keep the BVH between frames
    raw = root / RIO.RAW
    raw.mkdir(parents=True, exist_ok=True)

    W, H, fx = int(cam_info["w"]), int(cam_info["h"]), float(cam_info["fx"])
    t_bc = cam_info.get("t_bc")
    cam = bpy.data.objects.new("state_cam", bpy.data.cameras.new("state_cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    camera_intrinsics(cam, W, H, fx)

    def put(pose):
        # through Matrix: a nested list assigned to matrix_world is read column-major
        cam.matrix_world = Matrix([list(map(float, row)) for row in FL.blender_matrix(*pose, t_bc)])

    modes = Modes(scene, a.samples or int(course.get("rgb_samples", 64)))
    timings = {}
    t1 = time.time()
    looks_meta = frame_looks = None
    if a.looks is not None:
        looks_meta, frame_looks = render_looks(a, scene, modes, course, cam, todo, raw, t_bc, put)
        todo_rgb = []
    else:
        todo_rgb = todo
    modes.rgb(course.get("exposure"))
    # place-kit courses: camera motion blur over the drawn exposure time (the camera
    # keyed through prev -> state, extrapolated one step either side)
    shutter = float(course.get("shutter_s") or 0.0) if a.motion_blur == "on" else 0.0
    if looks_meta is not None:
        shutter = None
    for t in todo_rgb:
        if shutter > 0:
            p0 = 2 * t["prev"][0] - t["pose"][0]
            R0 = t["prev"][1] @ (t["pose"][1].T @ t["prev"][1])
            p3 = 2 * t["pose"][0] - t["prev"][0]
            R3 = t["pose"][1] @ (t["prev"][1].T @ t["pose"][1])
            key_camera(cam, [(1, (p0, R0)), (2, t["prev"]), (3, t["pose"]), (4, (p3, R3))], t_bc)
            scene.render.use_motion_blur = True
            scene.render.motion_blur_position = "CENTER"
            scene.render.motion_blur_shutter = shutter / a.dt           # in frames: one frame = dt
            for tag, f in (("rgb", 3),) if a.no_prev else (("rgb", 3), ("rgbprev", 2)):
                scene.frame_set(f)
                scene.render.filepath = str(raw / f"{tag}_{t['i']:08d}.png")
                bpy.ops.render.render(write_still=True)
            cam.animation_data_clear()
            scene.render.use_motion_blur = False
            continue
        for tag, pose in (("rgb", t["pose"]),) if a.no_prev else (("rgb", t["pose"]), ("rgbprev", t["prev"])):
            put(pose)
            scene.render.filepath = str(raw / f"{tag}_{t['i']:08d}.png")
            bpy.ops.render.render(write_still=True)
    timings["rgb_s_per_frame"] = round((time.time() - t1) / ((1 if a.no_prev else 2) * len(todo)
                                                            * (len(looks_meta) if looks_meta else 1)), 3)
    look = {"view_transform": scene.view_settings.view_transform, "look": scene.view_settings.look,
            "exposure": float(scene.view_settings.exposure), "gamma": float(scene.view_settings.gamma),
            "filter_width": float(scene.cycles.filter_width), "denoise": bool(scene.cycles.use_denoising),
            "denoiser": str(scene.cycles.denoiser),
            "adaptive_threshold": float(getattr(scene.cycles, "adaptive_threshold", 0.0)),
            "color_depth": "8", "format": "PNG"}

    t1 = time.time()
    modes.depth()
    tmp = raw / "__passes.exr"
    # the parity render shows the patch's own primitives and the ground only: every
    # other top-level collection is hidden (and a place kit's flat ground overlays)
    hide = [c for c in scene.collection.children if c.name not in ("obstacles", "ground", "library")]
    ground = bpy.data.collections.get("ground")
    hide_obs = [o for o in ground.objects if o.name != "ground_plane"] if ground is not None else []
    if a.no_prev:
        bpy.context.view_layer.use_pass_vector = False
    if looks_meta is not None and a.no_prev:
        # --looks: the same frames, batched -- every full depth, then the primitives-only depth
        # with the scenery hidden ONCE (each hide / unhide re-syncs a heavy course: ~15 s a toggle
        # on low_post_6105, two toggles a frame before)
        for t in todo:
            put(t["pose"])
            np.save(raw / f"depth_{t['i']:08d}.npy", render_passes(scene, tmp)["depth"])
        # the parity render shows our primitives (``obstacles``) and the ground only: Wenshan's
        # buildings, trees and props sit in the scene's master collection, which the collection
        # rule never hid (10-20 % of the pixels failed parity by metres on the first W20 frames).
        # Her ``ground`` (her terrain mesh: mm off z 0 even under a flat window, 3-7 cm of depth at
        # grazing range) is hidden too and an exact z = 0 plane stands in for the caster's ground.
        obst = bpy.data.collections.get("obstacles")
        keep = set(obst.all_objects) if obst is not None else set()
        hid = [o for o in scene.objects if o not in keep and o.name != "ground_plane"
               and o.type != "CAMERA" and not o.hide_render]
        plane = None
        if not any(o.name == "ground_plane" for o in scene.objects):
            me = bpy.data.meshes.new("w19_parity_ground")
            h = 4000.0
            me.from_pydata([(-h, -h, 0.0), (h, -h, 0.0), (h, h, 0.0), (-h, h, 0.0)], [], [(0, 1, 2, 3)])
            plane = bpy.data.objects.new("w19_parity_ground", me)
            scene.collection.objects.link(plane)
        for o in hid:
            o.hide_render = True
        for t in todo:
            put(t["pose"])
            np.save(raw / f"depthprim_{t['i']:08d}.npy", render_depth(scene, tmp))
        for o in hid:
            o.hide_render = False
        if plane is not None:
            bpy.data.objects.remove(plane, do_unlink=True)
        todo_depth = []
    else:
        todo_depth = todo
    for t in todo_depth:
        if a.no_prev:
            cam.animation_data_clear()
            put(t["pose"])
            got = render_passes(scene, tmp)
            np.save(raw / f"depth_{t['i']:08d}.npy", got["depth"])
        else:
            # t + dt: extrapolate the step t - dt -> t once more (only the Vector pass's
            # "next" half reads it)
            p_next = 2 * t["pose"][0] - t["prev"][0]
            R_next = t["pose"][1] @ (t["prev"][1].T @ t["pose"][1])
            key_camera(cam, [(1, t["prev"]), (2, t["pose"]), (3, (p_next, R_next))], t_bc)
            scene.frame_set(2)
            got = render_passes(scene, tmp)
            np.save(raw / f"depth_{t['i']:08d}.npy", got["depth"])
            if "vector" in got:
                np.save(raw / f"vector_{t['i']:08d}.npy", got["vector"])
            scene.frame_set(1)
            np.save(raw / f"depthprev_{t['i']:08d}.npy", render_depth(scene, tmp))
        if hide or hide_obs:
            for c in hide:
                c.hide_render = True
            for o in hide_obs:
                o.hide_render = True
            if not a.no_prev:
                scene.frame_set(2)
            np.save(raw / f"depthprim_{t['i']:08d}.npy", render_depth(scene, tmp))
            for c in hide:
                c.hide_render = False
            for o in hide_obs:
                o.hide_render = False
        cam.animation_data_clear()
    timings["depth_s_per_frame"] = round((time.time() - t1) / ((2 if a.no_prev else 3) * len(todo)), 3)

    K = FL.intrinsics(W, H, float(cam_info["hfov_deg"]))
    (raw / "poses.json").write_text(json.dumps({
        "mode": "rollout" if a.rollout is not None else "drawn",
        "states_source": str(a.states) if a.states is not None else None, "patch": patch,
        "dt": a.dt, "K": K.tolist(), "W": W, "H": H, "cam": cam_info, "prev": not a.no_prev, "variant": a.variant,
        "rgb_look": look, "blender": bpy.app.version_string, "device": backend,
        "threads": scene.render.threads, "rgb_samples": modes.samples, "motion_blur": a.motion_blur,
        "shutter_s": shutter, "course": course["patch"], "hdri": course.get("hdri"), "missing_images": missing,
        "course_source": src, "course_check": course_check, "states_total": states_total,
        **({"input_cloud_source": theirs} if theirs is not RIO.MISSING else {}),
        **timings,
        "seconds": round(time.time() - t0, 1),
        **({"looks": looks_meta} if looks_meta is not None else {}),
        "frames": [{**fr, **({"looks": frame_looks[fr["i"]]} if frame_looks else {})} for fr in RIO.frames_json(todo)]},
        indent=1))
    print(json.dumps({"frames": len(todo), **timings, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
