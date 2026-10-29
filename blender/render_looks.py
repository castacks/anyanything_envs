"""Render-time looks for ``render_frames.py --looks`` (W19, 2026-10-01): several appearances of
one course in one Blender session, with a per-frame auto exposure.

A *looks file* (JSON) lists the looks rendered for every state of a scene::

    {"looks": [
      {"variant": 0, "name": "own",       "light": null,                  "plain": false,
       "exposure": {"mode": "auto", "jitter_ev": 0.25}, "wb": {"temp_sd_k": 400, "tint_sd": 4},
       "shutter_ms": [5, 10]},
      {"variant": 1, "name": "clear_day", "light": {"seq": "204869", ...},  ...},
      {"variant": 2, "name": "plain",     "plain": true, ...}]}

* ``light`` -- another Wenshan scene's lighting rig (``kits.wenshan.relight``'s record: HDRI,
  sun-removed diffuse fill, strengths, rotation, background HSV, sun lamp). The two image files
  come from a *light pack* (``<light root>/<seq>/{hdri,fill}.*``, ``wenshan_render.py lightpack``),
  so a render node needs neither her 94 GB texture store nor a relit course build. Geometry and
  materials are untouched: depth and the cloud are the course's.
* ``plain`` -- every colliding object (our primitives and her buildings, props, trunks) in one
  plain material per instance (``draw_plain_look``: W15's rgb-albedo classes, copied from
  ``build_course`` on branch rgb-albedo d1f7b0e). Materials with a cut-out alpha (leaf cards) keep
  their own material, so the silhouette -- and the depth -- stay the course's.
* ``exposure`` -- ``auto``: the frame's own metering, Wenshan's generator rule (W9 reproduced her
  saved ``exposure_ev`` with it): meter ``M = 0.75 exp(mean ln L) + 0.25 mean L`` over the
  scene-linear Rec.709 luminance of the rendered (denoised) frame, ``EV = log2(0.18 / M) + 0.25``,
  plus a per-frame jitter ``N(0, jitter_ev)`` truncated at 2 sd, clamped to ``EV_RANGE``. The
  frame is rendered once; the meter reads the linear Render Result (saved as EXR), and the PNG is
  then saved through the view transform at the metered exposure -- no second render.
  ``course``: the course's saved exposure (her frame-1 metering; every render before this).
* ``wb`` -- white-balance jitter: Blender's view white balance, temperature and tint drawn
  ``N(0, sd)`` (truncated at 2 sd) around the scene's neutral setting.
* ``shutter_ms`` -- ``[lo, hi]``: camera motion blur over a per-frame shutter drawn uniform in
  ``[lo, hi]`` ms from the state's velocity and body rates (``render_frames``' blur path);
  ``0`` / absent: no blur.

Every draw is a function of ``(key, variant, what)`` only (``frame_rng``), ``key`` =
``<sample prefix>/<k>``, so a re-render reproduces it; every drawn value goes to ``poses.json``
and from there to the manifest's ``looks`` block.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
from pathlib import Path

import numpy as np

#: the exposure the metering may pick, EV (Blender view exposure); outside it the frame is
#: recorded ``clamped`` -- in the pilot the metered values were -0.1 .. 5.9
EV_RANGE = (-6.0, 10.0)
#: Wenshan's generator: mid-grey target and bias (W9 bl_render.py reproduces her exposure_ev)
AE_TARGET, AE_BIAS_EV = 0.18, 0.25
REC709 = np.array([0.2126, 0.7152, 0.0722], np.float32)


def frame_rng(key: str, variant, what: str) -> np.random.Generator:
    h = hashlib.sha256(f"{key}|{variant}|{what}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _trunc_normal(rng, sd: float, k: float = 2.0) -> float:
    if not sd:
        return 0.0
    return float(np.clip(rng.normal(0.0, sd), -k * sd, k * sd))


def draw_frame(look: dict, key: str) -> dict:
    """The per-frame random parts of ``look`` for the state ``key``."""
    v = look.get("variant", 0)
    ex = look.get("exposure") or {"mode": "course"}
    wb = look.get("wb") or {}
    sh = look.get("shutter_ms") or 0
    out = {"ev_jitter": round(_trunc_normal(frame_rng(key, v, "ev"), float(ex.get("jitter_ev", 0.0))), 4)
           if ex.get("mode") == "auto" else 0.0}
    if wb:
        r = frame_rng(key, v, "wb")
        out["wb_dtemp_k"] = round(_trunc_normal(r, float(wb.get("temp_sd_k", 0.0))), 1)
        out["wb_dtint"] = round(_trunc_normal(r, float(wb.get("tint_sd", 0.0))), 3)
    if isinstance(sh, (list, tuple)) and len(sh) == 2 and float(sh[1]) > 0:
        out["shutter_s"] = round(float(frame_rng(key, v, "shutter").uniform(float(sh[0]), float(sh[1]))) / 1000.0, 6)
    else:
        out["shutter_s"] = float(sh) / 1000.0 if sh else 0.0
    return out


#: centre-weighted metering (a camera's default AE mode): a Gaussian over the frame, sd 0.25 of the
#: width / 0.30 of the height, centred 0.10 of the height below the middle -- the sky band at the
#: top of a high or backlit frame counts for less
CW = {"sx": 0.25, "sy": 0.30, "cy": 0.10}


def _cw_weights(h: int, w: int) -> np.ndarray:
    y = (np.arange(h) + 0.5) / h - 0.5 - CW["cy"]
    x = (np.arange(w) + 0.5) / w - 0.5
    g = np.exp(-0.5 * (y[:, None] / CW["sy"]) ** 2 - 0.5 * (x[None, :] / CW["sx"]) ** 2)
    return (g / g.sum()).astype(np.float64)


def meter(rgb_linear: np.ndarray, mode: str = "wenshan") -> dict:
    """The metering of a scene-linear (H, W, >=3) frame: 0.75 log-average + 0.25 mean of the
    Rec.709 luminance (Wenshan's generator), uniform over the frame (``wenshan``) or
    centre-weighted (``center``, :data:`CW`). Both are recorded; ``meter`` is the one ``mode`` names."""
    h, w = rgb_linear.shape[:2]
    L = (rgb_linear[..., :3].reshape(-1, 3).astype(np.float32) @ REC709).reshape(h, w).astype(np.float64)
    lg = np.log(np.maximum(L, 1e-4))
    la, mean = float(np.exp(lg.mean())), float(L.mean())
    W = _cw_weights(h, w)
    la_c, mean_c = float(np.exp((lg * W).sum())), float((L * W).sum())
    out = {"meter_wenshan": 0.75 * la + 0.25 * mean, "meter_center": 0.75 * la_c + 0.25 * mean_c,
           "logavg": la, "mean": mean}
    out["meter"] = out["meter_center"] if mode == "center" else out["meter_wenshan"]
    return out


def ae_ev(m: float, jitter: float = 0.0) -> tuple[float, bool]:
    ev = math.log2(AE_TARGET / max(m, 1e-6)) + AE_BIAS_EV + jitter
    c = float(np.clip(ev, *EV_RANGE))
    return c, c != ev


# --- plain looks (W15 rgb-albedo d1f7b0e, build_course.draw_plain_look / plain_material) ------

PLAIN_CLASSES = (("sky_grey", 0.25), ("near_black", 0.12), ("near_white", 0.12), ("neutral", 0.15),
                 ("any_hue", 0.36))


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def draw_plain_look(rng) -> dict:
    names, p = zip(*PLAIN_CLASSES)
    cls = str(rng.choice(names, p=np.asarray(p) / sum(p)))
    if cls == "sky_grey":
        h, sat, v = rng.uniform(0.53, 0.66), rng.uniform(0.03, 0.35), rng.uniform(0.3, 0.85)
    elif cls == "near_black":
        h, sat, v = rng.uniform(0, 1), rng.uniform(0, 0.3), rng.uniform(0.01, 0.07)
    elif cls == "near_white":
        h, sat, v = rng.uniform(0, 1), rng.uniform(0, 0.08), rng.uniform(0.8, 0.96)
    elif cls == "neutral":
        h, sat, v = rng.uniform(0, 1), rng.uniform(0, 0.05), rng.uniform(0.1, 0.8)
    else:
        h, sat, v = rng.uniform(0, 1), rng.uniform(0.1, 0.8), rng.uniform(0.1, 0.85)
    rgb = colorsys.hsv_to_rgb(h, sat, v)
    look = {"class": cls, "srgb": [round(float(c), 4) for c in rgb],
            "roughness": float(rng.uniform(0.15, 1.0)), "specular": float(rng.uniform(0.0, 0.8)),
            "metallic": float(rng.uniform(0.2, 0.7)) if rng.uniform() < 0.1 else 0.0, "noise": None}
    if rng.uniform() < 0.35:
        k = float(rng.uniform(0.75, 0.92))
        rgb2 = colorsys.hsv_to_rgb((h + rng.uniform(-0.03, 0.03)) % 1.0, sat, v * k)
        look["noise"] = {"scale_per_m": float(rng.uniform(0.15, 0.8)), "detail": float(rng.uniform(0, 2)),
                         "srgb2": [round(float(c), 4) for c in rgb2]}
    return look


# --- Blender side (imported inside Blender only) -----------------------------------------------

def plain_material(name: str, look: dict):
    import bpy  # noqa: PLC0415
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    lin = lambda c: (*(_srgb_to_linear(x) for x in c), 1.0)  # noqa: E731
    bsdf.inputs["Base Color"].default_value = lin(look["srgb"])
    bsdf.inputs["Roughness"].default_value = look["roughness"]
    bsdf.inputs["Metallic"].default_value = look["metallic"]
    for key in ("Specular IOR Level", "Specular"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = look["specular"]
            break
    if look.get("noise"):
        tc = nt.nodes.new("ShaderNodeTexCoord")
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = look["noise"]["scale_per_m"]
        nz.inputs["Detail"].default_value = look["noise"]["detail"]
        nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        e0, e1 = ramp.color_ramp.elements[0], ramp.color_ramp.elements[1]
        e0.position, e0.color = 0.35, lin(look["srgb"])
        e1.position, e1.color = 0.65, lin(look["noise"]["srgb2"])
        nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def _instance_root(o, stop: str = "wenshan_T"):
    """Her instance = the root object under the ``wenshan_T`` Empty (metadata name); ours = itself."""
    r = o
    while r.parent is not None and r.parent.name != stop:
        r = r.parent
    return r


class PlainSwap:
    """Every colliding mesh object (custom prop ``collide`` = 1, or in the ``obstacles``
    collection) gets one plain material per instance through object-linked slots; cut-out alpha
    materials are kept. ``restore()`` puts the slots back."""

    def __init__(self, scene, seed: str):
        import bpy  # noqa: PLC0415
        import alpha_expr as AE  # noqa: PLC0415
        obst = bpy.data.collections.get("obstacles")
        ours = set(obst.all_objects) if obst is not None else set()
        self.undo, groups, kept = [], {}, set()
        alpha = {}
        for o in scene.objects:
            if o.type != "MESH" or not (int(o.get("collide", 0) or 0) == 1 or o in ours):
                continue
            # one colour per instance: her root object; ours per primitive kind (prim<NN>_<kind><j>:
            # a ring's logs, a stand's trunks, a wall's pieces share one plain colour, as one
            # object made of one material would -- per-primitive colours striped the ring)
            g = _instance_root(o).name if o not in ours else "ours:" + o.name.split("_", 1)[-1].rstrip("0123456789")
            if g not in groups:
                lk = draw_plain_look(np.random.default_rng(
                    int.from_bytes(hashlib.sha256(f"{seed}|plain|{g}".encode()).digest()[:8], "little")))
                groups[g] = (lk, plain_material(f"plain_{len(groups):04d}", lk))
            for s in o.material_slots:
                m = s.material
                if m is not None and m.name not in alpha:
                    try:
                        alpha[m.name] = AE.alpha_source(m, lambda _m: ("legacy",)) is not None
                    except Exception:  # noqa: BLE001 -- unknown chain: keep the material
                        alpha[m.name] = True
                if m is not None and alpha[m.name]:
                    kept.add(m.name)
                    continue
                self.undo.append((s, s.link, s.material))
                s.link = "OBJECT"
                s.material = groups[g][1]
        self.record = {"groups": len(groups), "slots": len(self.undo), "alpha_kept": sorted(kept)[:50],
                       "classes": {c: sum(1 for lk, _ in groups.values() if lk["class"] == c) for c, _ in PLAIN_CLASSES}}

    def restore(self):
        for s, link, m in reversed(self.undo):
            s.material = None
            s.link = link
            if link == "OBJECT":
                s.material = m
        self.undo = []


class LightSwap:
    """Her lighting rig v3 with another scene's lighting (``kits.wenshan.relight``'s record),
    the files from a light pack. ``restore()`` brings the course's own back."""

    NODES = ("Environment Texture", "DiffuseFillEnvironment", "Background", "Diffuse sky fill", "Mapping",
             "DatasetBackgroundHSV")

    def __init__(self, scene, rec: dict, root: Path):
        import bpy  # noqa: PLC0415
        from mathutils import Vector  # noqa: PLC0415
        nt = scene.world.node_tree
        miss = [n for n in self.NODES if n not in nt.nodes]
        if miss:
            raise RuntimeError(f"light swap: the world has no {miss} (not her lighting rig v3)")
        d = Path(root) / str(rec["seq"])
        rec = self.full_record(rec, d)
        hdri = next(iter(sorted(d.glob("hdri.*"))), None)
        fill = next(iter(sorted(d.glob("fill.*"))), None)
        if hdri is None or fill is None:
            raise RuntimeError(f"light pack {d}: no hdri.* / fill.*")
        N = nt.nodes
        self.scene, self.nt = scene, nt
        self.saved = {"env": N["Environment Texture"].image, "fill": N["DiffuseFillEnvironment"].image,
                      "bg": N["Background"].inputs["Strength"].default_value,
                      "dfs": N["Diffuse sky fill"].inputs["Strength"].default_value,
                      "rot": tuple(N["Mapping"].inputs["Rotation"].default_value),
                      "hsv": {k: N["DatasetBackgroundHSV"].inputs[k].default_value for k in ("Hue", "Saturation", "Value")},
                      "suns": [(o, o.hide_render) for o in scene.objects if o.type == "LIGHT" and o.data.type == "SUN"]}
        N["Environment Texture"].image = bpy.data.images.load(str(hdri), check_existing=True)
        N["DiffuseFillEnvironment"].image = bpy.data.images.load(str(fill), check_existing=True)
        N["Background"].inputs["Strength"].default_value = float(rec.get("camera_strength") or 0.8)
        N["Diffuse sky fill"].inputs["Strength"].default_value = float(rec.get("fill_strength") or 0.0)
        r = self.saved["rot"]
        N["Mapping"].inputs["Rotation"].default_value = (r[0], r[1], math.radians(float(rec.get("hdri_rotation_deg") or 0.0)))
        for k, inp in (("hue", "Hue"), ("saturation", "Saturation"), ("value", "Value")):
            if k in (rec.get("background_hsv") or {}):
                N["DatasetBackgroundHSV"].inputs[inp].default_value = float(rec["background_hsv"][k])
        for o, _ in self.saved["suns"]:
            o.hide_render = True
        self.sun = None
        s = rec.get("sun")
        if s and float(s.get("energy") or 0) > 0:
            ld = bpy.data.lights.new("W19 Sun", "SUN")
            ld.energy = float(s["energy"])
            ld.color = tuple(float(c) for c in s.get("color", (1, 1, 1)))[:3]
            ld.angle = math.radians(float(s.get("angle_deg") or 0.53))
            o = bpy.data.objects.new("W19 Sun", ld)
            scene.collection.objects.link(o)
            o.rotation_mode = "QUATERNION"
            o.rotation_quaternion = Vector([float(c) for c in s["direction"]]).normalized().to_track_quat("Z", "Y")
            self.sun = o
        self.record = {"seq": str(rec["seq"]), "preset": rec.get("preset"), "hdri": rec.get("hdri"),
                       "files": [hdri.name, fill.name]}

    #: the light record's lighting fields; a look without them renders with no sun and no diffuse fill
    LIGHT_KEYS = ("camera_strength", "fill_strength", "sun", "hdri_rotation_deg", "background_hsv")

    @classmethod
    def full_record(cls, rec: dict, pack: Path) -> dict:
        """``rec`` completed from the pack's ``light.json`` (``wenshan_render.py lightpack`` writes it).

        A looks file written without ``--lights`` carries only ``{seq, preset}``: the swap then fell back
        to camera strength 0.8, fill 0 and no sun, so the geometry got no light at all and every donor
        frame was a silhouette against the donor sky (wenshan_L0_v1: 69 % near-black on every donor
        preset, 2026-10-03). The looks file's own values still win over the pack's."""
        lj = Path(pack) / "light.json"
        if lj.exists():
            full = json.loads(lj.read_text())
            if str(full.get("seq")) != str(rec["seq"]):
                raise RuntimeError(f"light pack {pack}: light.json is seq {full.get('seq')}, the look wants {rec['seq']}")
            return {**full, **{k: v for k, v in rec.items() if v is not None}}
        if not any(k in rec for k in cls.LIGHT_KEYS):
            raise RuntimeError(f"light swap {rec['seq']}: the look has no lighting fields and the pack has no "
                               f"light.json -- it would render unlit")
        return rec

    def restore(self):
        import bpy  # noqa: PLC0415
        N, s = self.nt.nodes, self.saved
        N["Environment Texture"].image = s["env"]
        N["DiffuseFillEnvironment"].image = s["fill"]
        N["Background"].inputs["Strength"].default_value = s["bg"]
        N["Diffuse sky fill"].inputs["Strength"].default_value = s["dfs"]
        N["Mapping"].inputs["Rotation"].default_value = s["rot"]
        for k, v in s["hsv"].items():
            N["DatasetBackgroundHSV"].inputs[k].default_value = v
        for o, hid in s["suns"]:
            o.hide_render = hid
        if self.sun is not None:
            bpy.data.objects.remove(self.sun, do_unlink=True)
            self.sun = None
