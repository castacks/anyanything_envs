"""Materials, ground and sky for the place kits -- where the photorealism lives.

Every material is a Principled BSDF on real PBR scans (Poly Haven or
ambientCG, 2k), with what makes a scan stop looking like a tiled scan:

* **anti-tiling** -- a second lookup of the same maps, rotated and offset,
  blended in by a world-space noise mask, so a 60 m road does not repeat
  every 2 m;
* **macro variation** -- a low-frequency world-space noise on the colour
  (sun-bleached, damp and dirty patches), and a per-object hue/value jitter
  so twenty containers from one material are twenty shades;
* **AO, roughness, metal, normal and height** where the scan has them (the
  height map as a bump on top of the normal map);
* **rounded edges** -- Cycles' Bevel shader node bends the shading normal
  within ``bevel`` metres of an edge: a procedural box catches a highlight
  on its edges like a real one, with no change to its geometry (the cloud and
  the depth pass see the same box);
* **dirt** -- a grime scan (``Surface Imperfections``/``Leaking``) laid over
  in world space, stronger near the ground.

Alpha-cut materials (fences, nets, leaf cards) keep plain UV lookups -- no
mapping, no second sample -- because ``build_course.sample_cloud`` reads
their alpha through the mesh UVs to decide what is in the cloud.

The ground is one plane at z = 0 (the labeller's floor) with up to three
scans mixed by noise, optional puddles, and flat zone overlays (roads,
paths, lawns, painted lines) a few millimetres above it -- under the 5 cm
the cloud checks ignore, and non-colliding.

The sky is a Poly Haven HDRI with an exposure chosen from its own
irradiance, the way a camera's auto-exposure would, so a dusk sky and a
noon sky both give a usable image.
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
import numpy as np


#: scan categories whose opacity map is a real cut-out
ALPHA_CATS = {"fence", "net", "leaf", "scattered leaves", "grate", "leafset", "foliage"}


def _img(path: str, colour: bool):
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = "sRGB" if colour else "Non-Color"
    return im


class Looks:
    def __init__(self, lib, rng):
        self.lib, self.rng = lib, rng
        self.cache: dict = {}
        self.grime = None
        self.used: list[dict] = []

    # --- helpers ------------------------------------------------------------------------------

    def pick_material(self, cats_any=(), tags_any=(), cats_none=(), source=None, prefer=None):
        """A random material key from the library by category/tag, or from
        ``prefer`` (a list of keys) when any of those exist."""
        if prefer:
            keys = [k for k in prefer if k in self.lib.items]
            if keys:
                return self.lib.pick(self.rng, keys)
        return self.lib.pick(self.rng, self.lib.materials(cats_any=cats_any, tags_any=tags_any,
                                                           cats_none=cats_none, source=source))

    def _grime_key(self):
        if self.grime is None:
            ks = self.lib.materials(cats_any={"surface imperfections", "leaking"}, source="ambientcg")
            self.grime = self.lib.pick(self.rng, ks) or ""
        return self.grime or None

    # --- materials ----------------------------------------------------------------------------

    def pbr(self, key: str, tile_m: float | None = None, *, proj: str = "box", bevel: float = 0.0,
            dirt: float = 0.0, vary: float = 0.12, jitter: float = 0.06, antitile: bool = True,
            bump: float = 0.25, tint=None, name: str | None = None, emission=None, alpha: bool | None = None,
            into=None):
        """A material from library scan ``key``.

        ``proj``: ``box`` (object-space box projection -- procedural meshes are
        built in world coordinates, so this is world-scale), ``uv`` (the mesh's
        own UVs, untouched -- imported models, alpha-cut cards), ``flat``
        (object xy, for the ground). ``tile_m``: metres per texture repeat
        (default: the scan's own size, else 2 m). ``vary``: macro colour
        variation, ``jitter``: per-object hue/value spread, ``tint``: RGB
        multiplier (paint colours), ``bevel``: edge rounding radius (m),
        ``dirt``: grime overlay strength 0..1. ``alpha``: honour the scan's
        opacity map (default: only for cut-out scans -- fences, nets, leaves,
        grates; a painted-metal scan's peel mask is not a hole)."""
        ck = (key, tile_m, proj, bevel, dirt, vary, jitter, antitile, bump, tuple(tint) if tint else None, emission,
              alpha)
        if ck in self.cache and into is None:
            return self.cache[ck]
        info = self.lib.material(key)
        maps = info["maps"]
        tile = float(tile_m or info.get("tile_m") or 2.0)
        cats = set(self.lib.items[key].get("categories", []))
        if alpha is None:
            alpha = bool(cats & ALPHA_CATS)
        alpha = alpha and "alpha" in maps
        if alpha:
            proj, antitile = "uv", False
        if into is None:
            m = bpy.data.materials.new(name or f"{key.split(':')[1]}_{len(self.cache)}")
            m.use_nodes = True
            nt = m.node_tree
            bsdf = nt.nodes["Principled BSDF"]
        else:                                  # build into another tree (the ground mix): return the BSDF
            nt = into
            bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        N, L = nt.nodes, nt.links
        tc = N.new("ShaderNodeTexCoord")

        if proj == "uv":
            vec_a = tc.outputs["UV"]
            vec_b = None
        else:
            mp = N.new("ShaderNodeMapping")
            mp.inputs["Scale"].default_value = (1 / tile,) * 3
            L.new(tc.outputs["Object"], mp.inputs["Vector"])
            vec_a = mp.outputs["Vector"]
            vec_b = None
            if antitile:
                mp2 = N.new("ShaderNodeMapping")
                if proj == "flat":            # the ground: a rotated second lookup
                    mp2.inputs["Scale"].default_value = (1 / tile,) * 3
                    mp2.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(37.0 + 50 * self.rng.random()))
                else:                         # box projection: rotating about z would shear the side faces
                    mp2.inputs["Scale"].default_value = (0.87 / tile,) * 3
                mp2.inputs["Location"].default_value = tuple(self.rng.uniform(0, 1, 3))
                L.new(tc.outputs["Object"], mp2.inputs["Vector"])
                vec_b = mp2.outputs["Vector"]

        def tex(k, colour, vec):
            t = N.new("ShaderNodeTexImage")
            t.image = _img(maps[k], colour)
            if proj == "box":
                t.projection, t.projection_blend = "BOX", 0.25
            L.new(vec, t.inputs["Vector"])
            return t

        mask = None
        if vec_b is not None:
            nz = N.new("ShaderNodeTexNoise")
            nz.inputs["Scale"].default_value = 1.0 / (2.3 * tile)
            nz.inputs["Detail"].default_value = 2.0
            L.new(tc.outputs["Object"], nz.inputs["Vector"])
            ramp = N.new("ShaderNodeMapRange")
            ramp.inputs["From Min"].default_value, ramp.inputs["From Max"].default_value = 0.45, 0.55
            L.new(nz.outputs["Fac"], ramp.inputs["Value"])
            mask = ramp.outputs["Result"]

        def sample(k, colour, kind="RGBA"):
            a = tex(k, colour, vec_a)
            if mask is None:
                return a.outputs["Color"]
            b = tex(k, colour, vec_b)
            mix = N.new("ShaderNodeMix")
            mix.data_type = kind
            L.new(mask, mix.inputs["Factor"])
            L.new(a.outputs["Color"], mix.inputs[6 if kind == "RGBA" else "A"])
            L.new(b.outputs["Color"], mix.inputs[7 if kind == "RGBA" else "B"])
            return mix.outputs[2 if kind == "RGBA" else "Result"]

        col = sample("diff", True)
        if "ao" in maps:
            ao = sample("ao", False)
            mul = N.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"
            mul.inputs["Factor"].default_value = 0.6
            L.new(col, mul.inputs[6]); L.new(ao, mul.inputs[7])
            col = mul.outputs[2]
        if tint is not None:
            mt = N.new("ShaderNodeMix"); mt.data_type = "RGBA"; mt.blend_type = "MULTIPLY"
            mt.inputs["Factor"].default_value = 1.0
            L.new(col, mt.inputs[6]); mt.inputs[7].default_value = (*tint, 1.0)
            col = mt.outputs[2]
        if vary > 0:                                    # macro variation in world space
            gc = N.new("ShaderNodeNewGeometry")
            nz2 = N.new("ShaderNodeTexNoise")
            nz2.inputs["Scale"].default_value = 0.12
            nz2.inputs["Detail"].default_value = 3.0
            L.new(gc.outputs["Position"], nz2.inputs["Vector"])
            mr = N.new("ShaderNodeMapRange")
            mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 1.0 - vary, 1.0 + 0.5 * vary
            L.new(nz2.outputs["Fac"], mr.inputs["Value"])
            mv = N.new("ShaderNodeMix"); mv.data_type = "RGBA"; mv.blend_type = "MULTIPLY"
            mv.inputs["Factor"].default_value = 1.0
            L.new(col, mv.inputs[6]); L.new(mr.outputs["Result"], mv.inputs[7])
            col = mv.outputs[2]
        if jitter > 0:                                  # per object: no two identical
            oi = N.new("ShaderNodeObjectInfo")
            hsv = N.new("ShaderNodeHueSaturation")
            mr = N.new("ShaderNodeMapRange")
            mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 1 - 2 * jitter, 1 + jitter
            L.new(oi.outputs["Random"], mr.inputs["Value"])
            mh = N.new("ShaderNodeMapRange")
            mh.inputs["To Min"].default_value, mh.inputs["To Max"].default_value = 0.5 - 0.15 * jitter, 0.5 + 0.15 * jitter
            L.new(oi.outputs["Random"], mh.inputs["Value"])
            L.new(mr.outputs["Result"], hsv.inputs["Value"])
            L.new(mh.outputs["Result"], hsv.inputs["Hue"])
            L.new(col, hsv.inputs["Color"])
            col = hsv.outputs["Color"]
        if dirt > 0 and self._grime_key():
            g = self.lib.material(self._grime_key())["maps"]
            gt = N.new("ShaderNodeTexImage")
            gt.image = _img(g["diff"], False)
            gt.projection, gt.projection_blend = "BOX", 0.3
            gm = N.new("ShaderNodeMapping")
            gm.inputs["Scale"].default_value = (1 / 3.0,) * 3
            gpos = N.new("ShaderNodeNewGeometry")
            L.new(gpos.outputs["Position"], gm.inputs["Vector"])
            L.new(gm.outputs["Vector"], gt.inputs["Vector"])
            # grime is dark where the scan is dark; stronger low down (splash, soil)
            inv = N.new("ShaderNodeInvert")
            L.new(gt.outputs["Color"], inv.inputs["Color"])
            sep = N.new("ShaderNodeSeparateXYZ")
            L.new(gpos.outputs["Position"], sep.inputs["Vector"])
            low = N.new("ShaderNodeMapRange")
            low.inputs["From Min"].default_value, low.inputs["From Max"].default_value = 0.0, 1.5
            low.inputs["To Min"].default_value, low.inputs["To Max"].default_value = 1.0, 0.35
            L.new(sep.outputs["Z"], low.inputs["Value"])
            f = N.new("ShaderNodeMath"); f.operation = "MULTIPLY"
            L.new(inv.outputs["Color"], f.inputs[0]); L.new(low.outputs["Result"], f.inputs[1])
            f2 = N.new("ShaderNodeMath"); f2.operation = "MULTIPLY"; f2.inputs[1].default_value = dirt
            L.new(f.outputs["Value"], f2.inputs[0])
            md = N.new("ShaderNodeMix"); md.data_type = "RGBA"; md.blend_type = "MULTIPLY"
            L.new(f2.outputs["Value"], md.inputs["Factor"])
            L.new(col, md.inputs[6]); md.inputs[7].default_value = (0.32, 0.29, 0.25, 1.0)
            col = md.outputs[2]
        L.new(col, bsdf.inputs["Base Color"])

        if "rough" in maps:
            L.new(sample("rough", False), bsdf.inputs["Roughness"])
        if "metal" in maps:
            L.new(sample("metal", False), bsdf.inputs["Metallic"])
        if alpha:
            at = tex("alpha", False, vec_a)
            L.new(at.outputs["Color"], bsdf.inputs["Alpha"])
        normal = None
        if "nor_gl" in maps:
            nm = N.new("ShaderNodeNormalMap")
            L.new(sample("nor_gl", False), nm.inputs["Color"])
            normal = nm.outputs["Normal"]
        if "disp" in maps and bump > 0:
            bp = N.new("ShaderNodeBump")
            bp.inputs["Strength"].default_value = bump
            bp.inputs["Distance"].default_value = 0.02 * tile
            L.new(sample("disp", False), bp.inputs["Height"])
            if normal is not None:
                L.new(normal, bp.inputs["Normal"])
            normal = bp.outputs["Normal"]
        if bevel > 0:
            bv = N.new("ShaderNodeBevel")
            bv.samples = 6
            bv.inputs["Radius"].default_value = bevel
            if normal is not None:
                L.new(normal, bv.inputs["Normal"])
            normal = bv.outputs["Normal"]
        if normal is not None:
            L.new(normal, bsdf.inputs["Normal"])
        if emission is not None:
            bsdf.inputs["Emission Color"].default_value = (*emission[:3], 1.0)
            bsdf.inputs["Emission Strength"].default_value = emission[3] if len(emission) > 3 else 5.0
        self.used.append({"material": (m.name if into is None else "(mix)"), "scan": key, "tile_m": tile,
                          "proj": proj})
        if into is not None:
            return bsdf
        self.cache[ck] = m
        return m

    def paint(self, rgb, rough: float = 0.45, metal: float = 0.0, name: str = "paint", emission=None):
        """A plain painted/plastic surface (no scan): signs, markings, lamp glass."""
        ck = ("paint", tuple(rgb), rough, metal, emission)
        if ck in self.cache:
            return self.cache[ck]
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*rgb, 1.0)
        b.inputs["Roughness"].default_value = rough
        b.inputs["Metallic"].default_value = metal
        if emission is not None:
            b.inputs["Emission Color"].default_value = (*emission[:3], 1.0)
            b.inputs["Emission Strength"].default_value = emission[3] if len(emission) > 3 else 5.0
        self.cache[ck] = m
        return m

    def glass(self, name="glass", tint=(0.8, 0.85, 0.9), rough=0.05):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*tint, 1.0)
        b.inputs["Roughness"].default_value = rough
        b.inputs["Metallic"].default_value = 0.9            # reads as a dark reflective window, opaque for depth
        return m

    # --- ground -------------------------------------------------------------------------------

    def ground_material(self, keys: list[str], tile_m: float = 3.0, puddles: float = 0.0,
                        blend_scale: float = 9.0, name: str = "ground"):
        """Up to three scans mixed by world-space noise (e.g. grass, dirt, gravel),
        with puddles (glossy, dark, flat normal) on ``puddles`` of the area."""
        keys = [k for k in keys[:3] if k]
        if len(keys) == 1 and puddles <= 0:
            return self.pbr(keys[0], tile_m, proj="flat", vary=0.18, jitter=0.0)
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        N, L = nt.nodes, nt.links
        N.remove(N["Principled BSDF"])
        out = N["Material Output"]
        shaders = [self.pbr(k, tile_m, proj="flat", vary=0.18, jitter=0.0, into=nt).outputs["BSDF"] for k in keys]
        tc = N.new("ShaderNodeTexCoord")
        cur = shaders[0]
        for i, sh in enumerate(shaders[1:], 1):
            nz = N.new("ShaderNodeTexNoise")
            nz.inputs["Scale"].default_value = 1.0 / (blend_scale * (0.7 + 0.6 * i))
            nz.inputs["Detail"].default_value = 6.0
            nz.inputs["Roughness"].default_value = 0.65
            mpn = N.new("ShaderNodeMapping")
            mpn.inputs["Location"].default_value = tuple(self.rng.uniform(-50, 50, 3))
            L.new(tc.outputs["Object"], mpn.inputs["Vector"])
            L.new(mpn.outputs["Vector"], nz.inputs["Vector"])
            mr = N.new("ShaderNodeMapRange")
            c = 0.5 + 0.06 * (i - 1)
            mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = c - 0.04, c + 0.04
            L.new(nz.outputs["Fac"], mr.inputs["Value"])
            mix = N.new("ShaderNodeMixShader")
            L.new(mr.outputs["Result"], mix.inputs["Fac"])
            L.new(cur, mix.inputs[1]); L.new(sh, mix.inputs[2])
            cur = mix.outputs["Shader"]
        if puddles > 0:
            wet = N.new("ShaderNodeBsdfPrincipled")
            wet.inputs["Base Color"].default_value = (0.06, 0.055, 0.05, 1)
            wet.inputs["Roughness"].default_value = 0.06
            nz = N.new("ShaderNodeTexNoise")
            nz.inputs["Scale"].default_value = 0.5
            nz.inputs["Detail"].default_value = 5.0
            mpn = N.new("ShaderNodeMapping")
            mpn.inputs["Location"].default_value = tuple(self.rng.uniform(-50, 50, 3))
            L.new(tc.outputs["Object"], mpn.inputs["Vector"])
            L.new(mpn.outputs["Vector"], nz.inputs["Vector"])
            mr = N.new("ShaderNodeMapRange")
            lo = 0.70 - 0.12 * puddles                  # puddles=1 wets ~15 % of the ground
            mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = lo, lo + 0.03
            L.new(nz.outputs["Fac"], mr.inputs["Value"])
            mix = N.new("ShaderNodeMixShader")
            L.new(mr.outputs["Result"], mix.inputs["Fac"])
            L.new(cur, mix.inputs[1]); L.new(wet.outputs["BSDF"], mix.inputs[2])
            cur = mix.outputs["Shader"]
        L.new(cur, out.inputs["Surface"])
        return m

    # --- sky ----------------------------------------------------------------------------------

    def sky(self, key: str, rotation: float | None = None, strength: float = 1.0, target_ev: float = 0.0,
            exposure_jitter: float = 0.25) -> dict:
        """World = the HDRI; returns ``{"hdri", "rotation", "exposure"}``: the
        scene exposure that brings the HDRI's horizontal irradiance to a
        mid-grey target (a camera's auto-exposure), plus a little jitter."""
        path = self.lib.path(key)
        rot = float(self.rng.uniform(0, 2 * math.pi)) if rotation is None else rotation
        w = bpy.data.worlds.new("world")
        bpy.context.scene.world = w
        w.use_nodes = True
        nt = w.node_tree
        bg = nt.nodes["Background"]
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(path), check_existing=True)
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Rotation"].default_value = (0.0, 0.0, rot)
        nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = strength
        E = hdri_irradiance(env.image) * strength
        # E ~ 3.1 (pi x unit radiance) renders a grey card at mid-grey under AgX with exposure 0
        ev = float(np.clip(math.log2(3.1 / max(E, 1e-4)), -6.0, 6.0)) + target_ev
        ev += float(self.rng.normal(0, exposure_jitter))
        bpy.context.scene.view_settings.exposure = ev
        return {"hdri": key, "rotation": round(rot, 4), "exposure": round(ev, 3), "irradiance": round(E, 4)}


def hdri_irradiance(img) -> float:
    """Irradiance on an upward-facing surface, in HDRI units: the cosine-
    weighted integral of the upper hemisphere of an equirectangular image."""
    W, H = img.size
    px = np.empty(W * H * img.channels, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(H, W, img.channels)[::4, ::4, :3]            # rows bottom-up in Blender
    h, w = px.shape[:2]
    lum = px @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    theta = (np.arange(h) + 0.5) / h * math.pi                    # 0 = bottom row = nadir
    elev = theta - math.pi / 2                                   # -pi/2 .. pi/2
    up = elev > 0
    wgt = np.sin(elev) * np.cos(elev) * (math.pi / h) * (2 * math.pi / w)
    return float((lum[up] * wgt[up, None]).sum())
