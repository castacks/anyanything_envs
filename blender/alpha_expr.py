"""A material's alpha as Cycles' depth pass sees it: the whole surface shader, not just the
Principled BSDF's Alpha input (runs inside Blender; ``build_course`` uses it when
``build_course.ALPHA_EXACT`` is on -- the Wenshan kit turns it on).

Why (probe 2026-10-01, ``deliverables/2026-10-01/WENSHAN_KIT.md``): Wenshan's scenes carry
four kinds of leaf / twig / needle material, and ``build_course._alpha_source`` (the first
image upstream of the Principled Alpha, thresholded at 0.5) gets three of them wrong:

* glTF "MASK" cut-outs (``Tree_1Mat``, ``Tree_2Mat``, the cypress ``*_D``): Alpha =
  ``1 - (image.Alpha < cutoff)`` with cutoff 0.11-0.43. The old rule cut at 0.5 instead of
  the cutoff: crowns *too sparse*.
* Poly Haven twigs (``fir_tree_01_twig``, ``pine_tree_01_twig``): Material Output <- Mix Shader
  (Fac = alpha map, Transparent BSDF, Add(Principled, Translucent)), Principled Alpha = 1.
  The old rule saw an opaque Principled: *solid cards*.
* Poly Haven leaf groups (``island_tree_*_leaves``, ``jacaranda_tree_leaves``,
  ``tree_small_02_leaves``, ``quiver_tree_01_leaf``): the Principled BSDF lives inside a node
  group whose Alpha input is the alpha map. The old rule found no Principled: *solid cards*.
* direct Principled Alpha <- image (``fir_sapling_twigs``, ``pine_sapling_small_twig``): right
  before, right now.

The model: a surface's transparency T is the weight of its transparent closures --
Principled contributes ``1 - Alpha``, a Transparent BSDF 1, Mix Shader blends, Add Shader
adds, a group is followed through its output -- and Cycles' depth pass (with
``pass_alpha_threshold``) keeps a hit where ``alpha = 1 - clamp(T)`` reaches the threshold.
Scalars upstream are evaluated per sample: Image Texture (Color as luminance, or Alpha, or a
Separate Color channel), Math, Invert, Mix / MixRGB (MIX blend), Map Range (linear), Clamp,
Value, Reroute, group inputs. Vertex colours / attributes read as 1.0. The UV is followed
through Reroute / Mapping (2-D affine) to a UV Map node, a UV-named Attribute node or the
active UV map. Anything else (a ramp, a procedural texture, a non-UV projection) makes the
material *unsupported*: ``alpha_source`` then returns the legacy rule's answer and records the
material in ``UNSUPPORTED``.
"""

from __future__ import annotations

import math

import numpy as np

#: materials whose alpha chain had a node this evaluator does not model (name -> reason);
#: they fell back to the legacy rule. build_course records it in course.json.
UNSUPPORTED: dict[str, str] = {}
#: materials the exact rule handled, name -> a short form of the expression (course.json)
HANDLED: dict[str, str] = {}

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


class Unsupported(Exception):
    pass


# --- building the expression ---------------------------------------------------------------

def _linked(sock, env):
    """(node, out socket identifier, env) feeding ``sock``, following reroutes and group
    inputs; None when unlinked."""
    while sock.is_linked:
        ln = sock.links[0]
        if not ln.is_valid or ln.is_muted:
            return None
        node, out = ln.from_node, ln.from_socket
        if node.type == "REROUTE":
            sock = node.inputs[0]
            continue
        if node.type == "GROUP_INPUT":
            if env is None or out.identifier not in env:
                raise Unsupported("group input outside a group")
            outer_sock, outer_env = env[out.identifier]
            sock, env = outer_sock, outer_env
            continue
        return node, out.identifier, env
    return None


def _default(sock):
    v = getattr(sock, "default_value", 1.0)
    try:
        return float(v)
    except TypeError:
        vals = list(v)
        return float(np.dot(LUMA, vals[:3])) if len(vals) >= 3 else float(vals[0])


def _uv_of(tex, env):
    """The UV a texture node reads: ``(uv_name or None, 2x3 affine or None)``."""
    aff = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    sock = tex.inputs["Vector"]
    while True:
        got = _linked(sock, env)
        if got is None:
            return None, (None if np.allclose(aff, np.eye(2, 3)) else aff)
        node, out, env = got
        if node.type == "MAPPING":
            if node.vector_type not in ("POINT", "TEXTURE") or node.inputs["Location"].is_linked \
                    or node.inputs["Rotation"].is_linked or node.inputs["Scale"].is_linked:
                raise Unsupported(f"mapping {node.vector_type} / linked")
            loc = np.array(node.inputs["Location"].default_value)[:2]
            rot = np.array(node.inputs["Rotation"].default_value)
            sc = np.array(node.inputs["Scale"].default_value)[:2]
            if abs(rot[0]) > 1e-6 or abs(rot[1]) > 1e-6:
                raise Unsupported("mapping rotates out of the uv plane")
            c, s = math.cos(rot[2]), math.sin(rot[2])
            R = np.array([[c, -s], [s, c]])
            M = np.diag(sc) if node.vector_type == "POINT" else np.diag(1.0 / sc)
            A = R @ M if node.vector_type == "POINT" else M @ R.T
            t = loc if node.vector_type == "POINT" else -(M @ R.T @ loc)
            step = np.hstack([A, t[:, None]])            # this mapping applied after the upstream one
            aff = step @ np.vstack([aff, [0, 0, 1.0]])
            sock = node.inputs["Vector"]
            continue
        if node.type == "UVMAP":
            return (node.uv_map or None), (None if np.allclose(aff, np.eye(2, 3)) else aff)
        if node.type == "ATTRIBUTE" and node.attribute_type == "GEOMETRY":
            return (node.attribute_name or None), (None if np.allclose(aff, np.eye(2, 3)) else aff)
        if node.type == "TEX_COORD" and out == "UV":
            return None, (None if np.allclose(aff, np.eye(2, 3)) else aff)
        raise Unsupported(f"texture vector from {node.type}.{out}")


def _scalar(sock, env, depth=0):
    """The expression of a float (or colour, read as luminance) input socket."""
    if depth > 40:
        raise Unsupported("chain too deep")
    got = _linked(sock, env)
    if got is None:
        return ("c", _default(sock))
    return _out(*got, depth + 1, want="scalar")


def _out(node, out, env, depth, want="scalar"):
    t = node.type
    if t == "TEX_IMAGE":
        if node.image is None:
            return ("c", 0.0 if out == "Alpha" else 0.0)
        uv, aff = _uv_of(node, env)
        ch = "a" if out == "Alpha" else "lum"
        return ("img", node.image, ch, uv, None if aff is None else tuple(map(tuple, aff)))
    if t == "SEPARATE_COLOR" or t == "SEPRGB":
        got = _linked(node.inputs[0], env)
        if got is None:
            return ("c", 0.0)
        n2, o2, e2 = got
        if n2.type != "TEX_IMAGE" or o2 != "Color" or n2.image is None:
            raise Unsupported("separate colour of a non-image")
        if t == "SEPARATE_COLOR" and node.mode != "RGB":
            raise Unsupported(f"separate colour mode {node.mode}")
        uv, aff = _uv_of(n2, e2)
        ch = {"Red": "r", "Green": "g", "Blue": "b", "R": "r", "G": "g", "B": "b"}[out]
        return ("img", n2.image, ch, uv, None if aff is None else tuple(map(tuple, aff)))
    if t == "VALUE":
        return ("c", float(node.outputs[0].default_value))
    if t in ("VERTEX_COLOR", "ATTRIBUTE"):
        return ("c", 1.0)                           # a vertex tint / alpha: read as 1 (noted in the doc)
    if t == "MATH":
        args = [_scalar(i, env, depth) for i in node.inputs if i.enabled][:3]
        return ("math", node.operation, bool(node.use_clamp), *args)
    if t == "INVERT":
        return ("lerp", _scalar(node.inputs["Fac"], env, depth),
                _scalar(node.inputs["Color"], env, depth), ("1-", _scalar(node.inputs["Color"], env, depth)))
    if t == "MIX":
        if node.data_type not in ("FLOAT", "RGBA") or (node.data_type == "RGBA" and node.blend_type != "MIX"):
            raise Unsupported(f"mix {node.data_type}/{node.blend_type}")
        fac = _scalar(next(i for i in node.inputs if i.name == "Factor" and i.enabled), env, depth)
        a_s = next(i for i in node.inputs if i.name == "A" and i.enabled)
        b_s = next(i for i in node.inputs if i.name == "B" and i.enabled)
        return ("lerp", ("clamp", fac) if node.clamp_factor else fac, _scalar(a_s, env, depth), _scalar(b_s, env, depth))
    if t == "MIX_RGB":
        if node.blend_type != "MIX":
            raise Unsupported(f"mixrgb {node.blend_type}")
        return ("lerp", ("clamp", _scalar(node.inputs["Fac"], env, depth)),
                _scalar(node.inputs["Color1"], env, depth), _scalar(node.inputs["Color2"], env, depth))
    if t == "MAP_RANGE":
        if node.data_type != "FLOAT" or node.interpolation_type != "LINEAR":
            raise Unsupported(f"map range {node.data_type}/{node.interpolation_type}")
        ins = {i.identifier: i for i in node.inputs}
        g = lambda k: _scalar(ins[k], env, depth)  # noqa: E731
        return ("maprange", bool(node.clamp), g("Value"), g("From Min"), g("From Max"), g("To Min"), g("To Max"))
    if t == "CLAMP":
        return ("clampr", _scalar(node.inputs["Value"], env, depth), _scalar(node.inputs["Min"], env, depth),
                _scalar(node.inputs["Max"], env, depth))
    if t == "GROUP":
        if node.node_tree is None:
            raise Unsupported("empty group")
        inner = {i.identifier: (i, env) for i in node.inputs}
        go = next((n for n in node.node_tree.nodes if n.type == "GROUP_OUTPUT" and n.is_active_output), None) or \
            next((n for n in node.node_tree.nodes if n.type == "GROUP_OUTPUT"), None)
        if go is None:
            raise Unsupported("group without output")
        sock = next(i for i in go.inputs if i.identifier == out)
        return _scalar(sock, inner, depth)
    raise Unsupported(f"scalar from {t}.{out}")


def _transparency(sock, env, depth=0):
    """Transparent-closure weight of a shader socket (``("c", 0)`` when nothing is linked)."""
    if depth > 40:
        raise Unsupported("shader chain too deep")
    got = _linked(sock, env)
    if got is None:
        return ("c", 0.0)
    node, out, env = got
    t = node.type
    if t == "BSDF_PRINCIPLED":
        return ("1-", _scalar(node.inputs["Alpha"], env, depth))
    if t == "BSDF_TRANSPARENT":
        return ("c", 1.0)
    if t == "MIX_SHADER":
        sh = [i for i in node.inputs if i.type == "SHADER"]
        return ("lerp", ("clamp", _scalar(node.inputs["Fac"], env, depth)),
                _transparency(sh[0], env, depth + 1), _transparency(sh[1], env, depth + 1))
    if t == "ADD_SHADER":
        sh = [i for i in node.inputs if i.type == "SHADER"]
        return ("math", "ADD", False, _transparency(sh[0], env, depth + 1), _transparency(sh[1], env, depth + 1))
    if t == "GROUP":
        if node.node_tree is None:
            return ("c", 0.0)
        inner = {i.identifier: (i, env) for i in node.inputs}
        go = next((n for n in node.node_tree.nodes if n.type == "GROUP_OUTPUT" and n.is_active_output), None) or \
            next((n for n in node.node_tree.nodes if n.type == "GROUP_OUTPUT"), None)
        if go is None:
            return ("c", 0.0)
        return _transparency(next(i for i in go.inputs if i.identifier == out), inner, depth + 1)
    return ("c", 0.0)        # any other closure (diffuse, translucent, emission, glass, ...) is opaque to the pass


def _fold(e):
    """Constant-fold; returns the expression (a ``("c", v)`` when it is a constant)."""
    k = e[0]
    if k in ("c", "img"):
        return e
    if k == "1-":
        a = _fold(e[1])
        return ("c", 1.0 - a[1]) if a[0] == "c" else ("1-", a)
    if k == "clamp":
        a = _fold(e[1])
        return ("c", min(max(a[1], 0.0), 1.0)) if a[0] == "c" else ("clamp", a)
    if k == "lerp":
        f, a, b = _fold(e[1]), _fold(e[2]), _fold(e[3])
        if f[0] == "c":
            if f[1] <= 0.0:
                return a
            if f[1] >= 1.0:
                return b
        if f[0] == a[0] == b[0] == "c":
            return ("c", (1 - f[1]) * a[1] + f[1] * b[1])
        return ("lerp", f, a, b)
    if k == "math":
        args = [_fold(x) for x in e[3:]]
        if all(a[0] == "c" for a in args):
            return ("c", float(_math(e[1], e[2], *[np.float64(a[1]) for a in args])))
        return ("math", e[1], e[2], *args)
    if k in ("maprange", "clampr"):
        args = [_fold(x) for x in e[(2 if k == "maprange" else 1):]]
        return (k, e[1], *args) if k == "maprange" else (k, *args)
    return e


def _math(op, clamp, a, b=None, c=None):
    if op == "ADD": r = a + b
    elif op == "SUBTRACT": r = a - b
    elif op == "MULTIPLY": r = a * b
    elif op == "DIVIDE": r = np.where(b != 0, a / np.where(b != 0, b, 1), 0.0)
    elif op == "MULTIPLY_ADD": r = a * b + c
    elif op == "POWER": r = np.power(np.maximum(a, 0), b)
    elif op == "MINIMUM": r = np.minimum(a, b)
    elif op == "MAXIMUM": r = np.maximum(a, b)
    elif op == "LESS_THAN": r = (a < b).astype(np.float64)
    elif op == "GREATER_THAN": r = (a > b).astype(np.float64)
    elif op == "ABSOLUTE": r = np.abs(a)
    elif op == "ROUND": r = np.floor(a + 0.5)
    elif op == "FLOOR": r = np.floor(a)
    elif op == "CEIL": r = np.ceil(a)
    elif op == "SQRT": r = np.sqrt(np.maximum(a, 0))
    elif op == "SIGN": r = np.sign(a)
    elif op == "COMPARE": r = (np.abs(a - b) <= np.maximum(c, 1e-5)).astype(np.float64)
    elif op == "SNAP": r = np.floor(a / np.where(b != 0, b, 1)) * b
    elif op == "FRACT": r = a - np.floor(a)
    else:
        raise Unsupported(f"math {op}")
    return np.clip(r, 0.0, 1.0) if clamp else r


def _images(e, out):
    if e[0] == "img":
        out.append(e)
    elif e[0] != "c":
        for x in e[1:]:
            if isinstance(x, tuple):
                _images(x, out)
    return out


def _key(e):
    """``e`` with images replaced by their names: the source's identity in build_course."""
    if e[0] == "img":
        return ("img", e[1].name, *e[2:])
    return tuple(_key(x) if isinstance(x, tuple) else x for x in e)


def _short(e) -> str:
    k = e[0]
    if k == "c":
        return f"{e[1]:.3g}"
    if k == "img":
        return f"{e[1].name}.{e[2]}"
    if k == "math":
        return f"{e[1].lower()}({', '.join(_short(x) for x in e[3:])})"
    return f"{k}({', '.join(_short(x) for x in e[1:] if isinstance(x, tuple))})"


# --- the build_course interface ------------------------------------------------------------------

def alpha_source(mat, legacy):
    """``build_course._alpha_source``'s answer, exactly: ``None`` (opaque), ``("const", a)`` or
    ``("expr", key, expr, uv_name)``; ``legacy(mat)`` when the chain is not modelled."""
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return None
    nt = mat.node_tree
    outn = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output
                 and n.target in ("ALL", "CYCLES")), None) or \
        next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if outn is None:
        return None
    try:
        T = _transparency(outn.inputs["Surface"], None)
        a = _fold(("1-", ("clamp", T)))
    except Unsupported as e:
        UNSUPPORTED[mat.name] = str(e)
        return legacy(mat)
    if a[0] == "c":
        return None if a[1] >= 0.999 else ("const", float(a[1]))
    imgs = _images(a, [])
    uvs = {i[3] for i in imgs}
    uv = sorted(uvs, key=lambda u: (u is not None, u or ""))[0] if uvs else None
    if len(uvs) > 1:
        UNSUPPORTED[mat.name] = f"several uv maps {sorted(map(str, uvs))}: read through {uv!r}"
    HANDLED[mat.name] = _short(a)[:200]
    return ("expr", repr(_key(a)), a, uv)


def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def evaluate(e, uv: np.ndarray, look) -> np.ndarray:
    """``e`` at ``uv`` (n x 2); ``look.pixels(img) -> (H x W x C array, W, H)`` (build_course.AlphaLookup).
    Nearest texel, repeat wrap: the rule the legacy lookup used."""
    k = e[0]
    if k == "c":
        return np.full(len(uv), e[1], np.float64)
    if k == "img":
        img, ch, _uvn, aff = e[1], e[2], e[3], e[4]
        px, W, H = look.pixels(img)
        q = uv if aff is None else uv @ np.asarray(aff)[:, :2].T + np.asarray(aff)[:, 2]
        u = np.mod(q[:, 0], 1.0); v = np.mod(q[:, 1], 1.0)
        i = np.clip((v * H).astype(int), 0, H - 1); j = np.clip((u * W).astype(int), 0, W - 1)
        if ch == "a":
            return px[i, j, 3].astype(np.float64) if px.shape[2] >= 4 else np.ones(len(uv))
        rgb = px[i, j, :3].astype(np.float64) if px.shape[2] >= 3 else np.repeat(px[i, j, :1], 3, 1).astype(np.float64)
        if not img.is_float and img.colorspace_settings.name.lower().startswith("srgb"):
            rgb = _srgb_to_linear(rgb)
        if ch == "lum":
            return rgb @ LUMA.astype(np.float64)
        return rgb[:, "rgb".index(ch)]
    if k == "1-":
        return 1.0 - evaluate(e[1], uv, look)
    if k == "clamp":
        return np.clip(evaluate(e[1], uv, look), 0.0, 1.0)
    if k == "lerp":
        f = evaluate(e[1], uv, look)
        return (1 - f) * evaluate(e[2], uv, look) + f * evaluate(e[3], uv, look)
    if k == "math":
        return _math(e[1], e[2], *[evaluate(x, uv, look) for x in e[3:]])
    if k == "maprange":
        cl, v, a0, a1, b0, b1 = e[1], *[evaluate(x, uv, look) for x in e[2:]]
        d = np.where(a1 - a0 != 0, a1 - a0, 1.0)
        r = b0 + (v - a0) / d * (b1 - b0)
        return np.clip(r, np.minimum(b0, b1), np.maximum(b0, b1)) if cl else r
    if k == "clampr":
        v, lo, hi = [evaluate(x, uv, look) for x in e[1:]]
        return np.clip(v, lo, hi)
    raise ValueError(f"unknown alpha expression {k!r}")
