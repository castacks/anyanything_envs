#!/usr/bin/env python3
"""Build gates: is a built course fit to be drawn, routed and labelled?

    python blender/build_gates.py <course dir> [<course dir> ...] [--plan <batch>/plan.json]
        [--builds <batch>/builds] [--out gates.json] [--budget 400000] [--max-build-s 600]
        [--allow-dirty] [--expect-git-sha SHA] [--expect-library-sha SHA]
    from build_gates import gate_course          # one row per scene (dict)

One row per course (design BLENDER_V7_PIPELINE.md, "Gates", build row);
``ok`` is the AND of the gating checks, ``fails`` names the ones that failed:

* ``infra`` -- the build says nothing about the scene: refused by
  ``--require-clean`` (rc 3), bad arguments (rc 2), killed by a signal
  (negative rc, or > 128: the OOM killer, a timeout), no build record and
  no ``course.json`` yet (still building), or rc 0 with the course's files
  gone (deleted since). ``infra`` holds the reason and
  ``batch.resolve_scenes`` keeps the slot pending for a rebuild instead of
  spending a spare on it. A crash of the gate itself is infra too.
* ``rc`` -- the build script raised on this scene (rc 1: ``build_course``
  exits 1 on any exception; ``course.json`` is the last file it writes).
  ``rc`` comes from ``<builds>/<pid>.json`` (the build record ``batch.py
  build`` writes) when there is one.
* ``clearance`` -- the start (the origin) and the patch goal are at least
  ``CLEAR_M`` = 1.5 m (xy) from every *dressing* point in the flight band
  z 0.3-3.2 m -- the disc ``kits.base.Ctx`` claims round both (the 09-25
  contract check, ``kitcheck.py``, which looked at all points). The patch's
  own primitives are the drawn geometry, not dressing: a pillar may stand
  closer to its goal than that. ``*_clear_m`` (all points) and
  ``*_clear_prim_m`` (primitives only) are recorded beside it.
* ``slots`` -- no cloud point inside any wall opening (slot, window, door;
  the box of the opening shrunk by 6 cm, 0.6 m either side of the wall).
* ``gaps`` -- no *dressing* point (farther than ``PRIM_EPS`` from the patch's
  own primitive samples) in the passage a family is defined by: the ring's
  gap (the annulus r +- 0.6 m across the gap's angles) and the space under
  the lintel of ``bars_lintel``.
* ``points`` -- obstacle points (region 0) <= ``--budget`` (``OBSTACLE_BUDGET``).
* ``spec`` -- ``spec.json`` equals ``data_make_dataset.build_spec(pid)``
  field for field, and ``course.json`` ``patch`` is the pid (the course is
  of the patch it is named after). The pid is the plan row's, else the
  course directory's name (``<family>_<seed>``; a name that does not parse
  fails): never read back from ``spec.json`` itself.
* ``build_s`` -- the build's CPU seconds to the cloud (``course.json``
  ``cpu_seconds_build``: scene + cloud, not previews or packing; the wall
  ``seconds_build`` for a course that predates it) <= ``--max-build-s``. CPU,
  not wall: on a shared box under load 25-45 the wall time of one scene
  moves several-fold, and a slot must not be re-seeded by the load.
* ``provenance`` (contract 6) -- the ``course.json`` ``provenance`` block
  says a clean build: a git sha, ``dirty`` exactly False, no kit env hook,
  Python's str hash not salted (``provenance.check_clean(recorded=True)``);
  with ``expect`` also the given ``git_sha`` / ``library_sha``. Off with
  ``check_provenance=False`` (``--allow-dirty``: trial batches).
* ``pack`` -- a course meant to travel packed (a kit row or a plan3 v1 row,
  as ``batch.build_cmd`` packs them) records ``pack.packed``, and any packed
  course an empty ``pack.unpacked_left`` (``pack_all`` only warns on a
  missing source file); with a ``pack_check.py`` result (``batch.py gates
  --pack-check``), that says ok too. The gate reads the build's record, not
  ``course.blend`` itself, which ``pull`` skips and ``render --s3`` deletes.
* ``contract`` -- the cloud was built at this code's voxel and top
  (``CLOUD_VOXEL`` 0.10 m, ``CLOUD_ZMAX`` 4.75 m): a course built before the
  09-27 contract fails rather than mixing clouds.

Which failures spend a spare (``batch.resolve_scenes``): the scene's own --
``rc``, ``clearance_*``, ``slots``, ``gaps``, ``points``, ``build_s`` (``SCENE_FAILS``). Any other (``infra``, ``provenance``, ``pack``,
``contract``, ``spec``, a gate crash) says the build or the code is wrong,
not the scene: the row carries ``infra`` (the reasons) and its slot waits
for a rebuild, even when it fails a scene gate as well (a dirty or stale
build's geometry proves nothing).

Recorded, not gating: ``triangles``, ``obstacle_zmax_m``, ``approach_points``
(dressing inside the keep-out rectangles in front of and behind the wall
openings and round the bars, at the scene's keepout), ``interior_points``
(dressing inside a room or a ring, where the goal is), ``under_points``
(dressing under a floating box or bar: a post there changes the family), the
provenance (``git_sha``, ``dirty``, ``library_sha``), the wall seconds and the
pack stats.

numpy + scipy; imports the repo's ``src``/``scripts`` for ``build_spec`` and
``patch.sample`` (never the scene-cloud hook: ``SUPERFLY_SCENE_CLOUDS`` is
dropped from this process's environment first, so a tier B/C spec is rebuilt
on its primitives).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import provenance as PV  # noqa: E402  (stdlib only)
from band_shim import band as _band  # noqa: E402

CLEAR_M = 1.5
BAND = (0.3, 3.2)             # the flight band the clearance looks at (kitcheck.py): band.Band.gates
#: obstacle points (region 0) a course may carry. At 0.10 m the 09-25 kit previews
#: re-voxelised run p50 ~110k, p95 ~300k, max ~410k (the ruins arena); labelling cost is
#: linear in them (value_cost_torch chunked cdist). 400k keeps every tier-A preview in
#: and stops the warehouse-roof kind of outlier (see build_gates tests / the 09-27 report).
OBSTACLE_BUDGET = 400_000
MAX_BUILD_S = 600.0
CONTRACT = {"voxel_m": 0.10, "zmax_m": 4.75}   # build_course.CLOUD_VOXEL / CLOUD_ZMAX (bpy-free copy; band.cloud_top)
_band.bind(globals(), BAND=lambda b: b.gates, CONTRACT=lambda b: {"voxel_m": 0.10, "zmax_m": b.cloud_top})
PRIM_EPS = 0.2                # a cloud point this close to a primitive sample is the primitive's own
SLOT_INSET, SLOT_REACH = 0.06, 0.6
TIER_OF = {"corridor": "B", "arena": "C"}
#: gate failures that are the scene's own and spend a spare; every other is infra (module doc)
SCENE_FAILS = ("rc", "clearance_start", "clearance_goal", "slots", "gaps", "points", "build_s")


def pid_parts(pid: str) -> tuple[str, int] | None:
    """``(family, seed)`` of a pid ``<family>_<seed>``, None if it is not one."""
    m = re.fullmatch(r"([a-z][a-z_]*?)_(\d+)", pid)
    return (m.group(1), int(m.group(2))) if m else None


def infra_reason(rc, have: bool, build: dict | None) -> str | None:
    """Why a build's outcome says nothing about its scene (see ``infra`` in the module doc), else None.
    A build record with ``timed_out`` (seconds: the caller killed it at its wall-clock limit,
    ``pipeline.py build --timeout``) is a kill whatever rc it reports."""
    if build is not None and build.get("timed_out"):
        return f"killed at the {float(build['timed_out']):.0f} s wall-clock timeout (rc {rc})"
    if rc is None or rc == 0:
        if have:
            return None
        return ("no course.json yet and no build record" if build is None
                else "rc 0 but course.json/cloud.npz/spec.json missing (deleted since?)")
    if rc == 1:
        return None                                   # the script raised on this scene
    if rc == 3:
        return "refused by --require-clean (rc 3)"
    if rc < 0 or rc > 128:
        return f"killed by signal {-rc if rc < 0 else rc - 128} (rc {rc})"
    return f"rc {rc} (arguments or environment)"


def _src_path() -> None:
    """The repo root (``envgen``) and this folder (``specs``) on the path."""
    for p in (REPO, REPO / "blender"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


# --- geometry checks (pure numpy; the unit tests drive these) ------------------------------------

def band_points(O: np.ndarray, band=BAND) -> np.ndarray:
    return O[(O[:, 2] > band[0]) & (O[:, 2] < band[1])]


def clearance_xy(O: np.ndarray, xy, band=BAND) -> float:
    """xy distance from ``xy`` to the nearest obstacle point in the flight band (inf if none)."""
    B = band_points(O, band)
    if not len(B):
        return math.inf
    return float(np.hypot(B[:, 0] - xy[0], B[:, 1] - xy[1]).min())


def _wall_frame(p: dict):
    p0, p1 = np.asarray(p["p0"], float)[:2], np.asarray(p["p1"], float)[:2]
    L = float(np.linalg.norm(p1 - p0))
    d = (p1 - p0) / L
    return p0, d, np.array([-d[1], d[0]]), L


def slot_points(O: np.ndarray, prims: list, zmax: float = 4.4) -> int:
    """Cloud points inside any wall opening (``kitcheck.py``): along the wall
    the slot shrunk by ``SLOT_INSET``, across it the wall's thickness plus
    ``SLOT_REACH`` each side, in height the slot shrunk by ``SLOT_INSET``."""
    bad = 0
    for p in prims:
        if p["kind"] != "wall" or not p.get("slots"):
            continue
        p0, d, n, _ = _wall_frame(p)
        rel = O[:, :2] - p0
        u, v = rel @ d, rel @ n
        for s, w, z0, z1 in p["slots"]:
            m = ((u > s - w / 2 + SLOT_INSET) & (u < s + w / 2 - SLOT_INSET) & (np.abs(v) < p["t"] / 2 + SLOT_REACH)
                 & (O[:, 2] > z0 + SLOT_INSET) & (O[:, 2] < min(z1, zmax) - SLOT_INSET))
            bad += int(m.sum())
    return bad


def dressing_mask(O: np.ndarray, prim_pts: np.ndarray, eps: float = PRIM_EPS) -> np.ndarray:
    """True for cloud points farther than ``eps`` from every primitive sample: the kit's dressing."""
    if not len(prim_pts):
        return np.ones(len(O), bool)
    from scipy.spatial import cKDTree  # noqa: PLC0415
    d, _ = cKDTree(prim_pts).query(O, k=1, distance_upper_bound=eps + 1e-9)
    return ~np.isfinite(d)


def gap_points(D: np.ndarray, spec: dict) -> dict:
    """Dressing points (``D``) in the passage a family is defined by: the
    ring's gap, the space under the lintel. ``{}`` for other families."""
    fam, pr = spec["family"], spec.get("params", {})
    out = {}
    if fam == "ring" and not pr.get("closed") and pr.get("gap_deg", 0) > 0:
        c, r = np.asarray(pr["centre"], float), float(pr["r"])
        rel = D[:, :2] - c
        rho = np.hypot(rel[:, 0], rel[:, 1])
        ang = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
        dev = np.abs((ang - pr["gap_at_deg"] + 180.0) % 360.0 - 180.0)
        half = pr["gap_deg"] / 2 - math.degrees(0.45 / r)         # a pillar (0.18 m) + margin in from each side
        m = (np.abs(rho - r) < 0.6) & (dev < half) & (D[:, 2] > BAND[0]) & (D[:, 2] < BAND[1])
        out["ring_gap"] = int(m.sum())
    if fam == "bars_lintel":
        lint = next((p for p in spec["primitives"] if p["kind"] == "box"), None)
        if lint is not None:
            cx, hx = float(lint["c"][0]), float(lint["half"][0])
            y0, y1 = spec["bounds"][1] + 0.5, spec["bounds"][3] - 0.5
            m = ((D[:, 0] > cx - hx - 0.3) & (D[:, 0] < cx + hx + 0.3) & (D[:, 1] > y0) & (D[:, 1] < y1)
                 & (D[:, 2] > BAND[0]) & (D[:, 2] < pr["z_under"] - 0.1))
            out["under_lintel"] = int(m.sum())
    return out


def approach_points(D: np.ndarray, prims: list, keepout: float) -> int:
    """Dressing points in the flight band inside the keep-out rectangles
    ``build_course.build_primitives`` claims: in front of and behind every
    wall opening (half extents w/2 + 0.4 keepout along, keepout across) and
    round every horizontal bar (r + 0.6 keepout across)."""
    B = D[(D[:, 2] > BAND[0]) & (D[:, 2] < BAND[1])]
    n = np.zeros(len(B), bool)
    for p in prims:
        if p["kind"] == "wall":
            p0, d, nv, _ = _wall_frame(p)
            rel = B[:, :2] - p0
            u, v = rel @ d, rel @ nv
            for s, w, _, _ in p.get("slots", []):
                n |= (np.abs(u - s) < w / 2 + 0.4 * keepout) & (np.abs(v) < keepout)
        elif p["kind"] == "cyl_h":
            a, b = np.asarray(p["a"], float)[:2], np.asarray(p["b"], float)[:2]
            L = float(np.linalg.norm(b - a))
            d = (b - a) / max(L, 1e-9)
            rel = B[:, :2] - (a + b) / 2
            u, v = rel @ d, rel @ np.array([-d[1], d[0]])
            n |= (np.abs(u) < L / 2) & (np.abs(v) < p["r"] + 0.6 * keepout)
    return int(n.sum())


def interior_points(D: np.ndarray, spec: dict) -> int:
    """Dressing points in the band inside a room's walls or a ring (goal rule ``inside``)."""
    fam, pr = spec["family"], spec.get("params", {})
    B = D[(D[:, 2] > BAND[0]) & (D[:, 2] < BAND[1])]
    if fam == "ring":
        c = np.asarray(pr["centre"], float)
        return int((np.hypot(B[:, 0] - c[0], B[:, 1] - c[1]) < float(pr["r"]) - 0.4).sum())
    if fam == "room":
        walls = [p for p in spec["primitives"] if p["kind"] == "wall"]
        xy = np.array([q for p in walls for q in (p["p0"][:2], p["p1"][:2])], float)
        lo, hi = xy.min(0) + 0.3, xy.max(0) - 0.3
        return int(((B[:, 0] > lo[0]) & (B[:, 0] < hi[0]) & (B[:, 1] > lo[1]) & (B[:, 1] < hi[1])).sum())
    return 0


def under_points(D: np.ndarray, spec: dict) -> int:
    """Dressing points in the band under a floating box or bar (``float_*``): a post there changes the family."""
    if not spec["family"].startswith("float"):
        return 0
    n = 0
    for p in spec["primitives"]:
        if p["kind"] != "box":
            continue
        c, h = np.asarray(p["c"], float), np.asarray(p["half"], float)
        yaw = float(p.get("rpy", (0, 0, 0))[2])
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        q = (D[:, :2] - c[:2]) @ R
        n += int(((np.abs(q[:, 0]) < h[0]) & (np.abs(q[:, 1]) < h[1]) & (D[:, 2] > BAND[0])
                  & (D[:, 2] < c[2] - h[2] - 0.05)).sum())
    return n


# --- the row -------------------------------------------------------------------------------

def spec_for(scene: dict) -> dict:
    """``build_spec`` of a plan row (or of a pid, inferring the tier) as a dict."""
    _src_path()
    import specs as dmd  # noqa: PLC0415
    tier = scene.get("tier") or TIER_OF.get(scene["family"], "A")
    return json.loads(dmd.build_spec(5, tier, scene["family"], int(scene["seed"]), "train").to_json())


def primitive_samples(spec: dict) -> np.ndarray:
    """The patch's own primitive surface samples (``patch.sample`` at 0.1 m, no floor)."""
    _src_path()
    from envgen.primitives import patch as PT  # noqa: PLC0415
    pat = PT.Patch(seed=spec["seed"], regime="", family=spec["family"], bounds=list(spec["bounds"]),
                   flight_z=spec["flight_z"], state_pos=[0.0, 0.0, spec["flight_z"]], state_vel=[3.0, 0.0, 0.0],
                   goal_pos=list(spec["goal_pos"]), goal_speed=spec["goal_speed"], goal_radius=spec["goal_radius"],
                   primitives=spec["primitives"])
    P, reg = PT.sample(pat, spacing=0.1, floor_spacing=0.25)
    return np.asarray(P, float)[np.asarray(reg) >= 0]


def _diff(a, b, path="", out=None, tol=1e-9):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            _diff(a.get(k), b.get(k), f"{path}.{k}", out, tol)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(float(a) - float(b)) > tol:
            out.append(path)
    elif a != b:
        out.append(path)
    return out


def pack_problems(cj: dict, need: bool, check: dict | None = None) -> list[str]:
    """Why a course meant to travel packed would not render alone on a node; [] = fine."""
    pk = cj.get("pack") or {}
    why = []
    if need and not pk.get("packed"):
        why.append("not packed")
    if pk.get("packed") and pk.get("unpacked_left"):
        why.append(f"unpacked after pack_all: {pk['unpacked_left'][:5]}")
    if check is not None and not check.get("ok"):
        why.append("pack_check: " + json.dumps({k: check.get(k) for k in
                                                ("unpacked", "failed_to_load", "fonts_unpacked", "sounds_unpacked",
                                                 "libraries", "error") if check.get(k)})[:300])
    return why


def gate_course(course: Path, scene: dict | None = None, build: dict | None = None,
                budget: int = OBSTACLE_BUDGET, max_build_s: float = MAX_BUILD_S,
                check_spec: bool = True, contract: dict | None = CONTRACT, check_provenance: bool = True,
                expect: dict | None = None, require_pack: bool | None = None,
                pack_check: dict | None = None, clear_m: float = CLEAR_M) -> dict:
    """The gate row of one built course (see module doc). ``scene`` is its
    plan row (tier/family/seed/knobs; else the pid is the directory name),
    ``build`` its build record (``{"rc": ..., "seconds_wall": ...}``, plus
    ``timed_out`` when the caller killed it at a wall-clock limit),
    ``clear_m`` the start/goal dressing clearance (``CLEAR_M``),
    ``expect`` provenance values it must carry (``{"git_sha": ...,
    "library_sha": ...}``; a None value is not checked), ``require_pack``
    whether it must be packed (None: kit and plan3 v1 rows must; any course
    that is packed is checked for leftovers), ``pack_check`` a
    ``pack_check.py`` result."""
    course = Path(course)
    pid = course.name if scene is None else scene["pid"]
    row: dict = {"pid": pid, "course": str(course), "fails": []}
    rc = None if build is None else build.get("rc")
    cj_p, cl_p, sp_p = course / "course.json", course / "cloud.npz", course / "spec.json"
    have = cj_p.exists() and cl_p.exists() and sp_p.exists()
    row["rc"] = rc if rc is not None else (0 if have else None)
    if build is not None:
        row["seconds_wall"] = build.get("seconds_wall")
    infra = infra_reason(rc, have, build)
    if infra is not None:                       # not the scene's fault: rebuild, do not spend a spare
        row.update(infra=infra, ok=False)
        row["fails"].append("infra")
        return row
    if not have or (rc not in (None, 0)):
        row["fails"].append("rc")
        row["ok"] = False
        return row
    cj = json.loads(cj_p.read_text())
    spec = json.loads(sp_p.read_text())
    spec_why = []
    if scene is None:                           # the pid is the directory's name, never spec.json's
        parts = pid_parts(pid)
        if parts is None:
            spec_why.append(f"pid: {pid!r} is not <family>_<seed>")
            parts = (spec["family"], int(spec["seed"]))
        scene = {"pid": pid, "family": parts[0], "seed": parts[1], "tier": TIER_OF.get(parts[0], "A"),
                 "knobs": cj.get("knobs", {})}
    z = np.load(cl_p)
    P, reg = np.asarray(z["points"], float), np.asarray(z["region"])
    O = P[reg >= 0]
    cl = cj.get("cloud", {})
    row.update({"kit": cj.get("kit"), "theme": cj.get("theme"), "level": cj.get("level"),
                "obstacle_points": int(len(O)), "points": int(len(P)),
                "obstacle_zmax_m": round(float(O[:, 2].max()), 3) if len(O) else None,
                "voxel_m": cl.get("voxel_m"), "zmax_m": cl.get("zmax_m"),
                "triangles": cj.get("mesh", {}).get("triangles"), "seconds_build": cj.get("seconds_build"),
                "cpu_seconds_build": cj.get("cpu_seconds_build"),
                "seconds_total": cj.get("seconds_total"), "blend_bytes": (cj.get("pack") or {}).get("blend_bytes"),
                "git_sha": cj.get("git_sha"), "dirty": cj.get("dirty"), "library_sha": cj.get("library_sha"),
                "cloud_sha256": cl.get("sha256")})
    prim = primitive_samples(spec)
    D = O[dressing_mask(O, prim)]
    row["dressing_points"] = int(len(D))
    ends = {"start": (0.0, 0.0), "goal": tuple(spec["goal_pos"][:2])}
    for end, xy in ends.items():                # the gate is on the dressing; the primitives are the patch
        row[f"{end}_clear_dressing_m"] = min(round(clearance_xy(D, xy), 3), 999.0)
        row[f"{end}_clear_m"] = min(round(clearance_xy(O, xy), 3), 999.0)
        row[f"{end}_clear_prim_m"] = min(round(clearance_xy(prim, xy), 3), 999.0)
        if row[f"{end}_clear_dressing_m"] < clear_m:
            row["fails"].append(f"clearance_{end}")
    row["slot_points"] = slot_points(O, spec["primitives"])
    if row["slot_points"]:
        row["fails"].append("slots")
    gaps = gap_points(D, spec)
    row["gap_points"] = gaps
    if any(gaps.values()):
        row["fails"].append("gaps")
    knobs = scene.get("knobs") or {k: scene[k] for k in ("keepout",) if k in scene} or cj.get("knobs", {})
    row["approach_points"] = approach_points(D, spec["primitives"], float(knobs.get("keepout", 3.0)))
    row["interior_points"] = interior_points(D, spec)
    row["under_points"] = under_points(D, spec)
    if len(O) > budget:
        row["fails"].append("points")
    if check_spec:
        want = spec_for(scene)
        if cj.get("patch") != pid:              # the pid-level causes first: the field diff is truncated
            spec_why.append(f".patch (course.json) {cj.get('patch')!r} != {pid!r}")
        diff = spec_why + _diff(spec, want)
        row["spec_match"] = not diff
        if diff:
            row["spec_diff"] = diff[:10]
            row["fails"].append("spec")
    secs = cj.get("cpu_seconds_build")
    if secs is None:                            # a course from before the field: wall seconds
        secs = cj.get("seconds_build")
    if (secs or 0.0) > max_build_s:
        row["fails"].append("build_s")
    prov_why = (PV.check_clean(cj["provenance"], recorded=True) if isinstance(cj.get("provenance"), dict)
                else ["no provenance block (built before contract 6)"])
    row["provenance_why"] = prov_why
    exp_why = [f"{k} {cj.get(k)} != expected {v}" for k, v in (expect or {}).items()
               if v and not str(cj.get(k) or "").startswith(v)]       # a short sha names its commit
    if exp_why:
        row["provenance_expected"] = exp_why
    if (check_provenance and prov_why) or exp_why:
        row["fails"].append("provenance")
    if require_pack is None:                    # batch.build_cmd's rule: kit rows and plan3 v1 rows are packed
        require_pack = bool(scene.get("kit") or scene.get("source") == "v1")
    pack_why = pack_problems(cj, require_pack, pack_check)
    row["pack_why"] = pack_why
    if pack_check is not None:
        row["pack_check_ok"] = bool(pack_check.get("ok"))
    if pack_why:
        row["fails"].append("pack")
    if contract is not None and (cl.get("voxel_m") != contract["voxel_m"] or cl.get("zmax_m") != contract["zmax_m"]):
        row["fails"].append("contract")
    infra = [f for f in row["fails"] if f not in SCENE_FAILS]
    if infra:
        row["infra"] = "; ".join(infra)
    row["ok"] = not row["fails"]
    return row


def error_row(pid: str, e: BaseException) -> dict:
    """The row of a scene whose gate crashed: infra (rebuild or fix the gate), never a spent spare."""
    return {"pid": pid, "ok": False, "fails": ["error"], "infra": f"gate error: {type(e).__name__}: {e}",
            "error": f"{type(e).__name__}: {e}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("courses", nargs="+", type=Path, help="course dirs (<...>/courses/<pid>)")
    ap.add_argument("--plan", type=Path, default=None, help="plan.json whose rows (scenes + spares) name the courses")
    ap.add_argument("--builds", type=Path, default=None, help="dir of build records <pid>.json (batch.py build)")
    ap.add_argument("--out", type=Path, default=None, help="write the rows as JSON here")
    ap.add_argument("--budget", type=int, default=OBSTACLE_BUDGET)
    ap.add_argument("--max-build-s", type=float, default=MAX_BUILD_S)
    ap.add_argument("--clear-m", type=float, default=CLEAR_M, help="start/goal dressing clearance (m)")
    ap.add_argument("--no-spec", action="store_true", help="skip the build_spec comparison")
    ap.add_argument("--any-contract", action="store_true", help="do not gate on the cloud voxel/top")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="do not gate on the recorded provenance (trial builds from a dirty tree)")
    ap.add_argument("--expect-git-sha", default=None, help="fail a course built from another commit")
    ap.add_argument("--expect-library-sha", default=None, help="fail a course built from other library indexes")
    a = ap.parse_args(argv)
    expect = {"git_sha": a.expect_git_sha, "library_sha": a.expect_library_sha}
    rows_by = {}
    if a.plan:
        p = json.loads(a.plan.read_text())
        rows_by = {s["pid"]: s for s in p.get("scenes", []) + p.get("spares", [])}
    out = []
    for c in a.courses:
        b = None
        if a.builds and (a.builds / f"{c.name}.json").exists():
            b = json.loads((a.builds / f"{c.name}.json").read_text())
        r = gate_course(c, rows_by.get(c.name), b, a.budget, a.max_build_s, not a.no_spec,
                        None if a.any_contract else CONTRACT, check_provenance=not a.allow_dirty, expect=expect,
                        clear_m=a.clear_m)
        out.append(r)
        print(f"{r['pid']:24s} {'ok  ' if r['ok'] else 'FAIL'} {','.join(r['fails']):24s} "
              f"obs {r.get('obstacle_points', 0) or 0:>7,} start {r.get('start_clear_m')} goal {r.get('goal_clear_m')} "
              f"slot {r.get('slot_points')} gaps {r.get('gap_points')} "
              f"build {r.get('cpu_seconds_build') or r.get('seconds_build')} s"
              + (f"  infra: {r['infra']}" if r.get("infra") else "")
              + (f"  provenance: {r['provenance_why'] + r.get('provenance_expected', [])}"
                 if "provenance" in r["fails"] else "")
              + (f"  spec: {r.get('spec_diff')}" if "spec" in r["fails"] else ""), flush=True)
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0 if all(r["ok"] for r in out) else 1


if __name__ == "__main__":
    sys.exit(main())
