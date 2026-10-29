#!/usr/bin/env python3
"""What the Blender render reads and writes: the output layout (contract 5 of
``deliverables/2026-09-27/BLENDER_V7_PIPELINE.md``), the drawn-state input,
and the merge of a per-scene render into a labelled rollout.

stdlib + numpy only, so ``render_frames.py`` imports it inside Blender's
bundled python, ``finish_frames.py`` and ``render_runner.py`` on the repo side.

**Layout.** Frame ``i = k + 1`` for state ``k`` -- the writer's numbering
(``data_writer.add_state``: ``img/depth_{k+1:08d}.tif``, odometry row ``k``,
``trajectories_bf_{k:08d}.csv``), which is the student loader's
(``data_loader._decode_experiment_dir``: row ``r`` reads ``depth_{r+1}``,
``frame_left_{r+1}.png``, ``trajectories_<ref>_{r}``). A finished rollout's
``img/`` holds exactly two files per state and nothing else, because the loader
counts every ``img/`` name starting ``depth_`` (depth arms) or containing
``frame_left`` (RGB arms) against the odometry rows, and ``t5_label_all``'s
merge check counts every ``img/*.tif`` against the states:

    img/depth_%08d.tif          Blender mesh depth, 224² uint16 mm (render_depth.write_tif)
    img/frame_left_%08d.png     RGB 640x480 at the state
    aux/depth_prim_%08d.tif     the primitives' frame (what the labeller wrote)
    aux/depth640_%08d.png       Blender z-depth at the native 640x480, uint16 mm, 0 = no surface
                                (rgb-v9 renders; ``DEPTH640_*``)
    aux/depthprev_%08d.tif      depth dt earlier, same contract
    aux/rgbprev_%08d.png        RGB dt earlier
    aux/flow_%08d.npz           backward flow (finish_frames)
    aux/render_poses.json       both poses per frame, K, dt, device, samples, timings
    render_manifest.jsonl       one line per state with every modality's path
    render_manifest.primitives.jsonl   the writer's manifest, as it was (rollouts only)
    blender_checks.json         parity / cloud / flow checks

A *render dir* (``render_frames --states``: rendered from the drawn states,
beside labelling) has the same layout for its scene, numbered by the drawn
state index; :func:`merge_into_rollout` drops it into the labelled rollout,
mapping ``state_k`` to the rollout's frame through its ``render_manifest.jsonl``
and refusing a state whose pose differs, and a render of another course build.

**Which build** (:func:`course_source`): every render records the sha256 of the
course's ``cloud.npz`` / ``spec.json`` / ``course.json`` (and ``course.blend``,
plus ``course.json``'s git_sha / library_sha) in ``poses.json``, each manifest
line's ``render`` block, ``blender_checks.json`` and the runner's ``.done`` --
the same hashes the draw records and the labels carry as ``cloud_source``
(contract 1), so :func:`check_course` ties states, labels and pictures to one
build (render_frames before rendering, the merge before placing a file).

**Drawn states** (:func:`load_drawn`): a phase-R draw / routes record
(``prm_routes.draw_record``: ``goals[].states[].state``, the whole drawn state,
which is what phase L labels -- ``data_make_dataset.groups_from_record``) or a
set_M-style ``states/<pid>.json`` whose entries carry the whole state (``R`` or
``quat_wxyz``, ``omega``, ``acc_drawn``). Both are flattened in goal order and
renumbered ``k = 0..N-1`` exactly as ``data_make_dataset._renumber`` does.

    python blender/render_io.py merge --render <dir>/<pid> --rollout <data>/train/rollout_<seed5>_<family>
    python blender/render_io.py convert-legacy --rollout <rollout>     # batch-1 layout -> this one
    python blender/render_io.py check --rollout <rollout> [...] [--rgb] [--no-depth]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flowlib as FL  # noqa: E402

DEPTH = "img/depth_{:08d}.tif"
RGB = "img/frame_left_{:08d}.png"
DEPTH_PRIM = "aux/depth_prim_{:08d}.tif"
DEPTH_PREV = "aux/depthprev_{:08d}.tif"
RGB_PREV = "aux/rgbprev_{:08d}.png"
FLOW = "aux/flow_{:08d}.npz"
#: native-resolution z-depth (rgb-v9, 2026-10-01): uint16 PNG in mm, 0 = no surface (sky / beyond
#: 65.534 m is 65535), the auxiliary depth target an RGB student trains against
DEPTH640 = "aux/depth640_{:08d}.png"
DEPTH640_ENCODING = {"format": "png16", "unit": "mm", "round": "half-to-even (numpy.round)",
                     "zero": "no surface", "max_mm": 65535, "max_means": ">= 65.535 m", "depth": "z_along_optical_axis"}
POSES = "aux/render_poses.json"
MANIFEST = "render_manifest.jsonl"
MANIFEST_PRIM = "render_manifest.primitives.jsonl"
CHECKS = "blender_checks.json"
#: the raw Blender output (render_frames). finish_frames leaves it (batch_sheet reads it);
#: ``batch.py`` deletes it after the sheet unless ``--keep-raw``, render_runner never tars it
RAW = "blender_raw"

#: batch 1's layout (finish_frames before 2026-09-27): everything in img/
LEGACY = {"depth_prim": "img/depth_prim_{:08d}.tif", "depth_prev": "img/depthprev_{:08d}.tif",
          "rgb": "img/rgb_{:08d}.png", "rgb_prev": "img/rgbprev_{:08d}.png", "flow": "img/flow_{:08d}.npz"}


def frame_index(k: int) -> int:
    """The writer's frame number of state ``k``."""
    return int(k) + 1


def modalities(i: int, prev: bool = True, depth640: bool = False) -> dict:
    """Every stored modality of frame ``i``, relative to the rollout / render dir.
    ``prev=False``: a render without the previous frame (rgb-v9: no rgbprev, depthprev or
    flow); ``depth640``: with the native-resolution depth."""
    out = {"depth": DEPTH.format(i), "rgb": RGB.format(i), "depth_prim": DEPTH_PRIM.format(i)}
    if prev:
        out.update({"depth_prev": DEPTH_PREV.format(i), "rgb_prev": RGB_PREV.format(i), "flow": FLOW.format(i)})
    if depth640:
        out["depth640"] = DEPTH640.format(i)
    return out


def write_depth640(path, depth_m: np.ndarray) -> None:
    """``DEPTH640_ENCODING``: uint16 mm PNG, 0 where no surface (inf / nan / <= 0)."""
    from PIL import Image  # noqa: PLC0415 -- repo side only
    d = np.asarray(depth_m, np.float64)
    ok = np.isfinite(d) & (d > 0)
    mm = np.where(ok, np.clip(np.round(np.where(ok, d, 0.0) * 1000.0), 1, 65535), 0).astype(np.uint16)
    Image.fromarray(mm).save(str(path), format="PNG")


def read_depth640(path) -> np.ndarray:
    """metres, ``inf`` where no surface."""
    from PIL import Image  # noqa: PLC0415
    with Image.open(str(path)) as im:
        mm = np.asarray(im, np.uint16)
    return np.where(mm > 0, mm.astype(np.float32) / 1000.0, np.inf).astype(np.float32)


def index_of(frame: str) -> int:
    """``img/depth_00000007.tif`` -> 7."""
    stem = Path(frame).stem
    return int(stem.rsplit("_", 1)[1])


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


# --- which course build a render is of -----------------------------------------------------------

#: what two ``cloud_source`` values must share to name one course build: the files, not the
#: directory they sit in (``draw_states.CLOUD_SOURCE_KEYS``: a gs2 path and an OSMO job's
#: unpacked path name the same course)
CLOUD_SOURCE_KEYS = ("pid", "cloud_sha256", "spec_sha256", "course_json_sha256")
#: ``course.json``'s top-level contract-6 provenance, carried beside the hashes
COURSE_BUILD_KEYS = ("git_sha", "dirty", "library_sha", "blender")


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def course_source(course, pid: str | None = None, blend: bool = False) -> dict:
    """The course build a render reads, by file hash: ``draw_states.SceneCourse.source``'s
    ``{dir, pid, cloud_sha256, spec_sha256, course_json_sha256}`` (the draw records and the
    labels carry exactly this as ``cloud_source``, contract 1), plus ``course.json``'s
    ``git_sha`` / ``dirty`` / ``library_sha`` / ``blender`` and, with ``blend``, the hash of
    ``course.blend`` itself. ``pid``: ``course.json``'s ``patch``, else the directory name."""
    c = Path(course)
    cj_p = c / "course.json"
    cj = json.loads(cj_p.read_text()) if cj_p.is_file() else {}
    out = {"dir": str(c), "pid": pid or cj.get("patch") or c.name,
           "cloud_sha256": file_sha256(c / "cloud.npz") if (c / "cloud.npz").is_file() else None,
           "spec_sha256": file_sha256(c / "spec.json") if (c / "spec.json").is_file() else None,
           "course_json_sha256": file_sha256(cj_p) if cj_p.is_file() else None}
    out.update({k: cj.get(k) for k in COURSE_BUILD_KEYS})
    if blend:
        out["blend_sha256"] = file_sha256(c / "course.blend") if (c / "course.blend").is_file() else None
    return out


def same_source(a, b) -> bool:
    """``draw_states.same_cloud_source``: both None, or the same files."""
    if a is None or b is None:
        return a is None and b is None
    return all(a.get(k) == b.get(k) for k in CLOUD_SOURCE_KEYS)


def source_str(s) -> str:
    if not isinstance(s, dict):
        return repr(s)
    return (f"{s.get('pid')} cloud {str(s.get('cloud_sha256'))[:12]} spec {str(s.get('spec_sha256'))[:12]} "
            f"course.json {str(s.get('course_json_sha256'))[:12]}")


class CourseMismatch(RuntimeError):
    pass


def check_course(theirs, course: dict, what: str) -> str:
    """``theirs``: the ``cloud_source`` the drawn states or the labels record (``MISSING``
    when they record none) against ``course`` (:func:`course_source` of the files being
    rendered). ``"match"``, ``"unverified"`` (nothing recorded: batch 1, setM7) or raises
    :class:`CourseMismatch` -- a record of another build, or of the primitives (``None``),
    would pair pictures with states or labels planned in different geometry."""
    if theirs is MISSING:
        return "unverified"
    if theirs is None:
        raise CourseMismatch(f"{what} were made in the primitives' cloud (cloud_source null), not in the course "
                             f"{source_str(course)}")
    if not same_source(theirs, course):
        diff = [k for k in CLOUD_SOURCE_KEYS if theirs.get(k) != course.get(k)]
        raise CourseMismatch(f"{what} were made on another build of the course (differ in {', '.join(diff)}): "
                             f"{source_str(theirs)} vs the course {source_str(course)}")
    return "match"


#: what a re-skinned course's ``course.json`` ``reskin`` block must carry (``blender/reskin.py
#: course-check``): the hashes of the files it checked and its verdict
RESKIN_KEYS = ("ok", "geometry_hash", "spec_sha256", "cloud_sha256")


def check_reskin(course, geometry_hash, what: str) -> str:
    """Re-skin mode (PLAN_B phase 1): states labelled in the *primitives' cloud* (their
    ``cloud_source`` is null / absent) rendered in a course built ``--clutter 0 --hang 0``
    round the same primitives. The tie is the geometry, not the cloud file: ``course.json``'s
    ``reskin`` block (``blender/reskin.py course-check``) must say ``ok`` for this course's
    ``spec.json`` and ``cloud.npz`` (by sha256), and its ``geometry_hash`` -- the hash of the
    spec's primitives, the manifest's scene key -- must equal the states'. Returns
    ``"reskin-geometry"``; raises :class:`CourseMismatch`."""
    c = Path(course)
    cj = json.loads((c / "course.json").read_text())
    rs = cj.get("reskin")
    if not isinstance(rs, dict) or any(k not in rs for k in RESKIN_KEYS):
        raise CourseMismatch(f"{c}: no re-skin geometry check in course.json (blender/reskin.py course-check)")
    if not rs["ok"]:
        raise CourseMismatch(f"{c}: the re-skin geometry check failed: {rs.get('why')}")
    for key, name in (("spec_sha256", "spec.json"), ("cloud_sha256", "cloud.npz")):
        if rs[key] != file_sha256(c / name):
            raise CourseMismatch(f"{c}: {name} changed after the re-skin geometry check")
    if not geometry_hash:
        raise CourseMismatch(f"{what} carry no geometry_hash: nothing ties them to the course's primitives")
    if geometry_hash != rs["geometry_hash"]:
        raise CourseMismatch(f"{what} are of other primitives: geometry_hash {str(geometry_hash)[:19]} vs the "
                             f"course's {str(rs['geometry_hash'])[:19]}")
    return "reskin-geometry"


class _Missing:
    def __repr__(self):
        return "MISSING"


#: a record that says nothing about its cloud (key absent), as opposed to ``None`` (the primitives)
MISSING = _Missing()


def rollout_cloud_source(ro, lines: list[dict] | None = None):
    """The ``cloud_source`` a labelled rollout was planned in: its writer's manifest lines
    (``manifest_const``) and ``labels_meta.json`` (contract 1). :data:`MISSING` when none
    records one; a rollout whose records disagree raises :class:`CourseMismatch`."""
    ro = Path(ro)
    if lines is None:
        p = ro / MANIFEST_PRIM if (ro / MANIFEST_PRIM).exists() else ro / MANIFEST
        lines = read_jsonl(p) if p.exists() else []
    vals = [m["cloud_source"] for m in lines if "cloud_source" in m]
    meta_p = ro / "labels_meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if "cloud_source" in meta:
            vals.append(meta["cloud_source"])
    if not vals:
        return MISSING
    if not all(same_source(v, vals[0]) for v in vals[1:]):
        raise CourseMismatch(f"{ro}: its manifest and labels_meta.json name different clouds: "
                             f"{sorted({source_str(v) for v in vals})}")
    return vals[0]


# --- drawn states ---------------------------------------------------------------------------------

class DrawnStateError(ValueError):
    pass


def _quat_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _full_state(st: dict, k: int, where: str) -> dict:
    """The fields a render needs, from one drawn state: the pose (position,
    R_wb), and velocity, drawn acceleration and body rates for the previous
    frame. ``acc_drawn`` defaults to zero like the writer's odometry does;
    the rest must be there -- a pose made up here would render a different
    image from the one the labels were planned for."""
    miss = [f for f in ("position", "velocity", "omega") if st.get(f) is None]
    if st.get("R") is None and st.get("quat_wxyz") is None:
        miss.append("R")
    if miss:
        raise DrawnStateError(
            f"{where}: state {k} has no {', '.join(miss)} -- a set_M states/<pid>.json summary does not carry "
            f"the attitude; pass the phase-R draw record (draw/<pid>.json) or the routes record instead")
    R = np.asarray(st["R"], float).reshape(3, 3) if st.get("R") is not None else _quat_to_R(st["quat_wxyz"])
    return {"k": k, "k_record": st.get("k"), "position": [float(v) for v in st["position"]],
            "R_wb": R.tolist(), "velocity": [float(v) for v in st["velocity"]],
            "acc": [float(v) for v in (st.get("acc_drawn") or (0.0, 0.0, 0.0))],
            "omega": [float(v) for v in st["omega"]]}


def load_drawn(src) -> dict:
    """``{"patch", "format", "cloud_source", "states": [{k, position, R_wb, velocity, acc, omega}]}``
    from a draw/routes record or a states file (path or parsed dict). ``cloud_source`` is
    the record's (contract 1: the course cloud the states were drawn in), :data:`MISSING`
    when it carries none."""
    d = json.loads(Path(src).read_text()) if not isinstance(src, dict) else src
    where = str(src) if not isinstance(src, dict) else "drawn states"
    cs = d.get("cloud_source", MISSING)
    if isinstance(d.get("goals"), list) and any("states" in g for g in d["goals"]):
        flat = [s for g in d["goals"] for s in (g.get("states") or [])]
        out = []
        for k, s in enumerate(flat):
            st = s.get("state")
            if not st:
                raise DrawnStateError(f"{where}: goal state {k} carries no drawn state (a record that predates "
                                      f"the record-load phase L); re-run the draw")
            out.append(_full_state(st, k, where))
        return {"patch": d.get("id"), "format": "record", "cloud_source": cs, "states": out,
                "geometry_hash": d.get("geometry_hash")}
    if isinstance(d.get("states"), list):
        return {"patch": d.get("patch") or d.get("id"), "format": "states", "cloud_source": cs,
                "geometry_hash": d.get("geometry_hash"),
                "states": [_full_state(st, k, where) for k, st in enumerate(d["states"])]}
    raise DrawnStateError(f"{where}: neither a draw/routes record (goals[].states[].state) nor a states file (states[])")


def _cam(name: str, hfov: float, t_bc=(0.0, 0.0, 0.0)) -> dict:
    fx = 0.5 * 640 / math.tan(math.radians(hfov) / 2)
    return {"name": name, "w": 640, "h": 480, "hfov_deg": hfov, "fx": fx, "fy": fx, "cx": 319.5, "cy": 239.5,
            "t_bc": [float(v) for v in t_bc], "pitch_deg": 0.0}


#: ``render_depth.CAMERAS`` as plain dicts (Blender's python cannot import the package;
#: ``tests/test_blender_render.py`` checks the two tables agree). ``agile_91`` is every set
#: made before 2026-10-01 and what the label writer uses; ``starling_front_v1`` the Starling 2
#: Max nose camera (87 deg, FLU (0.071, 0, -0.007) m, pitch 0), the rgb-v9 renders' default.
CAMS = {"agile_91": _cam("agile_91", 91.0),
        "starling_front_v1": _cam("starling_front_v1", 87.0, (0.071, 0.0, -0.007))}
RGB_V9_CAM = "starling_front_v1"
#: render_depth.Camera()'s pinhole: what the labeller's frames and the old manifests use
DEFAULT_CAM = {k: CAMS["agile_91"][k] for k in ("w", "h", "hfov_deg", "fx")}


def cam_info(name_or_block) -> dict:
    """A named camera (``CAMS``), or a manifest ``cam`` block (old lines carry no name or
    offset: ``agile_91`` at the body origin)."""
    if isinstance(name_or_block, str):
        if name_or_block not in CAMS:
            raise KeyError(f"unknown camera {name_or_block!r}: known {sorted(CAMS)}")
        return dict(CAMS[name_or_block])
    c = name_or_block
    return {"name": c.get("name", "agile_91"), "w": int(c["w"]), "h": int(c["h"]), "hfov_deg": float(c["hfov_deg"]),
           "fx": float(c["fx"]), "fy": float(c.get("fy", c["fx"])), "cx": float(c.get("cx", (int(c["w"]) - 1) / 2)),
           "cy": float(c.get("cy", (int(c["h"]) - 1) / 2)),
           "t_bc": [float(v) for v in c.get("t_bc_flu_m", c.get("t_bc", (0.0, 0.0, 0.0)))], "pitch_deg": 0.0}


def same_camera(a: dict, b: dict, tol: float = 1e-6) -> bool:
    """Two camera blocks (``cam_info``) that render the same pixels."""
    a, b = cam_info(a), cam_info(b)
    return (a["w"], a["h"]) == (b["w"], b["h"]) and abs(a["fx"] - b["fx"]) < tol and \
        np.allclose(a["t_bc"], b["t_bc"], atol=tol)


def _item(k, i, p, R, v, acc, w, dt) -> dict:
    p, R = np.array(p, float), np.array(R, float).reshape(3, 3)
    v, acc, w = (np.array(x, float) for x in (v, acc, w))
    pp, Rp = FL.previous_pose(p, R, v, acc, w, dt)
    return {"k": int(k), "i": int(i), "pose": (p, R), "prev": (pp, Rp), "v": v, "acc": acc, "omega": w}


def rollout_todo(ro: Path, dt: float):
    """``(todo, cam, patch)`` of a labelled rollout: pose from its (writer's)
    manifest, velocity / drawn acceleration / body rates from ``odometry.csv``,
    the previous pose ``dt`` earlier (``flowlib.previous_pose``), frame index
    from the manifest's ``frame``."""
    ro = Path(ro)
    frames = read_jsonl(ro / MANIFEST_PRIM if (ro / MANIFEST_PRIM).exists() else ro / MANIFEST)
    with open(ro / "odometry.csv") as f:
        odo = {int(float(r["progress_idx"])): r for r in csv.DictReader(f)}
    todo = []
    for fr in frames:
        k = int(fr["state_k"])
        r = odo[k]
        todo.append(_item(k, index_of(fr["frame"]), fr["t_world"], fr["R_wb"],
                          [float(r[f"vel_{c}"]) for c in "xyz"], [float(r[f"acc_{c}"]) for c in "xyz"],
                          [float(r[f"omega_{c}"]) for c in "xyz"], dt))
    return todo, cam_info(frames[0]["cam"]), frames[0].get("patch")


def drawn_todo(path, dt: float, cam: str = RGB_V9_CAM):
    """``(todo, cam, patch)`` of drawn states (:func:`load_drawn`), frame ``k + 1``,
    with the named camera (default ``starling_front_v1``; ``agile_91`` reproduces the
    renders made before 2026-10-01)."""
    d = load_drawn(path)
    todo = [{**_item(s["k"], frame_index(s["k"]), s["position"], s["R_wb"], s["velocity"], s["acc"], s["omega"], dt),
             "k_record": s.get("k_record")} for s in d["states"]]
    return todo, cam_info(cam), d["patch"]


def frames_json(todo) -> list[dict]:
    """``poses.json``'s per-frame block (what finish_frames reads)."""
    return [{"k": t["k"], **({"sample_id": t["k_record"]} if t.get("k_record") is not None else {}),
             "i": t["i"], "p": t["pose"][0].tolist(), "R_wb": t["pose"][1].tolist(),
             "p_prev": t["prev"][0].tolist(), "R_wb_prev": t["prev"][1].tolist(),
             "v": t["v"].tolist(), "acc": t["acc"].tolist(), "omega": t["omega"].tolist()} for t in todo]


# --- merge a render dir into a labelled rollout ---------------------------------------------------

class MergeError(RuntimeError):
    pass


def _pose_err(a: dict, b: dict) -> float:
    dp = np.abs(np.asarray(a["t_world"], float) - np.asarray(b["t_world"], float)).max()
    dR = np.abs(np.asarray(a["R_wb"], float).ravel() - np.asarray(b["R_wb"], float).ravel()).max()
    return float(max(dp, dR))


def _tif_mm(path: Path) -> np.ndarray:
    from PIL import Image  # noqa: PLC0415 -- repo side only
    with Image.open(str(path)) as im:
        return np.asarray(im, dtype=np.int32)


def _place(src: Path, dst: Path, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))          # a rename on one filesystem; overwrites
    else:
        shutil.copyfile(src, dst)


def render_course_source(render_dir, rlines: list[dict] | None = None):
    """The course build a render dir is of: its manifest lines' ``render.course_source``
    (all equal), else ``blender_checks.json``'s; None for a render made before 2026-09-27."""
    render_dir = Path(render_dir)
    if rlines is None:
        rlines = read_jsonl(render_dir / MANIFEST) if (render_dir / MANIFEST).exists() else []
    vals = [(m.get("render") or {}).get("course_source") for m in rlines]
    vals = [v for v in vals if v]
    if not vals and (render_dir / CHECKS).exists():
        v = json.loads((render_dir / CHECKS).read_text()).get("course_source")
        vals = [v] if v else []
    if not vals:
        return None
    if not all(same_source(v, vals[0]) for v in vals[1:]):
        raise MergeError(f"{render_dir}: its frames name different course builds")
    return vals[0]


def _same_cam_lines(a: dict, b: dict) -> bool:
    """Two manifest lines whose ``cam`` blocks (and mount) render the same pixels; a line
    without a ``cam`` block is taken to be the other's camera (nothing to compare)."""
    if "cam" not in a or "cam" not in b:
        return True
    return same_camera(a["cam"], b["cam"])


def merge_into_rollout(render_dir, rollout, *, move: bool = False, pose_tol: float = 1e-6,
                       min_prim_within_1cm_pct: float = 99.0, allow_partial: bool = False,
                       allow_unverified_course: bool = False) -> dict:
    """Drop the render of one scene (``render_dir``: img/, aux/, render_manifest.jsonl,
    blender_checks.json, numbered by drawn state) into its labelled rollout.

    Which course: the render's ``course_source`` (its manifest lines' ``render`` block:
    the hashes of the ``cloud.npz`` / ``spec.json`` / ``course.json`` it rendered) must
    name the files the labels' ``cloud_source`` names (contract 1; manifest lines and
    ``labels_meta.json``) -- a course rebuilt between labelling and rendering, or a stale
    render kept by a resume, is refused. Labels that record no cloud (batch 1, setM7)
    cannot be checked: the report says ``course_check: unverified``. A render with no
    ``course_source`` (made before 2026-09-27) is refused unless ``allow_unverified_course``.
    Each line's ``geometry_hash`` (the primitives) must also equal the writer's.

    For each line of the rollout's writer manifest (``render_manifest.primitives.jsonl``
    once merged, else ``render_manifest.jsonl``): the render's line for the same
    ``state_k`` must exist and carry the same pose to ``pose_tol`` (so a label and
    a picture of two different draws never pair up). The writer's primitive frame
    moves to ``aux/depth_prim`` and is compared with the render's own primitive
    frame (both are the numpy caster's; below ``min_prim_within_1cm_pct`` % of
    pixels within 1 cm the course's primitives are not the labeller's and the
    merge is refused). The Blender files are placed at the *rollout's* frame
    index, the manifest is rewritten with every modality, and the checks copied.
    Idempotent: a second merge finds the writer's frame in ``aux/`` and redoes
    the rest. Returns a report."""
    render_dir, ro = Path(render_dir), Path(rollout)
    rlist = read_jsonl(render_dir / MANIFEST)
    rlines = {int(m["state_k"]): m for m in rlist}
    prim_path = ro / MANIFEST_PRIM
    if any((ro / pat.format(1)).exists() for pat in LEGACY.values()):
        raise MergeError(f"{ro}: batch-1 layout; run convert-legacy first")
    # the writer's lines, read without writing anything yet: a refused merge leaves the rollout as it was
    wlines = read_jsonl(prim_path if prim_path.exists() else ro / MANIFEST)
    patches = {m.get("patch") for m in wlines} | {m.get("patch") for m in rlines.values()}
    if len(patches - {None}) > 1:
        raise MergeError(f"{ro}: rollout and render are different patches: {sorted(p for p in patches if p)}")
    rsrc = render_course_source(render_dir, rlist)
    try:
        labels_src = rollout_cloud_source(ro, wlines)
        if rsrc is None:
            if not allow_unverified_course:
                raise MergeError(f"{render_dir}: the render records no course_source (rendered before 2026-09-27): "
                                 f"nothing ties it to a course build; re-render, or allow_unverified_course")
            course_check = "unverified (the render records no course)"
        else:
            course_check = check_course(labels_src, rsrc, "the labels")
            if course_check == "unverified":
                course_check = "unverified (the labels record no cloud_source)"
    except CourseMismatch as e:
        raise MergeError(f"{ro}: {e} -- labels and render are of different course builds") from e
    ghash = [(int(m["state_k"]), m.get("geometry_hash"), rlines[int(m["state_k"])].get("geometry_hash"))
             for m in wlines if int(m["state_k"]) in rlines]
    ghash = [g for g in ghash if g[1] and g[2] and g[1] != g[2]]
    if ghash:
        raise MergeError(f"{ro}: geometry_hash of the labels' primitives != the render course's spec.json "
                         f"(state {ghash[0][0]}: {ghash[0][1][:19]} vs {ghash[0][2][:19]})")
    missing = [int(m["state_k"]) for m in wlines if int(m["state_k"]) not in rlines]
    if missing and not allow_partial:
        raise MergeError(f"{ro}: the render has no frame for state(s) {missing}")
    bad = [(int(m["state_k"]), _pose_err(m, rlines[int(m["state_k"])])) for m in wlines
           if int(m["state_k"]) in rlines]
    bad = [(k, e) for k, e in bad if e > pose_tol]
    if bad:
        raise MergeError(f"{ro}: pose mismatch render vs labels (state, max abs err): {bad[:5]} -- "
                         f"rendered from a different draw")

    rep = {"rollout": str(ro), "render": str(render_dir), "states": len(wlines), "merged": 0,
           "missing": missing, "renumbered": 0, "course_check": course_check,
           "course": source_str(rsrc) if rsrc else None, "prim_vs_writer": []}
    # every check before the first file moves: a refused merge leaves the rollout as it was
    for m in wlines:
        k = int(m["state_k"])
        if k not in rlines:
            continue
        i_ro, i_r = index_of(m["frame"]), index_of(rlines[k]["frame"])
        for key in ("depth", "rgb", "depth_prim"):
            if not (render_dir / modalities(i_r)[key]).exists():
                raise MergeError(f"{render_dir}: no {modalities(i_r)[key]}")
        if not _same_cam_lines(m, rlines[k]):
            # another camera (rgb-v9: starling_front_v1 over agile_91 labels): the writer's frame
            # is another picture; the render's own primitive frame (render_depth at the render
            # camera, finish_frames' parity reference) replaces it
            rep["prim_vs_writer"].append({"k": k, "camera_changed": True})
            continue
        # the writer's primitive frame: in img/ before the first merge, in aux/ after
        writer = ro / DEPTH_PRIM.format(i_ro)
        a = _tif_mm(writer if writer.exists() else ro / DEPTH.format(i_ro))
        dmm = np.abs(a - _tif_mm(render_dir / DEPTH_PRIM.format(i_r)))
        pv = {"k": k, "identical_pct": round(100.0 * float(np.mean(dmm == 0)), 3),
              "within_1cm_pct": round(100.0 * float(np.mean(dmm <= 10)), 3), "max_mm": int(dmm.max())}
        rep["prim_vs_writer"].append(pv)
        if pv["within_1cm_pct"] < min_prim_within_1cm_pct:
            raise MergeError(f"{ro}: state {k}: the render's primitive frame is not the labeller's "
                             f"({pv}) -- course spec.json and the labelled patch disagree")

    if not prim_path.exists():
        shutil.copyfile(ro / MANIFEST, prim_path)       # the writer's lines, byte for byte
    out_lines = []
    for m in wlines:
        k = int(m["state_k"])
        i_ro = index_of(m["frame"])
        if k not in rlines:
            out_lines.append(m)
            continue
        r = rlines[k]
        i_r = index_of(r["frame"])
        rep["renumbered"] += int(i_r != i_ro)
        writer = ro / DEPTH_PRIM.format(i_ro)
        new_cam = not _same_cam_lines(m, r)
        if new_cam:
            # the writer's frame was taken by another camera: the render's own primitive frame replaces it
            if (ro / DEPTH.format(i_ro)).exists() and not writer.exists():
                (ro / DEPTH.format(i_ro)).unlink()
            _place(render_dir / DEPTH_PRIM.format(i_r), writer, move)
        elif not writer.exists():
            _place(ro / DEPTH.format(i_ro), writer, move=True)
        for key in ("depth", "rgb", "depth_prev", "rgb_prev", "flow", "depth640"):
            src = render_dir / modalities(i_r, depth640=True)[key]
            if src.exists():
                _place(src, ro / modalities(i_ro, depth640=True)[key], move)
        line = dict(m)
        line.update({k2: r[k2] for k2 in ("depth_source", "course", "camera_verified", "ground_verified", "prev",
                                          "render", "variant", "appearance") if k2 in r})
        if new_cam:
            line.update({k2: r[k2] for k2 in ("cam", "store", "T_bc") if k2 in r})
            line["writer_cam"] = m.get("cam")
        has_prev = (ro / RGB_PREV.format(i_ro)).exists()
        line.update({"frame": DEPTH.format(i_ro),
                     "modalities": modalities(i_ro, prev=has_prev,
                                              depth640=(ro / DEPTH640.format(i_ro)).exists())})
        out_lines.append(line)
        rep["merged"] += 1
    write_jsonl(ro / MANIFEST, out_lines)
    if (render_dir / POSES).exists():
        _place(render_dir / POSES, ro / POSES, move)
    if (render_dir / CHECKS).exists():
        checks = json.loads((render_dir / CHECKS).read_text())
        i_of = {int(m["state_k"]): index_of(m["frame"]) for m in wlines}
        for p in checks.get("per_frame", []):
            if int(p["k"]) in i_of:
                p["i"] = i_of[int(p["k"])]
        checks["merged"] = {k2: v for k2, v in rep.items() if k2 != "prim_vs_writer"}
        pv = [p["within_1cm_pct"] for p in rep["prim_vs_writer"] if "within_1cm_pct" in p]
        checks["merged"]["prim_vs_writer_within_1cm_pct_min"] = min(pv) if pv else None
        checks["merged"]["camera_changed"] = sum(1 for p in rep["prim_vs_writer"] if p.get("camera_changed"))
        (ro / CHECKS).write_text(json.dumps(checks, indent=1))
    return rep


def convert_legacy(rollout) -> dict:
    """A rollout finished in batch 1's layout (rgb_, rgbprev_, depthprev_,
    depth_prim_, flow_ all in img/) -> this layout, manifest paths included.
    Idempotent."""
    ro = Path(rollout)
    moved = 0
    for m in read_jsonl(ro / MANIFEST):
        i = index_of(m["frame"])
        new = modalities(i)
        for key, pat in LEGACY.items():
            src = ro / pat.format(i)
            if src.exists():
                _place(src, ro / new[key], move=True)
                moved += 1
    lines = read_jsonl(ro / MANIFEST)
    for m in lines:
        if m.get("depth_source") == "blender_course":
            m["modalities"] = modalities(index_of(m["frame"]))
    write_jsonl(ro / MANIFEST, lines)
    return {"rollout": str(ro), "moved": moved}


# --- what the readers need ------------------------------------------------------------------------

def _csv_rows(path: Path) -> tuple[int, list[str], list[list[str]]]:
    if not path.exists():
        return -1, [], []
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    return len(lines) - 1, lines[0].split(",") if lines else [], [l.split(",") for l in lines[1:]]


def check_rollout(rollout, *, use_rgb: bool = True, use_depth: bool = True, ref_frame: str = "bf",
                  check_images: bool = True) -> dict:
    """The student loader's and the merge check's view of one rollout, without
    TF: ``data_loader._decode_experiment_dir`` counts ``img/`` names starting
    ``depth_`` (depth-only ``ours``) or containing ``frame_left`` (RGB) against
    the odometry rows and asserts equality, then reads, for row ``r``,
    ``img/frame_left_{r+1}.png`` / ``img/depth_{r+1}.tif`` /
    ``trajectories/trajectories_<ref>_{r}`` (a missing one silently drops the
    sample, so it is checked here); ``goal.csv`` rows and ``pointcloud-unity.ply``
    are asserted. ``t5_label_all``'s merge check wants ``img/*.tif`` ==
    states == manifest lines. Row ``r`` must be state ``r`` (``progress_idx``),
    since the writer names frames by state and the loader by row."""
    ro = Path(rollout)
    errors = []
    names = sorted(os.listdir(ro / "img")) if (ro / "img").is_dir() else []
    n, hdr, rows = _csv_rows(ro / "odometry.csv")
    counts = {"odom_rows": n, "depth_": sum(f.startswith("depth_") for f in names),
              "frame_left": sum("frame_left" in f for f in names), "tif": sum(f.endswith(".tif") for f in names),
              "img_other": [f for f in names if not (f.startswith("depth_") or f.startswith("frame_left_"))][:5],
              "goal_rows": _csv_rows(ro / "goal.csv")[0],
              "manifest_lines": len(read_jsonl(ro / MANIFEST)) if (ro / MANIFEST).exists() else -1}
    if n < 0:
        errors.append("no odometry.csv")
    if use_depth and not use_rgb and counts["depth_"] != n:
        errors.append(f"loader (depth-only): {counts['depth_']} depth_* files != {n} odometry rows")
    if use_rgb and counts["frame_left"] != n:
        errors.append(f"loader (rgb): {counts['frame_left']} frame_left files != {n} odometry rows")
    for key in ("tif", "goal_rows", "manifest_lines"):
        if counts[key] != n:
            errors.append(f"{key} {counts[key]} != {n} states")
    if counts["img_other"]:
        errors.append(f"img/ holds other files: {counts['img_other']}")
    if not (ro / "pointcloud-unity.ply").exists():
        errors.append("no pointcloud-unity.ply")
    pi = hdr.index("progress_idx") if "progress_idx" in hdr else None
    traj = 0
    counts["traj_wf"] = sum(1 for _ in (ro / "trajectories").glob("trajectories_wf_*.csv")) if (ro / "trajectories").is_dir() else 0
    for r in range(max(n, 0)):
        if pi is not None and int(float(rows[r][pi])) != r:
            errors.append(f"row {r} is state {rows[r][pi]}: the loader would pair it with frame {r + 1}")
        if use_depth and not (ro / DEPTH.format(r + 1)).exists():
            errors.append(f"row {r}: no {DEPTH.format(r + 1)}")
        if use_rgb and not (ro / RGB.format(r + 1)).exists():
            errors.append(f"row {r}: no {RGB.format(r + 1)}")
        traj += (ro / "trajectories" / f"trajectories_{ref_frame}_{r:08d}.csv").exists()
    counts["traj_" + ref_frame] = traj
    if (ro / "labels_meta.json").exists():
        meta = json.loads((ro / "labels_meta.json").read_text())
        labelled = sum(1 for s in meta.get("states", []) if s.get("n_rows", len(s.get("rows", []))) > 0)
        counts["labelled"] = labelled
        if traj != labelled:
            errors.append(f"{traj} trajectories_{ref_frame} files != {labelled} labelled states")
        if counts["traj_wf"] != labelled:
            errors.append(f"{counts['traj_wf']} trajectories_wf files != {labelled} labelled states")
    if check_images and n > 0:
        from PIL import Image  # noqa: PLC0415
        if use_depth and (ro / DEPTH.format(1)).exists():
            with Image.open(ro / DEPTH.format(1)) as im:
                if im.size != (224, 224) or im.mode not in ("I;16", "I;16B", "I;16L"):
                    errors.append(f"{DEPTH.format(1)}: {im.size} {im.mode}, want (224, 224) uint16")
        if use_rgb and (ro / RGB.format(1)).exists():
            with Image.open(ro / RGB.format(1)) as im:
                if im.size != (640, 480) or im.mode != "RGB":
                    errors.append(f"{RGB.format(1)}: {im.size} {im.mode}, want (640, 480) RGB")
    return {"rollout": str(ro), "ok": not errors, "errors": errors, "counts": counts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge", help="drop a render dir into its labelled rollout")
    m.add_argument("--render", type=Path, required=True)
    m.add_argument("--rollout", type=Path, required=True)
    m.add_argument("--move", action="store_true", help="move instead of copy")
    m.add_argument("--allow-partial", action="store_true")
    m.add_argument("--allow-unverified-course", action="store_true",
                   help="merge a render that records no course_source (made before 2026-09-27)")
    c = sub.add_parser("convert-legacy", help="batch-1 layout (all in img/) -> img/ + aux/")
    c.add_argument("--rollout", type=Path, nargs="+", required=True)
    k = sub.add_parser("check", help="the loader's and the merge check's counts")
    k.add_argument("--rollout", type=Path, nargs="+", required=True)
    k.add_argument("--rgb", action="store_true", help="also as an RGB arm reads it")
    k.add_argument("--no-depth", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        try:
            rep = merge_into_rollout(a.render, a.rollout, move=a.move, allow_partial=a.allow_partial,
                                     allow_unverified_course=a.allow_unverified_course)
        except MergeError as e:
            print(f"merge refused: {e}", file=sys.stderr)
            return 2
        print(json.dumps({k2: v for k2, v in rep.items() if k2 != "prim_vs_writer"}))
        return 0
    if a.cmd == "convert-legacy":
        for ro in a.rollout:
            print(json.dumps(convert_legacy(ro)))
        return 0
    rc = 0
    for ro in a.rollout:
        rep = check_rollout(ro, use_rgb=a.rgb, use_depth=not a.no_depth)
        print(json.dumps(rep))
        rc |= not rep["ok"]
    return rc


if __name__ == "__main__":
    sys.exit(main())
