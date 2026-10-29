#!/usr/bin/env python3
"""Everything the inspection page shows about one Blender course, as one JSON.

    python blender/page_data.py --course <course dir> --data <dataset out> --out page_data.json

Images are embedded (JPEG, base64): the course previews and, per labelled
state, RGB now and dt earlier, the course depth, the primitives-only depth
(what the labeller rendered before the course existed) and the flow. The
cloud is subsampled for the 3-D view; labels carry their 0.1 s positions and
speeds.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flowlib as FL  # noqa: E402
import render_io as RIO  # noqa: E402

FAR = 20.0
# viridis-like control points, near = bright, far = dark
_CM = np.array([[253, 231, 37], [94, 201, 98], [33, 145, 140], [59, 82, 139], [68, 1, 84], [20, 12, 30]], float)


def jpeg(arr: np.ndarray, q: int = 82, size=None) -> str:
    im = Image.fromarray(arr)
    if size:
        im = im.resize(size, Image.LANCZOS)
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def depth_rgb(d: np.ndarray) -> np.ndarray:
    x = np.clip(np.where(np.isfinite(d), d, FAR) / FAR, 0, 1) ** 0.6 * (len(_CM) - 1)
    i = np.minimum(np.floor(x).astype(int), len(_CM) - 2)
    f = (x - i)[..., None]
    rgb = _CM[i] * (1 - f) + _CM[i + 1] * f
    rgb[~np.isfinite(d) | (d >= FAR)] = [14, 10, 20]
    return rgb.astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cloud-points", type=int, default=30000)
    a = ap.parse_args(argv)
    course = json.loads((a.course / "course.json").read_text())
    spec = json.loads((a.course / "spec.json").read_text())
    patch = course["patch"]
    ro = next((a.data / "train").glob(f"rollout_*_{spec['family']}"))
    raw = ro / "blender_raw"
    poses = json.loads((raw / "poses.json").read_text())
    checks = json.loads((ro / "blender_checks.json").read_text())
    per = {p["i"]: p for p in checks["per_frame"]}

    z = np.load(a.course / "cloud.npz")
    pts, reg = z["points"], z["region"]
    obs = pts[reg >= 0]
    rng = np.random.default_rng(0)
    sub = obs[rng.choice(len(obs), min(a.cloud_points, len(obs)), replace=False)]

    # one flow scale for every state: the 95th percentile of the per-state 99th percentiles
    mags = []
    for fr in poses["frames"]:
        fz = np.load(ro / RIO.FLOW.format(fr["i"]))
        f = fz["flow_full"].astype(np.float32)
        mags.append(np.percentile(np.hypot(f[0], f[1]), 99))
    fscale = float(np.percentile(mags, 95))

    states = []
    for fr in poses["frames"]:
        i, k = fr["i"], fr["k"]
        sj = json.loads((a.data / "states" / patch / f"{k}.json").read_text())
        d = np.load(raw / f"depth_{i:08d}.npy")
        dprim = np.load(raw / f"depthprim_{i:08d}.npy")
        fz = np.load(ro / RIO.FLOW.format(i))
        flow = fz["flow_full"].astype(np.float32)
        status = fz["status_full"] if "status_full" in fz else np.where(fz["valid_full"].astype(bool), 0, 1).astype(np.uint8)
        valid = status == FL.VALID
        bf = raw / f"blenderflow_{i:08d}.npy"
        bflow = np.load(bf).astype(np.float32) if bf.exists() else None
        rgb = np.asarray(Image.open(raw / f"rgb_{i:08d}.png").convert("RGB"))
        rgbp = np.asarray(Image.open(raw / f"rgbprev_{i:08d}.png").convert("RGB"))
        mm224 = np.asarray(Image.open(ro / "img" / f"depth_{i:08d}.tif"), np.float32) / 1000.0
        both = np.isfinite(d) | np.isfinite(dprim)
        changed = np.abs(np.minimum(d, FAR) - np.minimum(dprim, FAR)) > 0.05
        mag = np.hypot(flow[0], flow[1])
        fmax = max(float(np.percentile(mag[valid], 99)), 1.0) if valid.any() else 1.0
        rows = []
        for r in sj["labels"]:
            P = np.asarray(r["P"], float)
            rows.append({"P": np.round(P, 3).tolist(), "speeds": np.round(np.asarray(r["speeds"], float), 2).tolist(),
                         "cost": round(float(r["cost"]), 3), "rel_cost": round(float(r.get("rel_cost", 0.0)), 3),
                         "class": r.get("class_tag"), "arrived_at": r.get("arrived_at"),
                         "min_clearance": None if r.get("min_clearance") is None else round(float(r["min_clearance"]), 3),
                         "end_speed": r.get("end_speed")})
        st = sj["state"]
        states.append({
            "i": i, "k": k, "p": np.round(fr["p"], 3).tolist(), "R": np.round(np.asarray(fr["R_wb"]), 4).tolist(),
            "v": np.round(fr["v"], 3).tolist(), "speed": st["speed"], "heading_deg": st.get("heading_deg"),
            "bins": st.get("bins"), "goal": sj["goal"], "goal_speed": sj["goal_speed"], "goal_case": sj.get("goal_case"),
            "rows": rows, "checks": per[i], "flow_max_px": round(fscale, 1),
            "course_changed_pct": round(100.0 * float(changed[both].mean()), 1) if both.any() else 0.0,
            "img": {"rgb": jpeg(rgb), "rgbprev": jpeg(rgbp), "depth": jpeg(depth_rgb(d)),
                    "depthprim": jpeg(depth_rgb(dprim)),
                    "depth224": jpeg(depth_rgb(np.where(mm224 >= FAR, np.inf, mm224)), size=(224, 224)),
                    "flow": jpeg(FL.flow_to_rgb(np.moveaxis(flow, 0, -1), None, fscale, style="dark")),
                    "flowstatus": jpeg(FL.status_to_rgb(status, rgb)),
                    **({"blenderflow": jpeg(FL.flow_to_rgb(bflow, None, fscale, style="dark"))} if bflow is not None else {})},
        })

    previews = {}
    for name in course.get("previews", []):
        previews[name.removeprefix("preview_").removesuffix(".png")] = jpeg(
            np.asarray(Image.open(a.course / name).convert("RGB")), q=80)
    build = json.loads((a.data / "summary" / "build_train.json").read_text())
    out = {"patch": patch, "spec": {k: spec[k] for k in ("family", "seed", "bounds", "face_x", "far_x", "params", "primitives")},
           "course": course, "previews": previews, "cloud": np.round(sub, 2).tolist(),
           "cloud_total": int(len(pts)), "checks": {k: v for k, v in checks.items() if k != "per_frame"},
           "render": {k: poses[k] for k in ("dt", "rgb_samples", "device", "blender", "rgb_s_per_frame",
                                            "depth_s_per_frame", "seconds")},
           "label_build": {k: build[k] for k in build if isinstance(build[k], (int, float, str))},
           "states": states}
    a.out.write_text(json.dumps(out))
    print(a.out, round(a.out.stat().st_size / 1e6, 2), "MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
